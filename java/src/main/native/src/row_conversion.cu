/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <algorithm>
#include <iostream>
#include <iterator>
#include <limits>
#include <tuple>

#include <cooperative_groups.h>
#include <type_traits>

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
#include <cuda/barrier>
#endif

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/sequence.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/row_conversion.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
constexpr auto NUM_BLOCKS_PER_KERNEL_TO_COLUMNS = 8;
constexpr auto NUM_BLOCKS_PER_KERNEL_FROM_COLUMNS = 8;
constexpr auto NUM_BLOCKS_PER_KERNEL_LOADED = 2;
constexpr auto NUM_VALIDITY_BLOCKS_PER_KERNEL = 8;
constexpr auto NUM_VALIDITY_BLOCKS_PER_KERNEL_LOADED = 2;
#endif

using cudf::detail::make_device_uvector_async;
namespace cudf {

namespace detail {

static inline __host__ __device__ int32_t align_offset(int32_t offset, std::size_t alignment) {
  return (offset + alignment - 1) & ~(alignment - 1);
}

__global__ void copy_to_fixed_width_columns(const cudf::size_type num_rows,
                                            const cudf::size_type num_columns,
                                            const cudf::size_type row_size,
                                            const cudf::size_type *input_offset_in_row,
                                            const cudf::size_type *num_bytes, int8_t **output_data,
                                            cudf::bitmask_type **output_nm,
                                            const int8_t *input_data) {
  // We are going to copy the data in two passes.
  // The first pass copies a chunk of data into shared memory.
  // The second pass copies that chunk from shared memory out to the final location.

  // Because shared memory is limited we copy a subset of the rows at a time.
  // For simplicity we will refer to this as a row_group

  // In practice we have found writing more than 4 columns of data per thread
  // results in performance loss. As such we are using a 2 dimensional
  // kernel in terms of threads, but not in terms of blocks. Columns are
  // controlled by the y dimension (there is no y dimension in blocks). Rows
  // are controlled by the x dimension (there are multiple blocks in the x
  // dimension).

  cudf::size_type rows_per_group = blockDim.x;
  cudf::size_type row_group_start = blockIdx.x;
  cudf::size_type row_group_stride = gridDim.x;
  cudf::size_type row_group_end = (num_rows + rows_per_group - 1) / rows_per_group + 1;

  extern __shared__ int8_t shared_data[];

  // Because we are copying fixed width only data and we stride the rows
  // this thread will always start copying from shared data in the same place
  int8_t *row_tmp = &shared_data[row_size * threadIdx.x];
  int8_t *row_vld_tmp = &row_tmp[input_offset_in_row[num_columns - 1] + num_bytes[num_columns - 1]];

  for (cudf::size_type row_group_index = row_group_start; row_group_index < row_group_end;
       row_group_index += row_group_stride) {
    // Step 1: Copy the data into shared memory
    // We know row_size is always aligned with and a multiple of int64_t;
    int64_t *long_shared = reinterpret_cast<int64_t *>(shared_data);
    const int64_t *long_input = reinterpret_cast<int64_t const *>(input_data);

    cudf::size_type shared_output_index = threadIdx.x + (threadIdx.y * blockDim.x);
    cudf::size_type shared_output_stride = blockDim.x * blockDim.y;
    cudf::size_type row_index_end = ((row_group_index + 1) * rows_per_group);
    if (row_index_end > num_rows) {
      row_index_end = num_rows;
    }
    cudf::size_type num_rows_in_group = row_index_end - (row_group_index * rows_per_group);
    cudf::size_type shared_length = row_size * num_rows_in_group;

    cudf::size_type shared_output_end = shared_length / sizeof(int64_t);

    cudf::size_type start_input_index =
        (row_size * row_group_index * rows_per_group) / sizeof(int64_t);

    for (cudf::size_type shared_index = shared_output_index; shared_index < shared_output_end;
         shared_index += shared_output_stride) {
      long_shared[shared_index] = long_input[start_input_index + shared_index];
    }
    // Wait for all of the data to be in shared memory
    __syncthreads();

    // Step 2 copy the data back out

    // Within the row group there should be 1 thread for each row.  This is a
    // requirement for launching the kernel
    cudf::size_type row_index = (row_group_index * rows_per_group) + threadIdx.x;
    // But we might not use all of the threads if the number of rows does not go
    // evenly into the thread count. We don't want those threads to exit yet
    // because we may need them to copy data in for the next row group.
    uint32_t active_mask = __ballot_sync(0xffffffff, row_index < num_rows);
    if (row_index < num_rows) {
      cudf::size_type col_index_start = threadIdx.y;
      cudf::size_type col_index_stride = blockDim.y;
      for (cudf::size_type col_index = col_index_start; col_index < num_columns;
           col_index += col_index_stride) {
        cudf::size_type col_size = num_bytes[col_index];
        const int8_t *col_tmp = &(row_tmp[input_offset_in_row[col_index]]);
        int8_t *col_output = output_data[col_index];
        switch (col_size) {
          case 1: {
            col_output[row_index] = *col_tmp;
            break;
          }
          case 2: {
            int16_t *short_col_output = reinterpret_cast<int16_t *>(col_output);
            short_col_output[row_index] = *reinterpret_cast<const int16_t *>(col_tmp);
            break;
          }
          case 4: {
            int32_t *int_col_output = reinterpret_cast<int32_t *>(col_output);
            int_col_output[row_index] = *reinterpret_cast<const int32_t *>(col_tmp);
            break;
          }
          case 8: {
            int64_t *long_col_output = reinterpret_cast<int64_t *>(col_output);
            long_col_output[row_index] = *reinterpret_cast<const int64_t *>(col_tmp);
            break;
          }
          default: {
            cudf::size_type output_offset = col_size * row_index;
            // TODO this should just not be supported for fixed width columns, but just in case...
            for (cudf::size_type b = 0; b < col_size; b++) {
              col_output[b + output_offset] = col_tmp[b];
            }
            break;
          }
        }

        cudf::bitmask_type *nm = output_nm[col_index];
        int8_t *valid_byte = &row_vld_tmp[col_index / 8];
        cudf::size_type byte_bit_offset = col_index % 8;
        int predicate = *valid_byte & (1 << byte_bit_offset);
        uint32_t bitmask = __ballot_sync(active_mask, predicate);
        if (row_index % 32 == 0) {
          nm[word_index(row_index)] = bitmask;
        }
      } // end column loop
    }   // end row copy
    // wait for the row_group to be totally copied before starting on the next row group
    __syncthreads();
  }
}

__global__ void
copy_from_fixed_width_columns(const cudf::size_type start_row, const cudf::size_type num_rows,
                              const cudf::size_type num_columns, const cudf::size_type row_size,
                              const cudf::size_type *output_offset_in_row,
                              const cudf::size_type *num_bytes, const int8_t **input_data,
                              const cudf::bitmask_type **input_nm, int8_t *output_data) {
  // We are going to copy the data in two passes.
  // The first pass copies a chunk of data into shared memory.
  // The second pass copies that chunk from shared memory out to the final location.

  // Because shared memory is limited we copy a subset of the rows at a time.
  // We do not support copying a subset of the columns in a row yet, so we don't
  // currently support a row that is wider than shared memory.
  // For simplicity we will refer to this as a row_group

  // In practice we have found reading more than 4 columns of data per thread
  // results in performance loss. As such we are using a 2 dimensional
  // kernel in terms of threads, but not in terms of blocks. Columns are
  // controlled by the y dimension (there is no y dimension in blocks). Rows
  // are controlled by the x dimension (there are multiple blocks in the x
  // dimension).

  cudf::size_type rows_per_group = blockDim.x;
  cudf::size_type row_group_start = blockIdx.x;
  cudf::size_type row_group_stride = gridDim.x;
  cudf::size_type row_group_end = (num_rows + rows_per_group - 1) / rows_per_group + 1;

  extern __shared__ int8_t shared_data[];

  // Because we are copying fixed width only data and we stride the rows
  // this thread will always start copying to shared data in the same place
  int8_t *row_tmp = &shared_data[row_size * threadIdx.x];
  int8_t *row_vld_tmp =
      &row_tmp[output_offset_in_row[num_columns - 1] + num_bytes[num_columns - 1]];

  for (cudf::size_type row_group_index = row_group_start; row_group_index < row_group_end;
       row_group_index += row_group_stride) {
    // Within the row group there should be 1 thread for each row.  This is a
    // requirement for launching the kernel
    cudf::size_type row_index = start_row + (row_group_index * rows_per_group) + threadIdx.x;
    // But we might not use all of the threads if the number of rows does not go
    // evenly into the thread count. We don't want those threads to exit yet
    // because we may need them to copy data back out.
    if (row_index < (start_row + num_rows)) {
      cudf::size_type col_index_start = threadIdx.y;
      cudf::size_type col_index_stride = blockDim.y;
      for (cudf::size_type col_index = col_index_start; col_index < num_columns;
           col_index += col_index_stride) {
        cudf::size_type col_size = num_bytes[col_index];
        int8_t *col_tmp = &(row_tmp[output_offset_in_row[col_index]]);
        const int8_t *col_input = input_data[col_index];
        switch (col_size) {
          case 1: {
            *col_tmp = col_input[row_index];
            break;
          }
          case 2: {
            const int16_t *short_col_input = reinterpret_cast<const int16_t *>(col_input);
            *reinterpret_cast<int16_t *>(col_tmp) = short_col_input[row_index];
            break;
          }
          case 4: {
            const int32_t *int_col_input = reinterpret_cast<const int32_t *>(col_input);
            *reinterpret_cast<int32_t *>(col_tmp) = int_col_input[row_index];
            break;
          }
          case 8: {
            const int64_t *long_col_input = reinterpret_cast<const int64_t *>(col_input);
            *reinterpret_cast<int64_t *>(col_tmp) = long_col_input[row_index];
            break;
          }
          default: {
            cudf::size_type input_offset = col_size * row_index;
            // TODO this should just not be supported for fixed width columns, but just in case...
            for (cudf::size_type b = 0; b < col_size; b++) {
              col_tmp[b] = col_input[b + input_offset];
            }
            break;
          }
        }
        // atomicOr only works on 32 bit or 64 bit  aligned values, and not byte aligned
        // so we have to rewrite the addresses to make sure that it is 4 byte aligned
        int8_t *valid_byte = &row_vld_tmp[col_index / 8];
        cudf::size_type byte_bit_offset = col_index % 8;
        uint64_t fixup_bytes = reinterpret_cast<uint64_t>(valid_byte) % 4;
        int32_t *valid_int = reinterpret_cast<int32_t *>(valid_byte - fixup_bytes);
        cudf::size_type int_bit_offset = byte_bit_offset + (fixup_bytes * 8);
        // Now copy validity for the column
        if (input_nm[col_index]) {
          if (bit_is_set(input_nm[col_index], row_index)) {
            atomicOr_block(valid_int, 1 << int_bit_offset);
          } else {
            atomicAnd_block(valid_int, ~(1 << int_bit_offset));
          }
        } else {
          // It is valid so just set the bit
          atomicOr_block(valid_int, 1 << int_bit_offset);
        }
      } // end column loop
    }   // end row copy
    // wait for the row_group to be totally copied into shared memory
    __syncthreads();

    // Step 2: Copy the data back out
    // We know row_size is always aligned with and a multiple of int64_t;
    int64_t *long_shared = reinterpret_cast<int64_t *>(shared_data);
    int64_t *long_output = reinterpret_cast<int64_t *>(output_data);

    cudf::size_type shared_input_index = threadIdx.x + (threadIdx.y * blockDim.x);
    cudf::size_type shared_input_stride = blockDim.x * blockDim.y;
    cudf::size_type row_index_end = ((row_group_index + 1) * rows_per_group);
    if (row_index_end > num_rows) {
      row_index_end = num_rows;
    }
    cudf::size_type num_rows_in_group = row_index_end - (row_group_index * rows_per_group);
    cudf::size_type shared_length = row_size * num_rows_in_group;

    cudf::size_type shared_input_end = shared_length / sizeof(int64_t);

    cudf::size_type start_output_index =
        (row_size * row_group_index * rows_per_group) / sizeof(int64_t);

    for (cudf::size_type shared_index = shared_input_index; shared_index < shared_input_end;
         shared_index += shared_input_stride) {
      long_output[start_output_index + shared_index] = long_shared[shared_index];
    }
    __syncthreads();
    // Go for the next round
  }
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700

struct block_info {
  int start_col;
  int start_row;
  int end_col;
  int end_row;
  int buffer_num;

  __host__ __device__ size_type get_row_size(size_type const *const col_offsets,
                                             size_type const *const col_sizes) const {
    return align_offset(col_offsets[end_col] + col_sizes[end_col] - col_offsets[start_col], 8);
  }
  __host__ __device__ size_type num_cols() const { return end_col - start_col + 1; }

  __host__ __device__ size_type num_rows() const { return end_row - start_row + 1; }
};

// When building the columns to return, we have to be mindful of the offset limit in cudf.
// It is 32-bit and these data columns are capable of surpassing that easily. The data should
// not be cut off exactly at the limit though due to the validity buffers. The most efficient
// place to cut the validity is on a 32-row boundary, so as we calculate the row sizes
// we keep track of the cut points for the validity, which we call row batches. If the row
// is larger than can be represented with the 32-bit offsets, we use the last 32-row boundary we
// hit. Note that this boundary is for our book-keeping with column pointers and not anything that
// the kernel needs to worry about. We cut the output at convienient boundaries when assembling
// the outgoing data stream.
struct row_batch {
  size_type num_bytes;
  size_type row_count;
};

/**
 * @brief copy data from cudf columns into x format, which is row-based
 *
 * @param num_rows total number of rows in the table
 * @param num_columns total number of columns in the table
 * @param input_data pointer to raw table data
 * @param input_nm pointer to validity data
 * @param col_sizes array of sizes for each element in a column - one per column
 * @param col_offsets offset into input data row for each column's start
 * @param block_infos information about the blocks of work
 * @param row_offsets offset to a specific row in the input data
 * @param output_data pointer to output data
 *
 */
__global__ void copy_from_columns(const size_type num_rows, const size_type num_columns,
                                  const size_type shmem_used_per_block,
                                  const size_type num_block_infos, const int8_t **input_data,
                                  const size_type *col_sizes, const size_type *col_offsets,
                                  const block_info *block_infos, const size_type *row_offsets,
                                  int8_t **output_data) {
  // We are going to copy the data in two passes.
  // The first pass copies a chunk of data into shared memory.
  // The second pass copies that chunk from shared memory out to the final location.

  // Because shared memory is limited we copy a subset of the rows at a time.
  // This has been broken up for us in the block_info struct, so we don't have
  // any calculation to do here, but it is important to note.

  constexpr unsigned stages_count = NUM_BLOCKS_PER_KERNEL_LOADED;
  auto group = cooperative_groups::this_thread_block();
  extern __shared__ int8_t shared_data[];
  int8_t *shared[stages_count] = {shared_data, shared_data + shmem_used_per_block};

  __shared__ cuda::barrier<cuda::thread_scope_block> block_barrier[NUM_BLOCKS_PER_KERNEL_LOADED];
  if (group.thread_rank() == 0) {
    for (int i = 0; i < NUM_BLOCKS_PER_KERNEL_LOADED; ++i) {
      init(&block_barrier[i], group.size());
    }
  }

  group.sync();

  auto const blocks_remaining =
      std::min((uint)(num_block_infos % NUM_BLOCKS_PER_KERNEL_FROM_COLUMNS),
               std::min(num_block_infos - blockIdx.x * NUM_BLOCKS_PER_KERNEL_TO_COLUMNS,
                        (uint)NUM_BLOCKS_PER_KERNEL_TO_COLUMNS));

  size_t fetch;
  size_t subset;
  for (subset = fetch = 0; subset < blocks_remaining; ++subset) {
    // Fetch ahead up to stages_count subsets
    for (; fetch < blocks_remaining && fetch < (subset + stages_count); ++fetch) {
      auto const fetch_block = block_infos[blockIdx.x * NUM_BLOCKS_PER_KERNEL_TO_COLUMNS + fetch];

      auto const num_fetch_cols = fetch_block.num_cols();
      auto const num_fetch_rows = fetch_block.num_rows();
      auto const num_elements_in_block = num_fetch_cols * num_fetch_rows;
      auto const fetch_block_row_size = fetch_block.get_row_size(col_offsets, col_sizes);
      auto const starting_column_offset = col_offsets[fetch_block.start_col];
      auto &fetch_barrier = block_barrier[fetch % NUM_BLOCKS_PER_KERNEL_LOADED];

      // wait for the last use of the memory to be completed
      if (fetch > NUM_BLOCKS_PER_KERNEL_LOADED) {
        fetch_barrier.arrive_and_wait();
      }

      // to do the copy we need to do n column copies followed by m element copies OR
      // we have to do m element copies followed by r row copies. When going from column
      // to row it is much easier to copy by elements first otherwise we would need a running
      // total of the column sizes for our block, which isn't readily available. This makes it more
      // appealing to copy element-wise from input data into shared matching the end layout and do
      // row-based memcopies out.

      for (auto el = (int)threadIdx.x; el < num_elements_in_block; el += blockDim.x) {
        auto const relative_col = el / num_fetch_rows;
        auto const relative_row = el % num_fetch_rows;
        auto const absolute_col = relative_col + fetch_block.start_col;
        auto const absolute_row = relative_row + fetch_block.start_row;
        auto const col_size = col_sizes[absolute_col];
        auto const col_offset = col_offsets[absolute_col];
        auto const relative_col_offset = col_offset - starting_column_offset;

        auto const shared_offset = relative_row * fetch_block_row_size + relative_col_offset;
        auto const input_src = input_data[absolute_col] + col_size * absolute_row;

        // copy the main
        cuda::memcpy_async(&shared[fetch % stages_count][shared_offset], input_src, col_size,
                           fetch_barrier);
      }
    }

    auto &subset_barrier = block_barrier[subset % NUM_BLOCKS_PER_KERNEL_LOADED];
    subset_barrier.arrive_and_wait();

    auto block = block_infos[blockIdx.x * NUM_BLOCKS_PER_KERNEL_TO_COLUMNS + subset];
    /*    auto const rows_in_block  = block.num_rows();
        auto const cols_in_block  = block.num_cols();*/
    auto const block_row_size = block.get_row_size(col_offsets, col_sizes);
    auto const column_offset = col_offsets[block.start_col];

    // copy entire rows to final dest
    for (auto absolute_row = block.start_row + threadIdx.x; absolute_row <= block.end_row;
         absolute_row += blockDim.x) {
      auto const relative_row = absolute_row - block.start_row;
      auto const output_dest =
          output_data[block.buffer_num] + absolute_row * block_row_size + column_offset;
      auto const shared_offset = block_row_size * relative_row;
      cuda::memcpy_async(output_dest, &shared[subset % stages_count][shared_offset], block_row_size,
                         subset_barrier);
    }
  }

  // wait on the last copies to complete
  for (uint i = 0; i < std::min(stages_count, blocks_remaining); ++i) {
    block_barrier[i].arrive_and_wait();
  }
}

/**
 * @brief copy data from row-based format to cudf columns
 *
 * @param num_rows total number of rows in the table
 * @param num_columns total number of columns in the table
 * @param shmem_used_per_block amount of shared memory that is used by a block
 * @param offsets
 * @param output_data pointer to output data, partitioned by data size
 * @param validity_offsets offset into input data row for validity data
 * @param block_infos information about the blocks of work
 * @param num_block_infos number of infos in blocks array
 * @param input_data pointer to input data
 *
 */
__global__ void copy_validity_from_columns(
    const size_type num_rows, const size_type num_columns, const size_type shmem_used_per_block,
    const size_type *row_offsets, int8_t **output_data, const size_type validity_offset,
    const block_info *block_infos, const size_type num_block_infos, const bitmask_type **input_nm) {
  extern __shared__ int8_t shared_data[];
  int8_t *shared_blocks[NUM_VALIDITY_BLOCKS_PER_KERNEL_LOADED] = {
      shared_data, shared_data + shmem_used_per_block / 2};

  // per conversation with DaveB
  // each thread of warp reads a single int32 of validity - so we read 128 bytes
  // then ballot_sync the bits and write the result to shmem
  // after we fill shared mem memcpy it out in a blob.
  // probably need knobs for number of rows vs columns to balance read/write
  auto group = cooperative_groups::this_thread_block();

  int const blocks_remaining =
      std::min(num_block_infos - blockIdx.x * NUM_VALIDITY_BLOCKS_PER_KERNEL,
               (uint)NUM_VALIDITY_BLOCKS_PER_KERNEL);

  __shared__ cuda::barrier<cuda::thread_scope_block>
      shared_block_barriers[NUM_VALIDITY_BLOCKS_PER_KERNEL_LOADED];
  if (group.thread_rank() == 0) {
    for (int i = 0; i < NUM_VALIDITY_BLOCKS_PER_KERNEL_LOADED; ++i) {
      init(&shared_block_barriers[i], group.size());
    }
  }

  group.sync();

  for (int validity_block = 0; validity_block < blocks_remaining; ++validity_block) {
    if (validity_block != validity_block % NUM_VALIDITY_BLOCKS_PER_KERNEL_LOADED) {
      shared_block_barriers[validity_block % NUM_VALIDITY_BLOCKS_PER_KERNEL_LOADED]
          .arrive_and_wait();
    }
    int8_t *this_shared_block = shared_blocks[validity_block % 2];
    auto block = block_infos[blockIdx.x * NUM_VALIDITY_BLOCKS_PER_KERNEL + validity_block];

    auto const num_block_cols = block.num_cols();
    auto const num_block_rows = block.num_rows();

    auto const num_sections_x = (num_block_cols + 31) / 32;
    auto const num_sections_y = (num_block_rows + 7) / 8;
    auto const validity_data_row_length =
        align_offset(util::div_rounding_up_unsafe(num_block_cols, 8), 8);
    auto const total_sections = num_sections_x * num_sections_y;

    int const warp_id = threadIdx.x / detail::warp_size;
    int const lane_id = threadIdx.x % detail::warp_size;
    auto const warps_per_block = std::max(1u, blockDim.x / detail::warp_size);

    // the block is divided into sections. A warp operates on a section at a time.
    for (int my_section_idx = warp_id; my_section_idx < total_sections;
         my_section_idx += warps_per_block) {
      // convert to rows and cols
      auto const section_x = my_section_idx / num_sections_x;
      auto const section_y = my_section_idx % num_sections_x;

      auto const relative_col = section_x * 32 + lane_id;
      auto const relative_row = section_y * 8;
      auto const absolute_col = relative_col + block.start_col;
      auto const absolute_row = relative_row + block.start_row;
      auto const cols_left = num_columns - absolute_col;

      auto const participation_mask = __ballot_sync(0xFFFFFFFF, absolute_col < num_columns);

      if (absolute_col < num_columns) {
        auto my_byte =
            input_nm[absolute_col] != nullptr ? input_nm[absolute_col][absolute_row / 32] : 0xFF;

        // every thread that is participating in the warp has a byte, but it's column-based
        // data and we need it in row-based. So we shiffle the bits around with ballot_sync to make
        // the bytes we actually write.
        for (int i = 0, byte_mask = 1; i < 8 && relative_row + i < num_rows; ++i, byte_mask <<= 1) {
          auto validity_data = __ballot_sync(participation_mask, my_byte & byte_mask);
          // lead thread in each warp writes data
          auto const validity_write_offset =
              validity_data_row_length * (relative_row + i) + relative_col / 8;
          if (threadIdx.x % detail::warp_size == 0) {
            if (cols_left <= 8) {
              // write byte
              this_shared_block[validity_write_offset] = validity_data & 0xFF;
            } else if (cols_left <= 16) {
              // write int16
              *reinterpret_cast<int16_t *>(&this_shared_block[validity_write_offset]) =
                  validity_data & 0xFFFF;
            } else if (cols_left <= 24) {
              // write int16 and then int8
              *reinterpret_cast<int16_t *>(&this_shared_block[validity_write_offset]) =
                  validity_data & 0xFFFF;
              shared_data[validity_write_offset + 2] = (validity_data >> 16) & 0xFF;
            } else {
              // write int32
              *reinterpret_cast<int32_t *>(&this_shared_block[validity_write_offset]) =
                  validity_data;
            }
          }
        }
      }
    }

    // make sure entire block has finished copy
    group.sync();

    // now async memcpy the shared memory out to the final destination
    for (int row = block.start_row + threadIdx.x; row <= block.end_row; row += blockDim.x) {
      auto const relative_row = row - block.start_row;
      auto const output_ptr =
          output_data[block.buffer_num] + row_offsets[row] + validity_offset + block.start_col / 8;
      auto const num_bytes = util::div_rounding_up_unsafe(num_block_cols, 8);
      cuda::memcpy_async(
          output_ptr, &this_shared_block[validity_data_row_length * relative_row], num_bytes,
          shared_block_barriers[validity_block % NUM_VALIDITY_BLOCKS_PER_KERNEL_LOADED]);
    }
  }

  // wait for last blocks of data to arrive
  for (int validity_block = 0;
       validity_block < blocks_remaining % NUM_VALIDITY_BLOCKS_PER_KERNEL_LOADED;
       ++validity_block) {
    shared_block_barriers[validity_block].arrive_and_wait();
  }
}

static __device__ std::tuple<size_type, size_type>
get_admin_data_sizes(size_t col_size_size, size_t col_offset_size, int const num_cols) {
  auto const col_size_bytes = num_cols * col_size_size;
  auto const col_offset_bytes = num_cols * col_offset_size;

  return {col_size_bytes, col_offset_bytes};
}

/**
 * @brief ensure `read_ahead` buffer blocks are fetched
 *
 * @param fetch_index internal state passed into the function
 * @param processing_index index where processing is occuring
 * @param read_ahead_count how many blocks to read ahead
 * @param max_resident_blocks how many blocks can be loaded at once
 * @param total_blocks total number of blocks overall
 * @param block_infos pointer to the block infos
 * @param col_sizes pointer to column size information
 * @param col_offsets pointer to the table's column offsets
 * @param row_offsets pointer to offsets for each row in the table
 * @param input_data pointer to the input data
 * @param shared pointer to shared memory
 * @param group thread group participating in the fetch
 * @param block_barrier barriers used for each block
 * @return
 */
static __device__ void
fetch_blocks_for_row_to_column(size_t &fetch_index, size_t const processing_index,
                               int const read_ahead_count, int const max_resident_blocks,
                               int const total_blocks, block_info const *const block_infos,
                               size_type const *const col_sizes, size_type const *const col_offsets,
                               size_type const *const row_offsets, int8_t const *const input_data,
                               int8_t *shared[], cooperative_groups::thread_block const group,
                               cuda::barrier<cuda::thread_scope_block> *block_barrier) {
  for (; fetch_index < static_cast<size_t>(total_blocks) &&
         fetch_index < (processing_index + read_ahead_count);
       ++fetch_index) {
    auto const fetch_block =
        block_infos[blockIdx.x * NUM_BLOCKS_PER_KERNEL_TO_COLUMNS + fetch_index];
    auto const fetch_block_start_row = fetch_block.start_row;
    auto const fetch_block_end_row = fetch_block.end_row;
    auto const starting_col_offset = col_offsets[fetch_block.start_col];

    auto const fetch_block_row_size = fetch_block.get_row_size(col_offsets, col_sizes);
    auto const num_fetch_cols = fetch_block.num_cols();
    auto [col_size_bytes, col_offset_bytes] = get_admin_data_sizes(
        sizeof(decltype(*col_sizes)), sizeof(decltype(*col_offsets)), num_fetch_cols);
    auto &fetch_barrier = block_barrier[fetch_index % NUM_BLOCKS_PER_KERNEL_LOADED];

    // if we have fetched all buffers, we need to wait for processing
    // to complete on them before we can use them again
    if (fetch_index > NUM_BLOCKS_PER_KERNEL_LOADED) {
      fetch_barrier.arrive_and_wait();
    }

    auto shared_row_offset = 0;
    // copy the data for column sizes
    cuda::memcpy_async(group, &shared[fetch_index % max_resident_blocks][shared_row_offset],
                       &col_sizes[fetch_block.start_col], col_size_bytes, fetch_barrier);
    shared_row_offset += col_size_bytes;
    // copy the data for column offsets
    cuda::memcpy_async(group, &shared[fetch_index % max_resident_blocks][shared_row_offset],
                       &col_offsets[fetch_block.start_col], col_offset_bytes, fetch_barrier);
    shared_row_offset += col_offset_bytes;
    shared_row_offset = align_offset(shared_row_offset, 8);

    for (auto row = fetch_block_start_row + static_cast<int>(threadIdx.x);
         row <= fetch_block_end_row; row += blockDim.x) {
      auto shared_offset = (row - fetch_block_start_row) * fetch_block_row_size + shared_row_offset;
      // copy the main
      cuda::memcpy_async(&shared[fetch_index % max_resident_blocks][shared_offset],
                         &input_data[row_offsets[row] + starting_col_offset], fetch_block_row_size,
                         fetch_barrier);
    }
  }
}

/**
 * @brief copy data from row-based format to cudf columns
 *
 * @param num_rows total number of rows in the table
 * @param num_columns total number of columns in the table
 * @param shmem_used_per_block amount of shared memory that is used by a block
 * @param row_offsets
 * @param output_data
 * @param output_nm
 * @param col_sizes array of sizes for each element in a column - one per column
 * @param col_offsets offset into input data row for each column's start
 * @param block_infos information about the blocks of work
 * @param input_data pointer to input data
 *
 */
__global__ void copy_to_columns(const size_type num_rows, const size_type num_columns,
                                const size_type shmem_used_per_block, const size_type *row_offsets,
                                int8_t **output_data, const size_type *_col_sizes,
                                const size_type *_col_offsets, const block_info *block_infos,
                                const size_type num_block_infos, const int8_t *input_data) {
  // We are going to copy the data in two passes.
  // The first pass copies a chunk of data into shared memory.
  // The second pass copies that chunk from shared memory out to the final location.

  // Because shared memory is limited we copy a subset of the rows at a time.
  // This has been broken up for us in the block_info struct, so we don't have
  // any calculation to do here, but it is important to note.

  // to speed up some of the random access memory we do, we copy col_sizes and col_offsets
  // to shared memory for each of the blocks that we work on

  constexpr unsigned stages_count = NUM_BLOCKS_PER_KERNEL_LOADED;
  auto group = cooperative_groups::this_thread_block();
  extern __shared__ int8_t shared_data[];
  int8_t *shared[stages_count] = {shared_data, shared_data + shmem_used_per_block};

  __shared__ cuda::barrier<cuda::thread_scope_block> block_barrier[stages_count];
  if (group.thread_rank() == 0) {
    for (int i = 0; i < stages_count; ++i) {
      init(&block_barrier[i], group.size());
    }
  }

  group.sync();

  auto blocks_remaining = std::min(num_block_infos - blockIdx.x * NUM_BLOCKS_PER_KERNEL_TO_COLUMNS,
                                   (uint)NUM_BLOCKS_PER_KERNEL_TO_COLUMNS);

  auto get_admin_data_sizes = [col_size_size = sizeof(decltype(*_col_sizes)),
                               col_offset_size = sizeof(decltype(*_col_offsets))](
                                  int const num_cols,
                                  int const num_rows) -> std::tuple<size_type, size_type> {
    auto const col_size_bytes = num_cols * col_size_size;
    auto const col_offset_bytes = num_cols * col_offset_size;

    return {col_size_bytes, col_offset_bytes};
  };

  size_t fetch;
  size_t subset;
  for (subset = fetch = 0; subset < blocks_remaining; ++subset) {
    // Fetch ahead up to stages_count subsets
    fetch_blocks_for_row_to_column(fetch, subset, stages_count, stages_count, blocks_remaining,
                                   block_infos, _col_sizes, _col_offsets, row_offsets, input_data,
                                   shared, group, block_barrier);

    auto &subset_barrier = block_barrier[subset % stages_count];
    // ensure our data is ready
    subset_barrier.arrive_and_wait();

    auto const block = block_infos[blockIdx.x * NUM_BLOCKS_PER_KERNEL_TO_COLUMNS + subset];

    auto const rows_in_block = block.num_rows();
    auto const cols_in_block = block.num_cols();

    auto [col_size_bytes, col_offset_bytes] = get_admin_data_sizes(cols_in_block, rows_in_block);
    // auto shared_row_offsets = shared[subset];
    auto shared_col_sizes = reinterpret_cast<size_type *>(shared[subset % stages_count]);
    auto shared_col_offsets =
        reinterpret_cast<size_type *>(&shared[subset % stages_count][col_size_bytes]);

    auto const shared_row_offset = align_offset(col_size_bytes + col_offset_bytes, 8);

    auto block_row_size = block.get_row_size(_col_offsets, _col_sizes);

    // now we copy from shared memory to final destination.
    // the data is laid out in rows in shared memory, so the reads
    // for a column will be "vertical". Because of this and the different
    // sizes for each column, this portion is handled on row/column basis.
    // to prevent each thread working on a single row and also to ensure
    // that all threads can do work in the case of more threads than rows,
    // we do a global index instead of a double for loop with col/row.
    for (int index = threadIdx.x; index < rows_in_block * cols_in_block; index += blockDim.x) {
      auto const relative_col = index % cols_in_block;
      auto const relative_row = index / cols_in_block;
      auto const absolute_col = relative_col + block.start_col;
      auto const absolute_row = relative_row + block.start_row;

      auto const shared_memory_row_offset = block_row_size * relative_row;
      auto const shared_memory_offset = shared_col_offsets[relative_col] - shared_col_offsets[0] +
                                        shared_memory_row_offset + shared_row_offset;
      auto const column_size = shared_col_sizes[relative_col];

      int8_t *shmem_src = &shared[subset % stages_count][shared_memory_offset];
      int8_t *dst = &output_data[absolute_col][absolute_row * column_size];

      cuda::memcpy_async(dst, shmem_src, column_size, subset_barrier);
    }
    group.sync();
  }

  // wait on the last copies to complete
  for (uint i = 0; i < std::min(stages_count, blocks_remaining); ++i) {
    block_barrier[i].arrive_and_wait();
  }
}

/**
 * @brief copy data from row-based format to cudf columns
 *
 * @param num_rows total number of rows in the table
 * @param num_columns total number of columns in the table
 * @param shmem_used_per_block amount of shared memory that is used by a block
 * @param offsets
 * @param output_nm
 * @param validity_offsets offset into input data row for validity data
 * @param block_infos information about the blocks of work
 * @param num_block_infos number of infos in blocks array
 * @param input_data pointer to input data
 *
 */
__global__ void copy_validity_to_columns(
    const size_type num_rows, const size_type num_columns, const size_type shmem_used_per_block,
    const size_type *row_offsets, cudf::bitmask_type **output_nm, const size_type validity_offset,
    const block_info *block_infos, const size_type num_block_infos, const int8_t *input_data) {
  extern __shared__ int8_t shared_data[];
  int8_t *shared_blocks[NUM_VALIDITY_BLOCKS_PER_KERNEL_LOADED] = {
      shared_data, shared_data + shmem_used_per_block / 2};

  // per conversation with DaveB
  // each thread of warp reads a single byte of validity - so we read 32 bytes
  // then ballot_sync the bits and write the result to shmem
  // after we fill shared mem memcpy it out in a blob.
  // probably need knobs for number of rows vs columns to balance read/write
  auto group = cooperative_groups::this_thread_block();

  int const blocks_remaining =
      std::min(num_block_infos - blockIdx.x * NUM_VALIDITY_BLOCKS_PER_KERNEL,
               (uint)NUM_VALIDITY_BLOCKS_PER_KERNEL);

  __shared__ cuda::barrier<cuda::thread_scope_block>
      shared_block_barriers[NUM_VALIDITY_BLOCKS_PER_KERNEL_LOADED];
  if (group.thread_rank() == 0) {
    for (int i = 0; i < NUM_VALIDITY_BLOCKS_PER_KERNEL_LOADED; ++i) {
      init(&shared_block_barriers[i], group.size());
    }
  }

  group.sync();

  for (int validity_block = 0; validity_block < blocks_remaining; ++validity_block) {
    auto const validity_index = validity_block % NUM_VALIDITY_BLOCKS_PER_KERNEL_LOADED;
    if (validity_block != validity_index) {
      shared_block_barriers[validity_index].arrive_and_wait();
    }
    int8_t *this_shared_block = shared_blocks[validity_block % 2];
    auto const block = block_infos[blockIdx.x * NUM_VALIDITY_BLOCKS_PER_KERNEL + validity_block];
    auto const block_start_col = block.start_col;
    auto const block_start_row = block.start_row;

    auto const num_block_cols = block.num_cols();
    auto const num_block_rows = block.num_rows();

    auto const num_sections_x = (num_block_cols + 7) / 8;
    auto const num_sections_y = (num_block_rows + 31) / 32;
    auto const validity_data_col_length = align_offset(num_sections_y, 4);
    auto const total_sections = num_sections_x * num_sections_y;

    int const warp_id = threadIdx.x / detail::warp_size;
    int const lane_id = threadIdx.x % detail::warp_size;
    auto const warps_per_block = std::max(1u, blockDim.x / detail::warp_size);

    // the block is divided into sections. A warp operates on a section at a time.
    for (int my_section_idx = warp_id; my_section_idx < total_sections;
         my_section_idx += warps_per_block) {
      // convert to rows and cols
      auto const section_x = my_section_idx % num_sections_x;
      auto const section_y = my_section_idx / num_sections_x;

      auto const relative_col = section_x * 8;
      auto const relative_row = section_y * 32 + lane_id;
      auto const absolute_col = relative_col + block_start_col;
      auto const absolute_row = relative_row + block_start_row;
      auto const rows_left = num_rows - absolute_row;

      auto const participation_mask = __ballot_sync(0xFFFFFFFF, absolute_row < num_rows);

      if (absolute_row < num_rows) {
        auto const my_byte =
            input_data[row_offsets[absolute_row] + validity_offset + absolute_col / 8];

        // so every thread that is participating in the warp has a byte, but it's row-based
        // data and we need it in column-based. So we shiffle the bits around to make
        // the bytes we actually write.
        for (int i = 0, byte_mask = 1; i < 8 && relative_col + i < num_columns;
             ++i, byte_mask <<= 1) {
          auto validity_data = __ballot_sync(participation_mask, my_byte & byte_mask);
          // lead thread in each warp writes data
          if (threadIdx.x % detail::warp_size == 0) {
            auto const validity_write_offset =
                validity_data_col_length * (relative_col + i) + relative_row / 8;

            if (rows_left <= 8) {
              // write byte
              this_shared_block[validity_write_offset] = validity_data & 0xFF;
            } else if (rows_left <= 16) {
              // write int16
              *reinterpret_cast<int16_t *>(&this_shared_block[validity_write_offset]) =
                  validity_data & 0xFFFF;
            } else if (rows_left <= 24) {
              // write int16 and then int8
              *reinterpret_cast<int16_t *>(&this_shared_block[validity_write_offset]) =
                  validity_data & 0xFFFF;
              shared_data[validity_write_offset + 2] = (validity_data >> 16) & 0xFF;
            } else {
              // write int32
              *reinterpret_cast<int32_t *>(&this_shared_block[validity_write_offset]) =
                  validity_data;
            }
          }
        }
      }
    }

    // make sure entire block has finished copy
    group.sync();

    // now async memcpy the shared
    for (int col = block.start_col + threadIdx.x; col <= block.end_col; col += blockDim.x) {
      auto const relative_col = col - block.start_col;

      cuda::memcpy_async(
          output_nm[col] + word_index(block_start_row),
          &this_shared_block[validity_data_col_length * relative_col],
          util::div_rounding_up_unsafe(num_block_rows, 8),
          shared_block_barriers[validity_block % NUM_VALIDITY_BLOCKS_PER_KERNEL_LOADED]);
    }
  }

  // wait for last blocks of data to arrive
  auto const num_blocks_to_wait = blocks_remaining > NUM_VALIDITY_BLOCKS_PER_KERNEL_LOADED ?
                                      NUM_VALIDITY_BLOCKS_PER_KERNEL_LOADED :
                                      blocks_remaining;
  for (int validity_block = 0; validity_block < num_blocks_to_wait; ++validity_block) {
    shared_block_barriers[validity_block].arrive_and_wait();
  }
}

#endif // !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700

/**
 * Calculate the dimensions of the kernel for fixed width only columns.
 * @param [in] num_columns the number of columns being copied.
 * @param [in] num_rows the number of rows being copied.
 * @param [in] size_per_row the size each row takes up when padded.
 * @param [out] blocks the size of the blocks for the kernel
 * @param [out] threads the size of the threads for the kernel
 * @return the size in bytes of shared memory needed for each block.
 */
static int calc_fixed_width_kernel_dims(const cudf::size_type num_columns,
                                        const cudf::size_type num_rows,
                                        const cudf::size_type size_per_row, dim3 &blocks,
                                        dim3 &threads) {
  // We have found speed degrades when a thread handles more than 4 columns.
  // Each block is 2 dimensional. The y dimension indicates the columns.
  // We limit this to 32 threads in the y dimension so we can still
  // have at least 32 threads in the x dimension (1 warp) which should
  // result in better coalescing of memory operations. We also
  // want to guarantee that we are processing a multiple of 32 threads
  // in the x dimension because we use atomic operations at the block
  // level when writing validity data out to main memory, and that would
  // need to change if we split a word of validity data between blocks.
  int y_block_size = (num_columns + 3) / 4; // cudf::util::div_rounding_up_safe(num_columns, 4);
  if (y_block_size > 32)
    y_block_size = 32;
  int x_possible_block_size = 1024 / y_block_size;
  // 48KB is the default setting for shared memory per block according to the cuda tutorials
  // If someone configures the GPU to only have 16 KB this might not work.
  int max_shared_size = 48 * 1024;
  int max_block_size = max_shared_size / size_per_row;
  // If we don't have enough shared memory there is no point in having more threads
  // per block that will just sit idle
  max_block_size = max_block_size > x_possible_block_size ? x_possible_block_size : max_block_size;
  // Make sure that the x dimension is a multiple of 32 this not only helps
  // coalesce memory access it also lets us do a ballot sync for validity to write
  // the data back out the warp level.  If x is a multiple of 32 then each thread in the y
  // dimension is associated with one or more warps, that should correspond to the validity
  // words directly.
  int block_size = (max_block_size / 32) * 32;
  CUDF_EXPECTS(block_size != 0, "Row size is too large to fit in shared memory");

  int num_blocks = (num_rows + block_size - 1) / block_size;
  if (num_blocks < 1) {
    num_blocks = 1;
  } else if (num_blocks > 10240) {
    // The maximum number of blocks supported in the x dimension is 2 ^ 31 - 1
    // but in practice haveing too many can cause some overhead that I don't totally
    // understand. Playing around with this haveing as little as 600 blocks appears
    // to be able to saturate memory on V100, so this is an order of magnitude higher
    // to try and future proof this a bit.
    num_blocks = 10240;
  }
  blocks.x = num_blocks;
  blocks.y = 1;
  blocks.z = 1;
  threads.x = block_size;
  threads.y = y_block_size;
  threads.z = 1;
  return size_per_row * block_size;
}

/**
 * When converting to rows it is possible that the size of the table was too big to fit
 * in a single column. This creates an output column for a subset of the rows in a table
 * going from start row and containing the next num_rows.  Most of the parameters passed
 * into this function are common between runs and should be calculated once.
 */
static std::unique_ptr<cudf::column>
fixed_width_convert_to_rows(const cudf::size_type start_row, const cudf::size_type num_rows,
                            const cudf::size_type num_columns, const cudf::size_type size_per_row,
                            rmm::device_uvector<cudf::size_type> &column_start,
                            rmm::device_uvector<cudf::size_type> &column_size,
                            rmm::device_uvector<const int8_t *> &input_data,
                            rmm::device_uvector<const cudf::bitmask_type *> &input_nm,
                            const cudf::scalar &zero, const cudf::scalar &scalar_size_per_row,
                            rmm::cuda_stream_view stream, rmm::mr::device_memory_resource *mr) {
  int64_t total_allocation = size_per_row * num_rows;
  // We made a mistake in the split somehow
  CUDF_EXPECTS(total_allocation < std::numeric_limits<int>::max(), "Table is too large to fit!");

  // Allocate and set the offsets row for the byte array
  std::unique_ptr<cudf::column> offsets =
      cudf::detail::sequence(num_rows + 1, zero, scalar_size_per_row, stream);

  std::unique_ptr<cudf::column> data = cudf::make_numeric_column(
      cudf::data_type(cudf::type_id::INT8), static_cast<cudf::size_type>(total_allocation),
      cudf::mask_state::UNALLOCATED, stream, mr);

  dim3 blocks;
  dim3 threads;
  int shared_size =
      detail::calc_fixed_width_kernel_dims(num_columns, num_rows, size_per_row, blocks, threads);

  copy_from_fixed_width_columns<<<blocks, threads, shared_size, stream.value()>>>(
      start_row, num_rows, num_columns, size_per_row, column_start.data(), column_size.data(),
      input_data.data(), input_nm.data(), data->mutable_view().data<int8_t>());

  return cudf::make_lists_column(num_rows, std::move(offsets), std::move(data), 0,
                                 rmm::device_buffer{0, rmm::cuda_stream_default, mr}, stream, mr);
}

static cudf::data_type get_data_type(const cudf::column_view &v) {
  return v.type();
}

static inline bool are_all_fixed_width(std::vector<cudf::data_type> const &schema) {
  return std::all_of(schema.begin(), schema.end(),
                     [](const cudf::data_type &t) { return cudf::is_fixed_width(t); });
}

/**
 * Given a set of fixed width columns, calculate how the data will be laid out in memory.
 * @param [in] schema the types of columns that need to be laid out.
 * @param [out] column_start the byte offset where each column starts in the row.
 * @param [out] column_size the size in bytes of the data for each columns in the row.
 * @return the size in bytes each row needs.
 */
static inline int32_t compute_fixed_width_layout(std::vector<cudf::data_type> const &schema,
                                                 std::vector<cudf::size_type> &column_start,
                                                 std::vector<cudf::size_type> &column_size) {
  // We guarantee that the start of each column is 64-bit aligned so anything can go
  // there, but to make the code simple we will still do an alignment for it.
  int32_t at_offset = 0;
  for (auto col = schema.begin(); col < schema.end(); col++) {
    cudf::size_type s = cudf::size_of(*col);
    column_size.emplace_back(s);
    std::size_t allocation_needed = s;
    std::size_t alignment_needed = allocation_needed; // They are the same for fixed width types
    at_offset = align_offset(at_offset, alignment_needed);
    column_start.emplace_back(at_offset);
    at_offset += allocation_needed;
  }

  // Now we need to add in space for validity
  // Eventually we can think about nullable vs not nullable, but for now we will just always add it
  // in
  int32_t validity_bytes_needed =
      (schema.size() + 7) / 8; // cudf::util::div_rounding_up_safe<int32_t>(schema.size(), 8);
  // validity comes at the end and is byte aligned so we can pack more in.
  at_offset += validity_bytes_needed;
  // Now we need to pad the end so all rows are 64 bit aligned
  return align_offset(at_offset, 8); // 8 bytes (64 bits)
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700

template <typename iterator>
static size_type compute_column_information(iterator begin, iterator end,
                                            std::vector<size_type> &column_starts,
                                            std::vector<size_type> &column_sizes) //,
// std::function<void(T)> nested_type_cb)
{
  size_type fixed_width_size_per_row = 0;
  for (auto cv = begin; cv != end; ++cv) {
    auto col_type = std::get<0>(*cv);
    bool nested_type = col_type.id() == type_id::LIST || col_type.id() == type_id::STRING;

    //    if (nested_type && nested_type_cb) { nested_type_cb(cv->get<1>()); }

    // a list or string column will write a single uint64
    // of data here for offset/length
    auto col_size = nested_type ? 8 : size_of(col_type);

    // align size for this type
    std::size_t const alignment_needed = col_size; // They are the same for fixed width types
    fixed_width_size_per_row = detail::align_offset(fixed_width_size_per_row, alignment_needed);
    column_starts.push_back(fixed_width_size_per_row);
    column_sizes.push_back(col_size);
    fixed_width_size_per_row += col_size;
  }

  auto validity_offset = detail::align_offset(fixed_width_size_per_row, 4);
  column_starts.push_back(validity_offset);

  return fixed_width_size_per_row;
}

std::vector<detail::block_info>
build_validity_block_infos(size_type const &num_columns, size_type const &num_rows,
                           size_type const &shmem_limit_per_block,
                           std::vector<row_batch> const &row_batches) {
  auto const desired_rows_and_columns = (int)sqrt(shmem_limit_per_block);
  auto const column_stride = align_offset(
      [&]() {
        if (desired_rows_and_columns > num_columns) {
          // not many columns, group it into 8s and ship it off
          return std::min(8, num_columns);
        } else {
          return util::round_down_safe(desired_rows_and_columns, 8);
        }
      }(),
      8);
  // we fit as much as we can given the column stride
  auto const row_stride = std::min(num_rows, shmem_limit_per_block * 8 / column_stride);

  std::vector<detail::block_info> validity_block_infos;
  for (int col = 0; col < num_columns; col += column_stride) {
    int current_window_row_batch = 0;
    int rows_left_in_batch = row_batches[current_window_row_batch].row_count;
    int row = 0;
    while (row < num_rows) {
      if (rows_left_in_batch == 0) {
        current_window_row_batch++;
        rows_left_in_batch = row_batches[current_window_row_batch].row_count;
      }
      int const window_height = std::min(row_stride, rows_left_in_batch);

      validity_block_infos.emplace_back(detail::block_info{
          col, row, std::min(col + column_stride - 1, num_columns - 1), row + window_height - 1});
      row += window_height;
      rows_left_in_batch -= window_height;
    }
  }

  return validity_block_infos;
}

std::vector<block_info> build_block_infos(std::vector<size_type> const &column_sizes,
                                          std::vector<size_type> const &column_starts,
                                          std::vector<row_batch> const &row_batches,
                                          size_type const total_number_of_rows,
                                          size_type const &shmem_limit_per_block) {
  std::vector<block_info> block_infos;

  // block infos are organized with the windows going "down" the columns
  // this provides the most coalescing of memory access
  int current_window_width = 0;
  int current_window_start_col = 0;

  // build the blocks for a specific set of columns
  auto build_blocks = [&block_infos, &row_batches, total_number_of_rows](
                          int const start_col, int const end_col, int const desired_window_height) {
    int current_window_start_row = 0;
    int current_window_row_batch = 0;
    int rows_left_in_batch = row_batches[current_window_row_batch].row_count;
    int i = 0;
    while (i < total_number_of_rows) {
      if (rows_left_in_batch == 0) {
        current_window_row_batch++;
        rows_left_in_batch = row_batches[current_window_row_batch].row_count;
      }
      int const window_height = std::min(desired_window_height, rows_left_in_batch);

      block_infos.emplace_back(detail::block_info{
          start_col, current_window_start_row, end_col,
          std::min(current_window_start_row + window_height - 1, total_number_of_rows - 1),
          current_window_row_batch});

      i += window_height;
      current_window_start_row += window_height;
      rows_left_in_batch -= window_height;
    }
  };

  // the ideal window height has lots of 8-byte reads and 8-byte writes. The optimal read/write
  // would be memory cache line sized access, but since other blocks will read/write the edges this
  // may not turn out to be overly important. For now, we will attempt to build a square window as
  // far as byte sizes. x * y = shared_mem_size. Which translates to x^2 = shared_mem_size since we
  // want them equal, so height and width are sqrt(shared_mem_size). The trick is that it's in
  // bytes, not rows or columns.
  size_type const optimal_square_len = size_type(sqrt(shmem_limit_per_block));
  int const window_height =
      std::clamp(util::round_up_safe<int>(
                     optimal_square_len <= (size_type)column_sizes.size() ?
                         std::min(optimal_square_len / column_sizes[0], total_number_of_rows) :
                         row_batches[0].row_count / 2,
                     32),
                 1, row_batches[0].row_count);

  auto calc_admin_data_size = [](int num_cols) -> size_type {
    // admin data is the column sizes and column start information.
    // this is copied to shared memory as well and needs to be accounted for
    // in the window calculation.
    return num_cols * sizeof(size_type) + num_cols * sizeof(size_type);
  };

  int row_size = 0;

  // march each column and build the blocks of appropriate sizes
  for (unsigned int col = 0; col < column_sizes.size(); ++col) {
    auto const col_size = column_sizes[col];

    // align size for this type
    std::size_t alignment_needed = col_size; // They are the same for fixed width types
    auto row_size_aligned = detail::align_offset(row_size, alignment_needed);
    auto row_size_with_this_col = row_size_aligned + col_size;
    auto row_size_with_end_pad = detail::align_offset(row_size_with_this_col, 8);

    if (row_size_with_end_pad * window_height +
            calc_admin_data_size(col - current_window_start_col) >
        shmem_limit_per_block) {
      // too large, close this window, generate vertical blocks and restart
      build_blocks(current_window_start_col, col - 1, window_height);
      row_size =
          detail::align_offset((column_starts[col] + column_sizes[col]) & 7, alignment_needed);
      row_size += col_size; // alignment required for shared memory window boundary to match
                            // alignment of output row
      current_window_start_col = col;
      current_window_width = 0;
    } else {
      row_size = row_size_with_this_col;
      current_window_width++;
    }
  }

  // build last set of blocks
  if (current_window_width > 0) {
    build_blocks(current_window_start_col, (int)column_sizes.size() - 1, window_height);
  }

  return block_infos;
}

#endif // #if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700

} // namespace detail

std::vector<std::unique_ptr<cudf::column>> convert_to_rows(cudf::table_view const &tbl,
                                                           rmm::cuda_stream_view stream,
                                                           rmm::mr::device_memory_resource *mr) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
  // not scientifically chosen - the ideal window is long enough to allow coalesced reads of the
  // data, but small enough that multiple columns fit in memory so the writes can coalese as well.
  // Potential optimization for window sizes.
  const size_type num_columns = tbl.num_columns();
  const size_type num_rows = tbl.num_rows();

  int device_id;
  CUDA_TRY(cudaGetDevice(&device_id));
  int total_shmem;
  CUDA_TRY(cudaDeviceGetAttribute(&total_shmem, cudaDevAttrMaxSharedMemoryPerBlock, device_id));

  // TODO: kernels fail to launch if we use all the available shared memory.
  total_shmem -= 1024;

  int shmem_limit_per_block = total_shmem / NUM_VALIDITY_BLOCKS_PER_KERNEL_LOADED;

  // break up the work into blocks, which are a starting and ending row/col #.
  // this window size is calculated based on the shared memory size available
  // we want a single block to fill up the entire shared memory space available
  // for the transpose-like conversion.

  // There are two different processes going on here. The GPU conversion of the data
  // and the writing of the data into the list of byte columns that are a maximum of
  // 2 gigs each due to offset maximum size. The GPU conversion portion has to understand
  // this limitation because the column must own the data inside and as a result it must be
  // a distinct allocation for that column. Copying the data into these final buffers would
  // be prohibitively expensive, so care is taken to ensure the GPU writes to the proper buffer.
  // The windows are broken at the boundaries of specific rows based on the row sizes up
  // to that point. These are row batches and they are decided first before building the
  // windows so the windows can be properly cut around them.

  // Get the pointers to the input columnar data ready
  std::vector<int8_t const *> input_data;
  std::vector<bitmask_type const *> input_nm;
  input_data.reserve(num_columns);
  input_nm.reserve(num_columns);
  for (size_type column_number = 0; column_number < num_columns; column_number++) {
    column_view cv = tbl.column(column_number);
    auto const col_type = cv.type();
    bool nested_type = col_type.id() == type_id::LIST || col_type.id() == type_id::STRING;

    if (!nested_type) {
      input_data.emplace_back(cv.data<int8_t>());
      input_nm.emplace_back(cv.null_mask());
    }
  }

  auto dev_input_data = make_device_uvector_async(input_data, stream, mr);
  auto dev_input_nm = make_device_uvector_async(input_nm, stream, mr);

  std::vector<size_type> row_sizes;     // size of each row in bytes including any alignment padding
  std::vector<size_type> row_offsets;   // offset from the start of the data to this row
  std::vector<size_type> column_sizes;  // byte size of each column
  std::vector<size_type> column_starts; // offset of column inside a row including alignment
  std::vector<column_view>
      variable_width_columns; // list of the variable width columns in the table
  row_sizes.reserve(num_rows);
  row_offsets.reserve(num_rows);
  column_sizes.reserve(num_columns);
  column_starts.reserve(num_columns + 1); // we add a final offset for validity data start

  auto iter =
      thrust::make_transform_iterator(thrust::make_counting_iterator(0),
                                      [&tbl](auto i) -> std::tuple<data_type, column_view const> {
                                        return std::make_tuple(tbl.column(i).type(), tbl.column(i));
                                      });

  size_type fixed_width_size_per_row =
      detail::compute_column_information(iter, iter + num_columns, column_starts,
                                         column_sizes); //,
  //    [&variable_width_columns](column_view const &cv) { variable_width_columns.push_back(cv); });
  /*  size_type fixed_width_size_per_row = 0;
    for (int col = 0; col < num_columns; ++col) {
      auto cv          = tbl.column(col);
      auto col_type    = cv.type();
      bool nested_type = col_type.id() == type_id::LIST || col_type.id() == type_id::STRING;

      if (nested_type) { variable_width_columns.push_back(cv); }

      // a list or string column will write a single uint64
      // of data here for offset/length
      auto col_size = nested_type ? 8 : size_of(col_type);

      // align size for this type
      std::size_t const alignment_needed = col_size;  // They are the same for fixed width types
      fixed_width_size_per_row = detail::align_offset(fixed_width_size_per_row, alignment_needed);
      column_starts.push_back(fixed_width_size_per_row);
      column_sizes.push_back(col_size);
      fixed_width_size_per_row += col_size;
    }*/

  auto dev_col_sizes = make_device_uvector_async(column_sizes, stream, mr);
  auto dev_col_starts = make_device_uvector_async(column_starts, stream, mr);

  std::vector<detail::row_batch> row_batches;

  auto calculate_variable_width_row_data_size = [](int const row) {
    // each level of variable-width data will add an offset/length
    // uint64 of data. The first of which is inside the fixed-width
    // data itself and needs to be aligned based on what is around
    // that data. This is handled above with the fixed-width calculations
    // for that reason. We may still need to add more of these offset/length
    // combinations if the nesting is deeper than one level as these
    // will be included in the variable-width data blob at the end of the
    // row.
    return 0;
    /*      auto c = variable_width_columns[col];
            while (true) {
              auto col_offsets   = c.child(0).data<size_type>();
              auto col_data_size = size_of(c.child(1).type());
              std::size_t alignment_needed  = col_data_size;

            row_sizes[row] += (col_offsets[row + 1] - col_offsets[row]) * col_data_size;
            if (c.num_children() == 0) {
              break;
            }
            c = c.child(1);
          }
    */
  };

  uint64_t row_batch_size = 0;
  uint64_t total_table_size = 0;
  size_type row_batch_rows = 0;
  uint64_t row_offset = 0;

  // fixed_width_size_per_row is the size of the fixed-width portion of a row. We need to then
  // calculate the size of each row's variable-width data and validity as well.
  auto validity_size = num_bitmask_words(num_columns) * 4;
  for (int row = 0; row < num_rows; ++row) {
    auto aligned_row_batch_size =
        detail::align_offset(row_batch_size, 8); // rows are 8 byte aligned
    row_sizes[row] = fixed_width_size_per_row;
    // validity is byte aligned
    row_sizes[row] += validity_size;
    // variable width data is 8-byte aligned
    row_sizes[row] = detail::align_offset(row_sizes[row], 8) +
                     calculate_variable_width_row_data_size(row); // rows are 8 byte aligned

    if ((uint64_t)aligned_row_batch_size + row_sizes[row] >
        (uint64_t)std::numeric_limits<size_type>::max()) {
      // a new batch starts at the last 32-row boundary
      row_batches.push_back(
          detail::row_batch{static_cast<size_type>(row_batch_size), row_batch_rows & ~31});
      row_batch_size = 0;
      row_batch_rows = row_batch_rows & 31;
      row_offset = 0;
      aligned_row_batch_size = 0;
    }
    row_offset = detail::align_offset(row_offset, 8); // rows are 8 byte aligned
    row_offsets.push_back(row_offset);
    row_batch_size = aligned_row_batch_size + row_sizes[row];
    row_offset += row_sizes[row];
    total_table_size = detail::align_offset(total_table_size, 8); // rows are 8 byte aligned
    total_table_size += row_sizes[row];
    row_batch_rows++;
  }
  if (row_batch_size > 0) {
    row_batches.push_back(
        detail::row_batch{static_cast<size_type>(row_batch_size), row_batch_rows});
  }

  auto dev_row_offsets = make_device_uvector_async(row_offsets, stream, mr);

  std::vector<rmm::device_buffer> output_buffers;
  std::vector<int8_t *> output_data;
  output_data.reserve(row_batches.size());
  for (uint i = 0; i < row_batches.size(); ++i) {
    rmm::device_buffer temp(row_batches[i].num_bytes, stream, mr);
    output_data.push_back(static_cast<int8_t *>(temp.data()));
    output_buffers.push_back(std::move(temp));
  }
  auto dev_output_data = make_device_uvector_async(output_data, stream, mr);

  std::vector<detail::block_info> block_infos =
      build_block_infos(column_sizes, column_starts, row_batches, num_rows, shmem_limit_per_block);

  auto dev_block_infos = make_device_uvector_async(block_infos, stream, mr);

  // blast through the entire table and convert it
  dim3 blocks(util::div_rounding_up_unsafe(block_infos.size(), NUM_BLOCKS_PER_KERNEL_FROM_COLUMNS));
  dim3 threads(256);

  detail::copy_from_columns<<<blocks, threads, total_shmem, stream.value()>>>(
      num_rows, num_columns, shmem_limit_per_block, block_infos.size(), dev_input_data.data(),
      dev_col_sizes.data(), dev_col_starts.data(), dev_block_infos.data(), dev_row_offsets.data(),
      reinterpret_cast<int8_t **>(dev_output_data.data()));

  auto validity_block_infos =
      build_validity_block_infos(num_columns, num_rows, shmem_limit_per_block, row_batches);

  auto dev_validity_block_infos = make_device_uvector_async(validity_block_infos, stream, mr);
  dim3 validity_blocks(
      util::div_rounding_up_unsafe(validity_block_infos.size(), NUM_BLOCKS_PER_KERNEL_TO_COLUMNS));
  dim3 validity_threads(std::min(validity_block_infos.size() * 32, 128lu));
  detail::copy_validity_from_columns<<<validity_blocks, validity_threads, total_shmem,
                                       stream.value()>>>(
      num_rows, num_columns, shmem_limit_per_block, dev_row_offsets.data(), dev_output_data.data(),
      column_starts.back(), dev_validity_block_infos.data(), validity_block_infos.size(),
      dev_input_nm.data());

  // split up the output buffer into multiple buffers based on row batch sizes
  // and create list of byte columns
  int offset_offset = 0;
  std::vector<std::unique_ptr<cudf::column>> ret;
  for (uint i = 0; i < row_batches.size(); ++i) {
    // compute offsets for this row batch
    std::vector<size_type> offset_vals;
    offset_vals.reserve(row_batches[i].row_count + 1);
    size_type cur_offset = 0;
    offset_vals.push_back(cur_offset);
    for (int row = 0; row < row_batches[i].row_count; ++row) {
      cur_offset = detail::align_offset(cur_offset, 8) + row_sizes[row + offset_offset];
      offset_vals.push_back(cur_offset);
    }
    offset_offset += row_batches[i].row_count;

    auto dev_offsets = make_device_uvector_async(offset_vals, stream, mr);
    auto offsets = std::make_unique<column>(data_type{type_id::INT32},
                                            (size_type)offset_vals.size(), dev_offsets.release());

    auto data = std::make_unique<column>(data_type{cudf::type_id::INT8}, row_batches[i].num_bytes,
                                         std::move(output_buffers[i]));

    ret.push_back(
        cudf::make_lists_column(row_batches[i].row_count, std::move(offsets), std::move(data), 0,
                                rmm::device_buffer{0, rmm::cuda_stream_default, mr}, stream, mr));
  }

  return ret;
#else
  CUDF_FAIL("Column to row conversion optimization requires volta or later hardware.");
  return {};
#endif // #if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
}

std::vector<std::unique_ptr<cudf::column>>
old_convert_to_rows(cudf::table_view const &tbl, rmm::cuda_stream_view stream,
                    rmm::mr::device_memory_resource *mr) {
  const cudf::size_type num_columns = tbl.num_columns();

  std::vector<cudf::data_type> schema;
  schema.resize(num_columns);
  std::transform(tbl.begin(), tbl.end(), schema.begin(), detail::get_data_type);

  if (detail::are_all_fixed_width(schema)) {
    std::vector<cudf::size_type> column_start;
    std::vector<cudf::size_type> column_size;

    int32_t size_per_row = detail::compute_fixed_width_layout(schema, column_start, column_size);
    auto dev_column_start = make_device_uvector_async(column_start, stream, mr);
    auto dev_column_size = make_device_uvector_async(column_size, stream, mr);

    int32_t max_rows_per_batch = std::numeric_limits<int>::max() / size_per_row;
    // Make the number of rows per batch a multiple of 32 so we don't have to worry about
    // splitting validity at a specific row offset.  This might change in the future.
    max_rows_per_batch = (max_rows_per_batch / 32) * 32;

    cudf::size_type num_rows = tbl.num_rows();

    // Get the pointers to the input columnar data ready
    std::vector<const int8_t *> input_data;
    std::vector<cudf::bitmask_type const *> input_nm;
    for (cudf::size_type column_number = 0; column_number < num_columns; column_number++) {
      cudf::column_view cv = tbl.column(column_number);
      input_data.emplace_back(cv.data<int8_t>());
      input_nm.emplace_back(cv.null_mask());
    }
    auto dev_input_data = make_device_uvector_async(input_data, stream, mr);
    auto dev_input_nm = make_device_uvector_async(input_nm, stream, mr);

    using ScalarType = cudf::scalar_type_t<cudf::size_type>;
    auto zero = cudf::make_numeric_scalar(cudf::data_type(cudf::type_id::INT32), stream.value());
    zero->set_valid_async(true, stream);
    static_cast<ScalarType *>(zero.get())->set_value(0, stream);

    auto step = cudf::make_numeric_scalar(cudf::data_type(cudf::type_id::INT32), stream.value());
    step->set_valid_async(true, stream);
    static_cast<ScalarType *>(step.get())
        ->set_value(static_cast<cudf::size_type>(size_per_row), stream);

    std::vector<std::unique_ptr<cudf::column>> ret;
    for (cudf::size_type row_start = 0; row_start < num_rows; row_start += max_rows_per_batch) {
      cudf::size_type row_count = num_rows - row_start;
      row_count = row_count > max_rows_per_batch ? max_rows_per_batch : row_count;
      ret.emplace_back(detail::fixed_width_convert_to_rows(
          row_start, row_count, num_columns, size_per_row, dev_column_start, dev_column_size,
          dev_input_data, dev_input_nm, *zero, *step, stream, mr));
    }

    return ret;
  } else {
    CUDF_FAIL("Only fixed width types are currently supported");
  }
}

std::unique_ptr<cudf::table> convert_from_rows(cudf::lists_column_view const &input,
                                               std::vector<cudf::data_type> const &schema,
                                               rmm::cuda_stream_view stream,
                                               rmm::mr::device_memory_resource *mr) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
  // verify that the types are what we expect
  cudf::column_view child = input.child();
  cudf::type_id list_type = child.type().id();
  CUDF_EXPECTS(list_type == cudf::type_id::INT8 || list_type == cudf::type_id::UINT8,
               "Only a list of bytes is supported as input");

  cudf::size_type num_columns = schema.size();
  cudf::size_type num_rows = input.parent().size();

  int device_id;
  CUDA_TRY(cudaGetDevice(&device_id));
  int total_shmem;
  CUDA_TRY(cudaDeviceGetAttribute(&total_shmem, cudaDevAttrMaxSharedMemoryPerBlock, device_id));

  // TODO: unable to launch a kernel with all shared used
  total_shmem -= 1024;
  int shmem_limit_per_block = total_shmem / NUM_VALIDITY_BLOCKS_PER_KERNEL_LOADED;

  std::vector<cudf::size_type> column_starts;
  std::vector<cudf::size_type> column_sizes;

  auto iter = thrust::make_transform_iterator(thrust::make_counting_iterator(0), [&schema](auto i) {
    return std::make_tuple(schema[i], nullptr);
  });
  size_type fixed_width_size_per_row = detail::compute_column_information(
      iter, iter + num_columns, column_starts, column_sizes); //, [](void *) {});

  size_type validity_size = num_bitmask_words(num_columns) * 4;

  size_type row_size = detail::align_offset(fixed_width_size_per_row + validity_size, 8);

  // Ideally we would check that the offsets are all the same, etc. but for now
  // this is probably fine
  CUDF_EXPECTS(row_size * num_rows == child.size(), "The layout of the data appears to be off");
  auto dev_col_starts = make_device_uvector_async(column_starts, stream, mr);
  auto dev_col_sizes = make_device_uvector_async(column_sizes, stream, mr);

  // build the row_batches from the passed in list column
  std::vector<detail::row_batch> row_batches;

  row_batches.push_back(detail::row_batch{child.size(), num_rows});

  // Allocate the columns we are going to write into
  std::vector<std::unique_ptr<cudf::column>> output_columns;
  std::vector<int8_t *> output_data;
  std::vector<cudf::bitmask_type *> output_nm;
  for (cudf::size_type i = 0; i < num_columns; i++) {
    auto column = cudf::make_fixed_width_column(schema[i], num_rows,
                                                cudf::mask_state::UNINITIALIZED, stream, mr);
    auto mut = column->mutable_view();
    output_data.emplace_back(mut.data<int8_t>());
    output_nm.emplace_back(mut.null_mask());
    output_columns.emplace_back(std::move(column));
  }

  auto dev_output_data = make_device_uvector_async(output_data, stream, mr);
  auto dev_output_nm = make_device_uvector_async(output_nm, stream, mr);

  std::vector<detail::block_info> block_infos =
      build_block_infos(column_sizes, column_starts, row_batches, num_rows, shmem_limit_per_block);

  auto dev_block_infos = make_device_uvector_async(block_infos, stream, mr);

  dim3 blocks(util::div_rounding_up_unsafe(block_infos.size(), NUM_BLOCKS_PER_KERNEL_TO_COLUMNS));
#if defined(DEBUG)
  dim3 threads(std::min(std::min(128, shmem_limit_per_block / 8), (int)child.size()));
#else
  dim3 threads(std::min(256, (int)child.size()));
#endif
  detail::copy_to_columns<<<blocks, threads, total_shmem, stream.value()>>>(
      num_rows, num_columns, shmem_limit_per_block, input.offsets().data<size_type>(),
      dev_output_data.data(), dev_col_sizes.data(), dev_col_starts.data(), dev_block_infos.data(),
      block_infos.size(), child.data<int8_t>());

  auto const desired_rows_and_columns = (int)sqrt(shmem_limit_per_block);
  auto const column_stride = [&]() {
    if (desired_rows_and_columns > num_columns) {
      // not many columns, group it into 8s and ship it off
      return std::min(8, num_columns);
    } else {
      return util::round_down_safe(desired_rows_and_columns, 8);
    }
  }();
  auto const row_stride = [&]() {
    // we fit as much as we can, we know the column stride now, so calculate the row
    return std::min(num_rows, util::round_down_safe(shmem_limit_per_block * 8 / column_stride, 32));
    /*    if (desired_rows_and_columns > num_rows) {
          return std::min(32, num_rows);
        } else {
          return util::round_down_safe(desired_rows_and_columns, 32);
        }*/
  }();
  std::vector<detail::block_info> validity_block_infos;
  for (int col = 0; col < num_columns; col += column_stride) {
    for (int row = 0; row < num_rows; row += row_stride) {
      validity_block_infos.emplace_back(
          detail::block_info{col, row, std::min(col + column_stride - 1, num_columns - 1),
                             std::min(row + row_stride - 1, num_rows - 1)});
    }
  }
  auto dev_validity_block_infos = make_device_uvector_async(validity_block_infos, stream, mr);
  dim3 validity_blocks(
      util::div_rounding_up_unsafe(validity_block_infos.size(), NUM_BLOCKS_PER_KERNEL_TO_COLUMNS));

  dim3 validity_threads(std::min(validity_block_infos.size() * 32, 128lu));
  detail::
      copy_validity_to_columns<<<validity_blocks, validity_threads, total_shmem, stream.value()>>>(
          num_rows, num_columns, shmem_limit_per_block, input.offsets().data<size_type>(),
          dev_output_nm.data(), column_starts.back(), dev_validity_block_infos.data(),
          validity_block_infos.size(), child.data<int8_t>());

  return std::make_unique<cudf::table>(std::move(output_columns));
#else
  CUDF_FAIL("Row to column conversion optimization requires volta or later hardware.");
  return {};
#endif // #if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
}

std::unique_ptr<cudf::table> old_convert_from_rows(cudf::lists_column_view const &input,
                                                   std::vector<cudf::data_type> const &schema,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::mr::device_memory_resource *mr) {
  // verify that the types are what we expect
  cudf::column_view child = input.child();
  cudf::type_id list_type = child.type().id();
  CUDF_EXPECTS(list_type == cudf::type_id::INT8 || list_type == cudf::type_id::UINT8,
               "Only a list of bytes is supported as input");

  cudf::size_type num_columns = schema.size();

  if (detail::are_all_fixed_width(schema)) {
    std::vector<cudf::size_type> column_start;
    std::vector<cudf::size_type> column_size;

    cudf::size_type num_rows = input.parent().size();
    int32_t size_per_row = detail::compute_fixed_width_layout(schema, column_start, column_size);

    // Ideally we would check that the offsets are all the same, etc. but for now
    // this is probably fine
    CUDF_EXPECTS(size_per_row * num_rows == child.size(),
                 "The layout of the data appears to be off");
    auto dev_column_start = make_device_uvector_async(column_start, stream);
    auto dev_column_size = make_device_uvector_async(column_size, stream);

    // Allocate the columns we are going to write into
    std::vector<std::unique_ptr<cudf::column>> output_columns;
    std::vector<int8_t *> output_data;
    std::vector<cudf::bitmask_type *> output_nm;
    for (cudf::size_type i = 0; i < num_columns; i++) {
      auto column = cudf::make_fixed_width_column(schema[i], num_rows,
                                                  cudf::mask_state::UNINITIALIZED, stream, mr);
      auto mut = column->mutable_view();
      output_data.emplace_back(mut.data<int8_t>());
      output_nm.emplace_back(mut.null_mask());
      output_columns.emplace_back(std::move(column));
    }

    auto dev_output_data = make_device_uvector_async(output_data, stream, mr);
    auto dev_output_nm = make_device_uvector_async(output_nm, stream, mr);

    dim3 blocks;
    dim3 threads;
    int shared_size =
        detail::calc_fixed_width_kernel_dims(num_columns, num_rows, size_per_row, blocks, threads);

    detail::copy_to_fixed_width_columns<<<blocks, threads, shared_size, stream.value()>>>(
        num_rows, num_columns, size_per_row, dev_column_start.data(), dev_column_size.data(),
        dev_output_data.data(), dev_output_nm.data(), child.data<int8_t>());

    return std::make_unique<cudf::table>(std::move(output_columns));
  } else {
    CUDF_FAIL("Only fixed width types are currently supported");
  }
}

} // namespace cudf
