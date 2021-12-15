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
#include <cstdarg>
#include <cstdint>
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
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/sequence.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/lists/lists_column_device_view.cuh>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <thrust/binary_search.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scan.h>

#include "row_conversion.hpp"

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
constexpr auto NUM_BLOCKS_PER_KERNEL_FROM_ROWS = 2;
constexpr auto NUM_BLOCKS_PER_KERNEL_TO_ROWS = 2;
constexpr auto NUM_BLOCKS_PER_KERNEL_LOADED = 2;
constexpr auto NUM_VALIDITY_BLOCKS_PER_KERNEL = 8;
constexpr auto NUM_VALIDITY_BLOCKS_PER_KERNEL_LOADED = 2;

constexpr auto JCUDF_ROW_ALIGNMENT = 8;
constexpr auto MAX_BATCH_SIZE = std::numeric_limits<cudf::size_type>::max();

// needed to suppress warning about cuda::barrier
#pragma nv_diag_suppress static_var_with_dynamic_init
#endif

using namespace cudf;
using detail::make_device_uvector_async;
using rmm::device_uvector;
namespace cudf {
namespace java {
namespace detail {

/**
 * @brief Copies data from row-based JCUDF format to column-based cudf format.
 *
 * This optimized version of the conversion is faster for fixed-width tables
 * that do not have more than 100 columns.
 *
 * @param num_rows number of rows in the incoming table
 * @param num_columns number of columns in the incoming table
 * @param row_size length in bytes of each row
 * @param input_offset_in_row offset to each row of data
 * @param num_bytes total number of bytes in the incoming data
 * @param output_data array of pointers to the output data
 * @param output_nm array of pointers to the output null masks
 * @param input_data pointing to the incoming row data
 */
__global__ void
copy_from_rows_fixed_width_optimized(const size_type num_rows, const size_type num_columns,
                                     const size_type row_size, const size_type *input_offset_in_row,
                                     const size_type *num_bytes, int8_t **output_data,
                                     bitmask_type **output_nm, const int8_t *input_data) {
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

  size_type const rows_per_group = blockDim.x;
  size_type const row_group_start = blockIdx.x;
  size_type const row_group_stride = gridDim.x;
  size_type const row_group_end = (num_rows + rows_per_group - 1) / rows_per_group + 1;

  extern __shared__ int8_t shared_data[];

  // Because we are copying fixed width only data and we stride the rows
  // this thread will always start copying from shared data in the same place
  int8_t *row_tmp = &shared_data[row_size * threadIdx.x];
  int8_t *row_vld_tmp = &row_tmp[input_offset_in_row[num_columns - 1] + num_bytes[num_columns - 1]];

  for (auto row_group_index = row_group_start; row_group_index < row_group_end;
       row_group_index += row_group_stride) {
    // Step 1: Copy the data into shared memory
    // We know row_size is always aligned with and a multiple of int64_t;
    int64_t *long_shared = reinterpret_cast<int64_t *>(shared_data);
    int64_t const *long_input = reinterpret_cast<int64_t const *>(input_data);

    auto const shared_output_index = threadIdx.x + (threadIdx.y * blockDim.x);
    auto const shared_output_stride = blockDim.x * blockDim.y;
    auto const row_index_end = std::min(num_rows, ((row_group_index + 1) * rows_per_group));
    auto const num_rows_in_group = row_index_end - (row_group_index * rows_per_group);
    auto const shared_length = row_size * num_rows_in_group;

    size_type const shared_output_end = shared_length / sizeof(int64_t);

    auto const start_input_index = (row_size * row_group_index * rows_per_group) / sizeof(int64_t);

    for (size_type shared_index = shared_output_index; shared_index < shared_output_end;
         shared_index += shared_output_stride) {
      long_shared[shared_index] = long_input[start_input_index + shared_index];
    }
    // Wait for all of the data to be in shared memory
    __syncthreads();

    // Step 2 copy the data back out

    // Within the row group there should be 1 thread for each row.  This is a
    // requirement for launching the kernel
    auto const row_index = (row_group_index * rows_per_group) + threadIdx.x;
    // But we might not use all of the threads if the number of rows does not go
    // evenly into the thread count. We don't want those threads to exit yet
    // because we may need them to copy data in for the next row group.
    uint32_t active_mask = __ballot_sync(0xffffffff, row_index < num_rows);
    if (row_index < num_rows) {
      auto const col_index_start = threadIdx.y;
      auto const col_index_stride = blockDim.y;
      for (auto col_index = col_index_start; col_index < num_columns;
           col_index += col_index_stride) {
        auto const col_size = num_bytes[col_index];
        int8_t const *col_tmp = &(row_tmp[input_offset_in_row[col_index]]);
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
            auto const output_offset = col_size * row_index;
            // TODO this should just not be supported for fixed width columns, but just in case...
            for (auto b = 0; b < col_size; b++) {
              col_output[b + output_offset] = col_tmp[b];
            }
            break;
          }
        }

        bitmask_type *nm = output_nm[col_index];
        int8_t *valid_byte = &row_vld_tmp[col_index / 8];
        size_type byte_bit_offset = col_index % 8;
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

__global__ void copy_to_rows_fixed_width_optimized(
    const size_type start_row, const size_type num_rows, const size_type num_columns,
    const size_type row_size, const size_type *output_offset_in_row, const size_type *num_bytes,
    const int8_t **input_data, const bitmask_type **input_nm, int8_t *output_data) {
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

  size_type rows_per_group = blockDim.x;
  size_type row_group_start = blockIdx.x;
  size_type row_group_stride = gridDim.x;
  size_type row_group_end = (num_rows + rows_per_group - 1) / rows_per_group + 1;

  extern __shared__ int8_t shared_data[];

  // Because we are copying fixed width only data and we stride the rows
  // this thread will always start copying to shared data in the same place
  int8_t *row_tmp = &shared_data[row_size * threadIdx.x];
  int8_t *row_vld_tmp =
      &row_tmp[output_offset_in_row[num_columns - 1] + num_bytes[num_columns - 1]];

  for (size_type row_group_index = row_group_start; row_group_index < row_group_end;
       row_group_index += row_group_stride) {
    // Within the row group there should be 1 thread for each row.  This is a
    // requirement for launching the kernel
    size_type row_index = start_row + (row_group_index * rows_per_group) + threadIdx.x;
    // But we might not use all of the threads if the number of rows does not go
    // evenly into the thread count. We don't want those threads to exit yet
    // because we may need them to copy data back out.
    if (row_index < (start_row + num_rows)) {
      size_type col_index_start = threadIdx.y;
      size_type col_index_stride = blockDim.y;
      for (size_type col_index = col_index_start; col_index < num_columns;
           col_index += col_index_stride) {
        size_type col_size = num_bytes[col_index];
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
            size_type input_offset = col_size * row_index;
            // TODO this should just not be supported for fixed width columns, but just in case...
            for (size_type b = 0; b < col_size; b++) {
              col_tmp[b] = col_input[b + input_offset];
            }
            break;
          }
        }
        // atomicOr only works on 32 bit or 64 bit  aligned values, and not byte aligned
        // so we have to rewrite the addresses to make sure that it is 4 byte aligned
        int8_t *valid_byte = &row_vld_tmp[col_index / 8];
        size_type byte_bit_offset = col_index % 8;
        uint64_t fixup_bytes = reinterpret_cast<uint64_t>(valid_byte) % 4;
        int32_t *valid_int = reinterpret_cast<int32_t *>(valid_byte - fixup_bytes);
        size_type int_bit_offset = byte_bit_offset + (fixup_bytes * 8);
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

    size_type shared_input_index = threadIdx.x + (threadIdx.y * blockDim.x);
    size_type shared_input_stride = blockDim.x * blockDim.y;
    size_type row_index_end = ((row_group_index + 1) * rows_per_group);
    if (row_index_end > num_rows) {
      row_index_end = num_rows;
    }
    size_type num_rows_in_group = row_index_end - (row_group_index * rows_per_group);
    size_type shared_length = row_size * num_rows_in_group;

    size_type shared_input_end = shared_length / sizeof(int64_t);

    size_type start_output_index = (row_size * row_group_index * rows_per_group) / sizeof(int64_t);

    for (size_type shared_index = shared_input_index; shared_index < shared_input_end;
         shared_index += shared_input_stride) {
      long_output[start_output_index + shared_index] = long_shared[shared_index];
    }
    __syncthreads();
    // Go for the next round
  }
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700

/**
 * @brief The GPU blocks work on one or more block_info structs of data.
 *        This structure defined the workspace for the block.
 *
 */
struct block_info {
  int start_col;
  int start_row;
  int end_col;
  int end_row;
  int batch_number;

  constexpr size_type get_shared_row_size(size_type const *const col_offsets,
                                          size_type const *const col_sizes) const {
    return util::round_up_unsafe(col_offsets[end_col] + col_sizes[end_col] - col_offsets[start_col],
                                 JCUDF_ROW_ALIGNMENT);
  }
  constexpr size_type num_cols() const { return end_col - start_col + 1; }

  constexpr size_type num_rows() const { return end_row - start_row + 1; }
};

/**
 * @brief Returning rows is done in a byte cudf column. This is limited in size by
 *        `size_type` and so output is broken into batches of rows that fit inside
 *        this limit.
 *
 */
struct row_batch {
  size_type num_bytes;                   // number of bytes in this batch
  size_type row_count;                   // number of rows in the batch
  device_uvector<size_type> row_offsets; // offsets column of output cudf column
};

/**
 * @brief Holds information about the batches of data to be processed
 *
 */
struct batch_data {
  device_uvector<size_type> batch_row_offsets; // offset column of returned cudf column
  std::vector<size_type>
      batch_row_boundaries;           // row numbers for different batches like 0, 10000, 20000
  std::vector<row_batch> row_batches; // information about each batch such as byte count
};

/**
 * @brief copy data from cudf columns into JCUDF format, which is row-based
 *
 * @param num_rows total number of rows in the table
 * @param num_columns total number of columns in the table
 * @param shmem_used_per_block shared memory amount each `block_info` is using
 * @param block_infos span of `block_info` structs the define the work
 * @param input_data pointer to raw table data
 * @param col_sizes array of sizes for each element in a column - one per column
 * @param col_offsets offset into input data row for each column's start
 * @param row_offsets offset to a specific row in the output data
 * @param output_data pointer to output data
 *
 */
__global__ void copy_to_rows(const size_type num_rows, const size_type num_columns,
                             const size_type shmem_used_per_block,
                             device_span<const block_info> block_infos, const int8_t **input_data,
                             const size_type *col_sizes, const size_type *col_offsets,
                             const size_type *row_offsets, int8_t **output_data) {
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
      std::min(static_cast<uint>(block_infos.size()) - blockIdx.x * NUM_BLOCKS_PER_KERNEL_TO_ROWS,
               static_cast<uint>(NUM_BLOCKS_PER_KERNEL_TO_ROWS));

  size_t fetch;
  size_t subset;
  for (subset = fetch = 0; subset < blocks_remaining; ++subset) {
    // Fetch ahead up to stages_count subsets
    for (; fetch < blocks_remaining && fetch < (subset + stages_count); ++fetch) {
      auto const fetch_block = block_infos[blockIdx.x * NUM_BLOCKS_PER_KERNEL_TO_ROWS + fetch];
      auto const num_fetch_cols = fetch_block.num_cols();
      auto const num_fetch_rows = fetch_block.num_rows();
      auto const num_elements_in_block = num_fetch_cols * num_fetch_rows;
      auto const fetch_block_row_size = fetch_block.get_shared_row_size(col_offsets, col_sizes);
      auto const starting_column_offset = col_offsets[fetch_block.start_col];
      auto &fetch_barrier = block_barrier[fetch % NUM_BLOCKS_PER_KERNEL_LOADED];

      // wait for the last use of the memory to be completed
      if (fetch >= NUM_BLOCKS_PER_KERNEL_LOADED) {
        fetch_barrier.arrive_and_wait();
      }

      // to do the copy we need to do n column copies followed by m element copies OR
      // we have to do m element copies followed by r row copies. When going from column
      // to row it is much easier to copy by elements first otherwise we would need a running
      // total of the column sizes for our block, which isn't readily available. This makes it
      // more appealing to copy element-wise from input data into shared matching the end layout
      // and do row-based memcopies out.

      auto const shared_buffer_base = shared[fetch % stages_count];
      for (auto el = static_cast<int>(threadIdx.x); el < num_elements_in_block; el += blockDim.x) {
        auto const relative_col = el / num_fetch_rows;
        auto const relative_row = el % num_fetch_rows;
        auto const absolute_col = relative_col + fetch_block.start_col;
        auto const absolute_row = relative_row + fetch_block.start_row;
        auto const col_size = col_sizes[absolute_col];
        auto const col_offset = col_offsets[absolute_col];
        auto const relative_col_offset = col_offset - starting_column_offset;

        auto const shared_offset = relative_row * fetch_block_row_size + relative_col_offset;
        auto const input_src = input_data[absolute_col] + col_size * absolute_row;

        // copy the element from global memory
        switch (col_size) {
          case 2:
            cuda::memcpy_async(&shared_buffer_base[shared_offset], input_src,
                               cuda::aligned_size_t<2>(col_size), fetch_barrier);
            break;
          case 4:
            cuda::memcpy_async(&shared_buffer_base[shared_offset], input_src,
                               cuda::aligned_size_t<4>(col_size), fetch_barrier);
            break;
          case 8:
            cuda::memcpy_async(&shared_buffer_base[shared_offset], input_src,
                               cuda::aligned_size_t<8>(col_size), fetch_barrier);
            break;
          default:
            cuda::memcpy_async(&shared_buffer_base[shared_offset], input_src, col_size,
                               fetch_barrier);
            break;
        }
      }
    }

    auto &subset_barrier = block_barrier[subset % NUM_BLOCKS_PER_KERNEL_LOADED];
    subset_barrier.arrive_and_wait();

    auto const block = block_infos[blockIdx.x * NUM_BLOCKS_PER_KERNEL_TO_ROWS + subset];
    auto const block_row_size = block.get_shared_row_size(col_offsets, col_sizes);
    auto const column_offset = col_offsets[block.start_col];
    auto const block_output_buffer = output_data[block.batch_number];

    // copy entire row 8 bytes at a time
    auto const chunks_per_row = util::div_rounding_up_unsafe(block_row_size, 8);
    auto const total_chunks = chunks_per_row * block.num_rows();

    for (auto i = threadIdx.x; i < total_chunks; i += blockDim.x) {
      // determine source address of my chunk
      auto const relative_row = i / chunks_per_row;
      auto const relative_chunk_offset = (i % chunks_per_row) * 8;
      auto const output_dest = block_output_buffer + row_offsets[relative_row + block.start_row] +
                               column_offset + relative_chunk_offset;
      auto const input_src =
          &shared[subset % stages_count][block_row_size * relative_row + relative_chunk_offset];

      cuda::memcpy_async(output_dest, input_src, cuda::aligned_size_t<8>(8), subset_barrier);
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
 * @param row_offsets offset to a specific row in the output data
 * @param output_data pointer to output data, partitioned by data size
 * @param validity_offsets offset into input data row for validity data
 * @param block_infos information about the blocks of work
 * @param input_data pointer to input data
 *
 */
__global__ void copy_validity_to_rows(const size_type num_rows, const size_type num_columns,
                                      const size_type shmem_used_per_block,
                                      const size_type *row_offsets, int8_t **output_data,
                                      const size_type validity_offset,
                                      device_span<const block_info> block_infos,
                                      const bitmask_type **input_nm) {
  extern __shared__ int8_t shared_data[];
  int8_t *shared_blocks[NUM_VALIDITY_BLOCKS_PER_KERNEL_LOADED] = {
      shared_data, shared_data + shmem_used_per_block / 2};

  using cudf::detail::warp_size;

  // each thread of warp reads a single int32 of validity - so we read 128 bytes
  // then ballot_sync the bits and write the result to shmem
  // after we fill shared mem memcpy it out in a blob.
  // probably need knobs for number of rows vs columns to balance read/write
  auto group = cooperative_groups::this_thread_block();

  int const blocks_remaining =
      std::min(static_cast<uint>(block_infos.size()) - blockIdx.x * NUM_VALIDITY_BLOCKS_PER_KERNEL,
               static_cast<uint>(NUM_VALIDITY_BLOCKS_PER_KERNEL));

  __shared__ cuda::barrier<cuda::thread_scope_block>
      shared_block_barriers[NUM_VALIDITY_BLOCKS_PER_KERNEL_LOADED];
  if (group.thread_rank() == 0) {
    for (int i = 0; i < NUM_VALIDITY_BLOCKS_PER_KERNEL_LOADED; ++i) {
      init(&shared_block_barriers[i], group.size());
    }
  }

  group.sync();

  for (int validity_block = 0; validity_block < blocks_remaining; ++validity_block) {
    if (validity_block >= NUM_VALIDITY_BLOCKS_PER_KERNEL_LOADED) {
      shared_block_barriers[validity_block % NUM_VALIDITY_BLOCKS_PER_KERNEL_LOADED]
          .arrive_and_wait();
    }
    int8_t *this_shared_block = shared_blocks[validity_block % 2];
    auto block = block_infos[blockIdx.x * NUM_VALIDITY_BLOCKS_PER_KERNEL + validity_block];

    auto const num_block_cols = block.num_cols();
    auto const num_block_rows = block.num_rows();

    auto const num_sections_x = util::div_rounding_up_unsafe(num_block_cols, 32);
    auto const num_sections_y = util::div_rounding_up_unsafe(num_block_rows, 32);
    auto const validity_data_row_length =
        util::round_up_unsafe(util::div_rounding_up_unsafe(num_block_cols, 8), JCUDF_ROW_ALIGNMENT);
    auto const total_sections = num_sections_x * num_sections_y;

    int const warp_id = threadIdx.x / warp_size;
    int const lane_id = threadIdx.x % warp_size;
    auto const warps_per_block = std::max(1u, blockDim.x / warp_size);

    // the block is divided into sections. A warp operates on a section at a time.
    for (int my_section_idx = warp_id; my_section_idx < total_sections;
         my_section_idx += warps_per_block) {
      // convert to rows and cols
      auto const section_x = my_section_idx % num_sections_x;
      auto const section_y = my_section_idx / num_sections_x;
      auto const relative_col = section_x * 32 + lane_id;
      auto const relative_row = section_y * 32;
      auto const absolute_col = relative_col + block.start_col;
      auto const absolute_row = relative_row + block.start_row;
      auto const participation_mask = __ballot_sync(0xFFFFFFFF, absolute_col < num_columns);

      if (absolute_col < num_columns) {
        auto my_data = input_nm[absolute_col] != nullptr ?
                           input_nm[absolute_col][absolute_row / 32] :
                           std::numeric_limits<uint32_t>::max();

        // every thread that is participating in the warp has 4 bytes, but it's column-based
        // data and we need it in row-based. So we shuffle the bits around with ballot_sync to
        // make the bytes we actually write.
        bitmask_type dw_mask = 1;
        for (int i = 0; i < 32 && relative_row + i < num_rows; ++i, dw_mask <<= 1) {
          auto validity_data = __ballot_sync(participation_mask, my_data & dw_mask);
          // lead thread in each warp writes data
          auto const validity_write_offset =
              validity_data_row_length * (relative_row + i) + relative_col / 8;
          if (threadIdx.x % warp_size == 0) {
            *reinterpret_cast<int32_t *>(&this_shared_block[validity_write_offset]) = validity_data;
          }
        }
      }
    }

    // make sure entire block has finished copy
    group.sync();

    auto const output_data_base =
        output_data[block.batch_number] + validity_offset + block.start_col / 8;

    // now async memcpy the shared memory out to the final destination 4 bytes at a time since we do
    // 32-row chunks
    auto const row_bytes = util::div_rounding_up_unsafe(num_block_cols, 8);
    auto const chunks_per_row = util::div_rounding_up_unsafe(row_bytes, 8);
    auto const total_chunks = chunks_per_row * block.num_rows();
    auto &subset_barrier =
        shared_block_barriers[validity_block % NUM_VALIDITY_BLOCKS_PER_KERNEL_LOADED];
    auto const tail_bytes = row_bytes % 8;

    for (auto i = threadIdx.x; i < total_chunks; i += blockDim.x) {
      // determine source address of my chunk
      auto const relative_row = i / chunks_per_row;
      auto const col_chunk = i % chunks_per_row;
      auto const relative_chunk_offset = col_chunk * 8;
      auto const output_dest =
          output_data_base + row_offsets[relative_row + block.start_row] + relative_chunk_offset;
      auto const input_src =
          &this_shared_block[validity_data_row_length * relative_row + relative_chunk_offset];

      if (tail_bytes > 0 && col_chunk == chunks_per_row - 1)
        cuda::memcpy_async(output_dest, input_src, tail_bytes, subset_barrier);
      else
        cuda::memcpy_async(output_dest, input_src, cuda::aligned_size_t<8>(8), subset_barrier);
    }
  }

  // wait for last blocks of data to arrive
  for (int validity_block = 0;
       validity_block < blocks_remaining % NUM_VALIDITY_BLOCKS_PER_KERNEL_LOADED;
       ++validity_block) {
    shared_block_barriers[validity_block].arrive_and_wait();
  }
}

/**
 * @brief copy data from row-based format to cudf columns
 *
 * @param num_rows total number of rows in the table
 * @param num_columns total number of columns in the table
 * @param shmem_used_per_block amount of shared memory that is used by a block
 * @param row_offsets offset to a specific row in the input data
 * @param output_data pointers to column data
 * @param col_sizes array of sizes for each element in a column - one per column
 * @param col_offsets offset into input data row for each column's start
 * @param block_infos information about the blocks of work
 * @param input_data pointer to input data
 *
 */
__global__ void copy_from_rows(const size_type num_rows, const size_type num_columns,
                               const size_type shmem_used_per_block, const size_type *row_offsets,
                               int8_t **output_data, const size_type *col_sizes,
                               const size_type *col_offsets,
                               device_span<const block_info> block_infos,
                               const int8_t *input_data) {
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

  __shared__ cuda::barrier<cuda::thread_scope_block> block_barrier[NUM_BLOCKS_PER_KERNEL_LOADED];
  if (group.thread_rank() == 0) {
    for (int i = 0; i < NUM_BLOCKS_PER_KERNEL_LOADED; ++i) {
      init(&block_barrier[i], group.size());
    }
  }

  group.sync();

  auto blocks_remaining =
      std::min(static_cast<uint>(block_infos.size()) - blockIdx.x * NUM_BLOCKS_PER_KERNEL_FROM_ROWS,
               static_cast<uint>(NUM_BLOCKS_PER_KERNEL_FROM_ROWS));

  size_t fetch_index;
  size_t processing_index;
  for (processing_index = fetch_index = 0; processing_index < blocks_remaining;
       ++processing_index) {
    // Fetch ahead up to stages_count groups
    for (; fetch_index < static_cast<size_t>(blocks_remaining) &&
           fetch_index < (processing_index + stages_count);
         ++fetch_index) {
      auto const fetch_block =
          block_infos[blockIdx.x * NUM_BLOCKS_PER_KERNEL_FROM_ROWS + fetch_index];
      auto const fetch_block_start_row = fetch_block.start_row;
      auto const starting_col_offset = col_offsets[fetch_block.start_col];
      auto const fetch_block_row_size = fetch_block.get_shared_row_size(col_offsets, col_sizes);
      auto &fetch_barrier = block_barrier[fetch_index % NUM_BLOCKS_PER_KERNEL_LOADED];

      // if we have fetched all buffers, we need to wait for processing
      // to complete on them before we can use them again
      if (fetch_index > NUM_BLOCKS_PER_KERNEL_LOADED) {
        fetch_barrier.arrive_and_wait();
      }

      for (auto row = fetch_block_start_row + static_cast<int>(threadIdx.x);
           row <= fetch_block.end_row; row += blockDim.x) {
        auto shared_offset = (row - fetch_block_start_row) * fetch_block_row_size;
        // copy the data
        cuda::memcpy_async(&shared[fetch_index % stages_count][shared_offset],
                           &input_data[row_offsets[row] + starting_col_offset],
                           fetch_block_row_size, fetch_barrier);
      }
    }

    auto &processing_barrier = block_barrier[processing_index % NUM_BLOCKS_PER_KERNEL_LOADED];

    // ensure our data is ready
    processing_barrier.arrive_and_wait();

    auto const block = block_infos[blockIdx.x * NUM_BLOCKS_PER_KERNEL_FROM_ROWS + processing_index];
    auto const rows_in_block = block.num_rows();
    auto const cols_in_block = block.num_cols();
    auto const block_row_size = block.get_shared_row_size(col_offsets, col_sizes);

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
      auto const shared_memory_offset =
          col_offsets[absolute_col] - col_offsets[block.start_col] + shared_memory_row_offset;
      auto const column_size = col_sizes[absolute_col];

      int8_t *shmem_src = &shared[processing_index % stages_count][shared_memory_offset];
      int8_t *dst = &output_data[absolute_col][absolute_row * column_size];

      cuda::memcpy_async(dst, shmem_src, column_size, processing_barrier);
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
 * @param row_offsets offset to a specific row in the input data
 * @param output_nm pointers to null masks for columns
 * @param validity_offsets offset into input data row for validity data
 * @param block_infos information about the blocks of work
 * @param input_data pointer to input data
 *
 */
__global__ void copy_validity_from_rows(const size_type num_rows, const size_type num_columns,
                                        const size_type shmem_used_per_block,
                                        const size_type *row_offsets, bitmask_type **output_nm,
                                        const size_type validity_offset,
                                        device_span<const block_info> block_infos,
                                        const int8_t *input_data) {
  extern __shared__ int8_t shared_data[];
  int8_t *shared_blocks[NUM_VALIDITY_BLOCKS_PER_KERNEL_LOADED] = {
      shared_data, shared_data + shmem_used_per_block / 2};

  using cudf::detail::warp_size;

  // each thread of warp reads a single byte of validity - so we read 32 bytes
  // then ballot_sync the bits and write the result to shmem
  // after we fill shared mem memcpy it out in a blob.
  // probably need knobs for number of rows vs columns to balance read/write
  auto group = cooperative_groups::this_thread_block();

  int const blocks_remaining =
      std::min(static_cast<uint>(block_infos.size()) - blockIdx.x * NUM_VALIDITY_BLOCKS_PER_KERNEL,
               static_cast<uint>(NUM_VALIDITY_BLOCKS_PER_KERNEL));

  __shared__ cuda::barrier<cuda::thread_scope_block>
      shared_block_barriers[NUM_VALIDITY_BLOCKS_PER_KERNEL_LOADED];
  if (group.thread_rank() == 0) {
    for (int i = 0; i < NUM_VALIDITY_BLOCKS_PER_KERNEL_LOADED; ++i) {
      init(&shared_block_barriers[i], group.size());
    }
  }

  group.sync();

  for (int validity_block = 0; validity_block < blocks_remaining; ++validity_block) {
    if (validity_block >= NUM_VALIDITY_BLOCKS_PER_KERNEL_LOADED) {
      auto const validity_index = validity_block % NUM_VALIDITY_BLOCKS_PER_KERNEL_LOADED;
      shared_block_barriers[validity_index].arrive_and_wait();
    }
    int8_t *this_shared_block = shared_blocks[validity_block % 2];
    auto const block = block_infos[blockIdx.x * NUM_VALIDITY_BLOCKS_PER_KERNEL + validity_block];
    auto const block_start_col = block.start_col;
    auto const block_start_row = block.start_row;
    auto const num_block_cols = block.num_cols();
    auto const num_block_rows = block.num_rows();
    auto const num_sections_x = util::div_rounding_up_safe(num_block_cols, 8);
    auto const num_sections_y = util::div_rounding_up_safe(num_block_rows, 32);
    auto const validity_data_col_length = num_sections_y * 4; // words to bytes
    auto const total_sections = num_sections_x * num_sections_y;
    int const warp_id = threadIdx.x / warp_size;
    int const lane_id = threadIdx.x % warp_size;
    auto const warps_per_block = std::max(1u, blockDim.x / warp_size);

    // the block is divided into sections. A warp operates on a section at a time.
    for (int my_section_idx = warp_id; my_section_idx < total_sections;
         my_section_idx += warps_per_block) {
      // convert section to row and col
      auto const section_x = my_section_idx % num_sections_x;
      auto const section_y = my_section_idx / num_sections_x;
      auto const relative_col = section_x * 8;
      auto const relative_row = section_y * 32 + lane_id;
      auto const absolute_col = relative_col + block_start_col;
      auto const absolute_row = relative_row + block_start_row;

      auto const participation_mask = __ballot_sync(0xFFFFFFFF, absolute_row < num_rows);

      if (absolute_row < num_rows) {
        auto const my_byte =
            input_data[row_offsets[absolute_row] + validity_offset + absolute_col / 8];

        // so every thread that is participating in the warp has a byte, but it's row-based
        // data and we need it in column-based. So we shuffle the bits around to make
        // the bytes we actually write.
        for (int i = 0, byte_mask = 1; i < 8 && relative_col + i < num_columns;
             ++i, byte_mask <<= 1) {
          auto validity_data = __ballot_sync(participation_mask, my_byte & byte_mask);
          // lead thread in each warp writes data
          if (threadIdx.x % warp_size == 0) {
            auto const validity_write_offset =
                validity_data_col_length * (relative_col + i) + relative_row / 8;

            *reinterpret_cast<int32_t *>(&this_shared_block[validity_write_offset]) = validity_data;
          }
        }
      }
    }

    // make sure entire block has finished copy
    group.sync();

    // now async memcpy the shared memory out to the final destination 8 bytes at a time
    auto const col_bytes = util::div_rounding_up_unsafe(num_block_rows, 8);
    auto const chunks_per_col = util::div_rounding_up_unsafe(col_bytes, 8);
    auto const total_chunks = chunks_per_col * num_block_cols;
    auto &subset_barrier =
        shared_block_barriers[validity_block % NUM_VALIDITY_BLOCKS_PER_KERNEL_LOADED];
    auto const tail_bytes = col_bytes % 8;

    for (auto i = threadIdx.x; i < total_chunks; i += blockDim.x) {
      // determine source address of my chunk
      auto const relative_col = i / chunks_per_col;
      auto const row_chunk = i % chunks_per_col;
      auto const absolute_col = relative_col + block_start_col;
      auto const relative_chunk_byte_offset = row_chunk * 8;
      auto const output_dest =
          output_nm[absolute_col] + word_index(block_start_row) + row_chunk * 2;
      auto const input_src =
          &this_shared_block[validity_data_col_length * relative_col + relative_chunk_byte_offset];

      if (tail_bytes > 0 && row_chunk == chunks_per_col - 1)
        cuda::memcpy_async(output_dest, input_src, tail_bytes, subset_barrier);
      else
        cuda::memcpy_async(output_dest, input_src, cuda::aligned_size_t<8>(8), subset_barrier);
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
 * @brief Calculate the dimensions of the kernel for fixed width only columns.
 *
 * @param [in] num_columns the number of columns being copied.
 * @param [in] num_rows the number of rows being copied.
 * @param [in] size_per_row the size each row takes up when padded.
 * @param [out] blocks the size of the blocks for the kernel
 * @param [out] threads the size of the threads for the kernel
 * @return the size in bytes of shared memory needed for each block.
 */
static int calc_fixed_width_kernel_dims(const size_type num_columns, const size_type num_rows,
                                        const size_type size_per_row, dim3 &blocks, dim3 &threads) {
  // We have found speed degrades when a thread handles more than 4 columns.
  // Each block is 2 dimensional. The y dimension indicates the columns.
  // We limit this to 32 threads in the y dimension so we can still
  // have at least 32 threads in the x dimension (1 warp) which should
  // result in better coalescing of memory operations. We also
  // want to guarantee that we are processing a multiple of 32 threads
  // in the x dimension because we use atomic operations at the block
  // level when writing validity data out to main memory, and that would
  // need to change if we split a word of validity data between blocks.
  int const y_block_size = min(util::div_rounding_up_safe(num_columns, 4), 32);
  int const x_possible_block_size = 1024 / y_block_size;
  // 48KB is the default setting for shared memory per block according to the cuda tutorials
  // If someone configures the GPU to only have 16 KB this might not work.
  int const max_shared_size = 48 * 1024;
  // If we don't have enough shared memory there is no point in having more threads
  // per block that will just sit idle
  auto const max_block_size = std::min(x_possible_block_size, max_shared_size / size_per_row);
  // Make sure that the x dimension is a multiple of 32 this not only helps
  // coalesce memory access it also lets us do a ballot sync for validity to write
  // the data back out the warp level.  If x is a multiple of 32 then each thread in the y
  // dimension is associated with one or more warps, that should correspond to the validity
  // words directly.
  int const block_size = (max_block_size / 32) * 32;
  CUDF_EXPECTS(block_size != 0, "Row size is too large to fit in shared memory");

  // The maximum number of blocks supported in the x dimension is 2 ^ 31 - 1
  // but in practice haveing too many can cause some overhead that I don't totally
  // understand. Playing around with this haveing as little as 600 blocks appears
  // to be able to saturate memory on V100, so this is an order of magnitude higher
  // to try and future proof this a bit.
  int const num_blocks = std::clamp((num_rows + block_size - 1) / block_size, 1, 10240);

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
static std::unique_ptr<column> fixed_width_convert_to_rows(
    const size_type start_row, const size_type num_rows, const size_type num_columns,
    const size_type size_per_row, rmm::device_uvector<size_type> &column_start,
    rmm::device_uvector<size_type> &column_size, rmm::device_uvector<const int8_t *> &input_data,
    rmm::device_uvector<const bitmask_type *> &input_nm, const scalar &zero,
    const scalar &scalar_size_per_row, rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource *mr) {
  int64_t const total_allocation = size_per_row * num_rows;
  // We made a mistake in the split somehow
  CUDF_EXPECTS(total_allocation < std::numeric_limits<size_type>::max(),
               "Table is too large to fit!");

  // Allocate and set the offsets row for the byte array
  std::unique_ptr<column> offsets =
      cudf::detail::sequence(num_rows + 1, zero, scalar_size_per_row, stream);

  std::unique_ptr<column> data =
      make_numeric_column(data_type(type_id::INT8), static_cast<size_type>(total_allocation),
                          mask_state::UNALLOCATED, stream, mr);

  dim3 blocks;
  dim3 threads;
  int shared_size =
      detail::calc_fixed_width_kernel_dims(num_columns, num_rows, size_per_row, blocks, threads);

  copy_to_rows_fixed_width_optimized<<<blocks, threads, shared_size, stream.value()>>>(
      start_row, num_rows, num_columns, size_per_row, column_start.data(), column_size.data(),
      input_data.data(), input_nm.data(), data->mutable_view().data<int8_t>());

  return make_lists_column(num_rows, std::move(offsets), std::move(data), 0,
                           rmm::device_buffer{0, rmm::cuda_stream_default, mr}, stream, mr);
}

static inline bool are_all_fixed_width(std::vector<data_type> const &schema) {
  return std::all_of(schema.begin(), schema.end(),
                     [](const data_type &t) { return is_fixed_width(t); });
}

/**
 * @brief Given a set of fixed width columns, calculate how the data will be laid out in memory.
 *
 * @param [in] schema the types of columns that need to be laid out.
 * @param [out] column_start the byte offset where each column starts in the row.
 * @param [out] column_size the size in bytes of the data for each columns in the row.
 * @return the size in bytes each row needs.
 */
static inline int32_t compute_fixed_width_layout(std::vector<data_type> const &schema,
                                                 std::vector<size_type> &column_start,
                                                 std::vector<size_type> &column_size) {
  // We guarantee that the start of each column is 64-bit aligned so anything can go
  // there, but to make the code simple we will still do an alignment for it.
  int32_t at_offset = 0;
  for (auto col = schema.begin(); col < schema.end(); col++) {
    size_type s = size_of(*col);
    column_size.emplace_back(s);
    std::size_t allocation_needed = s;
    std::size_t alignment_needed = allocation_needed; // They are the same for fixed width types
    at_offset = util::round_up_unsafe(at_offset, static_cast<int32_t>(alignment_needed));
    column_start.emplace_back(at_offset);
    at_offset += allocation_needed;
  }

  // Now we need to add in space for validity
  // Eventually we can think about nullable vs not nullable, but for now we will just always add
  // it in
  int32_t const validity_bytes_needed = util::div_rounding_up_safe<int32_t>(schema.size(), 8);
  // validity comes at the end and is byte aligned so we can pack more in.
  at_offset += validity_bytes_needed;
  // Now we need to pad the end so all rows are 64 bit aligned
  return util::round_up_unsafe(at_offset, JCUDF_ROW_ALIGNMENT);
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700

/**
 * @brief Compute information about a table such as bytes per row and offsets.
 *
 * @tparam iterator iterator of column schema data
 * @param begin starting iterator of column schema
 * @param end ending iterator of column schema
 * @param column_starts column start offsets
 * @param column_sizes size in bytes of each column
 * @return size of the fixed_width data portion of a row.
 */
template <typename iterator>
static size_type compute_column_information(iterator begin, iterator end,
                                            std::vector<size_type> &column_starts,
                                            std::vector<size_type> &column_sizes) {
  size_type fixed_width_size_per_row = 0;
  for (auto cv = begin; cv != end; ++cv) {
    auto col_type = std::get<0>(*cv);
    bool nested_type = is_compound(col_type);

    // a list or string column will write a single uint64
    // of data here for offset/length
    auto col_size = nested_type ? 8 : size_of(col_type);

    // align size for this type
    size_type const alignment_needed = col_size; // They are the same for fixed width types
    fixed_width_size_per_row = util::round_up_unsafe(fixed_width_size_per_row, alignment_needed);
    column_starts.push_back(fixed_width_size_per_row);
    column_sizes.push_back(col_size);
    fixed_width_size_per_row += col_size;
  }

  auto validity_offset = fixed_width_size_per_row;
  column_starts.push_back(validity_offset);

  return fixed_width_size_per_row;
}

/**
 * @brief Build `block_info` for the validity data to break up the work.
 *
 * @param num_columns number of columns in the table
 * @param num_rows number of rows in the table
 * @param shmem_limit_per_block size of shared memory available to a single gpu block
 * @param row_batches batched row information for multiple output locations
 * @return vector of `block_info` structs for validity data
 */
std::vector<detail::block_info>
build_validity_block_infos(size_type const &num_columns, size_type const &num_rows,
                           size_type const &shmem_limit_per_block,
                           std::vector<row_batch> const &row_batches) {
  auto const desired_rows_and_columns = static_cast<int>(sqrt(shmem_limit_per_block));
  auto const column_stride = util::round_up_unsafe(
      [&]() {
        if (desired_rows_and_columns > num_columns) {
          // not many columns, group it into 8s and ship it off
          return std::min(8, num_columns);
        } else {
          return util::round_down_safe(desired_rows_and_columns, 8);
        }
      }(),
      JCUDF_ROW_ALIGNMENT);

  // we fit as much as we can given the column stride
  // note that an element in the table takes just 1 bit, but a row with a single
  // element still takes 8 bytes!
  auto const bytes_per_row =
      util::round_up_safe(util::div_rounding_up_unsafe(column_stride, 8), JCUDF_ROW_ALIGNMENT);
  auto const row_stride = std::min(num_rows, shmem_limit_per_block / bytes_per_row);

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

template <typename RowSize> struct row_size_functor {
  RowSize _row_sizes;
  size_type _num_rows;
  row_size_functor(RowSize row_sizes) : _row_sizes(row_sizes){};

  CUDA_DEVICE_CALLABLE
  uint64_t operator()(int row_index) { return static_cast<uint64_t>(_row_sizes[row_index]); }
};

/**
 * @brief Builds batches of rows that will fit in the size limit of a column.
 *
 * @tparam RowSize iterator that gives the size of a specific row of the table.
 * @param num_rows Total number of rows in the table
 * @param row_sizes iterator that gives the size of a specific row of the table.
 * @param stream stream to operate on for this work
 * @param mr memory resource used to allocate any returned data
 * @returns vector of size_type's that indicate row numbers for batch boundaries and a
 * device_uvector of row offsets
 */

template <typename RowSize>
batch_data build_batches(size_type num_rows, RowSize row_sizes, rmm::cuda_stream_view stream,
                         rmm::mr::device_memory_resource *mr) {
  auto uint64_row_sizes =
      cudf::detail::make_counting_transform_iterator(0, row_size_functor(row_sizes));
  auto const total_size =
      thrust::reduce(rmm::exec_policy(stream), uint64_row_sizes, uint64_row_sizes + num_rows);
  auto const num_batches = static_cast<int32_t>(
      util::div_rounding_up_safe(total_size, static_cast<uint64_t>(MAX_BATCH_SIZE)));
  auto const num_offsets = num_batches + 1;
  std::vector<row_batch> row_batches;
  std::vector<size_type> batch_row_boundaries;
  device_uvector<size_type> batch_row_offsets(num_rows, stream);

  // at most max gpu memory / 2GB iterations.
  batch_row_boundaries.reserve(num_offsets);
  batch_row_boundaries.push_back(0);
  size_type last_row_end = 0;
  device_uvector<uint64_t> cumulative_row_sizes(num_rows, stream);
  thrust::inclusive_scan(rmm::exec_policy(stream), uint64_row_sizes, uint64_row_sizes + num_rows,
                         cumulative_row_sizes.begin());

  while (static_cast<int>(batch_row_boundaries.size()) < num_offsets) {
    // find the next MAX_BATCH_SIZE boundary
    size_type const row_end =
        ((thrust::lower_bound(rmm::exec_policy(stream), cumulative_row_sizes.begin(),
                              cumulative_row_sizes.begin() + (num_rows - last_row_end),
                              MAX_BATCH_SIZE) -
          cumulative_row_sizes.begin()) +
         last_row_end);

    // build offset list for each row in this batch
    auto const num_rows_in_batch = row_end - last_row_end;

    // build offset list for each row in this batch
    auto const num_entries = row_end - last_row_end + 1;
    device_uvector<size_type> output_batch_row_offsets(num_entries, stream, mr);

    auto row_size_iter_bounded = thrust::make_transform_iterator(
        thrust::make_counting_iterator(0), [row_end, row_sizes, last_row_end] __device__(auto i) {
          return i >= row_end ? 0 : row_sizes[i + last_row_end];
        });

    thrust::exclusive_scan(rmm::exec_policy(stream), row_size_iter_bounded,
                           row_size_iter_bounded + num_entries, output_batch_row_offsets.begin());

    auto const batch_bytes = output_batch_row_offsets.element(num_rows_in_batch, stream);

    // The output_batch_row_offsets vector is used as the offset column of the returned data. This
    // needs to be individually allocated, but the kernel needs a contiguous array of offsets or
    // more global lookups are necessary.
    cudaMemcpy(batch_row_offsets.data() + last_row_end, output_batch_row_offsets.data(),
               num_rows_in_batch * sizeof(size_type), cudaMemcpyDeviceToDevice);

    batch_row_boundaries.push_back(row_end);
    row_batches.push_back({batch_bytes, num_rows_in_batch, std::move(output_batch_row_offsets)});

    last_row_end = row_end;
  }

  return {std::move(batch_row_offsets), batch_row_boundaries, std::move(row_batches)};
}

/**
 * @brief Computes the number of blocks necessary given a window height and batch offsets
 *
 * @param batch_row_boundaries row boundaries for each batch
 * @param desired_window_height height of each window in the table
 * @param stream stream to use
 * @return number of windows necessary
 */
int compute_block_counts(device_span<size_type const> const &batch_row_boundaries,
                         int desired_window_height, rmm::cuda_stream_view stream) {
  size_type const num_batches = batch_row_boundaries.size() - 1;
  device_uvector<size_type> num_blocks(num_batches, stream);
  auto iter = thrust::make_counting_iterator(0);
  thrust::transform(rmm::exec_policy(stream), iter, iter + num_batches, num_blocks.begin(),
                    [desired_window_height,
                     batch_row_boundaries =
                         batch_row_boundaries.data()] __device__(auto batch_index) -> size_type {
                      return util::div_rounding_up_unsafe(batch_row_boundaries[batch_index + 1] -
                                                              batch_row_boundaries[batch_index],
                                                          desired_window_height);
                    });
  return thrust::reduce(rmm::exec_policy(stream), num_blocks.begin(), num_blocks.end());
}

/**
 * @brief Builds the `block_info` structs for a given table.
 *
 * @param blocks span of blocks to populate
 * @param batch_row_boundaries boundary to row batches
 * @param column_start starting column of the window
 * @param column_end ending column of the window
 * @param desired_window_height height of the window
 * @param total_number_of_rows total number of rows in the table
 * @param stream stream to use
 * @return number of windows created
 */
size_type
build_blocks(device_span<block_info> blocks,
             device_uvector<size_type> const &batch_row_boundaries, // comes from build_batches
             int column_start, int column_end, int desired_window_height, int total_number_of_rows,
             rmm::cuda_stream_view stream) {
  size_type const num_batches = batch_row_boundaries.size() - 1;
  device_uvector<size_type> num_blocks(num_batches, stream);
  auto iter = thrust::make_counting_iterator(0);
  thrust::transform(rmm::exec_policy(stream), iter, iter + num_batches, num_blocks.begin(),
                    [desired_window_height,
                     batch_row_boundaries =
                         batch_row_boundaries.data()] __device__(auto batch_index) -> size_type {
                      return util::div_rounding_up_unsafe(batch_row_boundaries[batch_index + 1] -
                                                              batch_row_boundaries[batch_index],
                                                          desired_window_height);
                    });

  size_type const total_blocks =
      thrust::reduce(rmm::exec_policy(stream), num_blocks.begin(), num_blocks.end());

  device_uvector<size_type> block_starts(num_batches + 1, stream);
  auto block_iter = cudf::detail::make_counting_transform_iterator(
      0, [num_blocks = num_blocks.data(), num_batches] __device__(auto i) {
        return (i < num_batches) ? num_blocks[i] : 0;
      });
  thrust::exclusive_scan(rmm::exec_policy(stream), block_iter, block_iter + num_batches + 1,
                         block_starts.begin()); // in blocks

  thrust::transform(
      rmm::exec_policy(stream), iter, iter + total_blocks, blocks.begin(),
      [=, block_starts = block_starts.data(),
       batch_row_boundaries = batch_row_boundaries.data()] __device__(size_type block_index) {
        // what batch this block falls in
        auto const batch_index_iter =
            thrust::upper_bound(thrust::seq, block_starts, block_starts + num_batches, block_index);
        auto const batch_index = std::distance(block_starts, batch_index_iter) - 1;
        // local index within the block
        int const local_block_index = block_index - block_starts[batch_index];
        // the start row for this batch.
        int const batch_row_start = batch_row_boundaries[batch_index];
        // the start row for this block
        int const block_row_start = batch_row_start + (local_block_index * desired_window_height);
        // the end row for this block
        int const max_row =
            std::min(total_number_of_rows - 1,
                     batch_index + 1 > num_batches ?
                         std::numeric_limits<size_type>::max() :
                         static_cast<int>(batch_row_boundaries[batch_index + 1]) - 1);
        int const block_row_end = std::min(
            batch_row_start + ((local_block_index + 1) * desired_window_height) - 1, max_row);

        // stuff the block
        return block_info{column_start, block_row_start, column_end, block_row_end,
                          static_cast<int>(batch_index)};
      });

  return total_blocks;
}

/**
 * @brief Determines what data should be operated on by each block for the incoming table.
 *
 * @tparam WindowCallback Callback that receives the start and end columns of windows
 * @param column_sizes vector of the size of each column
 * @param column_starts vector of the offset of each column
 * @param first_row_batch_size size of the first row batch to limit max window size since a window
 * is unable to span batches
 * @param total_number_of_rows total number of rows in the table
 * @param shmem_limit_per_block shared memory allowed per block
 * @param f callback function called when building a window
 */
template <typename WindowCallback>
void determine_windows(std::vector<size_type> const &column_sizes,
                       std::vector<size_type> const &column_starts,
                       size_type const first_row_batch_size, size_type const total_number_of_rows,
                       size_type const &shmem_limit_per_block, WindowCallback f) {
  // block infos are organized with the windows going "down" the columns
  // this provides the most coalescing of memory access
  int current_window_width = 0;
  int current_window_start_col = 0;

  // the ideal window height has lots of 8-byte reads and 8-byte writes. The optimal read/write
  // would be memory cache line sized access, but since other blocks will read/write the edges
  // this may not turn out to be overly important. For now, we will attempt to build a square
  // window as far as byte sizes. x * y = shared_mem_size. Which translates to x^2 =
  // shared_mem_size since we want them equal, so height and width are sqrt(shared_mem_size). The
  // trick is that it's in bytes, not rows or columns.
  auto const optimal_square_len = static_cast<size_type>(sqrt(shmem_limit_per_block));
  auto const window_height =
      std::clamp(util::round_up_safe<int>(
                     std::min(optimal_square_len / column_sizes[0], total_number_of_rows), 32),
                 1, first_row_batch_size);

  int row_size = 0;

  // march each column and build the blocks of appropriate sizes
  for (uint col = 0; col < column_sizes.size(); ++col) {
    auto const col_size = column_sizes[col];

    // align size for this type
    auto const alignment_needed = col_size; // They are the same for fixed width types
    auto const row_size_aligned = util::round_up_unsafe(row_size, alignment_needed);
    auto const row_size_with_this_col = row_size_aligned + col_size;
    auto const row_size_with_end_pad =
        util::round_up_unsafe(row_size_with_this_col, JCUDF_ROW_ALIGNMENT);

    if (row_size_with_end_pad * window_height > shmem_limit_per_block) {
      // too large, close this window, generate vertical blocks and restart
      f(current_window_start_col, col == 0 ? col : col - 1, window_height);

      row_size =
          util::round_up_unsafe((column_starts[col] + column_sizes[col]) & 7, alignment_needed);
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
    f(current_window_start_col, static_cast<int>(column_sizes.size()) - 1, window_height);
  }
}

#endif // #if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700

} // namespace detail

std::vector<std::unique_ptr<column>> convert_to_rows(table_view const &tbl,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::mr::device_memory_resource *mr) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
  auto const num_columns = tbl.num_columns();
  auto const num_rows = tbl.num_rows();

  int device_id;
  CUDA_TRY(cudaGetDevice(&device_id));
  int total_shmem;
  CUDA_TRY(cudaDeviceGetAttribute(&total_shmem, cudaDevAttrMaxSharedMemoryPerBlock, device_id));

  // TODO: why is this needed. kernel fails to launch if all memory is requested.
  total_shmem -= 1024;
  auto const shmem_limit_per_block = total_shmem / NUM_BLOCKS_PER_KERNEL_LOADED;

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
  std::transform(tbl.begin(), tbl.end(), std::back_inserter(input_data),
                 [](column_view const &c) -> int8_t const * { return c.template data<int8_t>(); });
  std::transform(tbl.begin(), tbl.end(), std::back_inserter(input_nm),
                 [](auto c) { return c.null_mask(); });

  auto dev_input_data = make_device_uvector_async(input_data, stream, mr);
  auto dev_input_nm = make_device_uvector_async(input_nm, stream, mr);

  std::vector<size_type> column_sizes;  // byte size of each column
  std::vector<size_type> column_starts; // offset of column inside a row including alignment
  column_sizes.reserve(num_columns);
  column_starts.reserve(num_columns + 1); // we add a final offset for validity data start

  auto schema_column_iter =
      thrust::make_transform_iterator(thrust::make_counting_iterator(0),
                                      [&tbl](auto i) -> std::tuple<data_type, column_view const> {
                                        return std::make_tuple(tbl.column(i).type(), tbl.column(i));
                                      });

  auto const fixed_width_size_per_row = detail::compute_column_information(
      schema_column_iter, schema_column_iter + num_columns, column_starts, column_sizes);

  auto dev_col_sizes = make_device_uvector_async(column_sizes, stream, mr);
  auto dev_col_starts = make_device_uvector_async(column_starts, stream, mr);

  // total encoded row size. This includes fixed-width data, validity, and variable-width data.
  auto row_size_iter = cudf::detail::make_counting_transform_iterator(
      0, [fixed_width_size_per_row, num_columns] __device__(auto i) {
        auto const bytes_needed =
            fixed_width_size_per_row + util::div_rounding_up_safe<size_type>(num_columns, 8);
        return util::round_up_unsafe(bytes_needed, JCUDF_ROW_ALIGNMENT);
      });

  auto batch_info = detail::build_batches(num_rows, row_size_iter, stream, mr);
  auto gpu_batch_row_boundaries =
      make_device_uvector_async(batch_info.batch_row_boundaries, stream);

  // the first batch always exists unless we were sent an empty table
  auto const first_batch_size = batch_info.row_batches[0].row_count;

  std::vector<rmm::device_buffer> output_buffers;
  std::vector<int8_t *> output_data;
  output_data.reserve(batch_info.row_batches.size());
  output_buffers.reserve(batch_info.row_batches.size());
  std::transform(batch_info.row_batches.begin(), batch_info.row_batches.end(),
                 std::back_inserter(output_buffers), [&](auto const &batch) {
                   return rmm::device_buffer(batch.num_bytes, stream, mr);
                 });
  std::transform(output_buffers.begin(), output_buffers.end(), std::back_inserter(output_data),
                 [](auto &buf) { return static_cast<int8_t *>(buf.data()); });

  auto dev_output_data = make_device_uvector_async(output_data, stream, mr);

  int info_count = 0;
  detail::determine_windows(
      column_sizes, column_starts, first_batch_size, num_rows, shmem_limit_per_block,
      [&gpu_batch_row_boundaries, &info_count, &stream](int const start_col, int const end_col,
                                                        int const window_height) {
        int i = detail::compute_block_counts(gpu_batch_row_boundaries, window_height, stream);
        info_count += i;
      });

  // allocate space for blocks
  device_uvector<detail::block_info> gpu_block_infos(info_count, stream);
  int block_offset = 0;

  detail::determine_windows(
      column_sizes, column_starts, first_batch_size, num_rows, shmem_limit_per_block,
      [&gpu_batch_row_boundaries, &gpu_block_infos, num_rows, &block_offset,
       stream](int const start_col, int const end_col, int const window_height) {
        block_offset += detail::build_blocks(
            {gpu_block_infos.data() + block_offset, gpu_block_infos.size() - block_offset},
            gpu_batch_row_boundaries, start_col, end_col, window_height, num_rows, stream);
      });

  // blast through the entire table and convert it
  dim3 blocks(util::div_rounding_up_unsafe(gpu_block_infos.size(), NUM_BLOCKS_PER_KERNEL_TO_ROWS));
  dim3 threads(256);

  auto validity_block_infos = detail::build_validity_block_infos(
      num_columns, num_rows, shmem_limit_per_block, batch_info.row_batches);

  auto dev_validity_block_infos = make_device_uvector_async(validity_block_infos, stream);
  dim3 validity_blocks(
      util::div_rounding_up_unsafe(validity_block_infos.size(), NUM_VALIDITY_BLOCKS_PER_KERNEL));
  dim3 validity_threads(std::min(validity_block_infos.size() * 32, 128lu));

  detail::copy_to_rows<<<blocks, threads, total_shmem, stream.value()>>>(
      num_rows, num_columns, shmem_limit_per_block, gpu_block_infos, dev_input_data.data(),
      dev_col_sizes.data(), dev_col_starts.data(),
      batch_info.batch_row_offsets
          .data(), // needs to be row offsets per batch, not overall JUST for output.
      reinterpret_cast<int8_t **>(dev_output_data.data()));

  detail::copy_validity_to_rows<<<validity_blocks, validity_threads, total_shmem, stream.value()>>>(
      num_rows, num_columns, shmem_limit_per_block, batch_info.batch_row_offsets.data(),
      dev_output_data.data(), column_starts.back(), dev_validity_block_infos, dev_input_nm.data());

  // split up the output buffer into multiple buffers based on row batch sizes
  // and create list of byte columns
  std::vector<std::unique_ptr<column>> ret;
  auto counting_iter = thrust::make_counting_iterator(0);
  std::transform(counting_iter, counting_iter + batch_info.row_batches.size(),
                 std::back_inserter(ret), [&](auto batch) {
                   auto const offset_count = batch_info.row_batches[batch].row_offsets.size();
                   auto offsets = std::make_unique<column>(
                       data_type{type_id::INT32}, (size_type)offset_count,
                       batch_info.row_batches[batch].row_offsets.release());
                   auto data = std::make_unique<column>(data_type{type_id::INT8},
                                                        batch_info.row_batches[batch].num_bytes,
                                                        std::move(output_buffers[batch]));

                   return make_lists_column(
                       batch_info.row_batches[batch].row_count, std::move(offsets), std::move(data),
                       0, rmm::device_buffer{0, rmm::cuda_stream_default, mr}, stream, mr);
                 });

  return ret;
#else
  CUDF_FAIL("Column to row conversion optimization requires volta or later hardware.");
  return {};
#endif // #if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
}

std::vector<std::unique_ptr<column>>
convert_to_rows_fixed_width_optimized(table_view const &tbl, rmm::cuda_stream_view stream,
                                      rmm::mr::device_memory_resource *mr) {
  auto const num_columns = tbl.num_columns();

  std::vector<data_type> schema;
  schema.resize(num_columns);
  std::transform(tbl.begin(), tbl.end(), schema.begin(),
                 [](auto i) -> data_type { return i.type(); });

  if (detail::are_all_fixed_width(schema)) {
    std::vector<size_type> column_start;
    std::vector<size_type> column_size;

    int32_t const size_per_row =
        detail::compute_fixed_width_layout(schema, column_start, column_size);
    auto dev_column_start = make_device_uvector_async(column_start, stream, mr);
    auto dev_column_size = make_device_uvector_async(column_size, stream, mr);

    // Make the number of rows per batch a multiple of 32 so we don't have to worry about
    // splitting validity at a specific row offset.  This might change in the future.
    auto const max_rows_per_batch =
        util::round_down_safe(std::numeric_limits<size_type>::max() / size_per_row, 32);

    auto const num_rows = tbl.num_rows();

    // Get the pointers to the input columnar data ready
    std::vector<const int8_t *> input_data;
    std::vector<bitmask_type const *> input_nm;
    for (size_type column_number = 0; column_number < num_columns; column_number++) {
      column_view cv = tbl.column(column_number);
      input_data.emplace_back(cv.data<int8_t>());
      input_nm.emplace_back(cv.null_mask());
    }
    auto dev_input_data = make_device_uvector_async(input_data, stream, mr);
    auto dev_input_nm = make_device_uvector_async(input_nm, stream, mr);

    using ScalarType = scalar_type_t<size_type>;
    auto zero = make_numeric_scalar(data_type(type_id::INT32), stream.value());
    zero->set_valid_async(true, stream);
    static_cast<ScalarType *>(zero.get())->set_value(0, stream);

    auto step = make_numeric_scalar(data_type(type_id::INT32), stream.value());
    step->set_valid_async(true, stream);
    static_cast<ScalarType *>(step.get())->set_value(static_cast<size_type>(size_per_row), stream);

    std::vector<std::unique_ptr<column>> ret;
    for (size_type row_start = 0; row_start < num_rows; row_start += max_rows_per_batch) {
      size_type row_count = num_rows - row_start;
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

std::unique_ptr<table> convert_from_rows(lists_column_view const &input,
                                         std::vector<data_type> const &schema,
                                         rmm::cuda_stream_view stream,
                                         rmm::mr::device_memory_resource *mr) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
  // verify that the types are what we expect
  column_view child = input.child();
  auto const list_type = child.type().id();
  CUDF_EXPECTS(list_type == type_id::INT8 || list_type == type_id::UINT8,
               "Only a list of bytes is supported as input");

  auto const num_columns = schema.size();
  auto const num_rows = input.parent().size();

  int device_id;
  CUDA_TRY(cudaGetDevice(&device_id));
  int total_shmem;
  CUDA_TRY(cudaDeviceGetAttribute(&total_shmem, cudaDevAttrMaxSharedMemoryPerBlock, device_id));

  // TODO: why is this needed. kernel fails to launch if all memory is requested.
  total_shmem -= 1024;
  int shmem_limit_per_block = total_shmem / NUM_BLOCKS_PER_KERNEL_LOADED;

  std::vector<size_type> column_starts;
  std::vector<size_type> column_sizes;

  auto iter = thrust::make_transform_iterator(thrust::make_counting_iterator(0), [&schema](auto i) {
    return std::make_tuple(schema[i], nullptr);
  });
  auto const fixed_width_size_per_row =
      detail::compute_column_information(iter, iter + num_columns, column_starts, column_sizes);

  auto const validity_size = num_bitmask_words(num_columns) * 4;

  auto const row_size =
      util::round_up_unsafe(fixed_width_size_per_row + validity_size, JCUDF_ROW_ALIGNMENT);

  // Ideally we would check that the offsets are all the same, etc. but for now
  // this is probably fine
  CUDF_EXPECTS(row_size * num_rows == child.size(), "The layout of the data appears to be off");
  auto dev_col_starts = make_device_uvector_async(column_starts, stream, mr);
  auto dev_col_sizes = make_device_uvector_async(column_sizes, stream, mr);

  // Allocate the columns we are going to write into
  std::vector<std::unique_ptr<column>> output_columns;
  std::vector<int8_t *> output_data;
  std::vector<bitmask_type *> output_nm;
  for (int i = 0; i < static_cast<int>(num_columns); i++) {
    auto column =
        make_fixed_width_column(schema[i], num_rows, mask_state::UNINITIALIZED, stream, mr);
    auto mut = column->mutable_view();
    output_data.emplace_back(mut.data<int8_t>());
    output_nm.emplace_back(mut.null_mask());
    output_columns.emplace_back(std::move(column));
  }

  // build the row_batches from the passed in list column
  std::vector<detail::row_batch> row_batches;
  row_batches.push_back(
      {detail::row_batch{child.size(), num_rows, device_uvector<size_type>(0, stream)}});

  auto dev_output_data = make_device_uvector_async(output_data, stream, mr);
  auto dev_output_nm = make_device_uvector_async(output_nm, stream, mr);

  // only ever get a single batch when going from rows, so boundaries
  // are 0, num_rows
  device_uvector<size_type> gpu_batch_row_boundaries(2, stream);

  thrust::transform(rmm::exec_policy(stream), thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(2), gpu_batch_row_boundaries.begin(),
                    [num_rows] __device__(auto i) { return i == 0 ? 0 : num_rows; });

  int info_count = 0;
  detail::determine_windows(column_sizes, column_starts, num_rows, num_rows, shmem_limit_per_block,
                            [&gpu_batch_row_boundaries, &info_count, &stream](
                                int const start_col, int const end_col, int const window_height) {
                              info_count += detail::compute_block_counts(gpu_batch_row_boundaries,
                                                                         window_height, stream);
                            });

  // allocate space for blocks
  device_uvector<detail::block_info> gpu_block_infos(info_count, stream);

  int block_offset = 0;
  detail::determine_windows(
      column_sizes, column_starts, num_rows, num_rows, shmem_limit_per_block,
      [&gpu_batch_row_boundaries, &gpu_block_infos, num_rows, &block_offset,
       stream](int const start_col, int const end_col, int const window_height) {
        block_offset += detail::build_blocks(
            {gpu_block_infos.data() + block_offset, gpu_block_infos.size() - block_offset},
            gpu_batch_row_boundaries, start_col, end_col, window_height, num_rows, stream);
      });

  dim3 blocks(
      util::div_rounding_up_unsafe(gpu_block_infos.size(), NUM_BLOCKS_PER_KERNEL_FROM_ROWS));
  dim3 threads(std::min(std::min(256, shmem_limit_per_block / 8), static_cast<int>(child.size())));

  auto validity_block_infos =
      detail::build_validity_block_infos(num_columns, num_rows, shmem_limit_per_block, row_batches);

  auto dev_validity_block_infos = make_device_uvector_async(validity_block_infos, stream);

  dim3 validity_blocks(
      util::div_rounding_up_unsafe(validity_block_infos.size(), NUM_VALIDITY_BLOCKS_PER_KERNEL));

  dim3 validity_threads(std::min(validity_block_infos.size() * 32, 128lu));

  detail::copy_from_rows<<<blocks, threads, total_shmem, stream.value()>>>(
      num_rows, num_columns, shmem_limit_per_block, input.offsets().data<size_type>(),
      dev_output_data.data(), dev_col_sizes.data(), dev_col_starts.data(), gpu_block_infos,
      child.data<int8_t>());

  detail::
      copy_validity_from_rows<<<validity_blocks, validity_threads, total_shmem, stream.value()>>>(
          num_rows, num_columns, shmem_limit_per_block, input.offsets().data<size_type>(),
          dev_output_nm.data(), column_starts.back(), dev_validity_block_infos,
          child.data<int8_t>());

  return std::make_unique<table>(std::move(output_columns));
#else
  CUDF_FAIL("Row to column conversion optimization requires volta or later hardware.");
  return {};
#endif // #if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
}

std::unique_ptr<table> convert_from_rows_fixed_width_optimized(
    lists_column_view const &input, std::vector<data_type> const &schema,
    rmm::cuda_stream_view stream, rmm::mr::device_memory_resource *mr) {
  // verify that the types are what we expect
  column_view child = input.child();
  auto const list_type = child.type().id();
  CUDF_EXPECTS(list_type == type_id::INT8 || list_type == type_id::UINT8,
               "Only a list of bytes is supported as input");

  auto const num_columns = schema.size();

  if (detail::are_all_fixed_width(schema)) {
    std::vector<size_type> column_start;
    std::vector<size_type> column_size;

    auto const num_rows = input.parent().size();
    auto const size_per_row = detail::compute_fixed_width_layout(schema, column_start, column_size);

    // Ideally we would check that the offsets are all the same, etc. but for now
    // this is probably fine
    CUDF_EXPECTS(size_per_row * num_rows == child.size(),
                 "The layout of the data appears to be off");
    auto dev_column_start = make_device_uvector_async(column_start, stream);
    auto dev_column_size = make_device_uvector_async(column_size, stream);

    // Allocate the columns we are going to write into
    std::vector<std::unique_ptr<column>> output_columns;
    std::vector<int8_t *> output_data;
    std::vector<bitmask_type *> output_nm;
    for (int i = 0; i < static_cast<int>(num_columns); i++) {
      auto column =
          make_fixed_width_column(schema[i], num_rows, mask_state::UNINITIALIZED, stream, mr);
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

    detail::copy_from_rows_fixed_width_optimized<<<blocks, threads, shared_size, stream.value()>>>(
        num_rows, num_columns, size_per_row, dev_column_start.data(), dev_column_size.data(),
        dev_output_data.data(), dev_output_nm.data(), child.data<int8_t>());

    return std::make_unique<table>(std::move(output_columns));
  } else {
    CUDF_FAIL("Only fixed width types are currently supported");
  }
}

} // namespace java

} // namespace cudf
