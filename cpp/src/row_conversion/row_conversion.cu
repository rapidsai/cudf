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

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/sequence.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <cudf/row_conversion.hpp>
#include <tuple>
#include "cudf/types.hpp"
#include "rmm/device_buffer.hpp"
#include "thrust/iterator/counting_iterator.h"
#include "thrust/iterator/transform_iterator.h"

#define NUM_BLOCKS_PER_KERNEL_TO_COLUMNS (2)

using cudf::detail::make_device_uvector_async;
namespace cudf {

namespace detail {

static inline __host__ __device__ int32_t align_offset(int32_t offset, std::size_t alignment)
{
  return (offset + alignment - 1) & ~(alignment - 1);
}

__global__ void copy_to_fixed_width_columns(const cudf::size_type num_rows,
                                            const cudf::size_type num_columns,
                                            const cudf::size_type row_size,
                                            const cudf::size_type *input_offset_in_row,
                                            const cudf::size_type *num_bytes,
                                            int8_t **output_data,
                                            cudf::bitmask_type **output_nm,
                                            const int8_t *input_data)
{
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

  cudf::size_type rows_per_group   = blockDim.x;
  cudf::size_type row_group_start  = blockIdx.x;
  cudf::size_type row_group_stride = gridDim.x;
  cudf::size_type row_group_end    = (num_rows + rows_per_group - 1) / rows_per_group + 1;

  extern __shared__ int8_t shared_data[];

  // Because we are copying fixed width only data and we stride the rows
  // this thread will always start copying from shared data in the same place
  int8_t *row_tmp     = &shared_data[row_size * threadIdx.x];
  int8_t *row_vld_tmp = &row_tmp[input_offset_in_row[num_columns - 1] + num_bytes[num_columns - 1]];

  for (cudf::size_type row_group_index = row_group_start; row_group_index < row_group_end;
       row_group_index += row_group_stride) {
    // Step 1: Copy the data into shared memory
    // We know row_size is always aligned with and a multiple of int64_t;
    int64_t *long_shared      = reinterpret_cast<int64_t *>(shared_data);
    const int64_t *long_input = reinterpret_cast<int64_t const *>(input_data);

    cudf::size_type shared_output_index  = threadIdx.x + (threadIdx.y * blockDim.x);
    cudf::size_type shared_output_stride = blockDim.x * blockDim.y;
    cudf::size_type row_index_end        = ((row_group_index + 1) * rows_per_group);
    if (row_index_end > num_rows) { row_index_end = num_rows; }
    cudf::size_type num_rows_in_group = row_index_end - (row_group_index * rows_per_group);
    cudf::size_type shared_length     = row_size * num_rows_in_group;

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
      cudf::size_type col_index_start  = threadIdx.y;
      cudf::size_type col_index_stride = blockDim.y;
      for (cudf::size_type col_index = col_index_start; col_index < num_columns;
           col_index += col_index_stride) {
        cudf::size_type col_size = num_bytes[col_index];
        const int8_t *col_tmp    = &(row_tmp[input_offset_in_row[col_index]]);
        int8_t *col_output       = output_data[col_index];
        switch (col_size) {
          case 1: {
            col_output[row_index] = *col_tmp;
            break;
          }
          case 2: {
            int16_t *short_col_output   = reinterpret_cast<int16_t *>(col_output);
            short_col_output[row_index] = *reinterpret_cast<const int16_t *>(col_tmp);
            break;
          }
          case 4: {
            int32_t *int_col_output   = reinterpret_cast<int32_t *>(col_output);
            int_col_output[row_index] = *reinterpret_cast<const int32_t *>(col_tmp);
            break;
          }
          case 8: {
            int64_t *long_col_output   = reinterpret_cast<int64_t *>(col_output);
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

        cudf::bitmask_type *nm          = output_nm[col_index];
        int8_t *valid_byte              = &row_vld_tmp[col_index / 8];
        cudf::size_type byte_bit_offset = col_index % 8;
        int predicate                   = *valid_byte & (1 << byte_bit_offset);
        uint32_t bitmask                = __ballot_sync(active_mask, predicate);
        if (row_index % 32 == 0) { nm[row_index / 8] = bitmask; }
      }  // end column loop
    }    // end row copy
    // wait for the row_group to be totally copied before starting on the next row group
    __syncthreads();
  }
}

__global__ void copy_from_fixed_width_columns(const cudf::size_type start_row,
                                              const cudf::size_type num_rows,
                                              const cudf::size_type num_columns,
                                              const cudf::size_type row_size,
                                              const cudf::size_type *output_offset_in_row,
                                              const cudf::size_type *num_bytes,
                                              const int8_t **input_data,
                                              const cudf::bitmask_type **input_nm,
                                              int8_t *output_data)
{
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

  cudf::size_type rows_per_group   = blockDim.x;
  cudf::size_type row_group_start  = blockIdx.x;
  cudf::size_type row_group_stride = gridDim.x;
  cudf::size_type row_group_end    = (num_rows + rows_per_group - 1) / rows_per_group + 1;

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
      cudf::size_type col_index_start  = threadIdx.y;
      cudf::size_type col_index_stride = blockDim.y;
      for (cudf::size_type col_index = col_index_start; col_index < num_columns;
           col_index += col_index_stride) {
        cudf::size_type col_size = num_bytes[col_index];
        int8_t *col_tmp          = &(row_tmp[output_offset_in_row[col_index]]);
        const int8_t *col_input  = input_data[col_index];
        switch (col_size) {
          case 1: {
            *col_tmp = col_input[row_index];
            break;
          }
          case 2: {
            const int16_t *short_col_input        = reinterpret_cast<const int16_t *>(col_input);
            *reinterpret_cast<int16_t *>(col_tmp) = short_col_input[row_index];
            break;
          }
          case 4: {
            const int32_t *int_col_input          = reinterpret_cast<const int32_t *>(col_input);
            *reinterpret_cast<int32_t *>(col_tmp) = int_col_input[row_index];
            break;
          }
          case 8: {
            const int64_t *long_col_input         = reinterpret_cast<const int64_t *>(col_input);
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
        int8_t *valid_byte              = &row_vld_tmp[col_index / 8];
        cudf::size_type byte_bit_offset = col_index % 8;
        uint64_t fixup_bytes            = reinterpret_cast<uint64_t>(valid_byte) % 4;
        int32_t *valid_int              = reinterpret_cast<int32_t *>(valid_byte - fixup_bytes);
        cudf::size_type int_bit_offset  = byte_bit_offset + (fixup_bytes * 8);
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
      }  // end column loop
    }    // end row copy
    // wait for the row_group to be totally copied into shared memory
    __syncthreads();

    // Step 2: Copy the data back out
    // We know row_size is always aligned with and a multiple of int64_t;
    int64_t *long_shared = reinterpret_cast<int64_t *>(shared_data);
    int64_t *long_output = reinterpret_cast<int64_t *>(output_data);

    cudf::size_type shared_input_index  = threadIdx.x + (threadIdx.y * blockDim.x);
    cudf::size_type shared_input_stride = blockDim.x * blockDim.y;
    cudf::size_type row_index_end       = ((row_group_index + 1) * rows_per_group);
    if (row_index_end > num_rows) { row_index_end = num_rows; }
    cudf::size_type num_rows_in_group = row_index_end - (row_group_index * rows_per_group);
    cudf::size_type shared_length     = row_size * num_rows_in_group;

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

struct block_info {
  int start_col;
  int start_row;
  int end_col;
  int end_row;
  int buffer_num;
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
__global__ void copy_from_columns(const size_type num_rows,
                                  const size_type num_columns,
                                  const int8_t **input_data,
                                  const bitmask_type **input_nm,
                                  const size_type *col_sizes,
                                  const size_type *col_offsets,
                                  const block_info *block_infos,
                                  const size_type *row_offsets,
                                  int8_t **output_data)
{
  // We are going to copy the data in two passes.
  // The first pass copies a chunk of data into shared memory.
  // The second pass copies that chunk from shared memory out to the final location.

  // Because shared memory is limited we copy a subset of the rows at a time.
  // This has been broken up for us in the block_info struct, so we don't have
  // any calculation to do here, but it is important to note.

  bool debug_print = false; //blockIdx.x == 0 && threadIdx.x == 0;

  if (debug_print) {
    printf("%d %d - %d rows, %d columns\n", threadIdx.x, blockIdx.x, num_rows, num_columns);
    printf("Column Info:\n");
    for (int i = 0; i < num_columns; ++i) {
      printf("col %d is at %p with size %d and offset %d\n",
             i,
             input_data[i],
             col_sizes[i],
             col_offsets[i]);
    }
    printf("block infos are at %p and my index is %d\n", block_infos, blockIdx.x);
    /*    printf("Row Offsets:\n");
        for (int i=0; i<num_rows; ++i) {
          printf("%d: %d\n", i, row_offsets[i]);
        }*/
    printf("output data to %p\n", output_data[block_infos[blockIdx.x].buffer_num]);
  }
  // else { return; }
  auto block               = block_infos[blockIdx.x];
  auto const rows_in_block = block.end_row - block.start_row + 1;
  extern __shared__ int8_t shared_data[];
  uint64_t const output_start_offset = col_offsets[block.start_col] + row_offsets[block.start_row];
  uint8_t const dest_shim_offset =
    reinterpret_cast<uint64_t>(&output_data[0][output_start_offset]) &
    7;  // offset for alignment shim in order to match shared memory with final dest
  if (debug_print) {
    printf("outputting to offset %lu\n", output_start_offset);
    printf("dest shim offset is %d\n", dest_shim_offset);
    printf("Shared data is %p-%p\n", shared_data, shared_data + (48 * 1024));
    printf("my block is %d,%d -> %d,%d - buffer %d\n",
           block.start_col,
           block.start_row,
           block.end_col,
           block.end_row,
           block.buffer_num);
  }
  // each thread is responsible for every threadcount rows of data.
  // the data is copied into shared memory in the final layout.
  auto const real_bytes_in_row =
    col_offsets[block.end_col] + col_sizes[block.end_col] - col_offsets[block.start_col];
  auto const shmem_row_size  = align_offset(real_bytes_in_row + dest_shim_offset,
                                           8);  // 8 byte alignment required for shared memory rows
  auto const validity_offset = col_offsets[num_columns];
  if (debug_print) {
    printf("col_offsets[%d] = %d, col_sizes[%d] = %d, col_offsets[%d] = %d\n",
           block.end_col,
           col_offsets[block.end_col],
           block.end_col,
           col_sizes[block.end_col],
           block.start_col,
           col_offsets[block.start_col]);
    printf("shmem row size %d with real bytes %d\n", shmem_row_size, real_bytes_in_row);
    printf("validity offset is %d\n", validity_offset);
    printf("starting at %d,%d and going to %d, %d\n",
           block.start_col,
           block.start_row,
           block.end_col,
           block.end_row);
  }
  for (int col = block.start_col; col <= block.end_col; ++col) {
    /*if (!col_is_variable) */ {
      uint64_t col_offset      = 0;
      cudf::size_type col_size = col_sizes[col];
      auto const dest_col_offset =
        col_offsets[col] - col_offsets[block.start_col] + dest_shim_offset;
      if (debug_print) { printf("dest col offset %d\n", dest_col_offset); }
      for (int row = block.start_row + threadIdx.x; row <= block.end_row; row += blockDim.x) {
        if (debug_print) {
          printf("shmem row %d(%d) at offset %d(%d)\n",
                 row - block.start_row,
                 row,
                 (row - block.start_row) * shmem_row_size,
                 row * shmem_row_size);
        }
        int8_t *shmem_dest =
          &shared_data[dest_col_offset + shmem_row_size * (row - block.start_row)];
        switch (col_size) {
          case 1: {
            if (debug_print) { printf("%p <- byte %d\n", shmem_dest, input_data[col][row]); }
            *shmem_dest = input_data[col][row];
            break;
          }
          case 2: {
            const int16_t *short_col_input = reinterpret_cast<const int16_t *>(input_data[col]);
            if (debug_print) { printf("%p <- short %d\n", shmem_dest, short_col_input[row]); }
            *reinterpret_cast<int16_t *>(shmem_dest) = short_col_input[row];
            break;
          }
          case 4: {
            const int32_t *int_col_input = reinterpret_cast<const int32_t *>(input_data[col]);
            if (debug_print) {
              printf("shmem[%d][%d] - %p <- int 0x%x\n", row, col, shmem_dest, int_col_input[row]);
            }
            *reinterpret_cast<int32_t *>(shmem_dest) = int_col_input[row];
            break;
          }
          case 8: {
            const int64_t *long_col_input = reinterpret_cast<const int64_t *>(input_data[col]);
            if (debug_print) { printf("%p <- long %lu\n", shmem_dest, long_col_input[row]); }
            *reinterpret_cast<int64_t *>(shmem_dest) = long_col_input[row];
            break;
          }
          default: {
            cudf::size_type input_offset = col_size * row;
            if (debug_print) {
              printf("byte for byte copy due to size %d of column %d\n", col_size, col);
              printf("%p <- input_data[%d] which is %d\n",
                     shmem_dest,
                     input_offset,
                     input_data[col][input_offset]);
            }
            // TODO this should just not be supported for fixed width columns, but just in case...
            for (cudf::size_type b = 0; b < col_size; b++) {
              shmem_dest[b] = input_data[col][b + input_offset];
            }
            break;
          }
        }

        // atomicOr only works on 32 bit or 64 bit  aligned values, and not byte aligned
        // so we have to rewrite the addresses to make sure that it is 4 byte aligned
        // we do this directly in the final location because the entire row may not
        // fit in shared memory and may require many blocks to process it entirely
        int8_t *valid_byte =
          &output_data[block.buffer_num][row_offsets[row] + validity_offset + (col  / 8)];
        cudf::size_type byte_bit_offset = col % 8;
        uint64_t fixup_bytes            = reinterpret_cast<uint64_t>(valid_byte) % 4;
        int32_t *valid_int              = reinterpret_cast<int32_t *>(valid_byte - fixup_bytes);
        cudf::size_type int_bit_offset  = byte_bit_offset + (fixup_bytes * 8);
        if (debug_print) { printf("Outputting validity to %p\n", valid_byte); }
        // Now copy validity for the column
        if (input_nm[col]) {
          if (bit_is_set(input_nm[col], row)) {
            atomicOr_block(valid_int, 1 << int_bit_offset);
          } else {
            atomicAnd_block(valid_int, ~(1 << int_bit_offset));
          }
        } else {
          // It is valid so just set the bit
          atomicOr_block(valid_int, 1 << int_bit_offset);
        }
      }  // end row

      col_offset += col_sizes[col] * rows_in_block;
    }
  }  // end col

  // wait for the data to be totally copied into shared memory
  __syncthreads();

  // Step 2: Copy the data from shared memory to final destination
  // each block is potentially a slice of the table, so no assumptions
  // can be made about alignments. We do know that the alignment in shared
  // memory matches the final destination alignment. Also note that
  // we are not writing to entirely contiguous destinations as each
  // row in shared memory may not be an entire row of the destination.
  //
  auto const thread_start_offset = threadIdx.x * 8;
  auto const thread_stride       = blockDim.x * 8;
  auto const end_offset          = shmem_row_size * rows_in_block;

  if (debug_print) {
    printf("writing final data from %d to %d at stride %d\n",
           thread_start_offset,
           shmem_row_size * rows_in_block,
           thread_stride);
    printf("rows in block %d\n", rows_in_block);
  }
  for (auto src_offset = thread_start_offset; src_offset < end_offset;
       src_offset += thread_stride) {
    auto const output_row_num = src_offset / shmem_row_size;
    auto const row_offset     = row_offsets[block.start_row + output_row_num];
    auto const col_offset     = src_offset % shmem_row_size;
    int8_t *output_ptr        = &output_data[block.buffer_num][row_offset + col_offset];
    int8_t *input_ptr         = &shared_data[src_offset];

    // three cases to worry about here
    // 1) first 8-byte part of a large row - dest_shim_offset bytes of pad at the front
    // 2) last 8-byte part of a large row - some bytes of pad at the end
    // 3) corner case of <= 8 bytes of data, which means dest_shim_offset bytes of pad at the front
    // AND potentially pad at the rear

    // we know the real number of bytes in a row, so we can figure out if we are in case 3 easily.
    // 1st case is when we're at some even multiple of shmem_row_size offset.
    // 2nd case is when offset + 8 is some even multiple of shmem_row_size.
    // must be an 8 byte copy

    // there is a chance we have a 0 dest_shim_offset and an 8 byte thing to copy, optimize?
    if (real_bytes_in_row + dest_shim_offset <= 8) {
      // case 3, we want to copy real_bytes_in_row bytes
      auto const num_single_bytes = real_bytes_in_row - dest_shim_offset;
      for (auto i = 0; i < num_single_bytes; ++i) {
        if (debug_print) {
          printf("case 3 - %d single byte final write %p(%d) -> %p\n",
                 num_single_bytes,
                 &input_ptr[i + dest_shim_offset],
                 input_ptr[i + dest_shim_offset],
                 &output_ptr[i]);
        }
        output_ptr[i] = input_ptr[i + dest_shim_offset];
      }
    } else if (dest_shim_offset > 0 && src_offset % shmem_row_size == 0) {
      // first byte with leading pad
      auto const num_single_bytes = 8 - dest_shim_offset;
      for (auto i = 0; i < num_single_bytes; ++i) {
        if (debug_print) {
          printf(
            "single byte final write %p -> %p\n", &input_ptr[i + dest_shim_offset], &output_ptr[i]);
        }
        output_ptr[i] = input_ptr[i + dest_shim_offset];
      }
    } else if ((src_offset + 8) % shmem_row_size == 0 &&
               (real_bytes_in_row + dest_shim_offset) % 8 > 0) {
      // last bytes of a row
      auto const num_single_bytes = (real_bytes_in_row + dest_shim_offset) % 8;
      for (auto i = 0; i < num_single_bytes; ++i) {
        if (debug_print) {
          printf("single trailing byte final write %p -> %p\n",
                 &input_ptr[i + dest_shim_offset],
                 &output_ptr[i]);
        }
        output_ptr[i] = input_ptr[i + dest_shim_offset];
      }
    } else {
      // copy 8 bytes aligned
      const int64_t *long_col_input = reinterpret_cast<const int64_t *>(input_ptr);
      if (debug_print) {
        printf(
          "long final write %p -> %p\n", long_col_input, reinterpret_cast<int64_t *>(output_ptr));
      }
      *reinterpret_cast<int64_t *>(output_ptr) = *long_col_input;
    }
  }
}

/**
 * @brief copy data from row-based format to cudf columns
 *
 * @param num_rows total number of rows in the table
 * @param num_columns total number of columns in the table
 * @param shmem_used_per_block amount of shared memory that is used by a block
 * @param offsets
 * @param output_data
 * @param output_nm
 * @param col_sizes array of sizes for each element in a column - one per column
 * @param col_offsets offset into input data row for each column's start
 * @param block_infos information about the blocks of work
 * @param input_data pointer to input data
 *
 */
__global__ void copy_to_columns(const size_type num_rows,
                                const size_type num_columns,
                                const size_type shmem_used_per_block,
                                const size_type *offsets,
                                int8_t **output_data,
                                cudf::bitmask_type **output_nm,
                                const size_type *col_sizes,
                                const size_type *col_offsets,
                                const block_info *block_infos,
                                const int8_t *input_data)
{
  // We are going to copy the data in two passes.
  // The first pass copies a chunk of data into shared memory.
  // The second pass copies that chunk from shared memory out to the final location.

  // Because shared memory is limited we copy a subset of the rows at a time.
  // This has been broken up for us in the block_info struct, so we don't have
  // any calculation to do here, but it is important to note.

  constexpr bool debug_print = false; //blockIdx.x == 0 && threadIdx.x == 0;

  if (debug_print) {
    printf("%d %d - %d rows, %d columns\n", threadIdx.x, blockIdx.x, num_rows, num_columns);
    printf("block infos are at %p and my index is %d\n", block_infos, blockIdx.x);
    /*    printf("Row Offsets:\n");
    for (int i=0; i<num_rows; ++i) {
    printf("%d: %d\n", i, row_offsets[i]);
    }*/
    printf("output data to %p\n", output_data[block_infos[blockIdx.x].buffer_num]);
  }
//  else { return; }

  for (int block_offset = 0; block_offset < NUM_BLOCKS_PER_KERNEL_TO_COLUMNS; ++block_offset) {
    auto this_block_index = blockIdx.x*NUM_BLOCKS_PER_KERNEL_TO_COLUMNS + block_offset;
    if (this_block_index > blockDim.x) {
      break;
    }
    auto block               = block_infos[this_block_index];
  auto const rows_in_block = block.end_row - block.start_row + 1;
  auto const cols_in_block = block.end_col - block.start_col + 1;
  extern __shared__ int8_t shared_data[];

  // copy data from our block's window to shared memory
  // offsets information can get us on the row, then we need to know where the column
  // starts to offset into the row data.

  // each thread is responsible for 8-byte chunks starting at threadIdx.x and striding
  // at blockDim.x. If the 8-byte chunk falls on the boundary of the window, then the
  // thread may copy less than 8 bytes. Even if at the beginning of the window, because
  // every internal copy is aligned to 8-byte boundaries.
  //
  //  thread 0 thread 1 thread 2 thread 3 thread 4 thread 5
  //  01234567 89abcdef 01234567 89abcdef 01234567 89abcdef
  //  xxxbbbbb bbbbbbbb bbbbbbbb bbbbbbbb bbbbbbbb bbxxxxxx
  // |        |        |        |        |        |        |
  //
  //

  auto const window_start_quad = col_offsets[block.start_col] / 8;
  auto const window_end_quad   = (col_offsets[block.end_col] + col_sizes[block.end_col] + 7) / 8;
  auto const window_quad_width = window_end_quad - window_start_quad;
  auto const total_quads       = window_quad_width * rows_in_block;
  auto const shared_memory_starting_pad = col_offsets[block.start_col] & 0x7;

  if (debug_print) {
    printf("col_offsets[%d]: %d, col_offsets[%d]: %d col_sizes[%d]: %d\n", block.start_col, col_offsets[block.start_col], block.end_col, col_offsets[block.end_col], block.end_col, col_sizes[block.end_col]);
    printf("window start quad is %d, window end quad is %d\n", window_start_quad, window_end_quad);
    printf("window quad width is %d and there are %d total quads\n%d shared memory starting pad\n", window_quad_width, total_quads, shared_memory_starting_pad);
  }

  // the copy to shared memory will be greedy. We know that the data is 8-byte aligned, so we won't
  // access illegal memory by doing 8-byte aligned copies, so we can copy 8-byte aligned. This will
  // result in the window edges being duplicated across blocks, but we can copy the padding as well
  // to speed up our transfers to shared memory.
  for (int i = threadIdx.x; i < total_quads; i += blockDim.x) {
    auto const relative_row = i / window_quad_width;
    auto const absolute_row = relative_row + block.start_row;
    //auto const row           = i / window_quad_width;
    auto const offset_in_row = i % window_quad_width * 8;
    auto const shmem_dest    = &shared_data[i * 8];

    if (debug_print) {
      printf("relative_row: %d, absolute_row: %d, offset_in_row: %d, shmem_dest: %p\n", relative_row, absolute_row, offset_in_row, shmem_dest);
      printf("offsets is %p\n", offsets);
      printf("offsets[%d]: %d\n", absolute_row, offsets[absolute_row]);
      printf("input_data[%d] will be dereferenced\n", offsets[absolute_row] + offset_in_row);
    }

    // full 8-byte copy
    const int64_t *long_col_input =
      reinterpret_cast<const int64_t *>(&input_data[offsets[absolute_row] + offset_in_row]);
    if (debug_print) { 
      printf("which will be address %p\n", long_col_input);
      printf("%p <- long %lu\n", shmem_dest, *long_col_input); }
    *reinterpret_cast<int64_t *>(shmem_dest) = *long_col_input;
  }

  __syncthreads();

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

    auto const shared_memory_row_offset = window_quad_width * 8 * relative_row;
    auto const shared_memory_offset = col_offsets[absolute_col] - col_offsets[block.start_col] +
                                      shared_memory_row_offset + shared_memory_starting_pad;
    auto const column_size = col_sizes[absolute_col];

    int8_t *shmem_src = &shared_data[shared_memory_offset];
    int8_t *dst       = &output_data[absolute_col][absolute_row * column_size];

    if (debug_print) {
      printf("relative_col: %d, relative_row: %d, absolute_col: %d, absolute_row: %d, shared_mmeory_row_offset: %d, shared_memory_offset: %d,"
      " column_size: %d, shmem_src: %p, dst: %p\n", relative_col, relative_row, absolute_col, absolute_row, shared_memory_row_offset, shared_memory_offset, column_size,
    shmem_src, dst) ;
    }
    switch (column_size) {
      case 1: {
        if (debug_print) { printf("%p <- byte %d\n", dst, *shmem_src); }
        *dst = *shmem_src;
        break;
      }
      case 2: {
        const int16_t *short_col_input = reinterpret_cast<const int16_t *>(shmem_src);
        if (debug_print) { printf("%p <- short %d\n", dst, *short_col_input); }
        *reinterpret_cast<int16_t *>(dst) = *short_col_input;
        break;
      }
      case 4: {
        const int32_t *int_col_input = reinterpret_cast<const int32_t *>(shmem_src);
        if (debug_print) { printf("%p <- int 0x%x\n", dst, *int_col_input); }
        *reinterpret_cast<int32_t *>(dst) = *int_col_input;
        break;
      }
      case 8: {
        const int64_t *long_col_input = reinterpret_cast<const int64_t *>(shmem_src);
        if (debug_print) { printf("%p <- long %lu\n", dst, *long_col_input); }
        *reinterpret_cast<int64_t *>(dst) = *long_col_input;
        break;
      }
      default: {
        if (debug_print) {
          printf("byte for byte copy due to size %d of column %d\n", column_size, absolute_col);
        }
        // TODO this should just not be supported for fixed width columns, but just in case...
        for (cudf::size_type b = 0; b < column_size; b++) { dst[b] = shmem_src[b]; }
        break;
      }
    }
  }

  // now handle validity. Each thread is responsible for 32 rows in 8 columns.
  // to prevent indexing issues with a large number of threads, this is compressed
  // to a single loop like above. TODO: investigate using shared memory here
  auto const validity_batches_per_col = (num_rows + 31) / 32;
  auto const validity_batches_total   = std::max(1, validity_batches_per_col * (num_columns / 8));
  if (debug_print && threadIdx.x == 0 && blockIdx.x == 0) {
    printf("validity_batched_per_col is %d\nvalidity_batches_total is %d for %d rows\n%d blocks of %d threads\n", validity_batches_per_col, validity_batches_total, num_rows, gridDim.x, blockDim.x);
  }
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < validity_batches_total; index += blockDim.x * gridDim.x) {
    auto const start_col       = (index * 8) / validity_batches_per_col;
    auto const batch           = index % validity_batches_per_col;
    auto const starting_row    = batch * 32;
    auto const validity_offset = col_offsets[num_columns] + (start_col / 8);

    if (debug_print) {
      printf("%d-%d: cols: %d-%d, word index: %d, batch: %d, starting_row: %d, +validity_offset: %d, index: %d, stride: %d\n", threadIdx.x, blockIdx.x, start_col, start_col + 7, (start_col / 8), batch, starting_row, validity_offset, index, blockDim.x * gridDim.x);
    }

    // one for each column
    int32_t dst_validity[8] = {0};
    for (int row = starting_row; row < std::min(num_rows, starting_row + 32); ++row) {
      int8_t const * const validity_ptr = &input_data[offsets[row] + validity_offset];

      if (debug_print) {
        printf("%d: validity_ptr is %p for row %d\n", threadIdx.x, validity_ptr, row);
      }
  
      auto const val_byte     = *validity_ptr;

      for (int i=0; i<std::min(num_columns - start_col, 8); ++i) {
        auto const src_shift    = (start_col + i) % 8;
        auto const dst_shift    = row % 32;
        auto const src_bit_mask = 1 << src_shift;
        if (debug_print) {
          printf("%d-%d: src bit mask is 0x%x, src shift is 0x%x and dst shift is 0x%x, validity bit is 0x%x\n", threadIdx.x, blockIdx.x, src_bit_mask, src_shift, dst_shift, (val_byte & src_bit_mask) >> src_shift);
        }
  //      auto const dst_bit_mask = 1 << dst_shift;
        dst_validity[i] |= (((val_byte & src_bit_mask) >> src_shift) << dst_shift);
      }
    }
    

    for (int i=0; i<std::min(num_columns - start_col, 8); ++i) {
      int32_t *validity_ptr = reinterpret_cast<int32_t *>(output_nm[start_col + i] + (starting_row / 32));
      if (debug_print) {
        printf("%d-%d: validity write output_nm[%d][%d] - %p <- %d\n", threadIdx.x, blockIdx.x, start_col + i, starting_row, validity_ptr, dst_validity[i]);
      }
      *validity_ptr         = dst_validity[i];
    }
  }
}
}

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
                                        const cudf::size_type size_per_row,
                                        dim3 &blocks,
                                        dim3 &threads)
{
  // We have found speed degrades when a thread handles more than 4 columns.
  // Each block is 2 dimensional. The y dimension indicates the columns.
  // We limit this to 32 threads in the y dimension so we can still
  // have at least 32 threads in the x dimension (1 warp) which should
  // result in better coalescing of memory operations. We also
  // want to guarantee that we are processing a multiple of 32 threads
  // in the x dimension because we use atomic operations at the block
  // level when writing validity data out to main memory, and that would
  // need to change if we split a word of validity data between blocks.
  int y_block_size = (num_columns + 3) / 4;
  if (y_block_size > 32) { y_block_size = 32; }
  int x_possible_block_size = 1024 / y_block_size;
  // 48KB is the default setting for shared memory per block according to the cuda tutorials
  // If someone configures the GPU to only have 16 KB this might not work.
  int max_shared_size = 48 * 1024;
  int max_block_size  = max_shared_size / size_per_row;
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
  blocks.x  = num_blocks;
  blocks.y  = 1;
  blocks.z  = 1;
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
static std::unique_ptr<cudf::column> fixed_width_convert_to_rows(
  const cudf::size_type start_row,
  const cudf::size_type num_rows,
  const cudf::size_type num_columns,
  const cudf::size_type size_per_row,
  rmm::device_uvector<cudf::size_type> &column_start,
  rmm::device_uvector<cudf::size_type> &column_size,
  rmm::device_uvector<const int8_t *> &input_data,
  rmm::device_uvector<const cudf::bitmask_type *> &input_nm,
  const cudf::scalar &zero,
  const cudf::scalar &scalar_size_per_row,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource *mr)
{
  int64_t total_allocation = size_per_row * num_rows;
  // We made a mistake in the split somehow
  CUDF_EXPECTS(total_allocation < std::numeric_limits<int>::max(), "Table is too large to fit!");

  // Allocate and set the offsets row for the byte array
  std::unique_ptr<cudf::column> offsets =
    cudf::detail::sequence(num_rows + 1, zero, scalar_size_per_row, stream);

  std::unique_ptr<cudf::column> data =
    cudf::make_numeric_column(cudf::data_type(cudf::type_id::INT8),
                              static_cast<cudf::size_type>(total_allocation),
                              cudf::mask_state::UNALLOCATED,
                              stream,
                              mr);

  dim3 blocks;
  dim3 threads;
  int shared_size =
    detail::calc_fixed_width_kernel_dims(num_columns, num_rows, size_per_row, blocks, threads);

  copy_from_fixed_width_columns<<<blocks, threads, shared_size, stream.value()>>>(
    start_row,
    num_rows,
    num_columns,
    size_per_row,
    column_start.data(),
    column_size.data(),
    input_data.data(),
    input_nm.data(),
    data->mutable_view().data<int8_t>());

  return cudf::make_lists_column(num_rows,
                                 std::move(offsets),
                                 std::move(data),
                                 0,
                                 rmm::device_buffer{0, rmm::cuda_stream_default, mr},
                                 stream,
                                 mr);
}

static cudf::data_type get_data_type(const cudf::column_view &v) { return v.type(); }

static inline bool are_all_fixed_width(std::vector<cudf::data_type> const &schema)
{
  return std::all_of(
    schema.begin(), schema.end(), [](const cudf::data_type &t) { return cudf::is_fixed_width(t); });
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
                                                 std::vector<cudf::size_type> &column_size)
{
  // We guarantee that the start of each column is 64-bit aligned so anything can go
  // there, but to make the code simple we will still do an alignment for it.
  int32_t at_offset = 0;
  for (auto col = schema.begin(); col < schema.end(); col++) {
    cudf::size_type s = cudf::size_of(*col);
    column_size.emplace_back(s);
    std::size_t allocation_needed = s;
    std::size_t alignment_needed  = allocation_needed;  // They are the same for fixed width types
    at_offset                     = align_offset(at_offset, alignment_needed);
    column_start.emplace_back(at_offset);
    at_offset += allocation_needed;
  }

  // Now we need to add in space for validity
  // Eventually we can think about nullable vs not nullable, but for now we will just always add it
  // in
  int32_t validity_bytes_needed = (schema.size() + 7) / 8;
  // validity comes at the end and is byte aligned so we can pack more in.
  at_offset += validity_bytes_needed;
  // Now we need to pad the end so all rows are 64 bit aligned
  return align_offset(at_offset, 8);  // 8 bytes (64 bits)
}

template <typename iterator>
static size_type compute_column_information(
  iterator begin,
  iterator end,
  std::vector<size_type> &column_starts,
  std::vector<size_type> &column_sizes)//,
  //std::function<void(T)> nested_type_cb)
{
  size_type fixed_width_size_per_row = 0;
  for (auto cv = begin; cv != end; ++cv) {
    auto col_type    = std::get<0>(*cv);
    bool nested_type = col_type.id() == type_id::LIST || col_type.id() == type_id::STRING;

//    if (nested_type && nested_type_cb) { nested_type_cb(cv->get<1>()); }

    // a list or string column will write a single uint64
    // of data here for offset/length
    auto col_size = nested_type ? 8 : size_of(col_type);

    // align size for this type
    std::size_t const alignment_needed = col_size;  // They are the same for fixed width types
    fixed_width_size_per_row = detail::align_offset(fixed_width_size_per_row, alignment_needed);
    column_starts.push_back(fixed_width_size_per_row);
    column_sizes.push_back(col_size);
    fixed_width_size_per_row += col_size;
  }

  auto validity_offset = detail::align_offset(fixed_width_size_per_row, 4);
  column_starts.push_back(validity_offset);

  return fixed_width_size_per_row;
}

//#define DEBUG

static std::vector<block_info> build_block_infos(std::vector<size_type> const &column_sizes,
                                                 std::vector<size_type> const &column_starts,
                                                 std::vector<row_batch> const &row_batches,
                                                 size_type const total_number_of_rows,
                                                 size_type const &shmem_limit_per_block)
{
  std::vector<block_info> block_infos;

  // block infos are organized with the windows going "down" the columns
  // this provides the most coalescing of memory access
  int current_window_width     = 0;
  int current_window_start_col = 0;

  // build the blocks for a specific set of columns
  auto build_blocks = [&block_infos, &row_batches, total_number_of_rows](
                        int const start_col, int const end_col, int const desired_window_height) {
    int current_window_start_row = 0;
    int current_window_row_batch = 0;
    int rows_left_in_batch       = row_batches[current_window_row_batch].row_count;
    int i                        = 0;
    while (i < total_number_of_rows) {
      if (rows_left_in_batch == 0) {
        current_window_row_batch++;
        rows_left_in_batch = row_batches[current_window_row_batch].row_count;
      }
      int const window_height = std::min(desired_window_height, rows_left_in_batch);

      block_infos.emplace_back(detail::block_info{
        start_col,
        current_window_start_row,
        end_col,
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
  int const window_height = std::min(
    std::min(size_type(sqrt(shmem_limit_per_block)) / column_sizes[0], total_number_of_rows),
    row_batches[0].row_count);
#if defined(DEBUG)
  printf(
    "sqrt(shmem_limit_per_block) / column_sizes[0] is %d and num_rows is %d, batch row count is %d - which makes window height "
    "%d\n",
    size_type(sqrt(shmem_limit_per_block)) / column_sizes[0],
    total_number_of_rows,
    row_batches[0].row_count,
    window_height);
#endif

  int row_size = 0;

  // march each column and build the blocks of appropriate sizes
  for (unsigned int col = 0; col < column_sizes.size(); ++col) {
    auto const col_size = column_sizes[col];

    // align size for this type
    std::size_t alignment_needed = col_size;  // They are the same for fixed width types
    auto row_size_aligned        = detail::align_offset(row_size, alignment_needed);
    auto row_size_with_this_col  = row_size_aligned + col_size;
    auto row_size_with_end_pad   = detail::align_offset(row_size_with_this_col, 8);

    if (row_size_with_end_pad * window_height > shmem_limit_per_block) {
#if defined(DEBUG)
      printf(
        "Window size %d too large at column %d, bumping back to build windows of size %d(cols "
        "%d-%d), which is %d tall. Row size is too large at %d and ok at %d(aligned overall is %d) "
        "for shared mem size %d\n",
        row_size_with_end_pad * window_height,
        col,
        row_size * window_height,
        current_window_start_col,
        col - 1,
        window_height,
        row_size_with_end_pad,
        row_size,
        row_size_aligned,
        shmem_limit_per_block);
#endif
      // too large, close this window, generate vertical blocks and restart
      build_blocks(current_window_start_col, col - 1, window_height);
      row_size =
        detail::align_offset((column_starts[col] + column_sizes[col]) & 7, alignment_needed);
#if defined(DEBUG)
      printf(
        "New window starting with offset %d and row size %d to be %d (previous column offset %d+%d "
        "or %d)\n",
        row_size,
        col_size,
        row_size + col_size,
        column_starts[col - 1],
        column_sizes[col - 1],
        column_starts[col - 1] + column_sizes[col - 1]);
#endif
      row_size += col_size;  // alignment required for shared memory window boundary to match
                             // alignment of output row
      current_window_start_col = col;
      current_window_width     = 0;
    } else {
      row_size = row_size_with_this_col;
      current_window_width++;
    }
  }

  // build last set of blocks
  if (current_window_width > 0) {
    build_blocks(current_window_start_col, (int)column_sizes.size()-1, window_height);
  }

  return block_infos;
}
}  // namespace detail

#if defined(DEBUG)
  void pretty_print(uint64_t i) {
    if (i > (1 * 1024 * 1024 * 1024)) {
      printf("%.2f GB", i / float(1 * 1024 * 1024 * 1024));
    } else if (i > (1 * 1024 * 1024)) {
      printf("%.2f MB", i / float(1 * 1024 * 1024));
    } else if (i > (1 * 1024)) {
      printf("%.2f KB", float(i / 1024));
    } else {
      printf("%lu Bytes", i);
    }
  }
#endif

std::vector<std::unique_ptr<cudf::column>> convert_to_rows2(cudf::table_view const &tbl,
                                                            rmm::cuda_stream_view stream,
                                                            rmm::mr::device_memory_resource *mr)
{
  // not scientifically chosen - the ideal window is long enough to allow coalesced reads of the
  // data, but small enough that multiple columns fit in memory so the writes can coalese as well.
  // Potential optimization for window sizes.
  const size_type num_columns = tbl.num_columns();
  const size_type num_rows    = tbl.num_rows();

  int device_id;
  CUDA_TRY(cudaGetDevice(&device_id));
  int shmem_limit_per_block;
  CUDA_TRY(
    cudaDeviceGetAttribute(&shmem_limit_per_block, cudaDevAttrMaxSharedMemoryPerBlock, device_id));

#if defined(DEBUG)
  size_t free, total;
  cudaMemGetInfo(&free, &total);
  printf("%lu/%lu Memory\n", free, total);
#endif

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
  std::vector<const int8_t *> input_data;
  std::vector<bitmask_type const *> input_nm;
  input_data.reserve(num_columns);
  input_nm.reserve(num_columns);
  for (size_type column_number = 0; column_number < num_columns; column_number++) {
    column_view cv      = tbl.column(column_number);
    auto const col_type = cv.type();
    bool nested_type    = col_type.id() == type_id::LIST || col_type.id() == type_id::STRING;

    if (!nested_type) {
      input_data.emplace_back(cv.data<int8_t>());
      input_nm.emplace_back(cv.null_mask());
    }
  }

  auto dev_input_data = make_device_uvector_async(input_data, stream, mr);
  auto dev_input_nm   = make_device_uvector_async(input_nm, stream, mr);

  std::vector<size_type> row_sizes;     // size of each row in bytes including any alignment padding
  std::vector<size_type> row_offsets;   // offset from the start of the data to this row
  std::vector<size_type> column_sizes;  // byte size of each column
  std::vector<size_type> column_starts;  // offset of column inside a row including alignment
  std::vector<column_view>
    variable_width_columns;  // list of the variable width columns in the table
  row_sizes.reserve(num_rows);
  row_offsets.reserve(num_rows);
  column_sizes.reserve(num_columns);
  column_starts.reserve(num_columns + 1);  // we add a final offset for validity data start

  auto iter = thrust::make_transform_iterator(thrust::make_counting_iterator(0), [&tbl](auto i) -> std::tuple<data_type, column_view const> {
    return std::make_tuple(tbl.column(i).type(), tbl.column(i));
  });

  size_type fixed_width_size_per_row = detail::compute_column_information(
    iter,
    iter + num_columns,
    column_starts,
    column_sizes);//,
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

#if defined(DEBUG)
  printf("validity offset will be %d + %d = %d\n",
         column_starts.back(),
         column_sizes.back(),
         column_starts.back() + column_sizes.back());
#endif


  auto dev_col_sizes  = make_device_uvector_async(column_sizes, stream, mr);
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

  uint64_t row_batch_size   = 0;
  uint64_t total_table_size = 0;
  size_type row_batch_rows  = 0;
  uint64_t row_offset       = 0;

  // fixed_width_size_per_row is the size of the fixed-width portion of a row. We need to then
  // calculate the size of each row's variable-width data and validity as well.
  auto validity_size = num_bitmask_words(num_columns) * 4;
  for (int row = 0; row < num_rows; ++row) {
    auto aligned_row_batch_size =
      detail::align_offset(row_batch_size, 8);  // rows are 8 byte aligned
    row_sizes[row] = fixed_width_size_per_row;
    // validity is byte aligned
    row_sizes[row] += validity_size;
    // variable width data is 8-byte aligned
    row_sizes[row] = detail::align_offset(row_sizes[row], 8) +
                     calculate_variable_width_row_data_size(row);  // rows are 8 byte aligned

    if ((uint64_t)aligned_row_batch_size + row_sizes[row] >
        (uint64_t)std::numeric_limits<size_type>::max()) {
      // a new batch starts at the last 32-row boundary
      row_batches.push_back(
        detail::row_batch{static_cast<size_type>(row_batch_size), row_batch_rows & ~31});
      row_batch_size         = 0;
      row_batch_rows         = row_batch_rows & 31;
      row_offset             = 0;
      aligned_row_batch_size = 0;
    }
    row_offset = detail::align_offset(row_offset, 8);  // rows are 8 byte aligned
    row_offsets.push_back(row_offset);
    row_batch_size = aligned_row_batch_size + row_sizes[row];
    row_offset += row_sizes[row];
    total_table_size = detail::align_offset(total_table_size, 8);  // rows are 8 byte aligned
    total_table_size += row_sizes[row];
    row_batch_rows++;
  }
  if (row_batch_size > 0) {
    row_batches.push_back(detail::row_batch{static_cast<size_type>(row_batch_size), row_batch_rows});
  }

  auto dev_row_offsets = make_device_uvector_async(row_offsets, stream, mr);

#if defined(DEBUG)
  printf("%d rows and %d columns in table\n", num_rows, num_columns);
  printf("%lu batches:\n", row_batches.size());
  for (auto i = 0; i < (int)row_batches.size(); ++i) {
    printf("%d: %d rows, ", i, row_batches[i].row_count);
    pretty_print(row_batches[i].num_bytes);
    printf("\n");
  }
#endif

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

#if defined(DEBUG)
  printf("%lu windows for %d columns, %d rows to fit in ",
         block_infos.size(),
         block_infos[0].end_col - block_infos[0].start_col + 1,
         block_infos[0].end_row - block_infos[0].start_row);
  pretty_print(shmem_limit_per_block);
  printf(" shared mem(");
  pretty_print(fixed_width_size_per_row);
  printf("/row, %d columns, %d rows, ", num_columns, num_rows);
  pretty_print(total_table_size);
  printf(" total):\n");
#endif

  auto dev_block_infos = make_device_uvector_async(block_infos, stream, mr);

  // blast through the entire table and convert it
  dim3 blocks(block_infos.size());
  #if defined(DEBUG) || 1
  dim3 threads(std::min(std::min(512, shmem_limit_per_block / 8), (int)total_table_size));
  #else
  dim3 threads(std::min(std::min(1024, shmem_limit_per_block / 8), (int)total_table_size));
  #endif
#if defined(DEBUG)
  printf("Launching kernel with %d blocks, %d threads, ", blocks.x, threads.x);
  pretty_print(shmem_limit_per_block);
  printf(" shared memory\n");
#endif
  copy_from_columns<<<blocks, threads, shmem_limit_per_block, stream.value()>>>(
    num_rows,
    num_columns,
    dev_input_data.data(),
    dev_input_nm.data(),
    dev_col_sizes.data(),
    dev_col_starts.data(),
    dev_block_infos.data(),
    dev_row_offsets.data(),
    reinterpret_cast<int8_t **>(dev_output_data.data()));

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
    auto offsets     = std::make_unique<column>(
      data_type{type_id::INT32}, (size_type)offset_vals.size(), dev_offsets.release());

    auto data = std::make_unique<column>(
      data_type{cudf::type_id::INT8}, row_batches[i].num_bytes, std::move(output_buffers[i]));

    ret.push_back(cudf::make_lists_column(row_batches[i].row_count,
                                          std::move(offsets),
                                          std::move(data),
                                          0,
                                          rmm::device_buffer{0, rmm::cuda_stream_default, mr},
                                          stream,
                                          mr));
  }

  return ret;
}

std::vector<std::unique_ptr<cudf::column>> convert_to_rows(cudf::table_view const &tbl,
                                                           rmm::cuda_stream_view stream,
                                                           rmm::mr::device_memory_resource *mr)
{
  const cudf::size_type num_columns = tbl.num_columns();

  std::vector<cudf::data_type> schema;
  schema.resize(num_columns);
  std::transform(tbl.begin(), tbl.end(), schema.begin(), detail::get_data_type);

  if (detail::are_all_fixed_width(schema)) {
    std::vector<cudf::size_type> column_start;
    std::vector<cudf::size_type> column_size;

    int32_t size_per_row  = detail::compute_fixed_width_layout(schema, column_start, column_size);
    auto dev_column_start = make_device_uvector_async(column_start, stream, mr);
    auto dev_column_size  = make_device_uvector_async(column_size, stream, mr);

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
    auto dev_input_nm   = make_device_uvector_async(input_nm, stream, mr);

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
      row_count                 = row_count > max_rows_per_batch ? max_rows_per_batch : row_count;
      ret.emplace_back(detail::fixed_width_convert_to_rows(row_start,
                                                           row_count,
                                                           num_columns,
                                                           size_per_row,
                                                           dev_column_start,
                                                           dev_column_size,
                                                           dev_input_data,
                                                           dev_input_nm,
                                                           *zero,
                                                           *step,
                                                           stream,
                                                           mr));
    }

    return ret;
  } else {
    CUDF_FAIL("Only fixed width types are currently supported");
  }
}

std::unique_ptr<cudf::table> convert_from_rows2(cudf::lists_column_view const &input,
                                                std::vector<cudf::data_type> const &schema,
                                                rmm::cuda_stream_view stream,
                                                rmm::mr::device_memory_resource *mr)
{
  // verify that the types are what we expect
  cudf::column_view child = input.child();
  cudf::type_id list_type = child.type().id();
  CUDF_EXPECTS(list_type == cudf::type_id::INT8 || list_type == cudf::type_id::UINT8,
               "Only a list of bytes is supported as input");

  cudf::size_type num_columns = schema.size();
  cudf::size_type num_rows    = input.parent().size();

  int device_id;
  CUDA_TRY(cudaGetDevice(&device_id));
  int shmem_limit_per_block;
  CUDA_TRY(
    cudaDeviceGetAttribute(&shmem_limit_per_block, cudaDevAttrMaxSharedMemoryPerBlock, device_id));

  shmem_limit_per_block /= NUM_BLOCKS_PER_KERNEL_TO_COLUMNS;

  std::vector<cudf::size_type> column_starts;
  std::vector<cudf::size_type> column_sizes;

  auto iter = thrust::make_transform_iterator(thrust::make_counting_iterator(0), [&schema](auto i) {
    return std::make_tuple(schema[i], nullptr);
  });
  size_type fixed_width_size_per_row = detail::compute_column_information(
    iter, iter + num_columns, column_starts, column_sizes);//, [](void *) {});

  size_type validity_size = num_bitmask_words(num_columns) * 4;

  size_type row_size = detail::align_offset(fixed_width_size_per_row + validity_size, 8);

  // Ideally we would check that the offsets are all the same, etc. but for now
  // this is probably fine
  CUDF_EXPECTS(row_size * num_rows == child.size(),
               "The layout of the data appears to be off");
  auto dev_col_starts = make_device_uvector_async(column_starts, stream, mr);
  auto dev_col_sizes  = make_device_uvector_async(column_sizes, stream, mr);

  // build the row_batches from the passed in list column
  std::vector<detail::row_batch> row_batches;

  row_batches.push_back(detail::row_batch{child.size(), num_rows});

  // Allocate the columns we are going to write into
  std::vector<std::unique_ptr<cudf::column>> output_columns;
  std::vector<int8_t *> output_data;
  std::vector<cudf::bitmask_type *> output_nm;
  for (cudf::size_type i = 0; i < num_columns; i++) {
    auto column = cudf::make_fixed_width_column(
      schema[i], num_rows, cudf::mask_state::UNINITIALIZED, stream, mr);
    auto mut = column->mutable_view();
    output_data.emplace_back(mut.data<int8_t>());
    output_nm.emplace_back(mut.null_mask());
    output_columns.emplace_back(std::move(column));
  }

  auto dev_output_data = make_device_uvector_async(output_data, stream, mr);
  auto dev_output_nm   = make_device_uvector_async(output_nm, stream, mr);

  std::vector<detail::block_info> block_infos =
    build_block_infos(column_sizes, column_starts, row_batches, num_rows, shmem_limit_per_block);

  auto dev_block_infos = make_device_uvector_async(block_infos, stream, mr);

  dim3 blocks((block_infos.size() + (NUM_BLOCKS_PER_KERNEL_TO_COLUMNS - 1)) / NUM_BLOCKS_PER_KERNEL_TO_COLUMNS);
  #if defined(DEBUG) || 1
  dim3 threads(std::min(std::min(512, shmem_limit_per_block / 8), (int)child.size()));
  #else
  dim3 threads(std::min(std::min(1024, shmem_limit_per_block / 8), (int)child.size()));
  #endif
#if defined(DEBUG)
  printf("Launching kernel with %d blocks, %d threads, ", blocks.x, threads.x);
  pretty_print(shmem_limit_per_block);
  printf(" shared memory\n");
#endif
  detail::copy_to_columns<<<blocks, threads, shmem_limit_per_block, stream.value()>>>(
    num_rows,
    num_columns,
    shmem_limit_per_block,
    input.offsets().data<size_type>(),
    dev_output_data.data(),
    dev_output_nm.data(),
    dev_col_sizes.data(),
    dev_col_starts.data(),
    dev_block_infos.data(),
    child.data<int8_t>());

  return std::make_unique<cudf::table>(std::move(output_columns));
}

std::unique_ptr<cudf::table> convert_from_rows(cudf::lists_column_view const &input,
                                               std::vector<cudf::data_type> const &schema,
                                               rmm::cuda_stream_view stream,
                                               rmm::mr::device_memory_resource *mr)
{
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
      auto column = cudf::make_fixed_width_column(
        schema[i], num_rows, cudf::mask_state::UNINITIALIZED, stream, mr);
      auto mut = column->mutable_view();
      output_data.emplace_back(mut.data<int8_t>());
      output_nm.emplace_back(mut.null_mask());
      output_columns.emplace_back(std::move(column));
    }

    auto dev_output_data = make_device_uvector_async(output_data, stream, mr);
    auto dev_output_nm   = make_device_uvector_async(output_nm, stream, mr);

    dim3 blocks;
    dim3 threads;
    int shared_size =
      detail::calc_fixed_width_kernel_dims(num_columns, num_rows, size_per_row, blocks, threads);

    detail::copy_to_fixed_width_columns<<<blocks, threads, shared_size, stream.value()>>>(
      num_rows,
      num_columns,
      size_per_row,
      dev_column_start.data(),
      dev_column_size.data(),
      dev_output_data.data(),
      dev_output_nm.data(),
      child.data<int8_t>());

    return std::make_unique<cudf::table>(std::move(output_columns));
  } else {
    CUDF_FAIL("Only fixed width types are currently supported");
  }
}

std::unique_ptr<cudf::table> convert_from_rows(
  std::vector<std::unique_ptr<cudf::column>> const &input,
  std::vector<cudf::data_type> const &schema,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource *mr)
{
  CUDF_EXPECTS(input.size() == 1, "Too large of an input, need to concat the output tables...");

  //    for (uint i=0; i<input.size(); ++i) {
  cudf::lists_column_view lcv = input[0]->view();
  auto ret                    = convert_from_rows(lcv, schema, stream, mr);

  return ret;
  //    }
}

std::unique_ptr<cudf::table> convert_from_rows2(
  std::vector<std::unique_ptr<cudf::column>> const &input,
  std::vector<cudf::data_type> const &schema,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource *mr)
{
  CUDF_EXPECTS(input.size() == 1, "Too large of an input, need to concat the output tables...");

  //    for (uint i=0; i<input.size(); ++i) {
  cudf::lists_column_view lcv = input[0]->view();
  auto ret                    = convert_from_rows2(lcv, schema, stream, mr);

  return ret;
  //    }
}

}  // namespace cudf
