/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

#include <cooperative_groups.h>
#include <cuda/functional>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/sequence.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/lists/lists_column_device_view.cuh>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <thrust/binary_search.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scan.h>
#include <type_traits>

#include "row_conversion.hpp"

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
#define ASYNC_MEMCPY_SUPPORTED
#endif

#if !defined(__CUDA_ARCH__) || defined(ASYNC_MEMCPY_SUPPORTED)
#include <cuda/barrier>
#endif // #if !defined(__CUDA_ARCH__) || defined(ASYNC_MEMCPY_SUPPORTED)

#include <algorithm>
#include <cstdarg>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <limits>
#include <optional>
#include <tuple>

namespace {

constexpr auto JCUDF_ROW_ALIGNMENT = 8;

constexpr auto MAX_BATCH_SIZE = std::numeric_limits<cudf::size_type>::max();

// Number of rows each block processes in the two kernels. Tuned via nsight
constexpr auto NUM_STRING_ROWS_PER_BLOCK_TO_ROWS = 1024;
constexpr auto NUM_STRING_ROWS_PER_BLOCK_FROM_ROWS = 64;
constexpr auto MIN_STRING_BLOCKS = 32;
constexpr auto MAX_STRING_BLOCKS = MAX_BATCH_SIZE;

constexpr auto NUM_WARPS_IN_BLOCK = 32;

} // anonymous namespace

// needed to suppress warning about cuda::barrier
#pragma nv_diag_suppress static_var_with_dynamic_init

using namespace cudf;
using detail::make_device_uvector_async;
using rmm::device_uvector;

#ifdef ASYNC_MEMCPY_SUPPORTED
using cuda::aligned_size_t;
#else
template <std::size_t> using aligned_size_t = size_t; // Local stub for cuda::aligned_size_t.
#endif // ASYNC_MEMCPY_SUPPORTED

namespace cudf {
namespace jni {
namespace detail {

/*
 * This module converts data from row-major to column-major and from column-major to row-major. It
 * is a transpose of the data of sorts, but there are a few complicating factors. They are spelled
 * out below:
 *
 * Row Batches:
 * The row data has to fit inside a cuDF column, which limits it to 2 gigs currently. The calling
 * code attempts to keep the data size under 2 gigs, but due to padding this isn't always the case,
 * so being able to break this up into multiple columns is necessary. Internally, this is referred
 * to as the row batch, which is a group of rows that will fit into this 2 gig space requirement.
 * There are typically 1 of these batches, but there can be 2.
 *
 * Async Memcpy:
 * The CUDA blocks are using memcpy_async, which allows for the device to schedule memcpy operations
 * and then wait on them to complete at a later time with a barrier. On Ampere or later hardware
 * there is dedicated hardware to do this copy and on pre-Ampere it should generate the same code
 * that a hand-rolled loop would generate, so performance should be the same or better than a
 * hand-rolled kernel.
 *
 * Tile Info:
 * Each CUDA block will work on a single tile info before exiting. This single tile consumes all
 * available shared memory. The kernel reads data into shared memory and then back out from shared
 * memory to device memory via memcpy_async. This kernel is completely memory bound.
 *
 * Batch Data:
 * This structure contains all the row batches and some book-keeping data necessary for the batches
 * such as row numbers for the batches.
 *
 * Tiles:
 * The tile info describes a tile of data to process. In a GPU with 48KB this equates to about 221
 * bytes in each direction of a table. The tiles are kept as square as possible to attempt to
 * coalesce memory operations. The taller a tile is the better coalescing of columns, but row
 * coalescing suffers. The wider a tile is the better the row coalescing, but columns coalescing
 * suffers. The code attempts to produce a square tile to balance the coalescing. It starts by
 * figuring out the optimal byte length and then adding columns to the data until the tile is too
 * large. Since rows are different width with different alignment requirements, this isn't typically
 * exact. Once a width is found the tiles are generated vertically with that width and height and
 * then the process repeats. This means all the tiles will be the same height, but will have
 * different widths based on what columns they encompass. Tiles in a vertical row will all have the
 * same dimensions.
 *
 *   --------------------------------
 *   | 4   5.0f || True   8   3   1 |
 *   | 3   6.0f || False  3   1   1 |
 *   | 2   7.0f || True   7   4   1 |
 *   | 1   8.0f || False  2   5   1 |
 *   --------------------------------
 *   | 0   9.0f || True   6   7   1 |
 *   ...
 */

/**
 * @brief The CUDA blocks work on one tile_info struct of data.
 *        This structure defines the workspaces for the blocks.
 *
 */
struct tile_info {
  int start_col;
  int start_row;
  int end_col;
  int end_row;
  int batch_number;

  __device__ inline size_type get_shared_row_size(size_type const *const col_offsets,
                                                  size_type const *const col_sizes) const {
    // this calculation is invalid if there are holes in the data such as a variable-width column.
    // It is wrong in a safe way in that it will say this row size is larger than it should be, so
    // we are not losing data we are just not as efficient as we could be with shared memory. This
    // may be a problem if the tile is computed without regard to variable width offset/length sizes
    // in that we overrun shared memory.
    return util::round_up_unsafe(col_offsets[end_col] + col_sizes[end_col] - col_offsets[start_col],
                                 JCUDF_ROW_ALIGNMENT);
  }

  __device__ inline size_type num_cols() const { return end_col - start_col + 1; }

  __device__ inline size_type num_rows() const { return end_row - start_row + 1; }
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
  device_uvector<size_type> batch_row_offsets;      // offsets to each row in incoming data
  device_uvector<size_type> d_batch_row_boundaries; // row numbers for the start of each batch
  std::vector<size_type>
      batch_row_boundaries;           // row numbers for the start of each batch: 0, 1500, 2700
  std::vector<row_batch> row_batches; // information about each batch such as byte count
};

/**
 * @brief builds row size information for tables that contain strings
 *
 * @param tbl table from which to compute row size information
 * @param fixed_width_and_validity_size size of fixed-width and validity data in this table
 * @param stream cuda stream on which to operate
 * @return pair of device vector of size_types of the row sizes of the table and a device vector of
 * offsets into the string column
 */
std::pair<rmm::device_uvector<size_type>, rmm::device_uvector<strings_column_view::offset_iterator>>
build_string_row_offsets(table_view const &tbl, size_type fixed_width_and_validity_size,
                         rmm::cuda_stream_view stream) {
  auto const num_rows = tbl.num_rows();
  rmm::device_uvector<size_type> d_row_sizes(num_rows, stream);
  thrust::uninitialized_fill(rmm::exec_policy(stream), d_row_sizes.begin(), d_row_sizes.end(), 0);

  auto d_offsets_iterators = [&]() {
    std::vector<strings_column_view::offset_iterator> offsets_iterators;
    auto offsets_iter = thrust::make_transform_iterator(
        tbl.begin(), [](auto const &col) -> strings_column_view::offset_iterator {
          if (!is_fixed_width(col.type())) {
            CUDF_EXPECTS(col.type().id() == type_id::STRING, "only string columns are supported!");
            return strings_column_view(col).offsets_begin();
          } else {
            return nullptr;
          }
        });
    std::copy_if(offsets_iter, offsets_iter + tbl.num_columns(),
                 std::back_inserter(offsets_iterators),
                 [](auto const &offset_ptr) { return offset_ptr != nullptr; });
    return make_device_uvector_async(offsets_iterators, stream,
                                     rmm::mr::get_current_device_resource());
  }();

  auto const num_columns = static_cast<size_type>(d_offsets_iterators.size());

  thrust::for_each(rmm::exec_policy(stream), thrust::make_counting_iterator(0),
                   thrust::make_counting_iterator(num_columns * num_rows),
                   [d_offsets_iterators = d_offsets_iterators.data(), num_columns, num_rows,
                    d_row_sizes = d_row_sizes.data()] __device__(auto element_idx) {
                     auto const row = element_idx % num_rows;
                     auto const col = element_idx / num_rows;
                     auto const val =
                         d_offsets_iterators[col][row + 1] - d_offsets_iterators[col][row];
                     atomicAdd(&d_row_sizes[row], val);
                   });

  // transform the row sizes to include fixed width size and alignment
  thrust::transform(
      rmm::exec_policy(stream), d_row_sizes.begin(), d_row_sizes.end(), d_row_sizes.begin(),
      cuda::proclaim_return_type<size_type>([fixed_width_and_validity_size] __device__(
                                                auto row_size) {
        return util::round_up_unsafe(fixed_width_and_validity_size + row_size, JCUDF_ROW_ALIGNMENT);
      }));

  return {std::move(d_row_sizes), std::move(d_offsets_iterators)};
}

/**
 * @brief functor to return the offset of a row in a table with string columns
 *
 */
struct string_row_offset_functor {
  string_row_offset_functor(device_span<size_type const> d_row_offsets)
      : d_row_offsets(d_row_offsets){};

  __device__ inline size_type operator()(int row_number, int) const {
    return d_row_offsets[row_number];
  }

  device_span<size_type const> d_row_offsets;
};

/**
 * @brief functor to return the offset of a row in a table with only fixed-width columns
 *
 */
struct fixed_width_row_offset_functor {
  fixed_width_row_offset_functor(size_type fixed_width_only_row_size)
      : _fixed_width_only_row_size(fixed_width_only_row_size){};

  __device__ inline size_type operator()(int row_number, int tile_row_start) const {
    return (row_number - tile_row_start) * _fixed_width_only_row_size;
  }

  size_type _fixed_width_only_row_size;
};

/**
 * @brief Copies data from row-based JCUDF format to column-based cudf format.
 *
 * This optimized version of the conversion is faster for fixed-width tables that do not have more
 * than 100 columns.
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
    uint32_t active_mask = __ballot_sync(0xffff'ffffu, row_index < num_rows);
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

#ifdef ASYNC_MEMCPY_SUPPORTED
#define MEMCPY(dst, src, size, barrier) cuda::memcpy_async(dst, src, size, barrier)
#else
#define MEMCPY(dst, src, size, barrier) memcpy(dst, src, size)
#endif // ASYNC_MEMCPY_SUPPORTED

/**
 * @brief copy data from cudf columns into JCUDF format, which is row-based
 *
 * @tparam RowOffsetFunctor iterator that gives the size of a specific row of the table.
 * @param num_rows total number of rows in the table
 * @param num_columns total number of columns in the table
 * @param shmem_used_per_tile shared memory amount each `tile_info` is using
 * @param tile_infos span of `tile_info` structs the define the work
 * @param input_data pointer to raw table data
 * @param col_sizes array of sizes for each element in a column - one per column
 * @param col_offsets offset into input data row for each column's start
 * @param row_offsets offset to a specific row in the output data
 * @param batch_row_boundaries row numbers for batch starts
 * @param output_data pointer to output data
 *
 */
template <typename RowOffsetFunctor>
__global__ void copy_to_rows(const size_type num_rows, const size_type num_columns,
                             const size_type shmem_used_per_tile,
                             device_span<const tile_info> tile_infos, const int8_t **input_data,
                             const size_type *col_sizes, const size_type *col_offsets,
                             RowOffsetFunctor row_offsets, size_type const *batch_row_boundaries,
                             int8_t **output_data) {
  // We are going to copy the data in two passes.
  // The first pass copies a chunk of data into shared memory.
  // The second pass copies that chunk from shared memory out to the final location.

  // Because shared memory is limited we copy a subset of the rows at a time.
  // This has been broken up for us in the tile_info struct, so we don't have
  // any calculation to do here, but it is important to note.

  auto const group = cooperative_groups::this_thread_block();
  auto const warp = cooperative_groups::tiled_partition<cudf::detail::warp_size>(group);
  extern __shared__ int8_t shared_data[];

#ifdef ASYNC_MEMCPY_SUPPORTED
  __shared__ cuda::barrier<cuda::thread_scope_block> tile_barrier;
  if (group.thread_rank() == 0) {
    init(&tile_barrier, group.size());
  }
  group.sync();
#endif // ASYNC_MEMCPY_SUPPORTED

  auto const tile = tile_infos[blockIdx.x];
  auto const num_tile_cols = tile.num_cols();
  auto const num_tile_rows = tile.num_rows();
  auto const tile_row_size = tile.get_shared_row_size(col_offsets, col_sizes);
  auto const starting_column_offset = col_offsets[tile.start_col];

  // to do the copy we need to do n column copies followed by m element copies OR we have to do m
  // element copies followed by r row copies. When going from column to row it is much easier to
  // copy by elements first otherwise we would need a running total of the column sizes for our
  // tile, which isn't readily available. This makes it more appealing to copy element-wise from
  // input data into shared matching the end layout and do row-based memcopies out.

  // read each column across the tile
  // each warp takes a column with each thread of a warp taking a row this is done with cooperative
  // groups where each column is chosen by the tiled partition and each thread in that partition
  // works on a row
  for (int relative_col = warp.meta_group_rank(); relative_col < num_tile_cols;
       relative_col += warp.meta_group_size()) {

    auto const absolute_col = relative_col + tile.start_col;
    auto const col_size = col_sizes[absolute_col];
    auto const col_offset = col_offsets[absolute_col];
    auto const relative_col_offset = col_offset - starting_column_offset;
    auto const col_ptr = input_data[absolute_col];

    if (col_ptr == nullptr) {
      // variable-width data column
      continue;
    }

    for (int relative_row = warp.thread_rank(); relative_row < num_tile_rows;
         relative_row += warp.size()) {

      if (relative_row >= num_tile_rows) {
        // out of bounds
        continue;
      }
      auto const absolute_row = relative_row + tile.start_row;

      auto const shared_offset = relative_row * tile_row_size + relative_col_offset;
      auto const input_src = col_ptr + col_size * absolute_row;

      // copy the element from global memory
      switch (col_size) {
        case 2: {
          const int16_t *short_col_input = reinterpret_cast<const int16_t *>(input_src);
          *reinterpret_cast<int16_t *>(&shared_data[shared_offset]) = *short_col_input;
          break;
        }
        case 4: {
          const int32_t *int_col_input = reinterpret_cast<const int32_t *>(input_src);
          *reinterpret_cast<int32_t *>(&shared_data[shared_offset]) = *int_col_input;
          break;
        }
        case 8: {
          const int64_t *long_col_input = reinterpret_cast<const int64_t *>(input_src);
          *reinterpret_cast<int64_t *>(&shared_data[shared_offset]) = *long_col_input;
          break;
        }
        case 1: shared_data[shared_offset] = *input_src; break;
        default: {
          for (int i = 0; i < col_size; ++i) {
            shared_data[shared_offset] = *input_src;
          }
          break;
        }
      }
    }
  }

  auto const tile_output_buffer = output_data[tile.batch_number];
  auto const row_batch_start = tile.batch_number == 0 ? 0 : batch_row_boundaries[tile.batch_number];

  // no async copies above waiting on the barrier, so we sync the group here to ensure all copies to
  // shared memory are completed before copying data out
  group.sync();

  // each warp takes a row
  for (int copy_row = warp.meta_group_rank(); copy_row < tile.num_rows();
       copy_row += warp.meta_group_size()) {
    auto const src = &shared_data[tile_row_size * copy_row];
    auto const dst = tile_output_buffer + row_offsets(copy_row + tile.start_row, row_batch_start) +
                     starting_column_offset;
#ifdef ASYNC_MEMCPY_SUPPORTED
    cuda::memcpy_async(warp, dst, src, tile_row_size, tile_barrier);
#else
    for (int b = warp.thread_rank(); b < tile_row_size; b += warp.size()) {
      dst[b] = src[b];
    }
#endif
  }

#ifdef ASYNC_MEMCPY_SUPPORTED
  // wait on the last copies to complete
  tile_barrier.arrive_and_wait();
#else
  group.sync();
#endif // ASYNC_MEMCPY_SUPPORTED
}

/**
 * @brief copy data from row-based format to cudf columns
 *
 * @tparam RowOffsetFunctor iterator that gives the size of a specific row of the table.
 * @param num_rows total number of rows in the table
 * @param num_columns total number of columns in the table
 * @param shmem_used_per_tile amount of shared memory that is used by a tile
 * @param row_offsets offset to a specific row in the output data
 * @param batch_row_boundaries row numbers for batch starts
 * @param output_data pointer to output data, partitioned by data size
 * @param validity_offsets offset into input data row for validity data
 * @param tile_infos information about the tiles of work
 * @param input_nm pointer to input data
 *
 */
template <typename RowOffsetFunctor>
__global__ void
copy_validity_to_rows(const size_type num_rows, const size_type num_columns,
                      const size_type shmem_used_per_tile, RowOffsetFunctor row_offsets,
                      size_type const *batch_row_boundaries, int8_t **output_data,
                      const size_type validity_offset, device_span<const tile_info> tile_infos,
                      const bitmask_type **input_nm) {
  extern __shared__ int8_t shared_data[];

  // each thread of warp reads a single int32 of validity - so we read 128 bytes then ballot_sync
  // the bits and write the result to shmem after we fill shared mem memcpy it out in a blob.
  auto const group = cooperative_groups::this_thread_block();
  auto const warp = cooperative_groups::tiled_partition<cudf::detail::warp_size>(group);

#ifdef ASYNC_MEMCPY_SUPPORTED
  // Initialize cuda barriers for each tile.
  __shared__ cuda::barrier<cuda::thread_scope_block> shared_tile_barrier;
  if (group.thread_rank() == 0) {
    init(&shared_tile_barrier, group.size());
  }
  group.sync();
#endif // ASYNC_MEMCPY_SUPPORTED

  auto tile = tile_infos[blockIdx.x];
  auto const num_tile_cols = tile.num_cols();
  auto const num_tile_rows = tile.num_rows();

  auto const threads_per_warp = warp.size();
  auto const rows_per_read = cudf::detail::size_in_bits<bitmask_type>();

  auto const num_sections_x = util::div_rounding_up_unsafe(num_tile_cols, threads_per_warp);
  auto const num_sections_y = util::div_rounding_up_unsafe(num_tile_rows, rows_per_read);
  auto const validity_data_row_length = util::round_up_unsafe(
      util::div_rounding_up_unsafe(num_tile_cols, CHAR_BIT), JCUDF_ROW_ALIGNMENT);
  auto const total_sections = num_sections_x * num_sections_y;

  // the tile is divided into sections. A warp operates on a section at a time.
  for (int my_section_idx = warp.meta_group_rank(); my_section_idx < total_sections;
       my_section_idx += warp.meta_group_size()) {
    // convert to rows and cols
    auto const section_x = my_section_idx % num_sections_x;
    auto const section_y = my_section_idx / num_sections_x;
    auto const relative_col = section_x * threads_per_warp + warp.thread_rank();
    auto const relative_row = section_y * rows_per_read;
    auto const absolute_col = relative_col + tile.start_col;
    auto const absolute_row = relative_row + tile.start_row;
    auto const participating = absolute_col < num_columns && absolute_row < num_rows;
    auto const participation_mask = __ballot_sync(0xFFFF'FFFFu, participating);

    if (participating) {
      auto my_data = input_nm[absolute_col] != nullptr ?
                         input_nm[absolute_col][word_index(absolute_row)] :
                         std::numeric_limits<uint32_t>::max();

      // every thread that is participating in the warp has 4 bytes, but it's column-based data and
      // we need it in row-based. So we shuffle the bits around with ballot_sync to make the bytes
      // we actually write.
      bitmask_type dw_mask = 0x1;
      for (int i = 0; i < threads_per_warp && relative_row + i < num_rows; ++i, dw_mask <<= 1) {
        auto validity_data = __ballot_sync(participation_mask, my_data & dw_mask);
        // lead thread in each warp writes data
        auto const validity_write_offset =
            validity_data_row_length * (relative_row + i) + (relative_col / CHAR_BIT);
        if (warp.thread_rank() == 0) {
          *reinterpret_cast<bitmask_type *>(&shared_data[validity_write_offset]) = validity_data;
        }
      }
    }
  }

  auto const output_data_base =
      output_data[tile.batch_number] + validity_offset + tile.start_col / CHAR_BIT;

  // each warp copies a row at a time
  auto const row_bytes = util::div_rounding_up_unsafe(num_tile_cols, CHAR_BIT);
  auto const row_batch_start = tile.batch_number == 0 ? 0 : batch_row_boundaries[tile.batch_number];

  // make sure entire tile has finished copy
  // Note that this was copied from above just under the for loop due to nsight complaints about
  // divergent threads
  group.sync();

  for (int relative_row = warp.meta_group_rank(); relative_row < num_tile_rows;
       relative_row += warp.meta_group_size()) {
    auto const src = &shared_data[validity_data_row_length * relative_row];
    auto const dst = output_data_base + row_offsets(relative_row + tile.start_row, row_batch_start);
#ifdef ASYNC_MEMCPY_SUPPORTED
    cuda::memcpy_async(warp, dst, src, row_bytes, shared_tile_barrier);
#else
    for (int b = warp.thread_rank(); b < row_bytes; b += warp.size()) {
      dst[b] = src[b];
    }
#endif
  }

#ifdef ASYNC_MEMCPY_SUPPORTED
  // wait for tile of data to arrive
  shared_tile_barrier.arrive_and_wait();
#else
  group.sync();
#endif // ASYNC_MEMCPY_SUPPORTED
}

/**
 * @brief kernel to copy string data to JCUDF row format
 *
 * @tparam RowOffsetFunctor iterator for row offsets into the destination data
 * @param num_rows number of rows in this portion of the table
 * @param num_variable_columns number of columns of variable-width data
 * @param variable_input_data variable width data column pointers
 * @param variable_col_output_offsets output offset information for variable-width columns
 * @param variable_col_offsets input offset information for variable-width columns
 * @param fixed_width_row_size offset to variable-width data in a row
 * @param row_offsets offsets for each row in output data
 * @param batch_row_offset row start for this batch
 * @param output_data pointer to output data for this batch
 *
 */
template <typename RowOffsetFunctor>
__global__ void copy_strings_to_rows(size_type const num_rows, size_type const num_variable_columns,
                                     int8_t const **variable_input_data,
                                     size_type const *variable_col_output_offsets,
                                     size_type const **variable_col_offsets,
                                     size_type fixed_width_row_size, RowOffsetFunctor row_offsets,
                                     size_type const batch_row_offset, int8_t *output_data) {
  // Each block will take a group of rows controlled by NUM_STRING_ROWS_PER_BLOCK_TO_ROWS. Each warp
  // will copy a row at a time. The base thread will first go through column data and fill out
  // offset/length information for the column. Then all threads of the warp will participate in the
  // memcpy of the string data.
  auto const my_block = cooperative_groups::this_thread_block();
  auto const warp = cooperative_groups::tiled_partition<cudf::detail::warp_size>(my_block);
#ifdef ASYNC_MEMCPY_SUPPORTED
  cuda::barrier<cuda::thread_scope_block> block_barrier;
#endif

  auto const start_row =
      blockIdx.x * NUM_STRING_ROWS_PER_BLOCK_TO_ROWS + warp.meta_group_rank() + batch_row_offset;
  auto const end_row =
      std::min(num_rows, static_cast<size_type>(start_row + NUM_STRING_ROWS_PER_BLOCK_TO_ROWS));

  for (int row = start_row; row < end_row; row += warp.meta_group_size()) {
    auto offset = fixed_width_row_size; // initial offset to variable-width data
    auto const base_row_offset = row_offsets(row, 0);
    for (int col = 0; col < num_variable_columns; ++col) {
      auto const string_start_offset = variable_col_offsets[col][row];
      auto const string_length = variable_col_offsets[col][row + 1] - string_start_offset;
      if (warp.thread_rank() == 0) {
        // write the offset/length to column
        uint32_t *output_dest = reinterpret_cast<uint32_t *>(
            &output_data[base_row_offset + variable_col_output_offsets[col]]);
        output_dest[0] = offset;
        output_dest[1] = string_length;
      }
      auto string_output_dest = &output_data[base_row_offset + offset];
      auto string_output_src = &variable_input_data[col][string_start_offset];
      warp.sync();
#ifdef ASYNC_MEMCPY_SUPPORTED
      cuda::memcpy_async(warp, string_output_dest, string_output_src, string_length, block_barrier);
#else
      for (int c = warp.thread_rank(); c < string_length; c += warp.size()) {
        string_output_dest[c] = string_output_src[c];
      }
#endif
      offset += string_length;
    }
  }
}
/**
 * @brief copy data from row-based format to cudf columns
 *
 * @tparam RowOffsetFunctor iterator that gives the size of a specific row of the table.
 * @param num_rows total number of rows in the table
 * @param num_columns total number of columns in the table
 * @param shmem_used_per_tile amount of shared memory that is used by a tile
 * @param row_offsets offset to a specific row in the input data
 * @param batch_row_boundaries row numbers for batch starts
 * @param output_data pointers to column data
 * @param col_sizes array of sizes for each element in a column - one per column
 * @param col_offsets offset into input data row for each column's start
 * @param tile_infos information about the tiles of work
 * @param input_data pointer to input data
 *
 */
template <typename RowOffsetFunctor>
__global__ void copy_from_rows(const size_type num_rows, const size_type num_columns,
                               const size_type shmem_used_per_tile, RowOffsetFunctor row_offsets,
                               size_type const *batch_row_boundaries, int8_t **output_data,
                               const size_type *col_sizes, const size_type *col_offsets,
                               device_span<const tile_info> tile_infos, const int8_t *input_data) {
  // We are going to copy the data in two passes.
  // The first pass copies a chunk of data into shared memory.
  // The second pass copies that chunk from shared memory out to the final location.

  // Because shared memory is limited we copy a subset of the rows at a time. This has been broken
  // up for us in the tile_info struct, so we don't have any calculation to do here, but it is
  // important to note.

  // To speed up some of the random access memory we do, we copy col_sizes and col_offsets to shared
  // memory for each of the tiles that we work on

  auto const group = cooperative_groups::this_thread_block();
  auto const warp = cooperative_groups::tiled_partition<cudf::detail::warp_size>(group);
  extern __shared__ int8_t shared[];

#ifdef ASYNC_MEMCPY_SUPPORTED
  // Initialize cuda barriers for each tile.
  __shared__ cuda::barrier<cuda::thread_scope_block> tile_barrier;
  if (group.thread_rank() == 0) {
    init(&tile_barrier, group.size());
  }
  group.sync();
#endif // ASYNC_MEMCPY_SUPPORTED

  {
    auto const fetch_tile = tile_infos[blockIdx.x];
    auto const fetch_tile_start_row = fetch_tile.start_row;
    auto const starting_col_offset = col_offsets[fetch_tile.start_col];
    auto const fetch_tile_row_size = fetch_tile.get_shared_row_size(col_offsets, col_sizes);
    auto const row_batch_start =
        fetch_tile.batch_number == 0 ? 0 : batch_row_boundaries[fetch_tile.batch_number];

    for (int absolute_row = warp.meta_group_rank() + fetch_tile.start_row;
         absolute_row <= fetch_tile.end_row; absolute_row += warp.meta_group_size()) {
      warp.sync();
      auto shared_offset = (absolute_row - fetch_tile_start_row) * fetch_tile_row_size;
      auto dst = &shared[shared_offset];
      auto src = &input_data[row_offsets(absolute_row, row_batch_start) + starting_col_offset];
      // copy the data
#ifdef ASYNC_MEMCPY_SUPPORTED
      cuda::memcpy_async(warp, dst, src, fetch_tile_row_size, tile_barrier);
#else
      for (int b = warp.thread_rank(); b < fetch_tile_row_size; b += warp.size()) {
        dst[b] = src[b];
      }
#endif
    }
  }

  {
    auto const tile = tile_infos[blockIdx.x];
    auto const rows_in_tile = tile.num_rows();
    auto const cols_in_tile = tile.num_cols();
    auto const tile_row_size = tile.get_shared_row_size(col_offsets, col_sizes);

#ifdef ASYNC_MEMCPY_SUPPORTED
    // ensure our data is ready
    tile_barrier.arrive_and_wait();
#else
    group.sync();
#endif // ASYNC_MEMCPY_SUPPORTED

    // Now we copy from shared memory to final destination. The data is laid out in rows in shared
    // memory, so the reads for a column will be "vertical". Because of this and the different sizes
    // for each column, this portion is handled on row/column basis. to prevent each thread working
    // on a single row and also to ensure that all threads can do work in the case of more threads
    // than rows, we do a global index instead of a double for loop with col/row.
    for (int relative_row = warp.thread_rank(); relative_row < rows_in_tile;
         relative_row += warp.size()) {

      auto const absolute_row = relative_row + tile.start_row;
      auto const shared_memory_row_offset = tile_row_size * relative_row;

      for (int relative_col = warp.meta_group_rank(); relative_col < cols_in_tile;
           relative_col += warp.meta_group_size()) {
        auto const absolute_col = relative_col + tile.start_col;

        auto const shared_memory_offset =
            col_offsets[absolute_col] - col_offsets[tile.start_col] + shared_memory_row_offset;
        auto const column_size = col_sizes[absolute_col];

        int8_t *shmem_src = &shared[shared_memory_offset];
        int8_t *dst = &output_data[absolute_col][absolute_row * column_size];

        MEMCPY(dst, shmem_src, column_size, tile_barrier);
      }
    }
  }

#ifdef ASYNC_MEMCPY_SUPPORTED
  // wait on the last copies to complete
  tile_barrier.arrive_and_wait();
#else
  group.sync();
#endif // ASYNC_MEMCPY_SUPPORTED
}

/**
 * @brief copy data from row-based format to cudf columns
 *
 * @tparam RowOffsetFunctor iterator that gives the size of a specific row of the table.
 * @param num_rows total number of rows in the table
 * @param num_columns total number of columns in the table
 * @param shmem_used_per_tile amount of shared memory that is used by a tile
 * @param row_offsets offset to the first column a specific row in the input data
 * @param batch_row_boundaries row numbers for batch starts
 * @param output_nm pointers to null masks for columns
 * @param validity_offsets offset into input data row for validity data
 * @param tile_infos information about the tiles of work
 * @param input_data pointer to input data
 *
 */
template <typename RowOffsetFunctor>
__global__ void
copy_validity_from_rows(const size_type num_rows, const size_type num_columns,
                        const size_type shmem_used_per_tile, RowOffsetFunctor row_offsets,
                        size_type const *batch_row_boundaries, bitmask_type **output_nm,
                        const size_type validity_offset, device_span<const tile_info> tile_infos,
                        const int8_t *input_data) {
  extern __shared__ int8_t shared[];

  using cudf::detail::warp_size;

  // each thread of warp reads a single byte of validity - so we read 32 bytes then ballot_sync the
  // bits and write the result to shmem after we fill shared mem memcpy it out in a blob. Probably
  // need knobs for number of rows vs columns to balance read/write

  //        C0  C1  C2  C3  C4  C5  C6  C7
  //  R0    1   0   1   0   0   1   1   0       <-- thread 0 reads byte r0
  //  R1    1   1   1   1   1   1   1   0       <-- thread 1 reads byte r1
  //  R2    0   0   1   0   0   1   1   0       <-- thread 2 reads byte r2
  //  ...
  //  R31   1   1   1   1   1   1   1   1       <-- thread 31 reads byte r31
  //        ^
  //        |  1 bit of each input byte, by column, are swizzled into a single 32 bit word via
  //        __ballot_sync, representing 32 rows of that column.

  auto const group = cooperative_groups::this_thread_block();
  auto const warp = cooperative_groups::tiled_partition<cudf::detail::warp_size>(group);

#ifdef ASYNC_MEMCPY_SUPPORTED
  // Initialize cuda barriers for each tile.
  __shared__ cuda::barrier<cuda::thread_scope_block> shared_tile_barrier;
  if (group.thread_rank() == 0) {
    init(&shared_tile_barrier, group.size());
  }
  group.sync();
#endif // ASYNC_MEMCPY_SUPPORTED

  auto const tile = tile_infos[blockIdx.x];
  auto const tile_start_col = tile.start_col;
  auto const tile_start_row = tile.start_row;
  auto const num_tile_cols = tile.num_cols();
  auto const num_tile_rows = tile.num_rows();

  auto const threads_per_warp = warp.size();
  auto const cols_per_read = CHAR_BIT;

  auto const rows_per_read = static_cast<size_type>(threads_per_warp);
  auto const num_sections_x = util::div_rounding_up_safe(num_tile_cols, cols_per_read);
  auto const num_sections_y = util::div_rounding_up_safe(num_tile_rows, rows_per_read);
  auto const validity_data_col_length = num_sections_y * 4; // words to bytes
  auto const total_sections = num_sections_x * num_sections_y;

  // the tile is divided into sections. A warp operates on a section at a time.
  for (int my_section_idx = warp.meta_group_rank(); my_section_idx < total_sections;
       my_section_idx += warp.meta_group_size()) {
    // convert section to row and col
    auto const section_x = my_section_idx % num_sections_x;
    auto const section_y = my_section_idx / num_sections_x;
    auto const relative_col = section_x * cols_per_read;
    auto const relative_row = section_y * rows_per_read + warp.thread_rank();
    auto const absolute_col = relative_col + tile_start_col;
    auto const absolute_row = relative_row + tile_start_row;
    auto const row_batch_start =
        tile.batch_number == 0 ? 0 : batch_row_boundaries[tile.batch_number];

    auto const participation_mask = __ballot_sync(0xFFFF'FFFFu, absolute_row < num_rows);

    if (absolute_row < num_rows) {
      auto const my_byte = input_data[row_offsets(absolute_row, row_batch_start) + validity_offset +
                                      (absolute_col / cols_per_read)];

      // so every thread that is participating in the warp has a byte, but it's row-based data and
      // we need it in column-based. So we shuffle the bits around to make the bytes we actually
      // write.
      for (int i = 0, byte_mask = 0x1; (i < cols_per_read) && ((relative_col + i) < num_columns);
           ++i, byte_mask <<= 1) {
        auto const validity_data = __ballot_sync(participation_mask, my_byte & byte_mask);
        // lead thread in each warp writes data
        if (warp.thread_rank() == 0) {
          auto const validity_write_offset =
              validity_data_col_length * (relative_col + i) + relative_row / cols_per_read;
          *reinterpret_cast<bitmask_type *>(&shared[validity_write_offset]) = validity_data;
        }
      }
    }
  }

  // now memcpy the shared memory out to the final destination
  auto const col_words = util::div_rounding_up_unsafe(num_tile_rows, CHAR_BIT * 4);

  // make sure entire tile has finished copy
  group.sync();

  for (int relative_col = warp.meta_group_rank(); relative_col < num_tile_cols;
       relative_col += warp.meta_group_size()) {
    auto const absolute_col = relative_col + tile_start_col;
    auto dst = output_nm[absolute_col] + word_index(tile_start_row);
    auto const src =
        reinterpret_cast<bitmask_type *>(&shared[validity_data_col_length * relative_col]);

#ifdef ASYNC_MEMCPY_SUPPORTED
    cuda::memcpy_async(warp, dst, src, aligned_size_t<4>(validity_data_col_length),
                       shared_tile_barrier);
#else
    for (int b = warp.thread_rank(); b < col_words; b += warp.size()) {
      dst[b] = src[b];
    }
#endif
  }

#ifdef ASYNC_MEMCPY_SUPPORTED
  // wait for tile of data to arrive
  shared_tile_barrier.arrive_and_wait();
#else
  group.sync();
#endif // ASYNC_MEMCPY_SUPPORTED
}

/**
 * @brief copies string data from jcudf row format to cudf columns
 *
 * @tparam RowOffsetFunctor iterator for row offsets into the destination data
 * @param row_offsets offsets for each row in input data
 * @param string_row_offsets offset data into jcudf row data for each string
 * @param string_lengths length of each incoming string in each column
 * @param string_column_offsets offset column data for cudf column
 * @param string_col_data output cudf string column data
 * @param row_data jcudf row data
 * @param num_rows number of rows in data
 * @param num_string_columns number of string columns in the table
 */
template <typename RowOffsetFunctor>
__global__ void copy_strings_from_rows(RowOffsetFunctor row_offsets, int32_t **string_row_offsets,
                                       int32_t **string_lengths, size_type **string_column_offsets,
                                       char **string_col_data, int8_t const *row_data,
                                       size_type const num_rows,
                                       size_type const num_string_columns) {
  // Each warp takes a tile, which is a single column and up to ROWS_PER_BLOCK rows. A tile will not
  // wrap around the bottom of the table. The warp will copy the strings for each row in the tile.
  // Traversing in row-major order to coalesce the offsets and size reads.
  auto my_block = cooperative_groups::this_thread_block();
  auto warp = cooperative_groups::tiled_partition<cudf::detail::warp_size>(my_block);
#ifdef ASYNC_MEMCPY_SUPPORTED
  cuda::barrier<cuda::thread_scope_block> block_barrier;
#endif

  // workaround for not being able to take a reference to a constexpr host variable
  auto const ROWS_PER_BLOCK = NUM_STRING_ROWS_PER_BLOCK_FROM_ROWS;
  auto const tiles_per_col = util::div_rounding_up_unsafe(num_rows, ROWS_PER_BLOCK);
  auto const starting_tile = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
  auto const num_tiles = tiles_per_col * num_string_columns;
  auto const tile_stride = warp.meta_group_size() * gridDim.x;
  // Each warp will copy strings in its tile. This is handled by all the threads of a warp passing
  // the same parameters to async_memcpy and all threads in the warp participating in the copy.
  for (auto my_tile = starting_tile; my_tile < num_tiles; my_tile += tile_stride) {
    auto const starting_row = (my_tile % tiles_per_col) * ROWS_PER_BLOCK;
    auto const col = my_tile / tiles_per_col;
    auto const str_len = string_lengths[col];
    auto const str_row_off = string_row_offsets[col];
    auto const str_col_off = string_column_offsets[col];
    auto str_col_data = string_col_data[col];
    for (int row = starting_row; row < starting_row + ROWS_PER_BLOCK && row < num_rows; ++row) {
      auto const src = &row_data[row_offsets(row, 0) + str_row_off[row]];
      auto dst = &str_col_data[str_col_off[row]];

#ifdef ASYNC_MEMCPY_SUPPORTED
      cuda::memcpy_async(warp, dst, src, str_len[row], block_barrier);
#else
      for (int c = warp.thread_rank(); c < str_len[row]; c += warp.size()) {
        dst[c] = src[c];
      }
#endif
    }
  }
}

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
  // but in practice having too many can cause some overhead that I don't totally
  // understand. Playing around with this having as little as 600 blocks appears
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
      cudf::detail::sequence(num_rows + 1, zero, scalar_size_per_row, stream, mr);

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
                           rmm::device_buffer{0, stream, mr}, stream, mr);
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
  int32_t const validity_bytes_needed =
      util::div_rounding_up_safe<int32_t>(schema.size(), CHAR_BIT);
  // validity comes at the end and is byte aligned so we can pack more in.
  at_offset += validity_bytes_needed;
  // Now we need to pad the end so all rows are 64 bit aligned
  return util::round_up_unsafe(at_offset, JCUDF_ROW_ALIGNMENT);
}

/**
 * @brief column sizes and column start offsets for a table
 */
struct column_info_s {
  size_type size_per_row;
  std::vector<size_type> column_starts;
  std::vector<size_type> column_sizes;
  std::vector<size_type> variable_width_column_starts;

  column_info_s &operator=(column_info_s const &other) = delete;
  column_info_s &operator=(column_info_s &&other) = delete;
};

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
column_info_s compute_column_information(iterator begin, iterator end) {
  size_type size_per_row = 0;
  std::vector<size_type> column_starts;
  std::vector<size_type> column_sizes;
  std::vector<size_type> variable_width_column_starts;

  column_starts.reserve(std::distance(begin, end) + 1);
  column_sizes.reserve(std::distance(begin, end));

  for (auto col_type = begin; col_type != end; ++col_type) {
    bool const compound_type = is_compound(*col_type);

    // a list or string column will write a single uint64 of data here for offset/length
    auto const col_size = compound_type ? sizeof(uint32_t) + sizeof(uint32_t) : size_of(*col_type);

    // align size for this type - They are the same for fixed width types and 4 bytes for variable
    // width length/offset combos
    size_type const alignment_needed = compound_type ? __alignof(uint32_t) : col_size;
    size_per_row = util::round_up_unsafe(size_per_row, alignment_needed);
    if (compound_type) {
      variable_width_column_starts.push_back(size_per_row);
    }
    column_starts.push_back(size_per_row);
    column_sizes.push_back(col_size);
    size_per_row += col_size;
  }

  // add validity offset to the end of fixed_width offsets
  auto validity_offset = size_per_row;
  column_starts.push_back(validity_offset);

  // validity is byte-aligned in the JCUDF format
  size_per_row +=
      util::div_rounding_up_safe(static_cast<size_type>(std::distance(begin, end)), CHAR_BIT);

  return {size_per_row, std::move(column_starts), std::move(column_sizes),
          std::move(variable_width_column_starts)};
}

/**
 * @brief Build `tile_info` for the validity data to break up the work.
 *
 * @param num_columns number of columns in the table
 * @param num_rows number of rows in the table
 * @param shmem_limit_per_tile size of shared memory available to a single gpu tile
 * @param row_batches batched row information for multiple output locations
 * @return vector of `tile_info` structs for validity data
 */
std::vector<detail::tile_info>
build_validity_tile_infos(size_type const &num_columns, size_type const &num_rows,
                          size_type const &shmem_limit_per_tile,
                          std::vector<row_batch> const &row_batches) {
  auto const desired_rows_and_columns = static_cast<int>(sqrt(shmem_limit_per_tile));
  auto const column_stride = util::round_up_unsafe(
      [&]() {
        if (desired_rows_and_columns > num_columns) {
          // not many columns, build a single tile for table width and ship it off
          return num_columns;
        } else {
          return util::round_down_safe(desired_rows_and_columns, CHAR_BIT);
        }
      }(),
      JCUDF_ROW_ALIGNMENT);

  // we fit as much as we can given the column stride note that an element in the table takes just 1
  // bit, but a row with a single element still takes 8 bytes!
  auto const bytes_per_row = util::round_up_safe(
      util::div_rounding_up_unsafe(column_stride, CHAR_BIT), JCUDF_ROW_ALIGNMENT);
  auto const row_stride =
      std::min(num_rows, util::round_down_safe(shmem_limit_per_tile / bytes_per_row, 64));
  std::vector<detail::tile_info> validity_tile_infos;
  validity_tile_infos.reserve(num_columns / column_stride * num_rows / row_stride);
  for (int col = 0; col < num_columns; col += column_stride) {
    int current_tile_row_batch = 0;
    int rows_left_in_batch = row_batches[current_tile_row_batch].row_count;
    int row = 0;
    while (row < num_rows) {
      if (rows_left_in_batch == 0) {
        current_tile_row_batch++;
        rows_left_in_batch = row_batches[current_tile_row_batch].row_count;
      }
      int const tile_height = std::min(row_stride, rows_left_in_batch);
      validity_tile_infos.emplace_back(
          detail::tile_info{col, row, std::min(col + column_stride - 1, num_columns - 1),
                            row + tile_height - 1, current_tile_row_batch});
      row += tile_height;
      rows_left_in_batch -= tile_height;
    }
  }

  return validity_tile_infos;
}

/**
 * @brief functor that returns the size of a row or 0 is row is greater than the number of rows in
 * the table
 *
 * @tparam RowSize iterator that returns the size of a specific row
 */
template <typename RowSize> struct row_size_functor {
  row_size_functor(size_type row_end, RowSize row_sizes, size_type last_row_end)
      : _row_end(row_end), _row_sizes(row_sizes), _last_row_end(last_row_end) {}

  __device__ inline uint64_t operator()(int i) const {
    return i >= _row_end ? 0 : _row_sizes[i + _last_row_end];
  }

  size_type _row_end;
  RowSize _row_sizes;
  size_type _last_row_end;
};

/**
 * @brief Builds batches of rows that will fit in the size limit of a column.
 *
 * @tparam RowSize iterator that gives the size of a specific row of the table.
 * @param num_rows Total number of rows in the table
 * @param row_sizes iterator that gives the size of a specific row of the table.
 * @param all_fixed_width bool indicating all data in this table is fixed width
 * @param stream stream to operate on for this work
 * @param mr memory resource used to allocate any returned data
 * @returns vector of size_type's that indicate row numbers for batch boundaries and a
 * device_uvector of row offsets
 */
template <typename RowSize>
batch_data build_batches(size_type num_rows, RowSize row_sizes, bool all_fixed_width,
                         rmm::cuda_stream_view stream, rmm::mr::device_memory_resource *mr) {
  auto const total_size = thrust::reduce(rmm::exec_policy(stream), row_sizes, row_sizes + num_rows);
  auto const num_batches = static_cast<int32_t>(
      util::div_rounding_up_safe(total_size, static_cast<uint64_t>(MAX_BATCH_SIZE)));
  auto const num_offsets = num_batches + 1;
  std::vector<row_batch> row_batches;
  std::vector<size_type> batch_row_boundaries;
  device_uvector<size_type> batch_row_offsets(all_fixed_width ? 0 : num_rows, stream);

  // at most max gpu memory / 2GB iterations.
  batch_row_boundaries.reserve(num_offsets);
  batch_row_boundaries.push_back(0);
  size_type last_row_end = 0;
  device_uvector<uint64_t> cumulative_row_sizes(num_rows, stream);
  thrust::inclusive_scan(rmm::exec_policy(stream), row_sizes, row_sizes + num_rows,
                         cumulative_row_sizes.begin());

  // This needs to be split this into 2 gig batches. Care must be taken to avoid a batch larger than
  // 2 gigs. Imagine a table with 900 meg rows. The batches should occur every 2 rows, but if a
  // lower bound is run at 2 gigs, 4 gigs, 6 gigs. the batches will be 2 rows, 2 rows, 3 rows, which
  // will be invalid. The previous batch size must be taken into account when building a new batch.
  // One way is to pull the batch size back to the host and add it to MAX_BATCH_SIZE for the lower
  // bound search. The other method involves keeping everything on device, but subtracting the
  // previous batch from cumulative_row_sizes based on index. This involves no synchronization
  // between GPU and CPU, but involves more work on the GPU. These further need to be broken on a
  // 32-row boundary to match the fixed_width optimized versions.

  while (last_row_end < num_rows) {
    auto offset_row_sizes = thrust::make_transform_iterator(
        cumulative_row_sizes.begin(),
        cuda::proclaim_return_type<uint64_t>(
            [last_row_end, cumulative_row_sizes = cumulative_row_sizes.data()] __device__(auto i) {
              return i - cumulative_row_sizes[last_row_end];
            }));
    auto search_start = offset_row_sizes + last_row_end;
    auto search_end = offset_row_sizes + num_rows;

    // find the next MAX_BATCH_SIZE boundary
    auto const lb =
        thrust::lower_bound(rmm::exec_policy(stream), search_start, search_end, MAX_BATCH_SIZE);
    size_type const batch_size = lb - search_start;

    size_type const row_end = lb == search_end ?
                                  batch_size + last_row_end :
                                  last_row_end + util::round_down_safe(batch_size, 32);

    // build offset list for each row in this batch
    auto const num_rows_in_batch = row_end - last_row_end;

    // build offset list for each row in this batch
    auto const num_entries = row_end - last_row_end + 1;
    device_uvector<size_type> output_batch_row_offsets(num_entries, stream, mr);

    auto row_size_iter_bounded = cudf::detail::make_counting_transform_iterator(
        0, row_size_functor(row_end, row_sizes, last_row_end));

    thrust::exclusive_scan(rmm::exec_policy(stream), row_size_iter_bounded,
                           row_size_iter_bounded + num_entries, output_batch_row_offsets.begin());

    auto const batch_bytes = output_batch_row_offsets.element(num_rows_in_batch, stream);

    // The output_batch_row_offsets vector is used as the offset column of the returned data. This
    // needs to be individually allocated, but the kernel needs a contiguous array of offsets or
    // more global lookups are necessary.
    if (!all_fixed_width) {
      cudaMemcpy(batch_row_offsets.data() + last_row_end, output_batch_row_offsets.data(),
                 num_rows_in_batch * sizeof(size_type), cudaMemcpyDefault);
    }

    batch_row_boundaries.push_back(row_end);
    row_batches.push_back({batch_bytes, num_rows_in_batch, std::move(output_batch_row_offsets)});

    last_row_end = row_end;
  }

  return {std::move(batch_row_offsets),
          make_device_uvector_async(batch_row_boundaries, stream,
                                    rmm::mr::get_current_device_resource()),
          std::move(batch_row_boundaries), std::move(row_batches)};
}

/**
 * @brief Computes the number of tiles necessary given a tile height and batch offsets
 *
 * @param batch_row_boundaries row boundaries for each batch
 * @param desired_tile_height height of each tile in the table
 * @param stream stream to use
 * @return number of tiles necessary
 */
int compute_tile_counts(device_span<size_type const> const &batch_row_boundaries,
                        int desired_tile_height, rmm::cuda_stream_view stream) {
  size_type const num_batches = batch_row_boundaries.size() - 1;
  device_uvector<size_type> num_tiles(num_batches, stream);
  auto iter = thrust::make_counting_iterator(0);
  thrust::transform(
      rmm::exec_policy(stream), iter, iter + num_batches, num_tiles.begin(),
      cuda::proclaim_return_type<size_type>(
          [desired_tile_height, batch_row_boundaries = batch_row_boundaries.data()] __device__(
              auto batch_index) -> size_type {
            return util::div_rounding_up_unsafe(batch_row_boundaries[batch_index + 1] -
                                                    batch_row_boundaries[batch_index],
                                                desired_tile_height);
          }));
  return thrust::reduce(rmm::exec_policy(stream), num_tiles.begin(), num_tiles.end());
}

/**
 * @brief Builds the `tile_info` structs for a given table.
 *
 * @param tiles span of tiles to populate
 * @param batch_row_boundaries boundary to row batches
 * @param column_start starting column of the tile
 * @param column_end ending column of the tile
 * @param desired_tile_height height of the tile
 * @param total_number_of_rows total number of rows in the table
 * @param stream stream to use
 * @return number of tiles created
 */
size_type
build_tiles(device_span<tile_info> tiles,
            device_uvector<size_type> const &batch_row_boundaries, // comes from build_batches
            int column_start, int column_end, int desired_tile_height, int total_number_of_rows,
            rmm::cuda_stream_view stream) {
  size_type const num_batches = batch_row_boundaries.size() - 1;
  device_uvector<size_type> num_tiles(num_batches, stream);
  auto iter = thrust::make_counting_iterator(0);
  thrust::transform(
      rmm::exec_policy(stream), iter, iter + num_batches, num_tiles.begin(),
      cuda::proclaim_return_type<size_type>(
          [desired_tile_height, batch_row_boundaries = batch_row_boundaries.data()] __device__(
              auto batch_index) -> size_type {
            return util::div_rounding_up_unsafe(batch_row_boundaries[batch_index + 1] -
                                                    batch_row_boundaries[batch_index],
                                                desired_tile_height);
          }));

  size_type const total_tiles =
      thrust::reduce(rmm::exec_policy(stream), num_tiles.begin(), num_tiles.end());

  device_uvector<size_type> tile_starts(num_batches + 1, stream);
  auto tile_iter = cudf::detail::make_counting_transform_iterator(
      0, cuda::proclaim_return_type<size_type>(
             [num_tiles = num_tiles.data(), num_batches] __device__(auto i) {
               return (i < num_batches) ? num_tiles[i] : 0;
             }));
  thrust::exclusive_scan(rmm::exec_policy(stream), tile_iter, tile_iter + num_batches + 1,
                         tile_starts.begin()); // in tiles

  thrust::transform(
      rmm::exec_policy(stream), iter, iter + total_tiles, tiles.begin(),
      cuda::proclaim_return_type<tile_info>(
          [=, tile_starts = tile_starts.data(),
           batch_row_boundaries = batch_row_boundaries.data()] __device__(size_type tile_index) {
            // what batch this tile falls in
            auto const batch_index_iter = thrust::upper_bound(
                thrust::seq, tile_starts, tile_starts + num_batches, tile_index);
            auto const batch_index = std::distance(tile_starts, batch_index_iter) - 1;
            // local index within the tile
            int const local_tile_index = tile_index - tile_starts[batch_index];
            // the start row for this batch.
            int const batch_row_start = batch_row_boundaries[batch_index];
            // the start row for this tile
            int const tile_row_start = batch_row_start + (local_tile_index * desired_tile_height);
            // the end row for this tile
            int const max_row =
                std::min(total_number_of_rows - 1,
                         batch_index + 1 > num_batches ?
                             std::numeric_limits<size_type>::max() :
                             static_cast<int>(batch_row_boundaries[batch_index + 1]) - 1);
            int const tile_row_end = std::min(
                batch_row_start + ((local_tile_index + 1) * desired_tile_height) - 1, max_row);

            // stuff the tile
            return tile_info{column_start, tile_row_start, column_end, tile_row_end,
                             static_cast<int>(batch_index)};
          }));

  return total_tiles;
}

/**
 * @brief Determines what data should be operated on by each tile for the incoming table.
 *
 * @tparam TileCallback Callback that receives the start and end columns of tiles
 * @param column_sizes vector of the size of each column
 * @param column_starts vector of the offset of each column
 * @param first_row_batch_size size of the first row batch to limit max tile size since a tile
 * is unable to span batches
 * @param total_number_of_rows total number of rows in the table
 * @param shmem_limit_per_tile shared memory allowed per tile
 * @param f callback function called when building a tile
 */
template <typename TileCallback>
void determine_tiles(std::vector<size_type> const &column_sizes,
                     std::vector<size_type> const &column_starts,
                     size_type const first_row_batch_size, size_type const total_number_of_rows,
                     size_type const &shmem_limit_per_tile, TileCallback f) {
  // tile infos are organized with the tile going "down" the columns this provides the most
  // coalescing of memory access
  int current_tile_width = 0;
  int current_tile_start_col = 0;

  // the ideal tile height has lots of 8-byte reads and 8-byte writes. The optimal read/write would
  // be memory cache line sized access, but since other tiles will read/write the edges this may not
  // turn out to be overly important. For now, we will attempt to build a square tile as far as byte
  // sizes. x * y = shared_mem_size. Which translates to x^2 = shared_mem_size since we want them
  // equal, so height and width are sqrt(shared_mem_size). The trick is that it's in bytes, not rows
  // or columns.
  auto const square_bias = 32; // bias towards columns for performance reasons
  auto const optimal_square_len = static_cast<size_type>(sqrt(shmem_limit_per_tile));
  auto const desired_tile_height = util::round_up_safe<int>(
      std::min(optimal_square_len / square_bias, total_number_of_rows), cudf::detail::warp_size);
  auto const tile_height = std::clamp(desired_tile_height, 1, first_row_batch_size);

  int row_size = 0;

  // march each column and build the tiles of appropriate sizes
  for (uint col = 0; col < column_sizes.size(); ++col) {
    auto const col_size = column_sizes[col];

    // align size for this type
    auto const alignment_needed = col_size; // They are the same for fixed width types
    auto const row_size_aligned = util::round_up_unsafe(row_size, alignment_needed);
    auto const row_size_with_this_col = row_size_aligned + col_size;
    auto const row_size_with_end_pad =
        util::round_up_unsafe(row_size_with_this_col, JCUDF_ROW_ALIGNMENT);

    if (row_size_with_end_pad * tile_height > shmem_limit_per_tile) {
      // too large, close this tile, generate vertical tiles and restart
      f(current_tile_start_col, col == 0 ? col : col - 1, tile_height);

      row_size =
          util::round_up_unsafe((column_starts[col] + column_sizes[col]) & 7, alignment_needed);
      row_size += col_size; // alignment required for shared memory tile boundary to match alignment
                            // of output row
      current_tile_start_col = col;
      current_tile_width = 0;
    } else {
      row_size = row_size_with_this_col;
      current_tile_width++;
    }
  }

  // build last set of tiles
  if (current_tile_width > 0) {
    f(current_tile_start_col, static_cast<int>(column_sizes.size()) - 1, tile_height);
  }
}

/**
 * @brief convert cudf table into JCUDF row format
 *
 * @tparam offsetFunctor functor type for offset functor
 * @param tbl table to convert to JCUDF row format
 * @param batch_info information about the batches of data
 * @param offset_functor functor that returns the starting offset of each row
 * @param column_info information about incoming columns
 * @param variable_width_offsets optional vector of offsets for variable-with columns
 * @param stream stream used
 * @param mr selected memory resource for returned data
 * @return vector of list columns containing byte columns of the JCUDF row data
 */
template <typename offsetFunctor>
std::vector<std::unique_ptr<column>> convert_to_rows(
    table_view const &tbl, batch_data &batch_info, offsetFunctor offset_functor,
    column_info_s const &column_info,
    std::optional<rmm::device_uvector<strings_column_view::offset_iterator>> variable_width_offsets,
    rmm::cuda_stream_view stream, rmm::mr::device_memory_resource *mr) {
  int device_id;
  CUDF_CUDA_TRY(cudaGetDevice(&device_id));
  int total_shmem_in_bytes;
  CUDF_CUDA_TRY(
      cudaDeviceGetAttribute(&total_shmem_in_bytes, cudaDevAttrMaxSharedMemoryPerBlock, device_id));

#ifndef __CUDA_ARCH__ // __host__ code.
  // Need to reduce total shmem available by the size of barriers in the kernel's shared memory
  total_shmem_in_bytes -=
      util::round_up_unsafe(sizeof(cuda::barrier<cuda::thread_scope_block>), 16ul);
#endif // __CUDA_ARCH__

  auto const shmem_limit_per_tile = total_shmem_in_bytes;

  auto const num_rows = tbl.num_rows();
  auto const fixed_width_only = !variable_width_offsets.has_value();

  auto select_columns = [](auto const &tbl, auto column_predicate) {
    std::vector<column_view> cols;
    std::copy_if(tbl.begin(), tbl.end(), std::back_inserter(cols),
                 [&](auto c) { return column_predicate(c); });
    return table_view(cols);
  };

  auto dev_col_sizes = make_device_uvector_async(column_info.column_sizes, stream,
                                                 rmm::mr::get_current_device_resource());
  auto dev_col_starts = make_device_uvector_async(column_info.column_starts, stream,
                                                  rmm::mr::get_current_device_resource());

  // Get the pointers to the input columnar data ready
  auto const data_begin = thrust::make_transform_iterator(tbl.begin(), [](auto const &c) {
    return is_compound(c.type()) ? nullptr : c.template data<int8_t>();
  });
  std::vector<int8_t const *> input_data(data_begin, data_begin + tbl.num_columns());

  // validity code handles variable and fixed-width data, so give it everything
  auto const nm_begin =
      thrust::make_transform_iterator(tbl.begin(), [](auto const &c) { return c.null_mask(); });
  std::vector<bitmask_type const *> input_nm(nm_begin, nm_begin + tbl.num_columns());

  auto dev_input_data =
      make_device_uvector_async(input_data, stream, rmm::mr::get_current_device_resource());
  auto dev_input_nm =
      make_device_uvector_async(input_nm, stream, rmm::mr::get_current_device_resource());

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
  detail::determine_tiles(
      column_info.column_sizes, column_info.column_starts, first_batch_size, num_rows,
      shmem_limit_per_tile,
      [&gpu_batch_row_boundaries = batch_info.d_batch_row_boundaries, &info_count,
       &stream](int const start_col, int const end_col, int const tile_height) {
        int i = detail::compute_tile_counts(gpu_batch_row_boundaries, tile_height, stream);
        info_count += i;
      });

  // allocate space for tiles
  device_uvector<detail::tile_info> gpu_tile_infos(info_count, stream);
  int tile_offset = 0;

  detail::determine_tiles(
      column_info.column_sizes, column_info.column_starts, first_batch_size, num_rows,
      shmem_limit_per_tile,
      [&gpu_batch_row_boundaries = batch_info.d_batch_row_boundaries, &gpu_tile_infos, num_rows,
       &tile_offset, stream](int const start_col, int const end_col, int const tile_height) {
        tile_offset += detail::build_tiles(
            {gpu_tile_infos.data() + tile_offset, gpu_tile_infos.size() - tile_offset},
            gpu_batch_row_boundaries, start_col, end_col, tile_height, num_rows, stream);
      });

  // build validity tiles for ALL columns, variable and fixed width.
  auto validity_tile_infos = detail::build_validity_tile_infos(
      tbl.num_columns(), num_rows, shmem_limit_per_tile, batch_info.row_batches);

  auto dev_validity_tile_infos = make_device_uvector_async(validity_tile_infos, stream,
                                                           rmm::mr::get_current_device_resource());

  auto const validity_offset = column_info.column_starts.back();

  // blast through the entire table and convert it
  detail::copy_to_rows<<<gpu_tile_infos.size(), NUM_WARPS_IN_BLOCK * cudf::detail::warp_size,
                         total_shmem_in_bytes, stream.value()>>>(
      num_rows, tbl.num_columns(), shmem_limit_per_tile, gpu_tile_infos, dev_input_data.data(),
      dev_col_sizes.data(), dev_col_starts.data(), offset_functor,
      batch_info.d_batch_row_boundaries.data(),
      reinterpret_cast<int8_t **>(dev_output_data.data()));

  // note that validity gets the entire table and not the fixed-width portion
  detail::copy_validity_to_rows<<<validity_tile_infos.size(),
                                  NUM_WARPS_IN_BLOCK * cudf::detail::warp_size,
                                  total_shmem_in_bytes, stream.value()>>>(
      num_rows, tbl.num_columns(), shmem_limit_per_tile, offset_functor,
      batch_info.d_batch_row_boundaries.data(), dev_output_data.data(), validity_offset,
      dev_validity_tile_infos, dev_input_nm.data());

  if (!fixed_width_only) {
    // build table view for variable-width data only
    auto const variable_width_table =
        select_columns(tbl, [](auto col) { return is_compound(col.type()); });

    CUDF_EXPECTS(!variable_width_table.is_empty(), "No variable-width columns when expected!");
    CUDF_EXPECTS(variable_width_offsets.has_value(), "No variable width offset data!");

    auto const variable_data_begin =
        thrust::make_transform_iterator(variable_width_table.begin(), [](auto const &c) {
          strings_column_view const scv{c};
          return is_compound(c.type()) ? scv.chars().template data<int8_t>() : nullptr;
        });
    std::vector<int8_t const *> variable_width_input_data(
        variable_data_begin, variable_data_begin + variable_width_table.num_columns());

    auto dev_variable_input_data = make_device_uvector_async(
        variable_width_input_data, stream, rmm::mr::get_current_device_resource());
    auto dev_variable_col_output_offsets = make_device_uvector_async(
        column_info.variable_width_column_starts, stream, rmm::mr::get_current_device_resource());

    for (uint i = 0; i < batch_info.row_batches.size(); i++) {
      auto const batch_row_offset = batch_info.batch_row_boundaries[i];
      auto const batch_num_rows = batch_info.row_batches[i].row_count;

      dim3 const string_blocks(std::min(
          MAX_STRING_BLOCKS,
          util::div_rounding_up_unsafe(batch_num_rows, NUM_STRING_ROWS_PER_BLOCK_TO_ROWS)));

      detail::copy_strings_to_rows<<<string_blocks, NUM_WARPS_IN_BLOCK * cudf::detail::warp_size, 0,
                                     stream.value()>>>(
          batch_num_rows, variable_width_table.num_columns(), dev_variable_input_data.data(),
          dev_variable_col_output_offsets.data(), variable_width_offsets->data(),
          column_info.size_per_row, offset_functor, batch_row_offset,
          reinterpret_cast<int8_t *>(output_data[i]));
    }
  }

  // split up the output buffer into multiple buffers based on row batch sizes and create list of
  // byte columns
  std::vector<std::unique_ptr<column>> ret;
  ret.reserve(batch_info.row_batches.size());
  auto counting_iter = thrust::make_counting_iterator(0);
  std::transform(counting_iter, counting_iter + batch_info.row_batches.size(),
                 std::back_inserter(ret), [&](auto batch) {
                   auto const offset_count = batch_info.row_batches[batch].row_offsets.size();
                   auto offsets =
                       std::make_unique<column>(data_type{type_id::INT32}, (size_type)offset_count,
                                                batch_info.row_batches[batch].row_offsets.release(),
                                                rmm::device_buffer{}, 0);
                   auto data = std::make_unique<column>(
                       data_type{type_id::INT8}, batch_info.row_batches[batch].num_bytes,
                       std::move(output_buffers[batch]), rmm::device_buffer{}, 0);

                   return make_lists_column(
                       batch_info.row_batches[batch].row_count, std::move(offsets), std::move(data),
                       0, rmm::device_buffer{0, cudf::get_default_stream(), mr}, stream, mr);
                 });

  return ret;
}

} // namespace detail

/**
 * @brief convert a cudf table to JCUDF row format
 *
 * @param tbl incoming table to convert
 * @param stream stream to use for operations
 * @param mr memory resource used for returned data
 * @return vector of list columns containing byte columns of the JCUDF row data
 */
std::vector<std::unique_ptr<column>> convert_to_rows(table_view const &tbl,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::mr::device_memory_resource *mr) {
  auto const num_columns = tbl.num_columns();
  auto const num_rows = tbl.num_rows();

  auto const fixed_width_only = std::all_of(
      tbl.begin(), tbl.end(), [](column_view const &c) { return is_fixed_width(c.type()); });

  // Break up the work into tiles, which are a starting and ending row/col #. This tile size is
  // calculated based on the shared memory size available we want a single tile to fill up the
  // entire shared memory space available for the transpose-like conversion.

  // There are two different processes going on here. The GPU conversion of the data and the writing
  // of the data into the list of byte columns that are a maximum of 2 gigs each due to offset
  // maximum size. The GPU conversion portion has to understand this limitation because the column
  // must own the data inside and as a result it must be a distinct allocation for that column.
  // Copying the data into these final buffers would be prohibitively expensive, so care is taken to
  // ensure the GPU writes to the proper buffer. The tiles are broken at the boundaries of specific
  // rows based on the row sizes up to that point. These are row batches and they are decided first
  // before building the tiles so the tiles can be properly cut around them.

  auto schema_column_iter =
      thrust::make_transform_iterator(tbl.begin(), [](auto const &i) { return i.type(); });

  auto column_info =
      detail::compute_column_information(schema_column_iter, schema_column_iter + num_columns);
  auto const size_per_row = column_info.size_per_row;
  if (fixed_width_only) {
    // total encoded row size. This includes fixed-width data and validity only. It does not include
    // variable-width data since it isn't copied with the fixed-width and validity kernel.
    auto row_size_iter = thrust::make_constant_iterator<uint64_t>(
        util::round_up_unsafe(size_per_row, JCUDF_ROW_ALIGNMENT));

    auto batch_info = detail::build_batches(num_rows, row_size_iter, fixed_width_only, stream, mr);

    detail::fixed_width_row_offset_functor offset_functor(
        util::round_up_unsafe(size_per_row, JCUDF_ROW_ALIGNMENT));

    return detail::convert_to_rows(tbl, batch_info, offset_functor, std::move(column_info),
                                   std::nullopt, stream, mr);
  } else {
    auto offset_data = detail::build_string_row_offsets(tbl, size_per_row, stream);
    auto &row_sizes = std::get<0>(offset_data);

    auto row_size_iter = cudf::detail::make_counting_transform_iterator(
        0, detail::row_size_functor(num_rows, row_sizes.data(), 0));

    auto batch_info = detail::build_batches(num_rows, row_size_iter, fixed_width_only, stream, mr);

    detail::string_row_offset_functor offset_functor(batch_info.batch_row_offsets);

    return detail::convert_to_rows(tbl, batch_info, offset_functor, std::move(column_info),
                                   std::make_optional(std::move(std::get<1>(offset_data))), stream,
                                   mr);
  }
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

    // Make the number of rows per batch a multiple of 32 so we don't have to worry about splitting
    // validity at a specific row offset.  This might change in the future.
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

/**
 * @brief convert from JCUDF row format to cudf columns
 *
 * @param input vector of list columns containing byte columns of the JCUDF row data
 * @param schema incoming schema of the data
 * @param stream stream to use for compute
 * @param mr memory resource for returned data
 * @return cudf table of the data
 */
std::unique_ptr<table> convert_from_rows(lists_column_view const &input,
                                         std::vector<data_type> const &schema,
                                         rmm::cuda_stream_view stream,
                                         rmm::mr::device_memory_resource *mr) {
  // verify that the types are what we expect
  column_view child = input.child();
  auto const list_type = child.type().id();
  CUDF_EXPECTS(list_type == type_id::INT8 || list_type == type_id::UINT8,
               "Only a list of bytes is supported as input");

  // convert any strings in the schema to two int32 columns
  // This allows us to leverage the fixed-width copy code to fill in our offset and string length
  // data.
  std::vector<data_type> string_schema;
  string_schema.reserve(schema.size());
  for (auto i : schema) {
    if (i.id() == type_id::STRING) {
      string_schema.push_back(data_type(type_id::INT32));
      string_schema.push_back(data_type(type_id::INT32));
    } else {
      string_schema.push_back(i);
    }
  }

  auto const num_columns = string_schema.size();
  auto const num_rows = input.parent().size();

  int device_id;
  CUDF_CUDA_TRY(cudaGetDevice(&device_id));
  int total_shmem_in_bytes;
  CUDF_CUDA_TRY(
      cudaDeviceGetAttribute(&total_shmem_in_bytes, cudaDevAttrMaxSharedMemoryPerBlock, device_id));

#ifndef __CUDA_ARCH__ // __host__ code.
  // Need to reduce total shmem available by the size of barriers in the kernel's shared memory
  total_shmem_in_bytes -=
      util::round_up_unsafe(sizeof(cuda::barrier<cuda::thread_scope_block>), 16ul);
#endif // __CUDA_ARCH__

  auto const shmem_limit_per_tile = total_shmem_in_bytes;

  auto column_info = detail::compute_column_information(string_schema.begin(), string_schema.end());
  auto const size_per_row = util::round_up_unsafe(column_info.size_per_row, JCUDF_ROW_ALIGNMENT);

  // Ideally we would check that the offsets are all the same, etc. but for now this is probably
  // fine
  CUDF_EXPECTS(size_per_row * num_rows <= child.size(), "The layout of the data appears to be off");
  auto dev_col_starts = make_device_uvector_async(column_info.column_starts, stream,
                                                  rmm::mr::get_current_device_resource());
  auto dev_col_sizes = make_device_uvector_async(column_info.column_sizes, stream,
                                                 rmm::mr::get_current_device_resource());

  // Allocate the columns we are going to write into
  std::vector<std::unique_ptr<column>> output_columns;
  std::vector<std::unique_ptr<column>> string_row_offset_columns;
  std::vector<std::unique_ptr<column>> string_length_columns;
  std::vector<int8_t *> output_data;
  std::vector<bitmask_type *> output_nm;
  std::vector<int32_t *> string_row_offsets;
  std::vector<int32_t *> string_lengths;
  for (auto i : schema) {
    auto make_col = [&output_data, &output_nm](data_type type, size_type num_rows, bool include_nm,
                                               rmm::cuda_stream_view stream,
                                               rmm::mr::device_memory_resource *mr) {
      auto column = make_fixed_width_column(
          type, num_rows, include_nm ? mask_state::UNINITIALIZED : mask_state::UNALLOCATED, stream,
          mr);
      auto mut = column->mutable_view();
      output_data.emplace_back(mut.data<int8_t>());
      if (include_nm) {
        output_nm.emplace_back(mut.null_mask());
      }
      return column;
    };
    if (i.id() == type_id::STRING) {
      auto const int32type = data_type(type_id::INT32);
      auto offset_col =
          make_col(int32type, num_rows, true, stream, rmm::mr::get_current_device_resource());
      string_row_offsets.push_back(offset_col->mutable_view().data<int32_t>());
      string_row_offset_columns.emplace_back(std::move(offset_col));
      auto length_col =
          make_col(int32type, num_rows, false, stream, rmm::mr::get_current_device_resource());
      string_lengths.push_back(length_col->mutable_view().data<int32_t>());
      string_length_columns.emplace_back(std::move(length_col));
      // placeholder
      output_columns.emplace_back(make_empty_column(type_id::STRING));
    } else {
      output_columns.emplace_back(make_col(i, num_rows, true, stream, mr));
    }
  }

  auto dev_string_row_offsets =
      make_device_uvector_async(string_row_offsets, stream, rmm::mr::get_current_device_resource());
  auto dev_string_lengths =
      make_device_uvector_async(string_lengths, stream, rmm::mr::get_current_device_resource());

  // build the row_batches from the passed in list column
  std::vector<detail::row_batch> row_batches;
  row_batches.push_back(
      {detail::row_batch{child.size(), num_rows, device_uvector<size_type>(0, stream)}});

  auto dev_output_data =
      make_device_uvector_async(output_data, stream, rmm::mr::get_current_device_resource());
  auto dev_output_nm =
      make_device_uvector_async(output_nm, stream, rmm::mr::get_current_device_resource());

  // only ever get a single batch when going from rows, so boundaries are 0, num_rows
  constexpr auto num_batches = 2;
  device_uvector<size_type> gpu_batch_row_boundaries(num_batches, stream);

  thrust::transform(rmm::exec_policy(stream), thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(num_batches), gpu_batch_row_boundaries.begin(),
                    cuda::proclaim_return_type<size_type>(
                        [num_rows] __device__(auto i) { return i == 0 ? 0 : num_rows; }));

  int info_count = 0;
  detail::determine_tiles(
      column_info.column_sizes, column_info.column_starts, num_rows, num_rows, shmem_limit_per_tile,
      [&gpu_batch_row_boundaries, &info_count, &stream](int const start_col, int const end_col,
                                                        int const tile_height) {
        info_count += detail::compute_tile_counts(gpu_batch_row_boundaries, tile_height, stream);
      });

  // allocate space for tiles
  device_uvector<detail::tile_info> gpu_tile_infos(info_count, stream);

  int tile_offset = 0;
  detail::determine_tiles(
      column_info.column_sizes, column_info.column_starts, num_rows, num_rows, shmem_limit_per_tile,
      [&gpu_batch_row_boundaries, &gpu_tile_infos, num_rows, &tile_offset,
       stream](int const start_col, int const end_col, int const tile_height) {
        tile_offset += detail::build_tiles(
            {gpu_tile_infos.data() + tile_offset, gpu_tile_infos.size() - tile_offset},
            gpu_batch_row_boundaries, start_col, end_col, tile_height, num_rows, stream);
      });

  dim3 const blocks(gpu_tile_infos.size());

  // validity needs to be calculated based on the actual number of final table columns
  auto validity_tile_infos =
      detail::build_validity_tile_infos(schema.size(), num_rows, shmem_limit_per_tile, row_batches);

  auto dev_validity_tile_infos = make_device_uvector_async(validity_tile_infos, stream,
                                                           rmm::mr::get_current_device_resource());

  dim3 const validity_blocks(validity_tile_infos.size());

  if (dev_string_row_offsets.size() == 0) {
    detail::fixed_width_row_offset_functor offset_functor(size_per_row);

    detail::copy_from_rows<<<gpu_tile_infos.size(), NUM_WARPS_IN_BLOCK * cudf::detail::warp_size,
                             total_shmem_in_bytes, stream.value()>>>(
        num_rows, num_columns, shmem_limit_per_tile, offset_functor,
        gpu_batch_row_boundaries.data(), dev_output_data.data(), dev_col_sizes.data(),
        dev_col_starts.data(), gpu_tile_infos, child.data<int8_t>());

    detail::copy_validity_from_rows<<<validity_tile_infos.size(),
                                      NUM_WARPS_IN_BLOCK * cudf::detail::warp_size,
                                      total_shmem_in_bytes, stream.value()>>>(
        num_rows, num_columns, shmem_limit_per_tile, offset_functor,
        gpu_batch_row_boundaries.data(), dev_output_nm.data(), column_info.column_starts.back(),
        dev_validity_tile_infos, child.data<int8_t>());

  } else {
    detail::string_row_offset_functor offset_functor(device_span<size_type const>{input.offsets()});
    detail::copy_from_rows<<<gpu_tile_infos.size(), NUM_WARPS_IN_BLOCK * cudf::detail::warp_size,
                             total_shmem_in_bytes, stream.value()>>>(
        num_rows, num_columns, shmem_limit_per_tile, offset_functor,
        gpu_batch_row_boundaries.data(), dev_output_data.data(), dev_col_sizes.data(),
        dev_col_starts.data(), gpu_tile_infos, child.data<int8_t>());

    detail::copy_validity_from_rows<<<validity_tile_infos.size(),
                                      NUM_WARPS_IN_BLOCK * cudf::detail::warp_size,
                                      total_shmem_in_bytes, stream.value()>>>(
        num_rows, num_columns, shmem_limit_per_tile, offset_functor,
        gpu_batch_row_boundaries.data(), dev_output_nm.data(), column_info.column_starts.back(),
        dev_validity_tile_infos, child.data<int8_t>());

    std::vector<device_uvector<size_type>> string_col_offsets;
    std::vector<rmm::device_uvector<char>> string_data_cols;
    std::vector<size_type *> string_col_offset_ptrs;
    std::vector<char *> string_data_col_ptrs;
    for (auto &col_string_lengths : string_lengths) {
      device_uvector<size_type> output_string_offsets(num_rows + 1, stream, mr);
      auto tmp = cuda::proclaim_return_type<size_type>(
          [num_rows, col_string_lengths] __device__(auto const &i) {
            return i < num_rows ? col_string_lengths[i] : 0;
          });
      auto bounded_iter = cudf::detail::make_counting_transform_iterator(0, tmp);
      thrust::exclusive_scan(rmm::exec_policy(stream), bounded_iter, bounded_iter + num_rows + 1,
                             output_string_offsets.begin());

      // allocate destination string column
      rmm::device_uvector<char> string_data(output_string_offsets.element(num_rows, stream), stream,
                                            mr);

      string_col_offset_ptrs.push_back(output_string_offsets.data());
      string_data_col_ptrs.push_back(string_data.data());
      string_col_offsets.push_back(std::move(output_string_offsets));
      string_data_cols.push_back(std::move(string_data));
    }
    auto dev_string_col_offsets = make_device_uvector_async(string_col_offset_ptrs, stream,
                                                            rmm::mr::get_current_device_resource());
    auto dev_string_data_cols = make_device_uvector_async(string_data_col_ptrs, stream,
                                                          rmm::mr::get_current_device_resource());

    dim3 const string_blocks(
        std::min(std::max(MIN_STRING_BLOCKS, num_rows / NUM_STRING_ROWS_PER_BLOCK_FROM_ROWS),
                 MAX_STRING_BLOCKS));

    detail::copy_strings_from_rows<<<string_blocks, NUM_WARPS_IN_BLOCK * cudf::detail::warp_size, 0,
                                     stream.value()>>>(
        offset_functor, dev_string_row_offsets.data(), dev_string_lengths.data(),
        dev_string_col_offsets.data(), dev_string_data_cols.data(), child.data<int8_t>(), num_rows,
        static_cast<cudf::size_type>(string_col_offsets.size()));

    // merge strings back into output_columns
    int string_idx = 0;
    for (int i = 0; i < static_cast<int>(schema.size()); ++i) {
      if (schema[i].id() == type_id::STRING) {
        // stuff real string column
        auto const null_count = string_row_offset_columns[string_idx]->null_count();
        auto string_data = string_row_offset_columns[string_idx].release()->release();
        output_columns[i] = make_strings_column(
            num_rows,
            std::make_unique<cudf::column>(std::move(string_col_offsets[string_idx]),
                                           rmm::device_buffer{}, 0),
            std::make_unique<cudf::column>(std::move(string_data_cols[string_idx]),
                                           rmm::device_buffer{}, 0),
            null_count, std::move(*string_data.null_mask.release()));
        string_idx++;
      }
    }
  }

  for (auto &col : output_columns) {
    col->set_null_count(cudf::null_count(col->view().null_mask(), 0, col->size()));
  }
  return std::make_unique<table>(std::move(output_columns));
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

    // Ideally we would check that the offsets are all the same, etc. but for now this is probably
    // fine
    CUDF_EXPECTS(size_per_row * num_rows == child.size(),
                 "The layout of the data appears to be off");
    auto dev_column_start =
        make_device_uvector_async(column_start, stream, rmm::mr::get_current_device_resource());
    auto dev_column_size =
        make_device_uvector_async(column_size, stream, rmm::mr::get_current_device_resource());

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

    for (auto &col : output_columns) {
      col->set_null_count(cudf::null_count(col->view().null_mask(), 0, col->size()));
    }
    return std::make_unique<table>(std::move(output_columns));
  } else {
    CUDF_FAIL("Only fixed width types are currently supported");
  }
}

} // namespace jni

} // namespace cudf
