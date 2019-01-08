/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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

#include <thrust/gather.h>
#include "copying.hpp"
#include "cudf.h"
#include "rmm/thrust_rmm_allocator.h"
#include "utilities/type_dispatcher.hpp"

namespace {

struct bounds_checker {
  gdf_index_type const begin;
  gdf_index_type const end;

  __device__ bounds_checker(gdf_index_type begin_, gdf_index_type end_)
      : begin{begin_}, end{end_} {}

  __device__ __forceinline__ bool operator()(gdf_index_type const index) {
    return ((index >= begin) && (index < end));
  }
};

/* --------------------------------------------------------------------------*/
/**
 * @Synopsis  Conditionally gathers the bits of a validity bitmask.
 *
 * Gathers the bits of a validity bitmask according to a gather map.
 * If pred(stencil[i]) evaluates to true, then bit `i` in `destination_mask`
 * will be equal to bit `gather_map[i]` from the `source_mask`.
 *
 * @Param[in] source_mask The mask whose bits will be gathered
 * @Param[in] num_source_rows The number of bits in the source_mask
 * @Param[out] destination_mask The output after gathering the input
 * @Param[in] num_destination_rows The number of bits in the destination_mask
 * @Param[in] gather_map The map that indicates where elements from the input
 *  will be gathered to in the output. Length must be equal to
 * `num_destination_rows`.
 * @Param[in] stencil An array of values that will be evaluated by the
 * predicate. Length must be equal to `num_destination_rows`.
 * @Param[in] pred Unary predicate applied to the stencil values
 */
/* ----------------------------------------------------------------------------*/
template <typename T, typename P>
__global__ void gather_valid_if(gdf_valid_type const* const source_mask,
                                gdf_size_type const num_source_rows,
                                gdf_valid_type* const destination_mask,
                                gdf_size_type const num_destination_rows,
                                gdf_index_type const* gather_map,
                                T const* stencil, P pred) {
  using mask_type = uint32_t;
  constexpr uint32_t BITS_PER_MASK = 8 * sizeof(mask_type);

  // Cast the validity type to a type where atomicOr is natively supported
  // TODO: Update to use new bit_mask_t
  const mask_type* __restrict__ source_mask32 =
      reinterpret_cast<mask_type const*>(source_mask);
  mask_type* const __restrict__ destination_mask32 =
      reinterpret_cast<mask_type*>(destination_mask);

  gdf_index_type row_number = threadIdx.x + blockIdx.x * blockDim.x;

  while (row_number < num_destination_rows) {
    // If the predicate is false for this row, continue to the next row
    if (not pred(stencil[row_number])) {
      row_number += blockDim.x * gridDim.x;
      continue;
    }

    gdf_index_type const gather_location{gather_map[row_number]};

    // Get the bit corresponding from the gathered row
    // FIXME Replace with a standard `get_bit` function
    mask_type input_bit = (mask_type{1} << (gather_location % BITS_PER_MASK));
    if (nullptr != source_mask) {
      input_bit = input_bit & source_mask32[gather_location / BITS_PER_MASK];
    }

    // Only set the output bit if the input is valid
    if (input_bit > 0) {
      // FIXME Replace with a standard `set_bit` function
      // Construct the mask that sets the bit for the output row
      const mask_type output_bit = mask_type{1} << (row_number % BITS_PER_MASK);

      // Find the mask in the output that will hold the bit for output row
      const gdf_index_type output_location = row_number / BITS_PER_MASK;

      // Bitwise OR to set the gathered row's bit
      atomicOr(&destination_mask32[output_location], output_bit);
    }

    row_number += blockDim.x * gridDim.x;
  }
}

void gather_valid_kernel(gdf_valid_type const* source_mask,
                         gdf_size_type num_source_rows,
                         gdf_valid_type* destination_mask,
                         gdf_size_type num_destination_rows,
                         gdf_index_type const gather_map[],
                         cudaStream_t stream = 0) {
  constexpr gdf_size_type BLOCK_SIZE{256};
  const gdf_size_type gather_grid_size =
      (num_destination_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;

  gather_valid_if<<<gather_grid_size, BLOCK_SIZE, 0, stream>>>(
      source_mask, num_source_rows, destination_mask, num_destination_rows,
      gather_map, gather_map, bounds_checker{0, num_source_rows});
}

struct column_gatherer {
  template <typename ColumnType>
  void operator()(gdf_column const* source_column,
                  gdf_index_type const gather_map[],
                  gdf_column* destination_column, cudaStream_t stream = 0,
                  bool check_bounds = false) {
    ColumnType const* source_data{
        static_cast<ColumnType const*>(source_column->data)};
    ColumnType* destination_data{
        static_cast<ColumnType*>(destination_column->data)};

    gdf_size_type const num_destination_rows{destination_column->size};

    // Out-of-place gather
    if (source_data != destination_data) {
      if (check_bounds) {
        thrust::gather_if(rmm::exec_policy(stream), gather_map,
                          gather_map + num_destination_rows, gather_map,
                          source_data, destination_data,
                          bounds_checker{0, source_column->size});
      } else {
        thrust::gather(rmm::exec_policy(stream), gather_map,
                       gather_map + num_destination_rows, source_data,
                       destination_data);
      }

      // Gather bitmask
      if (nullptr != source_column->valid &&
          nullptr != destination_column->valid) {
        gather_valid_kernel(source_column->valid, source_column->size,
                            destination_column->valid, destination_column->size,
                            gather_map, stream);
      }
    }

    // In-place gather
    else {
      rmm::device_vector<ColumnType> temp_destination(num_destination_rows);

      if (check_bounds) {
        thrust::gather_if(rmm::exec_policy(stream), gather_map,
                          gather_map + num_destination_rows, gather_map,
                          source_data, temp_destination.begin(),
                          bounds_checker{0, source_column->size});
      } else {
        thrust::gather(rmm::exec_policy(stream), gather_map,
                       gather_map + num_destination_rows, source_data,
                       temp_destination.begin());
      }

      thrust::copy(rmm::exec_policy(stream), temp_destination.begin(),
                   temp_destination.end(), destination_data);

      // If the bitmask exists, gather it in-place
      if (nullptr != source_column->valid &&
          (source_column->valid == destination_column->valid)) {
        // Gather results into temporary buffer
        rmm::device_vector<gdf_valid_type> temp_valid_buffer(
            gdf_get_num_chars_bitmask(num_destination_rows));
        gather_valid_kernel(source_column->valid, source_column->size,
                            temp_valid_buffer.data().get(),
                            destination_column->size, gather_map, stream);

        // Copy temporary buffer to destination
        thrust::copy(rmm::exec_policy(stream), temp_valid_buffer.begin(),
                     temp_valid_buffer.end(), destination_column->valid);
      }
    }
  }
};
}  // namespace

namespace cudf {

/**
 * @brief Operations for copying from one column to another
 * @file copying_ops.cu
 */

gdf_error scatter(table const* source_table, gdf_index_type const scatter_map[],
                  table* destination_table) {
  return GDF_SUCCESS;
}

gdf_error gather(table const* source_table, gdf_index_type const gather_map[],
                 table* destination_table) {
  assert(source_table->size() == destination_table->size());

  gdf_error gdf_status{GDF_SUCCESS};

  auto gather_column = [&gather_map](gdf_column const* source,
                                     gdf_column* destination) {
    assert(source->dtype == destination->dtype);

    // TODO: Each column could be gathered on a separate stream
    cudf::type_dispatcher(source->dtype, column_gatherer{}, source, gather_map,
                          destination);
    return destination;
  };

  std::transform(source_table->begin(), source_table->end(),
                 destination_table->begin(), destination_table->begin(),
                 gather_column);

  return gdf_status;
}

}  // namespace cudf
