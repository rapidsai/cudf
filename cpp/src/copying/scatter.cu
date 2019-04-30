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

#include <bitmask/legacy_bitmask.hpp>
#include "copying.hpp"
#include "cudf.h"
#include "rmm/thrust_rmm_allocator.h"
#include "utilities/cudf_utils.h"
#include "utilities/type_dispatcher.hpp"
#include <table/table.hpp>

#include <algorithm>
#include <thrust/scatter.h>

namespace cudf {

namespace {

/**---------------------------------------------------------------------------*
 * @brief Scatters the bits from one bitmask to another.
 *
 * @param[in] source_mask The mask that will be scattered.
 * @param[out] destination_mask The output after scattering the input
 * @param[in] scatter_map The map that indicates where elements from the input
   will be scattered to in the output. output_bit[ scatter_map [i] ] =
 input_bit[i]
 * @param[in] num_source_rows The number of bits in the source mask
*---------------------------------------------------------------------------**/
__global__ void scatter_bitmask_kernel(
    gdf_valid_type const* const __restrict__ source_mask,
    gdf_size_type const num_source_rows,
    gdf_valid_type* const __restrict__ destination_mask,
    gdf_index_type const* const __restrict__ scatter_map) {
  using mask_type = uint32_t;
  constexpr uint32_t BITS_PER_MASK = 8 * sizeof(mask_type);

  // Cast the validity type to a type where atomicOr is natively supported
  mask_type* const __restrict__ destination_mask32 =
      reinterpret_cast<mask_type*>(destination_mask);

  gdf_size_type row_number = threadIdx.x + blockIdx.x * blockDim.x;

  while (row_number < num_source_rows) {
    // Only scatter the input bit if it is valid
    if (gdf_is_valid(source_mask, row_number)) {
      const gdf_index_type output_row = scatter_map[row_number];

      // Set the according output bit
      const mask_type output_bit = static_cast<mask_type>(1)
                                   << (output_row % BITS_PER_MASK);

      // Find the mask in the output that will hold the bit for the scattered
      // row
      gdf_index_type const output_location = output_row / BITS_PER_MASK;

      // Bitwise OR to set the scattered row's bit
      atomicOr(&destination_mask32[output_location], output_bit);
    }

    row_number += blockDim.x * gridDim.x;
  }
}

void scatter_bitmask(gdf_valid_type const* source_mask,
                     gdf_size_type num_source_rows,
                     gdf_valid_type* destination_mask,
                     gdf_size_type num_destination_rows,
                     gdf_index_type const scatter_map[],
                     cudaStream_t stream = 0) {
  constexpr gdf_size_type BLOCK_SIZE{256};
  const gdf_size_type scatter_grid_size =
      (num_source_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;

  gdf_valid_type* output_bitmask{destination_mask};

  // Allocate temporary output bitmask if scattering in-place
  bool const in_place{source_mask == destination_mask};
  rmm::device_vector<gdf_valid_type> temp_bitmask;
  if (in_place) {
    temp_bitmask.resize(gdf_valid_allocation_size(num_destination_rows));
    output_bitmask = temp_bitmask.data().get();
  }

  // Ensure the output bitmask is initialized to zero
  CUDA_TRY(cudaMemsetAsync(
      output_bitmask, 0,
      gdf_num_bitmask_elements(num_destination_rows) * sizeof(gdf_valid_type),
      stream));

  scatter_bitmask_kernel<<<scatter_grid_size, BLOCK_SIZE, 0, stream>>>(
      source_mask, num_source_rows, destination_mask, scatter_map);

  CHECK_STREAM(stream);

  // Copy temporary bitmask to destination mask
  if (in_place) {
    thrust::copy(rmm::exec_policy(stream)->on(stream), temp_bitmask.begin(),
                 temp_bitmask.end(), destination_mask);
  }
  CHECK_STREAM(stream);
}

/**---------------------------------------------------------------------------*
 * @brief Function object for scattering a type-erased
 * gdf_column. To be used with the cudf::type_dispatcher.
 *---------------------------------------------------------------------------**/
struct column_scatterer {
  /**---------------------------------------------------------------------------*
   * @brief Type-dispatched function to scatter from one column to another based
   * on a `scatter_map`.
   *
   * @tparam ColumnType Dispatched type for the column being gathered
   * @param source_column The column that will be scattered from
   * @param scatter_map Array of indices that maps source elements to
   * destination elements
   * @param destination_column The column that will be scattered into
   * @param stream Optional CUDA stream on which to execute kernels
   *---------------------------------------------------------------------------**/
  template <typename ColumnType>
  void operator()(gdf_column const* source_column,
                  gdf_index_type const scatter_map[],
                  gdf_column* destination_column, cudaStream_t stream = 0) {
    ColumnType const* const source_data{
        static_cast<ColumnType const*>(source_column->data)};
    ColumnType* destination_data{
        static_cast<ColumnType*>(destination_column->data)};

    gdf_size_type const num_source_rows{source_column->size};

    // If scattering in-place, allocate a temporary buffer to hold intermediate
    // results
    bool const in_place{source_data == destination_data};
    rmm::device_vector<ColumnType> temp_destination;
    if (in_place) {
      temp_destination.resize(num_source_rows);
      destination_data = temp_destination.data().get();
    }

    // Scatter the column's data
    thrust::scatter(rmm::exec_policy(stream)->on(stream), source_data,
                    source_data + source_column->size, scatter_map,
                    destination_data);

    // Copy temporary buffer result to destination column
    if (in_place) {
      thrust::copy(temp_destination.begin(), temp_destination.end(),
                   static_cast<ColumnType*>(destination_column->data));
    }

    bool const bitmasks_exist{(nullptr != source_column->valid) &&
                              (nullptr != destination_column->valid)};
    if (bitmasks_exist) {
      scatter_bitmask(source_column->valid, source_column->size,
                      destination_column->valid, destination_column->size,
                      scatter_map);

      // Update destination column's null count
      set_null_count(*destination_column);
    }

    CHECK_STREAM(stream);
  }
};
}  // namespace

namespace detail {
void scatter(table const* source_table, gdf_index_type const scatter_map[],
             table* destination_table, cudaStream_t stream = 0) {
  CUDF_EXPECTS(nullptr != source_table, "source table is null");
  CUDF_EXPECTS(nullptr != destination_table, "destination table is null");

  // If the source table is empty, return immediately because there is nothing
  // to scatter
  if (0 == source_table->num_rows()) {
    return;
  }

  CUDF_EXPECTS(nullptr != scatter_map, "scatter_map is null");
  CUDF_EXPECTS(source_table->num_columns() == destination_table->num_columns(),
               "Mismatched number of columns");

  auto scatter_column = [scatter_map, stream](gdf_column const* source,
                                              gdf_column* destination) {
    CUDF_EXPECTS(source->dtype == destination->dtype, "Column type mismatch");

    // If the source column has a valid buffer, the destination column must
    // also have one
    bool const source_has_nulls{source->valid != nullptr};
    bool const dest_has_nulls{destination->valid != nullptr};
    CUDF_EXPECTS((source_has_nulls && dest_has_nulls) || (not source_has_nulls),
                 "Missing destination validity buffer");

    // TODO: Each column could be scattered on a separate stream
    cudf::type_dispatcher(source->dtype, column_scatterer{}, source,
                          scatter_map, destination, stream);

    return destination;
  };

  // Gather columns one-by-one
  std::transform(source_table->begin(), source_table->end(),
                 destination_table->begin(), destination_table->begin(),
                 scatter_column);
}
}  // namespace detail

void scatter(table const* source_table, gdf_index_type const scatter_map[],
             table* destination_table) {
  detail::scatter(source_table, scatter_map, destination_table);
}
}  // namespace cudf
