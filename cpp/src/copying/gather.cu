/*
 * Copyright (c) 2018-2019, NVIDIA CORPORATION.
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

#include "copying.hpp"
#include "cudf.h"
#include "gather.hpp"
#include "rmm/thrust_rmm_allocator.h"
#include "utilities/cudf_utils.h"
#include "utilities/type_dispatcher.hpp"
#include <bitmask/legacy_bitmask.hpp>
#include <table/table.hpp>

#include <algorithm>
#include <thrust/gather.h>

/**
 * @brief Operations for copying from one column to another
 * @file copying_ops.cu
 */

namespace {

/**---------------------------------------------------------------------------*
 * @brief Function object to check if an index is within the bounds [begin,
 * end).
 *
 *---------------------------------------------------------------------------**/
struct bounds_checker {
  gdf_index_type const begin;
  gdf_index_type const end;

  __device__ bounds_checker(gdf_index_type begin_, gdf_index_type end_)
      : begin{begin_}, end{end_} {}

  __device__ __forceinline__ bool operator()(gdf_index_type const index) {
    return ((index >= begin) && (index < end));
  }
};

/**---------------------------------------------------------------------------*
 * @brief Conditionally gathers the bits of a validity bitmask.
 *
 * Gathers the bits of a validity bitmask according to a gather map.
 * If `pred(stencil[i])` evaluates to true, then bit `i` in `destination_mask`
 * will equal bit `gather_map[i]` from the `source_mask`.
 *
 * If `pred(stencil[i])` evaluates to false, then bit `i` in `destination_mask`
 * will be set to 0.
 *
 * If any value appears in `gather_map` more than once, the result is undefined.
 *
 * If any of the range [source_mask, source_mask + num_source_rows) overlaps
 * [destination_mask, destination_mask + num_destination_rows), the result is
 * undefined.
 *
 * @tparam T The type of the stencil array
 * @tparam P The type of the predicate
 * @param[in] source_mask The mask whose bits will be gathered
 * @param[in] num_source_rows The number of bits in the source_mask
 * @param[out] destination_mask The output after gathering the input
 * @param[in] num_destination_rows The number of bits in the
 * destination_mask
 * @param[in] gather_map The map that indicates where elements from the
 * input will be gathered to in the output. Length must be equal to
 * `num_destination_rows`.
 * @param[in] stencil An array of values that will be evaluated by the
 * predicate. Length must be equal to `num_destination_rows`.
 * @param[in] pred Unary predicate applied to the stencil values
 *---------------------------------------------------------------------------**/
template <typename T, typename P>
__global__ void gather_bitmask_if_kernel(
    gdf_valid_type const* const __restrict__ source_mask,
    gdf_size_type const num_source_rows, gdf_valid_type* const destination_mask,
    gdf_size_type const num_destination_rows, gdf_index_type const* gather_map,
    T const* stencil, P pred) {
  using MaskType = uint32_t;
  constexpr uint32_t BITS_PER_MASK{sizeof(MaskType) * 8};

  // TODO: Update to use new bit_mask_t
  MaskType* const __restrict__ destination_mask32 =
      reinterpret_cast<MaskType*>(destination_mask);

  gdf_index_type destination_row = threadIdx.x + blockIdx.x * blockDim.x;

  auto active_threads =
      __ballot_sync(0xffffffff, destination_row < num_destination_rows);

  while (destination_row < num_destination_rows) {
    bool source_bit_is_valid{false};
    bool const predicate_is_true{pred(stencil[destination_row])};
    if (predicate_is_true) {
      // If the predicate for `destination_row` is false, it's valid for
      // `gather_map[destination_row]` to be out of bounds,
      // therefore, only use it if the predicate evaluates to true
      source_bit_is_valid =
          gdf_is_valid(source_mask, gather_map[destination_row]);
    }

    bool const destination_bit_is_valid{
        gdf_is_valid(destination_mask, destination_row)};

    // Use ballot to find all valid bits in this warp and create the output
    // bitmask element
    // If the predicate is false, and the destination bit was valid, don't
    // overwrite it
    MaskType const result_mask =
        __ballot_sync(active_threads,
                      (predicate_is_true and source_bit_is_valid) or
                          (not predicate_is_true and destination_bit_is_valid));

    gdf_index_type const output_element = destination_row / BITS_PER_MASK;

    // Only one thread writes output
    if (0 == threadIdx.x % warpSize) {
      destination_mask32[output_element] = result_mask;
    }

    destination_row += blockDim.x * gridDim.x;
    active_threads =
        __ballot_sync(active_threads, destination_row < num_destination_rows);
  }
}

/**---------------------------------------------------------------------------*
 * @brief Gathers the bits of a validity bitmask.
 *
 * Gathers the bits from the source bitmask into the destination bitmask
 * according to a `gather_map` such that bit `i` in `destination_mask` will be
 * equal to bit `gather_map[i]` from `source_bitmask`.
 *
 * Undefined behavior results if any value in `gather_map` is outside the range
 * [0, num_source_rows).
 *
 * If any value appears in `gather_map` more than once, the result is undefined.
 *
 * If any of the range [source_mask, source_mask + num_source_rows) overlaps
 * [destination_mask, destination_mask + num_destination_rows), the result is
 * undefined.
 *
 * @param[in] source_mask The mask whose bits will be gathered
 * @param[in] num_source_rows The number of bits in the source_mask
 * @param[out] destination_mask The output after gathering the input
 * @param[in] num_destination_rows The number of bits in the
 * destination_mask
 * @param[in] gather_map The map that indicates where elements from the
 * input will be gathered to in the output. Length must be equal to
 * `num_destination_rows`.
 *---------------------------------------------------------------------------**/
__global__ void gather_bitmask_kernel(
    gdf_valid_type const* const __restrict__ source_mask,
    gdf_size_type const num_source_rows, gdf_valid_type* const destination_mask,
    gdf_size_type const num_destination_rows,
    gdf_index_type const* __restrict__ gather_map) {
  using MaskType = uint32_t;
  constexpr uint32_t BITS_PER_MASK{sizeof(MaskType) * 8};

  // Cast bitmask to a type to a 4B type
  // TODO: Update to use new bit_mask_t
  MaskType* const __restrict__ destination_mask32 =
      reinterpret_cast<MaskType*>(destination_mask);

  gdf_index_type destination_row = threadIdx.x + blockIdx.x * blockDim.x;

  auto active_threads =
      __ballot_sync(0xffffffff, destination_row < num_destination_rows);

  while (destination_row < num_destination_rows) {
    bool const source_bit_is_valid{
        gdf_is_valid(source_mask, gather_map[destination_row])};

    // Use ballot to find all valid bits in this warp and create the output
    // bitmask element
    MaskType const result_mask{
        __ballot_sync(active_threads, source_bit_is_valid)};

    gdf_index_type const output_element = destination_row / BITS_PER_MASK;

    // Only one thread writes output
    if (0 == threadIdx.x % warpSize) {
      destination_mask32[output_element] = result_mask;
    }

    destination_row += blockDim.x * gridDim.x;
    active_threads =
        __ballot_sync(active_threads, destination_row < num_destination_rows);
  }
}

/**---------------------------------------------------------------------------*
 * @brief Gathers the bits from a source bitmask into a destination bitmask
 * based on a map.
 *
 * Gathers the bits from the source bitmask into the destination bitmask
 * according to a `gather_map` such that bit `i` in `destination_mask` will be
 * equal to bit `gather_map[i]` from `source_bitmask`.
 *
 * Optionally performs bounds checking on the values of the `gather_map` that
 * ignores values outside [0, num_source_rows). It is undefined behavior if a
 * value in `gather_map` is outside these bounds and bounds checking is not
 * enabled.
 *
 * If the same value appears more than once in `gather_map`, the result is
 * undefined.
 *
 * @param[in] source_mask The mask from which bits will be gathered
 * @param[in] num_source_rows The number of bits in the source_mask
 * @param[in,out] destination_mask The mask to which bits will be gathered.
 * Buffer must be preallocated with sufficient storage to hold
 * `num_destination_rows` bits.
 * @param[in] num_destination_rows The number of bits in the destionation_mask
 * @param[in] gather_map An array of indices that maps the bits in the source
 * bitmask to bits in the destination bitmask. The number of elements in the
 * `gather_map` must be equal to `num_destination_rows`.
 * @param[in] check_bounds Optionally perform bounds checking of values in
 * `gather_map`
 * @param[in] stream Optional CUDA stream on which to execute kernels
 *---------------------------------------------------------------------------**/
void gather_bitmask(gdf_valid_type const* source_mask,
                    gdf_size_type num_source_rows,
                    gdf_valid_type* destination_mask,
                    gdf_size_type num_destination_rows,
                    gdf_index_type const gather_map[],
                    bool check_bounds = false, cudaStream_t stream = 0) {
  CUDF_EXPECTS(destination_mask != nullptr, "Missing valid buffer allocation");

  constexpr gdf_size_type BLOCK_SIZE{256};
  const gdf_size_type gather_grid_size =
      (num_destination_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;

  gdf_valid_type* output_bitmask{destination_mask};

  // Allocate a temporary results buffer if gathering in-place
  bool const in_place{source_mask == destination_mask};
  rmm::device_vector<gdf_valid_type> temp_bitmask;
  if (in_place) {
    temp_bitmask.resize(gdf_valid_allocation_size(num_destination_rows));
    output_bitmask = temp_bitmask.data().get();
  }

  if (check_bounds) {
    gather_bitmask_if_kernel<<<gather_grid_size, BLOCK_SIZE, 0, stream>>>(
        source_mask, num_source_rows, output_bitmask, num_destination_rows,
        gather_map, gather_map, bounds_checker{0, num_source_rows});
  } else {
    gather_bitmask_kernel<<<gather_grid_size, BLOCK_SIZE, 0, stream>>>(
        source_mask, num_source_rows, output_bitmask, num_destination_rows,
        gather_map);
  }

  CHECK_STREAM(stream);

  if (in_place) {
    thrust::copy(rmm::exec_policy(stream)->on(stream), temp_bitmask.begin(),
                 temp_bitmask.end(), destination_mask);
  }

  CHECK_STREAM(stream);
}

/**---------------------------------------------------------------------------*
 * @brief Function object for gathering a type-erased
 * gdf_column. To be used with the cudf::type_dispatcher.
 *
 *---------------------------------------------------------------------------**/
struct column_gatherer {
  /**---------------------------------------------------------------------------*
   * @brief Type-dispatched function to gather from one column to another based
   * on a `gather_map`.
   *
   * @tparam ColumnType Dispatched type for the column being gathered
   * @param source_column The column to gather from
   * @param gather_map Array of indices that maps source elements to destination
   * elements
   * @param destination_column The column to gather into
   * @param check_bounds Optionally perform bounds checking on the values of
   * `gather_map`
   * @param stream Optional CUDA stream on which to execute kernels
   *---------------------------------------------------------------------------**/
  template <typename ColumnType>
  void operator()(gdf_column const* source_column,
                  gdf_index_type const gather_map[],
                  gdf_column* destination_column, bool check_bounds = false,
                  cudaStream_t stream = 0) {
    ColumnType const* const source_data{
        static_cast<ColumnType const*>(source_column->data)};
    ColumnType* destination_data{
        static_cast<ColumnType*>(destination_column->data)};

    gdf_size_type const num_destination_rows{destination_column->size};

    // If gathering in-place, allocate temporary buffers to hold intermediate
    // results
    bool const in_place{source_data == destination_data};
    rmm::device_vector<ColumnType> temp_destination;
    if (in_place) {
      temp_destination.resize(num_destination_rows);
      destination_data = temp_destination.data().get();
    }

    if (check_bounds) {
      thrust::gather_if(rmm::exec_policy(stream)->on(stream), gather_map,
                        gather_map + num_destination_rows, gather_map,
                        source_data, destination_data,
                        bounds_checker{0, source_column->size});
    } else {
      thrust::gather(rmm::exec_policy(stream)->on(stream), gather_map,
                     gather_map + num_destination_rows, source_data,
                     destination_data);
    }

    // Copy temporary buffers used for in-place gather to destination column
    if (in_place) {
      thrust::copy(rmm::exec_policy(stream)->on(stream),
                   temp_destination.begin(), temp_destination.end(),
                   static_cast<ColumnType*>(destination_column->data));
    }

    if (destination_column->valid != nullptr) {
      gather_bitmask(source_column->valid, source_column->size,
                     destination_column->valid, num_destination_rows,
                     gather_map, check_bounds, stream);

      // TODO compute the null count in the gather_bitmask kernels
      set_null_count(*destination_column);
    }

    CHECK_STREAM(stream);
  }
};
}  // namespace

namespace cudf {
namespace detail {

void gather(table const* source_table, gdf_index_type const gather_map[],
            table* destination_table, bool check_bounds, cudaStream_t stream) {
  CUDF_EXPECTS(nullptr != source_table, "source table is null");
  CUDF_EXPECTS(nullptr != destination_table, "destination table is null");

  // If the destination is empty, return immediately as there is nothing to
  // gather
  if (0 == destination_table->num_rows()) {
    return;
  }

  CUDF_EXPECTS(nullptr != gather_map, "gather_map is null");
  CUDF_EXPECTS(source_table->num_columns() == destination_table->num_columns(),
               "Mismatched number of columns");

  auto gather_column = [gather_map, check_bounds, stream](
                           gdf_column const* source, gdf_column* destination) {
    CUDF_EXPECTS(source->dtype == destination->dtype, "Column type mismatch");

    // If the source column has a valid buffer, the destination column must
    // also have one
    bool const source_has_nulls{source->valid != nullptr};
    bool const dest_has_nulls{destination->valid != nullptr};
    CUDF_EXPECTS((source_has_nulls && dest_has_nulls) || (not source_has_nulls),
                 "Missing destination validity buffer");

    // TODO: Each column could be gathered on a separate stream
    cudf::type_dispatcher(source->dtype, column_gatherer{}, source, gather_map,
                          destination, check_bounds, stream);

    return destination;
  };

  // Gather columns one-by-one
  std::transform(source_table->begin(), source_table->end(),
                 destination_table->begin(), destination_table->begin(),
                 gather_column);
}

}  // namespace detail

void gather(table const* source_table, gdf_index_type const gather_map[],
            table* destination_table) {
  detail::gather(source_table, gather_map, destination_table);
}

}  // namespace cudf
