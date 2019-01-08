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
 * @brief Gathers a set bit from a source bitmask to a bit in a destination
 * mask.
 *
 * If bit `source_index` is set in `source_mask`, then bit `destination_index`
 * will be set in `destination_mask`.
 *
 * If bit `source_index` in `source_mask` is not set, then bit
 * `destination_index` in `destination_mask` is unmodified.
 *
 * @tparam MaskType The type of the bitmask
 * @param source_mask The mask to gather from
 * @param source_index The index of the bit to gather from
 * @param destination_mask The mask to gather to
 * @param destination_index The index of the bit to gather to
 *---------------------------------------------------------------------------**/
template <typename MaskType>
__device__ __forceinline__ void gather_bit(
    MaskType const* __restrict__ source_mask, gdf_index_type source_index,
    MaskType* __restrict__ destination_mask, gdf_index_type destination_index) {
  constexpr uint32_t BITS_PER_MASK = 8 * sizeof(MaskType);

  // Get the source bit
  // FIXME Replace with a standard `get_bit` function
  MaskType input_bit = (MaskType{1} << (source_index % BITS_PER_MASK));
  if (nullptr != source_mask) {
    input_bit = input_bit & source_mask[source_index / BITS_PER_MASK];
  }

  // Only set the output bit if the input is valid
  if (input_bit > 0) {
    // FIXME Replace with a standard `set_bit` function
    // Construct the mask that sets the bit for the output row
    MaskType const output_bit = MaskType{1}
                                << (destination_index % BITS_PER_MASK);

    // Find the mask in the output that will hold the bit for output row
    gdf_index_type const output_location = destination_index / BITS_PER_MASK;

    // Bitwise OR to set the gathered row's bit
    atomicOr(&destination_mask[output_location], output_bit);
  }
}

/**---------------------------------------------------------------------------*
 * @brief Conditionally gathers the set bits of a validity bitmask.
 *
 * Gathers the set bits of a validity bitmask according to a gather map.
 * If pred(stencil[i]) evaluates to true, then bit `i` in `destination_mask`
 * will be set if bit `gather_map[i]` from the `source_mask` is set.
 *
 * If bit `gather_map[i]` from `source_mask` is not set, then bit `i` in
 * `destination_mask` is unmodified.
 *
 * If any value appears in `gather_map` more than once, the result is undefined.
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
    gdf_valid_type const* const source_mask,
    gdf_size_type const num_source_rows, gdf_valid_type* const destination_mask,
    gdf_size_type const num_destination_rows, gdf_index_type const* gather_map,
    T const* stencil, P pred) {
  using MaskType = uint32_t;

  // Cast the validity type to a type where atomicOr is natively supported
  // TODO: Update to use new bit_mask_t
  const MaskType* __restrict__ source_mask32 =
      reinterpret_cast<MaskType const*>(source_mask);
  MaskType* const __restrict__ destination_mask32 =
      reinterpret_cast<MaskType*>(destination_mask);

  gdf_index_type destination_row = threadIdx.x + blockIdx.x * blockDim.x;

  while (destination_row < num_destination_rows) {
    // If the predicate is false for this row, continue to the next row
    if (not pred(stencil[destination_row])) {
      destination_row += blockDim.x * gridDim.x;
      continue;
    }

    gather_bit(source_mask32, gather_map[destination_row], destination_mask32,
               destination_row);

    destination_row += blockDim.x * gridDim.x;
  }
}

/**---------------------------------------------------------------------------*
 * @brief Gathers the set bits of a validity bitmask.
 *
 * Gathers the set bits of a validity bitmask according to a gather map.
 * If bit `gather_map[i]` in `source_mask` is set, then bit `i` in
 *`destination_mask` will be set.
 *
 * If bit `gather_map[i]` in `source_mask` is *not* set, then bit `i` in
 *`destination_mask` will be unmodified.
 *
 * Undefined behavior results if any value in `gather_map` is outside the range
 * [0, num_destination_rows).
 *
 * If any value appears in `gather_map` more than once, the result is undefined.
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
__global__ void gather_bitmask_kernel(gdf_valid_type const* const source_mask,
                                      gdf_size_type const num_source_rows,
                                      gdf_valid_type* const destination_mask,
                                      gdf_size_type const num_destination_rows,
                                      gdf_index_type const* gather_map) {
  using MaskType = uint32_t;

  // Cast the validity type to a type where atomicOr is natively supported
  // TODO: Update to use new bit_mask_t
  const MaskType* __restrict__ source_mask32 =
      reinterpret_cast<MaskType const*>(source_mask);
  MaskType* const __restrict__ destination_mask32 =
      reinterpret_cast<MaskType*>(destination_mask);

  gdf_index_type destination_row = threadIdx.x + blockIdx.x * blockDim.x;

  while (destination_row < num_destination_rows) {
    gather_bit(source_mask32, gather_map[destination_row], destination_mask32,
               destination_row);

    destination_row += blockDim.x * gridDim.x;
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
 * @return gdf_error
 *---------------------------------------------------------------------------**/
gdf_error gather_bitmask(gdf_valid_type const* source_mask,
                         gdf_size_type num_source_rows,
                         gdf_valid_type* destination_mask,
                         gdf_size_type num_destination_rows,
                         gdf_index_type const gather_map[],
                         bool check_bounds = false, cudaStream_t stream = 0) {
  constexpr gdf_size_type BLOCK_SIZE{256};
  const gdf_size_type gather_grid_size =
      (num_destination_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;

  gdf_valid_type* output_bitmask{destination_mask};

  // Allocate a temporary results buffer if gathering in-place
  bool const in_place{source_mask == destination_mask};
  rmm::device_vector<gdf_valid_type> temp_bitmask;
  if (in_place) {
    temp_bitmask.resize(num_destination_rows);
    output_bitmask = temp_bitmask.data().get();
  }

  // gather_bitmask kernels only gather the *set* bits, therefore we must
  // ensure the output bitmask is initialized to all unset bits
  CUDA_TRY(cudaMemsetAsync(output_bitmask, 0,
                           num_destination_rows * sizeof(gdf_valid_type),
                           stream);)

  if (check_bounds) {
    gather_bitmask_if_kernel<<<gather_grid_size, BLOCK_SIZE, 0, stream>>>(
        source_mask, num_source_rows, output_bitmask, num_destination_rows,
        gather_map, gather_map, bounds_checker{0, num_source_rows});
  } else {
    gather_bitmask_kernel<<<gather_grid_size, BLOCK_SIZE, 0, stream>>>(
        source_mask, num_source_rows, output_bitmask, num_destination_rows,
        gather_map);
  }

  if (in_place) {
    thrust::copy(rmm::exec_policy(stream), temp_bitmask.begin(),
                 temp_bitmask.end(), destination_mask);
  }

  return GDF_SUCCESS;
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
   * @param source_column The column that will be gathered from
   * @param gather_map Array of indices that maps source elements to destination
   * elements
   * @param destination_column The column that will be gathered into
   * @param check_bounds Optionally perform bounds checking on the values of
   * `gather_map`
   * @param stream Optional CUDA stream on which to execute kernels
   * @return gdf_error
   *---------------------------------------------------------------------------**/
  template <typename ColumnType>
  gdf_error operator()(gdf_column const* source_column,
                       gdf_index_type const gather_map[],
                       gdf_column* destination_column,
                       bool check_bounds = false, cudaStream_t stream = 0) {
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
      thrust::gather_if(rmm::exec_policy(stream), gather_map,
                        gather_map + num_destination_rows, gather_map,
                        source_data, destination_data,
                        bounds_checker{0, source_column->size});
    } else {
      thrust::gather(rmm::exec_policy(stream), gather_map,
                     gather_map + num_destination_rows, source_data,
                     destination_data);
    }

    // Copy temporary buffers used for in-place gather to destination column
    if (in_place) {
      thrust::copy(temp_destination.begin(), temp_destination.end(),
                   static_cast<ColumnType*>(destination_column->data));
    }

    // Gather bitmasks if they exist
    bool const bitmasks_exist{(nullptr != source_column->valid) &&
                              (nullptr != destination_column->valid)};
    if (bitmasks_exist) {
      gdf_error gdf_status = gather_bitmask(
          source_column->valid, source_column->size, destination_column->valid,
          num_destination_rows, gather_map, check_bounds, stream);

      GDF_REQUIRE(GDF_SUCCESS == gdf_status, gdf_status);
    }

    return GDF_SUCCESS;
  }
};
}  // namespace

namespace cudf {
namespace detail {
/**---------------------------------------------------------------------------*
 * @brief Gathers the rows (including null values) of a set of source columns
 * into a set of destination columns.
 *
 * Gathers the rows of the source columns into the destination columns according
 * to a gather map such that row "i" in the destination columns will contain
 * row "gather_map[i]" from the source columns.
 *
 * The datatypes between coresponding columns in the source and destination
 * columns must be the same.
 *
 * The number of elements in the gather_map must equal the number of rows in the
 * destination columns.
 *
 * Optionally performs bounds checking on the values of the `gather_map` that
 * ignores values outside [0, num_source_rows). It is undefined behavior if a
 * value in `gather_map` is outside these bounds and bounds checking is not
 * enabled.
 *
 * If the same index appears more than once in gather_map, the result is
 * undefined.
 *
 * @param[in] source_table The input columns whose rows will be gathered
 * @param[in] gather_map An array of indices that maps the rows in the source
 * columns to rows in the destination columns.
 * @param[out] destination_table A preallocated set of columns with a number
 * of rows equal in size to the number of elements in the gather_map that will
 * contain the rearrangement of the source columns based on the mapping
 * determined by the gather_map.
 * @param check_bounds Optionally perform bounds checking on the values of
 * `gather_map`
 * @param stream Optional CUDA stream on which to execute kernels
 * @return gdf_error
 *---------------------------------------------------------------------------**/
gdf_error gather(table const* source_table, gdf_index_type const gather_map[],
                 table* destination_table, bool check_bounds = false,
                 cudaStream_t stream = 0) {
  assert(source_table->size() == destination_table->size());

  gdf_error gdf_status{GDF_SUCCESS};

  auto gather_column = [gather_map, check_bounds, stream](
                           gdf_column const* source, gdf_column* destination) {
    if (source->dtype != destination->dtype) {
      throw GDF_DTYPE_MISMATCH;
    }

    // TODO: Each column could be gathered on a separate stream
    gdf_error gdf_status =
        cudf::type_dispatcher(source->dtype, column_gatherer{}, source,
                              gather_map, destination, check_bounds, stream);

    if (GDF_SUCCESS != gdf_status) {
      throw gdf_status;
    }

    return destination;
  };

  try {
    // Gather columns one-by-one
    std::transform(source_table->begin(), source_table->end(),
                   destination_table->begin(), destination_table->begin(),
                   gather_column);
  } catch (gdf_error e) {
    return e;
  }
  return gdf_status;
}

}  // namespace detail

gdf_error gather(table const* source_table, gdf_index_type const gather_map[],
                 table* destination_table) {
  return detail::gather(source_table, gather_map, destination_table);
}

}  // namespace cudf
