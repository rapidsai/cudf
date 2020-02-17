/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#pragma once

#include <cudf/groupby.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/aggregation/aggregation.cuh>
#include <cudf/utilities/bit.hpp>

namespace cudf {
namespace experimental {
namespace groupby {
namespace detail {
namespace hash {
 
/**
 * @brief Computes single-pass aggregations and store results into a sparse 
 * `output_values` table, and populate `map` with indices of unique keys
 *
 * The hash map is built by inserting every row `i` from the `keys` and
 * `values` tables as a single (key,value) pair. When the pair is inserted, if
 * the key was not already present in the map, then the corresponding value is
 * simply copied to the output. If the key was already present in the map,
 * then the inserted `values` row is aggregated with the existing row. This
 * aggregation is done for every element `j` in the row by applying aggregation
 * operation `j` between the new and existing element.
 *
 * Instead of storing the entire rows from `input_keys` and `input_values` in
 * the hashmap, we instead store the row indices. For example, when inserting
 * row at index `i` from `input_keys` into the hash map, the value `i` is what
 * gets stored for the hash map's "key". It is assumed the `map` was constructed
 * with a custom comparator that uses these row indices to check for equality
 * between key rows. For example, comparing two keys `k0` and `k1` will compare
 * the two rows `input_keys[k0] ?= input_keys[k1]`
 *
 * Likewise, we store the row indices for the hash maps "values". These indices
 * index into the `output_values` table. For a given key `k` (which is an index
 * into `input_keys`), the corresponding value `v` indexes into `output_values`
 * and stores the result of aggregating rows from `input_values` from rows of
 * `input_keys` equivalent to the row at `k`.
 *
 * The exact size of the result is not known a priori, but can be upper bounded
 * by the number of rows in `input_keys` & `input_values`. Therefore, it is
 * assumed `output_values` has sufficient storage for an equivalent number of
 * rows. In this way, after all rows are aggregated, `output_values` will likely
 * be "sparse", meaning that not all rows contain the result of an aggregation.
 *
 * @tparam skip_rows_with_nulls Indicates if rows in `input_keys` containing
 * null values should be skipped. It `true`, it is assumed `row_bitmask` is a
 * bitmask where bit `i` indicates the presence of a null value in row `i`.
 * @tparam Map The type of the hash map
 * @param map Hash map object to insert key,value pairs into.
 * @param num_keys The number of rows in input keys table
 * @param input_values The table whose rows will be aggregated in the values of
 * the hash map
 * @param output_values Table that stores the results of aggregating rows of
 * `input_values`.
 * @param aggs The set of aggregation operations to perform accross the columns
 * of the `input_values` rows
 * @param row_bitmask Bitmask where bit `i` indicates the presence of a null
 * value in row `i` of input keys. Only used if `skip_rows_with_nulls` is `true`
 */
template <bool skip_rows_with_nulls, typename Map>
__global__ void compute_single_pass_aggs(
    Map map, size_type num_keys, table_device_view input_values,
    mutable_table_device_view output_values, aggregation::Kind* aggs,
    bitmask_type const* const __restrict__ row_bitmask)
{  
  size_type i = threadIdx.x + blockIdx.x * blockDim.x;

  while (i < num_keys) {
    if (skip_rows_with_nulls and not cudf::bit_is_set(row_bitmask, i)) {
      i += blockDim.x * gridDim.x;
      continue;
    }

    auto result = map.insert(thrust::make_pair(i, i));

    experimental::detail::aggregate_row<true, true>(
      output_values, result.first->second, input_values, i, aggs);
    i += blockDim.x * gridDim.x;
  }
}

// TODO (dm): variance kernel

/**
 * @brief Extracts the populated elements from a hash map to get the indices of
 * result values in a groupby operation.
 * 
 * The hash map should be constructed such that it stores the indices of the
 * populated values in a sparse result table.
 * 
 * This method uses the @p map to check for populated map elements. It then 
 * gets the element and appends its value to the  @p gather_map.
 * The @p gather_map can be used in conjunction with sparse result values
 * written by compute_single_pass_aggs() to get the results in a dense form,
 * using a gather() operation.
 *
 * @tparam Map The type of the hash map object
 * @param map[in] The hash map that contains the indices into input keys that 
 * are unique and sparse values that are populated with aggregation results.
 * @param gather_map[out] The compressed array of populated values from @p map
 * @param output_write_index[in/out] Global counter used for determining write
 * location for output keys/values. When kernel is complete, indicates the final
 * result size.
 */
template <typename Map>
__global__ void extract_gather_map(Map map,
                                   size_type * const __restrict__ gather_map,
                                   size_type* output_write_index) {
  size_type i = threadIdx.x + blockIdx.x * blockDim.x;

  using pair_type = typename Map::value_type;

  pair_type const* const __restrict__ table_pairs{map.data()};

  while (i < map.capacity()) {
    size_type source_key_row_index;
    size_type source_value_row_index;

    // The way the aggregation map is built, these two indices will always be
    // equal.
    thrust::tie(source_key_row_index, source_value_row_index) = table_pairs[i];

    if (source_key_row_index != map.get_unused_key()) {
      auto output_index = atomicAdd(output_write_index, 1);
      gather_map[output_index] = source_value_row_index;
    }
    i += gridDim.x * blockDim.x;
  }
}

}  // namespace hash
}  // namespace detail
}  // namespace groupby
}  // namespace experimental
}  // namespace cudf
