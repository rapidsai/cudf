/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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
#ifndef GROUPBY_KERNELS_CUH
#define GROUPBY_KERNELS_CUH

#include <cudf/groupby.hpp>
#include "../common/type_info.hpp"
#include "../common/kernel_utils.hpp"

namespace cudf {
namespace groupby {
namespace hash {
 
template <bool nullable = true>
struct row_hasher {
  using result_type = hash_value_type;  // TODO Remove when aggregating
                                        // map::insert function is removed
  device_table table;
  row_hasher(device_table const& t) : table{t} {}

  __device__ auto operator()(gdf_size_type row_index) const {
    return hash_row<nullable>(table, row_index);
  }
};

/**---------------------------------------------------------------------------*
 * @brief Builds a hash map where the keys are the rows of a `keys` table, and
 * the values are the aggregation(s) of corresponding rows in a `values` table.
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
 * @tparam values_have_nulls Indicates if rows in `input_values` contain null
 * values
 * @tparam Map The type of the hash map
 * @param map Hash map object to insert key,value pairs into.
 * @param input_keys The table whose rows will be keys of the hash map
 * @param input_values The table whose rows will be aggregated in the values of
 * the hash map
 * @param output_values Table that stores the results of aggregating rows of
 * `input_values`.
 * @param ops The set of aggregation operations to perform accross the columns
 * of the `input_values` rows
 * @param row_bitmask Bitmask where bit `i` indicates the presence of a null
 * value in row `i` of `input_keys`. Only used if `skip_rows_with_nulls` is
 * `true`.
 *---------------------------------------------------------------------------**/
template <bool skip_rows_with_nulls, bool values_have_nulls, typename Map>
__global__ void build_aggregation_map(
    Map map, device_table input_keys, device_table input_values,
    device_table output_values, operators* ops,
    bit_mask::bit_mask_t const* const __restrict__ row_bitmask) {
  gdf_size_type i = threadIdx.x + blockIdx.x * blockDim.x;

  while (i < input_keys.num_rows()) {
    if (skip_rows_with_nulls and not bit_mask::is_valid(row_bitmask, i)) {
      i += blockDim.x * gridDim.x;
      continue;
    }

    auto result = map.insert(thrust::make_pair(i, i));

    aggregate_row<values_have_nulls>(output_values, result.first->second,
                                     input_values, i, ops);
    i += blockDim.x * gridDim.x;
  }
}

/**---------------------------------------------------------------------------*
 * @brief Extracts the resulting keys and values of the groupby operation from a
 * hash map and sparse output table.
 *
 * @tparam keys_have_nulls Indicates the presence of null values in the keys
 * table
 * @tparam values_have_nulls Indicates the presence of null values in the values
 * table
 * @tparam Map The type of the hash map object
 * @param map[in] The hash map whos "keys" are indices into the `input_keys`
 * table, and "values" are indices into the `sparse_output_values` table
 * @param input_keys The table whose rows were used as the keys to build the
 * hash map
 * @param output_keys[out] Resulting keys of the groupby operation. Contains the
 * unique set of key rows from `input_keys`. The "key" row at `i` corresponds to
 * the value row at `i` in `dense_output_values`.
 * @param sparse_output_values[in] The sparse table that holds the result of
 * aggregating the values corresponding to the rows in `input_keys`. The
 * "values" of the hash map index into this table.
 * @param dense_output_values[out] The compacted version of
 * `sparse_output_values` where row `i` corresponds to row `i` in the
 * `output_keys` table.
 * @param output_write_index[in/out] Global counter used for determining write
 * location for output keys/values. When kernel is complete, indicates the final
 * result size.
 *---------------------------------------------------------------------------**/
template <bool keys_have_nulls, bool values_have_nulls, typename Map>
__global__ void extract_groupby_result(Map map, device_table const input_keys,
                                       device_table output_keys,
                                       device_table const sparse_output_values,
                                       device_table dense_output_values,
                                       gdf_size_type* output_write_index) {
  gdf_size_type i = threadIdx.x + blockIdx.x * blockDim.x;

  using pair_type = typename Map::value_type;

  pair_type const* const __restrict__ table_pairs{map.data()};

  while (i < map.capacity()) {
    gdf_size_type source_key_row_index;
    gdf_size_type source_value_row_index;

    // The way the aggregation map is built, these two indices will always be
    // equal, but lets be generic just in case that ever changes.
    thrust::tie(source_key_row_index, source_value_row_index) = table_pairs[i];

    if (source_key_row_index != map.get_unused_key()) {
      auto output_index = atomicAdd(output_write_index, 1);

      // TODO: Optimize setting bits in output bitmask. Currently, we rely on
      // the functionality of `copy_row` to update the target's bitmask, which
      // is inefficient as it requires an atomic per bit. This could be done
      // here instead with warp intrinsics.
      copy_row<keys_have_nulls>(output_keys, output_index, input_keys,
                                source_key_row_index);

      copy_row<values_have_nulls>(dense_output_values, output_index,
                                  sparse_output_values, source_value_row_index);
    }
    i += gridDim.x * blockDim.x;
  }
}

}  // namespace hash
}  // namespace groupby
}  // namespace cudf

#endif
