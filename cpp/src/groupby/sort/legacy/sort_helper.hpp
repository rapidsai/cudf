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

#pragma once

#include <cudf/utilities/legacy/type_dispatcher.hpp>

#include <cudf/cudf.h>
#include <cudf/types.hpp>
#include <cudf/groupby.hpp>

#include <rmm/thrust_rmm_allocator.h>


namespace cudf {
namespace groupby {
namespace sort {
namespace detail { 

/**
 * @brief Helper class for computing sort-based groupby
 * 
 * This class serves the purpose of sorting the keys and values and provides
 * building blocks for aggregations. It can provide:
 * 1. On-demand grouping and sorting of a value column based on `keys`
 *   which is provided at construction
 * 2. Group offsets: starting offsets of all groups in sorted key table
 * 3. Group valid sizes: The number of valid values in each group in a sorted
 *   value column
 */
struct helper {
  using index_vector = rmm::device_vector<gdf_size_type>;
  using bitmask_vector = rmm::device_vector<bit_mask::bit_mask_t>;
  using gdf_col_pointer = std::unique_ptr<gdf_column, std::function<void(gdf_column*)>>;
  using index_vec_pointer = std::unique_ptr<index_vector>;
  using bitmask_vec_pointer = std::unique_ptr<bitmask_vector>;
  using null_order = cudf::groupby::sort::null_order;


  /**
   * @brief Construct a new helper object
   * 
   * If `include_nulls == false`, then any row in `keys` containing a null value
   * will effectively be discarded. I.e., any values corresponding to discarded
   * rows in `keys` will not contribute to any aggregation. 
   *
   * @param keys table to group by
   * @param include_nulls whether to include null keys in groupby
   * @param null_sort_behavior whether to put nulls before valid values or after
   * @param keys_pre_sorted if the keys are already sorted
   * @param stream used for all the computation in this helper object
   */
  helper(cudf::table const& keys, bool include_nulls = false,
          null_order null_sort_behavior = null_order::AFTER,
          bool keys_pre_sorted = false,
          cudaStream_t stream = 0)
  : _keys(keys)
  , _num_keys(-1)
  , _include_nulls(include_nulls)
  , _null_sort_behavior(null_sort_behavior)
  , _keys_pre_sorted(keys_pre_sorted)
  , _stream(stream)
  {};

  ~helper() = default;
  helper(helper const&) = delete;
  helper& operator=(helper const&) = delete;
  helper(helper&&) = default;
  helper& operator=(helper&&) = default;

  /**
   * @brief Groups a column of values according to `keys` and sorts within each group.
   * 
   * Groups the @p values where the groups are dictated by key table
   * and each group is sorted in ascending order, with NULL elements positioned at the end of each group. Calculates and
   * returns the number of valid values within each group. 
   * 
   * @throws cudf::logic_error if `values.size != keys.num_rows()`
   * 
   * @param values The value column to group and sort
   * @return the sorted and grouped column and per-group valid count
   */
  std::pair<gdf_column, rmm::device_vector<gdf_size_type> >
  sort_values(gdf_column const& values);

  /**
   * @brief Get a table of sorted unique keys
   * 
   * @return a new table in which each row is a unique row in the sorted key table.
   */
  cudf::table unique_keys();

  /**
   * @brief Get the number of groups in `keys`
   * 
   */
  gdf_size_type num_groups() { return group_offsets().size(); }

  /**
   * @brief Return the effective number of keys
   * 
   * When include_nulls = true, returned value is same as `keys.num_rows()`
   * When include_nulls = false, returned value is the number of rows in `keys`
   *  in which no element is null
   */
  gdf_size_type num_keys();

  /**
   * @brief Get the sorted order of `keys`.
   *
   * Gathering `keys` by sort order indices will produce the sorted key table.
   * 
   * Computes and stores the key sorted order on first invocation, and returns the
   * stored order on subsequent calls.
   * 
   * @return the sort order indices for `keys`.
   */
  gdf_column const& key_sort_order();

  /**
   * @brief Get each group's offset into the sorted order of `keys`. 
   * 
   * Computes and stores the group offsets on first invocation and returns
   * the stored group offsets on subsequent calls.
   * 
   * @return vector of offsets of the starting point of each group in the sorted key table
   */
  index_vector const& group_offsets();

  /**
   * @brief Get the group labels corresponding to the sorted order of `keys`. 
   * 
   * Each group is assigned a unique numerical "label" in `[0, 1, 2, ... , num_groups() - 1, num_groups())`.
   * For a row in sorted `keys`, its corresponding group label indicates which group it belongs to. 
   * 
   * Computes and stores labels on first invocation and returns stored labels on
   * subsequent calls.
   * 
   * @return vector of group labels for each row in the sorted key column
   */
  index_vector const& group_labels();

 private:
  /**
   * @brief Get the group labels for unsorted keys
   * 
   * Returns the group label for every row in the original `keys` table. For a given unique key row,
   * it's group label is equivalent to what is returned by `group_labels()`. However, 
   * if a row contains a null value, and `include_nulls == false`, then it's label is NULL. 
   * 
   * Computes and stores unsorted labels on first invocation and returns stored
   * labels on subsequent calls.
   * 
   * @return A nullable column of `GDF_INT32` containing group labels in the order of the unsorted key table
   */
  gdf_column const& unsorted_keys_labels();

  /**
   * @brief Get the row bitmask for the `keys`
   *
   * Computes a bitmask corresponding to the rows of `keys` where if bit `i` is zero,
   * then row `i` contains one or more null values. If bit `i` is one, then row `i` does not 
   * contain null values. 
   *
   * 
   * Computes and stores bitmask on first invocation and returns stored bitmask on
   * subsequent calls.
   */
  bitmask_vector& keys_row_bitmask();

 private:

  gdf_col_pointer     _key_sorted_order;
  gdf_col_pointer     _unsorted_keys_labels;
  cudf::table const&  _keys;

  index_vec_pointer   _group_offsets;
  index_vec_pointer   _group_labels;

  gdf_size_type       _num_keys;
  bool                _keys_pre_sorted;
  bool                _include_nulls;
  null_order          _null_sort_behavior;

  cudaStream_t        _stream;

  bitmask_vec_pointer _keys_row_bitmask;
};

}  // namespace detail
}  // namespace sort
}  // namespace groupby
}  // namespace cudf
