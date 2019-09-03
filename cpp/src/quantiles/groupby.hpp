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

#include <rmm/thrust_rmm_allocator.h>


namespace cudf {

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
struct groupby {
  using index_vector = rmm::device_vector<gdf_size_type>;
  using gdf_col_pointer = std::unique_ptr<gdf_column, std::function<void(gdf_column*)>>;
  using index_vec_pointer = std::unique_ptr<rmm::device_vector<gdf_size_type>>;

  groupby(cudf::table const& keys, bool include_nulls = false,
          cudaStream_t stream = 0)
  : _keys(keys)
  , _num_keys(keys.num_rows())
  , _include_nulls(include_nulls)
  , _stream(stream)
  {};

  ~groupby() {}

  /**
   * @brief Group and sort a column of values
   * 
   * Sorts and groups the @p values where the groups are dictated by key table
   * and the elements are sorted ascending within the groups. Calculates and
   * returns the number of valid values within each group. 
   * 
   * @note Size of @p values should be equal to number of rows in keys
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
  gdf_size_type num_keys() { return _num_keys; }

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
   * @brief Get the group offsets.
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
   * Each group is assigned a unique numerical "label" in `[0, 1, 2, ... , num_groups() - 1, num_groups())`. For 
   * a row in `keys`, its group label indicates which group it belongs to. 
   * 
   * Computes and stores labels on first invocation and returns stored labels on
   * subsequent calls.
   * 
   * @return vector of group labels for each row in the sorted key column
   */
  index_vector const& group_labels();

  /**
   * @brief Get the unsorted labels
   * 
   * For each row in `keys`, the unsorted label is the group it would
   * belong to after sorting.
   * 
   * Computes and stores unsorted labels on first invocation and returns stored
   * labels on subsequent calls.
   * 
   * @return column of group labels in the order of the unsorted key table
   */
  gdf_column const& unsorted_labels();

 private:

  gdf_col_pointer     _key_sorted_order;
  gdf_col_pointer     _unsorted_labels;
  cudf::table const&  _keys;

  index_vec_pointer   _group_offsets;
  index_vec_pointer   _group_labels;

  gdf_size_type       _num_keys;
  bool                _include_nulls;

  cudaStream_t        _stream;

};

} // namespace detail
  
} // namespace cudf
