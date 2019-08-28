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
 * 1. On-demand grouping and sorting of a value column based on the key table
 *   which is provided at construction
 * 2. Group indices: starting indices of all groups in sorted key table
 * 3. Group valid sizes: The number of valid values in each group in a sorted
 *   value column
 */
struct groupby {
  using index_vector = rmm::device_vector<gdf_size_type>;
  using gdf_col_pointer = std::unique_ptr<gdf_column, std::function<void(gdf_column*)>>;
  using index_vec_pointer = std::unique_ptr<rmm::device_vector<gdf_size_type>>;

  groupby(cudf::table const& key_table, bool include_nulls = false)
  : _key_table(key_table)
  , _num_keys(key_table.num_rows())
  , _include_nulls(include_nulls)
  {};

  ~groupby() {
    if (_key_sorted_order)
      gdf_column_free(_key_sorted_order.get());
    if (_unsorted_labels)
      gdf_column_free(_unsorted_labels.get());
  }

  /**
   * @brief Group and sort a column of values
   * values within each group
   * 
   * Sorts and groups the @p val_col where the groups are dictated by key table
   * and the elements are sorted ascending within the groups. Calculates and
   * returns the number of valid values within each group.
   * 
   * @param val_col The value column to group and sort
   * @return the sorted and grouped column and per-group valid count
   */
  std::pair<gdf_column, rmm::device_vector<gdf_size_type> >
  sort_values(gdf_column const& val_col);

  /**
   * @brief Get a table of sorted unique keys
   * 
   * @return a new table in which each row is a unique row in the sorted key table.
   */
  cudf::table unique_keys();

  /**
   * @brief Get the number of groups in the key table
   * 
   */
  gdf_size_type num_groups() const { return group_indices().size(); }

  /**
   * @brief Get the key sorted order.
   * 
   * @return the sort order indices for the key table.
   *
   * Gathering the key table by sort order indices will produce the sorted key table.
   * 
   * Computes and stores the key sorted order on first invocation, and returns the
   * stored order on subsequent calls.
   */
  gdf_column const& key_sort_order();

  /**
   * @brief Get the group indices.
   * 
   * @return vector of indices of the starting point of each group in the sorted key table
   * 
   * Computes and stores the group indices on first invocation and returns
   * the stored group indices on subsequent calls.
   */
  index_vector const& group_indices();

  /**
   * @brief Get the group labels
   * 
   * @return vector of group ID for each row in the sorted key column
   * 
   * Computes and stores labels on first invocation and returns stored labels on subsequent calls.
   */
  index_vector const& group_labels();

  /**
   * @brief Get the unsorted labels
   * 
   * @return column of group labels in the order of the unsorted key table
   * 
   * For each row in the key table, the unsorted label is the group it would belong to after sorting.
   * contains the group it would belong to, after sorting
   * 
   * Computes and stores unsorted labels on first invocation and returns stored labels on subsequent calls.
   */
  gdf_column const& unsorted_labels();

 private:

  gdf_col_pointer     _key_sorted_order;
  gdf_col_pointer     _unsorted_labels;
  cudf::table const&  _key_table;

  index_vec_pointer   _group_ids;
  index_vec_pointer   _group_labels;

  gdf_size_type       _num_keys;
  bool                _include_nulls;

};

} // namespace detail
  
} // namespace cudf
