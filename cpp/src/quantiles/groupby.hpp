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

namespace {

template <typename T>
struct optional {
  optional() : contains_value(false) {};
  optional(T value) : value(value), contains_value(true) {};

  operator bool() { return contains_value; }
  T* operator&() { return &value; }

 private:
  T value;
  bool contains_value;
};

} // namespace anonymous


namespace cudf {

namespace detail {

/**
 * @brief Helper class for computing sort-based groupby
 * 
 * This class serves the purpose of sorting the keys and values and provides
 * helping building blocks for aggregations. It can provide:
 * 1. On demand grouping and sorting of a value column based on the key table
 *   which is provided at construction
 * 2. Group indices: starting indices of all groups in sorted key table
 * 3. Group valid sizes: The number of valid values in each group in a sorted
 *   value column
 */
struct groupby {
  using index_vector = rmm::device_vector<gdf_size_type>;

  groupby(cudf::table const& key_table, bool include_nulls = false)
  : _key_table(key_table)
  , _num_keys(key_table.num_rows())
  , _include_nulls(include_nulls)
  {};

  ~groupby() {
    gdf_column_free(&_key_sorted_order);
    gdf_column_free(&_unsorted_labels);
  }

  /**
   * @brief Returns a grouped and sorted values column and a count of valid
   * values within each group
   * 
   * Sorts and groups the @p val_col where the groups are dictated by key table
   * and the elements are sorted ascending within the groups. Calculates the
   * number of valid values within each group and also returns this
   * 
   * @param val_col The value column to group and sort
   * @return std::pair<gdf_column, rmm::device_vector<gdf_size_type> > 
   *  The sorted and grouped column, and per group valid count
   */
  std::pair<gdf_column, rmm::device_vector<gdf_size_type> >
  sort_values(gdf_column const& val_col);

  /**
   * @brief Returns a table of sorted unique keys
   * 
   * The result contains a new table where each row is a unique row in the
   * sorted key table
   */
  cudf::table unique_keys();

  /**
   * @brief Returns the number of groups in the key table
   * 
   */
  gdf_size_type num_groups() { return group_indices().size(); }

  /**
   * @brief Get the member _key_sorted_order.
   * 
   * This member contains the sort order indices for _key_table. Gathering the
   * _key_table by _key_sorted_order would produce the sorted key table
   * 
   * This method will compute set the uninitialized _key_sorted_order on first 
   * call and return the precomputed value on every subsequent call
   */
  gdf_column const& key_sort_order();

  /**
   * @brief Get the member _group_ids.
   * 
   * _group_ids contains the indices for the starting points of each group in
   * the sorted key table
   * 
   * This method will compute set the uninitialized _group_ids on first 
   * call and return the precomputed value on every subsequent call
   */
  index_vector const& group_indices();

  /**
   * @brief Get the member _group_labels
   * 
   * _group_labels contains a value for each row in the sorted key column
   * signifying which group in _group_ids it belongs to
   * 
   * This method will compute set the uninitialized _key_sorted_order on first 
   * call and return the precomputed value on every subsequent call
   */
  index_vector const& group_labels();

  /**
   * @brief Get the member _unsorted_labels
   * 
   * _unsorted_labels contains the group labels but in the order of the 
   * unsorted _key_table so that for each row in _key_table, _unsorted_labels
   * contains the group it would belong to, after sorting
   * 
   * This method will compute set the uninitialized _key_sorted_order on first 
   * call and return the precomputed value on every subsequent call
   */
  gdf_column const& unsorted_labels();

 private:

  optional<gdf_column>    _key_sorted_order;
  optional<gdf_column>    _unsorted_labels;
  cudf::table const&      _key_table;

  optional<index_vector>  _group_ids;
  optional<index_vector>  _group_labels;

  gdf_size_type           _num_keys;
  bool                    _include_nulls;

};

} // namespace detail
  
} // namespace cudf
