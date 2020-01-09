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

#include <cudf/table/table_view.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>

#include <rmm/thrust_rmm_allocator.h>


namespace cudf {
namespace experimental {
namespace groupby {
namespace detail { 
namespace sort {

// TODO (dm): update documentation for whole file
/**
 * @brief Helper class for computing sort-based groupby
 * 
 * This class serves the purpose of sorting the keys and values and provides
 * building blocks for aggregations. It can provide:
 * 1. On-demand grouping or sorting of a value column based on `keys`
 *   which is provided at construction
 * 2. Group offsets: starting offsets of all groups in sorted key table
 * 3. Group valid sizes: The number of valid values in each group in a sorted
 *   value column
 */
struct sort_groupby_helper {
  using index_vector = rmm::device_vector<size_type>;
  using bitmask_vector = rmm::device_vector<bitmask_type>;
  using column_ptr = std::unique_ptr<column>;
  using index_vector_ptr = std::unique_ptr<index_vector>;
  using bitmask_vector_ptr = std::unique_ptr<bitmask_vector>;


  /**
   * @brief Construct a new helper object
   * 
   * If `ignore_null_keys == true`, then any row in `keys` containing a null
   * value will effectively be discarded. I.e., any values corresponding to
   * discarded rows in `keys` will not contribute to any aggregation. 
   *
   * @param keys table to group by
   * @param ignore_null_keys Ignore rows in keys with nulls
   * @param keys_pre_sorted Indicate if the keys are already sorted. Enables
   *                        optimizations to help skip re-sorting keys.
   */
  sort_groupby_helper(table_view const& keys, bool ignore_null_keys = true,
                      bool keys_pre_sorted = false)
  : _keys(keys),
    _num_keys(-1),
    _ignore_null_keys(ignore_null_keys),
    _keys_pre_sorted(keys_pre_sorted)
  {
    if (keys_pre_sorted and
        ignore_null_keys and
        has_nulls(keys))
    {
      _keys_pre_sorted = false;
    }
  };

  ~sort_groupby_helper() = default;
  sort_groupby_helper(sort_groupby_helper const&) = delete;
  sort_groupby_helper& operator=(sort_groupby_helper const&) = delete;
  sort_groupby_helper(sort_groupby_helper&&) = default;
  sort_groupby_helper& operator=(sort_groupby_helper&&) = default;

  /**
   * @brief Groups a column of values according to `keys` and sorts within each
   *  group.
   * 
   * Groups the @p values where the groups are dictated by key table and each 
   * group is sorted in ascending order, with NULL elements positioned at the 
   * end of each group.
   * 
   * @throw cudf::logic_error if `values.size() != keys.num_rows()`
   * 
   * @param values The value column to group and sort
   * @return the sorted and grouped column
   */
  std::unique_ptr<column> sorted_values(column_view const& values, 
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
    cudaStream_t stream = 0);

  /**
   * @brief Groups a column of values according to `keys`
   * 
   * The order of values within each group is undefined. 
   * 
   * @throw cudf::logic_error if `values.size() != keys.num_rows()`
   * 
   * @param values The value column to group
   * @return the grouped column
   */
  std::unique_ptr<column> grouped_values(column_view const& values, 
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
    cudaStream_t stream = 0);

  /**
   * @brief Get a table of sorted unique keys
   * 
   * @return a new table in which each row is a unique row in the sorted key table.
   */
  std::unique_ptr<table> unique_keys(
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
    cudaStream_t stream = 0);

  /**
   * @brief Get the number of groups in `keys`
   */
  size_type num_groups() { return group_offsets().size(); }

  /**
   * @brief Return the effective number of keys
   * 
   * When ignore_null_keys = false, returned value is same as `keys.num_rows()`
   * When ignore_null_keys = true, returned value is the number of rows in `keys`
   *  in which no element is null
   */
  size_type num_keys(cudaStream_t stream = 0);

  /**
   * @brief Get the sorted order of `keys`.
   *
   * Gathering `keys` by sort order indices will produce the sorted key table.
   * 
   * Computes and stores the key sorted order on first invocation, and returns
   * the stored order on subsequent calls.
   * 
   * @return the sort order indices for `keys`.
   */
  column_view key_sort_order(cudaStream_t stream = 0);

  /**
   * @brief Get each group's offset into the sorted order of `keys`. 
   * 
   * Computes and stores the group offsets on first invocation and returns
   * the stored group offsets on subsequent calls.
   * 
   * @return vector of offsets of the starting point of each group in the sorted key table
   */
  index_vector const& group_offsets(cudaStream_t stream = 0);

  /**
   * @brief Get the group labels corresponding to the sorted order of `keys`. 
   * 
   * Each group is assigned a unique numerical "label" in 
   * `[0, 1, 2, ... , num_groups() - 1, num_groups())`.
   * For a row in sorted `keys`, its corresponding group label indicates which
   * group it belongs to. 
   * 
   * Computes and stores labels on first invocation and returns stored labels on
   * subsequent calls.
   * 
   * @return vector of group labels for each row in the sorted key column
   */
  index_vector const& group_labels(cudaStream_t stream = 0);

 private:
  /**
   * @brief Get the group labels for unsorted keys
   * 
   * Returns the group label for every row in the original `keys` table. For a
   * given unique key row, its group label is equivalent to what is returned by
   * `group_labels()`. However, if a row contains a null value, and
   * `ignore_null_keys == true`, then its label is NULL. 
   * 
   * Computes and stores unsorted labels on first invocation and returns stored
   * labels on subsequent calls.
   * 
   * @return A nullable column of `INT32` containing group labels in the order 
   *         of the unsorted key table
   */
  column_view unsorted_keys_labels(cudaStream_t stream = 0);

  /**
   * @brief Get the column representing the row bitmask for the `keys`
   *
   * Computes a bitmask corresponding to the rows of `keys` where if bit `i` is
   * zero, then row `i` contains one or more null values. If bit `i` is one, 
   * then row `i` does not contain null values. This bitmask is added as null
   * mask of a column of type `INT8` where all the data values are the same and
   * the elements differ only in validity.
   * 
   * Computes and stores bitmask on first invocation and returns stored column
   * on subsequent calls.
   */
  column_view keys_bitmask_column(cudaStream_t stream = 0);

 private:

  column_ptr          _key_sorted_order;      ///< Indices to produce _keys in sorted order
  column_ptr          _unsorted_keys_labels;  ///< Group labels for unsorted _keys
  column_ptr          _keys_bitmask_column;   ///< Column representing rows with one or more nulls values
  table_view          _keys;                  ///< Input keys to sort by

  index_vector_ptr    _group_offsets;         ///< Indices into sorted _keys indicating starting index of each groups
  index_vector_ptr    _group_labels;          ///< Group labels for sorted _keys

  size_type           _num_keys;              ///< Number of effective rows in _keys (adjusted for _ignore_null_keys)
  bool                _keys_pre_sorted;       ///< Whether _keys are pre-sorted
  bool                _ignore_null_keys;      ///< Whether to use rows with nulls in _keys for grouping
};

}  // namespace sort
}  // namespace detail
}  // namespace groupby
}  // namespace experimental
}  // namespace cudf
