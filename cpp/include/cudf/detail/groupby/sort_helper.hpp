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
 * 1. On-demand grouping and sorting of a value column based on `keys`
 *   which is provided at construction
 * 2. Group offsets: starting offsets of all groups in sorted key table
 * 3. Group valid sizes: The number of valid values in each group in a sorted
 *   value column
 */
struct helper {
  using index_vector = rmm::device_vector<size_type>;
  using bitmask_vector = rmm::device_vector<bitmask_type>;
  using column_ptr = std::unique_ptr<column>;
  using index_vector_ptr = std::unique_ptr<index_vector>;
  using bitmask_vector_ptr = std::unique_ptr<bitmask_vector>;


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
  helper(table_view const& keys, bool include_nulls = false,
          std::vector<null_order> null_sort_order = {},
          bool keys_pre_sorted = false)
  : _keys(keys)
  , _num_keys(-1)
  , _include_nulls(include_nulls)
  , _keys_pre_sorted(keys_pre_sorted)
  {
    if (keys_pre_sorted and
        not include_nulls and
        std::any_of(null_sort_order.begin(), null_sort_order.end(), 
          [] (null_order order) { return order == null_order::BEFORE;})
       )
    {
      _keys_pre_sorted = false;
    }
  };

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
  std::pair<std::unique_ptr<column>, index_vector >
  sorted_values_and_num_valids(column_view const& values, 
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
    cudaStream_t stream = 0);

  std::unique_ptr<column> sorted_values(column_view const& values, 
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
    cudaStream_t stream = 0);

  // TODO (dm): implement
  std::pair<std::unique_ptr<column>, index_vector >
  grouped_values_and_num_valids(column_view const& values, 
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
    cudaStream_t stream = 0);

  // TODO (dm): implement
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
   * 
   */
  size_type num_groups() { return group_offsets().size(); }

  /**
   * @brief Return the effective number of keys
   * 
   * When include_nulls = true, returned value is same as `keys.num_rows()`
   * When include_nulls = false, returned value is the number of rows in `keys`
   *  in which no element is null
   */
  size_type num_keys();

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
   * Each group is assigned a unique numerical "label" in `[0, 1, 2, ... , num_groups() - 1, num_groups())`.
   * For a row in sorted `keys`, its corresponding group label indicates which group it belongs to. 
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
   * Returns the group label for every row in the original `keys` table. For a given unique key row,
   * it's group label is equivalent to what is returned by `group_labels()`. However, 
   * if a row contains a null value, and `include_nulls == false`, then it's label is NULL. 
   * 
   * Computes and stores unsorted labels on first invocation and returns stored
   * labels on subsequent calls.
   * 
   * @return A nullable column of `GDF_INT32` containing group labels in the order of the unsorted key table
   */
  column_view unsorted_keys_labels(cudaStream_t stream = 0);

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
  column_view keys_bitmask_column(cudaStream_t stream = 0);

  index_vector count_valids_in_groups(
    column const& grouped_values,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
    cudaStream_t stream = 0);

 private:

  column_ptr          _key_sorted_order;
  column_ptr          _unsorted_keys_labels;
  column_ptr          _keys_bitmask_column;
  table_view          _keys;

  index_vector_ptr    _group_offsets;
  index_vector_ptr    _group_labels;

  size_type           _num_keys;
  bool                _keys_pre_sorted;
  bool                _include_nulls;
};

}  // namespace sort
}  // namespace detail
}  // namespace groupby
}  // namespace experimental
}  // namespace cudf
