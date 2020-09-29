/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <cudf/types.hpp>

#include <memory>
#include <vector>

namespace cudf {

/**
 * @brief Tie-breaker method to use for ranking the column.
 *
 * @ingroup column_sort
 */
enum class rank_method {
  FIRST,    ///< stable sort order ranking (no ties)
  AVERAGE,  ///< mean of first in the group
  MIN,      ///< min of first in the group
  MAX,      ///< max of first in the group
  DENSE     ///< rank always increases by 1 between groups
};

/**
 * @addtogroup column_sort
 * @{
 * @file
 * @brief Column APIs for sort and rank
 */

/**
 * @brief Computes the row indices that would produce `input` in a lexicographical sorted order.
 *
 * @param input The table to sort
 * @param column_order The desired sort order for each column. Size must be
 * equal to `input.num_columns()` or empty. If empty, all columns will be sorted
 * in ascending order.
 * @param null_precedence The desired order of null compared to other elements
 * for each column.  Size must be equal to `input.num_columns()` or empty.
 * If empty, all columns will be sorted in `null_order::BEFORE`.
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return A non-nullable column of `size_type` elements containing the permuted row indices of
 * `input` if it were sorted
 */
std::unique_ptr<column> sorted_order(
  table_view input,
  std::vector<order> const& column_order         = {},
  std::vector<null_order> const& null_precedence = {},
  rmm::mr::device_memory_resource* mr            = rmm::mr::get_current_device_resource());

/**
 * @brief Computes the row indices that would produce `input` in a stable
 * lexicographical sorted order.
 *
 * The order of equivalent elements is guaranteed to be preserved.
 *
 * @copydoc cudf::sorted_order
 */
std::unique_ptr<column> stable_sorted_order(
  table_view input,
  std::vector<order> const& column_order         = {},
  std::vector<null_order> const& null_precedence = {},
  rmm::mr::device_memory_resource* mr            = rmm::mr::get_current_device_resource());

/**
 * @brief Checks whether the rows of a `table` are sorted in a lexicographical
 *        order.
 *
 * @param[in] in                table whose rows need to be compared for ordering
 * @param[in] column_order      The expected sort order for each column. Size
 *                              must be equal to `in.num_columns()` or empty. If
 *                              empty, it is expected all columns are in
 *                              ascending order.
 * @param[in] null_precedence   The desired order of null compared to other
 *                              elements for each column. Size must be equal to
 *                              `input.num_columns()` or empty. If empty,
 *                              `null_order::BEFORE` is assumed for all columns.
 *
 * @returns bool                true if sorted as expected, false if not.
 */
bool is_sorted(cudf::table_view const& table,
               std::vector<order> const& column_order,
               std::vector<null_order> const& null_precedence);

/**
 * @brief Performs a lexicographic sort of the rows of a table
 *
 * @param input The table to sort
 * @param column_order The desired order for each column. Size must be
 * equal to `input.num_columns()` or empty. If empty, all columns are sorted in
 * ascending order.
 * @param null_precedence The desired order of a null element compared to other
 * elements for each column in `input`. Size must be equal to
 * `input.num_columns()` or empty. If empty, all columns will be sorted with
 * `null_order::BEFORE`.
 * @param mr Device memory resource used to allocate the returned table's device memory
 * @return New table containing the desired sorted order of `input`
 */
std::unique_ptr<table> sort(
  table_view input,
  std::vector<order> const& column_order         = {},
  std::vector<null_order> const& null_precedence = {},
  rmm::mr::device_memory_resource* mr            = rmm::mr::get_current_device_resource());

/**
 * @brief Performs a key-value sort.
 *
 * Creates a new table that reorders the rows of `values` according to the
 * lexicographic ordering of the rows of `keys`.
 *
 * @throws cudf::logic_error if `values.num_rows() != keys.num_rows()`.
 *
 * @param values The table to reorder
 * @param keys The table that determines the ordering
 * @param column_order The desired order for each column in `keys`. Size must be
 * equal to `input.num_columns()` or empty. If empty, all columns are sorted in
 * ascending order.
 * @param null_precedence The desired order of a null element compared to other
 * elements for each column in `keys`. Size must be equal to
 * `keys.num_columns()` or empty. If empty, all columns will be sorted with
 * `null_order::BEFORE`.
 * @param mr Device memory resource used to allocate the returned table's device memory
 * @return The reordering of `values` determined by the lexicographic order of
 * the rows of `keys`.
 */
std::unique_ptr<table> sort_by_key(
  table_view const& values,
  table_view const& keys,
  std::vector<order> const& column_order         = {},
  std::vector<null_order> const& null_precedence = {},
  rmm::mr::device_memory_resource* mr            = rmm::mr::get_current_device_resource());

/**
 * @brief Computes the ranks of input column in sorted order.
 *
 * Rank indicate the position of each element in the sorted column and rank
 * value starts from 1.
 *
 * @code{.pseudo}
 * input = { 3, 4, 5, 4, 1, 2}
 * Result for different rank_method are
 * FIRST    = {3, 4, 6, 5, 1, 2}
 * AVERAGE  = {3, 4.5, 6, 4.5, 1, 2}
 * MIN      = {3, 4, 6, 4, 1, 2}
 * MAX      = {3, 5, 6, 5, 1, 2}
 * DENSE    = {3, 4, 5, 4, 1, 2}
 * @endcode
 *
 * @param input The column to rank
 * @param method The ranking method used for tie breaking (same values).
 * @param column_order The desired sort order for ranking
 * @param null_handling  flag to include nulls during ranking. If nulls are not
 * included, corresponding rank will be null.
 * @param null_precedence The desired order of null compared to other elements
 * for column
 * @param percentage flag to convert ranks to percentage in range (0,1}
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return std::unique_ptr<column> A column of containing the rank of the each
 * element of the column of `input`. The output column type will be `size_type`
 * column by default or else `double` when `method=rank_method::AVERAGE` or
 *`percentage=True`
 */
std::unique_ptr<column> rank(
  column_view const& input,
  rank_method method,
  order column_order,
  null_policy null_handling,
  null_order null_precedence,
  bool percentage,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/** @} */  // end of group
}  // namespace cudf
