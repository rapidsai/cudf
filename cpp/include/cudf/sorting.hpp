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

#include <cudf/types.hpp>

#include <memory>
#include <vector>

namespace cudf {
namespace experimental {

/**---------------------------------------------------------------------------*
 * @brief Computes the row indices that would produce `input`  in a
 * lexicographical sorted order.
 *
 * @param input The table to sort
 * @param column_order The desired sort order for each column. Size must be
 * equal to `input.num_columns()` or empty. If empty, all columns will be sorted
 * in ascending order.
 * @param null_precedence The desired order of null compared to other elements
 * for each column.  Size must be equal to `input.num_columns()` or empty.
 * If empty, all columns will be sorted in `null_order::BEFORE`.
 * @return std::unique_ptr<column> A non-nullable column of INT32 elements
 * containing the permuted row indices of `input` if it were sorted
 *---------------------------------------------------------------------------**/
std::unique_ptr<column> sorted_order(
    table_view input, std::vector<order> const& column_order = {},
    std::vector<null_order> const& null_precedence = {},
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**---------------------------------------------------------------------------*
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
 *---------------------------------------------------------------------------**/
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
 * @param mr The device memory resource used to allocate the returned table
 * @return New table containing the desired sorted order of `input`
 */
std::unique_ptr<table> sort(
    table_view input, std::vector<order> const& column_order = {},
    std::vector<null_order> const& null_precedence = {},
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());


/**
 * @brief Performs a key-value sort.
 *
 * Creates a new table that reorders the rows of `values` according to the
 * lexicographic ordering of the rows of `keys`.
 *
 * @throws `cudf::logic_error` if `values.num_rows() != keys.num_rows()`.
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
 * @param mr The device memory resource used to allocate the returned table
 * @return The reordering of `values` determined by the lexicographic order of
 * the rows of `keys`.
 */
std::unique_ptr<table> sort_by_key(
    table_view const& values, table_view const& keys,
    std::vector<order> const& column_order = {},
    std::vector<null_order> const& null_precedence = {},
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

}  // namespace experimental
}  // namespace cudf
