/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cudf/column/column.hpp>
#include <cudf/lists/lists_column_view.hpp>

#include <rmm/mr/device/device_memory_resource.hpp>

namespace cudf::lists {
/**
 * @addtogroup set_operations
 * @{
 * @file
 */

/**
 * @brief Check if lists at each row of the given lists columns overlap.
 *
 * Given two input lists columns, each list row in one column is checked if it has any common
 * elements with the corresponding row of the other column.
 *
 * A null input row in any of the input lists columns will result in a null output row.
 *
 * @throw cudf::logic_error if the input lists columns have different sizes.
 * @throw cudf::logic_error if children of the input lists columns have different data types.
 *
 * Example:
 * @code{.pseudo}
 * lhs    = { {0, 1, 2}, {1, 2, 3}, null,         {4, null, 5} }
 * rhs    = { {1, 2, 3}, {4, 5},    {null, 7, 8}, {null, null} }
 * result = { true, false, null, false }
 * @endcode
 *
 * @param lhs The input lists column for one side
 * @param rhs The input lists column for the other side
 * @param nulls_equal Flag to specify whether null elements should be considered as equal, default
 *        to be `UNEQUAL` which means only non-null elements are checked for overlapping
 * @param nans_equal Flag to specify whether floating-point NaNs should be considered as equal
 * @param mr Device memory resource used to allocate the returned object
 * @return A column of type BOOL containing the check results
 */
std::unique_ptr<column> have_overlap(
  lists_column_view const& lhs,
  lists_column_view const& rhs,
  null_equality nulls_equal           = null_equality::UNEQUAL,
  nan_equality nans_equal             = nan_equality::ALL_EQUAL,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Find the intersection without duplicate between lists at each row of the given lists
 *        columns.
 *
 * Given two input lists columns, an output lists column is created in a way such that each row
 * contains the common elements (without duplicates) of the lists from the input columns at the
 * corresponding row.
 *
 * A null input row in any of the input lists columns will result in a null output row.
 *
 * @throw cudf::logic_error if the input lists columns have different sizes.
 * @throw cudf::logic_error if children of the input lists columns have different data types.
 *
 * Example:
 * @code{.pseudo}
 * lhs    = { {2, 1, 2}, {1, 2, 3}, null,         {4, null, 5} }
 * rhs    = { {1, 2, 3}, {4, 5},    {null, 7, 8}, {null, null} }
 * result = { {1, 2}, {}, null, {null} }
 * @endcode
 *
 * @param lhs The input lists column for one side
 * @param rhs The input lists column for the other side
 * @param nulls_equal Flag to specify whether null elements should be considered as equal
 * @param nans_equal Flag to specify whether floating-point NaNs should be considered as equal
 * @param mr Device memory resource used to allocate the returned object
 * @return A lists column containing the intersection results
 */
std::unique_ptr<column> set_intersect(
  lists_column_view const& lhs,
  lists_column_view const& rhs,
  null_equality nulls_equal           = null_equality::EQUAL,
  nan_equality nans_equal             = nan_equality::ALL_EQUAL,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Find the union (without duplicate) between lists at each row of the given lists
 *        columns.
 *
 * Given two input lists columns, an output lists column is created by concatenating each pair of
 * lists at the same row from the input columns then removing duplicates.
 *
 * each pair of lists at the same row is concatenated then the result
 * is extracted without duplicates to the corresponding row in the output lists column.
 *
 * A null input row in any of the input lists columns will result in a null output row.
 *
 * @throw cudf::logic_error if the input lists columns have different sizes.
 * @throw cudf::logic_error if children of the input lists columns have different data types.
 *
 * Example:
 * @code{.pseudo}
 * lhs    = { {2, 1, 2}, {1, 2, 3}, null,         {4, null, 5} }
 * rhs    = { {1, 2, 3}, {4, 5},    {null, 7, 8}, {null, null} }
 * result = { {1, 2, 3}, {1, 2, 3, 4, 5}, null, {4, null, 5} }
 * @endcode
 *
 * @param lhs The input lists column for one side
 * @param rhs The input lists column for the other side
 * @param nulls_equal Flag to specify whether null elements should be considered as equal
 * @param nans_equal Flag to specify whether floating-point NaNs should be considered as equal
 * @param mr Device memory resource used to allocate the returned object
 * @return A lists column containing the union results
 */
std::unique_ptr<column> set_union(
  lists_column_view const& lhs,
  lists_column_view const& rhs,
  null_equality nulls_equal           = null_equality::EQUAL,
  nan_equality nans_equal             = nan_equality::ALL_EQUAL,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Find the elements (without duplicates) from each list of the left column that do not exist
 *        in the corresponding list of the right column.
 *
 * Given two input lists columns, an output lists column is created by finding the difference of
 * lists in the left column against the corresponding lists in the right column. Specifically, find
 * the elements (without duplicates) in each list of the left column such that they do not exist in
 * the corresponding list of the right column.
 *
 * A null input row in any of the input lists columns will result in a null output row.
 *
 * @throw cudf::logic_error if the input lists columns have different sizes.
 * @throw cudf::logic_error if children of the input lists columns have different data types.
 *
 * Example:
 * @code{.pseudo}
 * lhs    = { {2, 1, 2}, {1, 2, 3}, null,         {4, null, 5} }
 * rhs    = { {1, 2, 3}, {4, 5},    {null, 7, 8}, {null, null} }
 * result = { {}, {1, 2, 3}, null, {4, 5} }
 * @endcode
 *
 * @param lhs The input lists column containing the searching elements
 * @param rhs The input lists column for checking element existence
 * @param nulls_equal Flag to specify whether null elements should be considered as equal
 * @param nans_equal Flag to specify whether floating-point NaNs should be considered as equal
 * @param mr Device memory resource used to allocate the returned object
 * @return A lists column containing the difference results
 */
std::unique_ptr<column> set_difference(
  lists_column_view const& lhs,
  lists_column_view const& rhs,
  null_equality nulls_equal           = null_equality::EQUAL,
  nan_equality nans_equal             = nan_equality::ALL_EQUAL,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/** @} */  // end of group
}  // namespace cudf::lists
