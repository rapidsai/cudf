/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/mr/device/per_device_resource.hpp>

#include <vector>

namespace cudf {
/**
 * @addtogroup column_search
 * @{
 * @file
 * @brief Column APIs for lower_bound, upper_bound, and contains
 */

/**
 * @brief Find smallest indices in a sorted table where values should be inserted to maintain order.
 *
 * For each row in `needles`, find the first index in `haystack` where inserting the row still
 * maintains its sort order.
 *
 * @code{.pseudo}
 * Example:
 *
 *  Single column:
 *      idx        0   1   2   3   4
 *   haystack = { 10, 20, 20, 30, 50 }
 *   needles  = { 20 }
 *   result   = {  1 }
 *
 *  Multi Column:
 *      idx          0    1    2    3    4
 *   haystack = {{  10,  20,  20,  20,  20 },
 *               { 5.0,  .5,  .5,  .7,  .7 },
 *               {  90,  77,  78,  61,  61 }}
 *   needles  = {{ 20 },
 *               { .7 },
 *               { 61 }}
 *   result   = {   3 }
 * @endcode
 *
 * @param haystack The table containing search space
 * @param needles Values for which to find the insert locations in the search space
 * @param column_order Vector of column sort order
 * @param null_precedence Vector of null_precedence enums needles
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return A non-nullable column of cudf::size_type elements containing the insertion points
 */
std::unique_ptr<column> lower_bound(
  table_view const& haystack,
  table_view const& needles,
  std::vector<order> const& column_order,
  std::vector<null_order> const& null_precedence,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Find largest indices in a sorted table where values should be inserted to maintain order.
 *
 * For each row in `needles`, find the last index in `haystack` where inserting the row still
 * maintains its sort order.
 *
 * @code{.pseudo}
 * Example:
 *
 *  Single Column:
 *      idx        0   1   2   3   4
 *   haystack = { 10, 20, 20, 30, 50 }
 *   needles  = { 20 }
 *   result   = {  3 }
 *
 *  Multi Column:
 *      idx          0    1    2    3    4
 *   haystack = {{  10,  20,  20,  20,  20 },
 *               { 5.0,  .5,  .5,  .7,  .7 },
 *               {  90,  77,  78,  61,  61 }}
 *   needles  = {{ 20 },
 *               { .7 },
 *               { 61 }}
 *   result =     { 5 }
 * @endcode
 *
 * @param haystack The table containing search space
 * @param needles Values for which to find the insert locations in the search space
 * @param column_order Vector of column sort order
 * @param null_precedence Vector of null_precedence enums needles
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return A non-nullable column of cudf::size_type elements containing the insertion points
 */
std::unique_ptr<column> upper_bound(
  table_view const& haystack,
  table_view const& needles,
  std::vector<order> const& column_order,
  std::vector<null_order> const& null_precedence,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Check if the given `needle` value exists in the `haystack` column.
 *
 * @throws cudf::logic_error If `haystack.type() != needle.type()`.
 *
 * @code{.pseudo}
 *  Single Column:
 *   idx           0   1   2   3   4
 *   haystack = { 10, 20, 20, 30, 50 }
 *   needle   = { 20 }
 *   result   = true
 * @endcode
 *
 * @param haystack The column containing search space
 * @param needle A scalar value to check for existence in the search space
 * @return true if the given `needle` value exists in the `haystack` column
 */
bool contains(column_view const& haystack, scalar const& needle);

/**
 * @brief Check if the given `needles` values exists in the `haystack` column.
 *
 * The new column will have type BOOL and have the same size and null mask as the input `needles`
 * column. That is, any null row in the `needles` column will result in a nul row in the output
 * column.
 *
 * @throws cudf::logic_error If `haystack.type() != needles.type()`
 *
 * @code{.pseudo}
 *   haystack = { 10, 20, 30, 40, 50 }
 *   needles  = { 20, 40, 60, 80 }
 *   result   = { true, true, false, false }
 * @endcode
 *
 * @param haystack The column containing search space
 * @param needles A column of values to check for existence in the search space
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return A BOOL column indicating if each element in `needles` exists in the search space
 */
std::unique_ptr<column> contains(
  column_view const& haystack,
  column_view const& needles,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/** @} */  // end of group
}  // namespace cudf
