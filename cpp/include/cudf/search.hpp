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

#include <cudf/column/column.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <vector>

namespace cudf {
/**
 * @addtogroup column_search
 * @{
 * @file
 * @brief Column APIs for lower_bound, upper_bound, and contains
 */

/**
 * @brief Find smallest indices in a sorted table where values should be
 *  inserted to maintain order
 *
 * For each row v in @p values, find the first index in @p t where
 *  inserting the row will maintain the sort order of @p t
 *
 * @code{.pseudo}
 * Example:
 *
 *  Single column:
 *      idx      0   1   2   3   4
 *   column = { 10, 20, 20, 30, 50 }
 *   values = { 20 }
 *   result = {  1 }
 *
 *  Multi Column:
 *      idx        0    1    2    3    4
 *   t      = {{  10,  20,  20,  20,  20 },
 *             { 5.0,  .5,  .5,  .7,  .7 },
 *             {  90,  77,  78,  61,  61 }}
 *   values = {{ 20 },
 *             { .7 },
 *             { 61 }}
 *   result =  {  3 }
 * @endcode
 *
 * @param t               Table to search
 * @param values          Find insert locations for these values
 * @param column_order    Vector of column sort order
 * @param null_precedence Vector of null_precedence enums values
 * @param mr              Device memory resource used to allocate the returned column's device
 * memory
 * @return A non-nullable column of cudf::size_type elements containing the insertion points.
 */
std::unique_ptr<column> lower_bound(
  table_view const& t,
  table_view const& values,
  std::vector<order> const& column_order,
  std::vector<null_order> const& null_precedence,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Find largest indices in a sorted table where values should be
 *  inserted to maintain order
 *
 * For each row v in @p values, find the last index in @p t where
 *  inserting the row will maintain the sort order of @p t
 *
 * @code{.pseudo}
 * Example:
 *
 *  Single Column:
 *      idx      0   1   2   3   4
 *   column = { 10, 20, 20, 30, 50 }
 *   values = { 20 }
 *   result = {  3 }
 *
 *  Multi Column:
 *      idx        0    1    2    3    4
 *   t      = {{  10,  20,  20,  20,  20 },
 *             { 5.0,  .5,  .5,  .7,  .7 },
 *             {  90,  77,  78,  61,  61 }}
 *   values = {{ 20 },
 *             { .7 },
 *             { 61 }}
 *   result =  {  5 }
 * @endcode
 *
 * @param column          Table to search
 * @param values          Find insert locations for these values
 * @param column_order    Vector of column sort order
 * @param null_precedence Vector of null_precedence enums values
 * @param mr              Device memory resource used to allocate the returned column's device
 * memory
 * @return A non-nullable column of cudf::size_type elements containing the insertion points.
 */
std::unique_ptr<column> upper_bound(
  table_view const& t,
  table_view const& values,
  std::vector<order> const& column_order,
  std::vector<null_order> const& null_precedence,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Find if the `value` is present in the `col`
 *
 * @throws cudf::logic_error
 * If `col.type() != values.type()`
 *
 * @code{.pseudo}
 *  Single Column:
 *      idx      0   1   2   3   4
 *      col = { 10, 20, 20, 30, 50 }
 *  Scalar:
 *   value = { 20 }
 *   result = true
 * @endcode
 *
 * @param col      A column object
 * @param value    A scalar value to search for in `col`
 *
 * @return bool    If `value` is found in `column` true, else false.
 */
bool contains(column_view const& col, scalar const& value);

/**
 * @brief  Returns a new column of type bool identifying for each element of @p haystack column,
 *         if that element is contained in @p needles column.
 *
 * The new column will have the same dimension and null status as the @p haystack column.  That is,
 * any element that is invalid in the @p haystack column will be invalid in the returned column.
 *
 * @throws cudf::logic_error
 * If `haystack.type() != needles.type()`
 *
 * @code{.pseudo}
 *   haystack = { 10, 20, 30, 40, 50 }
 *   needles  = { 20, 40, 60, 80 }
 *
 *   result = { false, true, false, true, false }
 * @endcode
 *
 * @param haystack  A column object
 * @param needles   A column of values to search for in `col`
 * @param mr        Device memory resource used to allocate the returned column's device memory
 *
 * @return A column of bool elements containing true if the corresponding entry in haystack
 * appears in needles and false if it does not.
 */
std::unique_ptr<column> contains(
  column_view const& haystack,
  column_view const& needles,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/** @} */  // end of group
}  // namespace cudf
