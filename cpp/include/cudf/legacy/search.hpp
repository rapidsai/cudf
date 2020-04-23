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

#include <vector>
#include "cudf/cudf.h"
#include "cudf/types.hpp"

namespace cudf {

/**---------------------------------------------------------------------------*
 * @brief Find smallest indices in a sorted table where values should be
 *  inserted to maintain order
 *
 * For each row v in @p values, find the first index in @p t where
 *  inserting the row will maintain the sort order of @p t
 *
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
 *
 * @param t             Table to search
 * @param values        Find insert locations for these values
 * @param desc_flags    Vector of column sort order. False indicates the
 *  corresponding column is sorted ascending
 * @param nulls_as_largest If true, nulls are considered larger than valid
 *  values, otherwise, nulls are considered smaller than valid values
 *
 * @return gdf_column   Insertion points. Non-nullable column of type GDF_INT32 with same size as
 *values.
 *---------------------------------------------------------------------------**/
gdf_column lower_bound(table const& t,
                       table const& values,
                       std::vector<bool> const& desc_flags,
                       bool nulls_as_largest = true);

/**---------------------------------------------------------------------------*
 * @brief Find largest indices in a sorted table where values should be
 *  inserted to maintain order
 *
 * For each row v in @p values, find the last index in @p t where
 *  inserting the row will maintain the sort order of @p t
 *
 * Example:
 *
 *  Single Column:
 *      idx      0   1   2   3   4
 *   column = { 10, 20, 20, 30, 50 }
 *   values = { 20 }
 *   result = {  3 }
 *
 *  Multi Column:
 *    idx        0    1    2    3    4
 *   t      = {{  10,  20,  20,  20,  20 },
 *             { 5.0,  .5,  .5,  .7,  .7 },
 *             {  90,  77,  78,  61,  61 }}
 *   values = {{ 20 },
 *             { .7 },
 *             { 61 }}
 *   result =  {  5  *   *
 * @param column        Table to search
 * @param values        Find insert locations for these values
 * @param desc_flags    Vector of column sort order. False indicates the
 *  corresponding column is sorted ascending
 * @param nulls_as_largest If true, nulls are considered larger than valid
 *  values, otherwise, nulls are considered smaller than valid values
 *
 * @return gdf_column   Insertion points. Non-nullable column of type GDF_INT32 with same size as
 *values.
 *---------------------------------------------------------------------------**/
gdf_column upper_bound(table const& t,
                       table const& values,
                       std::vector<bool> const& desc_flags,
                       bool nulls_as_largest = true);

/**---------------------------------------------------------------------------*
 * @brief Find if the `value` is present in the `column` and dtype of both
 * `value` and `column` should match.
 *
 * @throws cudf::logic_error
 * If dtype of `column` and `value` doesn't match
 *
 * @example:
 *
 *  Single Column:
 *      idx      0   1   2   3   4
 *   column = { 10, 20, 20, 30, 50 }
 *  Scalar:
 *   value = { 20 }
 *   result = true
 *
 * @param column   A gdf column
 * @param value    A scalar value to search for in `column`
 *
 * @return bool    If `value` is found in `column` true, else false.
 *---------------------------------------------------------------------------**/
bool contains(gdf_column const& column, gdf_scalar const& value);
}  // namespace cudf
