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

#include "cudf.h"


namespace cudf {

/**---------------------------------------------------------------------------*
 * @brief Find smallest indices in sorted column where values should be 
 *  inserted to maintain order
 * 
 * @note The @p column is required to be sorted in ascending order otherwise
 *  the behaviour is undefined
 * 
 * @param column        Column to search in. Must be sorted in ascending order
 * @param values        Values to search insert locations for
 * @param nulls_as_largest If true, nulls are considered larger than valid
 *  values, otherwise, nulls are considered smaller than valid values
 * 
 * @return gdf_column   Insertion points with the same shape as values
 *---------------------------------------------------------------------------**/
gdf_column lower_bound(gdf_column const& column,
                       gdf_column const& values,
                       bool nulls_as_largest = true);

/**---------------------------------------------------------------------------*
 * @brief Find largest indices in sorted column where values should be 
 *  inserted to maintain order
 * 
 * @note The @p column is required to be sorted in ascending order otherwise
 *  the behaviour is undefined
 * 
 * @param column        Column to search in. Must be sorted in ascending order
 * @param values        Values to search insert locations for
 * @param nulls_as_largest If true, nulls are considered larger than valid
 *  values, otherwise, nulls are considered smaller than valid values
 * 
 * @return gdf_column   Insertion points with the same shape as values
 *---------------------------------------------------------------------------**/
gdf_column upper_bound(gdf_column const& column,
                       gdf_column const& values,
                       bool nulls_as_largest = true);

/**---------------------------------------------------------------------------*
 * @brief Find smallest indices in sorted table where values should be 
 *  inserted to maintain order
 * 
 * @note The table @p t is required to be sorted in ascending order otherwise
 *  the behaviour is undefined
 * 
 * @param t             Table to search in. Must be sorted in ascending order
 * @param values        Values to search insert locations for
 * @param nulls_as_largest If true, nulls are considered larger than valid
 *  values, otherwise, nulls are considered smaller than valid values
 * 
 * @return gdf_column   Insertion points with the same shape as values
 *---------------------------------------------------------------------------**/
gdf_column lower_bound(table const& t,
                       table const& values,
                       bool nulls_as_largest = true);

/**---------------------------------------------------------------------------*
 * @brief Find largest indices in sorted table where values should be 
 *  inserted to maintain order
 * 
 * @note The table @p t is required to be sorted in ascending order otherwise
 *  the behaviour is undefined
 * 
 * @param column        Column to search in. Must be sorted in ascending order
 * @param values        Values to search insert locations for
 * @param nulls_as_largest If true, nulls are considered larger than valid
 *  values, otherwise, nulls are considered smaller than valid values
 * 
 * @return gdf_column   Insertion points with the same shape as values
 *---------------------------------------------------------------------------**/
gdf_column upper_bound(table const& t,
                       table const& values,
                       bool nulls_as_largest = true);

} // namespace cudf

