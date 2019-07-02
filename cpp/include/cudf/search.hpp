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

#include <cudf/cudf.h>


namespace cudf {

/**---------------------------------------------------------------------------*
 * @brief Find indices in sorted column where values should be inserted to
 *  maintain order
 * 
 * @note The @p column is required to be sorted in ascending order otherwise
 *  the behaviour is undefined
 * 
 * @param column        Column to search in. Must be sorted in ascending order
 * @param values        Values to search insert locations for
 * @param find_first    If true, for each value, the index of the first suitable
 *  insert location is produced. If false, the *last* such index is produced
 * 
 * @return gdf_column   Insertion points with the same shape as values
 *---------------------------------------------------------------------------**/
gdf_column search_ordered(gdf_column const& column,
                          gdf_column const& values,
                          bool find_first = true,
                          bool nulls_as_largest = true);

} // namespace cudf

