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

#ifndef DATETIME_HPP
#define DATETIME_HPP

#include "cudf.h"

namespace cudf {
namespace datetime {

using gdf_col_pointer =
    typename std::unique_ptr<gdf_column, std::function<void(gdf_column*)>>;

/**
 * @brief Ensures GDF_DATE64 or GDF_TIMESTAMP columns share the same time unit.
 * 
 * If they don't, this function casts the values of the less granular column
 * to the more granular time unit.
 * 
 * If the time units are the same, or either column's dtype isn't GDF_DATE64 or
 * GDF_TIMESTAMP, no cast is performed and the unique_ptr pairs will reference
 * null_ptr.
 * 
 * @note This method treats GDF_DATE64 columns like GDF_TIMESTAMP[ms]
 *
 * @param[in] gdf_column* lhs column to compare against rhs
 * @param[in] gdf_column* rhs column to compare against lhs
 *
 * @returns std::pair<gdf_col_pointer, gdf_col_pointer> pair of unique_ptrs
 * whose initialization status indicates which input column was cast, if any.
 */
std::pair<gdf_col_pointer, gdf_col_pointer> resolve_common_time_unit(gdf_column const& lhs, gdf_column const& rhs);

}  // namespace datetime 
}  // namespace cudf

#endif  // DATETIME_HPP
