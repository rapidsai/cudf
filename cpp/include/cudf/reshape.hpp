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

#include <memory>
#include <cudf/column/column.hpp>
#include <cudf/table/table_view.hpp>

namespace cudf {

namespace experimental {

/**
 * @brief Stack rows of a Table into a single column
 *
 * Converts the column major table @p in into a row major contiguous buffer,
 * which is returned as a `gdf_column`.
 * Example:
 * ```
 * in = [[4,5,6], [1,2,3]]
 * return = [4,1,5,2,6,3]
 * ```
 *
 * @note: The dtype of all columns in @p input should be the same
 *
 * @param input Input table
 * @return gdf_column The result stacked buffer as column
 */
std::unique_ptr<column> stack(table_view const& in);

/*
 * @brief Constructs a new table with "rows" from @in stacked @p count times.
 *
 * When @p count = 0, the returned table is `empty_like(in)`.
 *
 * in     = [8, 5, 7]
 * count  = 3
 * return = [8, 5, 7, 8, 5, 7, 8, 5, 7]
 *
 * @param[in] in     Table containing "rows" columns to tile in to new table.
 * @param[in] count  Number of times to tile "rows". Must be non-negative.
 * @return           The table containing the tiled "rows".
 */
std::unique_ptr<table> tile(table_view const& in, size_type count);

} // namespace experimental

} // namespace cudf
