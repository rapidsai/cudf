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
gdf_column stack(const cudf::table &input);

}; // namespace cudf
