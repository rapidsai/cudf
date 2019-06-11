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

#ifndef FILLING_HPP
#define FILLING_HPP

#include "cudf.h"

namespace cudf {

/**
 * @brief Fills a range of elements in a column with a scalar value.
 * 
 * Fills N elements of @p column starting at @p begin with @p value, where
 * N = (@p end - @p begin)
 *
 * The datatypes of @p column and @p value must be the same.
 *
 * @param[out] column The preallocated column to fill into
 * @param[in] value The scalar value to fill
 * @param[in] begin The starting index of the fill range
 * @param[in] end The index one past the end of the fill range
 * 
 * @return void
 */
void fill(gdf_column *column, gdf_scalar const& value, 
          gdf_index_type begin, gdf_index_type end);

}; // namespace cudf

#endif // FILLING_HPP