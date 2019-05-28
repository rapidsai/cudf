/*
 * Copyright (c) 2018-2019, NVIDIA CORPORATION.
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

#ifndef REPLACE_HPP
#define REPLACE_HPP

#include <cudf.h>
#include <types.hpp>

namespace cudf {

/**
 * @brief Replaces all null values in a column with corresponding values of another column
 *
 * The first column is expected to be a regular gdf_column. The second column
 * must be of the same type and same size as the first.
 *
 * The function replaces all nulls of the first column with the
 * corresponding elements of the second column
 *
 * @param[in] input A gdf_column containing null values
 * @param[in] replacement A gdf_column whose values will replace null values in input
 *
 * @returns gdf_column Column with nulls replaced
 */
gdf_column replace_nulls(const gdf_column& input,
                         const gdf_column& replacement);

/**
  * @brief Replaces all null values in a column with a scalar.
  *
  * The column is expected to be a regular gdf_column. The scalar is expected to be
  * a gdf_scalar of the same data type.
  *
  * The function will replace all nulls of the column with the scalar value.
  *
  * @param[in] input A gdf_column containing null values
  * @param[in] replacement A gdf_scalar whose value will replace null values in input
  *
  * @returns gdf_column Column with nulls replaced
  */
gdf_column replace_nulls(const gdf_column& input,
                         const gdf_scalar& replacement);

}  // namespace cudf


#endif  // REPLACE_HPP
