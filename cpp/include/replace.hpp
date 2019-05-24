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
 * @brief Replaces all null values in a column with either a specific value or corresponding values of another column
 *
 * This function is a binary function. It will take in two gdf_columns.

 * The first one is expected to be a regular gdf_column, the second one
 * has to be a column of the same type as the first, and it has to be of
 * size one or of the same size as the other column.
 *
 * case 1: If the second column contains only one value, then this funciton will
 * replace all nulls in the first column with the value in the second
 * column.
 *
 * case 2: If the second column is of the same size as the first, then the function will
 * replace all nulls of the first column with the corresponding elemetns of the
 * second column
 *
 * @param[in] input A gdf_column that is the output of this function with null values replaced
 * @param[in] replacement_values A gdf_column that is of size 1 or same size as col_out, contains value / values to be placed in col_out
 *
 * @returns gdf_column Column with nulls replaced
 */
gdf_column replace_nulls(const gdf_column& input,
                         const gdf_column& replacement_values);

gdf_column replace_nulls(const gdf_column& input,
                         const gdf_scalar& replacement_value);

}  // namespace cudf


#endif  // REPLACE_HPP
