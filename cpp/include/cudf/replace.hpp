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

#include "cudf.h"
#include "types.hpp"

// Forward declaration
typedef struct CUstream_st* cudaStream_t;

namespace cudf {

namespace detail {

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
 * @param[in] stream Optional stream in which to perform allocations
 *
 * @returns gdf_column Column with nulls replaced
 */
gdf_column replace_nulls(const gdf_column& input,
                         const gdf_column& replacement,
                         cudaStream_t stream = 0);

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
  * @param[in] stream Optional stream in which to perform allocations
  *
  * @returns gdf_column Column with nulls replaced
  */
gdf_column replace_nulls(const gdf_column& input,
                         const gdf_scalar& replacement,
                         cudaStream_t stream = 0);

}  // namespace detail

/**
  * @brief Replaces all null values in a column with corresponding values of another column.
  *
  * Returns a column `output` such that if `input[i]` is valid, its value will be copied to
  * `output[i]`. Otherwise, `replacements[i]` will be copied to `output[i]`.
  *
  * The `input` and `replacement` columns must be of same size and have the same
  * data type.
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
  * Returns a column `output` such that if `input[i]` is valid, its value will be copied to
  * `output[i]`. Otherise, `replacement` will be coped to `output[i]`.
  *
  * `replacement` must have the same data type as `input`.
  *
  * @param[in] input A gdf_column containing null values
  * @param[in] replacement A gdf_scalar whose value will replace null values in input
  *
  * @returns gdf_column Column with nulls replaced
  */
gdf_column replace_nulls(const gdf_column& input,
                         const gdf_scalar& replacement);


/**
 * @brief Replace elements from `input_col` according to the mapping `old_values` to
 *        `new_values`, that is, replace all `old_values[i]` present in `col`
 *        with `new_values[i]` and return a new gdf_column `output`.
 *
 * @param[in] col gdf_column with the data to be modified
 * @param[in] values_to_replace gdf_column with the old values to be replaced
 * @param[in] replacement_values gdf_column with the new replacement values
 *
 * @return output gdf_column with the modified data
 *
 */
gdf_column find_and_replace_all(const gdf_column &input_col,
                                const gdf_column &values_to_replace,
                                const gdf_column &replacement_values);


} // namespace cudf

#endif // REPLACE_HPP
