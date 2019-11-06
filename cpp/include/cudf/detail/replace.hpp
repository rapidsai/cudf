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

#pragma once

#include "cudf.h"
#include "types.hpp"

// Forward declaration
typedef struct CUstream_st* cudaStream_t;

namespace cudf {
namespace detail {

/**
 * @brief Replaces all null values in a column with corresponding values of another column
 *
 * `input` and `replacement` must be of the same type and size.
 * must be of the same type and same size as the first.
 *
 * The function replaces all nulls of the first column with the
 * corresponding elements of the second column
 *
 * @param[in] input A cudf::column containing null values
 * @param[in] replacement A cudf::column whose values will replace null values in input
 * @param[in] stream Optional stream in which to perform allocations
 *
 * @returns Column with nulls replaced
 */
std::unique_ptr<cudf::column> replace_nulls(cudf::column_view const& input,
                                            cudf::column_view const& replacement,
                                            cudaStream_t stream = 0,
                                            device_memory_resource* mr = rmm::get_default_resource());

/**
  * @brief Replaces all null values in a column with a scalar.
  *
  * The column is expected to be a cudf::column. The scalar is expected to be
  * a gdf_scalar of the same data type as the column.
  *
  * The function will replace all nulls of the column with the scalar value.
  *
  * @param[in] input A gdf_column containing null values
  * @param[in] replacement A gdf_scalar whose value will replace null values in input
  * @param[in] stream Optional stream in which to perform allocations
  *
  * @returns Column with nulls replaced
  */
std::unique_ptr<cudf::column> replace_nulls(cudf::column_view const& input,
                                            const gdf_scalar& replacement,
                                            cudaStream_t stream = 0,
                                            device_memory_resource* mr = rmm::get_default_resource());

/**
 * @brief Replace elements from `input_col` according to the mapping `old_values` to
 *        `new_values`, that is, replace all `old_values[i]` present in `col`
 *        with `new_values[i]` and return a new gdf_column `output`.
 *
 * @param input_col The column to find and replace values in.
 * @param values_to_replace The values to replace
 * @param replacement_values The values to replace with
 * @param stream The CUDA stream to use for operations
 * @return The input column with specified values replaced.
 */
std::unique_ptr<cudf::column> find_and_replace_all(cudf::column_view const& input_col,
                                                   cudf::column_view const& values_to_replace,
                                                   cudf::column_view const& replacement_values,
                                                   cudaStream_t stream = 0,
                                                   device_memory_resource* mr = rmm::get_default_resource());
}  // namespace detail
} // namespace cudf
