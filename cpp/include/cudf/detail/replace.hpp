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

#include <cudf/types.hpp>
#include <memory>

// Forward declaration

namespace cudf {
namespace detail {

/**
 * @brief Replaces all null values in a column with corresponding values of another column
 *
 * `input` and `replacement` must be of the same type and size.
 * must be of the same type and same size as the first.
 *
 * @param[in] input A column whose null values will be replaced
 * @param[in] replacement A cudf::column whose values will replace null values in input
 * @param[in] mr A rmm::mr::device_memory_resource pointer to be used for allocations.
 * @param[in] stream Optional stream in which to perform allocations
 *
 * @returns A copy of `input` with the null values replaced with corresponding values from `replacement`.
 */
std::unique_ptr<column> replace_nulls(column_view const& input,
                                            cudf::column_view const& replacement,
                                            rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                            cudaStream_t stream = 0);

/**
  * @brief Replaces all null values in a column with a scalar.
  *
  * `input` and `replacement` must have the same type.
  * a cudf::scalar of the same data type as the column.
  *
  *
  * @param[in] input A column whose null values will be replaced
  * @param[in] replacement Scalar used to replace null values in `input`.
  * @param[in] mr A rmm::mr::device_memory_resource pointer to be used for allocations.
  * @param[in] stream Optional stream in which to perform allocations
  *
  * @returns Copy of `input` with null values replaced by `replacement`.
  */
std::unique_ptr<column> replace_nulls(column_view const& input,
                                            scalar const* replacement,
                                            rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                            cudaStream_t stream = 0);

/**
 *  @brief Replace all `old_values[i]` present in `input_col` with `new_values[i]`.
 *        with `new_values[i]`.
 *
 * @param input_col The column to find and replace values in.
 * @param values_to_replace The values to replace
 * @param replacement_values The values to replace with
 * @param[in] mr A rmm::mr::device_memory_resource pointer to be used for allocations.
 * @param stream The CUDA stream to use for operations
 *
 * @return Copy of `input` with specified values replaced.
 */
std::unique_ptr<column> find_and_replace_all(column_view const& input_col,
                                                   column_view const& values_to_replace,
                                                   column_view const& replacement_values,
                                                   rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                                   cudaStream_t stream = 0);
}  // namespace detail
} // namespace cudf
