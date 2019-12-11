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

namespace cudf {
namespace experimental {
  /**
   * @brief Replaces all null values in a column with corresponding values of another column
   *
   * If `input[i]` is NULL, then `output[i]` will contain `replacement[i]`. 
   * `input` and `replacement` must be of the same type and size.
   * must be of the same type and same size as the first.
   *
   * @param[in] input A column whose null values will be replaced
   * @param[in] replacement A cudf::column whose values will replace null values in input
   * @param[in] mr Optional device_memory_resource to use for allocations.
   *
   * @returns A copy of `input` with the null values replaced with corresponding values from `replacement`.
   */
std::unique_ptr<column> replace_nulls(column_view const& input,
                                      column_view const& replacement,
                                      rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
  * @brief Replaces all null values in a column with a scalar.
  *
  * If `input[i]` is NULL, then `output[i]` will contain `replacement`.
  * `input` and `replacement` must have the same type.
  * a cudf::scalar of the same data type as the column.
  *
  *
  * @param[in] input A column whose null values will be replaced
  * @param[in] replacement Scalar used to replace null values in `input`.
  * @param[in] mr Optional device_memory_resource to use for allocations.
  *
  * @returns Copy of `input` with null values replaced by `replacement`.
  */
std::unique_ptr<column> replace_nulls(column_view const& input,
                                      scalar const& replacement,
                                      rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());


/**
 * @brief Replace elements from `input_col` according to the mapping `old_values` to
 *  @brief Return a copy of `input_col` replacing all `old_values[i]` present with `new_values[i]`.
 *
 * @param input_col The column to find and replace values in.
 * @param values_to_replace The values to replace
 * @param replacement_values The values to replace with
 * @param mr Optional device_memory_resource to use for allocations.
 *
 * @returns Copy of `input` with specified values replaced.
 */
std::unique_ptr<column> find_and_replace_all(column_view const& input_col,
                                             column_view const& values_to_replace,
                                             column_view const& replacement_values,
                                             rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

} // namespace experimental

/**
 * @brief Copies from a column of floating-point elements and replaces `-NaN` and `-0.0` with `+NaN` and `+0.0`, respectively.
 *
 * Converts floating point values from @p input using the following rules:
 *        Convert  -NaN  -> NaN
 *        Convert  -0.0  -> 0.0
 *
 * @throws cudf::logic_error if column does not have floating point data type.
 * @param[in] Column of floating-point elements to copy and normalize
 * @param[in] device_memory_resource allocator for allocating output data
 *
 * @returns new column with the modified data
 */
std::unique_ptr<column> normalize_nans_and_zeros( column_view const& input,
                                                  rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource());

/**
 * @brief Modifies a column of floating-point elements to replace all `-NaN` and `-0.0` with `+NaN` and `+0.0`, respectively.
 *
 * Converts floating point values from @p in_out using the following rules:
 *        Convert  -NaN  -> NaN
 *        Convert  -0.0  -> 0.0
 *
 * @throws cudf::logic_error if column does not have floating point data type.
 * @param[in, out] Column of floating-point elements to normalize
 */
void normalize_nans_and_zeros(mutable_column_view& in_out);
} // namespace cudf
