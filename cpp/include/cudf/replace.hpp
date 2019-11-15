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
#include <rmm/mr/default_memory_resource.hpp>

namespace cudf {
namespace experimental {
/**
  * @brief Replaces all null values in a column with corresponding values of another column.
  *
  * Returns a column `output` such that if `input[i]` is valid, its value will be copied to
  * `output[i]`. Otherwise, `replacements[i]` will be copied to `output[i]`.
  *
  * The `input` and `replacement` columns must be of same size and have the same
  * data type.
  *
  * @param[in] input A cudf::column containing null values
  * @param[in] replacement A cudf::column whose values will replace null values in input
  *
  * @returns Column with nulls replaced
  */
std::unique_ptr<cudf::column> replace_nulls(cudf::column_view const& input,
                                            cudf::column_view const& replacement,
                                            rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
  * @brief Replaces all null values in a column with a scalar.
  *
  * Returns a column `output` such that if `input[i]` is valid, its value will be copied to
  * `output[i]`. Otherise, `replacement` will be coped to `output[i]`.
  *
  * `replacement` must have the same data type as `input`.
  *
  * @param[in] input A cudf::column containing null values
  * @param[in] replacement A cudf::scalar whose value will replace null values in input
  *
  * @returns Column with nulls replaced
  */
std::unique_ptr<cudf::column> replace_nulls(cudf::column_view const& input,
                                            cudf::scalar const& replacement,
                                            rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());


/**
 * @brief Finds and replaces values within a column.
 * 
 * Creates a copy of @p input, but replace elements according to the mapping `old_values` to
 *        `new_values`, that is, replace all `old_values[i]` present in `col`
 *        with `new_values[i]` and return a new gdf_column `output`.
 *
 * @param[in] col cudf::column with the data to be modified
 * @param[in] values_to_replace gdf_column with the old values to be replaced
 * @param[in] replacement_values gdf_column with the new replacement values
 *
 * @returns Copy of the input with the specified values replaced
 *
 */
std::unique_ptr<cudf::column> find_and_replace_all(cudf::column_view const& input_col,
                                                   cudf::column_view const& values_to_replace,
                                                   cudf::column_view const& replacement_values,
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
