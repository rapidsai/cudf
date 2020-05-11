/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.
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
/**
 * @ingroup transformation_replace
 * @{
 */

namespace experimental {

/**
 * @brief Replaces all null values in a column with corresponding values of another column
 *
 * @ingroup column_replace
 *
 * If `input[i]` is NULL, then `output[i]` will contain `replacement[i]`.
 * `input` and `replacement` must be of the same type and size.
 * must be of the same type and same size as the first.
 *
 * @param[in] input A column whose null values will be replaced
 * @param[in] replacement A cudf::column whose values will replace null values in input
 * @param[in] mr Optional device_memory_resource to use for allocations.
 *
 * @returns A copy of `input` with the null values replaced with corresponding values from
 * `replacement`.
 */
std::unique_ptr<column> replace_nulls(
  column_view const& input,
  column_view const& replacement,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Replaces all null values in a column with a scalar.
 *
 * @ingroup column_replace
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
std::unique_ptr<column> replace_nulls(
  column_view const& input,
  scalar const& replacement,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Return a copy of `input_col` replacing any `values_to_replace[i]`
 * found with `replacement_values[i]`.
 *
 * @param input_col The column to find and replace values in.
 * @param values_to_replace The values to replace
 * @param replacement_values The values to replace with
 * @param mr Optional device_memory_resource to use for allocations.
 *
 * @returns Copy of `input_col` with specified values replaced.
 */
std::unique_ptr<column> find_and_replace_all(
  column_view const& input_col,
  column_view const& values_to_replace,
  column_view const& replacement_values,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Replaces values less than `lo` in `input` with `lo_replace`,
 * and values greater than `hi` with `hi_replace`.
 *
 * @ingroup column_replace
 *
 * if `lo` is invalid, then lo will not be considered while
 * evaluating the input (Essentially considered minimum value of that type).
 * if `hi` is invalid, then hi will not be considered while
 * evaluating the input (Essentially considered maximum value of that type).
 *
 * @note: If `lo` is valid then `lo_replace` should be valid
 *        If `hi` is valid then `hi_replace` should be valid
 *
 * ```
 * Example:
 *    input: {1, 2, 3, NULL, 5, 6, 7}
 *
 *    valid lo and hi
 *    lo: 3, hi: 5, lo_replace : 0, hi_replace : 16
 *    output:{0, 0, 3, NULL, 5, 16, 16}
 *
 *    invalid lo
 *    lo: NULL, hi: 5, lo_replace : 0, hi_replace : 16
 *    output:{1, 2, 3, NULL, 5, 16, 16}
 *
 *    invalid hi
 *    lo: 3, hi: NULL, lo_replace : 0, hi_replace : 16
 *    output:{0, 0, 3, NULL, 5, 6, 7}
 * ```
 *
 * @throws cudf::logic_error if `lo.type() != hi.type()`
 * @throws cudf::logic_error if `lo_replace.type() != hi_replace.type()`
 * @throws cudf::logic_error if `lo.type() != lo_replace.type()`
 * @throws cudf::logic_error if `lo.type() != input.type()`
 *
 * @param[in] input Column whose elements will be clamped
 * @param[in] lo Minimum clamp value. All elements less than `lo` will be replaced by `lo_replace`.
 * Ignored if null.
 * @param[in] lo_replace All elements less than `lo` will be replaced by `lo_replace`.
 * @param[in] hi Maximum clamp value. All elements greater than `hi` will be replaced by
 * `hi_replace`. Ignored if null.
 * @param[in] hi_replace All elements greater than `hi` will be replaced by `hi_replace`.
 * @param[in] mr Optional resource to use for device memory
 *           allocation of the returned result column.
 *
 * @return Returns a clamped column as per `lo` and `hi` boundaries
 */
std::unique_ptr<column> clamp(
  column_view const& input,
  scalar const& lo,
  scalar const& lo_replace,
  scalar const& hi,
  scalar const& hi_replace,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Replaces values less than `lo` in `input` with `lo`,
 * and values greater than `hi` with `hi`.
 *
 * @ingroup column_replace
 *
 * if `lo` is invalid, then lo will not be considered while
 * evaluating the input (Essentially considered minimum value of that type).
 * if `hi` is invalid, then hi will not be considered while
 * evaluating the input (Essentially considered maximum value of that type).
 *
 * ```
 * Example:
 *    input: {1, 2, 3, NULL, 5, 6, 7}
 *
 *    valid lo and hi
 *    lo: 3, hi: 5
 *    output:{3, 3, 3, NULL, 5, 5, 5}
 *
 *    invalid lo
 *    lo: NULL, hi:5
 *    output:{1, 2, 3, NULL, 5, 5, 5}
 *
 *    invalid hi
 *    lo: 3, hi:NULL
 *    output:{3, 3, 3, NULL, 5, 6, 7}
 * ```
 *
 * @throws cudf::logic_error if `lo.type() != hi.type()`
 * @throws cudf::logic_error if `lo.type() != input.type()`
 *
 * @param[in] input Column whose elements will be clamped
 * @param[in] lo Minimum clamp value. All elements less than `lo` will be replaced by `lo`. Ignored
 * if null.
 * @param[in] hi Maximum clamp value. All elements greater than `hi` will be replaced by `hi`.
 * Ignored if null.
 * @param[in] mr Optional resource to use for device memory
 *           allocation of the returned result column.
 *
 * @return Returns a clamped column as per `lo` and `hi` boundaries
 */
std::unique_ptr<column> clamp(
  column_view const& input,
  scalar const& lo,
  scalar const& hi,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

}  // namespace experimental

/**
 * @brief Copies from a column of floating-point elements and replaces `-NaN` and `-0.0` with `+NaN`
 * and `+0.0`, respectively.
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
std::unique_ptr<column> normalize_nans_and_zeros(
  column_view const& input, rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Modifies a column of floating-point elements to replace all `-NaN` and `-0.0` with `+NaN`
 * and `+0.0`, respectively.
 *
 * Converts floating point values from @p in_out using the following rules:
 *        Convert  -NaN  -> NaN
 *        Convert  -0.0  -> 0.0
 *
 * @throws cudf::logic_error if column does not have floating point data type.
 * @param[in, out] Column of floating-point elements to normalize
 */
void normalize_nans_and_zeros(mutable_column_view& in_out);

/** @} */  // end of group
}  // namespace cudf
