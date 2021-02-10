/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <cudf/column/column.hpp>
#include <cudf/strings/strings_column_view.hpp>

namespace cudf {
namespace strings {
/**
 * @addtogroup strings_convert
 * @{
 * @file
 */

/**
 * @brief Returns a new fixed-point column parsing decimal values from the
 * provided strings column.
 *
 * Any null entries will result in corresponding null entries in the output column.
 *
 * Only characters [0-9] plus a prefix '-' and '+' and a single decimal point
 * are recognized. When any other character is encountered, the parsing ends
 * for that string and the current digits are converted into a fixed-point value.
 *
 * Overflow of the resulting value type is not checked.
 * The decimal point is used only for determining the output scale value.
 *
 * @throw cudf::logic_error if output_type is not a fixed-point type.
 *
 * @param strings Strings instance for this operation.
 * @param output_type Type of fixed-point column to return.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New column of `output_type`.
 */
std::unique_ptr<column> to_fixed_point(
  strings_column_view const& input,
  data_type output_type,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Returns a new strings column converting the fixed-point values from the
 * provided column into strings.
 *
 * Any null entries will result in corresponding null entries in the output column.
 *
 * For each value, a string is created in base-10 decimal.
 * Negative numbers will include a '-' prefix.
 * The column's scale value is used to place the decimal point.
 *
 * @throw cudf::logic_error if the input column is not a fixed-point type.
 *
 * @param input Fixed-point column to convert.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New strings column.
 */
std::unique_ptr<column> from_fixed_point(
  column_view const& input,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Returns a boolean column identifying strings in which all
 * characters are valid for conversion to fixed-point.
 *
 * The output row entry will be set to `true` if the corresponding string element
 * has at least one character in [+-0-9.].
 *
 * @code{.pseudo}
 * Example:
 * s = ['123', '-456', '', '1.2.3', '+17E30', '12.34' '.789']
 * b = s.is_fixed_point(s)
 * b is [true, false, false, false, false, true, true]
 * @endcode
 *
 * Any null row results in a null entry for that row in the output column.
 *
 * @param strings Strings instance for this operation.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New column of boolean results for each string.
 */
std::unique_ptr<column> is_fixed_point(
  strings_column_view const& input,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/** @} */  // end of doxygen group
}  // namespace strings
}  // namespace cudf
