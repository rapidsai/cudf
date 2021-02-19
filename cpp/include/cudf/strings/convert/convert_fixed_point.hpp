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
 * Any null entries result in corresponding null entries in the output column.
 *
 * The expected format is `[sign][integer][.][fraction]`, where the sign is either
 * not present, `-` or `+`, The decimal point `[.]` may or may not be present, and
 * `integer` and `fraction` are comprised of zero or more digits in [0-9].
 * An invalid data format results in undefined behavior in the corresponding
 * output row result.
 *
 * @code{.pseudo}
 * Example:
 * s = ['123', '-876', '543.2', '-0.12']
 * datatype = {DECIMAL32, scale=-2}
 * fp = to_fixed_point(s, datatype)
 * fp is [123400, -87600, 54320, -12]
 * @endcode
 *
 * Overflow of the resulting value type is not checked.
 * The scale in the `output_type` is used for setting the integer component.
 *
 * @throw cudf::logic_error if `output_type` is not a fixed-point decimal type.
 *
 * @param strings Strings instance for this operation.
 * @param output_type Type of fixed-point column to return including the scale value.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New column of `output_type`.
 */
std::unique_ptr<column> to_fixed_point(
  strings_column_view const& input,
  data_type output_type,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Returns a new strings column converting the fixed-point values
 * into a strings column.
 *
 * Any null entries result in corresponding null entries in the output column.
 *
 * For each value, a string is created in base-10 decimal.
 * Negative numbers include a '-' prefix in the output string.
 * The column's scale value is used to place the decimal point.
 * A negative scale value may add padded zeros after the decimal point.
 *
 * @code{.pseudo}
 * Example:
 * fp is [110, 222, 3330, -440, -1] with scale = -2
 * s = from_fixed_point(fp)
 * s is now ['1.10', '2.22', '33.30', '-4.40', '-0.01']
 * @endcode
 *
 * @throw cudf::logic_error if the `input` column is not a fixed-point decimal type.
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
 * The output row entry is set to `true` if the corresponding string element
 * has at least one character in [+-0123456789.]. The optional sign character
 * must only be in the first position. The decimal point may only appear once.
 * Also, the integer component must fit within the size limits of the
 * underlying fixed-point storage type. The value of the integer component
 * is based on the scale of the `decimal_type` provided.
 *
 * @code{.pseudo}
 * Example:
 * s = ['123', '-456', '', '1.2.3', '+17E30', '12.34' '.789', '-0.005]
 * b = is_fixed_point(s)
 * b is [true, true, false, false, false, true, true, true]
 * @endcode
 *
 * Any null entries result in corresponding null entries in the output column.
 *
 * @throw cudf::logic_error if the `decimal_type` is not a fixed-point decimal type.
 *
 * @param input Strings instance for this operation.
 * @param decimal_type Fixed-point type (with scale) used only for checking overflow.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New column of boolean results for each string.
 */
std::unique_ptr<column> is_fixed_point(
  strings_column_view const& input,
  data_type decimal_type              = data_type{type_id::DECIMAL64},
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/** @} */  // end of doxygen group
}  // namespace strings
}  // namespace cudf
