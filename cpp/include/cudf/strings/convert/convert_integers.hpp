/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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
 */

/**
 * @brief Returns a new integer numeric column parsing integer values from the
 * provided strings column.
 *
 * Any null entries will result in corresponding null entries in the output column.
 *
 * Only characters [0-9] plus a prefix '-' and '+' are recognized.
 * When any other character is encountered, the parsing ends for that string
 * and the current digits are converted into an integer.
 *
 * Overflow of the resulting integer type is not checked.
 * Each string is converted using an int64 type and then cast to the
 * target integer type before storing it into the output column.
 * If the resulting integer type is too small to hold the value,
 * the stored value will be undefined.
 *
 * @throw cudf::logic_error if output_type is not integral type.
 *
 * @param strings Strings instance for this operation.
 * @param output_type Type of integer numeric column to return.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New column with integers converted from strings.
 */
std::unique_ptr<column> to_integers(
  strings_column_view const& strings,
  data_type output_type,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Returns a new strings column converting the integer values from the
 * provided column into strings.
 *
 * Any null entries will result in corresponding null entries in the output column.
 *
 * For each integer, a string is created in base-10 decimal.
 * Negative numbers will include a '-' prefix.
 *
 * @throw cudf::logic_error if integers column is not integral type.
 *
 * @param integers Numeric column to convert.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New strings column with integers as strings.
 */
std::unique_ptr<column> from_integers(
  column_view const& integers,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Returns a new integer numeric column parsing hexadecimal values from the
 * provided strings column.
 *
 * Any null entries will result in corresponding null entries in the output column.
 *
 * Only characters [0-9] and [A-F] are recognized.
 * When any other character is encountered, the parsing ends for that string.
 * No interpretation is made on the sign of the integer.
 *
 * Overflow of the resulting integer type is not checked.
 * Each string is converted using an int64 type and then cast to the
 * target integer type before storing it into the output column.
 * If the resulting integer type is too small to hold the value,
 * the stored value will be undefined.
 *
 * @throw cudf::logic_error if output_type is not integral type.
 *
 * @param strings Strings instance for this operation.
 * @param output_type Type of integer numeric column to return.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New column with integers converted from strings.
 */
std::unique_ptr<column> hex_to_integers(
  strings_column_view const& strings,
  data_type output_type,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Returns a boolean column identifying strings in which all
 * characters are valid for conversion to integers from hex.
 *
 * The output row entry will be set to `true` if the corresponding string element
 * has at least one character in [0-9A-Za-z]. Also, the string may start
 * with '0x'.
 *
 * @code{.pseudo}
 * Example:
 * s = ['123', '-456', '', 'AGE', '+17EA', '0x9EF' '123ABC']
 * b = s.is_hex(s)
 * b is [true, false, false, false, false, true, true]
 * @endcode
 *
 * Any null row results in a null entry for that row in the output column.
 *
 * @param strings Strings instance for this operation.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New column of boolean results for each string.
 */
std::unique_ptr<column> is_hex(
  strings_column_view const& strings,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/** @} */  // end of doxygen group
}  // namespace strings
}  // namespace cudf
