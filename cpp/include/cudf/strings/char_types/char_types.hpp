/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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
 * @brief Character type values.
 * These types can be or'd to check for any combination of types.
 *
 * This cannot be turned into an enum class because or'd entries can
 * result in values that are not in the class. For example,
 * combining NUMERIC|SPACE is a valid, reasonable combination but
 * does not match to any explicitly named enumerator.
 */
enum string_character_types : uint32_t {
  DECIMAL    = 1 << 0,                             // binary 00000001
  NUMERIC    = 1 << 1,                             // binary 00000010
  DIGIT      = 1 << 2,                             // binary 00000100
  ALPHA      = 1 << 3,                             // binary 00001000
  SPACE      = 1 << 4,                             // binary 00010000
  UPPER      = 1 << 5,                             // binary 00100000
  LOWER      = 1 << 6,                             // binary 01000000
  ALPHANUM   = DECIMAL | NUMERIC | DIGIT | ALPHA,  // binary 00001111
  CASE_TYPES = UPPER | LOWER,
  ALL_TYPES  = ALPHANUM | CASE_TYPES | SPACE
};

// OR operators for combining types
string_character_types operator|(string_character_types lhs, string_character_types rhs)
{
  return static_cast<string_character_types>(
    static_cast<std::underlying_type_t<string_character_types>>(lhs) |
    static_cast<std::underlying_type_t<string_character_types>>(rhs));
}

string_character_types& operator|=(string_character_types& lhs, string_character_types rhs)
{
  lhs = static_cast<string_character_types>(
    static_cast<std::underlying_type_t<string_character_types>>(lhs) |
    static_cast<std::underlying_type_t<string_character_types>>(rhs));
  return lhs;
}

/**
 * @brief Returns a boolean column identifying strings entries in which all
 * characters are of the type specified.
 *
 * The output row entry will be set to false if the corresponding string element
 * is empty or has at least one character not of the specified type. If all
 * characters fit the type then true is set in that output row entry.
 *
 * To ignore all but specific types, set the `verify_types` to those types
 * which should be checked. Otherwise, the default ALL_TYPES will verify all
 * characters match `types`.
 *
 * ```
 * Example:
 * s = ['ab', 'a b', 'a7', 'a B']
 * b1 = s.all_characters_of_type(s,LOWER)
 * b1 is [true, false, false, false]
 * b2 = s.all_characters_of_type(s,LOWER,LOWER|UPPER)
 * b2 is [true, true, true, false]
 *
 * ```
 *
 * Any null row results in a null entry for that row in the output column.
 *
 * @param strings Strings instance for this operation.
 * @param types The character types to check in each string.
 * @param verify_types Only verify against these character types.
 *                     Default `ALL_TYPES` means return `true`
 *                     iff all characters match `types`.
 * @param mr Resource for allocating device memory.
 * @return New column of boolean results for each string.
 */
std::unique_ptr<column> all_characters_of_type(
  strings_column_view const& strings,
  string_character_types types,
  string_character_types verify_types = string_character_types::ALL_TYPES,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Returns a boolean column identifying strings in which all
 * characters are valid for conversion to integers.
 *
 * The output row entry will be set to `true` if the corresponding string element
 * has at least one character in [-+0-9].
 *
 * ```
 * Example:
 * s = ['123', '-456', '', 'A', '+7']
 * b = s.is_integer(s)
 * b is [true, true, false, false, true]
 *
 * ```
 *
 * Any null row results in a null entry for that row in the output column.
 *
 * @param strings Strings instance for this operation.
 * @param mr Resource for allocating device memory.
 * @return New column of boolean results for each string.
 */
std::unique_ptr<column> is_integer(
  strings_column_view const& strings,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Returns `true` if all strings contain
 * characters that are valid for conversion to integers.
 *
 * This function will return `true` if all string elements
 * has at least one character in [-+0-9].
 *
 * Any null entry or empty string will cause this function to return `false`.
 *
 * @param strings Strings instance for this operation.
 * @param mr Resource for allocating device memory.
 * @return true if all string are valid
 */
bool all_integer(strings_column_view const& strings,
                 rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Returns a boolean column identifying strings in which all
 * characters are valid for conversion to floats.
 *
 * The output row entry will be set to `true` if the corresponding string element
 * has at least one character in [-+0-9eE.].
 *
 * ```
 * Example:
 * s = ['123', '-456', '', 'A', '+7', '8.9' '3.7e+5']
 * b = s.is_float(s)
 * b is [true, true, false, false, true, true, true]
 *
 * ```
 *
 * Any null row results in a null entry for that row in the output column.
 *
 * @param strings Strings instance for this operation.
 * @param mr Resource for allocating device memory.
 * @return New column of boolean results for each string.
 */
std::unique_ptr<column> is_float(
  strings_column_view const& strings,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Returns `true` if all strings contain
 * characters that are valid for conversion to floats.
 *
 * This function will return `true` if all string elements
 * has at least one character in [-+0-9eE.].
 *
 * Any null entry or empty string will cause this function to return `false`.
 *
 * @param strings Strings instance for this operation.
 * @param mr Resource for allocating device memory.
 * @return true if all string are valid
 */
bool all_float(strings_column_view const& strings,
               rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

}  // namespace strings
}  // namespace cudf
