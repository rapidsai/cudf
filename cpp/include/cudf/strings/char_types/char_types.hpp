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
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>

namespace cudf {
namespace strings {
/**
 * @addtogroup strings_types
 * @{
 */

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
  DECIMAL    = 1 << 0,                             /// all decimal characters
  NUMERIC    = 1 << 1,                             /// all numeric characters
  DIGIT      = 1 << 2,                             /// all digit characters
  ALPHA      = 1 << 3,                             /// all alphabetic characters
  SPACE      = 1 << 4,                             /// all space characters
  UPPER      = 1 << 5,                             /// all upper case characters
  LOWER      = 1 << 6,                             /// all lower case characters
  ALPHANUM   = DECIMAL | NUMERIC | DIGIT | ALPHA,  /// all alphanumeric characters
  CASE_TYPES = UPPER | LOWER,                      /// all case-able characters
  ALL_TYPES  = ALPHANUM | CASE_TYPES | SPACE       /// all character types
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
 * which should be checked. Otherwise, the default `ALL_TYPES` will verify all
 * characters match `types`.
 *
 * @code{.pseudo}
 * Example:
 * s = ['ab', 'a b', 'a7', 'a B']
 * b1 = s.all_characters_of_type(s,LOWER)
 * b1 is [true, false, false, false]
 * b2 = s.all_characters_of_type(s,LOWER,LOWER|UPPER)
 * b2 is [true, true, true, false]
 * @endcode
 *
 * Any null row results in a null entry for that row in the output column.
 *
 * @param strings Strings instance for this operation.
 * @param types The character types to check in each string.
 * @param verify_types Only verify against these character types.
 *                     Default `ALL_TYPES` means return `true`
 *                     iff all characters match `types`.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New column of boolean results for each string.
 */
std::unique_ptr<column> all_characters_of_type(
  strings_column_view const& strings,
  string_character_types types,
  string_character_types verify_types = string_character_types::ALL_TYPES,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Filter specific character types from a column of strings.
 *
 * To remove all characters of a specific type, set that type in
 * `types_to_remove` and set `types_to_keep` to `ALL_TYPES`.
 *
 * To filter out characters NOT of a select type, specify `ALL_TYPES` for
 * `types_to_remove` and which types to not remove in `types_to_keep`.
 *
 * @code{.pseudo}
 * Example:
 * s = ['ab', 'a b', 'a7bb', 'A7B234']
 * s1 = s.filter_characters_of_type(s,NUMERIC,"",ALL_TYPES)
 * s1 is ['ab', 'a b', 'abb', 'AB']
 * s2 = s.filter_characters_of_type(s,ALL_TYPES,"-",LOWER)
 * s2 is ['ab', 'a-b', 'a-bb', '------']
 * @endcode
 *
 * In `s1` all NUMERIC types have been removed.
 * In `s2` all non-LOWER types have been replaced.
 *
 * One but not both parameters `types_to_remove` and `types_to_keep` must
 * be set to `ALL_TYPES`.
 *
 * Any null row results in a null entry for that row in the output column.
 *
 * @throw cudf::logic_error if neither or both `types_to_remove` and
 *        `types_to_keep` are set to `ALL_TYPES`.
 *
 * @param strings Strings instance for this operation.
 * @param types_to_remove The character types to check in each string.
 *        Use `ALL_TYPES` here to specify `types_to_keep` instead.
 * @param types_to_keep Default `ALL_TYPES` means all characters of
 *        `types_to_remove` will be filtered.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New column of boolean results for each string.
 */
std::unique_ptr<column> filter_characters_of_type(
  strings_column_view const& strings,
  string_character_types types_to_remove,
  string_scalar const& replacement     = string_scalar(""),
  string_character_types types_to_keep = string_character_types::ALL_TYPES,
  rmm::mr::device_memory_resource* mr  = rmm::mr::get_current_device_resource());

/**
 * @brief Returns a boolean column identifying strings in which all
 * characters are valid for conversion to integers.
 *
 * The output row entry will be set to `true` if the corresponding string element
 * has at least one character in [-+0-9].
 *
 * @code{.pseudo}
 * Example:
 * s = ['123', '-456', '', 'A', '+7']
 * b = s.is_integer(s)
 * b is [true, true, false, false, true]
 * @endcode
 *
 * Any null row results in a null entry for that row in the output column.
 *
 * @param strings Strings instance for this operation.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New column of boolean results for each string.
 */
std::unique_ptr<column> is_integer(
  strings_column_view const& strings,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

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
 * @return true if all string are valid
 */
bool all_integer(strings_column_view const& strings);

/**
 * @brief Returns a boolean column identifying strings in which all
 * characters are valid for conversion to floats.
 *
 * The output row entry will be set to `true` if the corresponding string element
 * has at least one character in [-+0-9eE.].
 *
 * @code{.pseudo}
 * Example:
 * s = ['123', '-456', '', 'A', '+7', '8.9' '3.7e+5']
 * b = s.is_float(s)
 * b is [true, true, false, false, true, true, true]
 * @endcode
 *
 * Any null row results in a null entry for that row in the output column.
 *
 * @param strings Strings instance for this operation.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New column of boolean results for each string.
 */
std::unique_ptr<column> is_float(
  strings_column_view const& strings,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

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
 * @return true if all string are valid
 */
bool all_float(strings_column_view const& strings);

/** @} */  // end of doxygen group
}  // namespace strings
}  // namespace cudf
