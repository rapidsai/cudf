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
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <rmm/mr/device/per_device_resource.hpp>

#include <vector>

namespace cudf {
namespace strings {
/**
 * @addtogroup strings_modify
 * @{
 * @file
 */

/**
 * @brief Translates individual characters within each string.
 *
 * This can also be used to remove a character by specifying 0 for the corresponding table entry.
 *
 * Null string entries result in null entries in the output column.
 *
 * @code{.pseudo}
 * Example:
 * s = ["aa","bbb","cccc","abcd"]
 * t = [['a','A'],['b',''],['d':'Q']]
 * r = translate(s,t)
 * r is now ["AA", "", "cccc", "AcQ"]
 * @endcode
 *
 * @param strings Strings instance for this operation.
 * @param chars_table Table of UTF-8 character mappings.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New column with padded strings.
 */
std::unique_ptr<column> translate(
  strings_column_view const& strings,
  std::vector<std::pair<char_utf8, char_utf8>> const& chars_table,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Removes or keeps the specified character ranges in cudf::strings::filter_characters
 */
enum class filter_type : bool { KEEP, REMOVE };

/**
 * @brief Removes ranges of characters from each string in a strings column.
 *
 * This can also be used to keep only the specified character ranges
 * and remove all others from each string.
 *
 * @code{.pseudo}
 * Example:
 * s = ["aeiou", "AEIOU", "0123456789", "bcdOPQ5"]
 * f = [{'M','Z'}, {'a','l'}, {'4','6'}]
 * r1 = filter_characters(s, f)
 * r1 is now ["aei", "OU", "456", "bcdOPQ5"]
 * r2 = filter_characters(s, f, REMOVE)
 * r2 is now ["ou", "AEI", "0123789", ""]
 * r3 = filter_characters(s, f, KEEP, "*")
 * r3 is now ["aei**", "***OU", "****456***", "bcdOPQ5"]
 * @endcode
 *
 * Null string entries result in null entries in the output column.
 *
 * @throw cudf::logic_error if `replacement` is invalid
 *
 * @param strings Strings instance for this operation.
 * @param characters_to_filter Table of character ranges to filter on.
 * @param keep_characters If true, the `characters_to_filter` are retained and all other characters
 * are removed.
 * @param replacement Optional replacement string for each character removed.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New column with filtered strings.
 */
std::unique_ptr<column> filter_characters(
  strings_column_view const& strings,
  std::vector<std::pair<cudf::char_utf8, cudf::char_utf8>> characters_to_filter,
  filter_type keep_characters         = filter_type::KEEP,
  string_scalar const& replacement    = string_scalar(""),
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/** @} */  // end of doxygen group
}  // namespace strings
}  // namespace cudf
