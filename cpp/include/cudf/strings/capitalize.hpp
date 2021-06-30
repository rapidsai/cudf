/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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
#include <cudf/strings/char_types/char_types.hpp>
#include <cudf/strings/strings_column_view.hpp>

namespace cudf {
namespace strings {
/**
 * @addtogroup strings_case
 * @{
 * @file
 */

/**
 * @brief Returns a column of capitalized strings.
 *
 * If the `delimiters` is an empty string, then only the first character of each
 * row is capitalized. Otherwise, a non-delimiter character is capitalized after
 * any delimiter character is found.
 *
 * @code{.pseudo}
 * Example:
 * input = ["tesT1", "a Test", "Another Test", "a\tb"];
 * output = capitalize(input)
 * output is ["Test1", "A test", "Another test", "A\tb"]
 * output = capitalize(input, " ")
 * output is ["Test1", "A Test", "Another Test", "A\tb"]
 * output = capitalize(input, " \t")
 * output is ["Test1", "A Test", "Another Test", "A\tB"]
 * @endcode
 *
 * Any null string entries return corresponding null output column entries.
 *
 * @throw cudf::logic_error if `delimiter.is_valid()` is  `false`.
 *
 * @param input String column.
 * @param delimiters Characters for identifying words to capitalize.
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Column of strings capitalized from the input column.
 */
std::unique_ptr<column> capitalize(
  strings_column_view const& input,
  string_scalar const& delimiters     = string_scalar(""),
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Modifies first character of each word to upper-case and lower-cases the rest.
 *
 * A word here is a sequence of characters of `sequence_type` delimited by
 * any characters not part of the `sequence_type` character set.
 *
 * This function returns a column of strings where, for each string row in the input,
 * the first character of each word is converted to upper-case,
 * while all the remaining characters in a word are converted to lower-case.
 *
 * @code{.pseudo}
 * Example:
 * input = ["   teST1", "a Test", " Another test ", "n2vidia"];
 * output = title(input)
 * output is ["   Test1", "A Test", " Another Test ", "N2Vidia"]
 * output = title(input,ALPHANUM)
 * output is ["   Test1", "A Test", " Another Test ", "N2vidia"]
 * @endcode
 *
 * Any null string entries return corresponding null output column entries.
 *
 * @param input String column.
 * @param sequence_type The character type that is used when identifying words.
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Column of titled strings.
 */
std::unique_ptr<column> title(
  strings_column_view const& input,
  string_character_types sequence_type = string_character_types::ALPHA,
  rmm::mr::device_memory_resource* mr  = rmm::mr::get_current_device_resource());

/** @} */  // end of doxygen group
}  // namespace strings
}  // namespace cudf
