/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

namespace nvtext {
/**
 * @addtogroup nvtext_tokenize
 * @{
 * @file
 */

/**
 * @brief Returns a single column of strings by tokenizing the input strings
 * column using the provided characters as delimiters.
 *
 * The `delimiter` may be zero or more characters. If the `delimiter` is empty,
 * whitespace (character code-point <= ' ') is used for identifying tokens.
 * Also, any consecutive delimiters found in a string are ignored.
 * This means only non-empty tokens are returned.
 *
 * Tokens are found by locating delimiter(s) starting at the beginning of each string.
 * As each string is tokenized, the tokens are appended using input column row order
 * to build the output column. That is, tokens found in input row[i] will be placed in
 * the output column directly before tokens found in input row[i+1].
 *
 * @code{.pseudo}
 * Example:
 * s = ["a", "b c", "d  e f "]
 * t = tokenize(s)
 * t is now ["a", "b", "c", "d", "e", "f"]
 * @endcode
 *
 * All null row entries are ignored and the output contains all valid rows.
 *
 * @param strings Strings column tokenize.
 * @param delimiter UTF-8 characters used to separate each string into tokens.
 *                  The default of empty string will separate tokens using whitespace.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New strings columns of tokens.
 */
std::unique_ptr<cudf::column> tokenize(
  cudf::strings_column_view const& strings,
  cudf::string_scalar const& delimiter = cudf::string_scalar{""},
  rmm::mr::device_memory_resource* mr  = rmm::mr::get_current_device_resource());

/**
 * @brief Returns a single column of strings by tokenizing the input strings
 * column using multiple strings as delimiters.
 *
 * Tokens are found by locating delimiter(s) starting at the beginning of each string.
 * Any consecutive delimiters found in a string are ignored.
 * This means only non-empty tokens are returned.
 *
 * As each string is tokenized, the tokens are appended using input column row order
 * to build the output column. That is, tokens found in input row[i] will be placed in
 * the output column directly before tokens found in input row[i+1].
 *
 * @code{.pseudo}
 * Example:
 * s = ["a", "b c", "d.e:f;"]
 * d = [".", ":", ";"]
 * t = tokenize(s,d)
 * t is now ["a", "b c", "d", "e", "f"]
 * @endcode
 *
 * All null row entries are ignored and the output contains all valid rows.
 *
 * @throw cudf::logic_error if the delimiters column is empty or contains nulls.
 *
 * @param strings Strings column to tokenize.
 * @param delimiters Strings used to separate individual strings into tokens.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New strings columns of tokens.
 */
std::unique_ptr<cudf::column> tokenize(
  cudf::strings_column_view const& strings,
  cudf::strings_column_view const& delimiters,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Returns the number of tokens in each string of a strings column.
 *
 * The `delimiter` may be zero or more characters. If the `delimiter` is empty,
 * whitespace (character code-point <= ' ') is used for identifying tokens.
 * Also, any consecutive delimiters found in a string are ignored.
 * This means that only empty strings or null rows will result in a token count of 0.
 *
 * @code{.pseudo}
 * Example:
 * s = ["a", "b c", " ", "d e f"]
 * t = count_tokens(s)
 * t is now [1, 2, 0, 3]
 * @endcode
 *
 * All null row entries are ignored and the output contains all valid rows.
 * The number of tokens for a null element is set to 0 in the output column.
 *
 * @param strings Strings column to use for this operation.
 * @param delimiter Strings used to separate each string into tokens.
 *                  The default of empty string will separate tokens using whitespace.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New INT32 column of token counts.
 */
std::unique_ptr<cudf::column> count_tokens(
  cudf::strings_column_view const& strings,
  cudf::string_scalar const& delimiter = cudf::string_scalar{""},
  rmm::mr::device_memory_resource* mr  = rmm::mr::get_current_device_resource());

/**
 * @brief Returns the number of tokens in each string of a strings column
 * by using multiple strings delimiters to identify tokens in each string.
 *
 * Also, any consecutive delimiters found in a string are ignored.
 * This means that only empty strings or null rows will result in a token count of 0.
 *
 * @code{.pseudo}
 * Example:
 * s = ["a", "b c", "d.e:f;"]
 * d = [".", ":", ";"]
 * t = count_tokens(s,d)
 * t is now [1, 1, 3]
 * @endcode
 *
 * All null row entries are ignored and the output contains all valid rows.
 * The number of tokens for a null element is set to 0 in the output column.
 *
 * @throw cudf::logic_error if the delimiters column is empty or contains nulls.
 *
 * @param strings Strings column to use for this operation.
 * @param delimiters Strings used to separate each string into tokens.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New INT32 column of token counts.
 */
std::unique_ptr<cudf::column> count_tokens(
  cudf::strings_column_view const& strings,
  cudf::strings_column_view const& delimiters,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Returns a single column of strings by converting each character to a string.
 *
 * Each string is converted to multiple strings -- one for each character.
 * Note that a character maybe more than one byte.
 *
 * @code{.pseudo}
 * Example:
 * s = ["hello world", null, "goodbye"]
 * t = character_tokenize(s)
 * t is now ["h","e","l","l","o"," ","w","o","r","l","d","g","o","o","d","b","y","e"]
 * @endcode
 *
 * All null row entries are ignored and the output contains all valid rows.
 *
 * @param strings Strings column to tokenize.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New strings columns of tokens.
 */
std::unique_ptr<cudf::column> character_tokenize(
  cudf::strings_column_view const& strings,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Creates a strings column from a strings column of tokens and an
 * associated column of row ids.
 *
 * Multiple tokens from the input column may be combined into a single row (string)
 * in the output column. The tokens are concatenated along with the `separator` string
 * in the order in which they appear in the `row_indices` column.
 *
 * @code{.pseudo}
 * Example:
 * s = ["hello", "world", "one", "two", "three"]
 * r = [0, 0, 1, 1, 1]
 * s1 = detokenize(s,r)
 * s1 is now ["hello world", "one two three"]
 * r = [0, 2, 1, 1, 0]
 * s2 = detokenize(s,r)
 * s2 is now ["hello three", "one two", "world"]
 * @endcode
 *
 * All null row entries are ignored and the output contains all valid rows.
 * The values in `row_indices` are expected to have positive, sequential
 * values without any missing row indices otherwise the output is undefined.
 *
 * @throw cudf::logic_error is `separator` is invalid
 * @throw cudf::logic_error if `row_indices.size() != strings.size()`
 * @throw cudf::logic_error if `row_indices` contains nulls
 *
 * @param strings Strings column to detokenize.
 * @param row_indices The relative output row index assigned for each token in the input column.
 * @param separator String to append after concatenating each token to the proper output row.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New strings columns of tokens.
 */
std::unique_ptr<cudf::column> detokenize(
  cudf::strings_column_view const& strings,
  cudf::column_view const& row_indices,
  cudf::string_scalar const& separator = cudf::string_scalar(" "),
  rmm::mr::device_memory_resource* mr  = rmm::mr::get_current_device_resource());

/** @} */  // end of tokenize group
}  // namespace nvtext
