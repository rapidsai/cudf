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

//! NVText APIs
namespace nvtext {
/**
 * @addtogroup nvtext_replace
 * @{
 * @file
 */

/**
 * @brief Replaces specified tokens with corresponding replacement strings.
 *
 * Tokens are identified in each string and if any match the specified `targets`
 * strings, they are replaced with corresponding `replacements` string such that
 * if `targets[i]` is found, then it is replaced by `replacements[i]`.
 *
 * The `delimiter` may be zero or more characters. If the `delimiter` is empty,
 * whitespace (character code-point <= ' ') is used for identifying tokens.
 * Also, any consecutive delimiters found in a string are ignored.
 *
 * @code{.pseudo}
 * Example:
 * s = ["this is me", "theme music"]
 * tgt = ["is", "me"]
 * rpl = ["+", "_"]
 * result = replace_tokens(s,tgt,rpl)
 * result is now ["this + _", "theme music"]
 * @endcode
 *
 * A null input element at row `i` produces a corresponding null entry
 * for row `i` in the output column.
 *
 * An empty string is allowed for a replacement string but the delimiters
 * will not be removed.
 *
 * @code{.pseudo}
 * Example:
 * s = ["this is me", "theme music"]
 * tgt = ["me", "this"]
 * rpl = ["", ""]
 * result = replace_tokens(s,tgt,rpl)
 * result is now [" is ", "theme music"]
 * @endcode
 *
 * Note the first string in `result` still retains the space delimiters.
 *
 * The `replacements.size()` must equal `targets.size()` unless `replacements.size()==1`.
 * In this case, all matching `targets` strings will be replaced with the
 * single `replacements[0]` string.
 *
 * @throw cudf::logic_error if `targets.size() != replacements.size()` and
 *                          if `replacements.size() != 1`
 * @throw cudf::logic_error if targets or replacements contain nulls
 * @throw cudf::logic_error if delimiter is invalid
 *
 * @param strings Strings column to replace.
 * @param targets Strings to compare against tokens found in `strings`
 * @param replacements Replacement strings for each string in `targets`
 * @param delimiter Characters used to separate each string into tokens.
 *                  The default of empty string will identify tokens using whitespace.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New strings columns of with replaced strings.
 */
std::unique_ptr<cudf::column> replace_tokens(
  cudf::strings_column_view const& strings,
  cudf::strings_column_view const& targets,
  cudf::strings_column_view const& replacements,
  cudf::string_scalar const& delimiter = cudf::string_scalar{""},
  rmm::mr::device_memory_resource* mr  = rmm::mr::get_current_device_resource());

/**
 * @brief Removes tokens whose lengths are less than a specified number of characters.
 *
 * Tokens identified in each string are removed from the corresponding output string.
 * The removed tokens can be replaced by specifying a `replacement` string as well.
 *
 * The `delimiter` may be zero or more characters. If the `delimiter` is empty,
 * whitespace (character code-point <= ' ') is used for identifying tokens.
 * Also, any consecutive delimiters found in a string are ignored.
 *
 * @code{.pseudo}
 * Example:
 * s = ["this is me", "theme music"]
 * result = filter_tokens(s,3)
 * result is now ["this  ", "theme music"]
 * @endcode
 *
 * Note the first string in `result` still retains the space delimiters.
 *
 * Example with a `replacement` string.
 *
 * @code{.pseudo}
 * Example:
 * s = ["this is me", "theme music"]
 * result = filter_tokens(s,5,"---")
 * result is now ["--- --- ---", "theme music"]
 * @endcode
 *
 * The `replacement` string is allowed to be shorter than min_token_length.
 *
 * @throw cudf::logic_error if `delimiter` or `replacement` is invalid
 *
 * @param strings Strings column to replace.
 * @param min_token_length The minimum number of characters to retain a token in the output string.
 * @param replacement Optional replacement string to be used in place of removed tokens.
 * @param delimiter Characters used to separate each string into tokens.
 *                  The default of empty string will identify tokens using whitespace.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New strings columns of with replaced strings.
 */
std::unique_ptr<cudf::column> filter_tokens(
  cudf::strings_column_view const& strings,
  cudf::size_type min_token_length,
  cudf::string_scalar const& replacement = cudf::string_scalar{""},
  cudf::string_scalar const& delimiter   = cudf::string_scalar{""},
  rmm::mr::device_memory_resource* mr    = rmm::mr::get_current_device_resource());

/** @} */  // end of group
}  // namespace nvtext
