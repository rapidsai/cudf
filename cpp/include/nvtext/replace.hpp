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
 */

/**
 * @brief Replaces specified tokens with corresponding replacement strings.
 *
 * Tokens are identified in each string and if any match the specified `targets`
 * strings, they are replaced with corresponding `repls` string.
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
 * There is no guaranteed order for replacing the targets. This means if
 * one target string is a substring over another (e.g. "the" is substring of "theme")
 * there is no guarantee that "the" will be replaced before "theme" or vice versa.
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
 * The `repls.size() == targets.size()` except of the `repls.size()==1`.
 * In this case, all matching `targets` strings will be replaced with the
 * single `repls` string.
 *
 * @throw cudf::logic_error if `targets.size() != repls.size()` && if `repls.size() != 1`
 * @throw cudf::logic_error if targets or repls contain nulls
 * @throw cudf::logic_error if delimiter is invalid
 *
 * @param strings Strings column to replace.
 * @param targets Strings to compare against tokens found in `strings`
 * @param repls Replacement strings for each string in `targets`
 * @param delimiter Characters used to separate each string into tokens.
 *                  The default of empty string will identify tokens using whitespace.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New strings columns of with replaced strings.
 */
std::unique_ptr<cudf::column> replace_tokens(
  cudf::strings_column_view const& strings,
  cudf::strings_column_view const& targets,
  cudf::strings_column_view const& repls,
  cudf::string_scalar const& delimiter = cudf::string_scalar{""},
  rmm::mr::device_memory_resource* mr  = rmm::mr::get_default_resource());

/** @} */  // end of group
}  // namespace nvtext
