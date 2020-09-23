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

namespace cudf {
namespace strings {
/**
 * @addtogroup strings_replace
 * @{
 * @file
 */

/**
 * @brief Replaces target string within each string with the specified
 * replacement string.
 *
 * This function searches each string in the column for the target string.
 * If found, the target string is replaced by the repl string within the
 * input string. If not found, the output entry is just a copy of the
 * corresponding input string.
 *
 * Specifing an empty string for repl will essentially remove the target
 * string if found in each string.
 *
 * Null string entries will return null output string entries.
 *
 * @code{.pseudo}
 * Example:
 * s = ["hello", "goodbye"]
 * r1 = replace(s,"o","OOO")
 * r1 is now ["hellOOO","gOOOOOOdbye"]
 * r2 = replace(s,"oo","")
 * r2 is now ["hello","gdbye"]
 * @endcode
 *
 * @throw cudf::logic_error if target is an empty string.
 *
 * @param strings Strings column for this operation.
 * @param target String to search for within each string.
 * @param repl Replacement string if target is found.
 * @param maxrepl Maximum times to replace if target appears multiple times in the input string.
 *        Default of -1 specifies replace all occurrences of target in each string.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New strings column.
 */
std::unique_ptr<column> replace(
  strings_column_view const& strings,
  string_scalar const& target,
  string_scalar const& repl,
  int32_t maxrepl                     = -1,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief This function replaces each string in the column with the provided
 * repl string within the [start,stop) character position range.
 *
 * Null string entries will return null output string entries.
 *
 * Position values are 0-based meaning position 0 is the first character
 * of each string.
 *
 * This function can be used to insert a string into specific position
 * by specifying the same position value for start and stop. The repl
 * string can be appended to each string by specifying -1 for both
 * start and stop.
 *
 * @code{.pseudo}
 * Example:
 * s = ["abcdefghij","0123456789"]
 * r = s.replace_slice(s,2,5,"z")
 * r is now ["abzfghij", "01z56789"]
 * @endcode
 *
 * @throw cudf::logic_error if start is greater than stop.
 *
 * @param strings Strings column for this operation.
 * @param repl Replacement string for specified positions found.
 *        Default is empty string.
 * @param start Start position where repl will be added.
 *        Default is 0, first character position.
 * @param stop End position (exclusive) to use for replacement.
 *        Default of -1 specifies the end of each string.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New strings column.
 */
std::unique_ptr<column> replace_slice(
  strings_column_view const& strings,
  string_scalar const& repl           = string_scalar(""),
  size_type start                     = 0,
  size_type stop                      = -1,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Replaces substrings matching a list of targets with the corresponding
 * replacement strings.
 *
 * For each string in strings, the list of targets is searched within that string.
 * If a target string is found, it is replaced by the corresponding entry in the repls column.
 * All occurrences found in each string are replaced.
 *
 * This does not use regex to match targets in the string.
 *
 * Null string entries will return null output string entries.
 *
 * The repls argument can optionally contain a single string. In this case, all
 * matching target substrings will be replaced by that single string.
 *
 * @code{.pseudo}
 * Example:
 * s = ["hello", "goodbye"]
 * tgts = ["e","o"]
 * repls = ["EE","OO"]
 * r1 = replace(s,tgts,repls)
 * r1 is now ["hEEllO", "gOOOOdbyEE"]
 * tgts = ["e","oo"]
 * repls = ["33",""]
 * r2 = replace(s,tgts,repls)
 * r2 is now ["h33llo", "gdby33"]
 * @endcode
 *
 * @throw cudf::logic_error if targets and repls are different sizes except
 * if repls is a single string.
 * @throw cudf::logic_error if targets or repls contain null entries.
 *
 * @param strings Strings column for this operation.
 * @param targets Strings to search for in each string.
 * @param repls Corresponding replacement strings for target strings.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New strings column.
 */
std::unique_ptr<column> replace(
  strings_column_view const& strings,
  strings_column_view const& targets,
  strings_column_view const& repls,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Replaces any null string entries with the given string.
 *
 * This returns a strings column with no null entries.
 *
 * @code{.pseudo}
 * Example:
 * s = ["hello", nullptr, "goodbye"]
 * r = replace_nulls(s,"**")
 * r is now ["hello", "**", "goodbye"]
 * @endcode
 *
 * @param strings Strings column for this operation.
 * @param repl Replacement string for null entries. Default is empty string.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New strings column.
 */
std::unique_ptr<column> replace_nulls(
  strings_column_view const& strings,
  string_scalar const& repl           = string_scalar(""),
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/** @} */  // end of doxygen group
}  // namespace strings
}  // namespace cudf
