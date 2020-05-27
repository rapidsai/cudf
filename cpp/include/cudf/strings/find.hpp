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
 * @addtogroup strings_find
 * @{
 */

/**
 * @brief Returns a column of character position values where the target
 * string is first found in each string of the provided column.
 *
 * If `target` is not found, -1 is returned for that row entry in the output column.
 *
 * The target string is searched within each string in the character
 * position range [start,stop). If the stop parameter is -1, then the
 * end of each string becomes the final position to include in the search.
 *
 * Any null string entries return corresponding null output column entries.
 *
 * @throw cudf::logic_error if start position is greater than stop position.
 *
 * @param strings Strings instance for this operation.
 * @param target UTF-8 encoded string to search for in each string.
 * @param start First character position to include in the search.
 * @param stop Last position (exclusive) to include in the search.
 *             Default of -1 will search to the end of the string.
 * @param mr Resource for allocating device memory.
 * @return New integer column with character position values.
 */
std::unique_ptr<column> find(strings_column_view const& strings,
                             string_scalar const& target,
                             size_type start                     = 0,
                             size_type stop                      = -1,
                             rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Returns a column of character position values where the target
 * string is first found searching from the end of each string.
 *
 * If `target` is not found, -1 is returned for that entry.
 *
 * The target string is searched within each string in the character
 * position range [start,stop). If the stop parameter is -1, then the
 * end of each string becomes the final position to include in the search.
 *
 * Any null string entries return corresponding null output column entries.
 *
 * @throw cudf::logic_error if start position is greater than stop position.
 *
 * @param strings Strings instance for this operation.
 * @param target UTF-8 encoded string to search for in each string.
 * @param start First position to include in the search.
 * @param stop Last position (exclusive) to include in the search.
 *             Default of -1 will search starting at the end of the string.
 * @param mr Resource for allocating device memory.
 * @return New integer column with character position values.
 */
std::unique_ptr<column> rfind(
  strings_column_view const& strings,
  string_scalar const& target,
  size_type start                     = 0,
  size_type stop                      = -1,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());
/**
 * @brief Returns a column of boolean values for each string where true indicates
 * the target string was found within that string in the provided column.
 *
 * If the `target` is not found for a string, false is returned for that entry in the output column.
 * If `target` is an empty string, true is returned for all non-null entries in the output column.
 *
 * Any null string entries return corresponding null entries in the output columns.
 *
 * @param strings Strings instance for this operation.
 * @param target UTF-8 encoded string to search for in each string.
 * @param mr Resource for allocating device memory.
 * @return New BOOL8 column.
 */
std::unique_ptr<column> contains(
  strings_column_view const& strings,
  string_scalar const& target,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Returns a column of boolean values for each string where true indicates
 * the target string was found at the beginning of that string in the provided column.
 *
 * If `target` is not found at the beginning of a string, false is set for
 * that row entry in the output column.
 * If `target` is an empty string, true is returned for all non-null entries in the output column.
 *
 * Any null string entries return corresponding null entries in the output columns.
 *
 * @param strings Strings instance for this operation.
 * @param target UTF-8 encoded string to search for in each string.
 * @param mr Resource for allocating device memory.
 * @return New BOOL8 column.
 */
std::unique_ptr<column> starts_with(
  strings_column_view const& strings,
  string_scalar const& target,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Returns a column of boolean values for each string where true indicates
 * the target string was found at the end of that string in the provided column.
 *
 * If `target` is not found at the end of a string, false is set for
 * that row entry in the output column.
 * If `target` is an empty string, true is returned for all non-null entries in the output column.
 *
 * Any null string entries return corresponding null entries in the output columns.
 *
 * @param strings Strings instance for this operation.
 * @param target UTF-8 encoded string to search for in each string.
 * @param mr Resource for allocating device memory.
 * @return New BOOL8 column.
 */
std::unique_ptr<column> ends_with(
  strings_column_view const& strings,
  string_scalar const& target,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Returns a column of strings that searches for the @p delimiter @p count number of
 * times in the source @p strings forward if @p count is positive or backwards if @p count is
 * negative. If @p count is positive, it returns a substring from the start of the source @p
 * strings up until @p count occurrence of the @delimiter not including the @p delimiter.
 * If @p count is negative, it returns a substring from the start of the @p count occurrence of
 * the @delimiter in the source @p strings past the delimiter until the end of the string.
 *
 * The search for @delimiter in @p strings is case sensitive.
 * If the @p count is 0, every row in the output column will be null.
 * If the row value of @p strings is null, the row value in the output column will be null.
 * If the @p delimiter is invalid or null, every row in the output column will be null.
 * If the @p delimiter or the column value for a row is empty, the row value in the output
 * column will be empty.
 * If @p count occurrences of @p delimiter isn't found, the row value in the output column will
 * be the row value from the input @p strings column.
 *
 * @code{.pseudo}
 * Example:
 * in_s = ['www.nvidia.com', null, 'www.google.com', '', 'foo' ]
 * r = substring_index(in_s, '.', 1)
 * r is ['www', null, 'www', '', 'foo']
 *
 * in_s = ['www.nvidia.com', null, 'www.google.com', '', 'foo' ]
 * r = substring_index(in_s, '.', -2)
 * r is ['nvidia.com', null, 'google.com', '', 'foo']
 * @endcode
 *
 * @param strings Strings instance for this operation.
 * @param delimiter UTF-8 encoded string to search for in each string.
 * @param count Number of times to search for delimiter in each string. If the value is positive,
 *              forward search of delimiter is performed; else, a backward search is performed.
 * @param mr Resource for allocating device memory.
 * @return New strings column containing the substrings.
 */
std::unique_ptr<column> substring_index(
  strings_column_view const& strings,
  string_scalar const& delimiter,
  size_type count,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Returns a column of strings that searches the delimiter for each row from
 * @p delimiter_strings @p count number of times in the source @p strings forward if @p count
 * is positive or backwards if @p count is negative. If @p count is positive, it returns a
 * substring from the start of the source @p strings up until @p count occurrence of the
 * delimiter for that row not including that delimiter. If @p count is negative, it returns a
 * substring from the start of the @p count occurrence of the delimiter for that row in the
 * source @p strings past the delimiter until the end of the string.
 *
 * The search for @p delimiter_strings in @p strings is case sensitive.
 * If the @p count is 0, every row in the output column will be null.
 * If the row value of @p strings is null, the row value in the output column will be null.
 * If the row value from @p delimiter_strings is invalid or null, the row value in the
 * output column will be null.
 * If the row value from @p delimiter_strings or the column value for a row is empty, the
 * row value in the output column will be empty.
 * If @p count occurrences of delimiter isn't found, the row value in the output column will
 * be the row value from the input @p strings column.
 *
 * @code{.pseudo}
 * Example:
 * in_s = ['www.nvidia.com', null, 'www.google.com', '', 'foo..bar....goo' ]
 * delimiters = ['.', '..', '', null, '..']
 * r = substring_index(in_s, delimiters, 2)
 * r is ['www.nvidia', null, '', null, 'foo..bar']
 *
 * in_s = ['www.nvidia.com', null, 'www.google.com', '', 'foo..bar....goo', 'apache.org' ]
 * delimiters = ['.', '..', '', null, '..', '.']
 * r = substring_index(in_s, delimiters, -2)
 * r is ['nvidia.com', null, '', null, '..goo', 'apache.org']
 * @endcode
 *
 * @throw cudf::logic_error if the number of rows in @p strings and @delimiter_strings do not match.
 *
 * @param strings Strings instance for this operation.
 * @param delimiter_strings UTF-8 encoded string for each row.
 * @param count Number of times to search for delimiter in each string. If the value is positive,
 *              forward search of delimiter is performed; else, a backward search is performed.
 * @param mr Resource for allocating device memory.
 * @return New strings column containing the substrings.
 */
std::unique_ptr<column> substring_index(
  strings_column_view const& strings,
  strings_column_view const& delimiter_strings,
  size_type count,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/** @} */  // end of doxygen group
}  // namespace strings
}  // namespace cudf
