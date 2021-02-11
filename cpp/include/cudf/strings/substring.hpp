/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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
 * @addtogroup strings_substring
 * @{
 * @file
 */

/**
 * @brief Returns a new strings column that contains substrings of the
 * strings in the provided column.
 *
 * The character positions to retrieve in each string are `[start,stop)`.
 * If the start position is outside a string's length, an empty
 * string is returned for that entry. If the stop position is past the
 * end of a string's length, the end of the string is used for
 * stop position for that string.
 *
 * Null string entries will return null output string entries.
 *
 * @code{.pseudo}
 * Example:
 * s = ["hello", "goodbye"]
 * r = substring(s,2,6)
 * r is now ["llo","odby"]
 * r2 = substring(s,2,5,2)
 * r2 is now ["lo","ob"]
 * @endcode
 *
 * @param strings Strings column for this operation.
 * @param start First character position to begin the substring.
 * @param stop Last character position (exclusive) to end the substring.
 * @param step Distance between input characters retrieved.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New strings column with sorted elements of this instance.
 */
std::unique_ptr<column> slice_strings(
  strings_column_view const& strings,
  numeric_scalar<size_type> const& start = numeric_scalar<size_type>(0, false),
  numeric_scalar<size_type> const& stop  = numeric_scalar<size_type>(0, false),
  numeric_scalar<size_type> const& step  = numeric_scalar<size_type>(1),
  rmm::mr::device_memory_resource* mr    = rmm::mr::get_current_device_resource());

/**
 * @brief Returns a new strings column that contains substrings of the
 * strings in the provided column using unique ranges for each string.
 *
 * The character positions to retrieve in each string are specified in
 * the `starts` and `stops` integer columns.
 * If a start position is outside a string's length, an empty
 * string is returned for that entry. If a stop position is past the
 * end of a string's length, the end of the string is used for
 * stop position for that string. Any stop position value set to -1 will
 * indicate to use the end of the string as the stop position for that
 * string.
 *
 * Null string entries will return null output string entries.
 *
 * The starts and stops column must both be the same integer type and
 * must be the same size as the strings column.
 *
 * @code{.pseudo}
 * Example:
 * s = ["hello", "goodbye"]
 * starts = [ 1, 2 ]
 * stops = [ 5, 4 ]
 * r = substring_from(s,starts,stops)
 * r is now ["ello","od"]
 * @endcode
 *
 * @throw cudf::logic_error if starts or stops is a different size than the strings column.
 * @throw cudf::logic_error if starts and stops are not same integer type.
 * @throw cudf::logic_error if starts or stops contains nulls.
 *
 * @param strings Strings column for this operation.
 * @param starts First character positions to begin the substring.
 * @param stops Last character (exclusive) positions to end the substring.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New strings column with sorted elements of this instance.
 */
std::unique_ptr<column> slice_strings(
  strings_column_view const& strings,
  column_view const& starts,
  column_view const& stops,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Slices a column of strings by using a delimiter as a slice point.
 *
 * Returns a column of strings after searching for @p delimiter @p count number of
 * times in the source @p strings from left to right if @p count is positive or from
 * right to left if @p count is negative. If @p count is positive, it returns a substring
 * from the start of the source @p strings up until @p count occurrence of the @delimiter
 * not including the @p delimiter. If @p count is negative, it returns a substring from
 * the start of the @p count occurrence of the @delimiter in the source @p strings past
 * the delimiter until the end of the string.
 *
 * The search for @delimiter in @p strings is case sensitive.
 * If the row value of @p strings is null, the row value in the output column will be null.
 * If the @p count is 0 or if @p delimiter is invalid or empty, every row in the output column
 * will be an empty string.
 * If the column value for a row is empty, the row value in the output column will be empty.
 * If @p count occurrences of @p delimiter isn't found, the row value in the output column will
 * be the row value from the input @p strings column.
 *
 * @code{.pseudo}
 * Example:
 * in_s = ['www.nvidia.com', null, 'www.google.com', '', 'foo']
 * r = slice_strings(in_s, '.', 1)
 * r =    ['www',            null, 'www',            '', 'foo']
 *
 * in_s = ['www.nvidia.com', null, 'www.google.com', '', 'foo']
 * r = slice_strings(in_s, '.', -2)
 * r =    ['nvidia.com',     null, 'google.com',     '', 'foo']
 * @endcode
 *
 * @param strings Strings instance for this operation.
 * @param delimiter UTF-8 encoded string to search for in each string.
 * @param count Number of times to search for delimiter in each string. If the value is positive,
 *              delimiter is searched from left to right; else, it is searched from right to left.
 * @param mr Resource for allocating device memory.
 * @return New strings column containing the substrings.
 */
std::unique_ptr<column> slice_strings(
  strings_column_view const& strings,
  string_scalar const& delimiter,
  size_type count,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Slices a column of strings by using a delimiter column as slice points.
 *
 * Returns a column of strings after searching the delimiter defined per row from
 * @p delimiter_strings @p count number of times in the source @p strings from left to right
 * if @p count is positive or from right to left if @p count is negative. If @p count is
 * positive, it returns a substring from the start of the source @p strings up until
 * @p count occurrence of the delimiter for that row not including that delimiter. If @p count
 * is negative, it returns a substring from the start of the @p count occurrence of the
 * delimiter for that row in the source @p strings past the delimiter until the end of the string.
 *
 * The search for @p delimiter_strings in @p strings is case sensitive.
 * If the @p count is 0, every row in the output column will be an empty string.
 * If the row value of @p strings is null, the row value in the output column will be null.
 * If the row value from @p delimiter_strings is invalid or null, the row value in the
 * output column will be an empty string.
 * If the row value from @p delimiter_strings or the column value for a row is empty, the
 * row value in the output column will be empty.
 * If @p count occurrences of delimiter isn't found, the row value in the output column will
 * be the row value from the input @p strings column.
 *
 * @code{.pseudo}
 * Example:
 * in_s =       ['www.nvidia.com', null, 'www.google.com', 'bar', 'foo..bar....goo']
 * delimiters = ['.',              '..', '',               null,  '..']
 * r = slice_strings(in_s, delimiters, 2)
 * r =          ['www.nvidia',     null, '',               '',   'foo..bar']
 *
 * in_s =       ['www.nvidia.com', null, 'www.google.com', '',  'foo..bar....goo', 'apache.org']
 * delimiters = ['.',              '..', '',               null,'..',              '.']
 * r = slice_strings(in_s, delimiters, -2)
 * r =          ['nvidia.com',     null, '',               '',  '..goo',           'apache.org']
 * @endcode
 *
 * @throw cudf::logic_error if the number of rows in @p strings and @delimiter_strings do not match.
 *
 * @param strings Strings instance for this operation.
 * @param delimiter_strings UTF-8 encoded string for each row.
 * @param count Number of times to search for delimiter in each string. If the value is positive,
 *              delimiter is searched from left to right; else, it is searched from right to left.
 * @param mr Resource for allocating device memory.
 * @return New strings column containing the substrings.
 */
std::unique_ptr<column> slice_strings(
  strings_column_view const& strings,
  strings_column_view const& delimiter_strings,
  size_type count,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Apply a JSONPath string to all rows in an input strings column.
 *
 * Applies a JSONPath string to an incoming strings column where each row in the column
 * is a valid json string.  The output is returned by row as a strings column.
 *
 * https://tools.ietf.org/id/draft-goessner-dispatch-jsonpath-00.html
 * Implements only the operators: $ . [] *
 *
 * @param col The input strings column. Each row must contain a valid json string
 * @param json_path The JSONPath string to be applied to each row
 * @param mr Resource for allocating device memory.
 * @return New strings column containing the retrieved json object strings
 */
std::unique_ptr<cudf::column> get_json_object(
  cudf::strings_column_view const& col,
  std::string const& json_path,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/** @} */  // end of doxygen group
}  // namespace strings
}  // namespace cudf
