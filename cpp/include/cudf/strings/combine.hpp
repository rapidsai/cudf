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
#include <cudf/table/table_view.hpp>

namespace cudf {
namespace strings {
/**
 * @addtogroup strings_combine
 * @{
 */

/**
 * @brief Row-wise concatenates the given list of strings columns and
 * returns a single strings column result.
 *
 * Each new string is created by concatenating the strings from the same
 * row delimited by the separator provided.
 *
 * Any row with a null entry will result in the corresponding output
 * row to be null entry unless a narep string is specified to be used
 * in its place.
 *
 * The number of strings in the columns provided must be the same.
 *
 * @code{.pseudo}
 * Example:
 * s1 = ['aa', null, '', 'aa']
 * s2 = ['', 'bb', 'bb', null]
 * r1 = concatenate([s1,s2])
 * r1 is ['aa', null, 'bb', null]
 * r2 = concatenate([s1,s2],':','_')
 * r2 is ['aa:', '_:bb', ':bb', 'aa:_']
 * @endcode
 *
 * @throw cudf::logic_error if input columns are not all strings columns.
 * @throw cudf::logic_error if separator is not valid.
 *
 * @param strings_columns List of string columns to concatenate.
 * @param separator String that should inserted between each string from each row.
 *        Default is an empty string.
 * @param narep String that should be used in place of any null strings
 *        found in any column. Default of invalid-scalar means any null entry in any column will
 *        produces a null result for that row.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New column with concatenated results.
 */
std::unique_ptr<column> concatenate(
  table_view const& strings_columns,
  string_scalar const& separator      = string_scalar(""),
  string_scalar const& narep          = string_scalar("", false),
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Concatenates all strings in the column into one new string delimited
 * by an optional separator string.
 *
 * This returns a column with one string. Any null entries are ignored unless
 * the narep parameter specifies a replacement string.
 *
 * @code{.pseudo}
 * Example:
 * s = ['aa', null, '', 'zz' ]
 * r = join_strings(s,':','_')
 * r is ['aa:_::zz']
 * @endcode
 *
 * @throw cudf::logic_error if separator is not valid.
 *
 * @param strings Strings for this operation.
 * @param separator String that should inserted between each string.
 *        Default is an empty string.
 * @param narep String that should represent any null strings found.
 *        Default of invalid-scalar will ignore any null entries.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New column containing one string.
 */
std::unique_ptr<column> join_strings(
  strings_column_view const& strings,
  string_scalar const& separator      = string_scalar(""),
  string_scalar const& narep          = string_scalar("", false),
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Concatenates a list of strings columns using separators for each row
 * and returns the result as a strings column.
 *
 * Each new string is created by concatenating the strings from the same
 * row delimited by the row separator provided for that row. The following rules
 * are applicable:
 *
 * - If row separator for a given row is null, output column for that row is null, unless
 *   there is a valid @p separator_narep
 * - If all column values for a given row is null, output column for that row is null, unless
 *   there is a valid @p col_narep
 * - null column values for a given row are skipped, if the column replacement isn't valid
 * - The separator is only applied between two valid column values
 * - If valid @p separator_narep and @p col_narep are provided, the output column is always
 *   non nullable
 *
 * @code{.pseudo}
 * Example:
 * c0   = ['aa', null, '',  'ee',  null, 'ff']
 * c1   = [null, 'cc', 'dd', null, null, 'gg']
 * c2   = ['bb', '',   null, null, null, 'hh']
 * sep  = ['::', '%%', '^^', '!',  '*',  null]
 * out0 = concatenate([c0, c1, c2], sep)
 * out0 is ['aa::bb', 'cc%%', '^^dd', 'ee', null, null]
 *
 * sep_rep = '+'
 * out1    = concatenate([c0, c1, c2], sep, sep_rep)
 * out1 is ['aa::bb', 'cc%%', '^^dd', 'ee', null, 'ff+gg+hh']
 *
 * col_rep = '-'
 * out2    = concatenate([c0, c1, c2], sep, invalid_sep_rep, col_rep)
 * out2 is ['aa::-::bb', '-%%cc%%', '^^dd^^-', 'ee!-!-', '-*-*-', null]
 * @endcode
 *
 * @throw cudf::logic_error if no input columns are specified - table view is empty
 * @throw cudf::logic_error if input columns are not all strings columns.
 * @throw cudf::logic_error if the number of rows from @p separators and @p strings_columns
 *                          do not match
 *
 * @param strings_columns List of strings columns to concatenate.
 * @param separators Strings column that provides the separator for a given row
 * @param separator_narep String that should be used in place of a null separator for a given
 *        row. Default of invalid-scalar means no row separator value replacements.
 *        Default is an invalid string.
 * @param col_narep String that should be used in place of any null strings
 *        found in any column. Default of invalid-scalar means no null column value replacements.
 *        Default is an invalid string.
 * @param mr Resource for allocating device memory.
 * @return New column with concatenated results.
 */
std::unique_ptr<column> concatenate(
  table_view const& strings_columns,
  strings_column_view const& separators,
  string_scalar const& separator_narep = string_scalar("", false),
  string_scalar const& col_narep       = string_scalar("", false),
  rmm::mr::device_memory_resource* mr  = rmm::mr::get_default_resource());

/** @} */  // end of doxygen group
}  // namespace strings
}  // namespace cudf
