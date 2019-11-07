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

#include <cudf/strings/strings_column_view.hpp>
#include <cudf/column/column.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/scalar/scalar.hpp>

namespace cudf
{
namespace strings
{

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
 * ```
 * s1 = ['aa', null, '', 'aa']
 * s2 = ['', 'bb', 'bb', null]
 * r1 = concatenate([s1,s2])
 * r1 is ['aa', null, 'bb', null]
 * r2 = concatenate([s1,s2],':','_')
 * r2 is ['aa:', '_:bb', ':bb', 'aa:_']
 * ```
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
 * @param mr Resource for allocating device memory.
 * @return New column with concatenated results.
 */
std::unique_ptr<column> concatenate( table_view const& strings_columns,
                                     string_scalar const& separator = string_scalar(""),
                                     string_scalar const& narep = string_scalar("",false),
                                     rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Concatenates all strings in the column into one new string delimited
 * by an optional separator string.
 *
 * This returns a column with one string. Any null entries are ignored unless
 * the narep parameter specifies a replacement string.
 *
 * ```
 * s = ['aa', null, '', 'zz' ]
 * r = join_strings(s,':','_')
 * r is ['aa:_::zz']
 * ```
 *
 * @throw cudf::logic_error if separator is not valid.
 *
 * @param strings Strings for this operation.
 * @param separator String that should inserted between each string.
 *        Default is an empty string.
 * @param narep String that should represent any null strings found.
 *        Default of invalid-scalar will ignore any null entries.
 * @param mr Resource for allocating device memory.
 * @return New column containing one string.
 */
std::unique_ptr<column> join_strings( strings_column_view const& strings,
                                      string_scalar const& separator = string_scalar(""),
                                      string_scalar const& narep = string_scalar("",false),
                                      rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

} // namespace strings
} // namespace cudf
