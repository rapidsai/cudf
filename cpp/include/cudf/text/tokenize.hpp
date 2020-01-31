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

namespace cudf
{
namespace nvtext
{

/**
 * @brief Returns a single column of strings by tokenizing the strings
 * in the given column using the provided characters as delimiters.
 *
 * As each string is tokenized, the tokens are appended in row order
 * to build the output column.
 *
 * Example:
 * ```
 * s = ["a", "b c", "d  e f "]
 * t = tokenize(s)
 * t is now ["a","b","c","d","e","f"]
 * ```
 *
 * All null row entries are ignored and the output contains all valid rows.
 *
 * @param strings Strings column to use for this operation.
 * @param delimiter UTF-8 characters used to separate each string into tokens.
 *                  The default of empty string will separate tokens using whitespace.
 * @param mr Resource for allocating device memory.
 * @return New strings columns of tokens.
 */
std::unique_ptr<column> tokenize( strings_column_view const& strings,
                                  string_scalar const& delimiter = string_scalar{""},
                                  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Returns a single column of strings by tokenizing the strings
 * in the given column using multiple strings delimiters.
 *
 * As each string is tokenized, the tokens are appended in row order
 * to build the output column.
 *
 * Example:
 * ```
 * s = ["a", "b c", "d,e:f;"]
 * d = [",",":",";"]
 * t = tokenize(s,d)
 * t is now ["a","b c","d","e","f"]
 * ```
 *
 * All null row entries are ignored and the output contains all valid rows.
 * 
 * @throw cudf::logic_error if the delimiters column is empty or contains nulls.
 *
 * @param strings Strings column to use for this operation.
 * @param delimiters Strings used to separate individual strings into tokens.
 * @param mr Resource for allocating device memory.
 * @return New strings columns of tokens.
 */
std::unique_ptr<column> tokenize( strings_column_view const& strings,
                                  strings_column_view const& delimiters,
                                  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Returns the number of tokens in each string of a strings column.
 *
 * Example:
 * ```
 * s = ["a", "b c", " ", "d e f"]
 * t = token_count(s)
 * t is now [1,2,0,3]
 * ```
 *
 * All null row entries are ignored and the output contains all valid rows.
 *
 * @param strings Strings column to use for this operation.
 * @param delimiter Strings used to separate each string into tokens.
 *                  The default of empty string will separate tokens using whitespace.
 * @param mr Resource for allocating device memory.
 * @return New INT32 column of token counts.
 */
std::unique_ptr<column> token_count( strings_column_view const& strings,
                                     string_scalar const& delimiter = string_scalar{""},
                                     rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Returns the number of tokens in a strings column
 * using multiple delimiters to tokenize each string.
 *
 * Example:
 * ```
 * s = ["a", "b c", "d,e:f;"]
 * d = [",",":",";"]
 * t = token_count(s,d)
 * t is now [1,1,3]
 * ```
 *
 * All null row entries are ignored and the output contains all valid rows.
 *
 * @throw cudf::logic_error if the delimiters column is empty or contains nulls.
 *
 * @param strings Strings column to use for this operation.
 * @param delimiters Strings used to separate each string into tokens.
 * @param mr Resource for allocating device memory.
 * @return New INT32 column of counts.
 */
std::unique_ptr<column> token_count( strings_column_view const& strings,
                                     strings_column_view const& delimiters,
                                     rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

} // namespace nvtext
} // namespace cudf
