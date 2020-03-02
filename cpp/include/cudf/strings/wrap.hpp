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

#include <cudf/strings/strings_column_view.hpp>
#include <cudf/column/column.hpp>

namespace cudf
{
namespace strings
{
/**
 * @brief Inserts new-line characters (ASCII 0x0A) into each string in place of spaces, depending on `width`.  
 *
 * Returns a column of strings where, for each string row in the input, 
 * words separated by spaces will become separated by newline characters,
 * as follows: if the string is longer than width, a new-line is inserted 
 * at space positions so that each line is no more than width characters 
 * without truncating any non-space character sequences.
 * Attempts to make each line less than or equal to width characters. 
 * If a string or characters is longer than width, 
 * the line is split on the next closest space character. 
 *
 * Any null string entries return corresponding null output column entries.
 *
 * Example 1:
 * ```
 * width = 5
 * input_string_tbl = ["tesT1 test2", "more longtest short1", " other test "];
 *
 * wrapped_string_tbl = wrap(input_string_tbl, width)
 * wrapped_string_tbl = ["test1\ntest2", "more\nlongt\nest\nshort1", "other\ntest"]
 * ```
 *
 * Example 2:
 * ```
 * width = 12;
 * input_string_tbl = ["the quick brown fox jumped over the lazy brown dog", "hello, world"]
 * 
 * wrapped_string_tbl = wrap(input_string_tbl, width)
 * wrapped_string_tbl = ["the quick\nbrown fox\njumped over\nthe lazy\nbrown dog", "hello, world"] 
 * ```
 *
 * @param[in] strings String column.
 * @param[in] width Maximum chararacter width of a line within each string. 
 * @param[in] mr Resource for allocating device memory.
 * @return Column of wrapped strings.
 */
std::unique_ptr<column> wrap( strings_column_view const& strings,
                              size_type width,
                              rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());
  
}//namespace strings
}//namespace cudf
