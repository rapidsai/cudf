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
 * @brief Wraps strings onto multiple lines shorter than `width` by replacing appropriate white space with new-line characters (ASCII 0x0A).
 *
 * For each string row in the input column longer than `width`, the corresponding output string row 
 * will have newline characters inserted so that each line is no more than `width characters`. Attempts to 
 * use existing white space locations to split the strings, but may split non-white-space sequences if necessary.  
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
