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
#include <cudf/table/table.hpp>

namespace cudf
{
namespace strings
{

/**
 * @brief Returns a vector of strings columns for each matching group specified in the given regular expression pattern.
 *
 * All the strings for the first group will go in the first output column; the second group
 * go in the second column and so on. Null entries are added if the string does match.
 *
 * Any null string entries return corresponding null output column entries.
 *
 * ```
 * s = ["a1","b2","c3"]
 * r = extract(s,"([ab])(\\d)")
 * r is now [["a","b",null],
 *           ["1","2",null]]
 * ```
 *
 * @param strings Strings instance for this operation.
 * @param pattern The regular expression pattern with group indicators.
 * @param mr Resource for allocating device memory.
 * @return Columns of strings extracted from the input column.
 */
std::unique_ptr<experimental::table> extract( strings_column_view const& strings,
                                              std::string const& pattern,
                                              rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());


} // namespace strings
} // namespace cudf
