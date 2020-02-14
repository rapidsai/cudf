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
#include <cudf/strings/strings_column_view.hpp>

namespace cudf
{
namespace strings
{

/**
 * @brief Translates individual characters within each string.
 *
 * This can also be used to remove a character by specifying 0 for the corresponding table entry.
 *
 * Null string entries result in null entries in the output column.
 *
 * ```
 * s = ["aa","bbb","cccc","abcd"]
 * t = [['a','A'],['b',''],['d':'Q']]
 * r = translate(s,t)
 * r is now ["AA", "", "cccc", "AcQ"]
 * ```
 *
 * @param strings Strings instance for this operation.
 * @param chars_table Table of UTF-8 character mappings.
 * @return New column with padded strings.
 */
std::unique_ptr<column> translate( strings_column_view const& strings,
                                   std::vector<std::pair<char_utf8,char_utf8>> const& chars_table,
                                   rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource() );

} // namespace strings
} // namespace cudf
