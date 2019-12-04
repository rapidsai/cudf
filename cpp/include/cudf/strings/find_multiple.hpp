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

namespace cudf
{
namespace strings
{

/**
 * @brief Returns a column with character position values where each
 * of the target strings are found in each string.
 *
 * The size of the output column is targets.size() * strings.size().
 * output[i] contains the position of target[i % targets.size()] in string[i/targets.size()]
 *
 * ```
 * s = ["abc","def"]
 * t = ["a","c","e"]
 * r = find_multiple(s,t)
 * r is now [ 0, 2,-1,   // for "abc": "a" at pos 0, "c" at pos 2, "e" not found
 *           -1,-1, 1 ]  // for "def": "a" and "b" not found, "e" at  pos 1
 * ```
 *
 * @throw cudf::logic_error targets is empty or contains nulls
 *
 * @param strings Strings instance for this operation.
 * @param targets Strings to search for in each string.
 * @param mr Resource for allocating device memory.
 * @return New integer column with character position values.
 */
std::unique_ptr<column> find_multiple( strings_column_view const& strings,
                                       strings_column_view const& targets,
                                       rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

} // namespace strings
} // namespace cudf
