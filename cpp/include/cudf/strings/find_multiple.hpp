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
 * @brief Returns a column with position values where each
 * of the target strings are found in each string.
 *
 * If the string is not found, -1 is returned for that entry.
 *
 * Any null string entries return corresponding null output column entries.
 *
 * @throw cudf::logic_error if start position is greater than stop position.
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
