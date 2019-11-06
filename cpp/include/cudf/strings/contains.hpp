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
 * @brief Returns a boolean column identifying strings entries which
 * contain a character sequence matching the given regex pattern.
 *
 * Any null string results in a null entry for that row in the output column.
 *
 * @param strings Strings instance for this operation.
 * @param pattern Regex pattern to match to each string.
 * @param mr Resource for allocating device memory.
 * @return New column of boolean results for each string.
 */
std::unique_ptr<column> contains_re( strings_column_view const& strings,
                                     std::string const& pattern,
                                     rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Returns a boolean column identifying strings entries which
 * contain a character sequence matching the given regex pattern
 * only at the beginning of each string.
 *
 * Any null string results in a null entry for that row in the output column.
 *
 * @param strings Strings instance for this operation.
 * @param pattern Regex pattern to match to each string.
 * @param mr Resource for allocating device memory.
 * @return New column of boolean results for each string.
 */
std::unique_ptr<column> matches_re( strings_column_view const& strings,
                                    std::string const& pattern,
                                    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Returns the number of times the given regex pattern
 * matches in each string.
 *
 * Any null string results in a null entry for that row in the output column.
 *
 * @param strings Strings instance for this operation.
 * @param pattern Regex pattern to match within each string.
 * @param mr Resource for allocating device memory.
 * @return New INT32 column with counts for each string.
 */
std::unique_ptr<column> count_re( strings_column_view const& strings,
                                  std::string const& pattern,
                                  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

} // namespace strings
} // namespace cudf
