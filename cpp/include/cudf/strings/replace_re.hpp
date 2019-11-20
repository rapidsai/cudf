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
#include <cudf/scalar/scalar.hpp>

namespace cudf
{
namespace strings
{

/**
 * @brief For each string, replaces any character sequence matching the given pattern
 * with the provided replacement string.
 *
 * Any null string entries return corresponding null output column entries.
 *
 * @param strings Strings instance for this operation.
 * @param pattern The regular expression pattern to search within each string.
 * @param repl The string used to replace the matched sequence in each string.
 *        Default is an empty string.
 * @param maxrepl The maximum number of times to replace the matched pattern within each string.
 * @param mr Resource for allocating device memory.
 * @return New strings column.
 */
std::unique_ptr<column> replace_re( strings_column_view const& strings,
                                    std::string const& pattern,
                                    string_scalar const& repl = string_scalar(""),
                                    size_type maxrepl = -1,
                                    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief For each string, replaces any character sequence matching the given patterns
 * with the corresponding string in the repls column.
 *
 * Any null string entries return corresponding null output column entries.
 *
 * @param strings Strings instance for this operation.
 * @param pattern The regular expression patterns to search within each string.
 * @param repls The strings used for replacement.
 * @param mr Resource for allocating device memory.
 * @return New strings column.
 */
std::unique_ptr<column> replace_re( strings_column_view const& strings,
                                    std::vector<std::string> const& patterns,
                                    strings_column_view const& repls,
                                    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief For each string, replaces any character sequence matching the given pattern
 * using the repl template for back-references.
 *
 * Any null string entries return corresponding null output column entries.
 *
 * @param strings Strings instance for this operation.
 * @param pattern The regular expression patterns to search within each string.
 * @param repl The replacement template for creating the output string.
 * @param mr Resource for allocating device memory.
 * @return New strings column.
 */
std::unique_ptr<column> replace_with_backrefs( strings_column_view const& strings,
                                               std::string const& pattern,
                                               std::string const& repl,
                                               rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

} // namespace strings
} // namespace cudf
