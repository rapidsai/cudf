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

enum class strip_type
{
    LEFT,   //<< strip characters from the beginning of the string
    RIGHT,  //<< strip characters from the end of the string
    BOTH    //<< strip characters from the beginning and end of the string
};

/**
 * @brief Removes the specified characters from the beginning or end
 * (or both) of each string.
 *
 * The to_strip parameter can contain one or more characters.
 * All characters in `to_strip` are removed from the input strings.
 *
 * If `to_strip` is the empty string, whitespace characters are removed.
 * Whitespace is considered the space character plus control characters
 * like tab and line feed.
 *
 * Any null string entries return corresponding null output column entries.
 *
 * ```
 * s = [" aaa ", "_bbbb ", "__cccc  ", "ddd", " ee _ff gg_"]
 * r = strip(s,both," _")
 * r is now ["aaa", "bbbb", "cccc", "ddd", "ee _ff gg"]
 * ```
 *
 * @throw cudf::logic_error if `to_strip` is invalid.
 *
 * @param strings Strings column for this operation.
 * @param stype Indicates characters are to be stripped from the beginning, end, or both of each string.
 *        Default is both.
 * @param to_strip UTF-8 encoded characters to strip from each string.
 *        Default is empty string which indicates strip whitespace characters.
 * @param mr Resource for allocating device memory.
 * @return New strings column.
 */
std::unique_ptr<column> strip( strings_column_view const& strings,
                               strip_type stype = strip_type::BOTH,
                               string_scalar const& to_strip = string_scalar(""),
                               rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());


} // namespace strings
} // namespace cudf
