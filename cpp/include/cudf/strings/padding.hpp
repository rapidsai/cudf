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
#include <cudf/scalar/scalar.hpp>

namespace cudf
{
namespace strings
{

/**
 * @brief Pad types for the pad method specify where the pad
 * character should be placed.
 */
enum class pad_side {
    left,   ///< Add padding to the left.
    right,  ///< Add padding to the right.
    both    ///< Add padding equally to the right and left.
};

/**
 * @brief Add padding to each string using a provided character.
 *
 * If the string is already width or more characters, no padding is performed.
 * No strings are truncated.
 *
 * Null string entries result in null entries in the output column.
 *
 * ```
 * s = ['aa','bbb','cccc','ddddd']
 * r = pad(s,4)
 * r is now ['aa  ','bbb ','cccc','ddddd']
 * ```
 *
 * @param strings Strings instance for this operation.
 * @param width The minimum number of characters for each string.
 * @param side Where to place the padding characters.
 *        Default is pad right (left justify).
 * @param fill_char Single UTF-8 character to use for padding.
 *        Default is the space character.
 * @return New column with padded strings.
 */
std::unique_ptr<column> pad( strings_column_view const& strings,
                             size_type width, pad_side side = pad_side::right,
                             std::string const& fill_char = " ",
                             rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource() );

/**
 * @brief Add '0' as padding to the left of each string.
 *
 * If the string is already width or more characters, no padding is performed.
 * No strings are truncated.
 *
 * This will skip any leading '+' or '-' character encountered in each string.
 *
 * Null string entries result in null entries in the output column.
 *
 * ```
 * s = ['1234','-9876','+0.34','-342567']
 * r = zfill(s,6)
 * r is now ['001234','-09876','+00.34','-342567']
 * ```
 *
 * @param strings Strings instance for this operation.
 * @param width The minimum number of characters for each string.
 * @return New column of strings.
 */
std::unique_ptr<column> zfill( strings_column_view const& strings,
                               size_type width,
                               rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource() );

} // namespace strings
} // namespace cudf
