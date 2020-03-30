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
 * @brief Decodes each string using URL encoding.
 *
 * Converts mostly non-ascii characters and control characters into UTF-8 hex code-points
 * prefixed with '%'. For example, the space character must be converted to characters '%20' where the
 * '20' indicates the hex value for space in UTF-8. Likewise, multi-byte characters are converted to
 * multiple hex charactes. For example, the é character is converted to characters '%C3%A9' where 'C3A9'
 * is the UTF-8 bytes 0xC3A9 for this character.
 *
 * Any null entries will result in corresponding null entries in the output column.
 *
 * @param strings Strings instance for this operation.
 * @param mr Resource for allocating device memory.
 * @return New strings column.
 */
std::unique_ptr<column> url_encode( strings_column_view const& strings,
                                    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Encodes each string using URL encoding.
 *
 * Converts all character sequences starting with '%' into character code-points
 * interpreting the 2 following characters as hex values to create the code-point.
 * For example, the sequence '%20' is converted into byte (0x20) which is a single
 * space character. Another example converts '%C3%A9' into 2 sequential bytes
 * (0xc3 and 0xa9 respectively). Overall, 3 characters are converted into one char byte
 * whenever a '%' character is encountered in the string.
 *
 * Any null entries will result in corresponding null entries in the output column.
 *
 * @param strings Strings instance for this operation.
 * @param mr Resource for allocating device memory.
 * @return New strings column.
 */
std::unique_ptr<column> url_decode( strings_column_view const& strings,
                                    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

} // namespace strings
} // namespace cudf
