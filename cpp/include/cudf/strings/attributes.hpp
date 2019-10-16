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

/**---------------------------------------------------------------------------*
 * @brief Returns a numeric column containing the length of each string in
 * characters. The column will have the same number of entries/rows as the
 * specified strings column. Each value is the number of characters in the
 * corresponding string entry in the strings column.
 *
 * A null entry will report a character length of 0.
 *
 * @param strings Strings instance for this operation.
 * @param mr Resource for allocating device memory.
 * @return New column with lengths for each string in strings.
 *---------------------------------------------------------------------------**/
std::unique_ptr<cudf::column> characters_counts( strings_column_view strings,
                                                 rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**---------------------------------------------------------------------------*
 * @brief Returns a numeric column containing the length of each string in
 * bytes. The column will have the same number of entries/rows as the
 * specified strings column. Each value is the number of bytes in the
 * corresponding string entry in the strings column.
 *
 * A null entry will report a length of 0.
 *
 * @param strings Strings instance for this operation.
 * @param mr Resource for allocating device memory.
 * @return New column with bytes for each string in strings.
 *---------------------------------------------------------------------------**/
std::unique_ptr<cudf::column> bytes_counts( strings_column_view strings,
                                            rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**---------------------------------------------------------------------------*
 * @brief Creates a numeric column with code point values (integers) for each
 * character of each string.
 * A code point is the integer value representation of a character.
 * For example, the code point value for the character 'A' in UTF-8 is 65.
 *
 * The size of the output column will be the total number of characters in the
 * strings column.
 *
 * @param strings Strings instance for this operation.
 * @param mr Resource for allocating device memory.
 * @return New column with code point integer values for each character.
 */
std::unique_ptr<cudf::column> code_points( strings_column_view strings,
                                           rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());


} // namespace strings
} // namespace cudf
