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
 * @brief Converts IPv4 addresses into integers.
 *
 * The IPv4 format is 1-3 character digits [0-9] between 3 dots
 * (e.g. 123.45.67.890). Each section can have a value between [0-255].
 *
 * The four sets of digits are converted to integers and placed in 8-bit fields inside
 * the resulting integer.
 * ```
 *   i0.i1.i2.i3 -> (i0 << 24) | (i1 << 16) | (i2 << 8) | (i3)
 * ```
 *
 * No checking is done on the format. If a string is not in IPv4 format, the resulting
 * integer is undefined.
 *
 * The resulting 32-bit integer is placed in an int64_t to avoid setting the sign-bit
 * in a int32_t type. This could be changed if cudf supported a UINT32 type in the future.
 *
 * Any null entries will result in corresponding null entries in the output column.
 *
 * @param strings Strings instance for this operation.
 * @param mr Resource for allocating device memory.
 * @return New INT64 column converted from strings.
 */
std::unique_ptr<column> ipv4_to_integers( strings_column_view const& strings,
                                          rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Converts integers into IPv4 addresses as strings.
 *
 * The IPv4 format is 1-3 character digits [0-9] between 3 dots
 * (e.g. 123.45.67.890). Each section can have a value between [0-255].
 *
 * Each input integer is dissected into four integers by dividing the input into 8-bit sections.
 * These sub-integers are then converted into [0-9] characters and placed between '.' characters.
 *
 * No checking is done on the input integer value. Only the lower 32-bits are used.
 *
 * Any null entries will result in corresponding null entries in the output column.
 *
 * @throw cudf::logic_error if the input column is not INT64 type.
 *
 * @param column Integer (INT64) column to convert.
 * @param mr Resource for allocating device memory.
 * @return New strings column.
 */
std::unique_ptr<column> integers_to_ipv4( column_view const& integers,
                                          rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

} // namespace strings
} // namespace cudf
