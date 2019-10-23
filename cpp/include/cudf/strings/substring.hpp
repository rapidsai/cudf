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
 * @brief Returns a new strings column that contains substrings of the
 * strings in the provided column.
 *
 * The character positions to retrieve in each string are [start,stop).
 * If the start position is outside a string's length, an empty
 * string is returned for that entry. If the stop position is past the
 * end of a string's length, the end of the string is used for
 * stop position for that string.
 *
 * Null string entries will return null output string entries.
 *
 * ```
 * s = ["hello", "goodbye"]
 * r = substring(s,2,6)
 * r is now ["llo","odby"]
 * r2 = substring(s,2,5,2)
 * r2 is now ["lo","ob"]
 * ```
 *
 * @param strings Strings instance for this operation.
 * @param start First character position to begin the substring.
 * @param stop Last character position (exclusive) to end the substring.
 *             Default of -1 indicates to the end of each string.
 * @param step Character count to skip when retrieving substring.
 * @param mr Resource for allocating device memory.
 * @return New strings column with sorted elements of this instance.
 */
std::unique_ptr<cudf::column> substring( strings_column_view strings,
                                         int32_t start, int32_t stop=-1, int32_t step=1,
                                         rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource() );


/**
 * @brief Returns a new strings column that contains substrings of the
 * strings in the provided column using unique ranges for each string.
 *
 * The character positions to retrieve in each string are [start,stop).
 * If a start position is outside a string's length, an empty
 * string is returned for that entry. If a stop position is past the
 * end of a string's length, the end of the string is used for
 * stop position for that string. Any stop position value set to -1 will
 * indicate to use the end of the string as the stop position for that
 * string.
 *
 * Null string entries will return null output string entries.
 *
 * ```
 * s = ["hello", "goodbye"]
 * starts = [ 1, 2 ]
 * stops = [ 5, 4 ]
 * r = substring_from(s,starts,stops)
 * r is now ["ello","od"]
 * ```
 *
 * @param strings Strings instance for this operation.
 * @param starts First character positions to begin the substring.
 * @param stops Last character (exclusive) positions to end the substring.
 * @param mr Resource for allocating device memory.
 * @return New strings column with sorted elements of this instance.
 */
std::unique_ptr<cudf::column> substring_from( strings_column_view strings,
                                              column_view starts, column_view stops,
                                              rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource() );


} // namespace strings
} // namespace cudf
