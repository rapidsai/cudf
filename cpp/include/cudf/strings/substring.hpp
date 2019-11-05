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
 * @throw cudf::logic_error if start position is not a positive integer or zero.
 * @throw cudf::logic_error if start is greater than stop.
 * @throw cudf::logic_error if step value is not a positive integer.
 *
 * @param strings Strings column for this operation.
 * @param start First character position to begin the substring.
 * @param stop Last character position (exclusive) to end the substring.
 *             Default of -1 indicates to use the end of each string.
 * @param step Distance between input characters retrieved.
 * @param mr Resource for allocating device memory.
 * @return New strings column with sorted elements of this instance.
 */
std::unique_ptr<column> slice_strings( strings_column_view const& strings,
                                       size_type start, size_type stop=-1, size_type step=1,
                                       rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());


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
 * The starts and stops column must both be the same integer type and
 * must be the same size as the strings column.
 *
 * ```
 * s = ["hello", "goodbye"]
 * starts = [ 1, 2 ]
 * stops = [ 5, 4 ]
 * r = substring_from(s,starts,stops)
 * r is now ["ello","od"]
 * ```
 *
 * @throw cudf::logic_error if starts or stops is a different size than the strings column.
 * @throw cudf::logic_error if starts and stops are not same integer type.
 * @throw cudf::logic_error if starts or stops contains nulls.
 *
 * @param strings Strings column for this operation.
 * @param starts First character positions to begin the substring.
 * @param stops Last character (exclusive) positions to end the substring.
 * @param mr Resource for allocating device memory.
 * @return New strings column with sorted elements of this instance.
 */
std::unique_ptr<column> slice_strings( strings_column_view const& strings,
                                       column_view const& starts, column_view const& stops,
                                       rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

} // namespace strings
} // namespace cudf
