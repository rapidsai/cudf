/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <rmm/mr/device/per_device_resource.hpp>

namespace cudf {
namespace strings {
/**
 * @addtogroup strings_slice
 * @{
 * @file
 */

/**
 * @brief Returns a new strings column that contains substrings of the
 * strings in the provided column.
 *
 * The character positions to retrieve in each string are `[start,stop)`.
 * If the start position is outside a string's length, an empty
 * string is returned for that entry. If the stop position is past the
 * end of a string's length, the end of the string is used for
 * stop position for that string.
 *
 * Null string entries will return null output string entries.
 *
 * @code{.pseudo}
 * Example:
 * s = ["hello", "goodbye"]
 * r = slice_strings(s,2,6)
 * r is now ["llo","odby"]
 * r2 = slice_strings(s,2,5,2)
 * r2 is now ["lo","ob"]
 * @endcode
 *
 * @param strings Strings column for this operation.
 * @param start First character position to begin the substring.
 * @param stop Last character position (exclusive) to end the substring.
 * @param step Distance between input characters retrieved.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New strings column with sorted elements of this instance.
 */
std::unique_ptr<column> slice_strings(
  strings_column_view const& strings,
  numeric_scalar<size_type> const& start = numeric_scalar<size_type>(0, false),
  numeric_scalar<size_type> const& stop  = numeric_scalar<size_type>(0, false),
  numeric_scalar<size_type> const& step  = numeric_scalar<size_type>(1),
  rmm::mr::device_memory_resource* mr    = rmm::mr::get_current_device_resource());

/**
 * @brief Returns a new strings column that contains substrings of the
 * strings in the provided column using unique ranges for each string.
 *
 * The character positions to retrieve in each string are specified in
 * the `starts` and `stops` integer columns.
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
 * @code{.pseudo}
 * Example:
 * s = ["hello", "goodbye"]
 * starts = [ 1, 2 ]
 * stops = [ 5, 4 ]
 * r = slice_strings(s,starts,stops)
 * r is now ["ello","od"]
 * @endcode
 *
 * @throw cudf::logic_error if starts or stops is a different size than the strings column.
 * @throw cudf::logic_error if starts and stops are not same integer type.
 * @throw cudf::logic_error if starts or stops contains nulls.
 *
 * @param strings Strings column for this operation.
 * @param starts First character positions to begin the substring.
 * @param stops Last character (exclusive) positions to end the substring.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New strings column with sorted elements of this instance.
 */
std::unique_ptr<column> slice_strings(
  strings_column_view const& strings,
  column_view const& starts,
  column_view const& stops,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/** @} */  // end of doxygen group
}  // namespace strings
}  // namespace cudf
