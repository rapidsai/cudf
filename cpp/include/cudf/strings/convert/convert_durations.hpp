/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

namespace cudf {
namespace strings {
/**
 * @addtogroup strings_convert
 * @{
 * @file
 */

/**
 * @brief Returns a new duration column converting a strings column into
 * durations using the provided format pattern.
 *
 * The format pattern can include the following specifiers:
 * "%%,%n,%t,%D,%H,%I,%M,%S,%p,%R,%T,%r,%OH,%OI,%OM,%OS"
 *
 * | Specifier | Description | Range |
 * | :-------: | ----------- | ---------------- |
 * | %% | A literal % character | % |
 * | \%n | A newline character | \\n |
 * | \%t | A horizontal tab character | \\t |
 * | \%D | Days | -2,147,483,648 to 2,147,483,647 |
 * | \%H | 24-hour of the day | 00 to 23 |
 * | \%I | 12-hour of the day | 00 to 11 |
 * | \%M | Minute of the hour | 00 to 59 |
 * | \%S | Second of the minute | 00 to 59.999999999 |
 * | \%OH | same as %H but without sign | 00 to 23 |
 * | \%OI | same as %I but without sign | 00 to 11 |
 * | \%OM | same as %M but without sign | 00 to 59 |
 * | \%OS | same as %S but without sign | 00 to 59 |
 * | \%p | AM/PM designations associated with a 12-hour clock | 'AM' or 'PM' |
 * | \%R | Equivalent to "%H:%M" |  |
 * | \%T | Equivalent to "%H:%M:%S" |  |
 * | \%r | Equivalent to "%OI:%OM:%OS %p" |  |
 *
 * Other specifiers are not currently supported.
 *
 * Invalid formats are not checked. If the string contains unexpected
 * or insufficient characters, that output row entry's duration value is undefined.
 *
 * Any null string entry will result in a corresponding null row in the output column.
 *
 * The resulting time units are specified by the `duration_type` parameter.
 *
 * @throw cudf::logic_error if duration_type is not a duration type.
 *
 * @param strings Strings instance for this operation.
 * @param duration_type The duration type used for creating the output column.
 * @param format String specifying the duration format in strings.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New duration column.
 */
std::unique_ptr<column> to_durations(
  strings_column_view const& strings,
  data_type duration_type,
  std::string const& format,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Returns a new strings column converting a duration column into
 * strings using the provided format pattern.
 *
 * The format pattern can include the following specifiers:
 * "%%,%n,%t,%D,%H,%I,%M,%S,%p,%R,%T,%r,%OH,%OI,%OM,%OS"
 *
 * | Specifier | Description | Range |
 * | :-------: | ----------- | ---------------- |
 * | %% | A literal % character | % |
 * | \%n | A newline character | \\n |
 * | \%t | A horizontal tab character | \\t |
 * | \%D | Days | -2,147,483,648 to 2,147,483,647 |
 * | \%H | 24-hour of the day | 00 to 23 |
 * | \%I | 12-hour of the day | 00 to 11 |
 * | \%M | Minute of the hour | 00 to 59 |
 * | \%S | Second of the minute | 00 to 59.999999999 |
 * | \%OH | same as %H but without sign | 00 to 23 |
 * | \%OI | same as %I but without sign | 00 to 11 |
 * | \%OM | same as %M but without sign | 00 to 59 |
 * | \%OS | same as %S but without sign | 00 to 59 |
 * | \%p | AM/PM designations associated with a 12-hour clock | 'AM' or 'PM' |
 * | \%R | Equivalent to "%H:%M" |  |
 * | \%T | Equivalent to "%H:%M:%S" |  |
 * | \%r | Equivalent to "%OI:%OM:%OS %p" |  |
 *
 * No checking is done for invalid formats or invalid duration values. Formatting sticks to
 * specifications of `std::formatter<std::chrono::duration>` as much as possible.
 *
 * Any null input entry will result in a corresponding null entry in the output column.
 *
 * The time units of the input column influence the number of digits in decimal of seconds.
 * It uses 3 digits for milliseconds, 6 digits for microseconds and 9 digits for nanoseconds.
 * If duration value is negative, only one negative sign is written to output string. The specifiers
 * with signs are "%H,%I,%M,%S,%R,%T".
 *
 * @throw cudf::logic_error if `durations` column parameter is not a duration type.
 *
 * @param durations Duration values to convert.
 * @param format The string specifying output format.
 *        Default format is ""%d days %H:%M:%S".
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New strings column with formatted durations.
 */
std::unique_ptr<column> from_durations(
  column_view const& durations,
  std::string const& format           = "%D days %H:%M:%S",
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/** @} */  // end of doxygen group
}  // namespace strings
}  // namespace cudf
