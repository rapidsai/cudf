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
 */

/**
 * @brief Returns a new duration column converting a strings column into
 * durations using the provided format pattern.
 *
 * The format pattern can include the following specifiers: "%d,%+,%H,%M,%S,%u,%f"
 *
 * | Specifier | Description |
 * | :-------: | ----------- |
 * | %%d | Days: -2,147,483,648-2,147,483,647 |
 * | %%+ | Optional '+' sign for hours in case of negative days: + |
 * | %%H | 24-hour of the day: 00-23 |
 * | %%M | Minute of the hour: 00-59|
 * | %%S | Second of the minute: 00-59 |
 * | %%u | 6-digit microsecond: 000000-999999 |
 * | %%f | 9-digit nanosecond: 000000000-999999999 |
 *
 * Other specifiers are not currently supported.
 *
 * Invalid formats are not checked. If the string contains unexpected
 * or insufficient characters, that output row entry's timestamp value is undefined.
 *
 * Any null string entry will result in a corresponding null row in the output column.
 *
 * The resulting time units are specified by the `timestamp_type` parameter.
 * The time units are independent of the number of digits parsed by the "%f" specifier.
 * The "%f" supports a precision value to read the numeric digits. Specify the
 * precision with a single integer value (1-9) as follows:
 * use "%3f" for milliseconds, "%6f" for microseconds and "%9f" for nanoseconds.
 *
 * @throw cudf::logic_error if timestamp_type is not a timestamp type.
 *
 * @param strings Strings instance for this operation.
 * @param timestamp_type The timestamp type used for creating the output column.
 * @param format String specifying the timestamp format in strings.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New datetime column.
 */
std::unique_ptr<column> to_durations(
  strings_column_view const& strings,
  data_type duration_type,
  std::string const& format,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Returns a new strings column converting a duration column into
 * strings using the provided format pattern.
 *
 * The format pattern can include the following specifiers: "%Y,%y,%m,%d,%H,%I,%p,%M,%S,%f,%z,%Z"
 *
 * | Specifier | Description |
 * | :-------: | ----------- |
 * | %%d | Days: -2,147,483,648-2,147,483,647 |
 * | %%+ | Optional '+' sign for hours in case of negative days: + |
 * | %%H | 24-hour of the day: 00-23 |
 * | %%M | Minute of the hour: 00-59|
 * | %%S | Second of the minute: 00-59 |
 * | %%u | 6-digit microsecond: 000000-999999 |
 * | %%f | 9-digit nanosecond: 000000000-999999999 |
 *
 * No checking is done for invalid formats or invalid timestamp values.
 * All timestamps values are formatted to UTC.
 *
 * Any null input entry will result in a corresponding null entry in the output column.
 *
 * The time units of the input column do not influence the number of digits written by
 * the "%f" specifier.
 * The "%f" supports a precision value to write out numeric digits for the subsecond value.
 * Specify the precision with a single integer value (1-9) between the "%" and the "f" as follows:
 * use "%3f" for milliseconds, "%6f" for microseconds and "%9f" for nanoseconds.
 * If the precision is higher than the units, then zeroes are padded to the right of
 * the subsecond value.
 * If the precision is lower than the units, the subsecond value may be truncated.
 *
 * @throw cudf::logic_error if `timestamps` column parameter is not a timestamp type.
 *
 * @param timestamps Timestamp values to convert.
 * @param format The string specifying output format.
 *        Default format is ""%d days %+%H:%M:%S".
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New strings column with formatted timestamps.
 */
std::unique_ptr<column> from_durations(
  column_view const& durations,
  std::string const& format           = "%d days %+%H:%M:%S",
  //"P%YY%MM%DDT%HH%MM%SS" is_iso_format() for skipping leading zeros.
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/** @} */  // end of doxygen group
}  // namespace strings
}  // namespace cudf
