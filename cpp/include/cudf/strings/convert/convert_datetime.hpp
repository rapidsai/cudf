/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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
 * @brief Returns a new timestamp column converting a strings column into
 * timestamps using the provided format pattern.
 *
 * The format pattern can include the following specifiers: "%Y,%y,%m,%d,%H,%I,%p,%M,%S,%f,%z"
 *
 * | Specifier | Description |
 * | :-------: | ----------- |
 * | %%d | Day of the month: 01-31 |
 * | %%m | Month of the year: 01-12 |
 * | %%y | Year without century: 00-99 |
 * | %%Y | Year with century: 0001-9999 |
 * | %%H | 24-hour of the day: 00-23 |
 * | %%I | 12-hour of the day: 01-12 |
 * | %%M | Minute of the hour: 00-59|
 * | %%S | Second of the minute: 00-59 |
 * | %%f | 6-digit microsecond: 000000-999999 |
 * | %%z | UTC offset with format ±HHMM Example +0500 |
 * | %%j | Day of the year: 001-366 |
 * | %%p | Only 'AM', 'PM' or 'am', 'pm' are recognized |
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
std::unique_ptr<column> to_timestamps(
  strings_column_view const& strings,
  data_type timestamp_type,
  std::string const& format,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Returns a new strings column converting a timestamp column into
 * strings using the provided format pattern.
 *
 * The format pattern can include the following specifiers: "%Y,%y,%m,%d,%H,%I,%p,%M,%S,%f,%z,%Z"
 *
 * | Specifier | Description |
 * | :-------: | ----------- |
 * | %%d | Day of the month: 01-31 |
 * | %%m | Month of the year: 01-12 |
 * | %%y | Year without century: 00-99 |
 * | %%Y | Year with century: 0001-9999 |
 * | %%H | 24-hour of the day: 00-23 |
 * | %%I | 12-hour of the day: 01-12 |
 * | %%M | Minute of the hour: 00-59|
 * | %%S | Second of the minute: 00-59 |
 * | %%f | 6-digit microsecond: 000000-999999 |
 * | %%z | Always outputs "+0000" |
 * | %%Z | Always outputs "UTC" |
 * | %%j | Day of the year: 001-366 |
 * | %%p | Only 'AM' or 'PM' |
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
 *        Default format is "%Y-%m-%dT%H:%M:%SZ".
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New strings column with formatted timestamps.
 */
std::unique_ptr<column> from_timestamps(
  column_view const& timestamps,
  std::string const& format           = "%Y-%m-%dT%H:%M:%SZ",
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/** @} */  // end of doxygen group
}  // namespace strings
}  // namespace cudf
