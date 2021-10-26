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

#include <cudf/types.hpp>

#include <memory>

/**
 * @file datetime.hpp
 * @brief DateTime column APIs.
 */

namespace cudf {
namespace datetime {
/**
 * @addtogroup datetime_extract
 * @{
 * @file
 */

/**
 * @brief  Extracts year from any date time type and returns an int16_t
 * cudf::column.
 *
 * @param column cudf::column_view of the input datetime values
 * @param mr Device memory resource used to allocate device memory of the returned column.
 *
 * @returns cudf::column of the extracted int16_t years
 * @throw cudf::logic_error if input column datatype is not TIMESTAMP
 */
std::unique_ptr<cudf::column> extract_year(
  cudf::column_view const& column,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief  Extracts month from any date time type and returns an int16_t
 * cudf::column.
 *
 * @param column cudf::column_view of the input datetime values
 * @param mr Device memory resource used to allocate device memory of the returned column.
 *
 * @returns cudf::column of the extracted int16_t months
 * @throw cudf::logic_error if input column datatype is not TIMESTAMP
 */
std::unique_ptr<cudf::column> extract_month(
  cudf::column_view const& column,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief  Extracts day from any date time type and returns an int16_t
 * cudf::column.
 *
 * @param column cudf::column_view of the input datetime values
 * @param mr Device memory resource used to allocate device memory of the returned column.
 *
 * @returns cudf::column of the extracted int16_t days
 * @throw cudf::logic_error if input column datatype is not TIMESTAMP
 */
std::unique_ptr<cudf::column> extract_day(
  cudf::column_view const& column,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief  Extracts day from any date time type and returns an int16_t
 * cudf::column.
 *
 * @param column cudf::column_view of the input datetime values
 * @param mr Device memory resource used to allocate device memory of the returned column.
 *
 * @returns cudf::column of the extracted int16_t days
 * @throw cudf::logic_error if input column datatype is not TIMESTAMP
 */
std::unique_ptr<cudf::column> extract_weekday(
  cudf::column_view const& column,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief  Extracts hour from any date time type and returns an int16_t
 * cudf::column.
 *
 * @param column cudf::column_view of the input datetime values
 * @param mr Device memory resource used to allocate device memory of the returned column.
 *
 * @returns cudf::column of the extracted int16_t hours
 * @throw cudf::logic_error if input column datatype is not TIMESTAMP
 */
std::unique_ptr<cudf::column> extract_hour(
  cudf::column_view const& column,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief  Extracts minute from any date time type and returns an int16_t
 * cudf::column.
 *
 * @param column cudf::column_view of the input datetime values
 * @param mr Device memory resource used to allocate device memory of the returned column.
 *
 * @returns cudf::column of the extracted int16_t minutes
 * @throw cudf::logic_error if input column datatype is not TIMESTAMP
 */
std::unique_ptr<cudf::column> extract_minute(
  cudf::column_view const& column,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief  Extracts second from any date time type and returns an int16_t
 * cudf::column.
 *
 * @param column cudf::column_view of the input datetime values
 * @param mr Device memory resource used to allocate device memory of the returned column.
 *
 * @returns cudf::column of the extracted int16_t seconds
 * @throw cudf::logic_error if input column datatype is not TIMESTAMP
 */
std::unique_ptr<cudf::column> extract_second(
  cudf::column_view const& column,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/** @} */  // end of group
/**
 * @addtogroup datetime_compute
 * @{
 * @file
 */

/**
 * @brief  Computes the last day of the month in date time type and returns a TIMESTAMP_DAYS
 * cudf::column.
 *
 * @param column cudf::column_view of the input datetime values
 * @param mr Device memory resource used to allocate device memory of the returned column.
 *
 * @returns cudf::column containing last day of the month as TIMESTAMP_DAYS
 * @throw cudf::logic_error if input column datatype is not TIMESTAMP
 */
std::unique_ptr<cudf::column> last_day_of_month(
  cudf::column_view const& column,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief  Computes the day number since the start of the year from the datetime and
 * returns an int16_t cudf::column. The value is between [1, {365-366}]
 *
 * @param column cudf::column_view of the input datetime values
 * @param mr Device memory resource used to allocate device memory of the returned column.
 *
 * @returns cudf::column of datatype INT16 containing the day number since the start of the year.
 * @throw cudf::logic_error if input column datatype is not a TIMESTAMP
 */
std::unique_ptr<cudf::column> day_of_year(
  cudf::column_view const& column,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief  Adds or subtracts a number of months from the date time type and returns a
 * timestamp column that is of the same type as the input `timestamps` column.
 *
 * For a given row, if the `timestamps` or the `months` column value is null,
 * the output for that row is null.
 * This method preserves the input time and the day where applicable. The date is rounded
 * down to the last day of the month for that year, if the new day is invalid for that month.
 *
 * @code{.pseudo}
 * Example:
 * timestamps = [5/31/20 08:00:00, 5/31/20 00:00:00, 5/31/20 13:00:00, 5/31/20 23:00:00,
 *               6/30/20 00:00:01, 6/30/20 14:12:13]
 * months     = [1               , -1              , -3              , -15             ,
 *               -1              , 1]
 * r = add_calendrical_months(timestamp_column, months_column)
 * r is [6/30/20 08:00:00, 4/30/20 00:00:00, 2/29/20 13:00:00, 2/28/19 23:00:00,
 *       5/30/20 00:00:01, 7/30/20 14:12:13]
 * @endcode

 * @throw cudf::logic_error if `timestamps` datatype is not a TIMESTAMP or if `months` datatype
 * is not INT16 or INT32.
 * @throw cudf::logic_error if `timestamps` column size is not equal to `months` column size.
 *
 * @param timestamps cudf::column_view of timestamp type.
 * @param months cudf::column_view of integer type containing the number of months to add.
 * @param mr Device memory resource used to allocate device memory of the returned column.
 *
 * @returns cudf::column of timestamp type containing the computed timestamps.
 */
std::unique_ptr<cudf::column> add_calendrical_months(
  cudf::column_view const& timestamps,
  cudf::column_view const& months,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief  Adds or subtracts a number of months from the date time type and returns a
 * timestamp column that is of the same type as the input `timestamps` column.
 *
 * For a given row, if the `timestamps` value is null, the output for that row is null.
 * A null months scalar would result in an all null column.
 * This method preserves the input time and the day where applicable. The date is rounded
 * down to the last day of the month for that year, if the new day is invalid for that month.
 *
 * @code{.pseudo}
 * Example:
 * timestamps = [5/31/20 08:00:00, 6/30/20 00:00:00, 7/31/20 13:00:00]
 * months     = -3
 * output is [2/29/20 08:00:00, 3/30/20 00:00:00, 4/30/20 13:00:00]
 *
 * timestamps = [4/28/20 04:00:00, 5/30/20 01:00:00, 6/30/20 21:00:00]
 * months     = 1
 * output is [5/28/20 04:00:00, 6/30/20 01:00:00, 7/30/20 21:00:00]
 * @endcode
 *
 * @throw cudf::logic_error if `timestamps` datatype is not a TIMESTAMP or if `months` datatype
 * is not INT16 or INT32.
 * @throw cudf::logic_error if `timestamps` column size is not equal to `months` column size.
 *
 * @param timestamps cudf::column_view of timestamp type.
 * @param months cudf::scalar of integer type containing the number of months to add.
 * @param mr Device memory resource used to allocate device memory of the returned column.
 *
 * @return cudf::column of timestamp type containing the computed timestamps.
 */
std::unique_ptr<cudf::column> add_calendrical_months(
  cudf::column_view const& timestamps,
  cudf::scalar const& months,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief  Check if the year of the given date is a leap year
 *
 * `output[i] == true` if year of `column[i]` is a leap year
 * `output[i] == false` if year of `column[i]` is not a leap year
 * `output[i] is null` if `column[i]` is null
 *
 * @param column cudf::column_view of the input datetime values
 * @param mr Device memory resource used to allocate device memory of the returned column.
 *
 * @returns cudf::column of datatype BOOL8 truth value of the corresponding date
 * @throw cudf::logic_error if input column datatype is not a TIMESTAMP
 */
std::unique_ptr<cudf::column> is_leap_year(
  cudf::column_view const& column,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Extract the number of days in the month
 *
 * output[i] contains the number of days in the month of date `column[i]`
 * output[i] is null if `column[i]` is null
 *
 * @throw cudf::logic_error if input column datatype is not a TIMESTAMP
 *
 * @param column cudf::column_view of the input datetime values
 * @param mr Device memory resource used to allocate device memory of the returned column.
 * @return cudf::column of datatype INT16 of days in month of the corresponding date
 */
std::unique_ptr<cudf::column> days_in_month(
  cudf::column_view const& column,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief  Returns the quarter of the date
 *
 * `output[i]` will be a value from {1, 2, 3, 4} corresponding to the quater of month given by
 * `column[i]`. It will be null if the input row at `column[i]` is null.
 *
 * @throw cudf::logic_error if input column datatype is not a TIMESTAMP
 *
 * @param column The input column containing datetime values
 * @param mr Device memory resource used to allocate device memory of the returned column.
 * @return A column of INT16 type indicating which quarter the date is in
 */
std::unique_ptr<cudf::column> extract_quarter(
  cudf::column_view const& column,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/** @} */  // end of group

/**
 * @brief Round up to the nearest day
 *
 * @param column cudf::column_view of the input datetime values
 * @param mr Device memory resource used to allocate device memory of the returned column.
 *
 * @throw cudf::logic_error if input column datatype is not TIMESTAMP
 * @return cudf::column of the same datetime resolution as the input column
 */
std::unique_ptr<cudf::column> ceil_day(
  cudf::column_view const& column,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Round up to the nearest hour
 *
 * @param column cudf::column_view of the input datetime values
 * @param mr Device memory resource used to allocate device memory of the returned column.
 *
 * @throw cudf::logic_error if input column datatype is not TIMESTAMP
 * @return cudf::column of the same datetime resolution as the input column
 */
std::unique_ptr<cudf::column> ceil_hour(
  cudf::column_view const& column,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Round up to the nearest minute
 *
 * @param column cudf::column_view of the input datetime values
 * @param mr Device memory resource used to allocate device memory of the returned column.
 *
 * @throw cudf::logic_error if input column datatype is not TIMESTAMP
 * @return cudf::column of the same datetime resolution as the input column
 */
std::unique_ptr<cudf::column> ceil_minute(
  cudf::column_view const& column,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Round up to the nearest second
 *
 * @param column cudf::column_view of the input datetime values
 * @param mr Device memory resource used to allocate device memory of the returned column.
 *
 * @throw cudf::logic_error if input column datatype is not TIMESTAMP
 * @return cudf::column of the same datetime resolution as the input column
 */
std::unique_ptr<cudf::column> ceil_second(
  cudf::column_view const& column,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Round up to the nearest millisecond
 *
 * @param column cudf::column_view of the input datetime values
 * @param mr Device memory resource used to allocate device memory of the returned column.
 *
 * @throw cudf::logic_error if input column datatype is not TIMESTAMP
 * @return cudf::column of the same datetime resolution as the input column
 */
std::unique_ptr<column> ceil_millisecond(
  column_view const& column,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Round up to the nearest microsecond
 *
 * @param column cudf::column_view of the input datetime values
 * @param mr Device memory resource used to allocate device memory of the returned column.
 *
 * @throw cudf::logic_error if input column datatype is not TIMESTAMP
 * @return cudf::column of the same datetime resolution as the input column
 */
std::unique_ptr<column> ceil_microsecond(
  column_view const& column,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Round up to the nearest nanosecond
 *
 * @param column cudf::column_view of the input datetime values
 * @param mr Device memory resource used to allocate device memory of the returned column.
 *
 * @throw cudf::logic_error if input column datatype is not TIMESTAMP
 * @return cudf::column of the same datetime resolution as the input column
 */
std::unique_ptr<column> ceil_nanosecond(
  column_view const& column,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

}  // namespace datetime
}  // namespace cudf
