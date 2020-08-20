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
 */

/**
 * @brief  Extracts year from any date time type and returns an int16_t
 * cudf::column.
 *
 * @param[in] cudf::column_view of the input datetime values
 *
 * @returns cudf::column of the extracted int16_t years
 * @throw cudf::logic_error if input column datatype is not TIMESTAMP
 */
std::unique_ptr<cudf::column> extract_year(
  cudf::column_view const& column,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief  Extracts month from any date time type and returns an int16_t
 * cudf::column.
 *
 * @param[in] cudf::column_view of the input datetime values
 *
 * @returns cudf::column of the extracted int16_t months
 * @throw cudf::logic_error if input column datatype is not TIMESTAMP
 */
std::unique_ptr<cudf::column> extract_month(
  cudf::column_view const& column,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief  Extracts day from any date time type and returns an int16_t
 * cudf::column.
 *
 * @param[in] cudf::column_view of the input datetime values
 *
 * @returns cudf::column of the extracted int16_t days
 * @throw cudf::logic_error if input column datatype is not TIMESTAMP
 */
std::unique_ptr<cudf::column> extract_day(
  cudf::column_view const& column,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief  Extracts day from any date time type and returns an int16_t
 * cudf::column.
 *
 * @param[in] cudf::column_view of the input datetime values
 *
 * @returns cudf::column of the extracted int16_t days
 * @throw cudf::logic_error if input column datatype is not TIMESTAMP
 */
std::unique_ptr<cudf::column> extract_weekday(
  cudf::column_view const& column,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief  Extracts hour from any date time type and returns an int16_t
 * cudf::column.
 *
 * @param[in] cudf::column_view of the input datetime values
 *
 * @returns cudf::column of the extracted int16_t hours
 * @throw cudf::logic_error if input column datatype is not TIMESTAMP
 */
std::unique_ptr<cudf::column> extract_hour(
  cudf::column_view const& column,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief  Extracts minute from any date time type and returns an int16_t
 * cudf::column.
 *
 * @param[in] cudf::column_view of the input datetime values
 *
 * @returns cudf::column of the extracted int16_t minutes
 * @throw cudf::logic_error if input column datatype is not TIMESTAMP
 */
std::unique_ptr<cudf::column> extract_minute(
  cudf::column_view const& column,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief  Extracts second from any date time type and returns an int16_t
 * cudf::column.
 *
 * @param[in] cudf::column_view of the input datetime values
 *
 * @returns cudf::column of the extracted int16_t seconds
 * @throw cudf::logic_error if input column datatype is not TIMESTAMP
 */
std::unique_ptr<cudf::column> extract_second(
  cudf::column_view const& column,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/** @} */  // end of group
/**
 * @addtogroup datetime_compute
 * @{
 */

/**
 * @brief  Computes the last day of the month in date time type and returns a TIMESTAMP_DAYS
 * cudf::column.
 *
 * @param[in] cudf::column_view of the input datetime values
 *
 * @returns cudf::column containing last day of the month as TIMESTAMP_DAYS
 * @throw cudf::logic_error if input column datatype is not TIMESTAMP
 */
std::unique_ptr<cudf::column> last_day_of_month(
  cudf::column_view const& column,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief  Computes the day number since the start of the year from the datetime and
 * returns an int16_t cudf::column. The value is between [1, {365-366}]
 *
 * @param[in] cudf::column_view of the input datetime values
 *
 * @returns cudf::column of datatype INT16 containing the day number since the start of the year.
 * @throw cudf::logic_error if input column datatype is not a TIMESTAMP
 */
std::unique_ptr<cudf::column> day_of_year(
  cudf::column_view const& column,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

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

 * @param[in] timestamps cudf::column_view of timestamp type.
 * @param[in] months cudf::column_view of integer type containing the number of months to add.
 *
 * @returns cudf::column of timestamp type containing the computed timestamps.
 * @throw cudf::logic_error if `timestamps` datatype is not a TIMESTAMP or if `months` datatype
 * is not INT16.
 * @throw cudf::logic_error if `timestamps` column size is not equal to `months` column size.
 */
std::unique_ptr<cudf::column> add_calendrical_months(
  cudf::column_view const& timestamps,
  cudf::column_view const& months,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());
/** @} */  // end of group
}  // namespace datetime
}  // namespace cudf
