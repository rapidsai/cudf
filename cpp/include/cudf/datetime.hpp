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

namespace cudf {
namespace datetime {
namespace detail {

enum class datetime_component {
  INVALID = 0,
  YEAR,
  MONTH,
  DAY,
  WEEKDAY,
  HOUR,
  MINUTE,
  SECOND,
};

/**
 * @brief  Extracts the supplied datetime component from any date time type
 * and returns an int16_t cudf::column.
 *
 * @param[in] cudf::column_view of the input datetime values
 * @returns cudf::column of the extracted int16_t datetime component
 * @throw cudf::logic_error if input column datatype is not TIMESTAMP
 */

template <datetime_component Component>
std::unique_ptr<column> extract_component(
    column_view const& column,
    cudaStream_t stream = 0,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());
}  // namespace detail

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

}  // namespace datetime
}  // namespace cudf
