/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <memory>

namespace CUDF_EXPORT cudf {
namespace datetime {
namespace detail {
/**
 * @copydoc cudf::extract_year(cudf::column_view const&, rmm::cuda_stream_view,
 * rmm::device_async_resource_ref)
 *
 */
std::unique_ptr<cudf::column> extract_year(cudf::column_view const& column,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::extract_month(cudf::column_view const&, rmm::cuda_stream_view,
 * rmm::device_async_resource_ref)
 *
 */
std::unique_ptr<cudf::column> extract_month(cudf::column_view const& column,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::extract_day(cudf::column_view const&, rmm::cuda_stream_view,
 * rmm::device_async_resource_ref)
 *
 */
std::unique_ptr<cudf::column> extract_day(cudf::column_view const& column,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::extract_weekday(cudf::column_view const&, rmm::cuda_stream_view,
 * rmm::device_async_resource_ref)
 *
 */
std::unique_ptr<cudf::column> extract_weekday(cudf::column_view const& column,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::extract_hour(cudf::column_view const&, rmm::cuda_stream_view,
 * rmm::device_async_resource_ref)
 *
 */
std::unique_ptr<cudf::column> extract_hour(cudf::column_view const& column,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::extract_minute(cudf::column_view const&, rmm::cuda_stream_view,
 * rmm::device_async_resource_ref)
 *
 */
std::unique_ptr<cudf::column> extract_minute(cudf::column_view const& column,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::extract_second(cudf::column_view const&, rmm::cuda_stream_view,
 * rmm::device_async_resource_ref)
 *
 */
std::unique_ptr<cudf::column> extract_second(cudf::column_view const& column,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::extract_millisecond_fraction(cudf::column_view const&, rmm::cuda_stream_view,
 * rmm::device_async_resource_ref)
 *
 */
std::unique_ptr<cudf::column> extract_millisecond_fraction(cudf::column_view const& column,
                                                           rmm::cuda_stream_view stream,
                                                           rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::extract_microsecond_fraction(cudf::column_view const&, rmm::cuda_stream_view,
 * rmm::device_async_resource_ref)
 *
 */
std::unique_ptr<cudf::column> extract_microsecond_fraction(cudf::column_view const& column,
                                                           rmm::cuda_stream_view stream,
                                                           rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::extract_nanosecond_fraction(cudf::column_view const&, rmm::cuda_stream_view,
 * rmm::device_async_resource_ref)
 *
 */
std::unique_ptr<cudf::column> extract_nanosecond_fraction(cudf::column_view const& column,
                                                          rmm::cuda_stream_view stream,
                                                          rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::extract_datetime_component(cudf::column_view const&, datetime_component,
 * rmm::cuda_stream_view, rmm::device_async_resource_ref)
 *
 */
std::unique_ptr<cudf::column> extract_datetime_component(cudf::column_view const& column,
                                                         datetime_component component,
                                                         rmm::cuda_stream_view stream,
                                                         rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::last_day_of_month(cudf::column_view const&, rmm::cuda_stream_view,
 * rmm::device_async_resource_ref)
 *
 */
std::unique_ptr<cudf::column> last_day_of_month(cudf::column_view const& column,
                                                rmm::cuda_stream_view stream,
                                                rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::day_of_year(cudf::column_view const&, rmm::cuda_stream_view,
 * rmm::device_async_resource_ref)
 *
 */
std::unique_ptr<cudf::column> day_of_year(cudf::column_view const& column,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::add_calendrical_months(cudf::column_view const&, cudf::column_view const&,
 * rmm::cuda_stream_view, rmm::device_async_resource_ref)
 *
 */
std::unique_ptr<cudf::column> add_calendrical_months(cudf::column_view const& timestamps,
                                                     cudf::column_view const& months,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::add_calendrical_months(cudf::column_view const&, cudf::scalar const&,
 * rmm::cuda_stream_view, rmm::device_async_resource_ref)
 *
 */
std::unique_ptr<cudf::column> add_calendrical_months(cudf::column_view const& timestamps,
                                                     cudf::scalar const& months,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::is_leap_year(cudf::column_view const&, rmm::cuda_stream_view,
 * rmm::device_async_resource_ref)
 *
 */
std::unique_ptr<cudf::column> is_leap_year(cudf::column_view const& column,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr);

std::unique_ptr<cudf::column> extract_quarter(cudf::column_view const& column,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr);

}  // namespace detail
}  // namespace datetime
}  // namespace CUDF_EXPORT cudf
