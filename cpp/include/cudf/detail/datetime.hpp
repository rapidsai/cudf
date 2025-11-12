/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
