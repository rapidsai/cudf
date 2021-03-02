/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
/**
 * @copydoc cudf::extract_year(cudf::column_view const&, rmm::mr::device_memory_resource *)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<cudf::column> extract_year(
  cudf::column_view const& column,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @copydoc cudf::extract_month(cudf::column_view const&, rmm::mr::device_memory_resource *)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<cudf::column> extract_month(
  cudf::column_view const& column,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @copydoc cudf::extract_day(cudf::column_view const&, rmm::mr::device_memory_resource *)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<cudf::column> extract_day(
  cudf::column_view const& column,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @copydoc cudf::extract_weekday(cudf::column_view const&, rmm::mr::device_memory_resource *)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<cudf::column> extract_weekday(
  cudf::column_view const& column,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @copydoc cudf::extract_hour(cudf::column_view const&, rmm::mr::device_memory_resource *)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<cudf::column> extract_hour(
  cudf::column_view const& column,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @copydoc cudf::extract_minute(cudf::column_view const&, rmm::mr::device_memory_resource *)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<cudf::column> extract_minute(
  cudf::column_view const& column,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @copydoc cudf::extract_second(cudf::column_view const&, rmm::mr::device_memory_resource *)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<cudf::column> extract_second(
  cudf::column_view const& column,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @copydoc cudf::last_day_of_month(cudf::column_view const&, rmm::mr::device_memory_resource *)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<cudf::column> last_day_of_month(
  cudf::column_view const& column,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @copydoc cudf::day_of_year(cudf::column_view const&, rmm::mr::device_memory_resource *)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<cudf::column> day_of_year(
  cudf::column_view const& column,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @copydoc cudf::add_calendrical_months(cudf::column_view const&, cudf::column_view const&,
 * rmm::mr::device_memory_resource *)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<cudf::column> add_calendrical_months(
  cudf::column_view const& timestamps,
  cudf::column_view const& months,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());
}  // namespace detail
}  // namespace datetime
}  // namespace cudf
