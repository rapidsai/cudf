/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

namespace cudf {
namespace strings {
namespace detail {

/**
 * @copydoc to_integers(strings_column_view const&,data_type,rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> to_integers(strings_column_view const& strings,
                                    data_type output_type,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr);

/**
 * @copydoc from_integers(strings_column_view const&,rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> from_integers(column_view const& integers,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr);

/**
 * @copydoc to_floats(strings_column_view const&,data_type,rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> to_floats(strings_column_view const& strings,
                                  data_type output_type,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr);

/**
 * @copydoc from_floats(strings_column_view const&,rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> from_floats(column_view const& floats,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr);

/**
 * @copydoc to_booleans(strings_column_view const&,string_scalar
 * const&,rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> to_booleans(strings_column_view const& strings,
                                    string_scalar const& true_string,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr);

/**
 * @copydoc from_booleans(strings_column_view const&,string_scalar const&,string_scalar
 * const&,rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> from_booleans(column_view const& booleans,
                                      string_scalar const& true_string,
                                      string_scalar const& false_string,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr);

/**
 * @copydoc to_timestamps(strings_column_view const&,data_type,std::string_view,
 * rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<cudf::column> to_timestamps(strings_column_view const& strings,
                                            data_type timestamp_type,
                                            std::string_view format,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr);

/**
 * @copydoc from_timestamps(strings_column_view const&,std::string_view,
 * strings_column_view const&,rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> from_timestamps(column_view const& timestamps,
                                        std::string_view format,
                                        strings_column_view const& names,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr);

/**
 * @copydoc to_durations(strings_column_view const&,data_type,std::string_view,
 * rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> to_durations(strings_column_view const& strings,
                                     data_type duration_type,
                                     std::string_view format,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr);

/**
 * @copydoc from_durations(strings_column_view const&,std::string_view.
 * rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> from_durations(column_view const& durations,
                                       std::string_view format,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr);

/**
 * @copydoc to_fixed_point(strings_column_view const&,data_type,rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> to_fixed_point(strings_column_view const& strings,
                                       data_type output_type,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr);

/**
 * @copydoc from_fixed_point(strings_column_view const&,rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> from_fixed_point(column_view const& integers,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr);

}  // namespace detail
}  // namespace strings
}  // namespace cudf
