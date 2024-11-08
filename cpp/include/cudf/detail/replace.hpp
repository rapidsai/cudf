/*
 * Copyright (c) 2018-2024, NVIDIA CORPORATION.
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

#include <cudf/replace.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>

namespace CUDF_EXPORT cudf {
namespace detail {
/**
 * @copydoc cudf::replace_nulls(column_view const&, column_view const&,
 * rmm::device_async_resource_ref)
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> replace_nulls(column_view const& input,
                                      cudf::column_view const& replacement,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::replace_nulls(column_view const&, scalar const&,
 * rmm::device_async_resource_ref)
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> replace_nulls(column_view const& input,
                                      scalar const& replacement,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::replace_nulls(column_view const&, replace_policy const&,
 * rmm::device_async_resource_ref)
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> replace_nulls(column_view const& input,
                                      replace_policy const& replace_policy,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::replace_nans(column_view const&, column_view const&,
 * rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> replace_nans(column_view const& input,
                                     column_view const& replacement,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::replace_nans(column_view const&, scalar const&,
 * rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> replace_nans(column_view const& input,
                                     scalar const& replacement,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::find_and_replace_all
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> find_and_replace_all(column_view const& input_col,
                                             column_view const& values_to_replace,
                                             column_view const& replacement_values,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::normalize_nans_and_zeros
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> normalize_nans_and_zeros(column_view const& input,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::device_async_resource_ref mr);

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
