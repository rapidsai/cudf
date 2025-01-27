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

#include <cudf/round.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace CUDF_EXPORT cudf {
//! Inner interfaces and implementations
namespace detail {

/**
 * @copydoc cudf::round(column_view const&, int32_t, rounding_method,
 * rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> round(column_view const& input,
                              int32_t decimal_places,
                              rounding_method method,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr);

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
