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

#include <cudf/interop.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/export.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

namespace CUDF_EXPORT cudf {
namespace detail {

/**
 * @copydoc cudf::from_dlpack
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<table> from_dlpack(DLManagedTensor const* managed_tensor,
                                   rmm::cuda_stream_view stream,
                                   rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::to_dlpack
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
DLManagedTensor* to_dlpack(table_view const& input,
                           rmm::cuda_stream_view stream,
                           rmm::device_async_resource_ref mr);

/**
 * @brief Return a maximum precision for a given type.
 *
 * @tparam T the type to get the maximum precision for
 */
template <typename T>
constexpr std::size_t max_precision()
{
  auto constexpr num_bits = sizeof(T) * 8;
  return std::floor(num_bits * std::log(2) / std::log(10));
}

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
