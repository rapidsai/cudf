/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <cstdint>

namespace cudf::io::detail {

/**
 * @brief A strong type for padding value.
 */
enum class padding : uint32_t {};

/**
 * @brief Create an `rmm::device_buffer` that is padded by a given padding value.
 *
 * @tparam T Type of the size parameter
 * @param size Size of the output buffer, in bytes
 * @param pad The padding amount (in bytes) that will be added to the output buffer
 * @param stream CUDA stream to use for device memory operations
 * @param mr Device memory resource to use for device memory allocation
 * @return A buffer with size equal to the input size value plus padding
 */
template <typename T>
rmm::device_buffer make_padded_device_buffer(
  T size,
  padding pad,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  return rmm::device_buffer(
    static_cast<std::size_t>(size) + static_cast<std::size_t>(pad), stream, mr);
}

}  // namespace cudf::io::detail
