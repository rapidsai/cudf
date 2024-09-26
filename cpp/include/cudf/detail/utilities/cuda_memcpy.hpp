/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <cudf/utilities/export.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace CUDF_EXPORT cudf {
namespace detail {

enum class host_memory_kind : uint8_t { PINNED, PAGEABLE };

/**
 * @brief Asynchronously copies data between the host and device.
 *
 * Implementation may use different strategies depending on the size and type of host data.
 *
 * @param dst Destination memory address
 * @param src Source memory address
 * @param size Number of bytes to copy
 * @param kind Type of host memory
 * @param stream CUDA stream used for the copy
 */
void cuda_memcpy_async(
  void* dst, void const* src, size_t size, host_memory_kind kind, rmm::cuda_stream_view stream);

/**
 * @brief Synchronously copies data between the host and device.
 *
 * Implementation may use different strategies depending on the size and type of host data.
 *
 * @param dst Destination memory address
 * @param src Source memory address
 * @param size Number of bytes to copy
 * @param kind Type of host memory
 * @param stream CUDA stream used for the copy
 */
void cuda_memcpy(
  void* dst, void const* src, size_t size, host_memory_kind kind, rmm::cuda_stream_view stream);

template <typename T>
void copy_to_device_async(device_span<T> dst, host_span<T const> src, rmm::cuda_stream_view stream)
{
  auto const is_pinned = src.is_device_accessible();
  cuda_memcpy_async(dst.data(),
                    src.data(),
                    src.size_bytes(),
                    is_pinned ? host_memory_kind::PINNED : host_memory_kind::PAGEABLE,
                    stream);
}

template <typename T>
void copy_to_device(device_span<T> dst, host_span<T const> src, rmm::cuda_stream_view stream)
{
  copy_to_device_async(dst, src, stream);
  stream.synchronize();
}

template <typename T>
void copy_from_device_async(host_span<T> dst,
                            device_span<T const> src,
                            rmm::cuda_stream_view stream)
{
  auto const is_pinned = dst.is_device_accessible();
  cuda_memcpy_async(dst.data(),
                    src.data(),
                    src.size_bytes(),
                    is_pinned ? host_memory_kind::PINNED : host_memory_kind::PAGEABLE,
                    stream);
}

template <typename T>
void copy_from_device(host_span<T> dst, device_span<T const> src, rmm::cuda_stream_view stream)
{
  copy_from_device_async(dst, src, stream);
  stream.synchronize();
}

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
