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

#include <rmm/cuda_stream_view.hpp>

namespace cudf::detail {

namespace impl {

void copy_pinned_to_device(void* dst, void const* src, size_t size, rmm::cuda_stream_view stream);
void copy_device_to_pinned(void* dst, void const* src, size_t size, rmm::cuda_stream_view stream);

void copy_pageable_to_device(void* dst, void const* src, size_t size, rmm::cuda_stream_view stream);
void copy_device_to_pageable(void* dst, void const* src, size_t size, rmm::cuda_stream_view stream);

}  // namespace impl

enum class copy_kind { PINNED_TO_DEVICE, DEVICE_TO_PINNED, PAGEABLE_TO_DEVICE, DEVICE_TO_PAGEABLE };

/**
 * @brief Asynchronously copies data between the host and device.
 *
 * Implementation may use different strategies depending on the size and type of host data.
 *
 * @param dst Destination memory address
 * @param src Source memory address
 * @param size Number of bytes to copy
 * @param kind Direction of the copy and type of host memory
 * @param stream CUDA stream used for the copy
 */

template <typename T>
void cuda_memcpy_async(
  T* dst, T const* src, size_t size, copy_kind kind, rmm::cuda_stream_view stream)
{
  if (kind == copy_kind::PINNED_TO_DEVICE) {
    impl::copy_pinned_to_device(dst, src, size * sizeof(T), stream);
  } else if (kind == copy_kind::DEVICE_TO_PINNED) {
    impl::copy_device_to_pinned(dst, src, size * sizeof(T), stream);
  } else if (kind == copy_kind::PAGEABLE_TO_DEVICE) {
    impl::copy_pageable_to_device(dst, src, size * sizeof(T), stream);
  } else if (kind == copy_kind::DEVICE_TO_PAGEABLE) {
    impl::copy_device_to_pageable(dst, src, size * sizeof(T), stream);
  }
}

/**
 * @brief Synchronously copies data between the host and device.
 *
 * Implementation may use different strategies depending on the size and type of host data.
 *
 * @param dst Destination memory address
 * @param src Source memory address
 * @param size Number of bytes to copy
 * @param kind Direction of the copy and type of host memory
 * @param stream CUDA stream used for the copy
 */
template <typename T>
void cuda_memcpy(T* dst, T const* src, size_t size, copy_kind kind, rmm::cuda_stream_view stream)
{
  cuda_memcpy_async(dst, src, size, kind, stream);
  stream.synchronize();
}

}  // namespace cudf::detail
