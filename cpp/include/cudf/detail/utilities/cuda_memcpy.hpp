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

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
