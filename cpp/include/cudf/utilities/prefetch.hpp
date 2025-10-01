/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

#include <cudf/utilities/default_stream.hpp>

#include <rmm/device_uvector.hpp>

#include <atomic>

namespace CUDF_EXPORT cudf {
namespace prefetch {

namespace detail {

std::atomic_bool& enabled();

std::atomic_bool& debug();

/**
 * @brief Enable prefetching for a particular structure or algorithm.
 *
 * @param ptr The pointer to prefetch.
 * @param size The size of the memory region to prefetch.
 * @param stream The stream to prefetch on.
 * @param device_id The device to prefetch on.
 */
void prefetch(void const* ptr,
              std::size_t size,
              rmm::cuda_stream_view stream,
              rmm::cuda_device_id device_id = rmm::get_current_cuda_device());

/**
 * @brief Enable prefetching for a particular structure or algorithm.
 *
 * @note This function will not throw exceptions, so it is safe to call in
 * noexcept contexts. If an error occurs, the error code is returned. This
 * function primarily exists for [mutable_]column_view::get_data and should be
 * removed once an method for stream-ordered data pointer access is added to
 * those data structures.
 *
 * @param ptr The pointer to prefetch.
 * @param size The size of the memory region to prefetch.
 * @param stream The stream to prefetch on.
 * @param device_id The device to prefetch on.
 */
cudaError_t prefetch_noexcept(
  void const* ptr,
  std::size_t size,
  rmm::cuda_stream_view stream,
  rmm::cuda_device_id device_id = rmm::get_current_cuda_device()) noexcept;

/**
 * @brief Prefetch the data in a device_uvector.
 *
 * @note At present this function does not support stream-ordered execution. Prefetching always
 * occurs on the default stream.
 *
 * @param v The device_uvector to prefetch.
 * @param stream The stream to prefetch on.
 * @param device_id The device to prefetch on.
 */
template <typename T>
void prefetch(rmm::device_uvector<T> const& v,
              rmm::cuda_stream_view stream,
              rmm::cuda_device_id device_id = rmm::get_current_cuda_device())
{
  if (v.is_empty()) { return; }
  prefetch(v.data(), v.size(), stream, device_id);
}

}  // namespace detail

/**
 * @brief Enable prefetching.
 *
 * Prefetching of managed memory in cudf currently always synchronizes on the
 * default stream and is not compatible with multi-stream applications.
 */
void enable() noexcept;

/**
 * @brief Disable prefetching.
 */
void disable() noexcept;

/**
 * @brief Enable debug mode for prefetching.
 *
 * In debug mode, the pointers being prefetched are printed to stderr.
 */
void enable_debugging() noexcept;

/**
 * @brief Enable debug mode for prefetching.
 *
 * In debug mode, the pointers being prefetched are printed to stderr.
 */
void disable_debugging() noexcept;

}  // namespace prefetch
}  // namespace CUDF_EXPORT cudf
