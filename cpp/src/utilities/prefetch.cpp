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

#include <cudf/utilities/error.hpp>
#include <cudf/utilities/prefetch.hpp>

#include <rmm/cuda_device.hpp>

#include <iostream>

namespace cudf::experimental::prefetch {

namespace detail {

prefetch_config& prefetch_config::instance()
{
  static prefetch_config instance;
  return instance;
}

bool prefetch_config::get(std::string_view key)
{
  std::shared_lock<std::shared_mutex> lock(config_mtx);
  auto const it = config_values.find(key.data());
  return it == config_values.end() ? false : it->second;  // default to not prefetching
}

void prefetch_config::set(std::string_view key, bool value)
{
  std::lock_guard<std::shared_mutex> lock(config_mtx);
  config_values[key.data()] = value;
}

cudaError_t prefetch_noexcept(std::string_view key,
                              void const* ptr,
                              std::size_t size,
                              rmm::cuda_stream_view stream,
                              rmm::cuda_device_id device_id) noexcept
{
  // Don't try to prefetch nullptrs or empty data. Sometimes libcudf has column
  // views that use nullptrs with a nonzero size as an optimization.
  if (ptr == nullptr) {
    if (prefetch_config::instance().debug) {
      std::cerr << "Skipping prefetch of nullptr" << std::endl;
    }
    return cudaSuccess;
  }
  if (size == 0) {
    if (prefetch_config::instance().debug) {
      std::cerr << "Skipping prefetch of size 0" << std::endl;
    }
    return cudaSuccess;
  }
  if (prefetch_config::instance().get(key)) {
    if (prefetch_config::instance().debug) {
      std::cerr << "Prefetching " << size << " bytes for key " << key << " at location " << ptr
                << std::endl;
    }
    auto result = cudaMemPrefetchAsync(ptr, size, device_id.value(), stream.value());
    // Need to flush the CUDA error so that the context is not corrupted.
    if (result == cudaErrorInvalidValue) { cudaGetLastError(); }
    return result;
  }
  return cudaSuccess;
}

void prefetch(std::string_view key,
              void const* ptr,
              std::size_t size,
              rmm::cuda_stream_view stream,
              rmm::cuda_device_id device_id)
{
  auto result = prefetch_noexcept(key, ptr, size, stream, device_id);
  // Ignore cudaErrorInvalidValue because that will be raised if prefetching is
  // attempted on unmanaged memory.
  if ((result != cudaErrorInvalidValue) && (result != cudaSuccess)) {
    std::cerr << "Prefetch failed" << std::endl;
    CUDF_CUDA_TRY(result);
  }
}

}  // namespace detail

void enable_prefetching(std::string_view key)
{
  detail::prefetch_config::instance().set(key, true);
}

void disable_prefetching(std::string_view key)
{
  detail::prefetch_config::instance().set(key, false);
}

void prefetch_debugging(bool enable) { detail::prefetch_config::instance().debug = enable; }
}  // namespace cudf::experimental::prefetch
