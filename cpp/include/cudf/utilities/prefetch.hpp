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

#include <cudf/utilities/default_stream.hpp>

#include <rmm/device_uvector.hpp>

#include <map>
#include <mutex>
#include <string>
#include <string_view>

namespace CUDF_EXPORT cudf {
namespace experimental::prefetch {

namespace detail {

/**
 * @brief A singleton class that manages the prefetching configuration.
 */
class prefetch_config {
 public:
  prefetch_config& operator=(const prefetch_config&) = delete;
  prefetch_config(const prefetch_config&)            = delete;

  /**
   * @brief Get the singleton instance of the prefetching configuration.
   *
   * @return The singleton instance of the prefetching configuration.
   */
  static prefetch_config& instance();

  /**
   * @brief Get the value of a configuration key.
   *
   * If the key does not exist, a `false` value will be returned.
   *
   * @param key The configuration key.
   * @return The value of the configuration key.
   */
  bool get(std::string_view key) const;
  /**
   * @brief Set the value of a configuration key.
   *
   * This is a thread-safe operation.
   *
   * @param key The configuration key.
   * @param value The value to set.
   */
  void set(std::string_view key, bool value);
  /**
   * @brief Enable or disable debug mode.
   *
   * In debug mode, the pointers being prefetched are printed to stderr.
   */
  bool debug{false};

 private:
  prefetch_config() = default;                //< Private constructor to enforce singleton pattern
  std::map<std::string, bool> config_values;  //< Map of configuration keys to values
  std::mutex config_mtx;                      //< Mutex for thread-safe config access
};

/**
 * @brief Enable prefetching for a particular structure or algorithm.
 *
 * @param key The key to enable prefetching for.
 * @param ptr The pointer to prefetch.
 * @param size The size of the memory region to prefetch.
 * @param stream The stream to prefetch on.
 * @param device_id The device to prefetch on.
 */
void prefetch(std::string_view key,
              void const* ptr,
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
 * @param key The key to enable prefetching for.
 * @param ptr The pointer to prefetch.
 * @param size The size of the memory region to prefetch.
 * @param stream The stream to prefetch on.
 * @param device_id The device to prefetch on.
 */
cudaError_t prefetch_noexcept(
  std::string_view key,
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
 * @param key The key to enable prefetching for.
 * @param v The device_uvector to prefetch.
 * @param stream The stream to prefetch on.
 * @param device_id The device to prefetch on.
 */
template <typename T>
void prefetch(std::string_view key,
              rmm::device_uvector<T> const& v,
              rmm::cuda_stream_view stream,
              rmm::cuda_device_id device_id = rmm::get_current_cuda_device())
{
  if (v.is_empty()) { return; }
  prefetch(key, v.data(), v.size(), stream, device_id);
}

}  // namespace detail

/**
 * @brief Enable prefetching for a particular structure or algorithm.
 *
 * @param key The key to enable prefetching for.
 */
void enable_prefetching(std::string_view key);

/**
 * @brief Disable prefetching for a particular structure or algorithm.
 *
 * @param key The key to disable prefetching for.
 */
void disable_prefetching(std::string_view key);

/**
 * @brief Enable or disable debug mode.
 *
 * In debug mode, the pointers being prefetched are printed to stderr.
 *
 * @param enable Whether to enable or disable debug mode.
 */
void prefetch_debugging(bool enable);

}  // namespace experimental::prefetch
}  // namespace CUDF_EXPORT cudf
