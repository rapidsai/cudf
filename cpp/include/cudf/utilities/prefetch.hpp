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

#include <map>
#include <string>
#include <string_view>

namespace cudf::experimental::prefetch {

namespace detail {

/**
 * @brief A singleton class that manages the prefetching configuration.
 */
class PrefetchConfig {
 public:
  PrefetchConfig& operator=(const PrefetchConfig&) = delete;
  PrefetchConfig(const PrefetchConfig&)            = delete;

  /**
   * @brief Get the singleton instance of the prefetching configuration.
   *
   * @return The singleton instance of the prefetching configuration.
   */
  static PrefetchConfig& instance();

  /**
   * @brief Get the value of a configuration key.
   *
   * @param key The configuration key.
   * @return The value of the configuration key.
   */
  bool get(std::string_view key);
  /**
   * @brief Set the value of a configuration key.
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
  PrefetchConfig() = default;                 //< Private constructor to enforce singleton pattern
  std::map<std::string, bool> config_values;  //< Map of configuration keys to values
};

// TODO: This function is currently called in noexcept contexts, so it should
// not throw exceptions, but currently it can. We need to decide what the
// appropriate behavior is: either we make those contexts (e.g. the data
// accessors in column_view) not noexcept, or we make this function noexcept
// and have it silently allow failures to prefetch.
/**
 * @brief Enable prefetching for a particular structure or algorithm.
 *
 * @note At present this function does not support stream-ordered execution. Prefetching always
 * occurs on the default stream.
 *
 * @param key The key to enable prefetching for.
 * @param ptr The pointer to prefetch.
 * @param size The size of the memory region to prefetch.
 */
void prefetch(std::string_view key, void const* ptr, std::size_t size);

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

}  // namespace cudf::experimental::prefetch
