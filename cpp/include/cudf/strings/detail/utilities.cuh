/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <cudf/strings/string_view.cuh>
#include <cudf/utilities/error.hpp>

#include <mutex>
#include <unordered_map>

namespace cudf {
namespace strings {
namespace detail {

/**
 * @brief Copies input string data into a buffer and increments the pointer by the number of bytes
 * copied.
 *
 * @param buffer Device buffer to copy to.
 * @param input Data to copy from.
 * @param bytes Number of bytes to copy.
 * @return Pointer to the end of the output buffer after the copy.
 */
__device__ inline char* copy_and_increment(char* buffer, const char* input, size_type bytes)
{
  memcpy(buffer, input, bytes);
  return buffer + bytes;
}

/**
 * @brief Copies input string data into a buffer and increments the pointer by the number of bytes
 * copied.
 *
 * @param buffer Device buffer to copy to.
 * @param d_string String to copy.
 * @return Pointer to the end of the output buffer after the copy.
 */
__device__ inline char* copy_string(char* buffer, const string_view& d_string)
{
  return copy_and_increment(buffer, d_string.data(), d_string.size_bytes());
}

// This template is a thin wrapper around per-context singleton objects.
// It maintains a single object for each CUDA context.
template <typename TableType>
class per_context_cache {
 public:
  // Find an object cached for a current CUDA context.
  // If there is no object available in the cache, it calls the initializer
  // `init` to create a new one and cache it for later uses.
  template <typename Initializer>
  TableType* find_or_initialize(const Initializer& init)
  {
    int device_id;
    CUDF_CUDA_TRY(cudaGetDevice(&device_id));

    auto finder = cache_.find(device_id);
    if (finder == cache_.end()) {
      TableType* result = init();
      cache_[device_id] = result;
      return result;
    } else
      return finder->second;
  }

 private:
  std::unordered_map<int, TableType*> cache_;
};

// This template is a thread-safe version of per_context_cache.
template <typename TableType>
class thread_safe_per_context_cache : public per_context_cache<TableType> {
 public:
  template <typename Initializer>
  TableType* find_or_initialize(const Initializer& init)
  {
    std::lock_guard<std::mutex> guard(mutex);
    return per_context_cache<TableType>::find_or_initialize(init);
  }

 private:
  std::mutex mutex;
};

}  // namespace detail
}  // namespace strings
}  // namespace cudf
