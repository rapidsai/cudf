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

#include <cudf/io/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

#include <thrust/host_vector.h>

#include <string>

/**
 * @file io_source.hpp
 * @brief Utilities for constructing the specified IO sources from the input parquet files.
 *
 */

/**
 * @brief Available IO source types
 */
enum class io_source_type { FILEPATH, HOST_BUFFER, PINNED_BUFFER, DEVICE_BUFFER };

/**
 * @brief Get io source type from the string keyword argument
 *
 * @param name io source type keyword name
 * @return io source type
 */
[[nodiscard]] io_source_type get_io_source_type(std::string name);

/**
 * @brief Create and return a reference to a static pinned memory pool
 *
 * @return Reference to a static pinned memory pool
 */
rmm::host_async_resource_ref pinned_memory_resource();

/**
 * @brief Custom allocator for pinned_buffer via RMM.
 */
template <typename T>
struct pinned_allocator : public std::allocator<T> {
  pinned_allocator(rmm::host_async_resource_ref _mr, rmm::cuda_stream_view _stream)
    : mr{_mr}, stream{_stream}
  {
  }

  T* allocate(std::size_t n)
  {
    auto ptr = mr.allocate_async(n * sizeof(T), rmm::RMM_DEFAULT_HOST_ALIGNMENT, stream);
    stream.synchronize();
    return static_cast<T*>(ptr);
  }

  void deallocate(T* ptr, std::size_t n)
  {
    mr.deallocate_async(ptr, n * sizeof(T), rmm::RMM_DEFAULT_HOST_ALIGNMENT, stream);
  }

 private:
  rmm::host_async_resource_ref mr;
  rmm::cuda_stream_view stream;
};

/**
 * @brief Class to create a cudf::io::source_info of given type from the input parquet file
 *
 */
class io_source {
 public:
  io_source(std::string_view file_path, io_source_type io_type, rmm::cuda_stream_view stream);

  // Get the internal source info
  [[nodiscard]] cudf::io::source_info get_source_info() const { return source_info; }

 private:
  // alias for pinned vector
  template <typename T>
  using pinned_vector = thrust::host_vector<T, pinned_allocator<T>>;
  cudf::io::source_info source_info;
  std::vector<char> h_buffer;
  pinned_vector<char> pinned_buffer;
  rmm::device_uvector<std::byte> d_buffer;
};
