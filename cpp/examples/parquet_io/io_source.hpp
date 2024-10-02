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
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>

#include <thrust/host_vector.h>

#include <filesystem>
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
 * @brief Create and return a reference to a static pinned memory pool
 *
 * @return Reference to a static pinned memory pool
 */
rmm::host_async_resource_ref pinned_memory_resource()
{
  static auto mr = rmm::mr::pinned_host_memory_resource{};
  return mr;
}

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
 * @brief Get io source type from the string keyword argument
 *
 * @param name io source type keyword name
 * @return io source type
 */
[[nodiscard]] io_source_type get_io_source_type(std::string name)
{
  static std::unordered_map<std::string_view, io_source_type> const map = {
    {"FILEPATH", io_source_type::FILEPATH},
    {"HOST_BUFFER", io_source_type::HOST_BUFFER},
    {"PINNED_BUFFER", io_source_type::PINNED_BUFFER},
    {"DEVICE_BUFFER", io_source_type::DEVICE_BUFFER}};

  std::transform(name.begin(), name.end(), name.begin(), ::toupper);
  if (map.find(name) != map.end()) {
    return map.at(name);
  } else {
    throw std::invalid_argument(name +
                                " is not a valid io source type. Available: FILEPATH,\n"
                                "HOST_BUFFER, PINNED_BUFFER, DEVICE_BUFFER.\n\n");
  }
}

/**
 * @brief Class to create a cudf::io::source_info of given type from the input parquet file
 *
 */
class io_source {
 public:
  io_source(std::string_view file_path, io_source_type io_type, rmm::cuda_stream_view stream)
    : type{io_type},
      file_name{file_path},
      file_size{std::filesystem::file_size(file_name)},
      pinned_buffer({pinned_memory_resource(), stream}),
      d_buffer{0, stream}
  {
    // For filepath make a quick source_info and return early
    if (type == io_source_type::FILEPATH) {
      source_info = cudf::io::source_info(file_name);
      return;
    }

    std::ifstream file{file_name, std::ifstream::binary};

    // Copy file contents to the specified io source buffer
    switch (type) {
      case io_source_type::HOST_BUFFER: {
        h_buffer.resize(file_size);
        file.read(h_buffer.data(), file_size);
        source_info = cudf::io::source_info(h_buffer.data(), file_size);
        break;
      }
      case io_source_type::PINNED_BUFFER: {
        pinned_buffer.resize(file_size);
        file.read(pinned_buffer.data(), file_size);
        source_info = cudf::io::source_info(pinned_buffer.data(), file_size);
        break;
      }
      case io_source_type::DEVICE_BUFFER: {
        h_buffer.resize(file_size);
        file.read(h_buffer.data(), file_size);
        d_buffer.resize(file_size, stream);
        CUDF_CUDA_TRY(cudaMemcpyAsync(
          d_buffer.data(), h_buffer.data(), file_size, cudaMemcpyDefault, stream.value()));

        source_info = cudf::io::source_info(d_buffer);
        break;
      }
      default: {
        throw std::runtime_error("Encountered unexpected source type\n\n");
      }
    }
  }

  // Get the internal source info
  [[nodiscard]] cudf::io::source_info get_source_info() const { return source_info; }

 private:
  // alias for pinned vector
  template <typename T>
  using pinned_vector = thrust::host_vector<T, pinned_allocator<T>>;

  io_source_type const type;
  std::string const file_name;
  size_t const file_size;
  cudf::io::source_info source_info;
  std::vector<char> h_buffer;
  pinned_vector<char> pinned_buffer;
  rmm::device_uvector<std::byte> d_buffer;
};
