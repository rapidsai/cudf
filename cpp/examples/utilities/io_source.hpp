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

#include "mr_utils.hpp"

#include <cudf/io/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <thrust/host_vector.h>

#include <filesystem>
#include <fstream>
#include <string>

/**
 * @file io_source.hpp
 * @brief Utilities for constructing the specified IO sources from the input parquet files.
 *
 */

namespace cudf::examples {

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
  io_source(std::string_view file_path, io_source_type io_type, rmm::cuda_stream_view stream)
    : io_type(io_type),
      pinned_buffer({create_pinned_memory_resource(), stream}),
      d_buffer{0, stream}
  {
    std::string const file_name{file_path};
    auto const file_size = std::filesystem::file_size(file_name);

    // For filepath make a quick source_info and return early
    if (io_type == io_source_type::FILEPATH) {
      source_info = cudf::io::source_info(file_name);
      return;
    }

    std::ifstream file{file_name, std::ifstream::binary};

    // Copy file contents to the specified io source buffer
    switch (io_type) {
      case io_source_type::HOST_BUFFER: {
        h_buffer.resize(file_size);
        file.read(h_buffer.data(), file_size);
        source_info = cudf::io::source_info(cudf::host_span<std::byte const>(
          reinterpret_cast<std::byte const*>(h_buffer.data()), file_size));
        break;
      }
      case io_source_type::PINNED_BUFFER: {
        pinned_buffer.resize(file_size);
        file.read(pinned_buffer.data(), file_size);
        source_info = cudf::io::source_info(cudf::host_span<std::byte const>(
          reinterpret_cast<std::byte const*>(pinned_buffer.data()), file_size));
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
  [[nodiscard]] cudf::io::source_info const& get_source_info() const { return source_info; }

  [[nodiscard]] io_source_type get_source_type() const { return io_type; }

  // Get the internal buffer span
  [[nodiscard]] cudf::host_span<uint8_t const> get_host_buffer_span() const
  {
    if (io_type == io_source_type::HOST_BUFFER) {
      return cudf::host_span<uint8_t const>(reinterpret_cast<uint8_t const*>(h_buffer.data()),
                                            h_buffer.size());
    } else if (io_type == io_source_type::PINNED_BUFFER) {
      return cudf::host_span<uint8_t const>(reinterpret_cast<uint8_t const*>(pinned_buffer.data()),
                                            pinned_buffer.size());
    } else {
      throw std::invalid_argument("Invalid io type to get host buffer span");
    }
  }

 private:
  // alias for pinned vector
  template <typename T>
  using pinned_vector = thrust::host_vector<T, pinned_allocator<T>>;

  cudf::io::source_info source_info;
  io_source_type io_type;
  std::vector<char> h_buffer;
  pinned_vector<char> pinned_buffer;
  rmm::device_uvector<std::byte> d_buffer;
};

}  // namespace cudf::examples
