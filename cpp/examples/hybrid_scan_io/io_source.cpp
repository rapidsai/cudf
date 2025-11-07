/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "io_source.hpp"

#include <cudf/io/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>

#include <thrust/host_vector.h>

#include <filesystem>
#include <fstream>
#include <string>

rmm::host_async_resource_ref pinned_memory_resource()
{
  static auto mr = rmm::mr::pinned_host_memory_resource{};
  return mr;
}

io_source_type get_io_source_type(std::string name)
{
  static std::unordered_map<std::string_view, io_source_type> const map = {
    {"HOST_BUFFER", io_source_type::HOST_BUFFER}, {"PINNED_BUFFER", io_source_type::PINNED_BUFFER}};

  std::transform(name.begin(), name.end(), name.begin(), ::toupper);
  if (map.find(name) != map.end()) {
    return map.at(name);
  } else {
    throw std::invalid_argument(name +
                                " is not a valid io source type. Available: HOST_BUFFER, "
                                "PINNED_BUFFER.\n\n");
  }
}

cudf::host_span<uint8_t const> io_source::get_host_buffer_span() const
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

io_source::io_source(std::string_view file_path, io_source_type type, rmm::cuda_stream_view stream)
  : io_type(type), pinned_buffer({pinned_memory_resource(), stream})
{
  std::string const file_name{file_path};
  auto const file_size = std::filesystem::file_size(file_name);

  std::ifstream file{file_name, std::ifstream::binary};

  // Copy file contents to the specified io source buffer
  switch (type) {
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
    default: {
      throw std::runtime_error("Encountered unexpected source type\n\n");
    }
  }
}
