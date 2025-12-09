/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "io_source.hpp"

#include <cudf/io/datasource.hpp>
#include <cudf/io/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>

#include <thrust/host_vector.h>

#include <cstdint>
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
    {"HOST_BUFFER", io_source_type::HOST_BUFFER},
    {"PINNED_BUFFER", io_source_type::PINNED_BUFFER},
    {"FILEPATH", io_source_type::FILEPATH}};

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
  if (_io_type == io_source_type::HOST_BUFFER) {
    return cudf::host_span<uint8_t const>(reinterpret_cast<uint8_t const*>(_h_buffer.data()),
                                          _h_buffer.size());
  } else if (_io_type == io_source_type::PINNED_BUFFER) {
    return cudf::host_span<uint8_t const>(reinterpret_cast<uint8_t const*>(_pinned_buffer.data()),
                                          _pinned_buffer.size());
  } else {
    throw std::invalid_argument("Invalid io type to get host buffer span");
  }
}

namespace {
template <typename T>
cudf::io::source_info generate_source_info_from_file(std::string const& file_path, T&& container)
{
  CUDF_EXPECTS(container.size() == 0, "Container must have an initial size of 0");

  auto kvikio_datasource = cudf::io::datasource::create(file_path);
  std::size_t const offset{0};
  auto const file_size = kvikio_datasource->size();

  auto const num_bytes_read =
    kvikio_datasource->host_read(offset, file_size, reinterpret_cast<uint8_t*>(container.data()));
  CUDF_EXPECTS(num_bytes_read == file_size, "Failed to read expected number of bytes");

  auto source_info = cudf::io::source_info(cudf::host_span<std::byte const>(
    reinterpret_cast<std::byte const*>(container.data()), file_size));
  return source_info;
}
}  // namespace

io_source::io_source(std::string_view file_path, io_source_type type, rmm::cuda_stream_view stream)
  : _io_type(type), _pinned_buffer({pinned_memory_resource(), stream})
{
  std::string const file_name{file_path};

  // Copy file contents to the specified io source buffer
  switch (type) {
    case io_source_type::HOST_BUFFER: {
      _source_info = generate_source_info_from_file(file_name, _h_buffer);
      break;
    }
    case io_source_type::PINNED_BUFFER: {
      _source_info = generate_source_info_from_file(file_name, _pinned_buffer);
      break;
    }
    case io_source_type::FILEPATH: {
      _source_info = cudf::io::source_info{file_name};
      break;
    }
    default: {
      throw std::runtime_error("Encountered unexpected source type\n\n");
    }
  }
}
