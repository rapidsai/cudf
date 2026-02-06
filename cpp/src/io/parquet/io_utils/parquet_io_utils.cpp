/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "io/parquet/parquet_common.hpp"

#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/io/text/byte_range_info.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device_memory_resource.hpp>

#include <numeric>

/**
 * @file parquet_io_utils.cpp
 * @brief Definitions for IO utilities for the Parquet and hybrid scan readers
 */

namespace cudf::io::parquet {

std::unique_ptr<cudf::io::datasource::buffer> fetch_footer_to_host(cudf::io::datasource& datasource)
{
  constexpr auto header_len = sizeof(file_header_s);
  constexpr auto ender_len  = sizeof(file_ender_s);
  size_t const len          = datasource.size();

  auto header_buffer = datasource.host_read(0, header_len);
  auto const header  = reinterpret_cast<file_header_s const*>(header_buffer->data());
  auto ender_buffer  = datasource.host_read(len - ender_len, ender_len);
  auto const ender   = reinterpret_cast<file_ender_s const*>(ender_buffer->data());
  CUDF_EXPECTS(len > header_len + ender_len, "Incorrect data source");
  CUDF_EXPECTS(header->magic == detail::parquet_magic, "Corrupted header");
  CUDF_EXPECTS(ender->magic == detail::parquet_magic, "Corrupted footer");
  CUDF_EXPECTS(ender->footer_len != 0 && ender->footer_len <= (len - header_len - ender_len),
               "Incorrect footer length");

  return datasource.host_read(len - ender->footer_len - ender_len, ender->footer_len);
}

std::unique_ptr<cudf::io::datasource::buffer> fetch_page_index_to_host(
  cudf::io::datasource& datasource, cudf::io::text::byte_range_info const page_index_bytes)
{
  return datasource.host_read(page_index_bytes.offset(), page_index_bytes.size());
}

std::tuple<std::vector<rmm::device_buffer>,
           std::vector<cudf::device_span<uint8_t const>>,
           std::future<void>>
fetch_byte_ranges_to_device_async(
  cudf::io::datasource& datasource,
  cudf::host_span<cudf::io::text::byte_range_info const> byte_ranges,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  static std::mutex mutex;

  // Allocate device spans for each column chunk
  std::vector<cudf::device_span<uint8_t const>> column_chunk_data{};
  column_chunk_data.reserve(byte_ranges.size());

  auto total_size = std::accumulate(
    byte_ranges.begin(), byte_ranges.end(), std::size_t{0}, [&](auto acc, auto const& range) {
      return acc + range.size();
    });

  // Allocate single device buffer for all column chunks
  std::vector<rmm::device_buffer> column_chunk_buffers{};
  column_chunk_buffers.emplace_back(total_size, stream, mr);
  auto buffer_data = static_cast<uint8_t*>(column_chunk_buffers.back().data());
  std::ignore      = std::accumulate(
    byte_ranges.begin(), byte_ranges.end(), std::size_t{0}, [&](auto acc, auto const& range) {
      column_chunk_data.emplace_back(buffer_data + acc, range.size());
      return acc + range.size();
    });

  std::vector<std::future<size_t>> device_read_tasks{};
  std::vector<std::future<size_t>> host_read_tasks{};
  device_read_tasks.reserve(byte_ranges.size());
  host_read_tasks.reserve(byte_ranges.size());
  {
    std::lock_guard<std::mutex> lock(mutex);

    for (size_t chunk = 0; chunk < byte_ranges.size();) {
      size_t const io_offset = byte_ranges[chunk].offset();
      size_t io_size         = byte_ranges[chunk].size();
      size_t next_chunk      = chunk + 1;
      while (next_chunk < byte_ranges.size()) {
        size_t const next_offset = byte_ranges[next_chunk].offset();
        if (next_offset != io_offset + io_size) { break; }
        io_size += byte_ranges[next_chunk].size();
        next_chunk++;
      }

      if (io_size != 0) {
        auto dest = const_cast<uint8_t*>(column_chunk_data[chunk].data());
        // Directly read the column chunk data to the device
        // buffer if supported
        if (datasource.supports_device_read() and datasource.is_device_read_preferred(io_size)) {
          device_read_tasks.emplace_back(
            datasource.device_read_async(io_offset, io_size, dest, stream));
        } else {
          // Read the column chunk data to the host buffer and
          // copy it to the device buffer
          host_read_tasks.emplace_back(
            std::async(std::launch::deferred, [&datasource, io_offset, io_size, dest, stream]() {
              auto host_buffer = datasource.host_read(io_offset, io_size);
              cudf::detail::cuda_memcpy_async(
                cudf::device_span<uint8_t>{dest, io_size},
                cudf::host_span<uint8_t const>{host_buffer->data(), io_size},
                stream);
              return io_size;
            }));
        }
      }
      chunk = next_chunk;
    }
  }

  auto sync_function = [](decltype(host_read_tasks) host_read_tasks,
                          decltype(device_read_tasks) device_read_tasks) {
    for (auto& task : host_read_tasks) {
      task.get();
    }
    for (auto& task : device_read_tasks) {
      task.get();
    }
  };
  return {std::move(column_chunk_buffers),
          std::move(column_chunk_data),
          std::async(std::launch::deferred,
                     sync_function,
                     std::move(host_read_tasks),
                     std::move(device_read_tasks))};
}

}  // namespace cudf::io::parquet
