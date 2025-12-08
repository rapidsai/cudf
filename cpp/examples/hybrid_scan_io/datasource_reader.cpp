/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "datasource_reader.hpp"

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/utilities/error.hpp>

#include <array>

datasource_reader::datasource_reader(std::string const& filepath, rmm::cuda_stream_view stream)
  : _datasource(cudf::io::datasource::create(filepath)), _stream(stream)
{
  CUDF_EXPECTS(_datasource != nullptr, "Failed to create datasource for: " + filepath);
}

std::vector<uint8_t> datasource_reader::fetch_footer_bytes()
{
  CUDF_FUNC_RANGE();

  using namespace cudf::io::parquet;

  constexpr auto header_len = sizeof(file_header_s);
  constexpr auto ender_len  = sizeof(file_ender_s);
  size_t const len          = _datasource->size();

  std::array<uint8_t, header_len> header_buffer;
  std::array<uint8_t, ender_len> ender_buffer;

  fetch_byte_ranges_to_host(0, header_len, header_buffer.data());
  auto const header = reinterpret_cast<file_header_s const*>(header_buffer.data());
  fetch_byte_ranges_to_host(len - ender_len, ender_len, ender_buffer.data());
  auto const ender = reinterpret_cast<file_ender_s const*>(ender_buffer.data());
  CUDF_EXPECTS(len > header_len + ender_len, "Incorrect data source");
  constexpr uint32_t parquet_magic = (('P' << 0) | ('A' << 8) | ('R' << 16) | ('1' << 24));
  CUDF_EXPECTS(header->magic == parquet_magic && ender->magic == parquet_magic,
               "Corrupted header or footer");
  CUDF_EXPECTS(ender->footer_len != 0 && ender->footer_len <= (len - header_len - ender_len),
               "Incorrect footer length");

  std::vector<uint8_t> footer(ender->footer_len);
  fetch_byte_ranges_to_host(len - ender->footer_len - ender_len, ender->footer_len, footer.data());
  return footer;
}

std::vector<uint8_t> datasource_reader::fetch_page_index_bytes(
  cudf::io::text::byte_range_info const page_index_bytes)
{
  CUDF_FUNC_RANGE();
  std::vector<uint8_t> page_index(page_index_bytes.size());
  _datasource->host_read(page_index_bytes.offset(), page_index_bytes.size(), page_index.data());
  return page_index;
}

void datasource_reader::fetch_byte_ranges_to_host(size_t offset, size_t size, uint8_t* dst)
{
  CUDF_FUNC_RANGE();
  auto num_bytes_read = _datasource->host_read(offset, size, dst);
  CUDF_EXPECTS(num_bytes_read == size, "Failed to read expected number of bytes");
}

std::vector<rmm::device_buffer> datasource_reader::fetch_byte_ranges_to_device(
  cudf::host_span<cudf::io::text::byte_range_info const> byte_ranges,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  if (byte_ranges.empty()) { return {}; }

  CUDF_EXPECTS(_datasource->supports_device_read(), "Device read not supported");

  std::vector<rmm::device_buffer> buffers;
  buffers.reserve(byte_ranges.size());

  std::vector<std::future<size_t>> futures;
  futures.reserve(byte_ranges.size());

  for (auto const& byte_range : byte_ranges) {
    buffers.emplace_back(byte_range.size(), stream, mr);

    if (byte_range.size() == 0) { continue; }

    auto future = _datasource->device_read_async(byte_range.offset(),
                                                 byte_range.size(),
                                                 static_cast<uint8_t*>(buffers.back().data()),
                                                 _stream);
    futures.push_back(std::move(future));
  }

  for (auto&& future : futures) {
    future.get();
  }

  return buffers;
}
