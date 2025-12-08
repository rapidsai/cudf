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

  host_read(0, header_len, header_buffer.data());
  auto const header = reinterpret_cast<file_header_s const*>(header_buffer.data());
  host_read(len - ender_len, ender_len, ender_buffer.data());
  auto const ender = reinterpret_cast<file_ender_s const*>(ender_buffer.data());
  CUDF_EXPECTS(len > header_len + ender_len, "Incorrect data source");
  constexpr uint32_t parquet_magic = (('P' << 0) | ('A' << 8) | ('R' << 16) | ('1' << 24));
  CUDF_EXPECTS(header->magic == parquet_magic && ender->magic == parquet_magic,
               "Corrupted header or footer");
  CUDF_EXPECTS(ender->footer_len != 0 && ender->footer_len <= (len - header_len - ender_len),
               "Incorrect footer length");

  std::vector<uint8_t> footer(ender->footer_len);
  host_read(len - ender->footer_len - ender_len, ender->footer_len, footer.data());
  return footer;
}

void datasource_reader::host_read(size_t offset, size_t size, uint8_t* dst)
{
  auto num_bytes_read = _datasource->host_read(offset, size, dst);
  CUDF_EXPECTS(num_bytes_read == size, "Failed to read expected number of bytes");
}
