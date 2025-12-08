/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/io/datasource.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/utilities/span.hpp>

#include <memory>

class datasource_reader {
 public:
  explicit datasource_reader(std::string const& filepath, rmm::cuda_stream_view stream);

  [[nodiscard]] std::vector<uint8_t> fetch_footer_bytes();

  [[nodiscard]] std::vector<uint8_t> fetch_page_index_bytes(
    cudf::io::text::byte_range_info const page_index_bytes);

  [[nodiscard]] std::vector<rmm::device_buffer> fetch_byte_ranges_to_device(
    cudf::host_span<cudf::io::text::byte_range_info const> byte_ranges,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

 private:
  void fetch_byte_ranges_to_host(size_t offset, size_t size, uint8_t* dst);

  std::unique_ptr<cudf::io::datasource> _datasource;
  rmm::cuda_stream_view _stream;
};
