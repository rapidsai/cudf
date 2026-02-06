/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/io/text/byte_range_info.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device_memory_resource.hpp>

/**
 * @file io_utils.cpp
 * @brief Definitions for IO utilities for hybrid_scan examples
 */

cudf::host_span<uint8_t const> make_host_span(
  std::reference_wrapper<cudf::io::datasource::buffer const> buffer)
{
  return cudf::host_span<uint8_t const>{static_cast<uint8_t const*>(buffer.get().data()),
                                        buffer.get().size()};
}

std::unique_ptr<cudf::io::datasource::buffer> fetch_footer_bytes(cudf::io::datasource& datasource)
{
  // Using libcudf utility but may have custom implementation in the future
  return cudf::io::parquet::fetch_footer_to_host(datasource);
}

std::unique_ptr<cudf::io::datasource::buffer> fetch_page_index_bytes(
  cudf::io::datasource& datasource, cudf::io::text::byte_range_info const page_index_bytes)
{
  // Using libcudf utility but may have custom implementation in the future
  return cudf::io::parquet::fetch_page_index_to_host(datasource, page_index_bytes);
}

std::tuple<std::vector<rmm::device_buffer>,
           std::vector<cudf::device_span<uint8_t const>>,
           std::future<void>>
fetch_byte_ranges_async(cudf::io::datasource& datasource,
                        cudf::host_span<cudf::io::text::byte_range_info const> byte_ranges,
                        rmm::cuda_stream_view stream,
                        rmm::device_async_resource_ref mr)
{
  // Using libcudf utility but may have custom implementation in the future
  return cudf::io::parquet::fetch_byte_ranges_to_device_async(datasource, byte_ranges, stream, mr);
}
