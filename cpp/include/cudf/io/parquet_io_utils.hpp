/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/io/datasource.hpp>
#include <cudf/io/detail/parquet_io_utils.hpp>
#include <cudf/io/text/byte_range_info.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device_memory_resource.hpp>

#include <future>
#include <vector>

/**
 * @file parquet_io_utils.hpp
 * @brief IO utilities for the Parquet and Hybrid scan readers
 */

namespace CUDF_EXPORT cudf {
namespace io::parquet {

/**
 * @brief Fetches a host buffer of Parquet footer bytes from the input data source
 *
 * @ingroup io_readers
 *
 * @param datasource Input data source
 * @return Host buffer containing footer bytes
 */
std::unique_ptr<cudf::io::datasource::buffer> fetch_footer_to_host(
  cudf::io::datasource& datasource);

/**
 * @brief Fetches a host buffer of Parquet page index from the input data source
 *
 * @ingroup io_readers
 *
 * @param datasource Input datasource
 * @param page_index_bytes Byte range of page index
 * @return Host buffer containing page index bytes
 */
std::unique_ptr<cudf::io::datasource::buffer> fetch_page_index_to_host(
  cudf::io::datasource& datasource, cudf::io::text::byte_range_info const page_index_bytes);

/**
 * @brief Fetches a list of byte ranges from a host buffer into device buffers
 *
 * @ingroup io_readers
 *
 * @param datasource Input datasource
 * @param byte_ranges Byte ranges to fetch
 * @param stream CUDA stream
 * @param mr Device memory resource
 *
 * @return A tuple containing the device buffers, the device spans of the fetched data, and a future
 * to wait on the read tasks
 */
std::tuple<std::vector<rmm::device_buffer>,
           std::vector<cudf::device_span<uint8_t const>>,
           std::future<void>>
fetch_byte_ranges_to_device_async(
  cudf::io::datasource& datasource,
  cudf::host_span<cudf::io::text::byte_range_info const> byte_ranges,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

}  // namespace io::parquet
}  // namespace CUDF_EXPORT cudf
