/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/io/datasource.hpp>
#include <cudf/io/text/byte_range_info.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <functional>
#include <future>
#include <tuple>
#include <vector>

/**
 * @file parquet_io_utils.hpp
 * @brief IO utilities for the Parquet and Hybrid scan readers
 */

namespace CUDF_EXPORT cudf {
namespace io::parquet {

/**
 * @addtogroup io_utils
 * @{
 * @file
 */

//! Using `byte_range_info` from cudf::io::text
using cudf::io::text::byte_range_info;

/**
 * @brief Fetches a host buffer of Parquet footer bytes from the input data source
 *
 * @ingroup io_utils
 *
 * @param datasource Input data source
 * @return Host buffer containing footer bytes
 */
[[nodiscard]] std::unique_ptr<cudf::io::datasource::buffer> fetch_footer_to_host(
  cudf::io::datasource& datasource);

/**
 * @brief Fetches host buffers of Parquet footer bytes from multiple input data sources
 *
 * @ingroup io_utils
 *
 * @param datasources Input data sources
 * @return Vector of host buffers containing footer bytes, one per datasource
 *
 * @throw cudf::logic_error if any datasource contains a corrupted Parquet magic number, header or
 * footer, or has an invalid footer length.
 */
[[nodiscard]] std::vector<std::unique_ptr<cudf::io::datasource::buffer>> fetch_footers_to_host(
  cudf::host_span<std::reference_wrapper<cudf::io::datasource> const> datasources);

/**
 * @brief Fetches a host buffer of Parquet page index from the input data source
 *
 * @ingroup io_utils
 *
 * @param datasource Input datasource
 * @param page_index_bytes Byte range of page index
 * @return Host buffer containing page index bytes
 */
[[nodiscard]] std::unique_ptr<cudf::io::datasource::buffer> fetch_page_index_to_host(
  cudf::io::datasource& datasource, byte_range_info const page_index_bytes);

/**
 * @brief Fetches host buffers of Parquet page index bytes from multiple input data sources
 *
 * @ingroup io_utils
 *
 * @param datasources Input datasources
 * @param page_index_bytes_per_source Byte ranges of page index, one per datasource
 * @return Vector of host buffers containing page index bytes, one per datasource
 *
 * @throw cudf::logic_error if the number of datasources does not match the number of page index
 * byte ranges
 * @throw std::out_of_range if any page index byte range is out of range for its datasource
 */
[[nodiscard]] std::vector<std::unique_ptr<cudf::io::datasource::buffer>> fetch_page_indexes_to_host(
  cudf::host_span<std::reference_wrapper<cudf::io::datasource> const> datasources,
  cudf::host_span<byte_range_info const> page_index_bytes_per_source);

/**
 * @brief Fetches a list of byte ranges from a datasource into device buffers
 *
 * @ingroup io_utils
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
fetch_byte_ranges_to_device_async(cudf::io::datasource& datasource,
                                  cudf::host_span<byte_range_info const> byte_ranges,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr);

/**
 * @brief Fetches lists of byte ranges from multiple datasources into device buffers
 *
 * @ingroup io_utils
 *
 * @param datasources Input datasources
 * @param byte_ranges_per_source Vector of byte ranges to fetch, one per datasource
 * @param stream CUDA stream
 * @param mr Device memory resource
 *
 * @return A tuple containing a vector of device buffers, a vector of vectors of device spans (one
 * per byte range per datasource), and a future to wait on the read tasks
 */
std::tuple<std::vector<rmm::device_buffer>,
           std::vector<std::vector<cudf::device_span<uint8_t const>>>,
           std::future<void>>
fetch_byte_ranges_to_device_async(
  cudf::host_span<std::reference_wrapper<cudf::io::datasource> const> datasources,
  cudf::host_span<std::vector<byte_range_info> const> byte_ranges_per_source,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

/** @} */  // end of group
}  // namespace io::parquet
}  // namespace CUDF_EXPORT cudf
