/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "host_buffer_source.hpp"

#include <cudf/ast/expressions.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>
#include <string>

/**
 * @file common_utils.hpp
 * @brief Utilities for `hybrid_scan_io` example
 */

/**
 * @brief Create memory resource for libcudf functions
 *
 * @param pool Whether to use a pool memory resource.
 * @return Memory resource instance
 */
std::shared_ptr<rmm::mr::device_memory_resource> create_memory_resource(bool is_pool_used);

/**
 * @brief Create a filter expression of the form `column_name == literal` for string type point
 * lookups
 *
 * @param column_name String column name
 * @param literal String literal value
 * @return Filter expression
 */
cudf::ast::operation create_filter_expression(std::string const& column_name,
                                              std::string const& literal_value);

/**
 * @brief Combine columns from filter and payload tables into a single table
 *
 * @param filter_table Filter table
 * @param payload_table Payload table
 * @return Combined table
 */
std::unique_ptr<cudf::table> combine_tables(std::unique_ptr<cudf::table> filter_table,
                                            std::unique_ptr<cudf::table> payload_table);

/**
 * @brief Check if two tables are identical, throw an error otherwise
 *
 * @param lhs_table View to lhs table
 * @param rhs_table View to rhs table
 * @param stream CUDA stream to use
 */
void check_tables_equal(cudf::table_view const& lhs_table,
                        cudf::table_view const& rhs_table,
                        rmm::cuda_stream_view stream = cudf::get_default_stream());

class io_backend {
 public:
  /**
   * @brief Constructs a new I/O backend object from an in-memory host data source
   *
   * @param buffer Host memory that contains the file data
   * @param stream CUDA stream
   */
  explicit io_backend(cudf::host_span<std::byte const> buffer, rmm::cuda_stream_view stream);

  /**
   * @brief Constructs a new I/O backend object from a file data source
   *
   * @param filepath Path to a file on the disk
   * @param stream CUDA stream
   */
  explicit io_backend(std::string const& filepath, rmm::cuda_stream_view stream);

  /**
   * @brief Fetches a host span of Parquet footer bytes from the data source
   *
   * @return A host span of the footer bytes
   */
  [[nodiscard]] std::vector<uint8_t> fetch_footer_bytes();

  /**
   * @brief Fetches a host span of Parquet PageIndexbytes from the data source
   *
   * @param page_index_bytes Byte range of `PageIndex` to fetch
   * @return A host span of the PageIndex bytes
   */
  [[nodiscard]] std::vector<uint8_t> fetch_page_index_bytes(
    cudf::io::text::byte_range_info const page_index_bytes);

  /**
   * @brief Fetches a list of byte ranges from the data source into a vector of device buffers
   *
   * @param byte_ranges Byte ranges to fetch
   * @param stream CUDA stream
   * @param mr Device memory resource to create device buffers with
   *
   * @return Vector of device buffers
   */
  [[nodiscard]] std::vector<rmm::device_buffer> fetch_byte_ranges_to_device(
    cudf::host_span<cudf::io::text::byte_range_info const> byte_ranges,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

 private:
  /**
   * @brief Fetches a byte range from the data source into a preallocated buffer
   *
   * @param offset File offset
   * @param size Number of bytes to read
   * @param dst Destination host buffer
   */
  void fetch_byte_range_to_host(size_t offset, size_t size, uint8_t* dst);

  std::unique_ptr<host_buffer_source> _host_buffer_source;
  std::unique_ptr<cudf::io::datasource> _datasource;
  rmm::cuda_stream_view _stream;
};
