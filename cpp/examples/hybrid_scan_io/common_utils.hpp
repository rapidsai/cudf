/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cudf/ast/expressions.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>

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

/**
 * @brief Fetches a host span of Parquet footer bytes from the input buffer span
 *
 * @param buffer Input buffer span
 * @return A host span of the footer bytes
 */
cudf::host_span<uint8_t const> fetch_footer_bytes(cudf::host_span<uint8_t const> buffer);
/**
 * @brief Fetches a host span of Parquet PageIndexbytes from the input buffer span
 *
 * @param buffer Input buffer span
 * @param page_index_bytes Byte range of `PageIndex` to fetch
 * @return A host span of the PageIndex bytes
 */
cudf::host_span<uint8_t const> fetch_page_index_bytes(
  cudf::host_span<uint8_t const> buffer, cudf::io::text::byte_range_info const page_index_bytes);

/**
 * @brief Fetches a list of byte ranges from a host buffer into a vector of device buffers
 *
 * @param host_buffer Host buffer span
 * @param byte_ranges Byte ranges to fetch
 * @param stream CUDA stream
 * @param mr Device memory resource to create device buffers with
 *
 * @return Vector of device buffers
 */
std::vector<rmm::device_buffer> fetch_byte_ranges(
  cudf::host_span<uint8_t const> host_buffer,
  cudf::host_span<cudf::io::text::byte_range_info const> byte_ranges,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);
