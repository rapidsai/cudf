/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "io_source.hpp"

#include <cudf/ast/expressions.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <string>
#include <unordered_set>

/**
 * @file common_utils.hpp
 * @brief Utilities for `hybrid_scan_io` example
 */

/**
 * @brief Enum to represent the available parquet filters
 */
enum class parquet_filter_type : uint8_t {
  ROW_GROUPS_WITH_STATS               = 0,
  ROW_GROUPS_WITH_DICT_PAGES          = 1,
  ROW_GROUPS_WITH_BLOOM_FILTERS       = 2,
  FILTER_COLUMN_PAGES_WITH_PAGE_INDEX = 3,
  PAYLOAD_COLUMN_PAGES_WITH_ROW_MASK  = 4,
};

/**
 * @brief Get boolean from they keyword
 *
 * @param input keyword affirmation string such as: Y, T, YES, TRUE, ON
 * @return true or false
 */
[[nodiscard]] bool get_boolean(std::string input);

/**
 * @brief Create memory resource for libcudf functions
 *
 * @param pool Whether to use a pool memory resource.
 * @return Memory resource instance
 */
std::shared_ptr<rmm::mr::device_memory_resource> create_memory_resource(bool is_pool_used);

/**
 * @brief Function to process comma delimited input paths string to parquet files and/or dirs
 *        and convert them to specified io sources.
 *
 * Process the input path string containing directories (of parquet files) and/or individual
 * parquet files into a list of input parquet files, multiple the list by `input_multiplier`,
 * make sure to have at least `thread_count` files to satisfy at least file per parallel thread,
 * and convert the final list of files to a list of `io_source` and return.
 *
 * @param paths Comma delimited input paths string
 * @param input_multiplier Multiplier for the input files list
 * @param thread_count Number of threads being used in the example
 * @param io_source_type Specified IO source type to convert input files to
 * @param stream CUDA stream to use
 *
 * @return Vector of input sources for the given paths
 */
[[nodiscard]] std::vector<io_source> extract_input_sources(std::string const& paths,
                                                           int32_t input_multiplier,
                                                           int32_t thread_count,
                                                           io_source_type io_source_type,
                                                           rmm::cuda_stream_view stream);

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
 * @brief Converts a span of device buffers into a vector of corresponding device spans
 *
 * @tparam T Type of output device spans
 * @param buffers Host span of device buffers
 * @return Device spans corresponding to the input device buffers
 */
template <typename T>
std::vector<cudf::device_span<T const>> make_device_spans(
  cudf::host_span<rmm::device_buffer const> buffers)
  requires(sizeof(T) == 1)
{
  std::vector<cudf::device_span<T const>> device_spans(buffers.size());
  std::transform(buffers.begin(), buffers.end(), device_spans.begin(), [](auto const& buffer) {
    return cudf::device_span<T const>{static_cast<T const*>(buffer.data()), buffer.size()};
  });
  return device_spans;
}

/**
 * @brief Fetches a list of byte ranges from a host buffer into device buffers
 *
 * @param host_buffer Host buffer span
 * @param byte_ranges Byte ranges to fetch
 * @param stream CUDA stream
 * @param mr Device memory resource
 *
 * @return Device buffers
 */
std::vector<rmm::device_buffer> fetch_byte_ranges(
  cudf::host_span<uint8_t const> host_buffer,
  cudf::host_span<cudf::io::text::byte_range_info const> byte_ranges,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

/**
 * @brief Read parquet file with the next-gen parquet reader
 *
 * @tparam print_progress Boolean indicating whether to print progress
 *
 * @param io_source io source to read
 * @param filter_expression Filter expression
 * @param filters Set of parquet filters to apply
 * @param stream CUDA stream for hybrid scan reader
 * @param mr Device memory resource
 *
 * @return Tuple of filter table, payload table, filter metadata, payload metadata, and the final
 *         row validity column
 */
template <bool print_progress, bool single_step_materialize>
std::unique_ptr<cudf::table> hybrid_scan(io_source const& io_source,
                                         cudf::ast::expression const& filter_expression,
                                         std::unordered_set<parquet_filter_type> const& filters,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr);

/**
 * @brief Concatenate a vector of tables and return the resultant table
 *
 * @param tables Vector of tables to concatenate
 * @param stream CUDA stream to use
 *
 * @return Unique pointer to the resultant concatenated table.
 */
std::unique_ptr<cudf::table> concatenate_tables(std::vector<std::unique_ptr<cudf::table>> tables,
                                                rmm::cuda_stream_view stream);
