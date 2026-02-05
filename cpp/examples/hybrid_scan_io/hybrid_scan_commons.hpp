/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "io_source.hpp"

#include <cudf/ast/expressions.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <unordered_set>

/**
 * @file hybrid_scan_commons.hpp
 * @brief Common hybrid scan related functions for hybrid_scan examples
 */

/**
 * @brief Enum to represent the available hybrid scan filters
 */
enum class hybrid_scan_filter_type : uint8_t {
  ROW_GROUPS_WITH_STATS               = 0,
  ROW_GROUPS_WITH_DICT_PAGES          = 1,
  ROW_GROUPS_WITH_BLOOM_FILTERS       = 2,
  FILTER_COLUMN_PAGES_WITH_PAGE_INDEX = 3,
  PAYLOAD_COLUMN_PAGES_WITH_ROW_MASK  = 4,
};

using cudf::io::parquet::experimental::hybrid_scan_reader;

namespace detail {

/**
 * @brief Sets up the a hybrid scan reader instance
 *
 * @param datasource Data source
 * @param options Parquet reader options
 * @param use_page_index Whether to set up page index
 * @param verbose Whether to print verbose output
 *
 * @return Hybrid scan reader
 */
std::unique_ptr<hybrid_scan_reader> setup_reader(cudf::io::datasource& datasource,
                                                 cudf::io::parquet_reader_options const& options,
                                                 bool use_page_index,
                                                 bool verbose);

/**
 * @brief Applies specified row group filters
 *
 * @param datasource Data source
 * @param reader Hybrid scan reader
 * @param filters Set of hybrid scanfilters to apply
 * @param input_row_group_indices Span of input row group indices
 * @param options Parquet reader options
 * @param verbose Whether to print verbose output
 * @param stream CUDA stream
 * @param mr Device memory resource
 *
 * @return Filtered row group indices
 */
std::vector<cudf::size_type> apply_row_group_filters(
  cudf::io::datasource& datasource,
  cudf::io::parquet::experimental::hybrid_scan_reader const& reader,
  std::unordered_set<hybrid_scan_filter_type> const& filters,
  cudf::host_span<cudf::size_type> input_row_group_indices,
  cudf::io::parquet_reader_options const& options,
  bool verbose,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

/**
 * @brief Materializes all parquet columns in single step mode
 *
 * @param datasource Data source
 * @param reader Hybrid scan reader
 * @param current_row_group_indices Span of current row group indices
 * @param options Parquet reader options
 * @param verbose Whether to print verbose output
 * @param stream CUDA stream
 * @param mr Device memory resource to allocate memory for the read table
 *
 * @return Unique pointer to the read table
 */
std::unique_ptr<cudf::table> single_step_materialize(
  cudf::io::datasource& datasource,
  cudf::io::parquet::experimental::hybrid_scan_reader const& reader,
  cudf::host_span<cudf::size_type> current_row_group_indices,
  cudf::io::parquet_reader_options const& options,
  bool verbose,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

/**
 * @brief Materializes all parquet columns in two step mode
 *
 * @param datasource Data source
 * @param reader Hybrid scan reader
 * @param filters Set of hybrid scan filters to apply
 * @param current_row_group_indices Span of current row group indices
 * @param options Parquet reader options
 * @param verbose Whether to print verbose output
 * @param stream CUDA stream
 * @param mr Device memory resource
 *
 * @return Unique pointer to the read table
 */
std::unique_ptr<cudf::table> two_step_materialize(
  cudf::io::datasource& datasource,
  cudf::io::parquet::experimental::hybrid_scan_reader const& reader,
  std::unordered_set<hybrid_scan_filter_type> const& filters,
  cudf::host_span<cudf::size_type> current_row_group_indices,
  cudf::io::parquet_reader_options const& options,
  bool verbose,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

}  // namespace detail

/**
 * @brief Read parquet file with the hybrid scan reader
 *
 * @param io_source IO source information
 * @param filter_expression AST filter expression
 * @param filters Set of hybrid scan filters to apply
 * @param stream CUDA stream for hybrid scan reader
 * @param mr Device memory resource to allocate memory for the read table
 *
 * @return Unique pointer to the read table
 */
template <bool single_step_read, bool use_page_index>
std::unique_ptr<cudf::table> inline hybrid_scan(
  io_source const& io_source,
  cudf::ast::expression const& filter_expression,
  std::unordered_set<hybrid_scan_filter_type> const& filters,
  bool verbose,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  static_assert(single_step_read or use_page_index,
                "Page index is needed for two-step parquet read");

  CUDF_FUNC_RANGE();

  auto options = cudf::io::parquet_reader_options::builder().filter(filter_expression).build();

  // Input file buffer span
  auto datasource     = std::move(cudf::io::make_datasources(io_source.get_source_info()).front());
  auto datasource_ref = std::ref(*datasource);

  // Setup reader
  auto reader           = detail::setup_reader(datasource_ref, options, use_page_index, verbose);
  auto const reader_ref = std::cref(*reader);

  // Start with all row groups
  auto all_row_group_indices = reader->all_row_groups(options);

  // Filter row groups
  auto filtered_row_group_indices = detail::apply_row_group_filters(
    datasource_ref, reader_ref, filters, all_row_group_indices, options, verbose, stream, mr);

  // Materialize filter and payload columns separately
  if constexpr (single_step_read) {
    return detail::single_step_materialize(
      datasource_ref, reader_ref, filtered_row_group_indices, options, verbose, stream, mr);
  } else {
    return detail::two_step_materialize(datasource_ref,
                                        reader_ref,
                                        filters,
                                        filtered_row_group_indices,
                                        options,
                                        verbose,
                                        stream,
                                        mr);
  }
}
