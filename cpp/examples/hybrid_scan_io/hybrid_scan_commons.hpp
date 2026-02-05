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

#include <thrust/iterator/counting_iterator.h>

#include <algorithm>
#include <thread>
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
 * @param verbose Whether to print verbose output
 *
 * @return Hybrid scan reader
 */
std::unique_ptr<hybrid_scan_reader> setup_reader(cudf::io::datasource& datasource,
                                                 cudf::io::parquet_reader_options const& options,
                                                 bool verbose);

/**
 * @brief Sets up the page index for the hybrid scan reader
 *
 * @param datasource Data source
 * @param reader Hybrid scan reader
 * @param single_step_read Whether to use single step read mode
 * @param verbose Whether to print verbose output
 */
void setup_page_index(cudf::io::datasource& datasource,
                      cudf::io::parquet::experimental::hybrid_scan_reader const& reader,
                      bool single_step_read,
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
  rmm::cuda_stream_view stream);

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
  std::optional<cudf::ast::operation const> filter_expression,
  std::unordered_set<hybrid_scan_filter_type> const& filters,
  bool verbose,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  static_assert(single_step_read or use_page_index,
                "Page index is needed for two-step parquet read");

  CUDF_FUNC_RANGE();

  auto const has_filter_expr = filter_expression.has_value();
  if (not single_step_read and not has_filter_expr) {
    throw std::runtime_error("Filter expression must be provided for two-step hybrid scan");
  }

  auto options = cudf::io::parquet_reader_options::builder().build();
  if (has_filter_expr) { options.set_filter(filter_expression.value()); }

  // Input file buffer span
  auto datasource     = std::move(cudf::io::make_datasources(io_source.get_source_info()).front());
  auto datasource_ref = std::ref(*datasource);

  // Setup reader
  auto reader           = detail::setup_reader(datasource_ref, options, verbose);
  auto const reader_ref = std::cref(*reader);

  // Setup page index if needed
  if constexpr (use_page_index) {
    detail::setup_page_index(datasource_ref, reader_ref, single_step_read, verbose);
  }

  // Start with all row groups
  auto row_group_indices         = reader->all_row_groups(options);
  auto current_row_group_indices = cudf::host_span<cudf::size_type>(row_group_indices);

  // Filter row groups
  if (has_filter_expr) {
    row_group_indices = detail::apply_row_group_filters(
      datasource_ref, reader_ref, filters, current_row_group_indices, options, verbose, stream);
    current_row_group_indices = cudf::host_span<cudf::size_type>(row_group_indices);
  }

  // Materialize filter and payload columns separately
  if constexpr (single_step_read) {
    return detail::single_step_materialize(
      datasource_ref, reader_ref, current_row_group_indices, options, verbose, stream, mr);
  } else {
    return detail::two_step_materialize(
      datasource_ref, reader_ref, filters, current_row_group_indices, options, verbose, stream, mr);
  }
}

/**
 * @brief Helper to set up multifile hybrid scan tasks
 *
 * @tparam Functor Type of the task functor to execute in each thread.
 *                 Must have an operator()(int tid) method.
 *
 * @param num_threads Number of threads to launch
 * @param hybrid_scan_fn Functor instance to execute in each thread with different tid values
 */
template <typename Functor>
void inline hybrid_scan_multifile(cudf::size_type num_threads, Functor const& hybrid_scan_fn)
{
  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  // Create and launch threads
  std::for_each(thrust::counting_iterator(0),
                thrust::counting_iterator(num_threads),
                [&](auto tid) { threads.emplace_back(hybrid_scan_fn, tid); });

  // Wait for all threads to complete
  for (auto& t : threads) {
    t.join();
  }
}
