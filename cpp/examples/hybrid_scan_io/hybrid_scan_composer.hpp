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
std::unique_ptr<cudf::table> hybrid_scan(
  io_source const& io_source,
  std::optional<cudf::ast::operation const> filter_expression,
  std::unordered_set<hybrid_scan_filter_type> const& filters,
  bool verbose,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

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
