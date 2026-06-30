/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan_multifile.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <functional>
#include <memory>
#include <random>
#include <utility>
#include <vector>

/**
 * @brief Helper to construct a random list<str> column
 *
 * @param gen Random engine
 * @param is_str_nullable Whether the string column should be nullable
 * @param is_list_nullable Whether the list column should be nullable
 *
 * @return Unique pointer to the constructed list<str> column
 */
[[nodiscard]] std::unique_ptr<cudf::column> make_list_str_column(std::mt19937& gen,
                                                                 bool is_str_nullable,
                                                                 bool is_list_nullable);

/**
 * @brief Struct to hold multifile datasources and footer buffers along with their byte spans
 */
struct multifile_inputs {
  /**
   * @brief Construct datasources, datasource refs, and footer byte spans from source info
   */
  explicit multifile_inputs(cudf::io::source_info const& source_info);

  std::vector<std::unique_ptr<cudf::io::datasource>> datasources;
  std::vector<std::reference_wrapper<cudf::io::datasource>> datasource_refs;
  std::vector<std::unique_ptr<cudf::io::datasource::buffer>> footer_buffers;
  std::vector<cudf::host_span<uint8_t const>> footer_byte_spans;
};

/**
 * @brief Device buffers and spans from multiple input sources
 */
struct multisource_device_data {
  std::vector<rmm::device_buffer> buffers;
  std::vector<std::vector<cudf::device_span<uint8_t const>>> per_source_spans;
  std::vector<cudf::device_span<uint8_t const>> flat_spans;
};

/**
 * @brief Construct source info from host buffers
 */
[[nodiscard]] cudf::io::source_info build_source_info(
  std::vector<std::vector<char>> const& file_buffers);

/**
 * @brief Fetch and set up page indexes for all sources in a multifile reader
 */
void setup_page_indexes(cudf::io::parquet::experimental::hybrid_scan_multifile const& reader,
                        multifile_inputs const& inputs);

/**
 * @brief Groups a flat byte range list by source using the specified source map
 */
[[nodiscard]] std::vector<std::vector<cudf::io::text::byte_range_info>> group_byte_ranges_by_source(
  std::pair<std::vector<cudf::io::text::byte_range_info>, std::vector<cudf::size_type>> const&
    byte_ranges_and_source_map,
  std::size_t num_sources);

/**
 * @brief Fetches byte ranges from multiple sources and returns per-source and flattened spans
 */
[[nodiscard]] multisource_device_data fetch_multisource_device_data(
  multifile_inputs const& inputs,
  std::pair<std::vector<cudf::io::text::byte_range_info>, std::vector<cudf::size_type>> const&
    byte_ranges_and_source_map,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

/**
 * @brief Concatenate a vector of tables and return the resultant table
 *
 * @param tables Vector of tables to concatenate
 * @param stream CUDA stream to use
 * @param mr Device memory resource used to allocate the returned table's device memory
 *
 * @return Unique pointer to the resultant concatenated table
 */
[[nodiscard]] std::unique_ptr<cudf::table> concatenate_tables(
  std::vector<std::unique_ptr<cudf::table>>&& tables,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

/**
 * @brief Creates a table and writes it to Parquet host buffer with column level statistics
 *
 * This function creates a table with three columns:
 * - col0: ascending T values
 * - col1: descending T values (reduced cardinality for timestamps and durations)
 * - col2: constant cudf::string_view values
 *
 * The function creates a table by concatenating the same set of columns NumTableConcats times.
 * It then writes this table to a Parquet host buffer with column level statistics.
 *
 * @tparam T Data type for columns 0 and 1
 * @tparam NumTableConcats Number of times to concatenate the base table (must be >= 1)
 * @tparam IsConstantStrings Whether to use constant strings for column 2
 * @tparam IsNullable Whether to create nullable columns
 *
 * @param str_col_value Value for the constant string column used when IsConstantStrings is true
 * @param compression Compression type
 * @param column_names Top-level column names assigned in `column_order` order (default
 *        {"col0", "col1", "col2"})
 * @param column_order Physical emit order of the base [col0, col1, col2] columns (default
 *        {0, 1, 2}). Reordering emits the same logical columns at different schema positions, which
 *        is used to build mismatched per-source schemas for the row-group filtering tests.
 * @param stream CUDA stream
 *
 * @return Tuple of table and Parquet host buffer
 */
template <typename T,
          size_t NumTableConcats,
          bool IsConstantStrings = true,
          bool IsNullable        = false>
[[nodiscard]] std::pair<std::unique_ptr<cudf::table>, std::vector<char>> create_parquet_with_stats(
  cudf::size_type str_col_value             = 100,
  cudf::io::compression_type compression    = cudf::io::compression_type::AUTO,
  std::vector<std::string> column_names     = {"col0", "col1", "col2"},
  std::vector<cudf::size_type> column_order = {0, 1, 2},
  rmm::cuda_stream_view stream              = cudf::get_default_stream())
{
  static_assert(NumTableConcats >= 1, "Concatenated table must contain at least one table");
  CUDF_EXPECTS(column_names.size() == column_order.size(),
               "Column names and column order must have the same size");
  CUDF_EXPECTS(column_order.size() == 3, "Column order must include all three test columns");
  CUDF_EXPECTS(std::all_of(cuda::counting_iterator<cudf::size_type>{0},
                           cuda::counting_iterator<cudf::size_type>{3},
                           [&](auto const col_idx) {
                             return std::count(column_order.begin(), column_order.end(), col_idx) ==
                                    1;
                           }),
               "Column order must be a permutation of the three test columns");

  auto col0 = testdata::ascending<T>();
  auto col1 = []() {
    if constexpr (cudf::is_chrono<T>()) {
      return descending_low_cardinality<T>();
    } else {
      return testdata::descending<T>();
    }
  }();

  auto col2 = [&]() {
    if constexpr (IsConstantStrings) {
      return constant_strings(str_col_value);  // constant stringified value
    } else {
      return testdata::ascending<cudf::string_view>();  // ascending strings
    }
  }();

  // Output table view
  auto output = table_view{{col0, col1, col2}};

  // Add nullmasks to the columns if specified
  std::vector<std::unique_ptr<cudf::column>> columns;
  if constexpr (IsNullable) {
    std::mt19937 gen(0xc0ffee);
    std::bernoulli_distribution bn(0.7f);
    auto valids =
      cudf::detail::make_counting_transform_iterator(0, [&](int index) { return bn(gen); });
    auto const num_rows = static_cast<cudf::column_view>(col0).size();

    columns.emplace_back(col0.release());
    auto [nullmask, nullcount] = cudf::test::detail::make_null_mask(valids, valids + num_rows);
    columns.back()->set_null_mask(std::move(nullmask), nullcount);

    columns.emplace_back(col1.release());
    std::tie(nullmask, nullcount) =
      cudf::test::detail::make_null_mask(valids + num_rows, valids + 2 * num_rows);
    columns.back()->set_null_mask(std::move(nullmask), nullcount);

    columns.emplace_back(col2.release());
    std::tie(nullmask, nullcount) =
      cudf::test::detail::make_null_mask(valids + 2 * num_rows, valids + 3 * num_rows);
    columns.back()->set_null_mask(std::move(nullmask), nullcount);

    // Purge non-empty nulls from the strings column only
    cudf::purge_nonempty_nulls(columns.back()->view());

    // Update the output table view with the nullable columns
    output = table_view{{columns[0]->view(), columns[1]->view(), columns[2]->view()}};
  }

  // Reorder the base [col0, col1, col2] columns into the requested physical order, naming them in
  // that new order.
  std::vector<cudf::column_view> reordered_columns;
  reordered_columns.reserve(column_order.size());
  for (auto const col_idx : column_order) {
    reordered_columns.emplace_back(output.column(col_idx));
  }
  output = table_view{reordered_columns};

  auto table = cudf::concatenate(std::vector<table_view>(NumTableConcats, output));
  output     = table->view();
  cudf::io::table_input_metadata output_metadata(output);
  for (std::size_t i = 0; i < column_names.size(); ++i) {
    output_metadata.column_metadata[i].set_name(column_names[i]);
  }

  std::vector<char> buffer;
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{&buffer}, output)
      .metadata(std::move(output_metadata))
      .row_group_size_rows(page_size_for_ordered_tests)
      .max_page_size_rows(page_size_for_ordered_tests / 5)
      .compression(compression)
      .dictionary_policy(cudf::io::dictionary_policy::ALWAYS)
      .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN);

  if constexpr (NumTableConcats > 1) {
    out_opts.set_row_group_size_rows(num_ordered_rows);
    out_opts.set_max_page_size_rows(page_size_for_ordered_tests);
  }

  cudf::io::write_parquet(out_opts);

  return std::pair{std::move(table), std::move(buffer)};
}
