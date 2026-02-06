/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "tests/io/parquet_common.hpp"

#include <cudf_test/column_wrapper.hpp>

#include <cudf/io/parquet.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/traits.hpp>

/**
 * @brief Creates a strings column with a constant stringified value between 0 and 9999
 *
 * @param value String value between 0 and 9999
 * @return Strings column wrapper
 */
cudf::test::strings_column_wrapper constant_strings(cudf::size_type value);

/**
 * @brief Fail for types other than duration or timestamp
 */
template <typename T, CUDF_ENABLE_IF(not cudf::is_chrono<T>())>
cudf::test::fixed_width_column_wrapper<T> descending_low_cardinality()
{
  static_assert(
    cudf::is_chrono<T>(),
    "Use testdata::descending<T>() to generate descending values for non-temporal types");
}

/**
 * @brief Creates a duration column wrapper with low cardinality descending values
 *
 * @tparam T Duration type
 * @return Column wrapper
 */
template <typename T, CUDF_ENABLE_IF(cudf::is_duration<T>())>
cudf::test::fixed_width_column_wrapper<T> descending_low_cardinality()
{
  auto elements = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return T((num_ordered_rows - i) / 100); });
  return cudf::test::fixed_width_column_wrapper<T>(elements, elements + num_ordered_rows);
}

/**
 * @brief Creates a timestamp column wrapper with low cardinality descending values
 *
 * @tparam T Timestamp type
 * @return Column wrapper
 */
template <typename T, CUDF_ENABLE_IF(cudf::is_timestamp<T>())>
cudf::test::fixed_width_column_wrapper<T> descending_low_cardinality()
{
  auto elements = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return T(typename T::duration((num_ordered_rows - i) / 100)); });
  return cudf::test::fixed_width_column_wrapper<T>(elements, elements + num_ordered_rows);
}

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
 * @param stream CUDA stream
 *
 * @return Tuple of table and Parquet host buffer
 */
template <typename T,
          size_t NumTableConcats,
          bool IsConstantStrings = true,
          bool IsNullable        = false>
auto create_parquet_with_stats(
  cudf::size_type str_col_value          = 100,
  cudf::io::compression_type compression = cudf::io::compression_type::AUTO,
  rmm::cuda_stream_view stream           = cudf::get_default_stream())
{
  static_assert(NumTableConcats >= 1, "Concatenated table must contain at least one table");

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

  auto table = cudf::concatenate(std::vector<table_view>(NumTableConcats, output));
  output     = table->view();
  cudf::io::table_input_metadata output_metadata(output);
  output_metadata.column_metadata[0].set_name("col0");
  output_metadata.column_metadata[1].set_name("col1");
  output_metadata.column_metadata[2].set_name("col2");

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