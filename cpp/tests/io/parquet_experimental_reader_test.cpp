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

#include "tests/io/parquet_common.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/io_metadata_utilities.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf/column/column.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <gtest/gtest.h>
#include <src/io/parquet/parquet_gpu.hpp>

// Base test fixture for tests
struct ParquetExperimentalReaderTest : public cudf::test::BaseFixture {};

namespace {

/**
 * @brief Fetches a host span of Parquet footer bytes from the input buffer span
 *
 * @param buffer Input buffer span
 * @return A host span of the footer bytes
 */
cudf::host_span<uint8_t const> fetch_footer_bytes(cudf::host_span<uint8_t const> buffer)
{
  using namespace cudf::io::parquet;

  constexpr auto header_len = sizeof(file_header_s);
  constexpr auto ender_len  = sizeof(file_ender_s);
  size_t const len          = buffer.size();

  auto const header_buffer = cudf::host_span<uint8_t const>(buffer.data(), header_len);
  auto const header        = reinterpret_cast<file_header_s const*>(header_buffer.data());
  auto const ender_buffer =
    cudf::host_span<uint8_t const>(buffer.data() + len - ender_len, ender_len);
  auto const ender = reinterpret_cast<file_ender_s const*>(ender_buffer.data());
  CUDF_EXPECTS(len > header_len + ender_len, "Incorrect data source");
  constexpr uint32_t parquet_magic = (('P' << 0) | ('A' << 8) | ('R' << 16) | ('1' << 24));
  CUDF_EXPECTS(header->magic == parquet_magic && ender->magic == parquet_magic,
               "Corrupted header or footer");
  CUDF_EXPECTS(ender->footer_len != 0 && ender->footer_len <= (len - header_len - ender_len),
               "Incorrect footer length");

  return cudf::host_span<uint8_t const>(buffer.data() + len - ender->footer_len - ender_len,
                                        ender->footer_len);
}

/**
 * @brief Fetches a host span of Parquet PageIndexbytes from the input buffer span
 *
 * @param buffer Input buffer span
 * @param page_index_bytes Byte range of `PageIndex` to fetch
 * @return A host span of the PageIndex bytes
 */
cudf::host_span<uint8_t const> fetch_page_index_bytes(
  cudf::host_span<uint8_t const> buffer, cudf::io::text::byte_range_info const page_index_bytes)
{
  return cudf::host_span<uint8_t const>(
    reinterpret_cast<uint8_t const*>(buffer.data()) + page_index_bytes.offset(),
    page_index_bytes.size());
}

/**
 * @brief Creates a table and writes it to Parquet host buffer with column level statistics
 *
 * This function creates a table with three columns:
 * - col_uint32: ascending uint32_t values
 * - col_int64: descending int64_t values
 * - col_str: ascending string values
 *
 * The function creates a table by concatenating the same set of columns NumTableConcats times.
 * It then writes this table to a Parquet host buffer with column level statistics.
 *
 * @tparam NumTableConcats Number of times to concatenate the base table (must be >= 1)
 * @return Tuple of table and Parquet host buffer
 */
template <size_t NumTableConcats>
auto create_parquet_with_stats(
  cudf::io::compression_type compression = cudf::io::compression_type::AUTO)
{
  static_assert(NumTableConcats >= 1, "Concatenated table must contain at least one table");

  auto col0 = testdata::ascending<uint32_t>();
  auto col1 = testdata::descending<int64_t>();
  auto col2 = testdata::ascending<cudf::string_view>();

  auto expected = table_view{{col0, col1, col2}};
  auto table    = cudf::concatenate(std::vector<table_view>(NumTableConcats, expected));
  expected      = table->view();

  cudf::io::table_input_metadata expected_metadata(expected);
  expected_metadata.column_metadata[0].set_name("col_uint32");
  expected_metadata.column_metadata[1].set_name("col_int64");
  expected_metadata.column_metadata[2].set_name("col_str");

  std::vector<char> buffer;
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{&buffer}, expected)
      .metadata(std::move(expected_metadata))
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

}  // namespace

TEST_F(ParquetExperimentalReaderTest, TestMetadata)
{
  // Create a table with several row groups each with a single page.
  auto constexpr num_concat         = 1;
  auto constexpr rows_per_row_group = page_size_for_ordered_tests;
  auto file_buffer                  = std::get<1>(create_parquet_with_stats<num_concat>());

  // Filtering AST - table[0] < 100
  auto literal_value     = cudf::numeric_scalar<uint32_t>(100);
  auto literal           = cudf::ast::literal(literal_value);
  auto col_ref_0         = cudf::ast::column_name_reference("col_uint32");
  auto filter_expression = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, literal);

  // Create reader options with empty source info
  cudf::io::parquet_reader_options options =
    cudf::io::parquet_reader_options::builder().filter(filter_expression);

  // Input file buffer span
  auto const file_buffer_span = cudf::host_span<uint8_t const>(
    reinterpret_cast<uint8_t const*>(file_buffer.data()), file_buffer.size());

  // Fetch footer and page index bytes from the buffer.
  auto const footer_buffer = fetch_footer_bytes(file_buffer_span);

  // Create hybrid scan reader with footer bytes
  auto const reader =
    std::make_unique<cudf::io::parquet::experimental::hybrid_scan_reader>(footer_buffer, options);

  // Get Parquet file metadata from the reader
  auto parquet_metadata = reader->parquet_metadata();

  // Check that the offset and column indices are not present
  ASSERT_FALSE(parquet_metadata.row_groups[0].columns[0].offset_index.has_value());
  ASSERT_FALSE(parquet_metadata.row_groups[0].columns[0].column_index.has_value());

  // Get page index byte range from the reader
  auto const page_index_byte_range = reader->page_index_byte_range();

  // Fetch page index bytes from the input buffer
  auto const page_index_buffer = fetch_page_index_bytes(file_buffer_span, page_index_byte_range);

  // Setup page index
  reader->setup_page_index(page_index_buffer);

  // Get Parquet file metadata from the reader again
  parquet_metadata = reader->parquet_metadata();

  // Check that the offset and column indices are now present
  ASSERT_TRUE(parquet_metadata.row_groups[0].columns[0].offset_index.has_value());
  ASSERT_TRUE(parquet_metadata.row_groups[0].columns[0].column_index.has_value());

  // Get all row groups from the reader
  auto input_row_group_indices = reader->all_row_groups(options);
  // Expect 4 = 20000 rows / 5000 rows per row group
  EXPECT_EQ(input_row_group_indices.size(), 4);

  // Explicitly set the row groups to read
  options.set_row_groups({{0, 1}});

  // Get all row groups from the reader again
  input_row_group_indices = reader->all_row_groups(options);
  // Expect only 2 row groups now
  EXPECT_EQ(input_row_group_indices.size(), 2);
  EXPECT_EQ(reader->total_rows_in_row_groups(input_row_group_indices), 2 * rows_per_row_group);
}

TEST_F(ParquetExperimentalReaderTest, TestFilterRowGroupWithStats)
{
  // Create a table with 4 row groups each with a single page.
  auto constexpr num_concat         = 1;
  auto constexpr rows_per_row_group = page_size_for_ordered_tests;
  auto [written_table, file_buffer] = create_parquet_with_stats<num_concat>();

  // Filtering AST - table[0] < 50
  auto literal_value     = cudf::numeric_scalar<uint32_t>(50);
  auto literal           = cudf::ast::literal(literal_value);
  auto col_ref_0         = cudf::ast::column_name_reference("col_uint32");
  auto filter_expression = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, literal);

  // Create reader options with empty source info
  cudf::io::parquet_reader_options options =
    cudf::io::parquet_reader_options::builder().filter(filter_expression);

  // Input file buffer span
  auto const file_buffer_span = cudf::host_span<uint8_t const>(
    reinterpret_cast<uint8_t const*>(file_buffer.data()), file_buffer.size());

  // Fetch footer and page index bytes from the buffer.
  auto const footer_buffer = fetch_footer_bytes(file_buffer_span);

  // Create hybrid scan reader with footer bytes
  auto const reader =
    std::make_unique<cudf::io::parquet::experimental::hybrid_scan_reader>(footer_buffer, options);

  // Get all row groups from the reader
  auto input_row_group_indices = reader->all_row_groups(options);
  // Expect 4 = 20000 rows / 5000 rows per row group
  EXPECT_EQ(input_row_group_indices.size(), 4);
  EXPECT_EQ(reader->total_rows_in_row_groups(input_row_group_indices), 4 * rows_per_row_group);

  auto stats_filtered_row_groups = reader->filter_row_groups_with_stats(
    input_row_group_indices, options, cudf::get_default_stream());
  // Expect 3 row groups to be filtered out with stats
  EXPECT_EQ(stats_filtered_row_groups.size(), 1);
  EXPECT_EQ(reader->total_rows_in_row_groups(stats_filtered_row_groups), rows_per_row_group);

  // Use custom input row group indices
  input_row_group_indices   = {1, 2};
  stats_filtered_row_groups = reader->filter_row_groups_with_stats(
    input_row_group_indices, options, cudf::get_default_stream());
  // Expect all row groups to be filtered out with stats
  EXPECT_EQ(stats_filtered_row_groups.size(), 0);
  EXPECT_EQ(reader->total_rows_in_row_groups(stats_filtered_row_groups), 0);
}

TEST_F(ParquetExperimentalReaderTest, TestFilterPagesWithPageIndexStats)
{
  srand(31337);

  // A table concatenated multiple times by itself with result in a parquet file with a row group
  // per concatenation with multiple pages per row group. Since all row groups will be identical, we
  // can only prune pages based on `PageIndex` stats
  auto constexpr num_concat = 2;
  auto const file_buffer    = std::get<1>(create_parquet_with_stats<num_concat>());

  // Input file buffer span
  auto const file_buffer_span = cudf::host_span<uint8_t const>(
    reinterpret_cast<uint8_t const*>(file_buffer.data()), file_buffer.size());

  // Filtering AST - table[0] < 100
  auto literal_value     = cudf::numeric_scalar<uint32_t>(100);
  auto literal           = cudf::ast::literal(literal_value);
  auto col_ref_0         = cudf::ast::column_name_reference("col_uint32");
  auto filter_expression = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, literal);

  // Create reader options with empty source info
  cudf::io::parquet_reader_options options =
    cudf::io::parquet_reader_options::builder().filter(filter_expression);

  // Fetch footer and page index bytes from the buffer.
  auto const footer_buffer = fetch_footer_bytes(file_buffer_span);

  // Create hybrid scan reader with footer bytes
  auto const reader =
    std::make_unique<cudf::io::parquet::experimental::hybrid_scan_reader>(footer_buffer, options);

  // Get all row groups from the reader
  auto input_row_group_indices = reader->all_row_groups(options);

  // Span to track current row group indices
  auto current_row_group_indices = cudf::host_span<cudf::size_type>(input_row_group_indices);

  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();

  // Calling `filter_data_pages_with_stats` before setting up the page index should raise an error
  EXPECT_THROW(std::ignore = reader->filter_data_pages_with_stats(
                 current_row_group_indices, options, stream, mr),
               std::runtime_error);

  // Set up the page index
  auto const page_index_byte_range = reader->page_index_byte_range();
  auto const page_index_buffer = fetch_page_index_bytes(file_buffer_span, page_index_byte_range);
  reader->setup_page_index(page_index_buffer);

  // Filter the data pages with page index stats
  auto [row_mask, data_page_mask] =
    reader->filter_data_pages_with_stats(current_row_group_indices, options, stream, mr);

  // Checks
  auto constexpr num_filter_columns = 1;
  EXPECT_EQ(data_page_mask.size(), num_filter_columns);

  auto const expected_num_rows = reader->total_rows_in_row_groups(current_row_group_indices);
  EXPECT_EQ(row_mask->type().id(), cudf::type_id::BOOL8);
  EXPECT_EQ(row_mask->size(), expected_num_rows);
  EXPECT_EQ(row_mask->null_count(), 0);

  // Half the pages should survive the page index filter
  auto constexpr expected_num_pages_after_page_index_filter =
    num_concat * (num_ordered_rows / page_size_for_ordered_tests) / 2;
  // Count the number of pages that survive the page index filter
  auto const num_pages_after_page_index_filter =
    std::accumulate(data_page_mask.begin(),
                    data_page_mask.end(),
                    cudf::size_type{0},
                    [](auto sum, auto const& page_mask) {
                      return sum + std::count(page_mask.cbegin(), page_mask.cend(), true);
                    });
  EXPECT_EQ(num_pages_after_page_index_filter, expected_num_pages_after_page_index_filter);
}