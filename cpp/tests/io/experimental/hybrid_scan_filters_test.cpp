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

#include "hybrid_scan_common.hpp"

#include <cudf_test/base_fixture.hpp>

#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <src/io/parquet/parquet_gpu.hpp>

namespace {

/**
 * @brief Filter input row groups using column chunk dictionaries via the experimental parquet
 * reader for hybrid scan
 *
 * @param file_buffer_span Input file buffer span
 * @param filter_expression Filter expression
 * @param stream CUDA stream
 * @param mr Device memory resource
 *
 * @return Vector of dictionary-filtered row group indices
 */
auto filter_row_groups_with_dictionaries(cudf::host_span<uint8_t const> file_buffer_span,
                                         cudf::ast::operation const& filter_expression,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  // Create reader options with empty source info
  cudf::io::parquet_reader_options options =
    cudf::io::parquet_reader_options::builder().filter(filter_expression);

  // Fetch footer and page index bytes from the buffer.
  auto const footer_buffer = fetch_footer_bytes(file_buffer_span);

  // Create hybrid scan reader with footer bytes
  auto const reader =
    std::make_unique<cudf::io::parquet::experimental::hybrid_scan_reader>(footer_buffer, options);

  // Get page index byte range from the reader
  auto const page_index_byte_range = reader->page_index_byte_range();

  // Fetch page index bytes from the input buffer
  auto const page_index_buffer = fetch_page_index_bytes(file_buffer_span, page_index_byte_range);

  // Setup page index
  reader->setup_page_index(page_index_buffer);

  // Get all row groups from the reader
  auto input_row_group_indices = reader->all_row_groups(options);

  // Span to track current row group indices
  auto current_row_group_indices = cudf::host_span<cudf::size_type>(input_row_group_indices);

  // Get dictionary page byte ranges from the reader
  auto const dict_page_byte_ranges =
    std::get<1>(reader->secondary_filters_byte_ranges(current_row_group_indices, options));

  // If we have dictionary page byte ranges, filter row groups with dictionary pages
  std::vector<cudf::size_type> dictionary_page_filtered_row_group_indices;
  dictionary_page_filtered_row_group_indices.reserve(current_row_group_indices.size());

  CUDF_EXPECTS(dict_page_byte_ranges.size() > 0, "No dictionary page byte ranges found");

  // Fetch dictionary page buffers from the input file buffer
  std::vector<rmm::device_buffer> dictionary_page_buffers =
    fetch_byte_ranges(file_buffer_span, dict_page_byte_ranges, stream, mr);

  dictionary_page_filtered_row_group_indices = reader->filter_row_groups_with_dictionary_pages(
    dictionary_page_buffers, current_row_group_indices, options, stream);

  return dictionary_page_filtered_row_group_indices;
}

}  // namespace

// Base test fixture for tests
struct HybridScanFiltersTest : public cudf::test::BaseFixture {};

TEST_F(HybridScanFiltersTest, TestMetadata)
{
  srand(0xf00d);
  using T = uint32_t;

  // Create a table with several row groups each with a single page.
  auto constexpr num_concat         = 1;
  auto constexpr rows_per_row_group = page_size_for_ordered_tests;
  auto file_buffer                  = std::get<1>(create_parquet_with_stats<T, num_concat>());

  // Filtering AST - table[0] < 100
  auto literal_value     = cudf::numeric_scalar<T>(100);
  auto literal           = cudf::ast::literal(literal_value);
  auto col_ref_0         = cudf::ast::column_name_reference("col0");
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

TEST_F(HybridScanFiltersTest, FilterRowGroupsWithStats)
{
  srand(0xc001);
  using T = uint32_t;

  // Create a table with 4 row groups each with a single page.
  auto constexpr num_concat         = 1;
  auto constexpr rows_per_row_group = page_size_for_ordered_tests;
  auto [written_table, file_buffer] = create_parquet_with_stats<T, num_concat, false>();

  // Filtering AST - table[0] < 50
  auto literal_value     = cudf::numeric_scalar<T>(50);
  auto literal           = cudf::ast::literal(literal_value);
  auto col_ref_0         = cudf::ast::column_name_reference("col0");
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

template <typename T>
struct PageFilteringWithPageIndexStats : public HybridScanFiltersTest {};

// Unsigned numeric types except booleans for columns 0 and 1 for page index stats tests
using SignedIntegralTypesNotBool =
  cudf::test::ContainedIn<cudf::test::Types<int8_t, int16_t, int32_t, int64_t>>;
using PageFilteringTestTypes =
  cudf::test::RemoveIf<SignedIntegralTypesNotBool,
                       cudf::test::Concat<cudf::test::IntegralTypesNotBool>>;

TYPED_TEST_SUITE(PageFilteringWithPageIndexStats, PageFilteringTestTypes);

TYPED_TEST(PageFilteringWithPageIndexStats, FilterPagesWithPageIndexStats)
{
  using T = TypeParam;

  srand(31337);

  // A table concatenated multiple times by itself with result in a parquet file with a row group
  // per concatenation with multiple pages per row group. Since all row groups will be identical, we
  // can only prune pages based on `PageIndex` stats
  auto constexpr num_concat = 2;
  auto const file_buffer    = std::get<1>(create_parquet_with_stats<T, num_concat, false>());

  // Input file buffer span
  auto const file_buffer_span = cudf::host_span<uint8_t const>(
    reinterpret_cast<uint8_t const*>(file_buffer.data()), file_buffer.size());

  // Helper function to test data page filteration using page index stats
  auto const test_filter_data_pages_with_stats = [&](cudf::ast::operation const& filter_expression,
                                                     cudf::size_type const expected_surviving_rows,
                                                     rmm::cuda_stream_view stream =
                                                       cudf::get_default_stream(),
                                                     rmm::device_async_resource_ref mr =
                                                       cudf::get_current_device_resource_ref()) {
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

    // Calling `filter_data_pages_with_stats` before setting up the page index should raise an
    // error
    EXPECT_THROW(std::ignore = reader->build_row_mask_with_page_index_stats(
                   current_row_group_indices, options, stream, mr),
                 std::runtime_error);

    // Set up the page index
    auto const page_index_byte_range = reader->page_index_byte_range();
    auto const page_index_buffer = fetch_page_index_bytes(file_buffer_span, page_index_byte_range);
    reader->setup_page_index(page_index_buffer);

    // Filter the data pages with page index stats
    auto const row_mask =
      reader->build_row_mask_with_page_index_stats(current_row_group_indices, options, stream, mr);

    auto const expected_num_rows = reader->total_rows_in_row_groups(current_row_group_indices);
    EXPECT_EQ(row_mask->type().id(), cudf::type_id::BOOL8);
    EXPECT_EQ(row_mask->size(), expected_num_rows);
    EXPECT_EQ(row_mask->null_count(), 0);

    // Copy the row mask to the host and count the number of surviving rows
    auto const host_row_mask = cudf::detail::make_host_vector<bool>(
      {row_mask->view().data<bool>(), static_cast<size_t>(row_mask->view().size())}, stream);
    EXPECT_EQ(std::count(host_row_mask.begin(), host_row_mask.end(), true),
              expected_surviving_rows);
  };

  // Filtering AST - table[0] < 100
  {
    auto literal_value     = cudf::numeric_scalar<T>(T{100});
    auto const literal     = cudf::ast::literal(literal_value);
    auto const col_ref     = cudf::ast::column_name_reference("col0");
    auto filter_expression = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref, literal);
    // Half the pages (unsigned) or 3/4th the pages (signed) should be filtered out by the page
    // index filter
    auto constexpr expected_surviving_rows =
      (num_concat * num_ordered_rows) / (std::is_signed_v<T> ? 4 : 2);
    test_filter_data_pages_with_stats(filter_expression, expected_surviving_rows);
  }

  // Filtering AST - table[2] >= 10000
  {
    auto literal_value = cudf::string_scalar("000010000");
    auto literal       = cudf::ast::literal(literal_value);
    auto col_ref       = cudf::ast::column_name_reference("col2");
    auto filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, col_ref, literal);
    // Half the pages (unsigned) or 3/4th the pages (signed) should be filtered out by the page
    // index filter
    auto constexpr expected_surviving_rows =
      (num_concat * num_ordered_rows) / (std::is_signed_v<T> ? 4 : 2);
    test_filter_data_pages_with_stats(filter_expression, expected_surviving_rows);
  }

  // Filtering AST - table[0] < 50 AND table[2] < "000010000"
  {
    auto literal_value1 = cudf::numeric_scalar<T>(T{50});
    auto const literal1 = cudf::ast::literal(literal_value1);
    auto const col_ref1 = cudf::ast::column_name_reference("col0");
    auto filter_expression1 =
      cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref1, literal1);

    auto literal_value2 = cudf::string_scalar("000010000");
    auto literal2       = cudf::ast::literal(literal_value2);
    auto col_ref2       = cudf::ast::column_name_reference("col2");
    auto filter_expression2 =
      cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref2, literal2);

    auto filter_expression = cudf::ast::operation(
      cudf::ast::ast_operator::LOGICAL_AND, filter_expression1, filter_expression2);
    // Only one page per num_concat per filter column should survive
    auto constexpr expected_surviving_rows = num_concat * page_size_for_ordered_tests;
    test_filter_data_pages_with_stats(filter_expression, expected_surviving_rows);
  }

  // Filtering AST - table[0] > 150 OR table[2] < "000005000"
  {
    auto literal_value1 = cudf::numeric_scalar<T>(T{150});
    auto const literal1 = cudf::ast::literal(literal_value1);
    auto const col_ref1 = cudf::ast::column_name_reference("col0");
    auto filter_expression1 =
      cudf::ast::operation(cudf::ast::ast_operator::GREATER, col_ref1, literal1);

    auto literal_value2 = cudf::string_scalar("000005000");
    auto literal2       = cudf::ast::literal(literal_value2);
    auto col_ref2       = cudf::ast::column_name_reference("col2");
    auto filter_expression2 =
      cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref2, literal2);

    auto filter_expression = cudf::ast::operation(
      cudf::ast::ast_operator::LOGICAL_OR, filter_expression1, filter_expression2);
    // Two pages (3rd and 0th from respective conditions) per num_concat per filter column should
    // survive
    auto constexpr expected_surviving_rows = 2 * num_concat * page_size_for_ordered_tests;
    test_filter_data_pages_with_stats(filter_expression, expected_surviving_rows);
  }
}

TEST_F(HybridScanFiltersTest, FilterRowGroupsWithDictBasic)
{
  srand(0xcafe);
  using T = uint32_t;

  // A table not concated with itself with result in a parquet file with several row groups each
  // with a single page. Since there is only one page per row group, the page and row group stats
  // are identical and we can only prune row groups.
  auto constexpr num_concat = 1;
  auto const buffer         = std::get<1>(create_parquet_with_stats<T, num_concat>());
  auto stream               = cudf::get_default_stream();
  auto mr                   = cudf::get_current_device_resource_ref();

  // Input file buffer span
  auto const file_buffer_span =
    cudf::host_span<uint8_t const>(reinterpret_cast<uint8_t const*>(buffer.data()), buffer.size());

  {
    // Filtering - table[0] != 1000
    auto uint_literal_value = cudf::numeric_scalar<T>(1000);
    auto uint_literal       = cudf::ast::literal(uint_literal_value);
    auto uint_col_ref       = cudf::ast::column_name_reference("col0");
    auto filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, uint_col_ref, uint_literal);
    constexpr size_t expected_row_groups = 4;
    EXPECT_EQ(
      filter_row_groups_with_dictionaries(file_buffer_span, filter_expression, stream, mr).size(),
      expected_row_groups);
  }

  {
    // Filtering - table[0] == 1000
    auto uint_literal_value = cudf::numeric_scalar<T>(1000);
    auto uint_literal       = cudf::ast::literal(uint_literal_value);
    auto uint_col_ref       = cudf::ast::column_name_reference("col0");
    auto filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::EQUAL, uint_col_ref, uint_literal);
    constexpr size_t expected_row_groups = 0;
    EXPECT_EQ(
      filter_row_groups_with_dictionaries(file_buffer_span, filter_expression, stream, mr).size(),
      expected_row_groups);
  }

  {
    // Filtering - table[2] != 0100
    auto str_literal_value = cudf::string_scalar("0100");  // in all row groups
    auto str_literal       = cudf::ast::literal(str_literal_value);
    auto str_col_ref       = cudf::ast::column_name_reference("col2");
    auto filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, str_col_ref, str_literal);

    constexpr size_t expected_row_groups = 0;
    EXPECT_EQ(
      filter_row_groups_with_dictionaries(file_buffer_span, filter_expression, stream, mr).size(),
      expected_row_groups);
  }

  {
    // Filtering - table[2] == 0100
    auto str_literal_value = cudf::string_scalar("0100");  // in all row groups
    auto str_literal       = cudf::ast::literal(str_literal_value);
    auto str_col_ref       = cudf::ast::column_name_reference("col2");
    auto filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::EQUAL, str_col_ref, str_literal);

    constexpr size_t expected_row_groups = 4;
    EXPECT_EQ(
      filter_row_groups_with_dictionaries(file_buffer_span, filter_expression, stream, mr).size(),
      expected_row_groups);
  }

  {
    // Filtering - table[0] != 50 AND table[2] == 0100
    auto uint_literal_value = cudf::numeric_scalar<T>(50);
    auto uint_literal       = cudf::ast::literal(uint_literal_value);
    auto uint_col_ref       = cudf::ast::column_name_reference("col0");
    auto uint_filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, uint_col_ref, uint_literal);

    auto str_literal_value = cudf::string_scalar("0100");
    auto str_literal       = cudf::ast::literal(str_literal_value);
    auto str_col_ref       = cudf::ast::column_name_reference("col2");
    auto str_filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::EQUAL, str_col_ref, str_literal);
    auto filter_expression = cudf::ast::operation(
      cudf::ast::ast_operator::LOGICAL_AND, uint_filter_expression, str_filter_expression);

    constexpr size_t expected_row_groups = 4;
    EXPECT_EQ(
      filter_row_groups_with_dictionaries(file_buffer_span, filter_expression, stream, mr).size(),
      expected_row_groups);
  }

  {
    // Filtering -  table[0] != 50 and table[0] != 100
    auto uint_literal_value  = cudf::numeric_scalar<T>(50);
    auto uint_literal_value2 = cudf::numeric_scalar<T>(100);
    auto uint_literal        = cudf::ast::literal(uint_literal_value);
    auto uint_literal2       = cudf::ast::literal(uint_literal_value2);
    auto uint_col_ref        = cudf::ast::column_name_reference("col0");
    auto uint_filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, uint_col_ref, uint_literal);
    auto uint_filter_expression2 =
      cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, uint_col_ref, uint_literal2);
    auto filter_expression = cudf::ast::operation(
      cudf::ast::ast_operator::LOGICAL_AND, uint_filter_expression, uint_filter_expression2);

    constexpr size_t expected_row_groups = 4;
    EXPECT_EQ(
      filter_row_groups_with_dictionaries(file_buffer_span, filter_expression, stream, mr).size(),
      expected_row_groups);
  }

  {
    // Filtering - table[0] != 50 and table[0] == 50
    auto uint_literal_value  = cudf::numeric_scalar<T>(50);
    auto uint_literal_value2 = cudf::numeric_scalar<T>(50);
    auto uint_literal        = cudf::ast::literal(uint_literal_value);
    auto uint_literal2       = cudf::ast::literal(uint_literal_value2);
    auto uint_col_ref        = cudf::ast::column_name_reference("col0");
    auto uint_filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, uint_col_ref, uint_literal);
    auto uint_filter_expression2 =
      cudf::ast::operation(cudf::ast::ast_operator::EQUAL, uint_col_ref, uint_literal2);
    auto filter_expression = cudf::ast::operation(
      cudf::ast::ast_operator::LOGICAL_AND, uint_filter_expression, uint_filter_expression2);

    constexpr size_t expected_row_groups = 1;
    EXPECT_EQ(
      filter_row_groups_with_dictionaries(file_buffer_span, filter_expression, stream, mr).size(),
      expected_row_groups);
  }

  {
    // Filtering - table[2] != 0100 or table[2] != 0101
    auto str_literal_value  = cudf::string_scalar("0100");  // in all row groups
    auto str_literal_value2 = cudf::string_scalar("0101");  // in no row group
    auto str_literal        = cudf::ast::literal(str_literal_value);
    auto str_literal2       = cudf::ast::literal(str_literal_value2);
    auto str_col_ref        = cudf::ast::column_name_reference("col2");
    auto str_filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, str_col_ref, str_literal);
    auto str_filter_expression2 =
      cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, str_col_ref, str_literal2);
    auto filter_expression = cudf::ast::operation(
      cudf::ast::ast_operator::LOGICAL_OR, str_filter_expression, str_filter_expression2);

    constexpr size_t expected_row_groups = 4;
    EXPECT_EQ(
      filter_row_groups_with_dictionaries(file_buffer_span, filter_expression, stream, mr).size(),
      expected_row_groups);
  }

  {
    // Filtering - table[0] != 50 or table[2] != 0100
    auto uint_literal_value = cudf::numeric_scalar<T>(50);
    auto uint_literal       = cudf::ast::literal(uint_literal_value);
    auto uint_col_ref       = cudf::ast::column_name_reference("col0");
    auto uint_filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, uint_col_ref, uint_literal);

    auto str_literal_value = cudf::string_scalar("0100");
    auto str_literal       = cudf::ast::literal(str_literal_value);
    auto str_col_ref       = cudf::ast::column_name_reference("col2");
    auto str_filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, str_col_ref, str_literal);
    auto filter_expression = cudf::ast::operation(
      cudf::ast::ast_operator::LOGICAL_OR, uint_filter_expression, str_filter_expression);

    constexpr size_t expected_row_groups = 4;
    EXPECT_EQ(
      filter_row_groups_with_dictionaries(file_buffer_span, filter_expression, stream, mr).size(),
      expected_row_groups);
  }

  {
    // Filtering - table[0] != 50 and table[2] != 0100
    auto uint_literal_value = cudf::numeric_scalar<T>(50);
    auto uint_literal       = cudf::ast::literal(uint_literal_value);
    auto uint_col_ref       = cudf::ast::column_name_reference("col0");
    auto uint_filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, uint_col_ref, uint_literal);

    auto str_literal_value = cudf::string_scalar("0100");
    auto str_literal       = cudf::ast::literal(str_literal_value);
    auto str_col_ref       = cudf::ast::column_name_reference("col2");
    auto str_filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, str_col_ref, str_literal);
    auto filter_expression = cudf::ast::operation(
      cudf::ast::ast_operator::LOGICAL_AND, uint_filter_expression, str_filter_expression);

    constexpr size_t expected_row_groups = 0;
    EXPECT_EQ(
      filter_row_groups_with_dictionaries(file_buffer_span, filter_expression, stream, mr).size(),
      expected_row_groups);
  }

  {
    // Filtering - table[0] == 50 or table[0] == 100 or table[0] == 150
    auto uint_literal_value  = cudf::numeric_scalar<T>(50);
    auto uint_literal_value2 = cudf::numeric_scalar<T>(100);
    auto uint_literal_value3 = cudf::numeric_scalar<T>(150);
    auto uint_literal        = cudf::ast::literal(uint_literal_value);
    auto uint_literal2       = cudf::ast::literal(uint_literal_value2);
    auto uint_literal3       = cudf::ast::literal(uint_literal_value3);
    auto uint_col_ref        = cudf::ast::column_name_reference("col0");
    auto uint_filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::EQUAL, uint_col_ref, uint_literal);
    auto uint_filter_expression2 =
      cudf::ast::operation(cudf::ast::ast_operator::EQUAL, uint_col_ref, uint_literal2);
    auto uint_filter_expression3 =
      cudf::ast::operation(cudf::ast::ast_operator::EQUAL, uint_col_ref, uint_literal3);
    auto composed_filter_expression = cudf::ast::operation(
      cudf::ast::ast_operator::LOGICAL_OR, uint_filter_expression, uint_filter_expression2);
    auto filter_expression = cudf::ast::operation(
      cudf::ast::ast_operator::LOGICAL_OR, composed_filter_expression, uint_filter_expression3);

    constexpr size_t expected_row_groups = 3;
    EXPECT_EQ(
      filter_row_groups_with_dictionaries(file_buffer_span, filter_expression, stream, mr).size(),
      expected_row_groups);
  }

  {
    // Filtering - table[0] != 50 or table[0] != 100 or table[0] != 150
    auto uint_literal_value  = cudf::numeric_scalar<T>(50);
    auto uint_literal_value2 = cudf::numeric_scalar<T>(100);
    auto uint_literal_value3 = cudf::numeric_scalar<T>(150);
    auto uint_literal        = cudf::ast::literal(uint_literal_value);
    auto uint_literal2       = cudf::ast::literal(uint_literal_value2);
    auto uint_literal3       = cudf::ast::literal(uint_literal_value3);
    auto uint_col_ref        = cudf::ast::column_name_reference("col0");
    auto uint_filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, uint_col_ref, uint_literal);
    auto uint_filter_expression2 =
      cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, uint_col_ref, uint_literal2);
    auto uint_filter_expression3 =
      cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, uint_col_ref, uint_literal3);
    auto composed_filter_expression = cudf::ast::operation(
      cudf::ast::ast_operator::LOGICAL_OR, uint_filter_expression, uint_filter_expression2);
    auto filter_expression = cudf::ast::operation(
      cudf::ast::ast_operator::LOGICAL_OR, composed_filter_expression, uint_filter_expression3);

    constexpr size_t expected_row_groups = 4;
    EXPECT_EQ(
      filter_row_groups_with_dictionaries(file_buffer_span, filter_expression, stream, mr).size(),
      expected_row_groups);
  }

  {
    // Filtering - table[2] != 0100 and table[2] != 0101 and table[2] != 0150
    auto str_literal_value  = cudf::string_scalar("0100");
    auto str_literal_value2 = cudf::string_scalar("0101");
    auto str_literal_value3 = cudf::string_scalar("0150");
    auto str_literal        = cudf::ast::literal(str_literal_value);
    auto str_literal2       = cudf::ast::literal(str_literal_value2);
    auto str_literal3       = cudf::ast::literal(str_literal_value3);
    auto str_col_ref        = cudf::ast::column_name_reference("col2");
    auto str_filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, str_col_ref, str_literal);
    auto str_filter_expression2 =
      cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, str_col_ref, str_literal2);
    auto str_filter_expression3 =
      cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, str_col_ref, str_literal3);
    auto composed_filter_expression = cudf::ast::operation(
      cudf::ast::ast_operator::LOGICAL_AND, str_filter_expression, str_filter_expression2);
    auto filter_expression = cudf::ast::operation(
      cudf::ast::ast_operator::LOGICAL_AND, composed_filter_expression, str_filter_expression3);

    constexpr size_t expected_row_groups = 0;
    EXPECT_EQ(
      filter_row_groups_with_dictionaries(file_buffer_span, filter_expression, stream, mr).size(),
      expected_row_groups);
  }
}

template <typename T>
struct RowGroupFilteringWithDictTest : public HybridScanFiltersTest {};

// Booleans are not supported for dictionary based filtering
using DictionaryTestTypes =
  cudf::test::RemoveIf<cudf::test::ContainedIn<cudf::test::Types<bool>>, SupportedTestTypes>;

TYPED_TEST_SUITE(RowGroupFilteringWithDictTest, DictionaryTestTypes);

TYPED_TEST(RowGroupFilteringWithDictTest, FilterFewLiteralsTyped)
{
  srand(0xace);
  using T = TypeParam;

  auto constexpr num_concat = 1;

  // Specifying ZSTD compression to explicitly test decompression of dictionary pages
  auto const buffer =
    std::get<1>(create_parquet_with_stats<T, num_concat>(100, cudf::io::compression_type::ZSTD));

  // For string tests use `col2` containing constant "0100" and for temporal types use `col1`
  // containing low cardinality descending values. For all other types use `col0`
  // containing ascending values.
  auto col_name = [&]() {
    if (cuda::std::is_same_v<T, cudf::string_view>) {
      return cudf::ast::column_name_reference("col2");
    } else if (cudf::is_duration<T>() or cudf::is_timestamp<T>()) {
      return cudf::ast::column_name_reference("col1");
    } else {
      return cudf::ast::column_name_reference("col0");
    }
  }();

  // Same logic as above for column reference
  auto col_ref = [&]() {
    if (cuda::std::is_same_v<T, cudf::string_view>) {
      return cudf::ast::column_reference(2);
    } else if (cudf::is_duration<T>() or cudf::is_timestamp<T>()) {
      return cudf::ast::column_reference(1);
    } else {
      return cudf::ast::column_reference(0);
    }
  }();

  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();

  // Input file buffer span
  auto const file_buffer_span =
    cudf::host_span<uint8_t const>(reinterpret_cast<uint8_t const*>(buffer.data()), buffer.size());

  // Filtering AST
  auto literal_value = []() {
    if constexpr (cudf::is_timestamp<T>()) {
      // table[1] == 100 timestamp d/s/ms/us/ns
      return cudf::timestamp_scalar<T>(T(typename T::duration(100)));  // i (0-200)
    } else if constexpr (cudf::is_duration<T>()) {
      // table[1] == 100 d/s/ms/us/ns
      return cudf::duration_scalar<T>(T(100));  // i (0-200)
    } else if constexpr (std::is_same_v<T, cudf::string_view>) {
      // table[2] == "0100"
      return cudf::string_scalar("0100");  // i (0-200)
    } else {
      // table[0] == 0 or 100u
      return cudf::numeric_scalar<T>((100 - 100 * std::is_signed_v<T>));  // i/100 (-100-100/ 0-200)
    }
  }();

  // Filtering AST - col_ref == 100
  {
    // Expected row group indices after filtering
    auto const expected_row_groups = [&]() {
      if constexpr (cuda::std::is_same_v<T, cudf::string_view>) {
        return std::vector<cudf::size_type>{
          0, 1, 2, 3};  // Constant string value "0100" is present in all RGs
      } else if constexpr (cudf::is_chrono<T>() or cuda::std::is_signed_v<T>) {
        return std::vector<cudf::size_type>{
          1, 2};  // Descending temporal and signed value (100) is present in RGs: 1,2
      } else {
        return std::vector<cudf::size_type>{2};  // Ascending value (100) is present in RG: 1
      }
    }();

    // Build the filter expression
    auto const literal = cudf::ast::literal(literal_value);
    auto const filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::EQUAL, col_name, literal);

    // Check the results
    EXPECT_EQ(filter_row_groups_with_dictionaries(file_buffer_span, filter_expression, stream, mr),
              expected_row_groups);
  }

  // Filtering AST - col_ref != 100
  {
    // Expected row group indices after filtering
    auto const expected_row_groups = [&]() {
      if constexpr (cuda::std::is_same_v<T, cudf::string_view>) {
        return std::vector<cudf::size_type>{};
      } else {
        return std::vector<cudf::size_type>{0, 1, 2, 3};
      }
    }();

    // Build the filter expression
    auto const literal = cudf::ast::literal(literal_value);
    auto const filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, col_name, literal);

    // Check the results
    EXPECT_EQ(filter_row_groups_with_dictionaries(file_buffer_span, filter_expression, stream, mr),
              expected_row_groups);
  }
}

TYPED_TEST(RowGroupFilteringWithDictTest, FilterManyLiteralsTyped)
{
  srand(0xcabab);
  using T = TypeParam;

  auto constexpr num_concat = 1;
  // Specifying no compression to explicitly test uncompressed dictionary pages
  auto const buffer =
    std::get<1>(create_parquet_with_stats<T, num_concat>(100, cudf::io::compression_type::NONE));

  // For string tests use `col2` containing constant "0100" and for temporal types use `col1`
  // containing low cardinality descending values. For all other types use `col0`
  // containing ascending values.
  auto col_name = [&]() {
    if (cuda::std::is_same_v<T, cudf::string_view>) {
      return cudf::ast::column_name_reference("col2");
    } else if (cudf::is_duration<T>() or cudf::is_timestamp<T>()) {
      return cudf::ast::column_name_reference("col1");
    } else {
      return cudf::ast::column_name_reference("col0");
    }
  }();

  // Same logic as above for column reference
  auto col_ref = [&]() {
    if (cuda::std::is_same_v<T, cudf::string_view>) {
      return cudf::ast::column_reference(2);
    } else if (cudf::is_duration<T>() or cudf::is_timestamp<T>()) {
      return cudf::ast::column_reference(1);
    } else {
      return cudf::ast::column_reference(0);
    }
  }();

  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();

  // Input file buffer span
  auto const file_buffer_span =
    cudf::host_span<uint8_t const>(reinterpret_cast<uint8_t const*>(buffer.data()), buffer.size());

  // First literal value
  auto literal_value1 = []() {
    if constexpr (cudf::is_timestamp<T>()) {
      // table[1] == 100 timestamp d/s/ms/us/ns
      return cudf::timestamp_scalar<T>(T(typename T::duration(100)));  // i (0-200)
    } else if constexpr (cudf::is_duration<T>()) {
      // table[1] == 100 d/s/ms/us/ns
      return cudf::duration_scalar<T>(T(100));  // i (0-200)
    } else if constexpr (std::is_same_v<T, cudf::string_view>) {
      // table[2] == "0100"
      return cudf::string_scalar("0100");  // i (0-200)
    } else {
      // table[0] == -100 or 100u
      return cudf::numeric_scalar<T>((100 - 200 * std::is_signed_v<T>));  // i/100 (-100-100/ 0-200)
    }
  }();

  // Second literal value
  auto literal_value2 = []() {
    if constexpr (cudf::is_timestamp<T>()) {
      // table[1] == 50 timestamp d/s/ms/us/ns
      return cudf::timestamp_scalar<T>(T(typename T::duration(50)));  // i (0-200)
    } else if constexpr (cudf::is_duration<T>()) {
      // table[1] == 50 d/s/ms/us/ns
      return cudf::duration_scalar<T>(T(50));  // i (0-200)
    } else if constexpr (std::is_same_v<T, cudf::string_view>) {
      // table[2] == "0050"
      return cudf::string_scalar("0050");  // i (0-200)
    } else {
      // table[0] == -50 or 50u
      return cudf::numeric_scalar<T>((50 - 100 * std::is_signed_v<T>));  // i/100 (-100-100/ 0-200)
    }
  }();

  // Third literal value
  auto literal_value3 = []() {
    if constexpr (cudf::is_timestamp<T>()) {
      // table[1] == 25 timestamp d/s/ms/us/ns
      return cudf::timestamp_scalar<T>(T(typename T::duration(25)));  // i (0-200)
    } else if constexpr (cudf::is_duration<T>()) {
      // table[1] == 25 d/s/ms/us/ns
      return cudf::duration_scalar<T>(T(25));  // i (0-200)
    } else if constexpr (std::is_same_v<T, cudf::string_view>) {
      // table[2] == "0025"
      return cudf::string_scalar("0025");  // i (0-200)
    } else {
      // table[0] == -25 or 25u
      return cudf::numeric_scalar<T>((25 - 50 * std::is_signed_v<T>));  // i/100 (-100-100/ 0-200)
    }
  }();

  // Filtering AST - col_ref == 100 or col_ref == 50 or col_ref == 25
  {
    // Expected row group indices after filtering
    auto const expected_row_groups = [&]() {
      if constexpr (cuda::std::is_same_v<T, cudf::string_view>) {
        return std::vector<cudf::size_type>{
          0, 1, 2, 3};  // Constant string value present in all RGs
      } else if constexpr (cudf::is_chrono<T>()) {
        return std::vector<cudf::size_type>{
          1, 2, 3};  // Descending temporal values present in three RGs: 1,2,3
      } else if constexpr (cuda::std::is_signed_v<T>) {
        return std::vector<cudf::size_type>{0,
                                            1};  // Signed ascending values present in two RGs: 0,1
      } else {
        return std::vector<cudf::size_type>{
          0, 1, 2};  // Ascending values present in three RGs: 0,1,2
      }
    }();

    // Build the filter expression
    auto const literal1 = cudf::ast::literal(literal_value1);
    auto const literal2 = cudf::ast::literal(literal_value2);
    auto const literal3 = cudf::ast::literal(literal_value3);

    auto const filter_expression1 =
      cudf::ast::operation(cudf::ast::ast_operator::EQUAL, col_name, literal1);
    auto const filter_expression2 =
      cudf::ast::operation(cudf::ast::ast_operator::EQUAL, col_name, literal2);
    auto const filter_expression3 =
      cudf::ast::operation(cudf::ast::ast_operator::EQUAL, col_name, literal3);
    auto const filter_expression12 = cudf::ast::operation(
      cudf::ast::ast_operator::LOGICAL_OR, filter_expression1, filter_expression2);
    auto const filter_expression = cudf::ast::operation(
      cudf::ast::ast_operator::LOGICAL_OR, filter_expression12, filter_expression3);

    // Check the results
    EXPECT_EQ(filter_row_groups_with_dictionaries(file_buffer_span, filter_expression, stream, mr),
              expected_row_groups);
  }

  // Filtering AST - col_ref != 100 and col_ref != 50 and col_ref != 25
  {
    // Expected row group indices after filtering
    auto const expected_row_groups = [&]() {
      if constexpr (cuda::std::is_same_v<T, cudf::string_view>) {
        return std::vector<cudf::size_type>{};
      } else {
        return std::vector<cudf::size_type>{0, 1, 2, 3};
      }
    }();

    // Build the filter expression
    auto const literal1 = cudf::ast::literal(literal_value1);
    auto const literal2 = cudf::ast::literal(literal_value2);
    auto const literal3 = cudf::ast::literal(literal_value3);

    auto const filter_expression1 =
      cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, col_name, literal1);
    auto const filter_expression2 =
      cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, col_name, literal2);
    auto const filter_expression3 =
      cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, col_name, literal3);
    auto const filter_expression12 = cudf::ast::operation(
      cudf::ast::ast_operator::LOGICAL_AND, filter_expression1, filter_expression2);
    auto const filter_expression = cudf::ast::operation(
      cudf::ast::ast_operator::LOGICAL_AND, filter_expression12, filter_expression3);

    // Check the results
    EXPECT_EQ(filter_row_groups_with_dictionaries(file_buffer_span, filter_expression, stream, mr),
              expected_row_groups);
  }
}
