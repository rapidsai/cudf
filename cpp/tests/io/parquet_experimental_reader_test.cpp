/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.
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

#include "parquet_common.hpp"

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
  rmm::device_async_resource_ref mr)
{
  std::vector<rmm::device_buffer> buffers{};
  buffers.reserve(byte_ranges.size());

  std::transform(
    byte_ranges.begin(),
    byte_ranges.end(),
    std::back_inserter(buffers),
    [&](auto const& byte_range) {
      auto const chunk_offset = host_buffer.data() + byte_range.offset();
      auto const chunk_size   = byte_range.size();
      auto buffer             = rmm::device_buffer(chunk_size, stream, mr);
      CUDF_CUDA_TRY(cudaMemcpyAsync(
        buffer.data(), chunk_offset, chunk_size, cudaMemcpyHostToDevice, stream.value()));
      return buffer;
    });

  stream.synchronize_no_throw();
  return buffers;
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
auto create_parquet_with_stats()
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
      .row_group_size_rows(5000)
      .max_page_size_rows(1000)
      .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN);

  if constexpr (NumTableConcats > 1) {
    out_opts.set_row_group_size_rows(20000);
    out_opts.set_max_page_size_rows(5000);
  }

  cudf::io::write_parquet(out_opts);

  return std::pair{std::move(table), std::move(buffer)};
}

/**
 * @brief Read parquet file with the hybrid scan reader
 *
 * @param buffer Buffer containing the parquet file
 * @param filter_expression Filter expression
 * @param num_filter_columns Number of filter columns
 * @param payload_column_names List of paths of select payload column names, if any
 * @param stream CUDA stream for hybrid scan reader
 * @param mr Device memory resource
 *
 * @return Tuple of filter table, payload table, filter metadata, payload metadata, and the final
 *         row validity column
 */
auto hybrid_scan(std::vector<char>& buffer,
                 cudf::ast::operation const& filter_expression,
                 cudf::size_type num_filter_columns,
                 std::optional<std::vector<std::string>> const& payload_column_names,
                 rmm::cuda_stream_view stream,
                 rmm::device_async_resource_ref mr)
{
  // Create reader options with empty source info
  cudf::io::parquet_reader_options options =
    cudf::io::parquet_reader_options::builder().filter(filter_expression);

  // Set payload column names if provided
  if (payload_column_names.has_value()) { options.set_columns(payload_column_names.value()); }

  // Input file buffer span
  auto const file_buffer_span =
    cudf::host_span<uint8_t const>(reinterpret_cast<uint8_t const*>(buffer.data()), buffer.size());

  // Fetch footer and page index bytes from the buffer.
  auto const footer_buffer = fetch_footer_bytes(file_buffer_span);

  // Create hybrid scan reader with footer bytes
  auto const reader =
    std::make_unique<cudf::io::parquet::experimental::hybrid_scan_reader>(footer_buffer, options);

  // Get Parquet file metadata from the reader - API # 1
  [[maybe_unused]] auto const parquet_metadata = reader->parquet_metadata();

  // Get page index byte range from the reader - API # 2
  auto const page_index_byte_range = reader->page_index_byte_range();

  // Fetch page index bytes from the input buffer
  auto const page_index_buffer = fetch_page_index_bytes(file_buffer_span, page_index_byte_range);

  // Setup page index - API # 3
  reader->setup_page_index(page_index_buffer);

  // Get all row groups from the reader - API # 4
  auto input_row_group_indices = reader->all_row_groups(options);

  // Span to track current row group indices
  auto current_row_group_indices = cudf::host_span<cudf::size_type>(input_row_group_indices);

  // Filter row groups with stats - API # 5
  auto stats_filtered_row_group_indices =
    reader->filter_row_groups_with_stats(current_row_group_indices, options, stream);

  // Update current row group indices
  current_row_group_indices = stats_filtered_row_group_indices;

  // Get bloom filter and dictionary page byte ranges from the reader - API # 6
  auto [bloom_filter_byte_ranges, dict_page_byte_ranges] =
    reader->secondary_filters_byte_ranges(current_row_group_indices, options);

  // If we have dictionary page byte ranges, filter row groups with dictionary pages - API # 7
  std::vector<cudf::size_type> dictionary_page_filtered_row_group_indices;
  dictionary_page_filtered_row_group_indices.reserve(current_row_group_indices.size());
  if (dict_page_byte_ranges.size()) {
    // Fetch dictionary page buffers from the input file buffer
    std::vector<rmm::device_buffer> dictionary_page_buffers =
      fetch_byte_ranges(file_buffer_span, dict_page_byte_ranges, stream, mr);

    // NOT YET IMPLEMENTED - Filter row groups with dictionary pages
    dictionary_page_filtered_row_group_indices = reader->filter_row_groups_with_dictionary_pages(
      dictionary_page_buffers, current_row_group_indices, options, stream);

    // Update current row group indices
    current_row_group_indices = dictionary_page_filtered_row_group_indices;
  }

  // If we have bloom filter byte ranges, filter row groups with bloom filters - API # 8
  std::vector<cudf::size_type> bloom_filtered_row_group_indices;
  bloom_filtered_row_group_indices.reserve(current_row_group_indices.size());
  if (bloom_filter_byte_ranges.size()) {
    // Fetch bloom filter data from the input file buffer
    std::vector<rmm::device_buffer> bloom_filter_data =
      fetch_byte_ranges(file_buffer_span, bloom_filter_byte_ranges, stream, mr);

    // Filter row groups with bloom filters
    bloom_filtered_row_group_indices = reader->filter_row_groups_with_bloom_filters(
      bloom_filter_data, current_row_group_indices, options, stream);

    // Update current row group indices
    current_row_group_indices = bloom_filtered_row_group_indices;
  }

  // Filter data pages with `PageIndex` stats - API # 9
  auto [row_mask, data_page_mask] =
    reader->filter_data_pages_with_stats(current_row_group_indices, options, stream, mr);

  EXPECT_EQ(data_page_mask.size(), num_filter_columns);

  // Get column chunk byte ranges from the reader - API # 10
  auto const filter_column_chunk_byte_ranges =
    reader->filter_column_chunks_byte_ranges(current_row_group_indices, options);

  // Fetch column chunk device buffers from the input buffer
  auto filter_column_chunk_buffers =
    fetch_byte_ranges(file_buffer_span, filter_column_chunk_byte_ranges, stream, mr);

  // Materialize the table with only the filter columns - API # 11
  auto [filter_table, filter_metadata] =
    reader->materialize_filter_columns(data_page_mask,
                                       current_row_group_indices,
                                       std::move(filter_column_chunk_buffers),
                                       row_mask->mutable_view(),
                                       options,
                                       stream);

  // Get column chunk byte ranges from the reader - API # 12
  auto const payload_column_chunk_byte_ranges =
    reader->payload_column_chunks_byte_ranges(current_row_group_indices, options);

  // Fetch column chunk device buffers from the input buffer
  [[maybe_unused]] auto payload_column_chunk_buffers =
    fetch_byte_ranges(file_buffer_span, payload_column_chunk_byte_ranges, stream, mr);

  // Materialize the table with only the payload columns - API # 13
  [[maybe_unused]] auto [payload_table, payload_metadata] =
    reader->materialize_payload_columns(current_row_group_indices,
                                        std::move(payload_column_chunk_buffers),
                                        row_mask->view(),
                                        options,
                                        stream);

  return std::tuple{std::move(filter_table),
                    std::move(payload_table),
                    std::move(filter_metadata),
                    std::move(payload_metadata),
                    std::move(row_mask)};
}

}  // namespace

TEST_F(ParquetExperimentalReaderTest, TestMetadata)
{
  // Create a table with several row groups each with a single page.
  auto constexpr num_concat    = 1;
  auto [written_table, buffer] = create_parquet_with_stats<num_concat>();

  // Filtering AST - table[0] < 100
  auto literal_value     = cudf::numeric_scalar<uint32_t>(100);
  auto literal           = cudf::ast::literal(literal_value);
  auto col_ref_0         = cudf::ast::column_name_reference("col_uint32");
  auto filter_expression = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, literal);

  // Create reader options with empty source info
  cudf::io::parquet_reader_options options =
    cudf::io::parquet_reader_options::builder().filter(filter_expression);

  // Input file buffer span
  auto const file_buffer_span =
    cudf::host_span<uint8_t const>(reinterpret_cast<uint8_t const*>(buffer.data()), buffer.size());

  // Fetch footer and page index bytes from the buffer.
  auto const footer_buffer = fetch_footer_bytes(file_buffer_span);

  // Create hybrid scan reader with footer bytes
  auto const reader =
    std::make_unique<cudf::io::parquet::experimental::hybrid_scan_reader>(footer_buffer, options);

  // Get Parquet file metadata from the reader - API # 1
  auto parquet_metadata = reader->parquet_metadata();

  // Check that the offset and column indices are not present
  EXPECT_FALSE(parquet_metadata.row_groups[0].columns[0].offset_index.has_value());
  EXPECT_FALSE(parquet_metadata.row_groups[0].columns[0].column_index.has_value());

  // Get page index byte range from the reader - API # 2
  auto const page_index_byte_range = reader->page_index_byte_range();

  // Fetch page index bytes from the input buffer
  auto const page_index_buffer = fetch_page_index_bytes(file_buffer_span, page_index_byte_range);

  // Setup page index - API # 3
  reader->setup_page_index(page_index_buffer);

  // Get Parquet file metadata from the reader again
  parquet_metadata = reader->parquet_metadata();

  // Check that the offset and column indices are now present
  EXPECT_TRUE(parquet_metadata.row_groups[0].columns[0].offset_index.has_value());
  EXPECT_TRUE(parquet_metadata.row_groups[0].columns[0].column_index.has_value());

  // Get all row groups from the reader - API # 4
  auto input_row_group_indices = reader->all_row_groups(options);
  // Expect 4 = 20000 rows / 5000 rows per row group
  EXPECT_EQ(input_row_group_indices.size(), 4);

  // Explicitly set the row groups to read
  options.set_row_groups({{0, 1}});

  // Get all row groups from the reader again
  input_row_group_indices = reader->all_row_groups(options);
  // Expect only 2 row groups now
  EXPECT_EQ(reader->all_row_groups(options).size(), 2);
}

TEST_F(ParquetExperimentalReaderTest, PruneRowGroupsOnly)
{
  srand(31337);

  // A table not concated with itself with result in a parquet file with several row groups each
  // with a single page. Since there is only one page per row group, the page and row group stats
  // are identical and we can only prune row groups.
  auto constexpr num_concat    = 1;
  auto [written_table, buffer] = create_parquet_with_stats<num_concat>();

  // Filtering AST - table[0] < 100
  auto constexpr num_filter_columns = 1;
  auto literal_value                = cudf::numeric_scalar<uint32_t>(100);
  auto literal                      = cudf::ast::literal(literal_value);
  auto col_ref_0                    = cudf::ast::column_name_reference("col_uint32");
  auto filter_expression = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, literal);

  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();

  // Read parquet using the hybrid scan reader
  auto [read_filter_table, read_payload_table, read_filter_meta, read_payload_meta, row_mask] =
    hybrid_scan(buffer, filter_expression, num_filter_columns, {}, stream, mr);

  CUDF_EXPECTS(read_filter_table->num_rows() == read_payload_table->num_rows(),
               "Filter and payload tables should have the same number of rows");

  // Check equivalence (equal without checking nullability) with the parquet file read with the
  // original reader
  {
    cudf::io::parquet_reader_options const options =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info(buffer.data(), buffer.size()))
        .filter(filter_expression);
    auto [expected_tbl, expected_meta] = cudf::io::read_parquet(options, stream);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected_tbl->select({0}), read_filter_table->view());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected_tbl->select({1, 2}), read_payload_table->view());
  }
}

TEST_F(ParquetExperimentalReaderTest, TestPayloadColumns)
{
  srand(31337);

  // A table not concated with itself with result in a parquet file with several row groups each
  // with a single page. Since there is only one page per row group, the page and row group stats
  // are identical and we can only prune row groups.
  auto constexpr num_concat    = 1;
  auto [written_table, buffer] = create_parquet_with_stats<num_concat>();

  // Filtering AST - table[0] < 100
  auto constexpr num_filter_columns = 1;
  auto literal_value                = cudf::numeric_scalar<uint32_t>(100);
  auto literal                      = cudf::ast::literal(literal_value);
  auto col_ref_0                    = cudf::ast::column_name_reference("col_uint32");
  auto filter_expression = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, literal);

  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();

  {
    auto const payload_column_names = std::vector<std::string>{"col_uint32", "col_str"};
    // Read parquet using the hybrid scan reader
    auto [read_filter_table, read_payload_table, read_filter_meta, read_payload_meta, row_mask] =
      hybrid_scan(buffer, filter_expression, num_filter_columns, payload_column_names, stream, mr);

    CUDF_EXPECTS(read_filter_table->num_rows() == read_payload_table->num_rows(),
                 "Filter and payload tables should have the same number of rows");
    cudf::io::parquet_reader_options const options =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info(buffer.data(), buffer.size()))
        .filter(filter_expression);
    auto [expected_tbl, expected_meta] = cudf::io::read_parquet(options, stream);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected_tbl->select({0}), read_filter_table->view());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected_tbl->select({2}), read_payload_table->view());
  }

  {
    auto const payload_column_names = std::vector<std::string>{"col_str", "col_int64"};
    // Read parquet using the hybrid scan reader
    auto [read_filter_table, read_payload_table, read_filter_meta, read_payload_meta, row_mask] =
      hybrid_scan(buffer, filter_expression, num_filter_columns, payload_column_names, stream, mr);

    CUDF_EXPECTS(read_filter_table->num_rows() == read_payload_table->num_rows(),
                 "Filter and payload tables should have the same number of rows");
    cudf::io::parquet_reader_options const options =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info(buffer.data(), buffer.size()))
        .filter(filter_expression);
    auto [expected_tbl, expected_meta] = cudf::io::read_parquet(options, stream);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected_tbl->select({0}), read_filter_table->view());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected_tbl->select({2, 1}), read_payload_table->view());
  }
}

TEST_F(ParquetExperimentalReaderTest, PrunePagesOnly)
{
  srand(31337);

  // A table concatenated multiple times by itself with result in a parquet file with a row group
  // per concatenation with multiple pages per row group. Since all row groups will be identical, we
  // can only prune pages based on `PageIndex` stats
  auto constexpr num_concat    = 2;
  auto [written_table, buffer] = create_parquet_with_stats<num_concat>();

  // Filtering AST - table[0] < 100
  auto constexpr num_filter_columns = 1;
  auto literal_value                = cudf::numeric_scalar<uint32_t>(100);
  auto literal                      = cudf::ast::literal(literal_value);
  auto col_ref_0                    = cudf::ast::column_name_reference("col_uint32");
  auto filter_expression = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, literal);

  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();

  // Read parquet using the hybrid scan reader
  auto [read_filter_table, read_payload_table, read_filter_meta, read_payload_meta, row_mask] =
    hybrid_scan(buffer, filter_expression, num_filter_columns, {}, stream, mr);

  CUDF_EXPECTS(read_filter_table->num_rows() == read_payload_table->num_rows(),
               "Filter and payload tables should have the same number of rows");

  // Check equivalence (equal without checking nullability) with the parquet file read with the
  // original reader
  {
    cudf::io::parquet_reader_options const options =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info(buffer.data(), buffer.size()))
        .filter(filter_expression);
    auto [expected_tbl, expected_meta] = cudf::io::read_parquet(options, stream);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected_tbl->select({0}), read_filter_table->view());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected_tbl->select({1, 2}), read_payload_table->view());
  }

  // Check equivalence (equal without checking nullability) with the original table with the
  // applied boolean mask
  {
    auto col_ref_0 = cudf::ast::column_reference(0);
    auto filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, literal);

    auto predicate = cudf::compute_column(written_table->view(), filter_expression);
    EXPECT_EQ(predicate->view().type().id(), cudf::type_id::BOOL8)
      << "Predicate filter should return a boolean";
    auto expected = cudf::apply_boolean_mask(written_table->view(), *predicate);
    // Check equivalence as the nullability between columns may be different
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected->select({0}), read_filter_table->view());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected->select({1, 2}), read_payload_table->view());
  }
}
