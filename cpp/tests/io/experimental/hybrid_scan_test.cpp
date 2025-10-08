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

#include "hybrid_scan_common.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/io_metadata_utilities.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf/column/column.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/aligned.hpp>
#include <rmm/mr/device/aligned_resource_adaptor.hpp>

auto constexpr bloom_filter_alignment = rmm::CUDA_ALLOCATION_ALIGNMENT;

namespace {

/**
 * @brief Concatenate a vector of tables and return the resultant table
 *
 * @param tables Vector of tables to concatenate
 * @param stream CUDA stream to use
 *
 * @return Unique pointer to the resultant concatenated table.
 */
std::unique_ptr<cudf::table> concatenate_tables(std::vector<std::unique_ptr<cudf::table>> tables,
                                                rmm::cuda_stream_view stream)
{
  if (tables.size() == 1) { return std::move(tables[0]); }

  std::vector<cudf::table_view> table_views;
  table_views.reserve(tables.size());
  std::transform(
    tables.begin(), tables.end(), std::back_inserter(table_views), [&](auto const& tbl) {
      return tbl->view();
    });
  // Construct the final table
  return cudf::concatenate(table_views, stream);
}

/**
 * @brief Apply parquet filters to the file buffer
 *
 * @param file_buffer_span Input file buffer span
 * @param options Reader options
 * @param stream CUDA stream
 * @param mr Device memory resource
 *
 * @return A tuple of the reader, filtered row group indices, and row mask and data page mask from
 * data page pruning
 */
auto apply_parquet_filters(cudf::host_span<uint8_t const> file_buffer_span,
                           cudf::io::parquet_reader_options const& options,
                           rmm::cuda_stream_view stream,
                           rmm::device_async_resource_ref mr)
{
  // Fetch footer and page index bytes from the buffer.
  auto const footer_buffer = fetch_footer_bytes(file_buffer_span);

  // Create hybrid scan reader with footer bytes
  auto reader =
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

  // Filter row groups with stats
  auto stats_filtered_row_group_indices =
    reader->filter_row_groups_with_stats(current_row_group_indices, options, stream);

  // Update current row group indices
  current_row_group_indices = stats_filtered_row_group_indices;

  // Get bloom filter and dictionary page byte ranges from the reader
  auto [bloom_filter_byte_ranges, dict_page_byte_ranges] =
    reader->secondary_filters_byte_ranges(current_row_group_indices, options);

  // If we have dictionary page byte ranges, filter row groups with dictionary pages
  std::vector<cudf::size_type> dictionary_page_filtered_row_group_indices;
  dictionary_page_filtered_row_group_indices.reserve(current_row_group_indices.size());
  if (dict_page_byte_ranges.size()) {
    // Fetch dictionary page buffers from the input file buffer
    std::vector<rmm::device_buffer> dictionary_page_buffers =
      fetch_byte_ranges(file_buffer_span, dict_page_byte_ranges, stream, mr);

    // Filter row groups with dictionary pages
    dictionary_page_filtered_row_group_indices = reader->filter_row_groups_with_dictionary_pages(
      dictionary_page_buffers, current_row_group_indices, options, stream);

    // Update current row group indices
    current_row_group_indices = dictionary_page_filtered_row_group_indices;
  }

  // If we have bloom filter byte ranges, filter row groups with bloom filters
  std::vector<cudf::size_type> bloom_filtered_row_group_indices;
  bloom_filtered_row_group_indices.reserve(current_row_group_indices.size());
  if (bloom_filter_byte_ranges.size()) {
    // Fetch 32 byte aligned bloom filter data buffers from the input file buffer
    auto aligned_mr = rmm::mr::aligned_resource_adaptor(cudf::get_current_device_resource(),
                                                        bloom_filter_alignment);

    std::vector<rmm::device_buffer> bloom_filter_data =
      fetch_byte_ranges(file_buffer_span, bloom_filter_byte_ranges, stream, aligned_mr);

    // Filter row groups with bloom filters
    bloom_filtered_row_group_indices = reader->filter_row_groups_with_bloom_filters(
      bloom_filter_data, current_row_group_indices, options, stream);

    // Update current row group indices
    current_row_group_indices = bloom_filtered_row_group_indices;
  }

  // Build row mask using page index stats
  auto row_mask =
    reader->build_row_mask_with_page_index_stats(current_row_group_indices, options, stream, mr);

  std::vector<cudf::size_type> final_row_group_indices(current_row_group_indices.begin(),
                                                       current_row_group_indices.end());

  return std::tuple{std::move(reader), std::move(final_row_group_indices), std::move(row_mask)};
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
                 rmm::device_async_resource_ref mr,
                 rmm::mr::aligned_resource_adaptor<rmm::mr::device_memory_resource>& aligned_mr)
{
  // Create reader options with empty source info
  cudf::io::parquet_reader_options options =
    cudf::io::parquet_reader_options::builder().filter(filter_expression);

  // Set payload column names if provided
  if (payload_column_names.has_value()) { options.set_columns(payload_column_names.value()); }

  // Input file buffer span
  auto const file_buffer_span =
    cudf::host_span<uint8_t const>(reinterpret_cast<uint8_t const*>(buffer.data()), buffer.size());

  auto [reader, filtered_row_group_indices, row_mask] =
    apply_parquet_filters(file_buffer_span, options, stream, mr);

  auto current_row_group_indices = cudf::host_span<cudf::size_type>(filtered_row_group_indices);

  // Get column chunk byte ranges from the reader
  auto const filter_column_chunk_byte_ranges =
    reader->filter_column_chunks_byte_ranges(current_row_group_indices, options);

  // Fetch column chunk device buffers from the input buffer
  auto filter_column_chunk_buffers =
    fetch_byte_ranges(file_buffer_span, filter_column_chunk_byte_ranges, stream, mr);

  // Materialize the table with only the filter columns
  auto row_mask_mutable_view = row_mask->mutable_view();
  auto [filter_table, filter_metadata] =
    reader->materialize_filter_columns(current_row_group_indices,
                                       std::move(filter_column_chunk_buffers),
                                       row_mask_mutable_view,
                                       cudf::io::parquet::experimental::use_data_page_mask::YES,
                                       options,
                                       stream);

  // Get column chunk byte ranges from the reader
  auto const payload_column_chunk_byte_ranges =
    reader->payload_column_chunks_byte_ranges(current_row_group_indices, options);

  // Fetch column chunk device buffers from the input buffer
  auto payload_column_chunk_buffers =
    fetch_byte_ranges(file_buffer_span, payload_column_chunk_byte_ranges, stream, mr);

  // Materialize the table with only the payload columns
  auto [payload_table, payload_metadata] =
    reader->materialize_payload_columns(current_row_group_indices,
                                        std::move(payload_column_chunk_buffers),
                                        row_mask->view(),
                                        cudf::io::parquet::experimental::use_data_page_mask::YES,
                                        options,
                                        stream);

  return std::tuple{std::move(filter_table),
                    std::move(payload_table),
                    std::move(filter_metadata),
                    std::move(payload_metadata),
                    std::move(row_mask)};
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
auto chunked_hybrid_scan(
  std::vector<char> const& buffer,
  cudf::ast::operation const& filter_expression,
  cudf::size_type num_filter_columns,
  std::optional<std::vector<std::string>> const& payload_column_names,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  rmm::mr::aligned_resource_adaptor<rmm::mr::device_memory_resource>& aligned_mr)
{
  // Create reader options with empty source info
  cudf::io::parquet_reader_options options =
    cudf::io::parquet_reader_options::builder().filter(filter_expression);

  // Set payload column names if provided
  if (payload_column_names.has_value()) { options.set_columns(payload_column_names.value()); }

  // Input file buffer span
  auto const file_buffer_span =
    cudf::host_span<uint8_t const>(reinterpret_cast<uint8_t const*>(buffer.data()), buffer.size());

  auto [reader, filtered_row_group_indices, row_mask] =
    apply_parquet_filters(file_buffer_span, options, stream, mr);

  auto current_row_group_indices = cudf::host_span<cudf::size_type>(filtered_row_group_indices);

  // Helper to split the materialization of filter columns into chunks
  auto tables          = std::vector<std::unique_ptr<cudf::table>>{};
  auto filter_metadata = cudf::io::table_metadata{};
  auto const materialize_filter_columns =
    [&](cudf::host_span<cudf::size_type const> row_group_indices) {
      // Get column chunk byte ranges from the reader and fetch device buffers
      auto const filter_column_chunk_byte_ranges =
        reader->filter_column_chunks_byte_ranges(row_group_indices, options);
      auto filter_column_chunk_buffers =
        fetch_byte_ranges(file_buffer_span, filter_column_chunk_byte_ranges, stream, mr);

      // Setup chunking for filter columns and materialize the columns
      reader->setup_chunking_for_filter_columns(
        1024,
        1024,
        row_group_indices,
        row_mask->view(),
        cudf::io::parquet::experimental::use_data_page_mask::YES,
        std::move(filter_column_chunk_buffers),
        options,
        stream);

      auto row_mask_mutable_view = row_mask->mutable_view();
      while (reader->has_next_table_chunk()) {
        auto chunk = reader->materialize_filter_columns_chunk(row_mask_mutable_view, stream);
        tables.push_back(std::move(chunk.tbl));
        filter_metadata = std::move(chunk.metadata);
      }
    };

  if (current_row_group_indices.size() > 1) {
    auto const row_group_split = current_row_group_indices.size() / 2;
    materialize_filter_columns(
      cudf::host_span<cudf::size_type const>{current_row_group_indices.begin(), row_group_split});
    materialize_filter_columns(
      cudf::host_span<cudf::size_type const>{current_row_group_indices.begin() + row_group_split,
                                             current_row_group_indices.size() - row_group_split});
  } else {
    materialize_filter_columns(current_row_group_indices);
  }

  auto filter_table = concatenate_tables(std::move(tables), stream);

  // Helper to split the materialization of payload columns into chunks
  tables.clear();
  auto payload_metadata = cudf::io::table_metadata{};
  auto const materialize_payload_columns =
    [&](cudf::host_span<cudf::size_type const> row_group_indices) {
      // Get column chunk byte ranges from the reader and fetch device buffers
      auto const payload_column_chunk_byte_ranges =
        reader->payload_column_chunks_byte_ranges(row_group_indices, options);
      auto payload_column_chunk_buffers =
        fetch_byte_ranges(file_buffer_span, payload_column_chunk_byte_ranges, stream, mr);

      // Setup chunking for payload columns and materialize the table
      reader->setup_chunking_for_payload_columns(
        1024,
        1024,
        row_group_indices,
        row_mask->view(),
        cudf::io::parquet::experimental::use_data_page_mask::YES,
        std::move(payload_column_chunk_buffers),
        options,
        stream);

      while (reader->has_next_table_chunk()) {
        auto chunk = reader->materialize_payload_columns_chunk(row_mask->view(), stream);
        tables.push_back(std::move(chunk.tbl));
        payload_metadata = std::move(chunk.metadata);
      }
    };

  if (current_row_group_indices.size() > 1) {
    auto const row_group_split = current_row_group_indices.size() / 2;
    materialize_payload_columns(
      cudf::host_span<cudf::size_type const>{current_row_group_indices.begin(), row_group_split});
    materialize_payload_columns(
      cudf::host_span<cudf::size_type const>{current_row_group_indices.begin() + row_group_split,
                                             current_row_group_indices.size() - row_group_split});
  } else {
    materialize_payload_columns(current_row_group_indices);
  }

  auto payload_table = concatenate_tables(std::move(tables), stream);

  // Return the filter table and metadata, payload table and metadata, and the final row mask
  return std::tuple{std::move(filter_table),
                    std::move(payload_table),
                    std::move(filter_metadata),
                    std::move(payload_metadata),
                    std::move(row_mask)};
}

}  // namespace

// Base test fixture for tests
struct HybridScanTest : public cudf::test::BaseFixture {};

TEST_F(HybridScanTest, PruneRowGroupsOnlyAndScanAllColumns)
{
  srand(0xc0ffee);
  using T = uint32_t;

  // A table with several row groups each containing a single page per column. The data page and row
  // group stats are identical so only row groups can be pruned using stats
  auto constexpr num_concat            = 1;
  auto [written_table, parquet_buffer] = create_parquet_with_stats<T, num_concat>();

  // Filtering AST - table[0] < 100
  auto constexpr num_filter_columns = 1;
  auto literal_value                = cudf::numeric_scalar<uint32_t>(100);
  auto literal                      = cudf::ast::literal(literal_value);
  auto col_ref_0                    = cudf::ast::column_name_reference("col0");
  auto filter_expression = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, literal);

  auto stream     = cudf::get_default_stream();
  auto mr         = cudf::get_current_device_resource_ref();
  auto aligned_mr = rmm::mr::aligned_resource_adaptor<rmm::mr::device_memory_resource>(
    cudf::get_current_device_resource(), bloom_filter_alignment);

  // Read parquet using the hybrid scan reader
  auto [read_filter_table, read_payload_table, read_filter_meta, read_payload_meta, row_mask] =
    hybrid_scan(parquet_buffer, filter_expression, num_filter_columns, {}, stream, mr, aligned_mr);

  // Read parquet using the chunked hybrid scan reader
  auto [read_filter_table_chunked,
        read_payload_table_chunked,
        read_filter_meta_chunked,
        read_payload_meta_chunked,
        row_mask_chunked] =
    chunked_hybrid_scan(
      parquet_buffer, filter_expression, num_filter_columns, {}, stream, mr, aligned_mr);

  CUDF_EXPECTS(read_filter_table->num_rows() == read_payload_table->num_rows(),
               "Filter and payload tables should have the same number of rows");
  CUDF_EXPECTS(read_filter_table_chunked->num_rows() == read_payload_table_chunked->num_rows(),
               "Filter and payload tables should have the same number of rows");

  // Check equivalence (equal without checking nullability) with the parquet file read with the
  // original reader
  {
    cudf::io::parquet_reader_options const options =
      cudf::io::parquet_reader_options::builder(
        cudf::io::source_info(cudf::host_span<char>(parquet_buffer.data(), parquet_buffer.size())))
        .filter(filter_expression);
    auto [expected_tbl, expected_meta] = read_parquet(options, stream);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected_tbl->select({0}), read_filter_table->view());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected_tbl->select({0}),
                                       read_filter_table_chunked->view());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected_tbl->select({1, 2}), read_payload_table->view());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected_tbl->select({1, 2}),
                                       read_payload_table_chunked->view());
  }
}

TEST_F(HybridScanTest, PruneRowGroupsOnlyAndScanSelectColumns)
{
  srand(0xcafe);
  using T = cudf::timestamp_ms;

  // A table with several row groups each containing a single page per column. The data page and row
  // group stats are identical so only row groups can be pruned using stats
  auto constexpr num_concat            = 1;
  auto [written_table, parquet_buffer] = create_parquet_with_stats<T, num_concat>();

  // Filtering AST - table[0] < 100
  auto constexpr num_filter_columns = 1;
  auto literal_value                = cudf::timestamp_scalar<T>(T{typename T::duration{100}});
  auto literal                      = cudf::ast::literal(literal_value);
  auto col_ref_0                    = cudf::ast::column_name_reference("col0");
  auto filter_expression = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, literal);

  auto stream     = cudf::get_default_stream();
  auto mr         = cudf::get_current_device_resource_ref();
  auto aligned_mr = rmm::mr::aligned_resource_adaptor<rmm::mr::device_memory_resource>(
    cudf::get_current_device_resource(), bloom_filter_alignment);

  {
    auto const payload_column_names = std::vector<std::string>{"col0", "col2"};
    // Read parquet using the hybrid scan reader
    auto [read_filter_table, read_payload_table, read_filter_meta, read_payload_meta, row_mask] =
      hybrid_scan(parquet_buffer,
                  filter_expression,
                  num_filter_columns,
                  payload_column_names,
                  stream,
                  mr,
                  aligned_mr);
    // Read parquet using the chunked hybrid scan reader
    auto [read_filter_table_chunked,
          read_payload_table_chunked,
          read_filter_meta_chunked,
          read_payload_meta_chunked,
          row_mask_chunked] = chunked_hybrid_scan(parquet_buffer,
                                                  filter_expression,
                                                  num_filter_columns,
                                                  payload_column_names,
                                                  stream,
                                                  mr,
                                                  aligned_mr);

    CUDF_EXPECTS(read_filter_table->num_rows() == read_payload_table->num_rows(),
                 "Filter and payload tables should have the same number of rows");
    CUDF_EXPECTS(read_filter_table_chunked->num_rows() == read_payload_table_chunked->num_rows(),
                 "Filter and payload tables should have the same number of rows");

    cudf::io::parquet_reader_options const options =
      cudf::io::parquet_reader_options::builder(
        cudf::io::source_info(cudf::host_span<char>(parquet_buffer.data(), parquet_buffer.size())))
        .filter(filter_expression);
    auto [expected_tbl, expected_meta] = read_parquet(options, stream);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected_tbl->select({0}), read_filter_table->view());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected_tbl->select({0}),
                                       read_filter_table_chunked->view());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected_tbl->select({2}), read_payload_table->view());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected_tbl->select({2}),
                                       read_payload_table_chunked->view());
  }

  {
    auto const payload_column_names = std::vector<std::string>{"col2", "col1"};
    // Read parquet using the hybrid scan reader
    auto [read_filter_table, read_payload_table, read_filter_meta, read_payload_meta, row_mask] =
      hybrid_scan(parquet_buffer,
                  filter_expression,
                  num_filter_columns,
                  payload_column_names,
                  stream,
                  mr,
                  aligned_mr);

    CUDF_EXPECTS(read_filter_table->num_rows() == read_payload_table->num_rows(),
                 "Filter and payload tables should have the same number of rows");
    cudf::io::parquet_reader_options const options =
      cudf::io::parquet_reader_options::builder(
        cudf::io::source_info(cudf::host_span<char>(parquet_buffer.data(), parquet_buffer.size())))
        .filter(filter_expression);
    auto [expected_tbl, expected_meta] = read_parquet(options, stream);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected_tbl->select({0}), read_filter_table->view());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected_tbl->select({2, 1}), read_payload_table->view());
  }
}

TEST_F(HybridScanTest, PruneDataPagesOnlyAndScanAllColumns)
{
  srand(0xf00d);
  using T = cudf::duration_ms;

  // A table concatenated with itself results in a parquet file with a row group per concatenated
  // table, each containing multiple pages per column. All row groups will be identical so only data
  // pages can be pruned using page index stats
  auto constexpr num_concat    = 2;
  auto [written_table, buffer] = create_parquet_with_stats<T, num_concat>();

  // Filtering AST - table[0] < 100
  auto constexpr num_filter_columns = 1;
  auto literal_value                = cudf::duration_scalar<T>(T{100});
  auto literal                      = cudf::ast::literal(literal_value);
  auto col_ref_0                    = cudf::ast::column_name_reference("col0");
  auto filter_expression = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, literal);

  auto stream     = cudf::get_default_stream();
  auto mr         = cudf::get_current_device_resource_ref();
  auto aligned_mr = rmm::mr::aligned_resource_adaptor<rmm::mr::device_memory_resource>(
    cudf::get_current_device_resource(), bloom_filter_alignment);

  // Read parquet using the hybrid scan reader
  auto [read_filter_table, read_payload_table, read_filter_meta, read_payload_meta, row_mask] =
    hybrid_scan(buffer, filter_expression, num_filter_columns, {}, stream, mr, aligned_mr);

  // Read parquet using the chunked hybrid scan reader
  auto [read_filter_table_chunked,
        read_payload_table_chunked,
        read_filter_meta_chunked,
        read_payload_meta_chunked,
        row_mask_chunked] =
    chunked_hybrid_scan(buffer, filter_expression, num_filter_columns, {}, stream, mr, aligned_mr);

  CUDF_EXPECTS(read_filter_table->num_rows() == read_payload_table->num_rows(),
               "Filter and payload tables should have the same number of rows");
  CUDF_EXPECTS(read_filter_table_chunked->num_rows() == read_payload_table_chunked->num_rows(),
               "Filter and payload tables should have the same number of rows");

  // Check equivalence (equal without checking nullability) with the parquet file read with the
  // original reader
  {
    cudf::io::parquet_reader_options const options =
      cudf::io::parquet_reader_options::builder(
        cudf::io::source_info(cudf::host_span<char>(buffer.data(), buffer.size())))
        .filter(filter_expression);
    auto [expected_tbl, expected_meta] = read_parquet(options, stream);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected_tbl->select({0}), read_filter_table->view());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected_tbl->select({0}),
                                       read_filter_table_chunked->view());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected_tbl->select({1, 2}), read_payload_table->view());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected_tbl->select({1, 2}),
                                       read_payload_table_chunked->view());
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

TEST_F(HybridScanTest, MaterializeMixedPayloadColumns)
{
  std::mt19937 gen(0xcaffe);

  // Parquet buffer
  std::vector<char> parquet_buffer;
  {
    auto constexpr num_rows = num_ordered_rows;

    // Validity helpers
    std::bernoulli_distribution bn(0.7f);
    auto valids =
      cudf::detail::make_counting_transform_iterator(0, [&](int index) { return bn(gen); });
    auto list_valids =
      cudf::detail::make_counting_transform_iterator(0, [&](int index) { return index % 100; });
    auto struct_valids_iter =
      cudf::detail::make_counting_transform_iterator(0, [&](int index) { return index % 121; });
    std::vector<bool> struct_valids(num_rows);
    std::copy(struct_valids_iter, struct_valids_iter + num_rows, struct_valids.begin());

    // str and list<str> helpers
    std::vector<std::string> strings{
      "abc", "x", "bananas", "gpu", "minty", "backspace", "", "cayenne", "turbine", "soft"};
    std::uniform_int_distribution<int> uni(0, strings.size() - 1);
    auto string_iter = cudf::detail::make_counting_transform_iterator(
      0, [&](cudf::size_type idx) { return strings[uni(gen)]; });

    // uint32_t(non-nullable)
    auto col0 = testdata::ascending<uint32_t>();
    // str(non-nullable)
    auto col1 = testdata::ascending<cudf::string_view>();

    // list<bool(nullable)>(nullable)
    auto bools_iter =
      cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 2; });
    auto bools_col =
      cudf::test::fixed_width_column_wrapper<bool>(bools_iter, bools_iter + num_rows, valids);
    auto offsets_iter = thrust::counting_iterator<int32_t>(0);
    auto offsets_col =
      cudf::test::fixed_width_column_wrapper<int32_t>(offsets_iter, offsets_iter + num_rows + 1);
    auto [null_mask, null_count] =
      cudf::test::detail::make_null_mask(list_valids, list_valids + num_rows);
    auto _col2 = cudf::make_lists_column(
      num_rows, offsets_col.release(), bools_col.release(), null_count, std::move(null_mask));
    auto col2 = cudf::purge_nonempty_nulls(*_col2);

    // list<list<bool(nullable)>(nullable)>(nullable)
    auto col3 = make_parquet_list_list_col<bool>(0, num_rows, 5, 8, true);

    // list<str(nullable)>(must be non-nullable)
    auto const make_list_str_column = [&](bool is_nullable) {
      constexpr int string_per_row  = 3;
      constexpr int num_string_rows = num_rows * string_per_row;
      cudf::test::strings_column_wrapper string_col{
        string_iter, string_iter + num_string_rows, valids};
      auto offset_iter = cudf::detail::make_counting_transform_iterator(
        0, [](cudf::size_type idx) { return idx * string_per_row; });
      cudf::test::fixed_width_column_wrapper<cudf::size_type> offsets(offset_iter,
                                                                      offset_iter + num_rows + 1);
      auto [null_mask, null_count] = [&]() {
        if (is_nullable) {
          return cudf::test::detail::make_null_mask(valids, valids + num_rows);
        } else {
          return std::make_pair(rmm::device_buffer{}, 0);
        }
      }();
      return cudf::make_lists_column(
        num_rows, offsets.release(), string_col.release(), null_count, std::move(null_mask));
    };

    // str(nullable)
    auto col4 = cudf::test::strings_column_wrapper{string_iter, string_iter + num_rows, valids};

    // list<str(nullable)>(non-nullable)
    auto col5 = make_list_str_column(false);

    // list<str(nullable)>(nullable)
    auto col6 = make_list_str_column(true);

    // struct<list<str(nullable)>(nullable), int(nullable), float(nullable)>(nullable)
    auto values    = thrust::make_counting_iterator(0);
    auto col7_list = make_list_str_column(true);
    cudf::test::fixed_width_column_wrapper<int> col7_ints(values, values + num_rows, valids);
    cudf::test::fixed_width_column_wrapper<float> col7_floats(values, values + num_rows, valids);
    std::vector<std::unique_ptr<cudf::column>> col7_children;
    col7_children.push_back(std::move(col7_list));
    col7_children.push_back(col7_ints.release());
    col7_children.push_back(col7_floats.release());
    cudf::test::structs_column_wrapper _col7(std::move(col7_children), struct_valids);
    auto col7 = cudf::purge_nonempty_nulls(_col7);

    // struct<str(nullable), bool(nullable)>(nullable)
    auto col8_str = cudf::test::strings_column_wrapper{string_iter, string_iter + num_rows, valids};
    cudf::test::fixed_width_column_wrapper<bool> col8_bools(values, values + num_rows, valids);
    std::vector<std::unique_ptr<cudf::column>> col8_children;
    col8_children.push_back(col8_str.release());
    col8_children.push_back(col8_bools.release());
    cudf::test::structs_column_wrapper _col8(std::move(col8_children), struct_valids);
    auto col8 = cudf::purge_nonempty_nulls(_col8);

    // list<list<str(nullable)>(nullable)>(nullable)
    constexpr int string_per_row  = 3;
    constexpr int lists_per_list  = 2;
    constexpr int num_string_rows = num_rows * string_per_row * lists_per_list;
    cudf::test::strings_column_wrapper string_col{
      string_iter, string_iter + num_string_rows, valids};
    auto offset_iter = cudf::detail::make_counting_transform_iterator(
      0, [](cudf::size_type idx) { return idx * string_per_row; });
    cudf::test::fixed_width_column_wrapper<cudf::size_type> list_offsets(
      offset_iter, offset_iter + (num_rows * lists_per_list) + 1);
    std::tie(null_mask, null_count) =
      cudf::test::detail::make_null_mask(list_valids, list_valids + (num_rows * lists_per_list));

    auto _list_col = cudf::make_lists_column(num_rows * lists_per_list,
                                             list_offsets.release(),
                                             string_col.release(),
                                             null_count,
                                             std::move(null_mask));
    auto list_col  = cudf::purge_nonempty_nulls(*_list_col);

    auto list_list_offsets_iter = cudf::detail::make_counting_transform_iterator(
      0, [](cudf::size_type idx) { return idx * lists_per_list; });
    cudf::test::fixed_width_column_wrapper<cudf::size_type> list_list_offsets(
      list_list_offsets_iter, list_list_offsets_iter + num_rows + 1);
    auto list_list_valids =
      cudf::detail::make_counting_transform_iterator(0, [&](int index) { return index % 80; });
    std::tie(null_mask, null_count) =
      cudf::test::detail::make_null_mask(list_list_valids, list_list_valids + num_rows);

    auto _col9 = cudf::make_lists_column(
      num_rows, list_list_offsets.release(), std::move(list_col), null_count, std::move(null_mask));

    auto col9 = cudf::purge_nonempty_nulls(*_col9);

    // Input table
    auto constexpr num_concat = 3;
    auto table =
      cudf::table_view{{col0, col1, *col2, *col3, col4, *col5, *col6, *col7, *col8, *col9}};
    auto expected = cudf::concatenate(std::vector<table_view>(num_concat, table));
    table         = expected->view();
    cudf::io::table_input_metadata expected_metadata(table);
    expected_metadata.column_metadata[0].set_name("col0");
    expected_metadata.column_metadata[1].set_name("col1");
    expected_metadata.column_metadata[2].set_name("col2");
    expected_metadata.column_metadata[3].set_name("col3");
    expected_metadata.column_metadata[4].set_name("col4");
    expected_metadata.column_metadata[5].set_name("col5");
    expected_metadata.column_metadata[6].set_name("col6");
    expected_metadata.column_metadata[7].set_name("col7");
    expected_metadata.column_metadata[8].set_name("col8");
    expected_metadata.column_metadata[9].set_name("col9");
    // Write to parquet buffer
    cudf::io::parquet_writer_options out_opts =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info{&parquet_buffer}, table)
        .metadata(std::move(expected_metadata))
        .row_group_size_rows(num_rows)
        .max_page_size_rows(page_size_for_ordered_tests)
        .compression(cudf::io::compression_type::AUTO)
        .dictionary_policy(cudf::io::dictionary_policy::ALWAYS)
        .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN);
    cudf::io::write_parquet(out_opts);
  }

  // Filtering AST - table[0] >= 100
  auto constexpr num_filter_columns = 1;
  auto literal_value                = cudf::numeric_scalar<uint32_t>(100);
  auto literal                      = cudf::ast::literal(literal_value);
  auto col_ref_0                    = cudf::ast::column_name_reference("col0");
  auto filter_expression =
    cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, col_ref_0, literal);

  auto stream     = cudf::get_default_stream();
  auto mr         = cudf::get_current_device_resource_ref();
  auto aligned_mr = rmm::mr::aligned_resource_adaptor<rmm::mr::device_memory_resource>(
    cudf::get_current_device_resource(), bloom_filter_alignment);

  // Read parquet using the hybrid scan reader
  auto [read_filter_table, read_payload_table, read_filter_meta, read_payload_meta, row_mask] =
    hybrid_scan(parquet_buffer, filter_expression, num_filter_columns, {}, stream, mr, aligned_mr);

  // Read parquet using the chunked hybrid scan reader
  auto [read_filter_table_chunked,
        read_payload_table_chunked,
        read_filter_meta_chunked,
        read_payload_meta_chunked,
        row_mask_chunked] =
    chunked_hybrid_scan(
      parquet_buffer, filter_expression, num_filter_columns, {}, stream, mr, aligned_mr);

  CUDF_EXPECTS(read_filter_table->num_rows() == read_payload_table->num_rows(),
               "Filter and payload tables should have the same number of rows");

  // Check equivalence (equal without checking nullability) with the parquet file read with the
  // original reader
  {
    cudf::io::parquet_reader_options const options =
      cudf::io::parquet_reader_options::builder(
        cudf::io::source_info(cudf::host_span<char>(parquet_buffer.data(), parquet_buffer.size())))
        .filter(filter_expression);
    auto [expected_tbl, expected_meta] = cudf::io::read_parquet(options, stream);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected_tbl->select({0}), read_filter_table->view());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected_tbl->select({0}),
                                       read_filter_table_chunked->view());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected_tbl->select({1, 2, 3, 4, 5, 6, 7, 8, 9}),
                                       read_payload_table->view());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected_tbl->select({1, 2, 3, 4, 5, 6, 7, 8, 9}),
                                       read_payload_table_chunked->view());
  }
}
