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

#include "cudf/io/text/byte_range_info.hpp"
#include "cudf/utilities/default_stream.hpp"
#include "cudf/utilities/memory_resource.hpp"
#include "cudf/utilities/span.hpp"
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
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>

#include <src/io/parquet/parquet_gpu.hpp>

#include <array>

// Base test fixture for tests
struct ParquetExperimentalReaderTest : public cudf::test::BaseFixture {};

namespace {

auto fetch_footer_bytes(cudf::host_span<uint8_t const> buffer)
{
  /**
   * @brief Struct that describes the Parquet file data header
   */
  struct file_header_s {
    uint32_t magic;
  };

  /**
   * @brief Struct that describes the Parquet file data postscript
   */
  struct file_ender_s {
    uint32_t footer_len;
    uint32_t magic;
  };

  constexpr auto header_len = sizeof(file_header_s);
  constexpr auto ender_len  = sizeof(file_ender_s);
  auto const len            = buffer.size();

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

auto fetch_page_index_bytes(cudf::host_span<uint8_t const> buffer,
                            cudf::io::text::byte_range_info const page_index_bytes)
{
  return cudf::host_span<uint8_t const>(
    reinterpret_cast<uint8_t const*>(buffer.data()) + page_index_bytes.offset(),
    page_index_bytes.size());
}

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

  auto columns = std::vector<std::unique_ptr<column>>{};
  if constexpr (NumTableConcats == 1) {
    columns.push_back(col0.release());
    columns.push_back(col1.release());
    columns.push_back(col2.release());
  } else {
    columns = table->release();
  }
  return std::pair{cudf::table{std::move(columns)}, buffer};
}

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

std::vector<rmm::device_buffer> fetch_column_chunk_buffers(
  cudf::host_span<uint8_t const> host_buffer,
  cudf::host_span<cudf::io::text::byte_range_info const> byte_ranges,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  std::vector<rmm::device_buffer> column_chunk_buffers{};
  column_chunk_buffers.reserve(byte_ranges.size());

  std::transform(
    byte_ranges.begin(),
    byte_ranges.end(),
    std::back_inserter(column_chunk_buffers),
    [&](auto const& byte_range) {
      auto const chunk_offset = host_buffer.data() + byte_range.offset();
      auto const chunk_size   = byte_range.size();
      auto chunk_buffer       = rmm::device_buffer(chunk_size, stream, mr);
      CUDF_CUDA_TRY(cudaMemcpyAsync(
        chunk_buffer.data(), chunk_offset, chunk_size, cudaMemcpyHostToDevice, stream.value()));
      return chunk_buffer;
    });

  stream.synchronize_no_throw();
  return column_chunk_buffers;
}

/**
 * @brief Read parquet file with the hybrid scan reader
 *
 * @param buffer Buffer containing the parquet file
 * @param num_filter_columns Number of filter columns
 * @param num_payload_columns Number of payload columns
 * @param filter_expression Filter expression
 * @param stream CUDA stream for hybrid scan reader
 * @param mr Device memory resource
 *
 * @return Tuple of filter table, payload table, filter metadata, payload metadata, and the final
 *         row validity column
 */
auto hybrid_scan(std::vector<char>& buffer,
                 cudf::size_type const num_filter_columns,
                 cudf::size_type const num_payload_columns,
                 cudf::ast::operation const& filter_expression,
                 rmm::cuda_stream_view stream,
                 rmm::device_async_resource_ref mr)
{
  // Create reader options with empty source info
  cudf::io::parquet_reader_options const options =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info(nullptr, 0))
      .filter(filter_expression);

  // Input file buffer span
  auto const file_buffer_span =
    cudf::host_span<uint8_t const>(reinterpret_cast<uint8_t const*>(buffer.data()), buffer.size());

  // Fetch footer and page index bytes from the buffer.
  auto const footer_buffer = fetch_footer_bytes(file_buffer_span);

  // Create hybrid scan reader with footer bytes - API # 1
  auto const reader = cudf::experimental::io::make_hybrid_scan_reader(footer_buffer, options);

  // Get Parquet file metadata from the reader - API # 2
  [[maybe_unused]] auto const parquet_metadata =
    cudf::experimental::io::get_parquet_metadata(reader);

  // Get page index byte range from the reader - API # 3
  auto const page_index_byte_range = cudf::experimental::io::get_page_index_bytes(reader);

  // Fetch page index bytes from the input buffer
  auto const page_index_buffer = fetch_page_index_bytes(file_buffer_span, page_index_byte_range);

  // Setup page index - API # 4
  cudf::experimental::io::setup_page_index(reader, page_index_buffer);

  // Get all row groups from the reader - API # 5
  auto input_row_group_indices = cudf::experimental::io::get_all_row_groups(reader, options);

  // Span to track current row group indices
  auto current_row_group_indices = cudf::host_span<cudf::size_type>(input_row_group_indices);

  // Filter row groups with stats - API # 6
  auto stats_filtered_row_group_indices = cudf::experimental::io::filter_row_groups_with_stats(
    reader, current_row_group_indices, options, stream);

  // Update current row group indices
  current_row_group_indices = cudf::host_span<cudf::size_type>(stats_filtered_row_group_indices);

  // Get bloom filter and dictionary page byte ranges from the reader - API # 7
  auto [bloom_filter_byte_ranges, dict_page_byte_ranges] =
    cudf::experimental::io::get_secondary_filters(reader, current_row_group_indices, options);

  // If we have dictionary page byte ranges, filter row groups with dictionary pages - API # 8
  std::vector<cudf::size_type> dictionary_page_filtered_row_group_indices;
  dictionary_page_filtered_row_group_indices.reserve(current_row_group_indices.size());
  if (dict_page_byte_ranges.size()) {
    // Fetch dictionary page buffers from the input file buffer
    std::vector<rmm::device_buffer> dictionary_page_buffers =
      fetch_byte_ranges(file_buffer_span, dict_page_byte_ranges, stream, mr);

    // NOT YET IMPLEMENTED - Filter row groups with dictionary pages
    dictionary_page_filtered_row_group_indices =
      cudf::experimental::io::filter_row_groups_with_dictionary_pages(
        reader, dictionary_page_buffers, current_row_group_indices, options, stream);
    // Update current row group indices
    current_row_group_indices =
      cudf::host_span<cudf::size_type>(dictionary_page_filtered_row_group_indices);
  }

  // If we have bloom filter byte ranges, filter row groups with bloom filters - API # 9
  std::vector<cudf::size_type> bloom_filtered_row_group_indices;
  bloom_filtered_row_group_indices.reserve(current_row_group_indices.size());
  if (bloom_filter_byte_ranges.size()) {
    // Fetch bloom filter data from the input file buffer
    std::vector<rmm::device_buffer> bloom_filter_data =
      fetch_byte_ranges(file_buffer_span, bloom_filter_byte_ranges, stream, mr);

    // Filter row groups with bloom filters
    bloom_filtered_row_group_indices = cudf::experimental::io::filter_row_groups_with_bloom_filters(
      reader, bloom_filter_data, current_row_group_indices, options, stream);
    // Update current row group indices
    current_row_group_indices = cudf::host_span<cudf::size_type>(bloom_filtered_row_group_indices);
  }

  // Filter data pages with `PageIndex` stats - API # 10
  auto [row_mask, data_page_mask] = cudf::experimental::io::filter_data_pages_with_stats(
    reader, current_row_group_indices, options, stream, mr);

  EXPECT_EQ(data_page_mask.size(), num_filter_columns);

  // Get column chunk byte ranges from the reader - API # 11
  auto const filter_column_chunk_byte_ranges =
    cudf::experimental::io::get_filter_column_chunk_byte_ranges(
      reader, current_row_group_indices, options);

  // Fetch column chunk device buffers from the input buffer
  auto filter_column_chunk_buffers =
    fetch_column_chunk_buffers(file_buffer_span, filter_column_chunk_byte_ranges, stream, mr);

  // Materialize the table with only the filter columns - API # 12
  auto [filter_table, filter_metadata] =
    cudf::experimental::io::materialize_filter_columns(reader,
                                                       data_page_mask,
                                                       current_row_group_indices,
                                                       std::move(filter_column_chunk_buffers),
                                                       row_mask->mutable_view(),
                                                       options,
                                                       stream);

  // Get column chunk byte ranges from the reader - API # 13
  auto const payload_column_chunk_byte_ranges =
    cudf::experimental::io::get_payload_column_chunk_byte_ranges(
      reader, current_row_group_indices, options);

  // Fetch column chunk device buffers from the input buffer
  [[maybe_unused]] auto payload_column_chunk_buffers =
    fetch_column_chunk_buffers(file_buffer_span, payload_column_chunk_byte_ranges, stream, mr);

  // Materialize the table with only the payload columns - API # 14
  [[maybe_unused]] auto [payload_table, payload_metadata] =
    cudf::experimental::io::materialize_payload_columns(reader,
                                                        current_row_group_indices,
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
    hybrid_scan(buffer,
                num_filter_columns,
                written_table.num_columns() - num_filter_columns,
                filter_expression,
                stream,
                mr);

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
    hybrid_scan(buffer,
                num_filter_columns,
                written_table.num_columns() - num_filter_columns,
                filter_expression,
                stream,
                mr);

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

    auto predicate = cudf::compute_column(written_table, filter_expression);
    EXPECT_EQ(predicate->view().type().id(), cudf::type_id::BOOL8)
      << "Predicate filter should return a boolean";
    auto expected = cudf::apply_boolean_mask(written_table, *predicate);
    // Check equivalence as the nullability between columns may be different
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected->select({0}), read_filter_table->view());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected->select({1, 2}), read_payload_table->view());
  }
}
