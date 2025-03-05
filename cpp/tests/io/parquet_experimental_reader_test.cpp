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
#include <limits>

// Base test fixture for tests
struct ParquetExperimentalReaderTest : public cudf::test::BaseFixture {};

namespace {

auto get_footer_bytes(cudf::host_span<uint8_t const> buffer)
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

auto create_parquet_with_stats(bool is_concatenated = false)
{
  auto col0 = testdata::ascending<uint32_t>();
  auto col1 = testdata::descending<int64_t>();
  auto col2 = testdata::unordered<double>();

  auto expected = table_view{{col0, col1, col2}};
  auto table    = std::unique_ptr<cudf::table>();
  if (is_concatenated) {
    table    = cudf::concatenate(std::vector<table_view>(5, expected));
    expected = table->view();
  }

  cudf::io::table_input_metadata expected_metadata(expected);
  expected_metadata.column_metadata[0].set_name("col_uint32");
  expected_metadata.column_metadata[1].set_name("col_int64");
  expected_metadata.column_metadata[2].set_name("col_double");

  std::vector<char> buffer;
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{&buffer}, expected)
      .metadata(std::move(expected_metadata))
      .row_group_size_rows(5000)
      .max_page_size_rows(1000)
      .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN);

  if (is_concatenated) {
    out_opts.set_row_group_size_rows(20000);
    out_opts.set_max_page_size_rows(5000);
  }

  cudf::io::write_parquet(out_opts);

  auto columns = std::vector<std::unique_ptr<column>>{};
  if (not is_concatenated) {
    columns.push_back(col0.release());
    columns.push_back(col1.release());
    columns.push_back(col2.release());
  } else {
    columns = table->release();
  }
  return std::pair{cudf::table{std::move(columns)}, buffer};
}

}  // namespace

TEST_F(ParquetExperimentalReaderTest, FilterRowGroupsWithStats)
{
  srand(31337);
  auto [table, buffer] = create_parquet_with_stats();

  // Filtering AST - table[0] < 150
  auto literal_value     = cudf::numeric_scalar<uint32_t>(100);
  auto literal           = cudf::ast::literal(literal_value);
  auto col_ref_0         = cudf::ast::column_name_reference("col_uint32");
  auto filter_expression = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, literal);

  cudf::io::parquet_reader_options options =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info(buffer.data(), buffer.size()))
      .filter(filter_expression);

  auto const stream = cudf::get_default_stream();

  auto const footer_bytes = get_footer_bytes(
    cudf::host_span<uint8_t const>(reinterpret_cast<uint8_t const*>(buffer.data()), buffer.size()));
  auto const page_index_bytes = cudf::host_span<uint8_t const>(  // nullptr, 0);
    reinterpret_cast<uint8_t const*>(buffer.data()),
    buffer.size());
  auto const reader =
    cudf::experimental::io::make_hybrid_scan_reader(footer_bytes, page_index_bytes, options);

  auto input_row_groups = cudf::experimental::io::get_valid_row_groups(reader, options);

  std::cout << "Num input row groups = " << input_row_groups.size() << std::endl;

  auto stats_filtered_row_groups =
    cudf::experimental::io::filter_row_groups_with_stats(reader, input_row_groups, options, stream);

  auto filtered_row_groups = cudf::host_span<cudf::size_type>(stats_filtered_row_groups);

  auto [bloom_filter_bytes, dict_page_bytes] =
    cudf::experimental::io::get_secondary_filters(reader, stats_filtered_row_groups, options);

  std::vector<cudf::size_type> bloom_filtered_row_groups;
  bloom_filtered_row_groups.reserve(filtered_row_groups.size());
  if (bloom_filter_bytes.size()) {
    // TODO: Read bloom filter data
    std::vector<rmm::device_buffer> bloom_filter_data;
    bloom_filtered_row_groups = cudf::experimental::io::filter_row_groups_with_bloom_filters(
      reader, bloom_filter_data, stats_filtered_row_groups, options, stream);
    filtered_row_groups = cudf::host_span<cudf::size_type>(bloom_filtered_row_groups);
  }

  std::cout << "Num filtered row groups = " << filtered_row_groups.size() << std::endl;

  auto mr = cudf::get_current_device_resource_ref();

  auto row_map = cudf::experimental::io::filter_data_pages_with_stats(
    reader, stats_filtered_row_groups, options, stream, mr);

  auto data_pages_bytes = cudf::experimental::io::get_filter_columns_data_pages(
    reader, row_map->view(), filtered_row_groups, options, stream);

  EXPECT_EQ(data_pages_bytes.size(), table.num_columns());
}

TEST_F(ParquetExperimentalReaderTest, FilterPagesWithStats)
{
  srand(31337);
  auto [table, buffer] = create_parquet_with_stats(true);

  // Filtering AST - table[0] < 150
  auto literal_value     = cudf::numeric_scalar<uint32_t>(100);
  auto literal           = cudf::ast::literal(literal_value);
  auto col_ref_0         = cudf::ast::column_name_reference("col_uint32");
  auto filter_expression = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, literal);

  cudf::io::parquet_reader_options options =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info(buffer.data(), buffer.size()))
      .filter(filter_expression);

  auto const stream = cudf::get_default_stream();

  auto const footer_bytes = get_footer_bytes(
    cudf::host_span<uint8_t const>(reinterpret_cast<uint8_t const*>(buffer.data()), buffer.size()));
  auto const page_index_bytes = cudf::host_span<uint8_t const>(  // nullptr, 0);
    reinterpret_cast<uint8_t const*>(buffer.data()),
    buffer.size());
  auto const reader =
    cudf::experimental::io::make_hybrid_scan_reader(footer_bytes, page_index_bytes, options);

  auto input_row_groups = cudf::experimental::io::get_valid_row_groups(reader, options);

  std::cout << "Num input row groups = " << input_row_groups.size() << std::endl;

  auto stats_filtered_row_groups =
    cudf::experimental::io::filter_row_groups_with_stats(reader, input_row_groups, options, stream);

  auto filtered_row_groups = cudf::host_span<cudf::size_type>(stats_filtered_row_groups);

  auto [bloom_filter_bytes, dict_page_bytes] =
    cudf::experimental::io::get_secondary_filters(reader, stats_filtered_row_groups, options);

  std::vector<cudf::size_type> bloom_filtered_row_groups;
  bloom_filtered_row_groups.reserve(filtered_row_groups.size());
  if (bloom_filter_bytes.size()) {
    // TODO: Read bloom filter data
    std::vector<rmm::device_buffer> bloom_filter_data;
    bloom_filtered_row_groups = cudf::experimental::io::filter_row_groups_with_bloom_filters(
      reader, bloom_filter_data, stats_filtered_row_groups, options, stream);
    filtered_row_groups = cudf::host_span<cudf::size_type>(bloom_filtered_row_groups);
  }

  std::cout << "Num filtered row groups = " << filtered_row_groups.size() << std::endl;

  auto mr = cudf::get_current_device_resource_ref();

  auto row_map = cudf::experimental::io::filter_data_pages_with_stats(
    reader, stats_filtered_row_groups, options, stream, mr);

  auto data_pages_bytes = cudf::experimental::io::get_filter_columns_data_pages(
    reader, row_map->view(), filtered_row_groups, options, stream);

  EXPECT_EQ(data_pages_bytes.size(), table.num_columns());
  EXPECT_EQ(data_pages_bytes.front().size(), 10);
}