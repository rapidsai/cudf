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
#include "rmm/device_uvector.hpp"

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
  auto col2 = testdata::ascending<cudf::string_view>();

  auto expected = table_view{{col0, col1, col2}};
  auto table    = std::unique_ptr<cudf::table>();
  if (is_concatenated) {
    table    = cudf::concatenate(std::vector<table_view>(2, expected));
    expected = table->view();
  }

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

std::vector<rmm::device_buffer> fetch_column_chunk_buffers(
  cudf::host_span<char const> host_buffer,
  cudf::host_span<cudf::io::text::byte_range_info const> byte_ranges,
  cudf::host_span<cudf::size_type const> /* chunk_source_map */,
  rmm::cuda_stream_view stream = cudf::get_default_stream())
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
      auto chunk_buffer       = rmm::device_buffer(chunk_size, stream);
      CUDF_CUDA_TRY(cudaMemcpyAsync(
        chunk_buffer.data(), chunk_offset, chunk_size, cudaMemcpyHostToDevice, stream.value()));
      return chunk_buffer;
    });

  stream.synchronize_no_throw();
  return column_chunk_buffers;
}

}  // namespace

TEST_F(ParquetExperimentalReaderTest, FilterWithStats)
{
  srand(31337);

  auto const hybrid_scan = [&](bool is_testing_pages) {
    auto [written_table, buffer] = create_parquet_with_stats(is_testing_pages);

    // Filtering AST - table[0] < 100
    auto literal_value = cudf::numeric_scalar<uint32_t>(100);
    auto literal       = cudf::ast::literal(literal_value);
    auto col_ref_0     = cudf::ast::column_name_reference("col_uint32");
    auto filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, literal);

    cudf::io::parquet_reader_options const options =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info(buffer.data(), buffer.size()))
        .filter(filter_expression);

    auto const stream = cudf::get_default_stream();

    auto const footer_bytes     = get_footer_bytes(cudf::host_span<uint8_t const>(
      reinterpret_cast<uint8_t const*>(buffer.data()), buffer.size()));
    auto const page_index_bytes = cudf::host_span<uint8_t const>(  // nullptr, 0);
      reinterpret_cast<uint8_t const*>(buffer.data()),
      buffer.size());
    auto const reader =
      cudf::experimental::io::make_hybrid_scan_reader(footer_bytes, page_index_bytes, options);

    auto input_row_groups = cudf::experimental::io::get_valid_row_groups(reader, options);

    auto stats_filtered_row_groups = cudf::experimental::io::filter_row_groups_with_stats(
      reader, input_row_groups, options, stream);

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

    auto mr = cudf::get_current_device_resource_ref();

    auto [row_validity, data_page_validity] = cudf::experimental::io::filter_data_pages_with_stats(
      reader, stats_filtered_row_groups, options, stream, mr);

    EXPECT_EQ(data_page_validity.size(), written_table.num_columns());

    auto [column_chunk_byte_ranges, _] = cudf::experimental::io::get_column_chunk_byte_ranges(
      reader, stats_filtered_row_groups, options);

    auto column_chunk_buffers =
      fetch_column_chunk_buffers(buffer, column_chunk_byte_ranges, {}, stream);

    auto const [read_table, read_meta] =
      cudf::experimental::io::materialize_filter_columns(reader,
                                                         data_page_validity,
                                                         stats_filtered_row_groups,
                                                         std::move(column_chunk_buffers),
                                                         row_validity->mutable_view(),
                                                         options,
                                                         stream);

    // Check equality with the parquet file read with the original reader
    {
      auto [expected_tbl, expected_meta] = cudf::io::read_parquet(options, stream);
      CUDF_TEST_EXPECT_TABLES_EQUAL(expected_tbl->view(), read_table->view());
    }

    // Check equivalence with the original table with the applied boolean mask
    {
      auto col_ref_0 = cudf::ast::column_reference(0);
      auto filter_expression =
        cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, literal);

      auto predicate = cudf::compute_column(written_table, filter_expression);
      EXPECT_EQ(predicate->view().type().id(), cudf::type_id::BOOL8)
        << "Predicate filter should return a boolean";
      auto expected = cudf::apply_boolean_mask(written_table, *predicate);
      // Check equivalence as the nullability between columns may be different
      CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected->view(), read_table->view());
    }
  };

  // Only test filtering row groups
  {
    auto constexpr filter_pages = false;
    hybrid_scan(filter_pages);
  }

  // Only test filtering data pages
  {
    auto constexpr filter_pages = true;
    hybrid_scan(filter_pages);
  }
}
