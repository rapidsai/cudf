/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hybrid_scan_common.hpp"

#include <cudf_test/base_fixture.hpp>

#include <cudf/ast/expressions.hpp>
#include <cudf/copying.hpp>
#include <cudf/io/experimental/hybrid_scan_multifile.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace {

/**
 * @brief Struct to hold multifile datasources, and footer buffers along with their byte spans
 */
struct multifile_inputs {
  std::vector<std::unique_ptr<cudf::io::datasource>> datasources;
  std::vector<std::unique_ptr<cudf::io::datasource::buffer>> footer_buffers;
  std::vector<cudf::host_span<uint8_t const>> footer_byte_spans;
};

template <typename Buffers>
multifile_inputs build_multifile_inputs(Buffers const& file_buffers)
{
  multifile_inputs out;
  out.datasources.reserve(file_buffers.size());
  out.footer_buffers.reserve(file_buffers.size());
  out.footer_byte_spans.reserve(file_buffers.size());
  for (auto const& buf : file_buffers) {
    out.datasources.emplace_back(cudf::io::datasource::create(cudf::host_span<std::byte const>(
      reinterpret_cast<std::byte const*>(buf.data()), buf.size())));
    out.footer_buffers.emplace_back(
      cudf::io::parquet::fetch_footer_to_host(*out.datasources.back()));
    out.footer_byte_spans.emplace_back(*out.footer_buffers.back());
  }
  return out;
}

/**
 * @brief Creates a parquet buffer with zero-rows and same schema as table from
 * `create_parquet_with_stats`
 */
template <typename T>
std::vector<char> create_empty_parquet_with_stats()
{
  auto const non_empty = std::get<0>(create_parquet_with_stats<T, 1>());
  auto const empty     = cudf::empty_like(non_empty->view());

  cudf::io::table_input_metadata output_metadata(empty->view());
  output_metadata.column_metadata[0].set_name("col0");
  output_metadata.column_metadata[1].set_name("col1");
  output_metadata.column_metadata[2].set_name("col2");

  std::vector<char> buffer;
  auto out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{&buffer}, empty->view())
      .metadata(std::move(output_metadata))
      .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN)
      .build();
  cudf::io::write_parquet(out_opts);
  return buffer;
}

}  // namespace

struct HybridScanMultifileFiltersTest : public cudf::test::BaseFixture {};

TEST_F(HybridScanMultifileFiltersTest, Metadata)
{
  using T = cudf::timestamp_ms;

  // Create two parquet sources, each with 4 row groups and 5000 rows per row
  // group
  auto constexpr rows_per_row_group = page_size_for_ordered_tests;
  auto constexpr num_sources        = 2;

  // Build sources with different seeds
  std::vector<std::vector<char>> file_buffers;
  file_buffers.reserve(num_sources);
  auto constexpr num_concat = 1;
  srand(0xbad);
  file_buffers.emplace_back(std::get<1>(create_parquet_with_stats<T, num_concat>()));
  srand(0xf00d);
  file_buffers.emplace_back(std::get<1>(create_parquet_with_stats<T, num_concat>()));

  // Filtering AST - col0 < 100
  auto literal_value =
    cudf::timestamp_scalar<T>(T(typename T::duration(100)), true, cudf::get_default_stream());
  auto literal           = cudf::ast::literal(literal_value);
  auto col_ref_0         = cudf::ast::column_name_reference("col0");
  auto filter_expression = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, literal);

  // Construct reader from footer bytes
  auto inputs = build_multifile_inputs(file_buffers);

  cudf::io::parquet_reader_options options =
    cudf::io::parquet_reader_options::builder().filter(filter_expression).build();
  auto const reader = std::make_unique<cudf::io::parquet::experimental::hybrid_scan_multifile>(
    cudf::host_span<cudf::host_span<uint8_t const> const>{inputs.footer_byte_spans}, options);

  // Get parquet metadata and check
  auto parquet_metadata = reader->parquet_metadatas();
  ASSERT_EQ(parquet_metadata.size(), num_sources);
  for (auto const& meta : parquet_metadata) {
    ASSERT_FALSE(meta.row_groups.empty());
    EXPECT_FALSE(meta.row_groups[0].columns[0].offset_index.has_value());
    EXPECT_FALSE(meta.row_groups[0].columns[0].column_index.has_value());
  }

  // Setup page index
  auto const page_index_byte_ranges = reader->page_index_byte_ranges();
  ASSERT_EQ(page_index_byte_ranges.size(), num_sources);
  EXPECT_TRUE(std::all_of(page_index_byte_ranges.begin(),
                          page_index_byte_ranges.end(),
                          [](auto const& range) { return not range.is_empty(); }));

  std::vector<std::unique_ptr<cudf::io::datasource::buffer>> page_index_buffers;
  std::vector<cudf::host_span<uint8_t const>> page_index_byte_spans;
  page_index_buffers.reserve(num_sources);
  page_index_byte_spans.reserve(num_sources);

  auto iter = cuda::zip_iterator(page_index_byte_ranges.begin(), inputs.datasources.begin());
  std::for_each(iter, iter + num_sources, [&](auto const& pair) {
    auto const& [pgidx_byte_range, datasource] = pair;
    page_index_buffers.emplace_back(
      cudf::io::parquet::fetch_page_index_to_host(*datasource, pgidx_byte_range));
    page_index_byte_spans.emplace_back(*page_index_buffers.back());
  });

  reader->setup_page_indexes(
    cudf::host_span<cudf::host_span<uint8_t const> const>{page_index_byte_spans});

  // Check if page index is now present in each parquet metadata
  parquet_metadata = reader->parquet_metadatas();
  for (auto const& meta : parquet_metadata) {
    EXPECT_TRUE(meta.row_groups[0].columns[0].offset_index.has_value());
    EXPECT_TRUE(meta.row_groups[0].columns[0].column_index.has_value());
  }

  // Check all row groups
  auto input_row_group_indices = reader->all_row_groups(options);
  ASSERT_EQ(input_row_group_indices.size(), num_sources);
  EXPECT_TRUE(std::all_of(
    input_row_group_indices.begin(), input_row_group_indices.end(), [](auto const& rgs) {
      return rgs == (std::vector<cudf::size_type>{0, 1, 2, 3});
    }));

  // Set explicit row groups (per-source) via options
  options.set_row_groups({{0, 1}, {2, 3}});
  input_row_group_indices = reader->all_row_groups(options);

  // Check if the row groups are set correctly
  ASSERT_EQ(input_row_group_indices.size(), num_sources);
  EXPECT_EQ(input_row_group_indices[0], (std::vector<cudf::size_type>{0, 1}));
  EXPECT_EQ(input_row_group_indices[1], (std::vector<cudf::size_type>{2, 3}));
  EXPECT_EQ(reader->total_rows_in_row_groups(input_row_group_indices),
            2 * rows_per_row_group * num_sources);

  // Construct a new reader from a span of existing FileMetaData
  auto const reader_with_existing_metadata =
    std::make_unique<cudf::io::parquet::experimental::hybrid_scan_multifile>(
      cudf::host_span<cudf::io::parquet::FileMetaData const>{parquet_metadata}, options);

  // Check if the new metadata is the same as the existing one
  auto const new_metadata = reader_with_existing_metadata->parquet_metadatas();
  ASSERT_EQ(new_metadata.size(), num_sources);
  EXPECT_TRUE(std::all_of(new_metadata.begin(), new_metadata.end(), [&](auto const& meta) {
    return meta.row_groups.size() == parquet_metadata.front().row_groups.size();
  }));
}

TEST_F(HybridScanMultifileFiltersTest, EmptySource)
{
  using T = uint32_t;

  srand(0xc0ffee);

  // Create two parquet source. First one with non-zero rows and the second one with zero rows.
  auto constexpr num_sources = 2;
  std::vector<std::vector<char>> file_buffers;
  file_buffers.reserve(num_sources);
  file_buffers.emplace_back(std::get<1>(create_parquet_with_stats<T, 1>()));
  file_buffers.emplace_back(create_empty_parquet_with_stats<T>());

  auto inputs = build_multifile_inputs(file_buffers);

  cudf::io::parquet_reader_options options = cudf::io::parquet_reader_options::builder().build();
  auto const reader = std::make_unique<cudf::io::parquet::experimental::hybrid_scan_multifile>(
    inputs.footer_byte_spans, options);

  // Check parquet metadata
  auto const parquet_metadata = reader->parquet_metadatas();
  ASSERT_EQ(parquet_metadata.size(), num_sources);
  EXPECT_FALSE(parquet_metadata.front().row_groups.empty());
  EXPECT_TRUE(parquet_metadata.back().row_groups.empty());

  // Check row group indices
  auto const all_rgs = reader->all_row_groups(options);
  ASSERT_EQ(all_rgs.size(), num_sources);
  EXPECT_EQ(all_rgs.front(), (std::vector<cudf::size_type>{0, 1, 2, 3}));
  EXPECT_TRUE(all_rgs.back().empty());

  // Check page index byte ranges
  auto const page_index_byte_ranges = reader->page_index_byte_ranges();
  ASSERT_EQ(page_index_byte_ranges.size(), num_sources);
  EXPECT_FALSE(page_index_byte_ranges.front().is_empty());
  EXPECT_TRUE(page_index_byte_ranges.back().is_empty());
}

TEST_F(HybridScanMultifileFiltersTest, ErrorFilterRowGroupsWithByteRanges)
{
  using T                    = uint32_t;
  auto constexpr num_sources = 2;
  srand(0xb47e);

  std::vector<std::vector<char>> file_buffers;
  file_buffers.reserve(num_sources);
  file_buffers.emplace_back(std::get<1>(create_parquet_with_stats<T, 1>()));
  file_buffers.emplace_back(std::get<1>(create_parquet_with_stats<T, 1>()));

  auto inputs = build_multifile_inputs(file_buffers);

  // Setting `skip_bytes` or `num_bytes` is ambiguous when reading multiple sources. The reader is
  // expected to throw an exception if row groups are filtered using byte range in this case.
  {
    auto const options = cudf::io::parquet_reader_options::builder().skip_bytes(1000).build();
    auto const reader  = std::make_unique<cudf::io::parquet::experimental::hybrid_scan_multifile>(
      inputs.footer_byte_spans, options);
    auto const row_group_indices = reader->all_row_groups(options);
    ASSERT_EQ(row_group_indices.size(), num_sources);
    EXPECT_THROW(
      std::ignore = reader->filter_row_groups_with_byte_range(row_group_indices, options),
      std::invalid_argument);
  }
  {
    auto const options = cudf::io::parquet_reader_options::builder().num_bytes(1000).build();
    auto const reader  = std::make_unique<cudf::io::parquet::experimental::hybrid_scan_multifile>(
      inputs.footer_byte_spans, options);
    auto const row_group_indices = reader->all_row_groups(options);
    ASSERT_EQ(row_group_indices.size(), num_sources);
    EXPECT_THROW(
      std::ignore = reader->filter_row_groups_with_byte_range(row_group_indices, options),
      std::invalid_argument);
  }
}

TEST_F(HybridScanMultifileFiltersTest, FilterRowGroupsWithStats)
{
  using T                           = cudf::duration_ms;
  auto constexpr num_sources        = 2;
  auto constexpr rows_per_row_group = page_size_for_ordered_tests;

  // Two sources, each with 4 row groups and ascending strings in col2
  std::vector<std::vector<char>> file_buffers;
  file_buffers.reserve(num_sources);
  srand(0xc001);
  file_buffers.emplace_back(std::get<1>(create_parquet_with_stats<T, 1, false>()));
  srand(0xbeef);
  file_buffers.emplace_back(std::get<1>(create_parquet_with_stats<T, 1, false>()));

  auto inputs = build_multifile_inputs(file_buffers);

  // Filter - col0 < 50 and col2 > "000010000"
  auto literal_value0 = cudf::duration_scalar<T>(T::rep(50), true, cudf::get_default_stream());
  auto literal0       = cudf::ast::literal(literal_value0);
  auto col_ref0       = cudf::ast::column_reference(0);
  auto filter1        = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref0, literal0);

  auto literal_value2 = cudf::string_scalar("000010000", true, cudf::get_default_stream());
  auto literal2       = cudf::ast::literal(literal_value2);
  auto col_ref2       = cudf::ast::column_reference(2);
  auto filter2        = cudf::ast::operation(cudf::ast::ast_operator::GREATER, literal2, col_ref2);

  auto filter_expression =
    cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, filter1, filter2);

  auto options      = cudf::io::parquet_reader_options::builder().filter(filter_expression).build();
  auto const reader = std::make_unique<cudf::io::parquet::experimental::hybrid_scan_multifile>(
    inputs.footer_byte_spans, options);

  // Each source has 4 row groups (20000 rows / 5000 rows per row group)
  auto input_row_group_indices = reader->all_row_groups(options);
  ASSERT_EQ(input_row_group_indices.size(), num_sources);
  EXPECT_EQ(reader->total_rows_in_row_groups(input_row_group_indices),
            num_sources * 4 * rows_per_row_group);

  // Each source prunes down to a single surviving row group
  auto stats_filtered = reader->filter_row_groups_with_stats(
    input_row_group_indices, options, cudf::get_default_stream());
  ASSERT_EQ(stats_filtered.size(), num_sources);
  for (std::size_t i = 0; i < stats_filtered.size(); ++i) {
    EXPECT_EQ(stats_filtered[i].size(), 1) << "Source index: " << i;
  }
  EXPECT_EQ(reader->total_rows_in_row_groups(stats_filtered), num_sources * rows_per_row_group);

  // Custom per-source indices that prune all row groups via stats, including an empty source
  input_row_group_indices = {{1, 2}, {}};
  stats_filtered          = reader->filter_row_groups_with_stats(
    input_row_group_indices, options, cudf::get_default_stream());
  ASSERT_EQ(stats_filtered.size(), num_sources);
  EXPECT_TRUE(stats_filtered.front().empty());
  EXPECT_TRUE(stats_filtered.back().empty());
}

TEST_F(HybridScanMultifileFiltersTest, FilterRowGroupsWithBloomFilters)
{
  using T                    = uint32_t;
  auto constexpr num_sources = 2;
  auto const stream          = cudf::get_default_stream();

  // Two sources, each with 4 row groups
  std::vector<std::vector<char>> file_buffers;
  file_buffers.reserve(num_sources);
  srand(0xb100);
  file_buffers.emplace_back(std::get<1>(create_parquet_with_stats<T, 1>()));
  srand(0x600d);
  file_buffers.emplace_back(std::get<1>(create_parquet_with_stats<T, 1>()));

  auto inputs = build_multifile_inputs(file_buffers);

  // An equality predicate makes col0 eligible for bloom filtering. cuDF's Parquet writer does not
  // emit bloom filters, so the per-source bloom byte ranges come back empty (same as single-file).
  {
    auto literal_value = cudf::numeric_scalar<T>(T{42}, true, stream);
    auto literal       = cudf::ast::literal(literal_value);
    auto col_ref       = cudf::ast::column_name_reference("col0");
    auto filter        = cudf::ast::operation(cudf::ast::ast_operator::EQUAL, col_ref, literal);

    auto options      = cudf::io::parquet_reader_options::builder().filter(filter).build();
    auto const reader = std::make_unique<cudf::io::parquet::experimental::hybrid_scan_multifile>(
      inputs.footer_byte_spans, options);

    auto const input_row_group_indices = reader->all_row_groups(options);
    ASSERT_EQ(input_row_group_indices.size(), num_sources);

    auto const bloom_byte_ranges =
      reader->bloom_filters_byte_ranges(input_row_group_indices, options);
    EXPECT_TRUE(bloom_byte_ranges.empty());
  }

  // Without any bloom-eligible (equality) predicate, bloom filtering is a no-op: the reader returns
  // the input row groups unchanged, one inner vector per source. Validates the multifile bloom
  // filter API delegation and per-source output shape.
  {
    auto literal_value = cudf::numeric_scalar<T>(T{50}, true, stream);
    auto literal       = cudf::ast::literal(literal_value);
    auto col_ref       = cudf::ast::column_name_reference("col0");
    auto filter        = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref, literal);

    auto options      = cudf::io::parquet_reader_options::builder().filter(filter).build();
    auto const reader = std::make_unique<cudf::io::parquet::experimental::hybrid_scan_multifile>(
      inputs.footer_byte_spans, options);

    auto const input_row_group_indices = reader->all_row_groups(options);
    ASSERT_EQ(input_row_group_indices.size(), num_sources);

    auto const bloom_filtered =
      reader->filter_row_groups_with_bloom_filters({}, input_row_group_indices, options, stream);
    EXPECT_EQ(bloom_filtered, input_row_group_indices);
  }
}
