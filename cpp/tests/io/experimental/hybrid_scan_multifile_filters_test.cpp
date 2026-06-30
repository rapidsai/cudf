/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hybrid_scan_common.hpp"
#include "hybrid_scan_multifile_common.hpp"

#include <cudf_test/base_fixture.hpp>

#include <cudf/ast/expressions.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/experimental/hybrid_scan_multifile.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/aligned.hpp>
#include <rmm/mr/aligned_resource_adaptor.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <numeric>
#include <vector>

namespace {

auto constexpr bloom_filter_alignment = rmm::CUDA_ALLOCATION_ALIGNMENT;

/**
 * @brief Copy fixed-width column data to a host vector
 */
template <typename T>
auto host_row_mask_data(cudf::column_view const& column, rmm::cuda_stream_view stream)
{
  return cudf::detail::make_host_vector<T>(
    cudf::device_span<T const>(column.data<T>(), static_cast<size_t>(column.size())), stream);
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

/**
 * @brief Build a scalar literal matching a filter column type
 */
template <typename T>
auto make_scalar(cudf::size_type value, rmm::cuda_stream_view stream)
{
  if constexpr (cudf::is_timestamp<T>()) {
    return cudf::timestamp_scalar<T>(T(typename T::duration(value)), true, stream);
  } else if constexpr (cudf::is_duration<T>()) {
    return cudf::duration_scalar<T>(T(value), true, stream);
  } else {
    return cudf::numeric_scalar<T>(static_cast<T>(value), true, stream);
  }
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
  auto constexpr seed       = 0xbad;
  std::transform(cuda::counting_iterator{seed},
                 cuda::counting_iterator{seed + num_sources},
                 std::back_inserter(file_buffers),
                 [](auto const src_seed) {
                   srand(src_seed);
                   return std::get<1>(create_parquet_with_stats<T, num_concat>());
                 });

  // Filtering AST - col0 < 100
  auto literal_value     = make_scalar<T>(100, cudf::get_default_stream());
  auto literal           = cudf::ast::literal(literal_value);
  auto col_ref_0         = cudf::ast::column_name_reference("col0");
  auto filter_expression = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, literal);

  // Construct reader from footer bytes
  auto inputs = multifile_inputs(build_source_info(file_buffers));

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

  // Create two parquet source. First one with non-zero rows and the second one with zero rows.
  auto constexpr num_sources = 2;
  std::vector<std::vector<char>> file_buffers;
  file_buffers.reserve(num_sources);
  file_buffers.emplace_back(std::get<1>(create_parquet_with_stats<T, 1>()));
  file_buffers.emplace_back(create_empty_parquet_with_stats<T>());

  auto inputs = multifile_inputs(build_source_info(file_buffers));

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

  auto const passes = reader->construct_row_group_passes(all_rgs, 1);
  ASSERT_EQ(passes.size(), all_rgs.front().size());
  for (auto const& pass : passes) {
    ASSERT_EQ(pass.size(), num_sources);
    ASSERT_EQ(pass.front().size(), 1);
    EXPECT_TRUE(pass.back().empty());
  }
}

TEST_F(HybridScanMultifileFiltersTest, RowGroupPasses)
{
  using T = uint32_t;

  auto constexpr num_sources = 2;
  std::vector<std::vector<char>> file_buffers;
  file_buffers.reserve(num_sources);
  std::transform(cuda::counting_iterator(0),
                 cuda::counting_iterator(num_sources),
                 std::back_inserter(file_buffers),
                 [&](auto i) {
                   srand(0xced + i);
                   return std::get<1>(create_parquet_with_stats<T, 1>());
                 });
  auto inputs = multifile_inputs(build_source_info(file_buffers));

  cudf::io::parquet_reader_options options = cudf::io::parquet_reader_options::builder().build();
  auto const reader = std::make_unique<cudf::io::parquet::experimental::hybrid_scan_multifile>(
    inputs.footer_byte_spans, options);

  auto const all_rgs = reader->all_row_groups(options);
  ASSERT_EQ(all_rgs.size(), num_sources);
  EXPECT_TRUE(std::all_of(all_rgs.begin(), all_rgs.end(), [](auto const& rgs) {
    return rgs == (std::vector<cudf::size_type>{0, 1, 2, 3});
  }));

  // Invalid row group indices => throw error
  {
    auto invalid_rgs = all_rgs;
    invalid_rgs.pop_back();
    EXPECT_THROW(static_cast<void>(reader->construct_row_group_passes(invalid_rgs, 0)),
                 std::invalid_argument);
  }

  // Empty row group indices => throw error
  {
    auto const empty_rgs = std::vector<std::vector<cudf::size_type>>(num_sources);
    EXPECT_THROW(static_cast<void>(reader->construct_row_group_passes(empty_rgs, 0)),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(reader->construct_row_group_passes(empty_rgs, 1)),
                 std::invalid_argument);
  }

  // Zero pass read limit => single pass with all row groups
  {
    auto const passes = reader->construct_row_group_passes(all_rgs, 0);
    ASSERT_EQ(passes.size(), 1);
    ASSERT_EQ(passes.front().size(), num_sources);
    EXPECT_EQ(passes.front(), all_rgs);
  }

  // Small pass read limit => each row group in its own pass
  {
    auto const passes = reader->construct_row_group_passes(all_rgs, 1);
    ASSERT_EQ(passes.size(), num_sources * all_rgs.front().size());
    for (auto const& pass : passes) {
      ASSERT_EQ(pass.size(), num_sources);
      auto const pass_num_row_groups =
        std::accumulate(pass.begin(), pass.end(), std::size_t{0}, [](auto sum, auto const& rgs) {
          return sum + rgs.size();
        });
      EXPECT_EQ(pass_num_row_groups, 1);
    }
  }

  // Large pass read limit => multiple passes
  {
    auto const passes = reader->construct_row_group_passes(all_rgs, 10'000);
    ASSERT_GT(passes.size(), 1);
    auto const pass_num_row_groups = [](auto const& pass) {
      return std::accumulate(
        pass.begin(), pass.end(), std::size_t{0}, [](auto sum, auto const& rgs) {
          return sum + rgs.size();
        });
    };
    EXPECT_TRUE(std::any_of(passes.begin(), passes.end(), [&](auto const& pass) {
      return pass_num_row_groups(pass) > 1;
    }));

    auto flattened = std::vector<std::pair<std::size_t, cudf::size_type>>{};
    for (auto const& pass : passes) {
      ASSERT_EQ(pass.size(), num_sources);
      for (auto source_index = std::size_t{0}; source_index < pass.size(); ++source_index) {
        std::transform(
          pass[source_index].begin(),
          pass[source_index].end(),
          std::back_inserter(flattened),
          [source_index](auto const rg_index) { return std::pair{source_index, rg_index}; });
      }
    }

    auto expected = std::vector<std::pair<std::size_t, cudf::size_type>>{};
    for (auto source_index = std::size_t{0}; source_index < all_rgs.size(); ++source_index) {
      std::transform(
        all_rgs[source_index].begin(),
        all_rgs[source_index].end(),
        std::back_inserter(expected),
        [source_index](auto const rg_index) { return std::pair{source_index, rg_index}; });
    }
    EXPECT_EQ(flattened, expected);
  }
}

TEST_F(HybridScanMultifileFiltersTest, RowGroupPassesSingleSourceParity)
{
  using T = cudf::duration_ms;

  srand(0xced);

  std::vector<std::vector<char>> file_buffers;
  file_buffers.emplace_back(std::get<1>(create_parquet_with_stats<T, 1>()));

  auto inputs = multifile_inputs(build_source_info(file_buffers));

  cudf::io::parquet_reader_options options = cudf::io::parquet_reader_options::builder().build();
  auto const multifile_reader =
    std::make_unique<cudf::io::parquet::experimental::hybrid_scan_multifile>(
      inputs.footer_byte_spans, options);
  auto const single_file_reader =
    std::make_unique<cudf::io::parquet::experimental::hybrid_scan_reader>(
      inputs.footer_byte_spans.front(), options);

  auto const all_rgs = multifile_reader->all_row_groups(options);
  ASSERT_EQ(all_rgs.size(), 1);
  auto constexpr pass_read_limit = std::size_t{10'000};
  auto const multifile_passes =
    multifile_reader->construct_row_group_passes(all_rgs, pass_read_limit);
  auto const single_file_passes =
    single_file_reader->construct_row_group_passes(all_rgs.front(), pass_read_limit);

  auto projected_passes = std::vector<std::vector<cudf::size_type>>{};
  projected_passes.reserve(multifile_passes.size());
  for (auto const& pass : multifile_passes) {
    ASSERT_EQ(pass.size(), 1);
    projected_passes.push_back(pass.front());
  }
  EXPECT_EQ(projected_passes, single_file_passes);
}

TEST_F(HybridScanMultifileFiltersTest, ErrorFilterRowGroupsWithByteRanges)
{
  using T                    = uint32_t;
  auto constexpr num_sources = 2;

  std::vector<std::vector<char>> file_buffers;
  file_buffers.reserve(num_sources);
  auto constexpr seed = 0xb47e;
  std::transform(cuda::counting_iterator{seed},
                 cuda::counting_iterator{seed + num_sources},
                 std::back_inserter(file_buffers),
                 [](auto const src_seed) {
                   srand(src_seed);
                   return std::get<1>(create_parquet_with_stats<T, 1>());
                 });

  auto inputs = multifile_inputs(build_source_info(file_buffers));

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
  auto constexpr seed = 0xc001;
  std::transform(cuda::counting_iterator{seed},
                 cuda::counting_iterator{seed + num_sources},
                 std::back_inserter(file_buffers),
                 [](auto const src_seed) {
                   srand(src_seed);
                   return std::get<1>(create_parquet_with_stats<T, 1, false>());
                 });

  auto inputs = multifile_inputs(build_source_info(file_buffers));

  // Filter - col0 < 50 and col2 > "000010000"
  auto literal_value0 = make_scalar<T>(50, cudf::get_default_stream());
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
  auto constexpr num_sources = 32;
  auto const stream          = cudf::get_default_stream();

  // num_sources sources, each with the same schema
  std::vector<std::vector<char>> file_buffers;
  file_buffers.reserve(num_sources);
  for (int i = 0; i < num_sources; ++i) {
    srand(0xb100 + i);
    file_buffers.emplace_back(std::get<1>(create_parquet_with_stats<T, 1>()));
  }

  auto inputs = multifile_inputs(build_source_info(file_buffers));

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

    auto const [bloom_byte_ranges, bloom_source_map] =
      reader->bloom_filters_byte_ranges(input_row_group_indices, options);
    // cuDF's Parquet writer does not emit bloom filters, so the ranges come back empty. The source
    // map is parallel to the ranges and must always match them in length (here, both empty).
    EXPECT_TRUE(bloom_byte_ranges.empty());
    EXPECT_EQ(bloom_byte_ranges.size(), bloom_source_map.size());
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

    auto const empty_bloom_data = std::vector<cudf::device_span<uint8_t const>>{};
    auto const bloom_filtered   = reader->filter_row_groups_with_bloom_filters(
      empty_bloom_data, input_row_group_indices, options, stream);
    EXPECT_EQ(bloom_filtered, input_row_group_indices);
  }
}

TEST_F(HybridScanMultifileFiltersTest, FilterRowGroupsWithBloomFiltersRealData)
{
  auto constexpr num_sources = 32;
  auto const stream          = cudf::get_default_stream();
  auto aligned_mr = rmm::mr::aligned_resource_adaptor(cudf::get_current_device_resource_ref(),
                                                      bloom_filter_alignment);

  // Embedded copy of cuDF's committed bloom-filter fixture (cuDF cannot write bloom filters, and
  // cuDF tests avoid committed data files). Source:
  // python/cudf/cudf/tests/data/parquet/bloom_filter_alignment.parquet (DuckDB-written; bloom
  // filter on the r_reason_desc column). Regenerate with: xxd -i bloom_filter_alignment.parquet
  constexpr std::array<unsigned char, 1805> bloom_filter_alignment_parquet{
    0x50, 0x41, 0x52, 0x31, 0x15, 0x04, 0x15, 0x98, 0x02, 0x15, 0xa0, 0x02, 0x4c, 0x15, 0x46, 0x15,
    0x00, 0x00, 0x00, 0x8c, 0x01, 0xf0, 0x8b, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03,
    0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00, 0x07,
    0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x0b,
    0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x0d, 0x00, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00, 0x0f,
    0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x11, 0x00, 0x00, 0x00, 0x12, 0x00, 0x00, 0x00, 0x13,
    0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x15, 0x00, 0x00, 0x00, 0x16, 0x00, 0x00, 0x00, 0x17,
    0x00, 0x00, 0x00, 0x18, 0x00, 0x00, 0x00, 0x19, 0x00, 0x00, 0x00, 0x1a, 0x00, 0x00, 0x00, 0x1b,
    0x00, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x00, 0x1d, 0x00, 0x00, 0x00, 0x1e, 0x00, 0x00, 0x00, 0x1f,
    0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0x21, 0x00, 0x00, 0x00, 0x22, 0x00, 0x00, 0x00, 0x23,
    0x00, 0x00, 0x00, 0x15, 0x00, 0x15, 0x90, 0x03, 0x15, 0x62, 0x2c, 0x15, 0x46, 0x15, 0x10, 0x15,
    0x06, 0x15, 0x06, 0x00, 0x00, 0xc8, 0x01, 0x90, 0x02, 0x00, 0x00, 0x00, 0x46, 0x01, 0x06, 0x41,
    0x40, 0x20, 0x0c, 0x44, 0x61, 0x1c, 0x48, 0xa2, 0x2c, 0x4c, 0xe3, 0x3c, 0x50, 0x24, 0x4d, 0x54,
    0x65, 0x5d, 0x58, 0xa6, 0x6d, 0x5c, 0xe7, 0x7d, 0x60, 0x28, 0x02, 0x00, 0x00, 0xfe, 0x02, 0x00,
    0xfe, 0x02, 0x00, 0x8a, 0x02, 0x00, 0x15, 0x04, 0x15, 0xf8, 0x0a, 0x15, 0x86, 0x03, 0x4c, 0x15,
    0x46, 0x15, 0x00, 0x00, 0x00, 0xbc, 0x05, 0x10, 0x10, 0x00, 0x00, 0x00, 0x41, 0x0d, 0x01, 0x00,
    0x42, 0x0d, 0x08, 0x2e, 0x14, 0x00, 0x00, 0x43, 0x4a, 0x14, 0x00, 0x00, 0x44, 0x4a, 0x14, 0x00,
    0x00, 0x45, 0x4a, 0x14, 0x00, 0x00, 0x46, 0x4a, 0x14, 0x00, 0x00, 0x47, 0x4a, 0x14, 0x00, 0x00,
    0x48, 0x4a, 0x14, 0x00, 0x00, 0x49, 0x4a, 0x14, 0x00, 0x00, 0x4a, 0x4a, 0x14, 0x00, 0x00, 0x4b,
    0x4a, 0x14, 0x00, 0x00, 0x4c, 0x4a, 0x14, 0x00, 0x00, 0x4d, 0x4a, 0x14, 0x00, 0x00, 0x4e, 0x4a,
    0x14, 0x00, 0x00, 0x4f, 0x4a, 0x14, 0x00, 0x00, 0x50, 0x4a, 0x14, 0x00, 0x31, 0x2d, 0x2e, 0x2c,
    0x01, 0x00, 0x42, 0x2d, 0x41, 0x2e, 0x14, 0x00, 0x00, 0x43, 0x4a, 0x14, 0x00, 0x00, 0x44, 0x4a,
    0x14, 0x00, 0x00, 0x45, 0x4a, 0x14, 0x00, 0x00, 0x46, 0x4a, 0x14, 0x00, 0x00, 0x47, 0x4a, 0x14,
    0x00, 0x00, 0x48, 0x4a, 0x14, 0x00, 0x00, 0x49, 0x4a, 0x14, 0x00, 0x00, 0x4a, 0x4a, 0x14, 0x00,
    0x00, 0x4b, 0x4a, 0x14, 0x00, 0x00, 0x4c, 0x4a, 0x14, 0x00, 0x00, 0x4d, 0x4a, 0x14, 0x00, 0x00,
    0x4e, 0x4a, 0x14, 0x00, 0x00, 0x4f, 0x4a, 0x14, 0x00, 0x00, 0x50, 0x4a, 0x14, 0x00, 0x51, 0x59,
    0x2e, 0x2c, 0x01, 0x00, 0x42, 0x4d, 0x6d, 0x2e, 0x14, 0x00, 0x00, 0x43, 0x4a, 0x14, 0x00, 0x1c,
    0x44, 0x43, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x15, 0x00, 0x15, 0x90, 0x03, 0x15, 0x62, 0x2c,
    0x15, 0x46, 0x15, 0x10, 0x15, 0x06, 0x15, 0x06, 0x00, 0x00, 0xc8, 0x01, 0x90, 0x02, 0x00, 0x00,
    0x00, 0x46, 0x01, 0x06, 0x41, 0x40, 0x20, 0x0c, 0x44, 0x61, 0x1c, 0x48, 0xa2, 0x2c, 0x4c, 0xe3,
    0x3c, 0x50, 0x24, 0x4d, 0x54, 0x65, 0x5d, 0x58, 0xa6, 0x6d, 0x5c, 0xe7, 0x7d, 0x60, 0x28, 0x02,
    0x00, 0x00, 0xfe, 0x02, 0x00, 0xfe, 0x02, 0x00, 0x8a, 0x02, 0x00, 0x15, 0x04, 0x15, 0x82, 0x0b,
    0x15, 0xe0, 0x07, 0x4c, 0x15, 0x44, 0x15, 0x00, 0x00, 0x00, 0xc1, 0x05, 0xf0, 0x55, 0x13, 0x00,
    0x00, 0x00, 0x50, 0x61, 0x63, 0x6b, 0x61, 0x67, 0x65, 0x20, 0x77, 0x61, 0x73, 0x20, 0x64, 0x61,
    0x6d, 0x61, 0x67, 0x65, 0x64, 0x0f, 0x00, 0x00, 0x00, 0x53, 0x74, 0x6f, 0x70, 0x70, 0x65, 0x64,
    0x20, 0x77, 0x6f, 0x72, 0x6b, 0x69, 0x6e, 0x67, 0x16, 0x00, 0x00, 0x00, 0x44, 0x69, 0x64, 0x20,
    0x6e, 0x6f, 0x74, 0x20, 0x67, 0x65, 0x74, 0x20, 0x69, 0x74, 0x20, 0x6f, 0x6e, 0x20, 0x74, 0x69,
    0x6d, 0x65, 0x1f, 0x00, 0x00, 0x00, 0x4e, 0x6f, 0x74, 0x20, 0x74, 0x68, 0x65, 0x20, 0x70, 0x72,
    0x6f, 0x64, 0x75, 0x63, 0x01, 0x0c, 0x04, 0x61, 0x74, 0x05, 0x51, 0x18, 0x6f, 0x72, 0x64, 0x72,
    0x65, 0x64, 0x0d, 0x05, 0x67, 0x2c, 0x72, 0x74, 0x73, 0x20, 0x6d, 0x69, 0x73, 0x73, 0x69, 0x6e,
    0x67, 0x28, 0x01, 0x4e, 0x08, 0x6f, 0x65, 0x73, 0x05, 0x4f, 0x01, 0x62, 0x1c, 0x20, 0x77, 0x69,
    0x74, 0x68, 0x20, 0x61, 0x20, 0x32, 0x41, 0x00, 0x14, 0x49, 0x20, 0x68, 0x61, 0x76, 0x65, 0x01,
    0x3d, 0x34, 0x47, 0x69, 0x66, 0x74, 0x20, 0x65, 0x78, 0x63, 0x68, 0x61, 0x6e, 0x67, 0x65, 0x16,
    0x01, 0x3d, 0x0d, 0x8b, 0x34, 0x6c, 0x69, 0x6b, 0x65, 0x20, 0x74, 0x68, 0x65, 0x20, 0x63, 0x6f,
    0x6c, 0x6f, 0x72, 0x52, 0x1a, 0x00, 0x14, 0x6d, 0x6f, 0x64, 0x65, 0x6c, 0x15, 0x4e, 0x34, 0x00,
    0x10, 0x6d, 0x61, 0x6b, 0x65, 0x19, 0x4e, 0x19, 0x00, 0xf0, 0x3e, 0x77, 0x61, 0x72, 0x72, 0x61,
    0x6e, 0x74, 0x79, 0x1e, 0x00, 0x00, 0x00, 0x4e, 0x6f, 0x20, 0x73, 0x65, 0x72, 0x76, 0x69, 0x63,
    0x65, 0x20, 0x6c, 0x6f, 0x63, 0x61, 0x74, 0x69, 0x6f, 0x6e, 0x20, 0x69, 0x6e, 0x20, 0x6d, 0x79,
    0x20, 0x61, 0x72, 0x65, 0x61, 0x1f, 0x00, 0x00, 0x00, 0x46, 0x6f, 0x75, 0x6e, 0x64, 0x20, 0x61,
    0x20, 0x62, 0x65, 0x74, 0x74, 0x65, 0x72, 0x20, 0x70, 0x72, 0x01, 0x2c, 0x28, 0x69, 0x6e, 0x20,
    0x61, 0x20, 0x73, 0x74, 0x6f, 0x72, 0x65, 0x2b, 0x46, 0x23, 0x00, 0x14, 0x65, 0x78, 0x74, 0x65,
    0x6e, 0x64, 0x21, 0x5d, 0x0d, 0x69, 0x1d, 0x2f, 0x00, 0x14, 0x05, 0x74, 0x04, 0x74, 0x20, 0x21,
    0x16, 0x30, 0x69, 0x6e, 0x67, 0x20, 0x61, 0x6e, 0x79, 0x20, 0x6d, 0x6f, 0x72, 0x65, 0x0b, 0x1d,
    0xa9, 0x3c, 0x66, 0x69, 0x74, 0x0a, 0x00, 0x00, 0x00, 0x57, 0x72, 0x6f, 0x6e, 0x67, 0x20, 0x73,
    0x69, 0x7a, 0x05, 0x1d, 0x28, 0x4c, 0x6f, 0x73, 0x74, 0x20, 0x6d, 0x79, 0x20, 0x6a, 0x6f, 0x62,
    0x01, 0x44, 0x80, 0x75, 0x6e, 0x61, 0x75, 0x74, 0x68, 0x6f, 0x69, 0x7a, 0x65, 0x64, 0x20, 0x70,
    0x75, 0x72, 0x63, 0x68, 0x61, 0x73, 0x65, 0x12, 0x00, 0x00, 0x00, 0x64, 0x75, 0x70, 0x6c, 0x69,
    0x63, 0x61, 0x74, 0x65, 0x15, 0x16, 0x24, 0x0c, 0x00, 0x00, 0x00, 0x69, 0x74, 0x73, 0x20, 0x69,
    0x73, 0x01, 0xc5, 0x04, 0x6f, 0x79, 0x09, 0x10, 0x09, 0x0f, 0x40, 0x67, 0x69, 0x72, 0x6c, 0x09,
    0x00, 0x00, 0x00, 0x72, 0x65, 0x61, 0x73, 0x6f, 0x6e, 0x20, 0x32, 0x33, 0x2e, 0x0d, 0x00, 0x00,
    0x34, 0x2e, 0x0d, 0x00, 0x00, 0x35, 0x2e, 0x0d, 0x00, 0x00, 0x36, 0x2e, 0x0d, 0x00, 0x00, 0x37,
    0x2e, 0x0d, 0x00, 0x00, 0x38, 0x2e, 0x0d, 0x00, 0x00, 0x39, 0x1d, 0x0d, 0x04, 0x33, 0x31, 0x2e,
    0x0d, 0x00, 0x00, 0x32, 0x2e, 0x0d, 0x00, 0x2e, 0x75, 0x00, 0x38, 0x33, 0x34, 0x09, 0x00, 0x00,
    0x00, 0x72, 0x65, 0x61, 0x73, 0x6f, 0x6e, 0x20, 0x33, 0x35, 0x15, 0x00, 0x15, 0x90, 0x03, 0x15,
    0x62, 0x2c, 0x15, 0x46, 0x15, 0x10, 0x15, 0x06, 0x15, 0x06, 0x00, 0x00, 0xc8, 0x01, 0x90, 0x02,
    0x00, 0x00, 0x00, 0x46, 0x01, 0x06, 0x41, 0x40, 0x20, 0x0c, 0x44, 0x61, 0x1c, 0x48, 0xa2, 0x2c,
    0x4c, 0xe3, 0x3c, 0x50, 0x24, 0x4d, 0x54, 0x65, 0x5d, 0x58, 0xa6, 0x6d, 0x5c, 0xd7, 0x79, 0x1f,
    0x18, 0x02, 0x00, 0x00, 0xfe, 0x02, 0x00, 0xfe, 0x02, 0x00, 0x8a, 0x02, 0x00, 0x15, 0x80, 0x01,
    0x1c, 0x1c, 0x00, 0x00, 0x1c, 0x1c, 0x00, 0x00, 0x1c, 0x1c, 0x00, 0x00, 0x00, 0x2e, 0x25, 0xce,
    0x22, 0x55, 0x71, 0x2a, 0xda, 0x42, 0x80, 0xf4, 0xf3, 0xe7, 0x0e, 0x91, 0x23, 0x38, 0x21, 0x89,
    0x8d, 0xb0, 0x63, 0x93, 0xe8, 0x01, 0xfd, 0x58, 0x11, 0xda, 0x28, 0x63, 0x87, 0x3f, 0x40, 0x7c,
    0x32, 0x1c, 0x08, 0xa8, 0x2f, 0xe2, 0xd8, 0xa3, 0x80, 0x2e, 0x4e, 0xa8, 0x4a, 0x86, 0x16, 0x24,
    0xce, 0xad, 0xea, 0x68, 0x00, 0x20, 0x36, 0xe6, 0xa2, 0x10, 0x99, 0x80, 0x6d, 0x15, 0x80, 0x01,
    0x1c, 0x1c, 0x00, 0x00, 0x1c, 0x1c, 0x00, 0x00, 0x1c, 0x1c, 0x00, 0x00, 0x00, 0x4e, 0xf3, 0x6e,
    0x00, 0x4b, 0x4f, 0x44, 0x74, 0x0e, 0x73, 0xa0, 0x90, 0xe3, 0xbd, 0xe2, 0x0a, 0x32, 0x87, 0x50,
    0xc5, 0x41, 0x2a, 0x2c, 0xee, 0x3a, 0x12, 0x58, 0xa3, 0x0d, 0x05, 0xe5, 0x89, 0x66, 0x0f, 0xa3,
    0xc0, 0xb8, 0xa2, 0xe6, 0xc1, 0x4f, 0x00, 0xf6, 0x8c, 0x9a, 0xa2, 0xf0, 0x17, 0xc4, 0x29, 0x1f,
    0x06, 0x89, 0xbf, 0x13, 0x88, 0x58, 0x84, 0xc7, 0x38, 0xf9, 0x18, 0x01, 0x78, 0x15, 0x80, 0x01,
    0x1c, 0x1c, 0x00, 0x00, 0x1c, 0x1c, 0x00, 0x00, 0x1c, 0x1c, 0x00, 0x00, 0x00, 0x50, 0x35, 0x06,
    0xbd, 0xf8, 0xeb, 0x00, 0x26, 0xc1, 0x0a, 0x5a, 0xd2, 0x4e, 0x69, 0x47, 0x82, 0xf4, 0x61, 0x61,
    0x68, 0x11, 0x32, 0x90, 0xe6, 0xa4, 0x8e, 0xa8, 0x44, 0x41, 0x53, 0x08, 0xd5, 0xea, 0x24, 0xfd,
    0x91, 0xae, 0x84, 0x81, 0xb8, 0xa6, 0x9a, 0x41, 0x75, 0xdc, 0x2b, 0x5c, 0x92, 0x06, 0xf0, 0x87,
    0xea, 0x38, 0x38, 0x52, 0x26, 0x08, 0x11, 0x7d, 0x6d, 0x28, 0xfc, 0x60, 0x89, 0x15, 0x02, 0x19,
    0x4c, 0x35, 0x00, 0x18, 0x0d, 0x64, 0x75, 0x63, 0x6b, 0x64, 0x62, 0x5f, 0x73, 0x63, 0x68, 0x65,
    0x6d, 0x61, 0x15, 0x06, 0x00, 0x15, 0x02, 0x25, 0x02, 0x18, 0x0b, 0x72, 0x5f, 0x72, 0x65, 0x61,
    0x73, 0x6f, 0x6e, 0x5f, 0x73, 0x6b, 0x25, 0x22, 0x00, 0x15, 0x0c, 0x25, 0x02, 0x18, 0x0b, 0x72,
    0x5f, 0x72, 0x65, 0x61, 0x73, 0x6f, 0x6e, 0x5f, 0x69, 0x64, 0x25, 0x00, 0x00, 0x15, 0x0c, 0x25,
    0x02, 0x18, 0x0d, 0x72, 0x5f, 0x72, 0x65, 0x61, 0x73, 0x6f, 0x6e, 0x5f, 0x64, 0x65, 0x73, 0x63,
    0x25, 0x00, 0x00, 0x16, 0x46, 0x19, 0x1c, 0x19, 0x3c, 0x26, 0x00, 0x1c, 0x15, 0x02, 0x19, 0x15,
    0x10, 0x19, 0x18, 0x0b, 0x72, 0x5f, 0x72, 0x65, 0x61, 0x73, 0x6f, 0x6e, 0x5f, 0x73, 0x6b, 0x15,
    0x02, 0x16, 0x46, 0x16, 0xea, 0x05, 0x16, 0xc4, 0x03, 0x26, 0xc6, 0x02, 0x26, 0x08, 0x1c, 0x18,
    0x04, 0x23, 0x00, 0x00, 0x00, 0x18, 0x04, 0x01, 0x00, 0x00, 0x00, 0x16, 0x00, 0x16, 0x46, 0x18,
    0x04, 0x23, 0x00, 0x00, 0x00, 0x18, 0x04, 0x01, 0x00, 0x00, 0x00, 0x11, 0x11, 0x00, 0x26, 0xfa,
    0x10, 0x15, 0xa0, 0x01, 0x00, 0x00, 0x26, 0x00, 0x1c, 0x15, 0x0c, 0x19, 0x15, 0x10, 0x19, 0x18,
    0x0b, 0x72, 0x5f, 0x72, 0x65, 0x61, 0x73, 0x6f, 0x6e, 0x5f, 0x69, 0x64, 0x15, 0x02, 0x16, 0x46,
    0x16, 0xca, 0x0e, 0x16, 0xaa, 0x04, 0x26, 0xf0, 0x06, 0x26, 0xcc, 0x03, 0x1c, 0x18, 0x10, 0x41,
    0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x50, 0x42, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x18,
    0x10, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x42, 0x41, 0x41, 0x41, 0x41, 0x41,
    0x41, 0x16, 0x00, 0x16, 0x46, 0x18, 0x10, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x50,
    0x42, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x18, 0x10, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41,
    0x41, 0x41, 0x42, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x11, 0x11, 0x00, 0x26, 0x9a, 0x12, 0x15,
    0xa0, 0x01, 0x00, 0x00, 0x26, 0x00, 0x1c, 0x15, 0x0c, 0x19, 0x15, 0x10, 0x19, 0x18, 0x0d, 0x72,
    0x5f, 0x72, 0x65, 0x61, 0x73, 0x6f, 0x6e, 0x5f, 0x64, 0x65, 0x73, 0x63, 0x15, 0x02, 0x16, 0x46,
    0x16, 0xd4, 0x0e, 0x16, 0x84, 0x09, 0x26, 0xf4, 0x0f, 0x26, 0xf6, 0x07, 0x1c, 0x18, 0x14, 0x75,
    0x6e, 0x61, 0x75, 0x74, 0x68, 0x6f, 0x69, 0x7a, 0x65, 0x64, 0x20, 0x70, 0x75, 0x72, 0x63, 0x68,
    0x61, 0x73, 0x65, 0x18, 0x0b, 0x44, 0x69, 0x64, 0x20, 0x6e, 0x6f, 0x74, 0x20, 0x66, 0x69, 0x74,
    0x16, 0x00, 0x16, 0x44, 0x18, 0x14, 0x75, 0x6e, 0x61, 0x75, 0x74, 0x68, 0x6f, 0x69, 0x7a, 0x65,
    0x64, 0x20, 0x70, 0x75, 0x72, 0x63, 0x68, 0x61, 0x73, 0x65, 0x18, 0x0b, 0x44, 0x69, 0x64, 0x20,
    0x6e, 0x6f, 0x74, 0x20, 0x66, 0x69, 0x74, 0x11, 0x11, 0x00, 0x26, 0xba, 0x13, 0x15, 0xa0, 0x01,
    0x00, 0x00, 0x16, 0x88, 0x23, 0x16, 0x46, 0x26, 0x08, 0x00, 0x28, 0x28, 0x44, 0x75, 0x63, 0x6b,
    0x44, 0x42, 0x20, 0x76, 0x65, 0x72, 0x73, 0x69, 0x6f, 0x6e, 0x20, 0x76, 0x31, 0x2e, 0x33, 0x2e,
    0x30, 0x20, 0x28, 0x62, 0x75, 0x69, 0x6c, 0x64, 0x20, 0x37, 0x31, 0x63, 0x35, 0x63, 0x30, 0x37,
    0x63, 0x64, 0x64, 0x29, 0x00, 0xd8, 0x01, 0x00, 0x00, 0x50, 0x41, 0x52, 0x31,
  };
  // Sources backed by the embedded bloom-filter fixture
  std::vector<char> const fixture(bloom_filter_alignment_parquet.begin(),
                                  bloom_filter_alignment_parquet.end());
  std::vector<std::vector<char>> file_buffers(num_sources, fixture);

  auto inputs = multifile_inputs(build_source_info(file_buffers));

  // An equality predicate makes the "r_reason_desc" column bloom-eligible
  auto literal_value = cudf::string_scalar("Did not like the color", true, stream);
  auto literal       = cudf::ast::literal(literal_value);
  auto col_ref       = cudf::ast::column_name_reference("r_reason_desc");
  auto filter        = cudf::ast::operation(cudf::ast::ast_operator::EQUAL, col_ref, literal);

  auto options      = cudf::io::parquet_reader_options::builder().filter(filter).build();
  auto const reader = std::make_unique<cudf::io::parquet::experimental::hybrid_scan_multifile>(
    cudf::host_span<cudf::host_span<uint8_t const> const>{inputs.footer_byte_spans}, options);

  auto const input_row_group_indices = reader->all_row_groups(options);
  ASSERT_EQ(input_row_group_indices.size(), num_sources);

  auto const [bloom_byte_ranges, bloom_source_map] =
    reader->bloom_filters_byte_ranges(input_row_group_indices, options);
  ASSERT_EQ(bloom_byte_ranges.size(), static_cast<size_t>(num_sources));
  ASSERT_EQ(bloom_byte_ranges.size(), bloom_source_map.size());
  std::vector<cudf::size_type> expected_source_map(num_sources);
  std::iota(expected_source_map.begin(), expected_source_map.end(), 0);
  EXPECT_EQ(bloom_source_map, expected_source_map);
  EXPECT_TRUE(std::none_of(bloom_byte_ranges.begin(), bloom_byte_ranges.end(), [](auto const& r) {
    return r.is_empty();
  }));

  std::vector<std::vector<byte_range_info>> ranges_per_source(num_sources);
  for (size_t i = 0; i < bloom_byte_ranges.size(); ++i) {
    ASSERT_LT(bloom_source_map[i], num_sources);
    ranges_per_source[bloom_source_map[i]].push_back(bloom_byte_ranges[i]);
  }
  auto [bloom_buffers, bloom_data_per_source, bloom_tasks] =
    cudf::io::parquet::fetch_byte_ranges_to_device_async(
      inputs.datasource_refs, ranges_per_source, stream, aligned_mr);
  bloom_tasks.get();
  for (auto const& per_source : bloom_data_per_source) {
    ASSERT_EQ(per_source.size(), 1);
  }

  // Flatten the per-source bloom filter data in source order
  std::vector<cudf::device_span<uint8_t const>> bloom_filter_data;
  for (auto const& per_source : bloom_data_per_source) {
    bloom_filter_data.insert(bloom_filter_data.end(), per_source.begin(), per_source.end());
  }
  auto const bloom_filtered = reader->filter_row_groups_with_bloom_filters(
    bloom_filter_data, input_row_group_indices, options, stream);

  // Shouldn't filter out any RG, since the queried value is present in every source.
  EXPECT_EQ(bloom_filtered, input_row_group_indices);
}

TEST_F(HybridScanMultifileFiltersTest, BuildAllTrueRowMask)
{
  using T                    = uint64_t;
  auto constexpr num_sources = 2;

  std::vector<std::vector<char>> file_buffers;
  file_buffers.reserve(num_sources);
  auto constexpr seed = 0xa11;
  std::transform(cuda::counting_iterator{seed},
                 cuda::counting_iterator{seed + num_sources},
                 std::back_inserter(file_buffers),
                 [](auto const src_seed) {
                   srand(src_seed);
                   return std::get<1>(create_parquet_with_stats<T, 1>());
                 });

  auto inputs = multifile_inputs(build_source_info(file_buffers));

  auto const options = cudf::io::parquet_reader_options::builder().build();
  auto const reader  = std::make_unique<cudf::io::parquet::experimental::hybrid_scan_multifile>(
    inputs.footer_byte_spans, options);

  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();

  auto test_all_true_row_mask =
    [&](cudf::host_span<std::vector<cudf::size_type> const> row_group_indices) {
      auto const row_mask = reader->build_all_true_row_mask(row_group_indices, stream, mr);

      auto const expected_num_rows = reader->total_rows_in_row_groups(row_group_indices);

      EXPECT_EQ(row_mask->type().id(), cudf::type_id::BOOL8);
      EXPECT_EQ(row_mask->size(), expected_num_rows);
      EXPECT_EQ(row_mask->null_count(), 0);
      auto const host_row_mask = host_row_mask_data<bool>(row_mask->view(), stream);
      EXPECT_EQ(std::count(host_row_mask.begin(), host_row_mask.end(), true), expected_num_rows);
    };

  auto row_group_indices = std::vector<std::vector<cudf::size_type>>{{0, 2}, {1, 3}};
  test_all_true_row_mask(row_group_indices);

  row_group_indices = reader->all_row_groups(options);
  test_all_true_row_mask(row_group_indices);
}

template <typename T>
struct HybridScanMultifilePageIndexRowMaskTest : public HybridScanMultifileFiltersTest {};

// Unsigned numeric types except booleans for page index stats tests
using SignedIntegralTypesNotBool =
  cudf::test::ContainedIn<cudf::test::Types<int8_t, int16_t, int32_t, int64_t>>;
using PageIndexRowMaskTestTypes =
  cudf::test::RemoveIf<SignedIntegralTypesNotBool,
                       cudf::test::Concat<cudf::test::IntegralTypesNotBool>>;

TYPED_TEST_SUITE(HybridScanMultifilePageIndexRowMaskTest, PageIndexRowMaskTestTypes);

TYPED_TEST(HybridScanMultifilePageIndexRowMaskTest, BuildRowMaskWithPageIndexStats)
{
  using T                    = TypeParam;
  auto constexpr num_sources = 4;

  std::vector<std::vector<char>> file_buffers;
  file_buffers.reserve(num_sources);
  auto constexpr seed = 0xa11b;
  std::transform(cuda::counting_iterator{seed},
                 cuda::counting_iterator{seed + num_sources},
                 std::back_inserter(file_buffers),
                 [](auto const src_seed) {
                   srand(src_seed);
                   return std::get<1>(create_parquet_with_stats<T, 1, false>());
                 });

  auto inputs = multifile_inputs(build_source_info(file_buffers));

  auto options      = cudf::io::parquet_reader_options::builder().build();
  auto const reader = std::make_unique<cudf::io::parquet::experimental::hybrid_scan_multifile>(
    inputs.footer_byte_spans, options);

  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();

  auto const input_row_group_indices = reader->all_row_groups(options);

  auto const test_filter_data_pages_with_stats = [&](
                                                   cudf::ast::operation const& filter_expression,
                                                   cudf::size_type const expected_surviving_rows) {
    options.set_filter(filter_expression);
    reader->reset_column_selection();

    auto const row_mask =
      reader->build_row_mask_with_page_index_stats(input_row_group_indices, options, stream, mr);

    auto const expected_num_rows = reader->total_rows_in_row_groups(input_row_group_indices);
    EXPECT_EQ(row_mask->type().id(), cudf::type_id::BOOL8);
    EXPECT_EQ(row_mask->size(), expected_num_rows);
    EXPECT_EQ(row_mask->null_count(), 0);

    auto const host_row_mask = host_row_mask_data<bool>(row_mask->view(), stream);
    EXPECT_EQ(std::count(host_row_mask.begin(), host_row_mask.end(), true),
              expected_surviving_rows);
  };

  // Calling the page-index row mask builder before setting up the page index should raise an error.
  {
    auto literal_value     = make_scalar<T>(100, stream);
    auto const literal     = cudf::ast::literal(literal_value);
    auto const col_ref     = cudf::ast::column_name_reference("col0");
    auto filter_expression = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref, literal);
    options.set_filter(filter_expression);
    EXPECT_THROW(std::ignore = reader->build_row_mask_with_page_index_stats(
                   input_row_group_indices, options, stream, mr),
                 std::runtime_error);
  }

  setup_page_indexes(*reader, inputs);

  // Filtering AST - table[0] < 100
  {
    auto literal_value = make_scalar<T>(100, stream);
    auto const literal = cudf::ast::literal(literal_value);
    auto const col_ref = cudf::ast::column_name_reference("col0");
    auto filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::GREATER, literal, col_ref);
    auto constexpr expected_surviving_rows =
      num_sources * num_ordered_rows / (std::is_signed_v<T> ? 4 : 2);
    test_filter_data_pages_with_stats(filter_expression, expected_surviving_rows);
  }

  // Filtering AST - table[2] >= 10000
  {
    auto literal_value = cudf::string_scalar("000010000", true, stream);
    auto literal       = cudf::ast::literal(literal_value);
    auto col_ref       = cudf::ast::column_name_reference("col2");
    auto filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, col_ref, literal);
    auto constexpr expected_surviving_rows =
      num_sources * num_ordered_rows / (std::is_signed_v<T> ? 4 : 2);
    test_filter_data_pages_with_stats(filter_expression, expected_surviving_rows);
  }

  // Filtering AST - table[0] < 50 AND table[2] < "000010000"
  {
    auto literal_value1 = make_scalar<T>(50, stream);
    auto const literal1 = cudf::ast::literal(literal_value1);
    auto const col_ref1 = cudf::ast::column_name_reference("col0");
    auto filter_expression1 =
      cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref1, literal1);

    auto literal_value2 = cudf::string_scalar("000010000", true, stream);
    auto literal2       = cudf::ast::literal(literal_value2);
    auto col_ref2       = cudf::ast::column_name_reference("col2");
    auto filter_expression2 =
      cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref2, literal2);

    auto filter_expression = cudf::ast::operation(
      cudf::ast::ast_operator::LOGICAL_AND, filter_expression1, filter_expression2);
    auto constexpr expected_surviving_rows = num_sources * page_size_for_ordered_tests;
    test_filter_data_pages_with_stats(filter_expression, expected_surviving_rows);
  }

  // Filtering AST - table[0] > 150 OR table[2] < "000005000"
  {
    auto literal_value1 = make_scalar<T>(150, stream);
    auto const literal1 = cudf::ast::literal(literal_value1);
    auto const col_ref1 = cudf::ast::column_name_reference("col0");
    auto filter_expression1 =
      cudf::ast::operation(cudf::ast::ast_operator::GREATER, col_ref1, literal1);

    auto literal_value2 = cudf::string_scalar("000005000", true, stream);
    auto literal2       = cudf::ast::literal(literal_value2);
    auto col_ref2       = cudf::ast::column_name_reference("col2");
    auto filter_expression2 =
      cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref2, literal2);

    auto filter_expression = cudf::ast::operation(
      cudf::ast::ast_operator::LOGICAL_OR, filter_expression1, filter_expression2);
    auto constexpr expected_surviving_rows = 2 * num_sources * page_size_for_ordered_tests;
    test_filter_data_pages_with_stats(filter_expression, expected_surviving_rows);
  }
}
