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
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_buffer.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
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

/**
 * @brief Writes `tbl` to a Parquet host buffer with the given top-level column names, column-chunk
 * statistics + page index (STATISTICS_COLUMN) and ALWAYS dictionary encoding.
 *
 * Used to build reordered/mismatched per-source schemas that exercise the dictionary-page pruning
 * path under `allow_mismatched_pq_schemas`.
 */
[[nodiscard]] std::vector<char> write_mismatched_source(cudf::table_view const& tbl,
                                                        std::vector<std::string> const& names)
{
  cudf::io::table_input_metadata md{tbl};
  for (std::size_t i = 0; i < names.size(); ++i) {
    md.column_metadata[i].set_name(names[i]);
  }
  std::vector<char> buffer;
  auto out_opts = cudf::io::parquet_writer_options::builder(cudf::io::sink_info{&buffer}, tbl)
                    .metadata(std::move(md))
                    .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN)
                    .dictionary_policy(cudf::io::dictionary_policy::ALWAYS)
                    .build();
  cudf::io::write_parquet(out_opts);
  return buffer;
}

/**
 * @brief Fetches and sets up the per-source page index on the reader.
 *
 * The page index buffers and their host spans are stored in the caller-provided vectors so that
 * they outlive subsequent reader calls that rely on the page index.
 */
void setup_multifile_page_index(
  cudf::io::parquet::experimental::hybrid_scan_multifile const& reader,
  multifile_inputs& inputs,
  std::vector<std::unique_ptr<cudf::io::datasource::buffer>>& page_index_buffers,
  std::vector<cudf::host_span<uint8_t const>>& page_index_byte_spans)
{
  auto const page_index_byte_ranges = reader.page_index_byte_ranges();
  auto const num_sources            = inputs.datasources.size();
  page_index_buffers.reserve(num_sources);
  page_index_byte_spans.reserve(num_sources);

  auto iter = cuda::zip_iterator(page_index_byte_ranges.begin(), inputs.datasources.begin());
  std::for_each(iter, iter + num_sources, [&](auto const& pair) {
    auto const& [pgidx_byte_range, datasource] = pair;
    page_index_buffers.emplace_back(
      cudf::io::parquet::fetch_page_index_to_host(*datasource, pgidx_byte_range));
    page_index_byte_spans.emplace_back(*page_index_buffers.back());
  });

  reader.setup_page_indexes(
    cudf::host_span<cudf::host_span<uint8_t const> const>{page_index_byte_spans});
}

/**
 * @brief Filter input row groups using column chunk dictionaries via the experimental parquet
 * reader for hybrid scan (multi-file)
 *
 * Multi-file counterpart of the single-file `filter_row_groups_with_dictionaries` helper, kept as
 * close to it as possible. The dictionary page byte ranges are flat and source-major, so each
 * source's slice is fetched from its own datasource before filtering.
 *
 * @param inputs Multi-file datasources
 * @param reader Hybrid scan multi-file reader
 * @param input_row_group_indices Input per-source row group indices
 * @param options Parquet reader options
 * @param stream CUDA stream
 * @param mr Device memory resource
 *
 * @return Vector of per-source dictionary-filtered row group indices
 */
auto filter_row_groups_with_dictionaries(
  multifile_inputs& inputs,
  cudf::io::parquet::experimental::hybrid_scan_multifile const& reader,
  std::vector<std::vector<cudf::size_type>> const& input_row_group_indices,
  cudf::io::parquet_reader_options const& options,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  // Get dictionary page byte ranges from the reader
  auto const dict_page_byte_ranges =
    reader.dictionary_pages_byte_ranges(input_row_group_indices, options);

  // If we have dictionary page byte ranges, filter row groups with dictionary pages
  std::vector<std::vector<cudf::size_type>> dict_page_filtered_row_group_indices;
  dict_page_filtered_row_group_indices.reserve(input_row_group_indices.size());

  CUDF_EXPECTS(dict_page_byte_ranges.size() > 0, "No dictionary page byte ranges found");

  // Dictionary page byte ranges are flat and source-major, so derive the dictionary column count
  // and fetch each source's slice from its own datasource
  std::size_t total_row_groups = 0;
  for (auto const& rgs : input_row_group_indices) {
    total_row_groups += rgs.size();
  }
  auto const num_dictionary_columns = dict_page_byte_ranges.size() / total_row_groups;

  // Fetch dictionary page buffers from the input file buffers
  std::vector<rmm::device_buffer> dict_page_buffers;
  std::vector<cudf::device_span<uint8_t const>> dict_page_data;
  std::size_t offset = 0;
  for (std::size_t src = 0; src < input_row_group_indices.size(); ++src) {
    auto const count = input_row_group_indices[src].size() * num_dictionary_columns;
    std::vector<cudf::io::text::byte_range_info> const src_ranges(
      dict_page_byte_ranges.begin() + offset, dict_page_byte_ranges.begin() + offset + count);
    offset += count;

    auto [buffers, data, task] = cudf::io::parquet::fetch_byte_ranges_to_device_async(
      *inputs.datasources[src], src_ranges, stream, mr);
    task.get();
    for (auto& buffer : buffers) {
      dict_page_buffers.emplace_back(std::move(buffer));
    }
    dict_page_data.insert(dict_page_data.end(), data.begin(), data.end());
  }

  dict_page_filtered_row_group_indices = reader.filter_row_groups_with_dictionary_pages(
    dict_page_data, input_row_group_indices, options, stream);

  return dict_page_filtered_row_group_indices;
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

// Matched-schema real dictionary pruning across two sources. Both sources share the same schema but
// `col2` is a per-source constant string ("0100" in source A, "0200" in source B). With an equality
// predicate `col2 == "0100"` source A keeps all of its row groups while source B is fully pruned.
TEST_F(HybridScanMultifileFiltersTest, FilterRowGroupsWithDictionaryPages)
{
  using T                           = uint32_t;
  auto constexpr num_sources        = 2;
  auto constexpr num_row_groups     = 4;
  auto constexpr rows_per_row_group = page_size_for_ordered_tests;
  auto stream                       = cudf::get_default_stream();
  auto mr                           = cudf::get_current_device_resource_ref();

  // Each source has 4 row groups (20000 rows / 5000 rows per row group) and is dictionary encoded
  // under `dictionary_policy::ALWAYS`. `col2` is a per-source constant string.
  std::vector<std::vector<char>> file_buffers;
  file_buffers.reserve(num_sources);
  srand(0xd1c7);
  file_buffers.emplace_back(std::get<1>(create_parquet_with_stats<T, 1>(100)));  // col2 == "0100"
  srand(0xfeed);
  file_buffers.emplace_back(std::get<1>(create_parquet_with_stats<T, 1>(200)));  // col2 == "0200"

  auto inputs = build_multifile_inputs(file_buffers);

  // Filter - col2 == "0100" (present only in source A's dictionary)
  auto literal_value = cudf::string_scalar("0100", true, stream);
  auto literal       = cudf::ast::literal(literal_value);
  auto col_ref       = cudf::ast::column_name_reference("col2");
  auto filter        = cudf::ast::operation(cudf::ast::ast_operator::EQUAL, col_ref, literal);

  auto options      = cudf::io::parquet_reader_options::builder().filter(filter).build();
  auto const reader = std::make_unique<cudf::io::parquet::experimental::hybrid_scan_multifile>(
    inputs.footer_byte_spans, options);

  // Page index is needed to detect dictionary-only encoded pages
  std::vector<std::unique_ptr<cudf::io::datasource::buffer>> page_index_buffers;
  std::vector<cudf::host_span<uint8_t const>> page_index_byte_spans;
  setup_multifile_page_index(*reader, inputs, page_index_buffers, page_index_byte_spans);

  auto const input_row_group_indices = reader->all_row_groups(options);
  ASSERT_EQ(input_row_group_indices.size(), num_sources);
  EXPECT_EQ(reader->total_rows_in_row_groups(input_row_group_indices),
            num_sources * num_row_groups * rows_per_row_group);

  auto const dict_filtered = filter_row_groups_with_dictionaries(
    inputs, *reader, input_row_group_indices, options, stream, mr);

  // Source A keeps all 4 row groups (col2 == "0100"); source B is fully pruned (only "0200")
  ASSERT_EQ(dict_filtered.size(), num_sources);
  EXPECT_EQ(dict_filtered.front(), (std::vector<cudf::size_type>{0, 1, 2, 3}));
  EXPECT_TRUE(dict_filtered.back().empty());
}

// Mismatched-schema regression for the dictionary-page pruning path under
// `allow_mismatched_pq_schemas`. `get_dictionary_page_bytes` used to resolve column chunks by the
// raw zeroth-source `schema_idx` (no `map_schema_index`), so a reordered source read the wrong
// column chunk. Here `price` is schema 2 in source A but schema 3 in source B; the buggy lookup
// reads source B's int64 `id` (schema 2) instead of its double `price`. With `price == 40` (present
// only in source B) the correct result keeps source B and prunes source A.
TEST_F(HybridScanMultifileFiltersTest, MismatchedSchemaDictionaryPruningCollision)
{
  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();

  // Use enough low-cardinality rows that the writer actually emits dictionary-encoded pages even
  // under `dictionary_policy::ALWAYS` (a tiny all-unique column falls back to PLAIN because the
  // dictionary would not save space). `price == 40` is present only in source B's `price` column.
  auto constexpr num_rows = cudf::size_type{600};
  std::array<double, 3> const price_a_cycle{50.0, 150.0, 75.0};  // no 40
  std::array<double, 3> const price_b_cycle{40.0, 200.0, 99.0};  // has 40
  std::array<std::string, 3> const cat_cycle{"x", "y", "z"};

  std::vector<int64_t> id_a_vals(num_rows);
  std::vector<int64_t> id_b_vals(num_rows);
  std::vector<double> price_a_vals(num_rows);
  std::vector<double> price_b_vals(num_rows);
  std::vector<std::string> category_b_vals(num_rows);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    id_a_vals[i]       = (i % 3) + 1;     // {1, 2, 3}
    id_b_vals[i]       = 1000 + (i % 3);  // {1000, 1001, 1002}
    price_a_vals[i]    = price_a_cycle[i % 3];
    price_b_vals[i]    = price_b_cycle[i % 3];
    category_b_vals[i] = cat_cycle[i % 3];
  }
  cudf::test::fixed_width_column_wrapper<int64_t> const id_a(id_a_vals.begin(), id_a_vals.end());
  cudf::test::fixed_width_column_wrapper<double> const price_a(price_a_vals.begin(),
                                                               price_a_vals.end());
  cudf::test::strings_column_wrapper const category_b(category_b_vals.begin(),
                                                      category_b_vals.end());
  cudf::test::fixed_width_column_wrapper<int64_t> const id_b(id_b_vals.begin(), id_b_vals.end());
  cudf::test::fixed_width_column_wrapper<double> const price_b(price_b_vals.begin(),
                                                               price_b_vals.end());

  std::vector<std::vector<char>> file_buffers;
  file_buffers.emplace_back(
    write_mismatched_source(cudf::table_view{{id_a, price_a}}, {"id", "price"}));
  file_buffers.emplace_back(write_mismatched_source(cudf::table_view{{category_b, id_b, price_b}},
                                                    {"category", "id", "price"}));

  auto inputs = build_multifile_inputs(file_buffers);

  // Filter - price == 40 (dictionary pruning participates for equality predicates)
  auto literal_value = cudf::numeric_scalar<double>(40.0, true, stream);
  auto literal       = cudf::ast::literal(literal_value);
  auto col_ref       = cudf::ast::column_name_reference("price");
  auto filter        = cudf::ast::operation(cudf::ast::ast_operator::EQUAL, col_ref, literal);

  auto options = cudf::io::parquet_reader_options::builder()
                   .allow_mismatched_pq_schemas(true)
                   .column_names({"id", "price"})
                   .filter(filter)
                   .build();

  auto const reader = std::make_unique<cudf::io::parquet::experimental::hybrid_scan_multifile>(
    cudf::host_span<cudf::host_span<uint8_t const> const>{inputs.footer_byte_spans}, options);

  // Page index is needed to detect dictionary-only encoded pages
  std::vector<std::unique_ptr<cudf::io::datasource::buffer>> page_index_buffers;
  std::vector<cudf::host_span<uint8_t const>> page_index_byte_spans;
  setup_multifile_page_index(*reader, inputs, page_index_buffers, page_index_byte_spans);

  auto const input_row_group_indices = reader->all_row_groups(options);
  ASSERT_EQ(input_row_group_indices.size(), 2);

  auto const dict_filtered = filter_row_groups_with_dictionaries(
    inputs, *reader, input_row_group_indices, options, stream, mr);

  // Correct behavior: source A is pruned (no price == 40), source B survives (price == 40 present)
  ASSERT_EQ(dict_filtered.size(), 2);
  EXPECT_TRUE(dict_filtered.front().empty()) << "Source A should be pruned (no price == 40)";
  EXPECT_EQ(dict_filtered.back(), (std::vector<cudf::size_type>{0}))
    << "Source B should survive (price == 40 present)";
}
