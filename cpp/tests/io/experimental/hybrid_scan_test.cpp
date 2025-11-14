/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
#include <rmm/mr/aligned_resource_adaptor.hpp>

namespace {

/**
 * @brief Helper to construct a random list<str> column
 *
 * @param gen Random engine
 * @param is_str_nullable Whether the string column should be nullable
 * @param is_list_nullable Whether the list column should be nullable
 *
 * @return Unique pointer to the constructed list<str> column
 */
auto make_list_str_column(std::mt19937& gen, bool is_str_nullable, bool is_list_nullable)
{
  auto constexpr num_rows        = num_ordered_rows;
  auto constexpr string_per_row  = 3;
  auto constexpr num_string_rows = num_rows * string_per_row;

  // str and list<str> helpers
  std::vector<std::string> strings{
    "abc", "x", "bananas", "gpu", "minty", "backspace", "", "cayenne", "turbine", "soft"};
  std::uniform_int_distribution<int> uni(0, strings.size() - 1);
  auto string_iter = cudf::detail::make_counting_transform_iterator(
    0, [&](cudf::size_type idx) { return strings[uni(gen)]; });

  std::bernoulli_distribution bn(0.7f);
  auto string_valids = cudf::detail::make_counting_transform_iterator(
    0, [&](int index) { return is_str_nullable ? bn(gen) : true; });
  cudf::test::strings_column_wrapper string_col{
    string_iter, string_iter + num_string_rows, string_valids};

  auto offset_iter = cudf::detail::make_counting_transform_iterator(
    0, [](cudf::size_type idx) { return idx * string_per_row; });
  cudf::test::fixed_width_column_wrapper<cudf::size_type> offsets(offset_iter,
                                                                  offset_iter + num_rows + 1);

  auto list_valids =
    cudf::detail::make_counting_transform_iterator(0, [&](int index) { return index % 100; });
  auto [null_mask, null_count] = [&]() {
    if (is_list_nullable) {
      return cudf::test::detail::make_null_mask(list_valids, list_valids + num_rows);
    } else {
      return std::make_pair(rmm::device_buffer{}, 0);
    }
  }();
  return cudf::make_lists_column(
    num_rows, offsets.release(), string_col.release(), null_count, std::move(null_mask));
}

/**
 * @brief Helper to test the hybrid scan reader
 *
 * Concatenates the input table and writes it to parquet. Then reads it back using the regular and
 * the chunked hybrid scan readers. The filter expression used is as follows: table[0] >= 100. The
 * output filter and payload tables from both readers are compared with the expected table (read via
 * the mainline parquet reader) for equivalence (as nullability may be different for the read
 * tables).
 *
 *
 * @note The first column in the input table must be constructed with
 * `cudf::test::ascending<uint32_t>()`
 *
 * @tparam num_concat Number of times to concatenate the table before writing to parquet
 * @tparam num_rows Number of rows in the input table
 *
 * @param columns List of column views in the input table
 */
template <int num_concat = 2, int num_rows = num_ordered_rows>
void test_hybrid_scan(std::vector<cudf::column_view> const& columns)
{
  // Input table
  auto table    = cudf::table_view{columns};
  auto expected = cudf::concatenate(std::vector<table_view>(num_concat, table));
  table         = expected->view();
  cudf::io::table_input_metadata expected_metadata(table);
  expected_metadata.column_metadata[0].set_name("col0");

  // Parquet buffer
  std::vector<char> parquet_buffer;
  {
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
    cudf::get_current_device_resource_ref(), bloom_filter_alignment);

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
               "Filter and payload tables must have the same number of rows");
  CUDF_EXPECTS(read_filter_table_chunked->num_rows() == read_payload_table_chunked->num_rows(),
               "Chunked filter and payload tables must have the same number of rows");
  CUDF_EXPECTS(read_filter_table->num_rows() == read_filter_table_chunked->num_rows(),
               "Tables from the chunked and non-chunked hybrid scan readers must have the same "
               "number of rows");

  // Check equivalence (equal without checking nullability) with the parquet file read with the
  // original reader
  auto const options =
    cudf::io::parquet_reader_options::builder(
      cudf::io::source_info(cudf::host_span<char>(parquet_buffer.data(), parquet_buffer.size())))
      .filter(filter_expression)
      .build();
  auto [expected_tbl, expected_meta] = cudf::io::read_parquet(options, stream);

  CUDF_EXPECTS(
    expected_tbl->num_rows() == read_filter_table->num_rows(),
    "Tables read by the mainline and hybrid scan readers must have the same number of rows");

  // Check equivalence for the filter column
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected_tbl->select({0}), read_filter_table->view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected_tbl->select({0}), read_filter_table_chunked->view());

  // Check equivalence for the payload columns: [num_filter_columns, num_columns)
  auto payload_column_indices = std::vector<cudf::size_type>(columns.size() - num_filter_columns);
  std::iota(payload_column_indices.begin(), payload_column_indices.end(), num_filter_columns);
  auto const expected_payload_table = expected_tbl->select(payload_column_indices);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected_payload_table, read_payload_table->view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected_payload_table, read_payload_table_chunked->view());
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
    cudf::get_current_device_resource_ref(), bloom_filter_alignment);

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
    cudf::get_current_device_resource_ref(), bloom_filter_alignment);

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
    cudf::get_current_device_resource_ref(), bloom_filter_alignment);

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

TEST_F(HybridScanTest, MaterializeStrings)
{
  std::mt19937 gen(0xbee);

  auto const num_concat   = 2;
  auto constexpr num_rows = num_ordered_rows;

  // uint32_t(non-nullable)
  auto col0 = testdata::ascending<uint32_t>();

  // str(non-nullable)
  auto col1 = testdata::ascending<cudf::string_view>();

  // str(nullable)
  std::vector<std::string> strings{
    "abc", "x", "bananas", "gpu", "minty", "backspace", "", "cayenne", "turbine", "soft"};
  std::uniform_int_distribution<int> uni(0, strings.size() - 1);
  auto string_iter = cudf::detail::make_counting_transform_iterator(
    0, [&](cudf::size_type idx) { return strings[uni(gen)]; });
  std::bernoulli_distribution bn(0.7f);
  auto string_valids =
    cudf::detail::make_counting_transform_iterator(0, [&](int index) { return bn(gen); });
  auto col2 =
    cudf::test::strings_column_wrapper{string_iter, string_iter + num_rows, string_valids};

  test_hybrid_scan<num_concat, num_rows>({col0, col1, col2});
}

TEST_F(HybridScanTest, MaterializeLists)
{
  std::mt19937 gen(0xadd);

  auto const num_concat            = 2;
  auto constexpr num_rows          = num_ordered_rows;
  auto constexpr lists_per_row     = 2;
  auto constexpr max_vals_per_list = 3;

  // uint32_t(non-nullable)
  auto col0 = testdata::ascending<uint32_t>();

  // list<int32_t(nullable)>(nullable)
  auto col1 = make_parquet_list_col<int32_t>(gen, num_rows, max_vals_per_list, true);

  // list<list<int32_t(nullable)>(nullable)>(nullable)
  auto col2 =
    make_parquet_list_list_col<int32_t>(0, num_rows, lists_per_row, max_vals_per_list, true);

  // list<int32_t(non-nullable)>(non-nullable)
  auto col3 = make_parquet_list_col<int32_t>(gen, num_rows, max_vals_per_list, false);

  // list<list<int32_t(non-nullable)>(non-nullable)>(non-nullable)
  auto col4 =
    make_parquet_list_list_col<int32_t>(0, num_rows, lists_per_row, max_vals_per_list, false);

  // list<list<bool(nullable)>(nullable)>(nullable)
  auto col5 = make_parquet_list_list_col<bool>(0, num_rows, lists_per_row, max_vals_per_list, true);

  test_hybrid_scan<num_concat, num_rows>({col0, *col1, *col2, *col3, *col4, *col5});
}

TEST_F(HybridScanTest, MaterializeListsOfStrings)
{
  std::mt19937 gen(0xc0c0a);

  // uint32_t(non-nullable)
  auto col0 = testdata::ascending<uint32_t>();

  // list<str(non-nullable)>(non-nullable)
  auto col1 = make_list_str_column(gen, false, false);

  // list<str(non-nullable)>(nullable)
  auto col2 = make_list_str_column(gen, false, true);

  // list<str(nullable)>(non-nullable)
  auto col3 = make_list_str_column(gen, true, false);

  // list<list<str(nullable)>>(nullable)
  auto col4 = make_list_str_column(gen, true, true);

  test_hybrid_scan({col0, *col1, *col2, *col3, *col4});
}

TEST_F(HybridScanTest, MaterializeStructs)
{
  std::mt19937 gen(0xbaLL);

  auto const num_concat   = 2;
  auto constexpr num_rows = num_ordered_rows;

  // uint32_t(non-nullable)
  auto col0 = testdata::ascending<uint32_t>();

  // Validity helpers
  std::bernoulli_distribution bn(0.7f);
  auto valids =
    cudf::detail::make_counting_transform_iterator(0, [&](int index) { return bn(gen); });
  auto struct_valids_iter =
    cudf::detail::make_counting_transform_iterator(0, [&](int index) { return index % 121; });
  std::vector<bool> struct_valids(num_rows);
  std::copy(struct_valids_iter, struct_valids_iter + num_rows, struct_valids.begin());

  // strings helpers
  std::vector<std::string> strings{
    "abc", "x", "bananas", "gpu", "minty", "backspace", "", "cayenne", "turbine", "soft"};
  std::uniform_int_distribution<int> uni(0, strings.size() - 1);
  auto string_iter = cudf::detail::make_counting_transform_iterator(
    0, [&](cudf::size_type idx) { return strings[uni(gen)]; });

  // struct<list<str(nullable)>(nullable), int(nullable), float(non-nullable)>(nullable)
  auto values    = thrust::make_counting_iterator(0);
  auto col1_list = make_list_str_column(gen, true, true);
  cudf::test::fixed_width_column_wrapper<int> col1_ints(values, values + num_rows, valids);
  cudf::test::fixed_width_column_wrapper<float> col1_floats(values, values + num_rows);
  std::vector<std::unique_ptr<cudf::column>> col1_children;
  col1_children.push_back(std::move(col1_list));
  col1_children.push_back(col1_ints.release());
  col1_children.push_back(col1_floats.release());
  cudf::test::structs_column_wrapper _col1(std::move(col1_children), struct_valids);
  auto col1 = cudf::purge_nonempty_nulls(_col1);

  // struct<str(nullable), str(non-nullable), bool(nullable)>(nullable)
  auto col2_str = cudf::test::strings_column_wrapper{string_iter, string_iter + num_rows, valids};
  auto col2_str_non_nullable =
    cudf::test::strings_column_wrapper{string_iter, string_iter + num_rows};
  auto col2_bool = cudf::test::fixed_width_column_wrapper<bool>(values, values + num_rows, valids);
  std::vector<std::unique_ptr<cudf::column>> col2_children;
  col2_children.push_back(col2_str.release());
  col2_children.push_back(col2_str_non_nullable.release());
  col2_children.push_back(col2_bool.release());
  cudf::test::structs_column_wrapper _col2(std::move(col2_children));
  auto col2 = cudf::purge_nonempty_nulls(_col2);

  test_hybrid_scan<num_concat, num_rows>({col0, *col1, *col2});
}

TEST_F(HybridScanTest, MaterializeListsOfStructs)
{
  std::mt19937 gen(0xcaLL);

  auto constexpr num_concat = 2;
  auto constexpr num_rows   = num_ordered_rows;

  // uint32_t(non-nullable)
  auto col0 = testdata::ascending<uint32_t>();

  // Validity helpers
  std::bernoulli_distribution bn(0.7f);
  auto valids =
    cudf::detail::make_counting_transform_iterator(0, [&](int index) { return bn(gen); });
  auto list_valids =
    cudf::detail::make_counting_transform_iterator(0, [&](int index) { return index % 100; });
  auto struct_valids_iter =
    cudf::detail::make_counting_transform_iterator(0, [&](int index) { return index % 150; });
  std::vector<bool> struct_valids(num_rows);
  std::copy(struct_valids_iter, struct_valids_iter + num_rows, struct_valids.begin());

  // list<struct<list<str(nullable)>(nullable), int(nullable),
  // float(non-nullable)>(nullable)>(nullable)
  auto struct1_list = make_list_str_column(gen, true, true);
  auto values       = thrust::make_counting_iterator(0);
  cudf::test::fixed_width_column_wrapper<float> struct1_floats(values, values + num_rows, valids);
  std::vector<std::unique_ptr<cudf::column>> struct1_children;
  struct1_children.push_back(std::move(struct1_list));
  struct1_children.push_back(struct1_floats.release());
  cudf::test::structs_column_wrapper _struct1(std::move(struct1_children), struct_valids);
  auto struct1 = cudf::purge_nonempty_nulls(_struct1);

  auto col1_offsets_iter = thrust::counting_iterator<int32_t>(0);
  auto col1_offsets_col  = cudf::test::fixed_width_column_wrapper<int32_t>(
    col1_offsets_iter, col1_offsets_iter + num_rows + 1);
  auto [null_mask, null_count] =
    cudf::test::detail::make_null_mask(list_valids, list_valids + num_rows);
  auto col1 = cudf::make_lists_column(
    num_rows, col1_offsets_col.release(), std::move(struct1), null_count, std::move(null_mask));

  // strings helpers
  std::vector<std::string> strings{
    "abc", "x", "bananas", "gpu", "minty", "backspace", "", "cayenne", "turbine", "soft"};
  std::uniform_int_distribution<int> uni(0, strings.size() - 1);
  auto string_iter = cudf::detail::make_counting_transform_iterator(
    0, [&](cudf::size_type idx) { return strings[uni(gen)]; });

  // list<struct<str(nullable), str(non-nullable), bool(nullable)>(non-nullable)>(nullable)
  auto struct2_str =
    cudf::test::strings_column_wrapper{string_iter, string_iter + num_rows, valids};
  auto struct2_str_non_nullable =
    cudf::test::strings_column_wrapper{string_iter, string_iter + num_rows};
  auto struct2_bool =
    cudf::test::fixed_width_column_wrapper<bool>(values, values + num_rows, valids);
  std::vector<std::unique_ptr<cudf::column>> struct2_children;
  struct2_children.push_back(struct2_str.release());
  struct2_children.push_back(struct2_str_non_nullable.release());
  struct2_children.push_back(struct2_bool.release());
  cudf::test::structs_column_wrapper _struct2(std::move(struct2_children));
  auto struct2 = cudf::purge_nonempty_nulls(_struct2);

  auto col2_offsets_iter = thrust::counting_iterator<int32_t>(0);
  auto col2_offsets_col  = cudf::test::fixed_width_column_wrapper<int32_t>(
    col2_offsets_iter, col2_offsets_iter + num_rows + 1);
  std::tie(null_mask, null_count) =
    cudf::test::detail::make_null_mask(list_valids, list_valids + num_rows);
  auto col2 = cudf::make_lists_column(
    num_rows, col2_offsets_col.release(), std::move(struct2), null_count, std::move(null_mask));

  test_hybrid_scan<num_concat, num_rows>({col0, *col1, *col2});
}

TEST_F(HybridScanTest, MaterializeMixedPayloadColumns)
{
  std::mt19937 gen(0xcaffe);

  auto constexpr num_rows = num_ordered_rows;

  // uint32_t(non-nullable)
  auto col0 = testdata::ascending<uint32_t>();
  // str(non-nullable)
  auto col1 = testdata::ascending<cudf::string_view>();

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

  // list<bool(nullable)>(nullable)
  auto bools_iter = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 2; });
  auto bools_col =
    cudf::test::fixed_width_column_wrapper<bool>(bools_iter, bools_iter + num_rows, valids);
  auto offsets_iter = thrust::counting_iterator<int32_t>(0);
  auto offsets_col =
    cudf::test::fixed_width_column_wrapper<int32_t>(offsets_iter, offsets_iter + num_rows + 1);
  auto [null_mask, null_count] =
    cudf::test::detail::make_null_mask(list_valids, list_valids + num_rows);
  auto col2 = cudf::make_lists_column(
    num_rows, offsets_col.release(), bools_col.release(), null_count, std::move(null_mask));
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
        return cudf::test::detail::make_null_mask(list_valids, list_valids + num_rows);
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
  cudf::test::strings_column_wrapper string_col{string_iter, string_iter + num_string_rows, valids};
  auto offset_iter = cudf::detail::make_counting_transform_iterator(
    0, [](cudf::size_type idx) { return idx * string_per_row; });
  cudf::test::fixed_width_column_wrapper<cudf::size_type> list_offsets(
    offset_iter, offset_iter + (num_rows * lists_per_list) + 1);
  std::tie(null_mask, null_count) =
    cudf::test::detail::make_null_mask(list_valids, list_valids + (num_rows * lists_per_list));

  auto list_col = cudf::make_lists_column(num_rows * lists_per_list,
                                          list_offsets.release(),
                                          string_col.release(),
                                          null_count,
                                          std::move(null_mask));

  auto list_list_offsets_iter = cudf::detail::make_counting_transform_iterator(
    0, [](cudf::size_type idx) { return idx * lists_per_list; });
  cudf::test::fixed_width_column_wrapper<cudf::size_type> list_list_offsets(
    list_list_offsets_iter, list_list_offsets_iter + num_rows + 1);
  auto list_list_valids =
    cudf::detail::make_counting_transform_iterator(0, [&](int index) { return index % 80; });
  std::tie(null_mask, null_count) =
    cudf::test::detail::make_null_mask(list_list_valids, list_list_valids + num_rows);

  auto col9 = cudf::make_lists_column(
    num_rows, list_list_offsets.release(), std::move(list_col), null_count, std::move(null_mask));

  auto constexpr num_concat = 3;
  test_hybrid_scan<num_concat, num_rows>(
    {col0, col1, *col2, *col3, col4, *col5, *col6, *col7, *col8, *col9});
}
