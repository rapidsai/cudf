/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
#include <cudf/io/parquet.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>

#include <src/io/parquet/parquet_gpu.hpp>

#include <array>

TEST_F(ParquetReaderTest, UserBounds)
{
  // trying to read more rows than there are should result in
  // receiving the properly capped # of rows
  {
    srand(31337);
    auto expected = create_random_fixed_table<int>(4, 4, false);

    auto filepath = temp_env->get_temp_filepath("TooManyRows.parquet");
    cudf::io::parquet_writer_options args =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, *expected);
    cudf::io::write_parquet(args);

    // attempt to read more rows than there actually are
    cudf::io::parquet_reader_options read_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath}).num_rows(16);
    auto result = cudf::io::read_parquet(read_opts);

    // we should only get back 4 rows
    EXPECT_EQ(result.tbl->view().column(0).size(), 4);
  }

  // trying to read past the end of the # of actual rows should result
  // in empty columns.
  {
    srand(31337);
    auto expected = create_random_fixed_table<int>(4, 4, false);

    auto filepath = temp_env->get_temp_filepath("PastBounds.parquet");
    cudf::io::parquet_writer_options args =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, *expected);
    cudf::io::write_parquet(args);

    // attempt to read more rows than there actually are
    cudf::io::parquet_reader_options read_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath}).skip_rows(4);
    auto result = cudf::io::read_parquet(read_opts);

    // we should get empty columns back
    EXPECT_EQ(result.tbl->view().num_columns(), 4);
    EXPECT_EQ(result.tbl->view().column(0).size(), 0);
  }

  // trying to read 0 rows should result in empty columns
  {
    srand(31337);
    auto expected = create_random_fixed_table<int>(4, 4, false);

    auto filepath = temp_env->get_temp_filepath("ZeroRows.parquet");
    cudf::io::parquet_writer_options args =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, *expected);
    cudf::io::write_parquet(args);

    // attempt to read more rows than there actually are
    cudf::io::parquet_reader_options read_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath}).num_rows(0);
    auto result = cudf::io::read_parquet(read_opts);

    EXPECT_EQ(result.tbl->view().num_columns(), 4);
    EXPECT_EQ(result.tbl->view().column(0).size(), 0);
  }

  // trying to read 0 rows past the end of the # of actual rows should result
  // in empty columns.
  {
    srand(31337);
    auto expected = create_random_fixed_table<int>(4, 4, false);

    auto filepath = temp_env->get_temp_filepath("ZeroRowsPastBounds.parquet");
    cudf::io::parquet_writer_options args =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, *expected);
    cudf::io::write_parquet(args);

    // attempt to read more rows than there actually are
    cudf::io::parquet_reader_options read_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
        .skip_rows(4)
        .num_rows(0);
    auto result = cudf::io::read_parquet(read_opts);

    // we should get empty columns back
    EXPECT_EQ(result.tbl->view().num_columns(), 4);
    EXPECT_EQ(result.tbl->view().column(0).size(), 0);
  }
}

TEST_F(ParquetReaderTest, UserBoundsWithNulls)
{
  // clang-format off
  cudf::test::fixed_width_column_wrapper<float> col{{1,1,1,1,1,1,1,1, 2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3, 4,4,4,4,4,4,4,4,  5,5,5,5,5,5,5,5, 6,6,6,6,6,6,6,6, 7,7,7,7,7,7,7,7, 8,8,8,8,8,8,8,8}
                                                   ,{true,true,true,false,false,false,true,true, true,true,true,true,true,true,true,true, false,false,false,false,false,false,false,false, true,true,true,true,true,true,false,false,  true,false,true,true,true,true,true,true, true,true,true,true,true,true,true,true, true,true,true,true,true,true,true,true, true,true,true,true,true,true,true,false}};
  // clang-format on
  cudf::table_view tbl({col});
  auto filepath = temp_env->get_temp_filepath("UserBoundsWithNulls.parquet");
  cudf::io::parquet_writer_options out_args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, tbl);
  cudf::io::write_parquet(out_args);

  // skip_rows / num_rows
  // clang-format off
  std::vector<std::pair<int, int>> params{ {-1, -1}, {1, 3}, {3, -1},
                                           {31, -1}, {32, -1}, {33, -1},
                                           {31, 5}, {32, 5}, {33, 5},
                                           {-1, 7}, {-1, 31}, {-1, 32}, {-1, 33},
                                           {62, -1}, {63, -1},
                                           {62, 2}, {63, 1}};
  // clang-format on
  for (auto p : params) {
    cudf::io::parquet_reader_options read_args =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
    if (p.first >= 0) { read_args.set_skip_rows(p.first); }
    if (p.second >= 0) { read_args.set_num_rows(p.second); }
    auto result = cudf::io::read_parquet(read_args);

    p.first  = p.first < 0 ? 0 : p.first;
    p.second = p.second < 0 ? static_cast<cudf::column_view>(col).size() - p.first : p.second;
    std::vector<cudf::size_type> slice_indices{p.first, p.first + p.second};
    auto expected = cudf::slice(col, slice_indices);

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(0), expected[0]);
  }
}

TEST_F(ParquetReaderTest, UserBoundsWithNullsMixedTypes)
{
  constexpr int num_rows = 32 * 1024;

  std::mt19937 gen(6542);
  std::bernoulli_distribution bn(0.7f);
  auto valids =
    cudf::detail::make_counting_transform_iterator(0, [&](int index) { return bn(gen); });
  auto values = thrust::make_counting_iterator(0);

  // int64
  cudf::test::fixed_width_column_wrapper<int64_t> c0(values, values + num_rows, valids);

  // list<float>
  constexpr int floats_per_row = 4;
  auto c1_offset_iter          = cudf::detail::make_counting_transform_iterator(
    0, [](cudf::size_type idx) { return idx * floats_per_row; });
  cudf::test::fixed_width_column_wrapper<cudf::size_type> c1_offsets(c1_offset_iter,
                                                                     c1_offset_iter + num_rows + 1);
  cudf::test::fixed_width_column_wrapper<float> c1_floats(
    values, values + (num_rows * floats_per_row), valids);
  auto [null_mask, null_count] = cudf::test::detail::make_null_mask(valids, valids + num_rows);

  auto _c1 = cudf::make_lists_column(
    num_rows, c1_offsets.release(), c1_floats.release(), null_count, std::move(null_mask));
  auto c1 = cudf::purge_nonempty_nulls(*_c1);

  // list<list<int>>
  auto c2 = make_parquet_list_list_col<int>(0, num_rows, 5, 8, true);

  // struct<list<string>, int, float>
  std::vector<std::string> strings{
    "abc", "x", "bananas", "gpu", "minty", "backspace", "", "cayenne", "turbine", "soft"};
  std::uniform_int_distribution<int> uni(0, strings.size() - 1);
  auto string_iter = cudf::detail::make_counting_transform_iterator(
    0, [&](cudf::size_type idx) { return strings[uni(gen)]; });
  constexpr int string_per_row  = 3;
  constexpr int num_string_rows = num_rows * string_per_row;
  cudf::test::strings_column_wrapper string_col{string_iter, string_iter + num_string_rows};
  auto offset_iter = cudf::detail::make_counting_transform_iterator(
    0, [](cudf::size_type idx) { return idx * string_per_row; });
  cudf::test::fixed_width_column_wrapper<cudf::size_type> offsets(offset_iter,
                                                                  offset_iter + num_rows + 1);

  auto _c3_valids =
    cudf::detail::make_counting_transform_iterator(0, [&](int index) { return index % 200; });
  std::vector<bool> c3_valids(num_rows);
  std::copy(_c3_valids, _c3_valids + num_rows, c3_valids.begin());
  std::tie(null_mask, null_count) = cudf::test::detail::make_null_mask(valids, valids + num_rows);
  auto _c3_list                   = cudf::make_lists_column(
    num_rows, offsets.release(), string_col.release(), null_count, std::move(null_mask));
  auto c3_list = cudf::purge_nonempty_nulls(*_c3_list);
  cudf::test::fixed_width_column_wrapper<int> c3_ints(values, values + num_rows, valids);
  cudf::test::fixed_width_column_wrapper<float> c3_floats(values, values + num_rows, valids);
  std::vector<std::unique_ptr<cudf::column>> c3_children;
  c3_children.push_back(std::move(c3_list));
  c3_children.push_back(c3_ints.release());
  c3_children.push_back(c3_floats.release());
  cudf::test::structs_column_wrapper _c3(std::move(c3_children), c3_valids);
  auto c3 = cudf::purge_nonempty_nulls(_c3);

  // write it out
  cudf::table_view tbl({c0, *c1, *c2, *c3});
  auto filepath = temp_env->get_temp_filepath("UserBoundsWithNullsMixedTypes.parquet");
  cudf::io::parquet_writer_options out_args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, tbl);
  cudf::io::write_parquet(out_args);

  // read it back
  std::vector<std::pair<int, int>> params{
    {-1, -1}, {0, num_rows}, {1, num_rows - 1}, {num_rows - 1, 1}, {517, 22000}};
  for (auto p : params) {
    cudf::io::parquet_reader_options read_args =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
    if (p.first >= 0) { read_args.set_skip_rows(p.first); }
    if (p.second >= 0) { read_args.set_num_rows(p.second); }
    auto result = cudf::io::read_parquet(read_args);

    p.first  = p.first < 0 ? 0 : p.first;
    p.second = p.second < 0 ? num_rows - p.first : p.second;
    std::vector<cudf::size_type> slice_indices{p.first, p.first + p.second};
    auto expected = cudf::slice(tbl, slice_indices);

    CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, expected[0]);
  }
}

TEST_F(ParquetReaderTest, UserBoundsWithNullsLarge)
{
  constexpr int num_rows = 30 * 10000;

  std::mt19937 gen(6747);
  std::bernoulli_distribution bn(0.7f);
  auto valids =
    cudf::detail::make_counting_transform_iterator(0, [&](int index) { return bn(gen); });
  auto values = thrust::make_counting_iterator(0);

  cudf::test::fixed_width_column_wrapper<int> col(values, values + num_rows, valids);

  // this file will have row groups of 10,000 each
  cudf::table_view tbl({col});
  auto filepath = temp_env->get_temp_filepath("UserBoundsWithNullsLarge.parquet");
  cudf::io::parquet_writer_options out_args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, tbl)
      .row_group_size_rows(10000)
      .max_page_size_rows(1000);
  cudf::io::write_parquet(out_args);

  // skip_rows / num_rows
  // clang-format off
  std::vector<std::pair<int, int>> params{ {-1, -1}, {31, -1}, {32, -1}, {33, -1}, {16130, -1}, {19999, -1},
                                           {31, 1}, {32, 1}, {33, 1},
                                           // deliberately span some row group boundaries
                                           {9900, 1001}, {9900, 2000}, {29999, 2}, {139997, -1},
                                           {167878, 3}, {229976, 31},
                                           {240031, 17}, {290001, 9899}, {299999, 1} };
  // clang-format on
  for (auto p : params) {
    cudf::io::parquet_reader_options read_args =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
    if (p.first >= 0) { read_args.set_skip_rows(p.first); }
    if (p.second >= 0) { read_args.set_num_rows(p.second); }
    auto result = cudf::io::read_parquet(read_args);

    p.first  = p.first < 0 ? 0 : p.first;
    p.second = p.second < 0 ? static_cast<cudf::column_view>(col).size() - p.first : p.second;
    std::vector<cudf::size_type> slice_indices{p.first, p.first + p.second};
    auto expected = cudf::slice(col, slice_indices);

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(0), expected[0]);
  }
}

TEST_F(ParquetReaderTest, ListUserBoundsWithNullsLarge)
{
  constexpr int num_rows = 5 * 10000;
  auto colp              = make_parquet_list_list_col<int>(0, num_rows, 5, 8, true);
  cudf::column_view col  = *colp;

  // this file will have row groups of 10,000 each
  cudf::table_view tbl({col});
  auto filepath = temp_env->get_temp_filepath("ListUserBoundsWithNullsLarge.parquet");
  cudf::io::parquet_writer_options out_args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, tbl)
      .row_group_size_rows(10000)
      .max_page_size_rows(1000);
  cudf::io::write_parquet(out_args);

  // skip_rows / num_rows
  // clang-format off
  std::vector<std::pair<int, int>> params{ {-1, -1}, {31, -1}, {32, -1}, {33, -1}, {1670, -1}, {44997, -1},
                                           {31, 1}, {32, 1}, {33, 1},
                                           // deliberately span some row group boundaries
                                           {9900, 1001}, {9900, 2000}, {29999, 2},
                                           {16567, 3}, {42976, 31},
                                           {40231, 17}, {19000, 9899}, {49999, 1} };
  // clang-format on
  for (auto p : params) {
    cudf::io::parquet_reader_options read_args =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
    if (p.first >= 0) { read_args.set_skip_rows(p.first); }
    if (p.second >= 0) { read_args.set_num_rows(p.second); }
    auto result = cudf::io::read_parquet(read_args);

    p.first  = p.first < 0 ? 0 : p.first;
    p.second = p.second < 0 ? static_cast<cudf::column_view>(col).size() - p.first : p.second;
    std::vector<cudf::size_type> slice_indices{p.first, p.first + p.second};
    auto expected = cudf::slice(col, slice_indices);

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(0), expected[0]);
  }
}

TEST_F(ParquetReaderTest, ReorderedColumns)
{
  {
    auto a = cudf::test::strings_column_wrapper{{"a", "", "c"}, {true, false, true}};
    auto b = cudf::test::fixed_width_column_wrapper<int>{1, 2, 3};

    cudf::table_view tbl{{a, b}};
    auto filepath = temp_env->get_temp_filepath("ReorderedColumns.parquet");
    cudf::io::table_input_metadata md(tbl);
    md.column_metadata[0].set_name("a");
    md.column_metadata[1].set_name("b");
    cudf::io::parquet_writer_options opts =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, tbl).metadata(md);
    cudf::io::write_parquet(opts);

    // read them out of order
    cudf::io::parquet_reader_options read_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
        .columns({"b", "a"});
    auto result = cudf::io::read_parquet(read_opts);

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(0), b);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(1), a);
  }

  {
    auto a = cudf::test::fixed_width_column_wrapper<int>{1, 2, 3};
    auto b = cudf::test::strings_column_wrapper{{"a", "", "c"}, {true, false, true}};

    cudf::table_view tbl{{a, b}};
    auto filepath = temp_env->get_temp_filepath("ReorderedColumns2.parquet");
    cudf::io::table_input_metadata md(tbl);
    md.column_metadata[0].set_name("a");
    md.column_metadata[1].set_name("b");
    cudf::io::parquet_writer_options opts =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, tbl).metadata(md);
    cudf::io::write_parquet(opts);

    // read them out of order
    cudf::io::parquet_reader_options read_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
        .columns({"b", "a"});
    auto result = cudf::io::read_parquet(read_opts);

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(0), b);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(1), a);
  }

  auto a = cudf::test::fixed_width_column_wrapper<int>{1, 2, 3, 10, 20, 30};
  auto b = cudf::test::strings_column_wrapper{{"a", "", "c", "cats", "dogs", "owls"},
                                              {true, false, true, true, false, true}};
  auto c = cudf::test::fixed_width_column_wrapper<int>{{15, 16, 17, 25, 26, 32},
                                                       {false, true, true, true, true, false}};
  auto d = cudf::test::strings_column_wrapper{"ducks", "sheep", "cows", "fish", "birds", "ants"};

  cudf::table_view tbl{{a, b, c, d}};
  auto filepath = temp_env->get_temp_filepath("ReorderedColumns3.parquet");
  cudf::io::table_input_metadata md(tbl);
  md.column_metadata[0].set_name("a");
  md.column_metadata[1].set_name("b");
  md.column_metadata[2].set_name("c");
  md.column_metadata[3].set_name("d");
  cudf::io::parquet_writer_options opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, tbl)
      .metadata(std::move(md));
  cudf::io::write_parquet(opts);

  {
    // read them out of order
    cudf::io::parquet_reader_options read_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
        .columns({"d", "a", "b", "c"});
    auto result = cudf::io::read_parquet(read_opts);

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(0), d);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(1), a);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(2), b);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(3), c);
  }

  {
    // read them out of order
    cudf::io::parquet_reader_options read_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
        .columns({"c", "d", "a", "b"});
    auto result = cudf::io::read_parquet(read_opts);

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(0), c);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(1), d);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(2), a);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(3), b);
  }

  {
    // read them out of order
    cudf::io::parquet_reader_options read_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
        .columns({"d", "c", "b", "a"});
    auto result = cudf::io::read_parquet(read_opts);

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(0), d);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(1), c);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(2), b);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(3), a);
  }
}

TEST_F(ParquetReaderTest, SelectNestedColumn)
{
  // Struct<is_human:bool,
  //        Struct<weight:float,
  //               ages:int,
  //               land_unit:List<int>>,
  //               flats:List<List<int>>
  //              >
  //       >

  auto weights_col = cudf::test::fixed_width_column_wrapper<float>{1.1, 2.4, 5.3, 8.0, 9.6, 6.9};

  auto ages_col = cudf::test::fixed_width_column_wrapper<int32_t>{
    {48, 27, 25, 31, 351, 351}, {true, true, true, true, true, false}};

  auto struct_1 = cudf::test::structs_column_wrapper{{weights_col, ages_col},
                                                     {true, true, true, true, false, true}};

  auto is_human_col = cudf::test::fixed_width_column_wrapper<bool>{
    {true, true, false, false, false, false}, {true, true, false, true, true, false}};

  auto struct_2 = cudf::test::structs_column_wrapper{{is_human_col, struct_1},
                                                     {false, true, true, true, true, true}}
                    .release();

  auto input = table_view({*struct_2});

  cudf::io::table_input_metadata input_metadata(input);
  input_metadata.column_metadata[0].set_name("being");
  input_metadata.column_metadata[0].child(0).set_name("human?");
  input_metadata.column_metadata[0].child(1).set_name("particulars");
  input_metadata.column_metadata[0].child(1).child(0).set_name("weight");
  input_metadata.column_metadata[0].child(1).child(1).set_name("age");

  auto filepath = temp_env->get_temp_filepath("SelectNestedColumn.parquet");
  cudf::io::parquet_writer_options args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, input)
      .metadata(std::move(input_metadata));
  cudf::io::write_parquet(args);

  {  // Test selecting a single leaf from the table
    cudf::io::parquet_reader_options read_args =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info(filepath))
        .columns({"being.particulars.age"});
    auto const result = cudf::io::read_parquet(read_args);

    auto expect_ages_col = cudf::test::fixed_width_column_wrapper<int32_t>{
      {48, 27, 25, 31, 351, 351}, {true, true, true, true, true, false}};
    auto expect_s_1 =
      cudf::test::structs_column_wrapper{{expect_ages_col}, {true, true, true, true, false, true}};
    auto expect_s_2 =
      cudf::test::structs_column_wrapper{{expect_s_1}, {false, true, true, true, true, true}}
        .release();
    auto expected = table_view({*expect_s_2});

    cudf::io::table_input_metadata expected_metadata(expected);
    expected_metadata.column_metadata[0].set_name("being");
    expected_metadata.column_metadata[0].child(0).set_name("particulars");
    expected_metadata.column_metadata[0].child(0).child(0).set_name("age");

    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
    cudf::test::expect_metadata_equal(expected_metadata, result.metadata);
  }

  {  // Test selecting a non-leaf and expecting all hierarchy from that node onwards
    cudf::io::parquet_reader_options read_args =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info(filepath))
        .columns({"being.particulars"});
    auto const result = cudf::io::read_parquet(read_args);

    auto expected_weights_col =
      cudf::test::fixed_width_column_wrapper<float>{1.1, 2.4, 5.3, 8.0, 9.6, 6.9};

    auto expected_ages_col = cudf::test::fixed_width_column_wrapper<int32_t>{
      {48, 27, 25, 31, 351, 351}, {true, true, true, true, true, false}};

    auto expected_s_1 = cudf::test::structs_column_wrapper{
      {expected_weights_col, expected_ages_col}, {true, true, true, true, false, true}};

    auto expect_s_2 =
      cudf::test::structs_column_wrapper{{expected_s_1}, {false, true, true, true, true, true}}
        .release();
    auto expected = table_view({*expect_s_2});

    cudf::io::table_input_metadata expected_metadata(expected);
    expected_metadata.column_metadata[0].set_name("being");
    expected_metadata.column_metadata[0].child(0).set_name("particulars");
    expected_metadata.column_metadata[0].child(0).child(0).set_name("weight");
    expected_metadata.column_metadata[0].child(0).child(1).set_name("age");

    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
    cudf::test::expect_metadata_equal(expected_metadata, result.metadata);
  }

  {  // Test selecting struct children out of order
    cudf::io::parquet_reader_options read_args =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info(filepath))
        .columns({"being.particulars.age", "being.particulars.weight", "being.human?"});
    auto const result = cudf::io::read_parquet(read_args);

    auto expected_weights_col =
      cudf::test::fixed_width_column_wrapper<float>{1.1, 2.4, 5.3, 8.0, 9.6, 6.9};

    auto expected_ages_col = cudf::test::fixed_width_column_wrapper<int32_t>{
      {48, 27, 25, 31, 351, 351}, {true, true, true, true, true, false}};

    auto expected_is_human_col = cudf::test::fixed_width_column_wrapper<bool>{
      {true, true, false, false, false, false}, {true, true, false, true, true, false}};

    auto expect_s_1 = cudf::test::structs_column_wrapper{{expected_ages_col, expected_weights_col},
                                                         {true, true, true, true, false, true}};

    auto expect_s_2 = cudf::test::structs_column_wrapper{{expect_s_1, expected_is_human_col},
                                                         {false, true, true, true, true, true}}
                        .release();

    auto expected = table_view({*expect_s_2});

    cudf::io::table_input_metadata expected_metadata(expected);
    expected_metadata.column_metadata[0].set_name("being");
    expected_metadata.column_metadata[0].child(0).set_name("particulars");
    expected_metadata.column_metadata[0].child(0).child(0).set_name("age");
    expected_metadata.column_metadata[0].child(0).child(1).set_name("weight");
    expected_metadata.column_metadata[0].child(1).set_name("human?");

    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
    cudf::test::expect_metadata_equal(expected_metadata, result.metadata);
  }
}

TEST_F(ParquetReaderTest, DecimalRead)
{
  {
    /* We could add a dataset to include this file, but we don't want tests in cudf to have data.
       This test is a temporary test until python gains the ability to write decimal, so we're
       embedding
       a parquet file directly into the code here to prevent issues with finding the file */
    constexpr unsigned int decimals_parquet_len = 2366;
    std::array<unsigned char, decimals_parquet_len> const decimals_parquet{
      0x50, 0x41, 0x52, 0x31, 0x15, 0x00, 0x15, 0xb0, 0x03, 0x15, 0xb8, 0x03, 0x2c, 0x15, 0x6a,
      0x15, 0x00, 0x15, 0x06, 0x15, 0x08, 0x1c, 0x36, 0x02, 0x28, 0x04, 0x7f, 0x96, 0x98, 0x00,
      0x18, 0x04, 0x81, 0x69, 0x67, 0xff, 0x00, 0x00, 0x00, 0xd8, 0x01, 0xf0, 0xd7, 0x04, 0x00,
      0x00, 0x00, 0x64, 0x01, 0x03, 0x06, 0x68, 0x12, 0xdc, 0xff, 0xbd, 0x18, 0xfd, 0xff, 0x64,
      0x13, 0x80, 0x00, 0xb3, 0x5d, 0x62, 0x00, 0x90, 0x35, 0xa9, 0xff, 0xa2, 0xde, 0xe3, 0xff,
      0xe9, 0xbf, 0x96, 0xff, 0x1f, 0x8a, 0x98, 0xff, 0xb1, 0x50, 0x34, 0x00, 0x88, 0x24, 0x59,
      0x00, 0x2a, 0x33, 0xbe, 0xff, 0xd5, 0x16, 0xbc, 0xff, 0x13, 0x50, 0x8d, 0xff, 0xcb, 0x63,
      0x2d, 0x00, 0x80, 0x8f, 0xbe, 0xff, 0x82, 0x40, 0x10, 0x00, 0x84, 0x68, 0x70, 0xff, 0x9b,
      0x69, 0x78, 0x00, 0x14, 0x6c, 0x10, 0x00, 0x50, 0xd9, 0xe1, 0xff, 0xaa, 0xcd, 0x6a, 0x00,
      0xcf, 0xb1, 0x28, 0x00, 0x77, 0x57, 0x8d, 0x00, 0xee, 0x05, 0x79, 0x00, 0xf0, 0x15, 0xeb,
      0xff, 0x02, 0xe2, 0x06, 0x00, 0x87, 0x43, 0x86, 0x00, 0xf8, 0x2d, 0x2e, 0x00, 0xee, 0x2e,
      0x98, 0xff, 0x39, 0xcb, 0x4d, 0x00, 0x1e, 0x6b, 0xea, 0xff, 0x80, 0x8e, 0x6c, 0xff, 0x97,
      0x25, 0x26, 0x00, 0x4d, 0x0d, 0x0a, 0x00, 0xca, 0x64, 0x7f, 0x00, 0xf4, 0xbe, 0xa1, 0xff,
      0xe2, 0x12, 0x6c, 0xff, 0xbd, 0x77, 0xae, 0xff, 0xf9, 0x4b, 0x36, 0x00, 0xb0, 0xe3, 0x79,
      0xff, 0xa2, 0x2a, 0x29, 0x00, 0xcd, 0x06, 0xbc, 0xff, 0x2d, 0xa3, 0x7e, 0x00, 0xa9, 0x08,
      0xa1, 0xff, 0xbf, 0x81, 0xd0, 0xff, 0x4f, 0x03, 0x73, 0x00, 0xb0, 0x99, 0x0c, 0x00, 0xbd,
      0x6f, 0xf8, 0xff, 0x6b, 0x02, 0x05, 0x00, 0xc1, 0xe1, 0xba, 0xff, 0x81, 0x69, 0x67, 0xff,
      0x7f, 0x96, 0x98, 0x00, 0x15, 0x00, 0x15, 0xd0, 0x06, 0x15, 0xda, 0x06, 0x2c, 0x15, 0x6a,
      0x15, 0x00, 0x15, 0x06, 0x15, 0x08, 0x1c, 0x36, 0x02, 0x28, 0x08, 0xff, 0x3f, 0x7a, 0x10,
      0xf3, 0x5a, 0x00, 0x00, 0x18, 0x08, 0x01, 0xc0, 0x85, 0xef, 0x0c, 0xa5, 0xff, 0xff, 0x00,
      0x00, 0x00, 0xa8, 0x03, 0xf4, 0xa7, 0x01, 0x04, 0x00, 0x00, 0x00, 0x64, 0x01, 0x03, 0x06,
      0x55, 0x6f, 0xc5, 0xe4, 0x9f, 0x1a, 0x00, 0x00, 0x47, 0x89, 0x0a, 0xe8, 0x58, 0xf0, 0xff,
      0xff, 0x63, 0xee, 0x21, 0xdd, 0xdd, 0xca, 0xff, 0xff, 0xbe, 0x6f, 0x3b, 0xaa, 0xe9, 0x3d,
      0x00, 0x00, 0xd6, 0x91, 0x2a, 0xb7, 0x08, 0x02, 0x00, 0x00, 0x75, 0x45, 0x2c, 0xd7, 0x76,
      0x0c, 0x00, 0x00, 0x54, 0x49, 0x92, 0x44, 0x9c, 0xbf, 0xff, 0xff, 0x41, 0xa9, 0x6d, 0xec,
      0x7a, 0xd0, 0xff, 0xff, 0x27, 0xa0, 0x23, 0x41, 0x44, 0xc1, 0xff, 0xff, 0x18, 0xd4, 0xe1,
      0x30, 0xd3, 0xe0, 0xff, 0xff, 0x59, 0xac, 0x14, 0xf4, 0xec, 0x58, 0x00, 0x00, 0x2c, 0x17,
      0x29, 0x57, 0x44, 0x13, 0x00, 0x00, 0xa2, 0x0d, 0x4a, 0xcc, 0x63, 0xff, 0xff, 0xff, 0x81,
      0x33, 0xbc, 0xda, 0xd5, 0xda, 0xff, 0xff, 0x4c, 0x05, 0xf4, 0x78, 0x19, 0xea, 0xff, 0xff,
      0x06, 0x71, 0x25, 0xde, 0x5a, 0xaf, 0xff, 0xff, 0x95, 0x32, 0x5f, 0x76, 0x98, 0xb3, 0xff,
      0xff, 0xf1, 0x34, 0x3c, 0xbf, 0xa8, 0xbe, 0xff, 0xff, 0x27, 0x73, 0x40, 0x0c, 0x7d, 0xcd,
      0xff, 0xff, 0x68, 0xa9, 0xc2, 0xe9, 0x2c, 0x03, 0x00, 0x00, 0x3f, 0x79, 0xd9, 0x04, 0x8c,
      0xe5, 0xff, 0xff, 0x91, 0xb4, 0x9b, 0xe3, 0x8f, 0x21, 0x00, 0x00, 0xb8, 0x20, 0xc8, 0xc2,
      0x4d, 0xa6, 0xff, 0xff, 0x47, 0xfa, 0xde, 0x36, 0x4a, 0xf3, 0xff, 0xff, 0x72, 0x80, 0x94,
      0x59, 0xdd, 0x4e, 0x00, 0x00, 0x29, 0xe4, 0xd6, 0x43, 0xb0, 0xf0, 0xff, 0xff, 0x68, 0x36,
      0xbc, 0x2d, 0xd1, 0xa9, 0xff, 0xff, 0xbc, 0xe4, 0xbe, 0xd7, 0xed, 0x1b, 0x00, 0x00, 0x02,
      0x8b, 0xcb, 0xd7, 0xed, 0x47, 0x00, 0x00, 0x3c, 0x06, 0xe4, 0xda, 0xc7, 0x47, 0x00, 0x00,
      0xf3, 0x39, 0x55, 0x28, 0x97, 0xba, 0xff, 0xff, 0x07, 0x79, 0x38, 0x4e, 0xe0, 0x21, 0x00,
      0x00, 0xde, 0xed, 0x1c, 0x23, 0x09, 0x49, 0x00, 0x00, 0x49, 0x46, 0x49, 0x5d, 0x8f, 0x34,
      0x00, 0x00, 0x38, 0x18, 0x50, 0xf6, 0xa1, 0x11, 0x00, 0x00, 0xdf, 0xb8, 0x19, 0x14, 0xd1,
      0xe1, 0xff, 0xff, 0x2c, 0x56, 0x72, 0x93, 0x64, 0x3f, 0x00, 0x00, 0x1c, 0xe0, 0xbe, 0x87,
      0x7d, 0xf9, 0xff, 0xff, 0x73, 0x0e, 0x3c, 0x01, 0x91, 0xf9, 0xff, 0xff, 0xb2, 0x37, 0x85,
      0x81, 0x5f, 0x54, 0x00, 0x00, 0x58, 0x44, 0xb0, 0x1a, 0xac, 0xbb, 0xff, 0xff, 0x36, 0xbf,
      0xbe, 0x5e, 0x22, 0xff, 0xff, 0xff, 0x06, 0x20, 0xa0, 0x23, 0x0d, 0x3b, 0x00, 0x00, 0x19,
      0xc6, 0x49, 0x0a, 0x00, 0xcf, 0xff, 0xff, 0x4f, 0xcd, 0xc6, 0x95, 0x4b, 0xf1, 0xff, 0xff,
      0xa3, 0x59, 0xaf, 0x65, 0xec, 0xe9, 0xff, 0xff, 0x58, 0xef, 0x05, 0x50, 0x63, 0xe4, 0xff,
      0xff, 0xc7, 0x6a, 0x9e, 0xf1, 0x69, 0x20, 0x00, 0x00, 0xd1, 0xb3, 0xc9, 0x14, 0xb2, 0x29,
      0x00, 0x00, 0x1d, 0x48, 0x16, 0x70, 0xf0, 0x40, 0x00, 0x00, 0x01, 0xc0, 0x85, 0xef, 0x0c,
      0xa5, 0xff, 0xff, 0xff, 0x3f, 0x7a, 0x10, 0xf3, 0x5a, 0x00, 0x00, 0x15, 0x00, 0x15, 0x90,
      0x0d, 0x15, 0x9a, 0x0d, 0x2c, 0x15, 0x6a, 0x15, 0x00, 0x15, 0x06, 0x15, 0x08, 0x1c, 0x36,
      0x02, 0x28, 0x10, 0x4b, 0x3b, 0x4c, 0xa8, 0x5a, 0x86, 0xc4, 0x7a, 0x09, 0x8a, 0x22, 0x3f,
      0xff, 0xff, 0xff, 0xff, 0x18, 0x10, 0xb4, 0xc4, 0xb3, 0x57, 0xa5, 0x79, 0x3b, 0x85, 0xf6,
      0x75, 0xdd, 0xc0, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0xc8, 0x06, 0xf4, 0x47, 0x03,
      0x04, 0x00, 0x00, 0x00, 0x64, 0x01, 0x03, 0x06, 0x05, 0x49, 0xf7, 0xfc, 0x89, 0x3d, 0x3e,
      0x20, 0x07, 0x72, 0x3e, 0xa1, 0x66, 0x81, 0x67, 0x80, 0x23, 0x78, 0x06, 0x68, 0x0e, 0x78,
      0xf5, 0x08, 0xed, 0x20, 0xcd, 0x0e, 0x7f, 0x9c, 0x70, 0xa0, 0xb9, 0x16, 0x44, 0xb2, 0x41,
      0x62, 0xba, 0x82, 0xad, 0xe1, 0x12, 0x9b, 0xa6, 0x53, 0x8d, 0x20, 0x27, 0xd5, 0x84, 0x63,
      0xb8, 0x07, 0x4b, 0x5b, 0xa4, 0x1c, 0xa4, 0x1c, 0x17, 0xbf, 0x4b, 0x00, 0x24, 0x04, 0x56,
      0xa8, 0x52, 0xaf, 0x33, 0xf7, 0xad, 0x7c, 0xc8, 0x83, 0x25, 0x13, 0xaf, 0x80, 0x25, 0x6f,
      0xbd, 0xd1, 0x15, 0x69, 0x64, 0x20, 0x7b, 0xd7, 0x33, 0xba, 0x66, 0x29, 0x8a, 0x00, 0xda,
      0x42, 0x07, 0x2c, 0x6c, 0x39, 0x76, 0x9f, 0xdc, 0x17, 0xad, 0xb6, 0x58, 0xdf, 0x5f, 0x00,
      0x18, 0x3a, 0xae, 0x1c, 0xd6, 0x5f, 0x9d, 0x78, 0x8d, 0x73, 0xdd, 0x3e, 0xd6, 0x18, 0x33,
      0x40, 0xe4, 0x36, 0xde, 0xb0, 0xb7, 0x33, 0x2a, 0x6b, 0x08, 0x03, 0x6c, 0x6d, 0x8f, 0x13,
      0x93, 0xd0, 0xd7, 0x87, 0x62, 0x63, 0x53, 0xfb, 0xd8, 0xbb, 0xc9, 0x54, 0x90, 0xd6, 0xa9,
      0x8f, 0xc8, 0x60, 0xbd, 0xec, 0x75, 0x23, 0x9a, 0x21, 0xec, 0xe4, 0x86, 0x43, 0xd7, 0xc1,
      0x88, 0xdc, 0x82, 0x00, 0x32, 0x79, 0xc9, 0x2b, 0x70, 0x85, 0xb7, 0x25, 0xa1, 0xcc, 0x7d,
      0x0b, 0x29, 0x03, 0xea, 0x80, 0xff, 0x9b, 0xf3, 0x24, 0x7f, 0xd1, 0xff, 0xf0, 0x22, 0x65,
      0x85, 0x99, 0x17, 0x63, 0xc2, 0xc0, 0xb7, 0x62, 0x05, 0xda, 0x7a, 0xa0, 0xc3, 0x2a, 0x6f,
      0x1f, 0xee, 0x1f, 0x31, 0xa8, 0x42, 0x80, 0xe4, 0xb7, 0x6c, 0xf6, 0xac, 0x47, 0xb0, 0x17,
      0x69, 0xcb, 0xff, 0x66, 0x8a, 0xd6, 0x25, 0x00, 0xf3, 0xcf, 0x0a, 0xaf, 0xf8, 0x92, 0x8a,
      0xa0, 0xdf, 0x71, 0x13, 0x8d, 0x9d, 0xff, 0x7e, 0xe0, 0x0a, 0x52, 0xf1, 0x97, 0x01, 0xa9,
      0x73, 0x27, 0xfd, 0x63, 0x58, 0x00, 0x32, 0xa6, 0xf6, 0x78, 0xb8, 0xe4, 0xfd, 0x20, 0x7c,
      0x90, 0xee, 0xad, 0x8c, 0xc9, 0x71, 0x35, 0x66, 0x71, 0x3c, 0xe0, 0xe4, 0x0b, 0xbb, 0xa0,
      0x50, 0xe9, 0xf2, 0x81, 0x1d, 0x3a, 0x95, 0x94, 0x00, 0xd5, 0x49, 0x00, 0x07, 0xdf, 0x21,
      0x53, 0x36, 0x8d, 0x9e, 0xd9, 0xa5, 0x52, 0x4d, 0x0d, 0x29, 0x74, 0xf0, 0x40, 0xbd, 0xda,
      0x63, 0x4e, 0xdd, 0x91, 0x8e, 0xa6, 0xa7, 0xf6, 0x78, 0x58, 0x3b, 0x0a, 0x5c, 0x60, 0x3c,
      0x15, 0x34, 0xf8, 0x2c, 0x21, 0xe3, 0x56, 0x1b, 0x9e, 0xd9, 0x56, 0xd3, 0x13, 0x2e, 0x80,
      0x2c, 0x36, 0xda, 0x1d, 0xc8, 0xfb, 0x52, 0xee, 0x17, 0xb3, 0x2b, 0xf3, 0xd2, 0xeb, 0x29,
      0xa0, 0x37, 0xa0, 0x12, 0xce, 0x1c, 0x50, 0x6a, 0xf4, 0x11, 0xcd, 0x96, 0x88, 0x3f, 0x43,
      0x78, 0xc0, 0x2c, 0x53, 0x6c, 0xa6, 0xdf, 0xb9, 0x9e, 0x93, 0xd4, 0x1e, 0xa9, 0x7f, 0x67,
      0xa6, 0xc1, 0x80, 0x46, 0x0f, 0x63, 0x7d, 0x15, 0xf2, 0x4c, 0xc5, 0xda, 0x11, 0x9a, 0x20,
      0x67, 0x27, 0xe8, 0x00, 0xec, 0x03, 0x1d, 0x15, 0xa7, 0x92, 0xb3, 0x1f, 0xda, 0x20, 0x92,
      0xd8, 0x00, 0xfb, 0x06, 0x80, 0xeb, 0x4b, 0x0c, 0xc1, 0x1f, 0x49, 0x40, 0x06, 0x8d, 0x8a,
      0xf8, 0x34, 0xb1, 0x0c, 0x1d, 0x20, 0xd0, 0x47, 0xe5, 0xb1, 0x7e, 0xf7, 0xe4, 0xb4, 0x7e,
      0x9c, 0x84, 0x18, 0x61, 0x32, 0x4f, 0xc0, 0xc2, 0xb2, 0xcc, 0x63, 0xf6, 0xe1, 0x16, 0xd6,
      0xd9, 0x4b, 0x74, 0x13, 0x01, 0xa1, 0xe2, 0x00, 0xb7, 0x9e, 0xc1, 0x3a, 0xc5, 0xaf, 0xe8,
      0x54, 0x07, 0x2a, 0x20, 0xfd, 0x2c, 0x6f, 0xb9, 0x80, 0x18, 0x92, 0x87, 0xa0, 0x81, 0x24,
      0x60, 0x47, 0x17, 0x4f, 0xbc, 0xbe, 0xf5, 0x03, 0x69, 0x80, 0xe3, 0x10, 0x54, 0xd6, 0x68,
      0x7d, 0x75, 0xd3, 0x0a, 0x45, 0x38, 0x9e, 0xa9, 0xfd, 0x05, 0x40, 0xd2, 0x1e, 0x6f, 0x5c,
      0x30, 0x10, 0xfe, 0x9b, 0x9f, 0x6d, 0xc0, 0x9d, 0x6c, 0x17, 0x7d, 0x00, 0x09, 0xb6, 0x8a,
      0x31, 0x8e, 0x1b, 0x6b, 0x84, 0x1e, 0x79, 0xce, 0x10, 0x55, 0x59, 0x6a, 0x40, 0x16, 0xdc,
      0x9a, 0xcf, 0x4d, 0xb0, 0x8f, 0xac, 0xe3, 0x8d, 0xee, 0xd2, 0xef, 0x01, 0x8c, 0xe0, 0x2b,
      0x24, 0xe5, 0xb4, 0xe1, 0x86, 0x72, 0x00, 0x30, 0x07, 0xce, 0x02, 0x23, 0x41, 0x33, 0x40,
      0xf0, 0x9b, 0xc2, 0x2d, 0x30, 0xec, 0x3b, 0x17, 0xb2, 0x8f, 0x64, 0x7d, 0xcd, 0x70, 0x9e,
      0x80, 0x22, 0xb5, 0xdf, 0x6d, 0x2a, 0x43, 0xd4, 0x2b, 0x5a, 0xf6, 0x96, 0xa6, 0xea, 0x91,
      0x62, 0x80, 0x39, 0xf2, 0x5a, 0x8e, 0xc0, 0xb9, 0x29, 0x99, 0x17, 0xe7, 0x35, 0x2c, 0xf6,
      0x4d, 0x18, 0x00, 0x48, 0x10, 0x85, 0xb4, 0x3f, 0x89, 0x60, 0x49, 0x6e, 0xf0, 0xcd, 0x9d,
      0x92, 0xeb, 0x96, 0x80, 0xcf, 0xf9, 0xf1, 0x46, 0x1d, 0xc0, 0x49, 0xb3, 0x36, 0x2e, 0x24,
      0xc8, 0xdb, 0x41, 0x72, 0x20, 0xf5, 0xde, 0x5c, 0xf9, 0x4a, 0x6e, 0xa0, 0x0b, 0x13, 0xfc,
      0x2d, 0x17, 0x07, 0x16, 0x5e, 0x00, 0x3c, 0x54, 0x41, 0x0e, 0xa2, 0x0d, 0xf3, 0x48, 0x12,
      0x2e, 0x7c, 0xab, 0x3c, 0x59, 0x1c, 0x40, 0xca, 0xb0, 0x71, 0xc7, 0x29, 0xf0, 0xbb, 0x9f,
      0xf4, 0x3f, 0x25, 0x49, 0xad, 0xc2, 0x8f, 0x80, 0x04, 0x38, 0x6d, 0x35, 0x02, 0xca, 0xe6,
      0x02, 0x83, 0x89, 0x4e, 0x74, 0xdb, 0x08, 0x5a, 0x80, 0x13, 0x99, 0xd4, 0x26, 0xc1, 0x27,
      0xce, 0xb0, 0x98, 0x99, 0xca, 0xf6, 0x3e, 0x50, 0x49, 0xd0, 0xbf, 0xcb, 0x6f, 0xbe, 0x5b,
      0x92, 0x63, 0xde, 0x94, 0xd3, 0x8f, 0x07, 0x06, 0x0f, 0x2b, 0x80, 0x36, 0xf1, 0x77, 0xf6,
      0x29, 0x33, 0x13, 0xa9, 0x4a, 0x55, 0x3d, 0x6c, 0xca, 0xdb, 0x4e, 0x40, 0xc4, 0x95, 0x54,
      0xf4, 0xe2, 0x8c, 0x1b, 0xa0, 0xfe, 0x30, 0x50, 0x9d, 0x62, 0xbc, 0x5c, 0x00, 0xb4, 0xc4,
      0xb3, 0x57, 0xa5, 0x79, 0x3b, 0x85, 0xf6, 0x75, 0xdd, 0xc0, 0x00, 0x00, 0x00, 0x01, 0x4b,
      0x3b, 0x4c, 0xa8, 0x5a, 0x86, 0xc4, 0x7a, 0x09, 0x8a, 0x22, 0x3f, 0xff, 0xff, 0xff, 0xff,
      0x15, 0x02, 0x19, 0x4c, 0x48, 0x0c, 0x73, 0x70, 0x61, 0x72, 0x6b, 0x5f, 0x73, 0x63, 0x68,
      0x65, 0x6d, 0x61, 0x15, 0x06, 0x00, 0x15, 0x02, 0x25, 0x02, 0x18, 0x06, 0x64, 0x65, 0x63,
      0x37, 0x70, 0x34, 0x25, 0x0a, 0x15, 0x08, 0x15, 0x0e, 0x00, 0x15, 0x04, 0x25, 0x02, 0x18,
      0x07, 0x64, 0x65, 0x63, 0x31, 0x34, 0x70, 0x35, 0x25, 0x0a, 0x15, 0x0a, 0x15, 0x1c, 0x00,
      0x15, 0x0e, 0x15, 0x20, 0x15, 0x02, 0x18, 0x08, 0x64, 0x65, 0x63, 0x33, 0x38, 0x70, 0x31,
      0x38, 0x25, 0x0a, 0x15, 0x24, 0x15, 0x4c, 0x00, 0x16, 0x6a, 0x19, 0x1c, 0x19, 0x3c, 0x26,
      0x08, 0x1c, 0x15, 0x02, 0x19, 0x35, 0x06, 0x08, 0x00, 0x19, 0x18, 0x06, 0x64, 0x65, 0x63,
      0x37, 0x70, 0x34, 0x15, 0x02, 0x16, 0x6a, 0x16, 0xf6, 0x03, 0x16, 0xfe, 0x03, 0x26, 0x08,
      0x3c, 0x36, 0x02, 0x28, 0x04, 0x7f, 0x96, 0x98, 0x00, 0x18, 0x04, 0x81, 0x69, 0x67, 0xff,
      0x00, 0x19, 0x1c, 0x15, 0x00, 0x15, 0x00, 0x15, 0x02, 0x00, 0x00, 0x00, 0x26, 0x86, 0x04,
      0x1c, 0x15, 0x04, 0x19, 0x35, 0x06, 0x08, 0x00, 0x19, 0x18, 0x07, 0x64, 0x65, 0x63, 0x31,
      0x34, 0x70, 0x35, 0x15, 0x02, 0x16, 0x6a, 0x16, 0xa6, 0x07, 0x16, 0xb0, 0x07, 0x26, 0x86,
      0x04, 0x3c, 0x36, 0x02, 0x28, 0x08, 0xff, 0x3f, 0x7a, 0x10, 0xf3, 0x5a, 0x00, 0x00, 0x18,
      0x08, 0x01, 0xc0, 0x85, 0xef, 0x0c, 0xa5, 0xff, 0xff, 0x00, 0x19, 0x1c, 0x15, 0x00, 0x15,
      0x00, 0x15, 0x02, 0x00, 0x00, 0x00, 0x26, 0xb6, 0x0b, 0x1c, 0x15, 0x0e, 0x19, 0x35, 0x06,
      0x08, 0x00, 0x19, 0x18, 0x08, 0x64, 0x65, 0x63, 0x33, 0x38, 0x70, 0x31, 0x38, 0x15, 0x02,
      0x16, 0x6a, 0x16, 0x86, 0x0e, 0x16, 0x90, 0x0e, 0x26, 0xb6, 0x0b, 0x3c, 0x36, 0x02, 0x28,
      0x10, 0x4b, 0x3b, 0x4c, 0xa8, 0x5a, 0x86, 0xc4, 0x7a, 0x09, 0x8a, 0x22, 0x3f, 0xff, 0xff,
      0xff, 0xff, 0x18, 0x10, 0xb4, 0xc4, 0xb3, 0x57, 0xa5, 0x79, 0x3b, 0x85, 0xf6, 0x75, 0xdd,
      0xc0, 0x00, 0x00, 0x00, 0x01, 0x00, 0x19, 0x1c, 0x15, 0x00, 0x15, 0x00, 0x15, 0x02, 0x00,
      0x00, 0x00, 0x16, 0xa2, 0x19, 0x16, 0x6a, 0x00, 0x19, 0x2c, 0x18, 0x18, 0x6f, 0x72, 0x67,
      0x2e, 0x61, 0x70, 0x61, 0x63, 0x68, 0x65, 0x2e, 0x73, 0x70, 0x61, 0x72, 0x6b, 0x2e, 0x76,
      0x65, 0x72, 0x73, 0x69, 0x6f, 0x6e, 0x18, 0x05, 0x33, 0x2e, 0x30, 0x2e, 0x31, 0x00, 0x18,
      0x29, 0x6f, 0x72, 0x67, 0x2e, 0x61, 0x70, 0x61, 0x63, 0x68, 0x65, 0x2e, 0x73, 0x70, 0x61,
      0x72, 0x6b, 0x2e, 0x73, 0x71, 0x6c, 0x2e, 0x70, 0x61, 0x72, 0x71, 0x75, 0x65, 0x74, 0x2e,
      0x72, 0x6f, 0x77, 0x2e, 0x6d, 0x65, 0x74, 0x61, 0x64, 0x61, 0x74, 0x61, 0x18, 0xf4, 0x01,
      0x7b, 0x22, 0x74, 0x79, 0x70, 0x65, 0x22, 0x3a, 0x22, 0x73, 0x74, 0x72, 0x75, 0x63, 0x74,
      0x22, 0x2c, 0x22, 0x66, 0x69, 0x65, 0x6c, 0x64, 0x73, 0x22, 0x3a, 0x5b, 0x7b, 0x22, 0x6e,
      0x61, 0x6d, 0x65, 0x22, 0x3a, 0x22, 0x64, 0x65, 0x63, 0x37, 0x70, 0x34, 0x22, 0x2c, 0x22,
      0x74, 0x79, 0x70, 0x65, 0x22, 0x3a, 0x22, 0x64, 0x65, 0x63, 0x69, 0x6d, 0x61, 0x6c, 0x28,
      0x37, 0x2c, 0x34, 0x29, 0x22, 0x2c, 0x22, 0x6e, 0x75, 0x6c, 0x6c, 0x61, 0x62, 0x6c, 0x65,
      0x22, 0x3a, 0x74, 0x72, 0x75, 0x65, 0x2c, 0x22, 0x6d, 0x65, 0x74, 0x61, 0x64, 0x61, 0x74,
      0x61, 0x22, 0x3a, 0x7b, 0x7d, 0x7d, 0x2c, 0x7b, 0x22, 0x6e, 0x61, 0x6d, 0x65, 0x22, 0x3a,
      0x22, 0x64, 0x65, 0x63, 0x31, 0x34, 0x70, 0x35, 0x22, 0x2c, 0x22, 0x74, 0x79, 0x70, 0x65,
      0x22, 0x3a, 0x22, 0x64, 0x65, 0x63, 0x69, 0x6d, 0x61, 0x6c, 0x28, 0x31, 0x34, 0x2c, 0x35,
      0x29, 0x22, 0x2c, 0x22, 0x6e, 0x75, 0x6c, 0x6c, 0x61, 0x62, 0x6c, 0x65, 0x22, 0x3a, 0x74,
      0x72, 0x75, 0x65, 0x2c, 0x22, 0x6d, 0x65, 0x74, 0x61, 0x64, 0x61, 0x74, 0x61, 0x22, 0x3a,
      0x7b, 0x7d, 0x7d, 0x2c, 0x7b, 0x22, 0x6e, 0x61, 0x6d, 0x65, 0x22, 0x3a, 0x22, 0x64, 0x65,
      0x63, 0x33, 0x38, 0x70, 0x31, 0x38, 0x22, 0x2c, 0x22, 0x74, 0x79, 0x70, 0x65, 0x22, 0x3a,
      0x22, 0x64, 0x65, 0x63, 0x69, 0x6d, 0x61, 0x6c, 0x28, 0x33, 0x38, 0x2c, 0x31, 0x38, 0x29,
      0x22, 0x2c, 0x22, 0x6e, 0x75, 0x6c, 0x6c, 0x61, 0x62, 0x6c, 0x65, 0x22, 0x3a, 0x74, 0x72,
      0x75, 0x65, 0x2c, 0x22, 0x6d, 0x65, 0x74, 0x61, 0x64, 0x61, 0x74, 0x61, 0x22, 0x3a, 0x7b,
      0x7d, 0x7d, 0x5d, 0x7d, 0x00, 0x18, 0x4a, 0x70, 0x61, 0x72, 0x71, 0x75, 0x65, 0x74, 0x2d,
      0x6d, 0x72, 0x20, 0x76, 0x65, 0x72, 0x73, 0x69, 0x6f, 0x6e, 0x20, 0x31, 0x2e, 0x31, 0x30,
      0x2e, 0x31, 0x20, 0x28, 0x62, 0x75, 0x69, 0x6c, 0x64, 0x20, 0x61, 0x38, 0x39, 0x64, 0x66,
      0x38, 0x66, 0x39, 0x39, 0x33, 0x32, 0x62, 0x36, 0x65, 0x66, 0x36, 0x36, 0x33, 0x33, 0x64,
      0x30, 0x36, 0x30, 0x36, 0x39, 0x65, 0x35, 0x30, 0x63, 0x39, 0x62, 0x37, 0x39, 0x37, 0x30,
      0x62, 0x65, 0x62, 0x64, 0x31, 0x29, 0x19, 0x3c, 0x1c, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x1c,
      0x00, 0x00, 0x00, 0xd3, 0x02, 0x00, 0x00, 0x50, 0x41, 0x52, 0x31};

    cudf::io::parquet_reader_options read_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{
        reinterpret_cast<char const*>(decimals_parquet.data()), decimals_parquet_len});
    auto result = cudf::io::read_parquet(read_opts);

    auto validity =
      cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 50; });

    EXPECT_EQ(result.tbl->view().num_columns(), 3);

    std::array<int32_t, 53> col0_data{
      -2354584, -190275,  8393572,  6446515,  -5687920, -1843550, -6897687, -6780385, 3428529,
      5842056,  -4312278, -4450603, -7516141, 2974667,  -4288640, 1065090,  -9410428, 7891355,
      1076244,  -1975984, 6999466,  2666959,  9262967,  7931374,  -1370640, 451074,   8799111,
      3026424,  -6803730, 5098297,  -1414370, -9662848, 2499991,  658765,   8348874,  -6177036,
      -9694494, -5343299, 3558393,  -8789072, 2697890,  -4454707, 8299309,  -6223703, -3112513,
      7537487,  825776,   -495683,  328299,   -4529727, 0,        -9999999, 9999999};

    EXPECT_EQ(static_cast<std::size_t>(result.tbl->view().column(0).size()),
              sizeof(col0_data) / sizeof(col0_data[0]));
    cudf::test::fixed_point_column_wrapper<int32_t> col0(
      std::begin(col0_data), std::end(col0_data), validity, numeric::scale_type{-4});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(0), col0);

    std::array<int64_t, 53> col1_data{29274040266581,  -17210335917753, -58420730139037,
                                      68073792696254,  2236456014294,   13704555677045,
                                      -70797090469548, -52248605513407, -68976081919961,
                                      -34277313883112, 97774730521689,  21184241014572,
                                      -670882460254,   -40862944054399, -24079852370612,
                                      -88670167797498, -84007574359403, -71843004533519,
                                      -55538016554201, 3491435293032,   -29085437167297,
                                      36901882672273,  -98622066122568, -13974902998457,
                                      86712597643378,  -16835133643735, -94759096142232,
                                      30708340810940,  79086853262082,  78923696440892,
                                      -76316597208589, 37247268714759,  80303592631774,
                                      57790350050889,  19387319851064,  -33186875066145,
                                      69701203023404,  -7157433049060,  -7073790423437,
                                      92769171617714,  -75127120182184, -951893180618,
                                      64927618310150,  -53875897154023, -16168039035569,
                                      -24273449166429, -30359781249192, 35639397345991,
                                      45844829680593,  71401416837149,  0,
                                      -99999999999999, 99999999999999};

    EXPECT_EQ(static_cast<std::size_t>(result.tbl->view().column(1).size()), col1_data.size());
    cudf::test::fixed_point_column_wrapper<int64_t> col1(
      col1_data.begin(), col1_data.end(), validity, numeric::scale_type{-5});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(1), col1);

    cudf::io::parquet_reader_options read_strict_opts = read_opts;
    read_strict_opts.set_columns({"dec7p4", "dec14p5"});
    EXPECT_NO_THROW(cudf::io::read_parquet(read_strict_opts));
  }
  {
    // dec7p3: Decimal(precision=7, scale=3) backed by FIXED_LENGTH_BYTE_ARRAY(length = 4)
    // dec12p11: Decimal(precision=12, scale=11) backed by FIXED_LENGTH_BYTE_ARRAY(length = 6)
    // dec20p1: Decimal(precision=20, scale=1) backed by FIXED_LENGTH_BYTE_ARRAY(length = 9)
    std::array<unsigned char, 1226> const fixed_len_bytes_decimal_parquet{
      0x50, 0x41, 0x52, 0x31, 0x15, 0x00, 0x15, 0xA8, 0x01, 0x15, 0xAE, 0x01, 0x2C, 0x15, 0x28,
      0x15, 0x00, 0x15, 0x06, 0x15, 0x08, 0x1C, 0x36, 0x02, 0x28, 0x04, 0x00, 0x97, 0x45, 0x72,
      0x18, 0x04, 0x00, 0x01, 0x81, 0x3B, 0x00, 0x00, 0x00, 0x54, 0xF0, 0x53, 0x04, 0x00, 0x00,
      0x00, 0x26, 0x01, 0x03, 0x00, 0x00, 0x61, 0x10, 0xCF, 0x00, 0x0A, 0xA9, 0x08, 0x00, 0x77,
      0x58, 0x6F, 0x00, 0x6B, 0xEE, 0xA4, 0x00, 0x92, 0xF8, 0x94, 0x00, 0x2E, 0x18, 0xD4, 0x00,
      0x4F, 0x45, 0x33, 0x00, 0x97, 0x45, 0x72, 0x00, 0x0D, 0xC2, 0x75, 0x00, 0x76, 0xAA, 0xAA,
      0x00, 0x30, 0x9F, 0x86, 0x00, 0x4B, 0x9D, 0xB1, 0x00, 0x4E, 0x4B, 0x3B, 0x00, 0x01, 0x81,
      0x3B, 0x00, 0x22, 0xD4, 0x53, 0x00, 0x72, 0xC4, 0xAF, 0x00, 0x43, 0x9B, 0x72, 0x00, 0x1D,
      0x91, 0xC3, 0x00, 0x45, 0x27, 0x48, 0x15, 0x00, 0x15, 0xF4, 0x01, 0x15, 0xFA, 0x01, 0x2C,
      0x15, 0x28, 0x15, 0x00, 0x15, 0x06, 0x15, 0x08, 0x1C, 0x36, 0x02, 0x28, 0x06, 0x00, 0xD5,
      0xD7, 0x31, 0x99, 0xA6, 0x18, 0x06, 0xFF, 0x17, 0x2B, 0x5A, 0xF0, 0x01, 0x00, 0x00, 0x00,
      0x7A, 0xF0, 0x79, 0x04, 0x00, 0x00, 0x00, 0x24, 0x01, 0x03, 0x02, 0x00, 0x54, 0x23, 0xCF,
      0x13, 0x0A, 0x00, 0x07, 0x22, 0xB1, 0x21, 0x7E, 0x00, 0x64, 0x19, 0xD6, 0xD2, 0xA5, 0x00,
      0x61, 0x7F, 0xF6, 0xB9, 0xB0, 0x00, 0xD0, 0x7F, 0x9C, 0xA9, 0xE9, 0x00, 0x65, 0x58, 0xF0,
      0xAD, 0xFB, 0x00, 0xBC, 0x61, 0xE2, 0x03, 0xDA, 0xFF, 0x17, 0x2B, 0x5A, 0xF0, 0x01, 0x00,
      0x63, 0x4B, 0x4C, 0xFE, 0x45, 0x00, 0x7A, 0xA0, 0xD8, 0xD1, 0xC0, 0x00, 0xC0, 0x63, 0xF7,
      0x9D, 0x0A, 0x00, 0x88, 0x22, 0x0F, 0x1B, 0x25, 0x00, 0x1A, 0x80, 0x56, 0x34, 0xC7, 0x00,
      0x5F, 0x48, 0x61, 0x09, 0x7C, 0x00, 0x61, 0xEF, 0x92, 0x42, 0x2F, 0x00, 0xD5, 0xD7, 0x31,
      0x99, 0xA6, 0xFF, 0x17, 0x2B, 0x5A, 0xF0, 0x01, 0x00, 0x71, 0xDD, 0xE2, 0x22, 0x7B, 0x00,
      0x54, 0xBF, 0xAE, 0xE9, 0x3C, 0x15, 0x00, 0x15, 0xD4, 0x02, 0x15, 0xDC, 0x02, 0x2C, 0x15,
      0x28, 0x15, 0x00, 0x15, 0x06, 0x15, 0x08, 0x1C, 0x36, 0x04, 0x28, 0x09, 0x00, 0x7D, 0xFE,
      0x02, 0xDA, 0xB2, 0x62, 0xA3, 0xFB, 0x18, 0x09, 0x00, 0x03, 0x9C, 0xCD, 0x5A, 0xAC, 0xBB,
      0xF1, 0xE3, 0x00, 0x00, 0x00, 0xAA, 0x01, 0xF0, 0xA9, 0x04, 0x00, 0x00, 0x00, 0x07, 0xBF,
      0xBF, 0x0F, 0x00, 0x7D, 0xFE, 0x02, 0xDA, 0xB2, 0x62, 0xA3, 0xFB, 0x00, 0x7D, 0x9A, 0xCB,
      0xDA, 0x4B, 0x10, 0x8B, 0xAC, 0x00, 0x20, 0xBA, 0x97, 0x87, 0x2E, 0x3B, 0x4E, 0x04, 0x00,
      0x15, 0xBB, 0xC2, 0xDF, 0x2D, 0x25, 0x08, 0xB6, 0x00, 0x5C, 0x67, 0x0E, 0x36, 0x30, 0xF1,
      0xAC, 0xA4, 0x00, 0x44, 0xF1, 0x8E, 0xFB, 0x17, 0x5E, 0xE1, 0x96, 0x00, 0x64, 0x69, 0xF9,
      0x66, 0x3F, 0x11, 0xED, 0xB9, 0x00, 0x45, 0xB5, 0xDA, 0x14, 0x9C, 0xA3, 0xFA, 0x64, 0x00,
      0x26, 0x5F, 0xDE, 0xD7, 0x67, 0x95, 0xEF, 0xB1, 0x00, 0x35, 0xDB, 0x9B, 0x88, 0x46, 0xD0,
      0xA1, 0x0E, 0x00, 0x45, 0xA9, 0x92, 0x8E, 0x89, 0xD1, 0xAC, 0x4C, 0x00, 0x4C, 0xF1, 0xCB,
      0x27, 0x82, 0x3A, 0x7D, 0xB7, 0x00, 0x64, 0xD3, 0xD2, 0x2F, 0x9C, 0x83, 0x16, 0x75, 0x00,
      0x15, 0xDF, 0xC2, 0xA9, 0x63, 0xB8, 0x33, 0x65, 0x00, 0x27, 0x40, 0x28, 0x97, 0x05, 0x8E,
      0xE3, 0x46, 0x00, 0x03, 0x9C, 0xCD, 0x5A, 0xAC, 0xBB, 0xF1, 0xE3, 0x00, 0x22, 0x23, 0xF5,
      0xE8, 0x9D, 0x55, 0xD4, 0x9C, 0x00, 0x25, 0xB9, 0xD8, 0x87, 0x2D, 0xF1, 0xF2, 0x17, 0x15,
      0x02, 0x19, 0x4C, 0x48, 0x0C, 0x73, 0x70, 0x61, 0x72, 0x6B, 0x5F, 0x73, 0x63, 0x68, 0x65,
      0x6D, 0x61, 0x15, 0x06, 0x00, 0x15, 0x0E, 0x15, 0x08, 0x15, 0x02, 0x18, 0x06, 0x64, 0x65,
      0x63, 0x37, 0x70, 0x33, 0x25, 0x0A, 0x15, 0x06, 0x15, 0x0E, 0x00, 0x15, 0x0E, 0x15, 0x0C,
      0x15, 0x02, 0x18, 0x08, 0x64, 0x65, 0x63, 0x31, 0x32, 0x70, 0x31, 0x31, 0x25, 0x0A, 0x15,
      0x16, 0x15, 0x18, 0x00, 0x15, 0x0E, 0x15, 0x12, 0x15, 0x02, 0x18, 0x07, 0x64, 0x65, 0x63,
      0x32, 0x30, 0x70, 0x31, 0x25, 0x0A, 0x15, 0x02, 0x15, 0x28, 0x00, 0x16, 0x28, 0x19, 0x1C,
      0x19, 0x3C, 0x26, 0x08, 0x1C, 0x15, 0x0E, 0x19, 0x35, 0x06, 0x08, 0x00, 0x19, 0x18, 0x06,
      0x64, 0x65, 0x63, 0x37, 0x70, 0x33, 0x15, 0x02, 0x16, 0x28, 0x16, 0xEE, 0x01, 0x16, 0xF4,
      0x01, 0x26, 0x08, 0x3C, 0x36, 0x02, 0x28, 0x04, 0x00, 0x97, 0x45, 0x72, 0x18, 0x04, 0x00,
      0x01, 0x81, 0x3B, 0x00, 0x19, 0x1C, 0x15, 0x00, 0x15, 0x00, 0x15, 0x02, 0x00, 0x00, 0x00,
      0x26, 0xFC, 0x01, 0x1C, 0x15, 0x0E, 0x19, 0x35, 0x06, 0x08, 0x00, 0x19, 0x18, 0x08, 0x64,
      0x65, 0x63, 0x31, 0x32, 0x70, 0x31, 0x31, 0x15, 0x02, 0x16, 0x28, 0x16, 0xC2, 0x02, 0x16,
      0xC8, 0x02, 0x26, 0xFC, 0x01, 0x3C, 0x36, 0x02, 0x28, 0x06, 0x00, 0xD5, 0xD7, 0x31, 0x99,
      0xA6, 0x18, 0x06, 0xFF, 0x17, 0x2B, 0x5A, 0xF0, 0x01, 0x00, 0x19, 0x1C, 0x15, 0x00, 0x15,
      0x00, 0x15, 0x02, 0x00, 0x00, 0x00, 0x26, 0xC4, 0x04, 0x1C, 0x15, 0x0E, 0x19, 0x35, 0x06,
      0x08, 0x00, 0x19, 0x18, 0x07, 0x64, 0x65, 0x63, 0x32, 0x30, 0x70, 0x31, 0x15, 0x02, 0x16,
      0x28, 0x16, 0xAE, 0x03, 0x16, 0xB6, 0x03, 0x26, 0xC4, 0x04, 0x3C, 0x36, 0x04, 0x28, 0x09,
      0x00, 0x7D, 0xFE, 0x02, 0xDA, 0xB2, 0x62, 0xA3, 0xFB, 0x18, 0x09, 0x00, 0x03, 0x9C, 0xCD,
      0x5A, 0xAC, 0xBB, 0xF1, 0xE3, 0x00, 0x19, 0x1C, 0x15, 0x00, 0x15, 0x00, 0x15, 0x02, 0x00,
      0x00, 0x00, 0x16, 0xDE, 0x07, 0x16, 0x28, 0x00, 0x19, 0x2C, 0x18, 0x18, 0x6F, 0x72, 0x67,
      0x2E, 0x61, 0x70, 0x61, 0x63, 0x68, 0x65, 0x2E, 0x73, 0x70, 0x61, 0x72, 0x6B, 0x2E, 0x76,
      0x65, 0x72, 0x73, 0x69, 0x6F, 0x6E, 0x18, 0x05, 0x33, 0x2E, 0x30, 0x2E, 0x31, 0x00, 0x18,
      0x29, 0x6F, 0x72, 0x67, 0x2E, 0x61, 0x70, 0x61, 0x63, 0x68, 0x65, 0x2E, 0x73, 0x70, 0x61,
      0x72, 0x6B, 0x2E, 0x73, 0x71, 0x6C, 0x2E, 0x70, 0x61, 0x72, 0x71, 0x75, 0x65, 0x74, 0x2E,
      0x72, 0x6F, 0x77, 0x2E, 0x6D, 0x65, 0x74, 0x61, 0x64, 0x61, 0x74, 0x61, 0x18, 0xF4, 0x01,
      0x7B, 0x22, 0x74, 0x79, 0x70, 0x65, 0x22, 0x3A, 0x22, 0x73, 0x74, 0x72, 0x75, 0x63, 0x74,
      0x22, 0x2C, 0x22, 0x66, 0x69, 0x65, 0x6C, 0x64, 0x73, 0x22, 0x3A, 0x5B, 0x7B, 0x22, 0x6E,
      0x61, 0x6D, 0x65, 0x22, 0x3A, 0x22, 0x64, 0x65, 0x63, 0x37, 0x70, 0x33, 0x22, 0x2C, 0x22,
      0x74, 0x79, 0x70, 0x65, 0x22, 0x3A, 0x22, 0x64, 0x65, 0x63, 0x69, 0x6D, 0x61, 0x6C, 0x28,
      0x37, 0x2C, 0x33, 0x29, 0x22, 0x2C, 0x22, 0x6E, 0x75, 0x6C, 0x6C, 0x61, 0x62, 0x6C, 0x65,
      0x22, 0x3A, 0x74, 0x72, 0x75, 0x65, 0x2C, 0x22, 0x6D, 0x65, 0x74, 0x61, 0x64, 0x61, 0x74,
      0x61, 0x22, 0x3A, 0x7B, 0x7D, 0x7D, 0x2C, 0x7B, 0x22, 0x6E, 0x61, 0x6D, 0x65, 0x22, 0x3A,
      0x22, 0x64, 0x65, 0x63, 0x31, 0x32, 0x70, 0x31, 0x31, 0x22, 0x2C, 0x22, 0x74, 0x79, 0x70,
      0x65, 0x22, 0x3A, 0x22, 0x64, 0x65, 0x63, 0x69, 0x6D, 0x61, 0x6C, 0x28, 0x31, 0x32, 0x2C,
      0x31, 0x31, 0x29, 0x22, 0x2C, 0x22, 0x6E, 0x75, 0x6C, 0x6C, 0x61, 0x62, 0x6C, 0x65, 0x22,
      0x3A, 0x74, 0x72, 0x75, 0x65, 0x2C, 0x22, 0x6D, 0x65, 0x74, 0x61, 0x64, 0x61, 0x74, 0x61,
      0x22, 0x3A, 0x7B, 0x7D, 0x7D, 0x2C, 0x7B, 0x22, 0x6E, 0x61, 0x6D, 0x65, 0x22, 0x3A, 0x22,
      0x64, 0x65, 0x63, 0x32, 0x30, 0x70, 0x31, 0x22, 0x2C, 0x22, 0x74, 0x79, 0x70, 0x65, 0x22,
      0x3A, 0x22, 0x64, 0x65, 0x63, 0x69, 0x6D, 0x61, 0x6C, 0x28, 0x32, 0x30, 0x2C, 0x31, 0x29,
      0x22, 0x2C, 0x22, 0x6E, 0x75, 0x6C, 0x6C, 0x61, 0x62, 0x6C, 0x65, 0x22, 0x3A, 0x74, 0x72,
      0x75, 0x65, 0x2C, 0x22, 0x6D, 0x65, 0x74, 0x61, 0x64, 0x61, 0x74, 0x61, 0x22, 0x3A, 0x7B,
      0x7D, 0x7D, 0x5D, 0x7D, 0x00, 0x18, 0x4A, 0x70, 0x61, 0x72, 0x71, 0x75, 0x65, 0x74, 0x2D,
      0x6D, 0x72, 0x20, 0x76, 0x65, 0x72, 0x73, 0x69, 0x6F, 0x6E, 0x20, 0x31, 0x2E, 0x31, 0x30,
      0x2E, 0x31, 0x20, 0x28, 0x62, 0x75, 0x69, 0x6C, 0x64, 0x20, 0x61, 0x38, 0x39, 0x64, 0x66,
      0x38, 0x66, 0x39, 0x39, 0x33, 0x32, 0x62, 0x36, 0x65, 0x66, 0x36, 0x36, 0x33, 0x33, 0x64,
      0x30, 0x36, 0x30, 0x36, 0x39, 0x65, 0x35, 0x30, 0x63, 0x39, 0x62, 0x37, 0x39, 0x37, 0x30,
      0x62, 0x65, 0x62, 0x64, 0x31, 0x29, 0x19, 0x3C, 0x1C, 0x00, 0x00, 0x1C, 0x00, 0x00, 0x1C,
      0x00, 0x00, 0x00, 0xC5, 0x02, 0x00, 0x00, 0x50, 0x41, 0x52, 0x31,
    };

    unsigned int parquet_len = 1226;

    cudf::io::parquet_reader_options read_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{
        reinterpret_cast<char const*>(fixed_len_bytes_decimal_parquet.data()), parquet_len});
    auto result = cudf::io::read_parquet(read_opts);
    EXPECT_EQ(result.tbl->view().num_columns(), 3);

    auto validity_c0 = cudf::test::iterators::nulls_at({19});
    std::array col0_data{6361295, 698632,  7821423, 7073444, 9631892, 3021012, 5195059,
                         9913714, 901749,  7776938, 3186566, 4955569, 5131067, 98619,
                         2282579, 7521455, 4430706, 1937859, 4532040, 0};

    EXPECT_EQ(static_cast<std::size_t>(result.tbl->view().column(0).size()), col0_data.size());
    cudf::test::fixed_point_column_wrapper<int32_t> col0(
      col0_data.begin(), col0_data.end(), validity_c0, numeric::scale_type{-3});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(0), col0);

    auto validity_c1 = cudf::test::iterators::nulls_at({18});
    std::array<int64_t, 20> col1_data{361378026250,
                                      30646804862,
                                      429930238629,
                                      418758703536,
                                      895494171113,
                                      435283865083,
                                      809096053722,
                                      -999999999999,
                                      426465099333,
                                      526684574144,
                                      826310892810,
                                      584686967589,
                                      113822282951,
                                      409236212092,
                                      420631167535,
                                      918438386086,
                                      -999999999999,
                                      489053889147,
                                      0,
                                      363993164092};

    EXPECT_EQ(static_cast<std::size_t>(result.tbl->view().column(1).size()), col1_data.size());
    cudf::test::fixed_point_column_wrapper<int64_t> col1(
      col1_data.begin(), col1_data.end(), validity_c1, numeric::scale_type{-11});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(1), col1);

    auto validity_c2 = cudf::test::iterators::nulls_at({6, 14});
    std::array<__int128_t, 20> col2_data{9078697037144433659,
                                         9050770539577117612,
                                         2358363961733893636,
                                         1566059559232276662,
                                         6658306200002735268,
                                         4967909073046397334,
                                         0,
                                         7235588493887532473,
                                         5023160741463849572,
                                         2765173712965988273,
                                         3880866513515749646,
                                         5019704400576359500,
                                         5544435986818825655,
                                         7265381725809874549,
                                         0,
                                         1576192427381240677,
                                         2828305195087094598,
                                         260308667809395171,
                                         2460080200895288476,
                                         2718441925197820439};

    EXPECT_EQ(static_cast<std::size_t>(result.tbl->view().column(2).size()), col2_data.size());
    cudf::test::fixed_point_column_wrapper<__int128_t> col2(
      col2_data.begin(), col2_data.end(), validity_c2, numeric::scale_type{-1});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(2), col2);
  }
}

TEST_F(ParquetReaderTest, EmptyOutput)
{
  cudf::test::fixed_width_column_wrapper<int> c0;
  cudf::test::strings_column_wrapper c1;
  cudf::test::fixed_point_column_wrapper<int> c2({}, numeric::scale_type{2});
  cudf::test::lists_column_wrapper<float> _c3{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}};
  auto c3 = cudf::empty_like(_c3);

  cudf::test::fixed_width_column_wrapper<int> sc0;
  cudf::test::strings_column_wrapper sc1;
  cudf::test::lists_column_wrapper<int> _sc2{{1, 2}};
  std::vector<std::unique_ptr<cudf::column>> struct_children;
  struct_children.push_back(sc0.release());
  struct_children.push_back(sc1.release());
  struct_children.push_back(cudf::empty_like(_sc2));
  cudf::test::structs_column_wrapper c4(std::move(struct_children));

  table_view expected({c0, c1, c2, *c3, c4});

  // set precision on the decimal column
  cudf::io::table_input_metadata expected_metadata(expected);
  expected_metadata.column_metadata[2].set_decimal_precision(1);

  auto filepath = temp_env->get_temp_filepath("EmptyOutput.parquet");
  cudf::io::parquet_writer_options out_args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected);
  out_args.set_metadata(std::move(expected_metadata));
  cudf::io::write_parquet(out_args);

  cudf::io::parquet_reader_options read_args =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(read_args);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
}

TEST_F(ParquetReaderTest, EmptyColumnsParam)
{
  srand(31337);
  auto const expected = create_random_fixed_table<int>(2, 4, false);

  std::vector<char> out_buffer;
  cudf::io::parquet_writer_options args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{&out_buffer}, *expected);
  cudf::io::write_parquet(args);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(
      cudf::io::source_info{out_buffer.data(), out_buffer.size()})
      .columns({});
  auto const result = cudf::io::read_parquet(read_opts);

  EXPECT_EQ(result.tbl->num_columns(), 0);
  EXPECT_EQ(result.tbl->num_rows(), 0);
}

TEST_F(ParquetReaderTest, BinaryAsStrings)
{
  std::vector<char const*> strings{
    "Monday", "Wednesday", "Friday", "Monday", "Friday", "Friday", "Friday", "Funday"};
  auto const num_rows = strings.size();

  auto seq_col0 = random_values<int>(num_rows);
  auto seq_col2 = random_values<float>(num_rows);
  auto seq_col3 = random_values<uint8_t>(num_rows);
  auto validity = cudf::test::iterators::no_nulls();

  column_wrapper<int> int_col{seq_col0.begin(), seq_col0.end(), validity};
  column_wrapper<cudf::string_view> string_col{strings.begin(), strings.end()};
  column_wrapper<float> float_col{seq_col2.begin(), seq_col2.end(), validity};
  cudf::test::lists_column_wrapper<uint8_t> list_int_col{
    {'M', 'o', 'n', 'd', 'a', 'y'},
    {'W', 'e', 'd', 'n', 'e', 's', 'd', 'a', 'y'},
    {'F', 'r', 'i', 'd', 'a', 'y'},
    {'M', 'o', 'n', 'd', 'a', 'y'},
    {'F', 'r', 'i', 'd', 'a', 'y'},
    {'F', 'r', 'i', 'd', 'a', 'y'},
    {'F', 'r', 'i', 'd', 'a', 'y'},
    {'F', 'u', 'n', 'd', 'a', 'y'}};

  auto output = table_view{{int_col, string_col, float_col, string_col, list_int_col}};
  cudf::io::table_input_metadata output_metadata(output);
  output_metadata.column_metadata[0].set_name("col_other");
  output_metadata.column_metadata[1].set_name("col_string");
  output_metadata.column_metadata[2].set_name("col_float");
  output_metadata.column_metadata[3].set_name("col_string2").set_output_as_binary(true);
  output_metadata.column_metadata[4].set_name("col_binary").set_output_as_binary(true);

  auto filepath = temp_env->get_temp_filepath("BinaryReadStrings.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, output)
      .metadata(std::move(output_metadata));
  cudf::io::write_parquet(out_opts);

  auto expected_string = table_view{{int_col, string_col, float_col, string_col, string_col}};
  auto expected_mixed  = table_view{{int_col, string_col, float_col, list_int_col, list_int_col}};

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
      .set_column_schema({{}, {}, {}, {}, {}});
  auto result = cudf::io::read_parquet(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_string, result.tbl->view());

  cudf::io::parquet_reader_options default_in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  result = cudf::io::read_parquet(default_in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_string, result.tbl->view());

  std::vector<cudf::io::reader_column_schema> md{
    {},
    {},
    {},
    cudf::io::reader_column_schema().set_convert_binary_to_strings(false),
    cudf::io::reader_column_schema().set_convert_binary_to_strings(false)};

  cudf::io::parquet_reader_options mixed_in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
      .set_column_schema(md);
  result = cudf::io::read_parquet(mixed_in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_mixed, result.tbl->view());
}

TEST_F(ParquetReaderTest, NestedByteArray)
{
  constexpr auto num_rows = 8;

  auto seq_col0       = random_values<int>(num_rows);
  auto seq_col2       = random_values<float>(num_rows);
  auto seq_col3       = random_values<uint8_t>(num_rows);
  auto const validity = cudf::test::iterators::no_nulls();

  column_wrapper<int> int_col{seq_col0.begin(), seq_col0.end(), validity};
  column_wrapper<float> float_col{seq_col2.begin(), seq_col2.end(), validity};
  cudf::test::lists_column_wrapper<uint8_t> list_list_int_col{
    {{'M', 'o', 'n', 'd', 'a', 'y'},
     {'W', 'e', 'd', 'n', 'e', 's', 'd', 'a', 'y'},
     {'F', 'r', 'i', 'd', 'a', 'y'}},
    {{'M', 'o', 'n', 'd', 'a', 'y'}, {'F', 'r', 'i', 'd', 'a', 'y'}},
    {{'M', 'o', 'n', 'd', 'a', 'y'},
     {'W', 'e', 'd', 'n', 'e', 's', 'd', 'a', 'y'},
     {'F', 'r', 'i', 'd', 'a', 'y'}},
    {{'F', 'r', 'i', 'd', 'a', 'y'},
     {'F', 'r', 'i', 'd', 'a', 'y'},
     {'F', 'u', 'n', 'd', 'a', 'y'}},
    {{'M', 'o', 'n', 'd', 'a', 'y'},
     {'W', 'e', 'd', 'n', 'e', 's', 'd', 'a', 'y'},
     {'F', 'r', 'i', 'd', 'a', 'y'}},
    {{'F', 'r', 'i', 'd', 'a', 'y'},
     {'F', 'r', 'i', 'd', 'a', 'y'},
     {'F', 'u', 'n', 'd', 'a', 'y'}},
    {{'M', 'o', 'n', 'd', 'a', 'y'},
     {'W', 'e', 'd', 'n', 'e', 's', 'd', 'a', 'y'},
     {'F', 'r', 'i', 'd', 'a', 'y'}},
    {{'M', 'o', 'n', 'd', 'a', 'y'}, {'F', 'r', 'i', 'd', 'a', 'y'}}};

  auto const expected = table_view{{int_col, float_col, list_list_int_col}};
  cudf::io::table_input_metadata output_metadata(expected);
  output_metadata.column_metadata[0].set_name("col_other");
  output_metadata.column_metadata[1].set_name("col_float");
  output_metadata.column_metadata[2].set_name("col_binary").child(1).set_output_as_binary(true);

  auto filepath = temp_env->get_temp_filepath("NestedByteArray.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .metadata(std::move(output_metadata));
  cudf::io::write_parquet(out_opts);

  auto source = cudf::io::datasource::create(filepath);
  cudf::io::parquet::detail::FileMetaData fmd;

  read_footer(source, &fmd);
  EXPECT_EQ(fmd.schema[5].type, cudf::io::parquet::detail::Type::BYTE_ARRAY);

  std::vector<cudf::io::reader_column_schema> md{
    {},
    {},
    cudf::io::reader_column_schema().add_child(
      cudf::io::reader_column_schema().set_convert_binary_to_strings(false))};

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
      .set_column_schema(md);
  auto result = cudf::io::read_parquet(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
}

TEST_F(ParquetReaderTest, StructByteArray)
{
  constexpr auto num_rows = 100;

  auto seq_col0       = random_values<uint8_t>(num_rows);
  auto const validity = cudf::test::iterators::no_nulls();

  column_wrapper<uint8_t> int_col{seq_col0.begin(), seq_col0.end(), validity};
  cudf::test::lists_column_wrapper<uint8_t> list_of_int{{seq_col0.begin(), seq_col0.begin() + 50},
                                                        {seq_col0.begin() + 50, seq_col0.end()}};
  auto struct_col = cudf::test::structs_column_wrapper{{list_of_int}, validity};

  auto const expected = table_view{{struct_col}};
  EXPECT_EQ(1, expected.num_columns());
  cudf::io::table_input_metadata output_metadata(expected);
  output_metadata.column_metadata[0]
    .set_name("struct_binary")
    .child(0)
    .set_name("a")
    .set_output_as_binary(true);

  auto filepath = temp_env->get_temp_filepath("StructByteArray.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .metadata(std::move(output_metadata));
  cudf::io::write_parquet(out_opts);

  std::vector<cudf::io::reader_column_schema> md{cudf::io::reader_column_schema().add_child(
    cudf::io::reader_column_schema().set_convert_binary_to_strings(false))};

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
      .set_column_schema(md);
  auto result = cudf::io::read_parquet(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
}

TEST_F(ParquetReaderTest, NestingOptimizationTest)
{
  // test nesting levels > cudf::io::parquet::detail::max_cacheable_nesting_decode_info deep.
  constexpr cudf::size_type num_nesting_levels = 16;
  static_assert(num_nesting_levels > cudf::io::parquet::detail::max_cacheable_nesting_decode_info);
  constexpr cudf::size_type rows_per_level = 2;

  constexpr cudf::size_type num_values = (1 << num_nesting_levels) * rows_per_level;
  auto value_iter                      = thrust::make_counting_iterator(0);
  auto validity =
    cudf::detail::make_counting_transform_iterator(0, [](cudf::size_type i) { return i % 2; });
  cudf::test::fixed_width_column_wrapper<int> values(value_iter, value_iter + num_values, validity);

  // ~256k values with num_nesting_levels = 16
  auto prev_col = values.release();
  for (int idx = 0; idx < num_nesting_levels; idx++) {
    auto const num_rows = (1 << (num_nesting_levels - idx));

    auto offsets_iter = cudf::detail::make_counting_transform_iterator(
      0, [](cudf::size_type i) { return i * rows_per_level; });

    cudf::test::fixed_width_column_wrapper<cudf::size_type> offsets(offsets_iter,
                                                                    offsets_iter + num_rows + 1);
    auto c   = cudf::make_lists_column(num_rows, offsets.release(), std::move(prev_col), 0, {});
    prev_col = std::move(c);
  }
  auto const& expect = prev_col;

  auto filepath = temp_env->get_temp_filepath("NestingDecodeCache.parquet");
  cudf::io::parquet_writer_options opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, table_view{{*expect}});
  cudf::io::write_parquet(opts);

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(in_opts);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expect, result.tbl->get_column(0));
}

TEST_F(ParquetReaderTest, SingleLevelLists)
{
  std::array<unsigned char, 214> list_bytes{
    0x50, 0x41, 0x52, 0x31, 0x15, 0x00, 0x15, 0x28, 0x15, 0x28, 0x15, 0xa7, 0xce, 0x91, 0x8c, 0x06,
    0x1c, 0x15, 0x04, 0x15, 0x00, 0x15, 0x06, 0x15, 0x06, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03,
    0x02, 0x02, 0x00, 0x00, 0x00, 0x03, 0x03, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x15,
    0x02, 0x19, 0x3c, 0x48, 0x0c, 0x73, 0x70, 0x61, 0x72, 0x6b, 0x5f, 0x73, 0x63, 0x68, 0x65, 0x6d,
    0x61, 0x15, 0x02, 0x00, 0x35, 0x00, 0x18, 0x01, 0x66, 0x15, 0x02, 0x15, 0x06, 0x4c, 0x3c, 0x00,
    0x00, 0x00, 0x15, 0x02, 0x25, 0x04, 0x18, 0x05, 0x61, 0x72, 0x72, 0x61, 0x79, 0x00, 0x16, 0x02,
    0x19, 0x1c, 0x19, 0x1c, 0x26, 0x08, 0x1c, 0x15, 0x02, 0x19, 0x25, 0x00, 0x06, 0x19, 0x28, 0x01,
    0x66, 0x05, 0x61, 0x72, 0x72, 0x61, 0x79, 0x15, 0x00, 0x16, 0x04, 0x16, 0x56, 0x16, 0x56, 0x26,
    0x08, 0x3c, 0x18, 0x04, 0x01, 0x00, 0x00, 0x00, 0x18, 0x04, 0x00, 0x00, 0x00, 0x00, 0x16, 0x00,
    0x28, 0x04, 0x01, 0x00, 0x00, 0x00, 0x18, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x19, 0x1c, 0x15,
    0x00, 0x15, 0x00, 0x15, 0x02, 0x00, 0x00, 0x00, 0x16, 0x56, 0x16, 0x02, 0x26, 0x08, 0x16, 0x56,
    0x14, 0x00, 0x00, 0x28, 0x13, 0x52, 0x41, 0x50, 0x49, 0x44, 0x53, 0x20, 0x53, 0x70, 0x61, 0x72,
    0x6b, 0x20, 0x50, 0x6c, 0x75, 0x67, 0x69, 0x6e, 0x19, 0x1c, 0x1c, 0x00, 0x00, 0x00, 0x9f, 0x00,
    0x00, 0x00, 0x50, 0x41, 0x52, 0x31};

  // read single level list reproducing parquet file
  cudf::io::parquet_reader_options read_opts = cudf::io::parquet_reader_options::builder(
    cudf::io::source_info{reinterpret_cast<char const*>(list_bytes.data()), list_bytes.size()});
  auto table = cudf::io::read_parquet(read_opts);

  auto const c0 = table.tbl->get_column(0);
  EXPECT_TRUE(c0.type().id() == cudf::type_id::LIST);

  auto const lc    = cudf::lists_column_view(c0);
  auto const child = lc.child();
  EXPECT_TRUE(child.type().id() == cudf::type_id::INT32);
}

TEST_F(ParquetReaderTest, ChunkedSingleLevelLists)
{
  std::array<unsigned char, 214> list_bytes{
    0x50, 0x41, 0x52, 0x31, 0x15, 0x00, 0x15, 0x28, 0x15, 0x28, 0x15, 0xa7, 0xce, 0x91, 0x8c, 0x06,
    0x1c, 0x15, 0x04, 0x15, 0x00, 0x15, 0x06, 0x15, 0x06, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03,
    0x02, 0x02, 0x00, 0x00, 0x00, 0x03, 0x03, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x15,
    0x02, 0x19, 0x3c, 0x48, 0x0c, 0x73, 0x70, 0x61, 0x72, 0x6b, 0x5f, 0x73, 0x63, 0x68, 0x65, 0x6d,
    0x61, 0x15, 0x02, 0x00, 0x35, 0x00, 0x18, 0x01, 0x66, 0x15, 0x02, 0x15, 0x06, 0x4c, 0x3c, 0x00,
    0x00, 0x00, 0x15, 0x02, 0x25, 0x04, 0x18, 0x05, 0x61, 0x72, 0x72, 0x61, 0x79, 0x00, 0x16, 0x02,
    0x19, 0x1c, 0x19, 0x1c, 0x26, 0x08, 0x1c, 0x15, 0x02, 0x19, 0x25, 0x00, 0x06, 0x19, 0x28, 0x01,
    0x66, 0x05, 0x61, 0x72, 0x72, 0x61, 0x79, 0x15, 0x00, 0x16, 0x04, 0x16, 0x56, 0x16, 0x56, 0x26,
    0x08, 0x3c, 0x18, 0x04, 0x01, 0x00, 0x00, 0x00, 0x18, 0x04, 0x00, 0x00, 0x00, 0x00, 0x16, 0x00,
    0x28, 0x04, 0x01, 0x00, 0x00, 0x00, 0x18, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x19, 0x1c, 0x15,
    0x00, 0x15, 0x00, 0x15, 0x02, 0x00, 0x00, 0x00, 0x16, 0x56, 0x16, 0x02, 0x26, 0x08, 0x16, 0x56,
    0x14, 0x00, 0x00, 0x28, 0x13, 0x52, 0x41, 0x50, 0x49, 0x44, 0x53, 0x20, 0x53, 0x70, 0x61, 0x72,
    0x6b, 0x20, 0x50, 0x6c, 0x75, 0x67, 0x69, 0x6e, 0x19, 0x1c, 0x1c, 0x00, 0x00, 0x00, 0x9f, 0x00,
    0x00, 0x00, 0x50, 0x41, 0x52, 0x31};

  auto reader = cudf::io::chunked_parquet_reader(
    1L << 31,
    cudf::io::parquet_reader_options::builder(
      cudf::io::source_info{reinterpret_cast<char const*>(list_bytes.data()), list_bytes.size()}));
  int iterations = 0;
  while (reader.has_next() && iterations < 10) {
    auto chunk = reader.read_chunk();
  }
  EXPECT_TRUE(iterations < 10);
}

TEST_F(ParquetReaderTest, ReorderedReadMultipleFiles)
{
  constexpr auto num_rows    = 50'000;
  constexpr auto cardinality = 20'000;

  // table 1
  auto str1 = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return "cat " + std::to_string(i % cardinality); });
  auto cols1 = cudf::test::strings_column_wrapper(str1, str1 + num_rows);

  auto int1 =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % cardinality; });
  auto coli1 = cudf::test::fixed_width_column_wrapper<int>(int1, int1 + num_rows);

  auto const expected1 = table_view{{cols1, coli1}};
  auto const swapped1  = table_view{{coli1, cols1}};

  auto const filepath1 = temp_env->get_temp_filepath("LargeReorderedRead1.parquet");
  auto out_opts1 =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath1}, expected1)
      .compression(cudf::io::compression_type::NONE);
  cudf::io::write_parquet(out_opts1);

  // table 2
  auto str2 = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return "dog " + std::to_string(i % cardinality); });
  auto cols2 = cudf::test::strings_column_wrapper(str2, str2 + num_rows);

  auto int2 = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return (i % cardinality) + cardinality; });
  auto coli2 = cudf::test::fixed_width_column_wrapper<int>(int2, int2 + num_rows);

  auto const expected2 = table_view{{cols2, coli2}};
  auto const swapped2  = table_view{{coli2, cols2}};

  auto const filepath2 = temp_env->get_temp_filepath("LargeReorderedRead2.parquet");
  auto out_opts2 =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath2}, expected2)
      .compression(cudf::io::compression_type::NONE);
  cudf::io::write_parquet(out_opts2);

  // read in both files swapping the columns
  auto read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{{filepath1, filepath2}})
      .columns({"_col1", "_col0"});
  auto result = cudf::io::read_parquet(read_opts);
  auto sliced = cudf::slice(result.tbl->view(), {0, num_rows, num_rows, 2 * num_rows});
  CUDF_TEST_EXPECT_TABLES_EQUAL(sliced[0], swapped1);
  CUDF_TEST_EXPECT_TABLES_EQUAL(sliced[1], swapped2);
}

TEST_F(ParquetReaderTest, FilterSimple)
{
  srand(31337);
  auto written_table = create_random_fixed_table<int>(9, 9, false);

  auto filepath = temp_env->get_temp_filepath("FilterSimple.parquet");
  cudf::io::parquet_writer_options args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, *written_table);
  cudf::io::write_parquet(args);

  // Filtering AST - table[0] < RAND_MAX/2
  auto literal_value     = cudf::numeric_scalar<decltype(RAND_MAX)>(RAND_MAX / 2);
  auto literal           = cudf::ast::literal(literal_value);
  auto col_ref_0         = cudf::ast::column_reference(0);
  auto filter_expression = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, literal);

  auto predicate = cudf::compute_column(*written_table, filter_expression);
  EXPECT_EQ(predicate->view().type().id(), cudf::type_id::BOOL8)
    << "Predicate filter should return a boolean";
  auto expected = cudf::apply_boolean_mask(*written_table, *predicate);
  // To make sure AST filters out some elements
  EXPECT_LT(expected->num_rows(), written_table->num_rows());

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
      .filter(filter_expression);
  auto result = cudf::io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *expected);
}

auto create_parquet_with_stats(std::string const& filename)
{
  auto col0 = testdata::ascending<uint32_t>();
  auto col1 = testdata::descending<int64_t>();
  auto col2 = testdata::unordered<double>();

  auto const expected = table_view{{col0, col1, col2}};

  cudf::io::table_input_metadata expected_metadata(expected);
  expected_metadata.column_metadata[0].set_name("col_uint32");
  expected_metadata.column_metadata[1].set_name("col_int64");
  expected_metadata.column_metadata[2].set_name("col_double");

  auto const filepath = temp_env->get_temp_filepath(filename);
  const cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .metadata(std::move(expected_metadata))
      .row_group_size_rows(8000)
      .stats_level(cudf::io::statistics_freq::STATISTICS_ROWGROUP);
  cudf::io::write_parquet(out_opts);

  std::vector<std::unique_ptr<column>> columns;
  columns.push_back(col0.release());
  columns.push_back(col1.release());
  columns.push_back(col2.release());

  return std::pair{cudf::table{std::move(columns)}, filepath};
}

TEST_F(ParquetReaderTest, FilterIdentity)
{
  auto [src, filepath] = create_parquet_with_stats("FilterIdentity.parquet");

  // Filtering AST - identity function, always true.
  auto literal_value     = cudf::numeric_scalar<bool>(true);
  auto literal           = cudf::ast::literal(literal_value);
  auto filter_expression = cudf::ast::operation(cudf::ast::ast_operator::IDENTITY, literal);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
      .filter(filter_expression);
  auto result = cudf::io::read_parquet(read_opts);

  cudf::io::parquet_reader_options read_opts2 =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result2 = cudf::io::read_parquet(read_opts2);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *result2.tbl);
}

TEST_F(ParquetReaderTest, FilterWithColumnProjection)
{
  // col_uint32, col_int64, col_double
  auto [src, filepath] = create_parquet_with_stats("FilterWithColumnProjection.parquet");
  auto val             = cudf::numeric_scalar<uint32_t>{10};
  auto lit             = cudf::ast::literal{val};
  auto col_ref         = cudf::ast::column_name_reference{"col_uint32"};
  auto col_index       = cudf::ast::column_reference{0};
  auto filter_expr     = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_index, lit);

  auto predicate = cudf::compute_column(src, filter_expr);

  {  // column_name_reference in parquet filter (not present in column projection)
    auto read_expr       = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref, lit);
    auto projected_table = cudf::table_view{{src.get_column(2)}};
    auto expected        = cudf::apply_boolean_mask(projected_table, *predicate);

    auto read_opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                       .columns({"col_double"})
                       .filter(read_expr);
    auto result = cudf::io::read_parquet(read_opts);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *expected);
  }

  {  // column_reference in parquet filter (indices as per order of column projection)
    auto col_index2    = cudf::ast::column_reference{1};
    auto read_ref_expr = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_index2, lit);

    auto projected_table = cudf::table_view{{src.get_column(2), src.get_column(0)}};
    auto expected        = cudf::apply_boolean_mask(projected_table, *predicate);
    auto read_opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                       .columns({"col_double", "col_uint32"})
                       .filter(read_ref_expr);
    auto result = cudf::io::read_parquet(read_opts);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *expected);
  }

  // Error cases
  {  // column_reference is not same type as literal, column_reference index is out of bounds
    for (auto const index : {0, 2}) {
      auto col_index2    = cudf::ast::column_reference{index};
      auto read_ref_expr = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_index2, lit);
      auto read_opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                         .columns({"col_double", "col_uint32"})
                         .filter(read_ref_expr);
      EXPECT_THROW(cudf::io::read_parquet(read_opts), cudf::logic_error);
    }
  }
}

TEST_F(ParquetReaderTest, FilterReferenceExpression)
{
  auto [src, filepath] = create_parquet_with_stats("FilterReferenceExpression.parquet");
  // Filtering AST - table[0] < 150
  auto literal_value     = cudf::numeric_scalar<uint32_t>(150);
  auto literal           = cudf::ast::literal(literal_value);
  auto col_ref_0         = cudf::ast::column_reference(0);
  auto filter_expression = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, literal);

  // Expected result
  auto predicate = cudf::compute_column(src, filter_expression);
  auto expected  = cudf::apply_boolean_mask(src, *predicate);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
      .filter(filter_expression);
  auto result = cudf::io::read_parquet(read_opts);
  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *expected);
}

TEST_F(ParquetReaderTest, FilterNamedExpression)
{
  auto [src, filepath] = create_parquet_with_stats("NamedExpression.parquet");
  // Filtering AST - table["col_uint32"] < 150
  auto literal_value  = cudf::numeric_scalar<uint32_t>(150);
  auto literal        = cudf::ast::literal(literal_value);
  auto col_name_0     = cudf::ast::column_name_reference("col_uint32");
  auto parquet_filter = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_name_0, literal);
  auto col_ref_0      = cudf::ast::column_reference(0);
  auto table_filter   = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, literal);

  // Expected result
  auto predicate = cudf::compute_column(src, table_filter);
  auto expected  = cudf::apply_boolean_mask(src, *predicate);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
      .filter(parquet_filter);
  auto result = cudf::io::read_parquet(read_opts);

  // tests
  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *expected);
}

TEST_F(ParquetReaderTest, FilterMultiple1)
{
  using T = cudf::string_view;

  auto const [src, filepath] = create_parquet_typed_with_stats<T>("FilterMultiple1.parquet");
  auto const written_table   = src.view();

  // Filtering AST - 10000 < table[0] < 12000
  std::string const low  = "000010000";
  std::string const high = "000012000";
  auto lov               = cudf::string_scalar(low, true);
  auto hiv               = cudf::string_scalar(high, true);
  auto filter_col        = cudf::ast::column_reference(0);
  auto lo_lit            = cudf::ast::literal(lov);
  auto hi_lit            = cudf::ast::literal(hiv);
  auto expr_1 = cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, filter_col, lo_lit);
  auto expr_2 = cudf::ast::operation(cudf::ast::ast_operator::LESS, filter_col, hi_lit);
  auto expr_3 = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, expr_1, expr_2);

  // Expected result
  auto predicate = cudf::compute_column(written_table, expr_3);
  auto expected  = cudf::apply_boolean_mask(written_table, *predicate);

  auto si                  = cudf::io::source_info(filepath);
  auto builder             = cudf::io::parquet_reader_options::builder(si).filter(expr_3);
  auto table_with_metadata = cudf::io::read_parquet(builder);
  auto result              = table_with_metadata.tbl->view();

  // tests
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), result);
}

TEST_F(ParquetReaderTest, FilterMultiple2)
{
  // multiple conditions on same column.
  using T = cudf::string_view;

  auto const [src, filepath] = create_parquet_typed_with_stats<T>("FilterMultiple2.parquet");
  auto const written_table   = src.view();
  // 0-8000, 8001-16000, 16001-20000

  // Filtering AST
  // (table[0] >= "000010000" AND table[0] < "000012000") OR
  // (table[0] >= "000017000" AND table[0] < "000019000")
  std::string const low1  = "000010000";
  std::string const high1 = "000012000";
  auto lov                = cudf::string_scalar(low1, true);
  auto hiv                = cudf::string_scalar(high1, true);
  auto filter_col         = cudf::ast::column_reference(0);
  auto lo_lit             = cudf::ast::literal(lov);
  auto hi_lit             = cudf::ast::literal(hiv);
  auto expr_1 = cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, filter_col, lo_lit);
  auto expr_2 = cudf::ast::operation(cudf::ast::ast_operator::LESS, filter_col, hi_lit);
  auto expr_3 = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, expr_1, expr_2);
  std::string const low2  = "000017000";
  std::string const high2 = "000019000";
  auto lov2               = cudf::string_scalar(low2, true);
  auto hiv2               = cudf::string_scalar(high2, true);
  auto lo_lit2            = cudf::ast::literal(lov2);
  auto hi_lit2            = cudf::ast::literal(hiv2);
  auto expr_4 = cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, filter_col, lo_lit2);
  auto expr_5 = cudf::ast::operation(cudf::ast::ast_operator::LESS, filter_col, hi_lit2);
  auto expr_6 = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, expr_4, expr_5);
  auto expr_7 = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_OR, expr_3, expr_6);

  // Expected result
  auto predicate = cudf::compute_column(written_table, expr_7);
  auto expected  = cudf::apply_boolean_mask(written_table, *predicate);

  auto si                  = cudf::io::source_info(filepath);
  auto builder             = cudf::io::parquet_reader_options::builder(si).filter(expr_7);
  auto table_with_metadata = cudf::io::read_parquet(builder);
  auto result              = table_with_metadata.tbl->view();

  // tests
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), result);
}

TEST_F(ParquetReaderTest, FilterMultiple3)
{
  // multiple conditions with reference to multiple columns.
  // index and name references mixed.
  using T                    = uint32_t;
  auto const [src, filepath] = create_parquet_typed_with_stats<T>("FilterMultiple3.parquet");
  auto const written_table   = src.view();

  // Filtering AST - (table[0] >= 70 AND table[0] < 90) OR (table[1] >= 100 AND table[1] < 120)
  // row groups min, max:
  // table[0] 0-80, 81-160, 161-200.
  // table[1] 200-121, 120-41, 40-0.
  auto filter_col1  = cudf::ast::column_reference(0);
  auto filter_col2  = cudf::ast::column_name_reference("col1");
  T constexpr low1  = 70;
  T constexpr high1 = 90;
  T constexpr low2  = 100;
  T constexpr high2 = 120;
  auto lov          = cudf::numeric_scalar(low1, true);
  auto hiv          = cudf::numeric_scalar(high1, true);
  auto lo_lit1      = cudf::ast::literal(lov);
  auto hi_lit1      = cudf::ast::literal(hiv);
  auto expr_1  = cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, filter_col1, lo_lit1);
  auto expr_2  = cudf::ast::operation(cudf::ast::ast_operator::LESS, filter_col1, hi_lit1);
  auto expr_3  = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, expr_1, expr_2);
  auto lov2    = cudf::numeric_scalar(low2, true);
  auto hiv2    = cudf::numeric_scalar(high2, true);
  auto lo_lit2 = cudf::ast::literal(lov2);
  auto hi_lit2 = cudf::ast::literal(hiv2);
  auto expr_4  = cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, filter_col2, lo_lit2);
  auto expr_5  = cudf::ast::operation(cudf::ast::ast_operator::LESS, filter_col2, hi_lit2);
  auto expr_6  = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, expr_4, expr_5);
  // expression to test
  auto expr_7 = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_OR, expr_3, expr_6);

  // Expected result
  auto filter_col2_ref = cudf::ast::column_reference(1);
  auto expr_4_ref =
    cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, filter_col2_ref, lo_lit2);
  auto expr_5_ref = cudf::ast::operation(cudf::ast::ast_operator::LESS, filter_col2_ref, hi_lit2);
  auto expr_6_ref =
    cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, expr_4_ref, expr_5_ref);
  auto expr_7_ref = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_OR, expr_3, expr_6_ref);
  auto predicate  = cudf::compute_column(written_table, expr_7_ref);
  auto expected   = cudf::apply_boolean_mask(written_table, *predicate);

  auto si                  = cudf::io::source_info(filepath);
  auto builder             = cudf::io::parquet_reader_options::builder(si).filter(expr_7);
  auto table_with_metadata = cudf::io::read_parquet(builder);
  auto result              = table_with_metadata.tbl->view();

  // tests
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), result);
}

TEST_F(ParquetReaderTest, FilterSupported)
{
  using T                    = uint32_t;
  auto const [src, filepath] = create_parquet_typed_with_stats<T>("FilterSupported.parquet");
  auto const written_table   = src.view();

  // Filtering AST - ((table[0] > 70 AND table[0] <= 90) OR (table[1] >= 100 AND table[1] < 120))
  //              AND (table[1] != 110)
  // row groups min, max:
  // table[0] 0-80, 81-160, 161-200.
  // table[1] 200-121, 120-41, 40-0.
  auto filter_col1       = cudf::ast::column_reference(0);
  auto filter_col2       = cudf::ast::column_reference(1);
  T constexpr low1       = 70;
  T constexpr high1      = 90;
  T constexpr low2       = 100;
  T constexpr high2      = 120;
  T constexpr skip_value = 110;
  auto lov               = cudf::numeric_scalar(low1, true);
  auto hiv               = cudf::numeric_scalar(high1, true);
  auto lo_lit1           = cudf::ast::literal(lov);
  auto hi_lit1           = cudf::ast::literal(hiv);
  auto expr_1  = cudf::ast::operation(cudf::ast::ast_operator::GREATER, filter_col1, lo_lit1);
  auto expr_2  = cudf::ast::operation(cudf::ast::ast_operator::LESS_EQUAL, filter_col1, hi_lit1);
  auto expr_3  = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, expr_1, expr_2);
  auto lov2    = cudf::numeric_scalar(low2, true);
  auto hiv2    = cudf::numeric_scalar(high2, true);
  auto lo_lit2 = cudf::ast::literal(lov2);
  auto hi_lit2 = cudf::ast::literal(hiv2);
  auto expr_4  = cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, filter_col2, lo_lit2);
  auto expr_5  = cudf::ast::operation(cudf::ast::ast_operator::LESS, filter_col2, hi_lit2);
  auto expr_6  = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, expr_4, expr_5);
  auto expr_7  = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_OR, expr_3, expr_6);
  auto skip_ov = cudf::numeric_scalar(skip_value, true);
  auto skip_lit = cudf::ast::literal(skip_ov);
  auto expr_8   = cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, filter_col2, skip_lit);
  auto expr_9   = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, expr_7, expr_8);

  // Expected result
  auto predicate = cudf::compute_column(written_table, expr_9);
  auto expected  = cudf::apply_boolean_mask(written_table, *predicate);

  auto si                  = cudf::io::source_info(filepath);
  auto builder             = cudf::io::parquet_reader_options::builder(si).filter(expr_9);
  auto table_with_metadata = cudf::io::read_parquet(builder);
  auto result              = table_with_metadata.tbl->view();

  // tests
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), result);
}

TEST_F(ParquetReaderTest, FilterSupported2)
{
  using T                 = uint32_t;
  constexpr auto num_rows = 4000;
  auto elements0 =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i / 2000; });
  auto elements1 =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i / 1000; });
  auto elements2 =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i / 500; });
  auto col0 = cudf::test::fixed_width_column_wrapper<T>(elements0, elements0 + num_rows);
  auto col1 = cudf::test::fixed_width_column_wrapper<T>(elements1, elements1 + num_rows);
  auto col2 = cudf::test::fixed_width_column_wrapper<T>(elements2, elements2 + num_rows);
  auto const written_table = table_view{{col0, col1, col2}};
  auto const filepath      = temp_env->get_temp_filepath("FilterSupported2.parquet");
  {
    const cudf::io::parquet_writer_options out_opts =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, written_table)
        .row_group_size_rows(1000);
    cudf::io::write_parquet(out_opts);
  }
  auto si          = cudf::io::source_info(filepath);
  auto filter_col0 = cudf::ast::column_reference(0);
  auto filter_col1 = cudf::ast::column_reference(1);
  auto filter_col2 = cudf::ast::column_reference(2);
  auto s_value     = cudf::numeric_scalar<T>(1, true);
  auto lit_value   = cudf::ast::literal(s_value);

  auto test_expr = [&](auto& expr) {
    // Expected result
    auto predicate = cudf::compute_column(written_table, expr);
    auto expected  = cudf::apply_boolean_mask(written_table, *predicate);

    // tests
    auto builder             = cudf::io::parquet_reader_options::builder(si).filter(expr);
    auto table_with_metadata = cudf::io::read_parquet(builder);
    auto result              = table_with_metadata.tbl->view();

    CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), result);
  };

  // row groups min, max:
  // table[0] 0-0, 0-0, 1-1, 1-1
  // table[1] 0-0, 1-1, 2-2, 3-3
  // table[2] 0-1, 2-3, 4-5, 6-7

  // Filtering AST -   table[i] == 1
  {
    auto expr0 = cudf::ast::operation(cudf::ast::ast_operator::EQUAL, filter_col0, lit_value);
    test_expr(expr0);

    auto expr1 = cudf::ast::operation(cudf::ast::ast_operator::EQUAL, filter_col1, lit_value);
    test_expr(expr1);

    auto expr2 = cudf::ast::operation(cudf::ast::ast_operator::EQUAL, filter_col2, lit_value);
    test_expr(expr2);
  }
  // Filtering AST -   table[i] != 1
  {
    auto expr0 = cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, filter_col0, lit_value);
    test_expr(expr0);

    auto expr1 = cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, filter_col1, lit_value);
    test_expr(expr1);

    auto expr2 = cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, filter_col2, lit_value);
    test_expr(expr2);
  }
}

// Error types - type mismatch, invalid column name, invalid literal type, invalid operator,
// non-bool filter output type.
TEST_F(ParquetReaderTest, FilterErrors)
{
  using T                    = uint32_t;
  auto const [src, filepath] = create_parquet_typed_with_stats<T>("FilterErrors.parquet");
  auto const written_table   = src.view();
  auto si                    = cudf::io::source_info(filepath);

  // Filtering AST - invalid column index
  {
    auto filter_col1 = cudf::ast::column_reference(3);
    T constexpr low  = 100;
    auto lov         = cudf::numeric_scalar(low, true);
    auto low_lot     = cudf::ast::literal(lov);
    auto expr        = cudf::ast::operation(cudf::ast::ast_operator::LESS, filter_col1, low_lot);

    auto builder = cudf::io::parquet_reader_options::builder(si).filter(expr);
    EXPECT_THROW(cudf::io::read_parquet(builder), cudf::logic_error);
  }

  // Filtering AST - invalid column name
  {
    auto filter_col1 = cudf::ast::column_name_reference("col3");
    T constexpr low  = 100;
    auto lov         = cudf::numeric_scalar(low, true);
    auto low_lot     = cudf::ast::literal(lov);
    auto expr        = cudf::ast::operation(cudf::ast::ast_operator::LESS, filter_col1, low_lot);
    auto builder     = cudf::io::parquet_reader_options::builder(si).filter(expr);
    EXPECT_THROW(cudf::io::read_parquet(builder), cudf::logic_error);
  }

  // Filtering AST - incompatible literal type
  {
    auto filter_col1      = cudf::ast::column_name_reference("col0");
    auto filter_col2      = cudf::ast::column_reference(1);
    int64_t constexpr low = 100;
    auto lov              = cudf::numeric_scalar(low, true);
    auto low_lot          = cudf::ast::literal(lov);
    auto expr1    = cudf::ast::operation(cudf::ast::ast_operator::LESS, filter_col1, low_lot);
    auto expr2    = cudf::ast::operation(cudf::ast::ast_operator::LESS, filter_col2, low_lot);
    auto builder1 = cudf::io::parquet_reader_options::builder(si).filter(expr1);
    EXPECT_THROW(cudf::io::read_parquet(builder1), cudf::logic_error);

    auto builder2 = cudf::io::parquet_reader_options::builder(si).filter(expr2);
    EXPECT_THROW(cudf::io::read_parquet(builder2), cudf::logic_error);
  }

  // Filtering AST - "table[0] + 110" is invalid filter expression
  {
    auto filter_col1      = cudf::ast::column_reference(0);
    T constexpr add_value = 110;
    auto add_v            = cudf::numeric_scalar(add_value, true);
    auto add_lit          = cudf::ast::literal(add_v);
    auto expr_8 = cudf::ast::operation(cudf::ast::ast_operator::ADD, filter_col1, add_lit);

    auto si      = cudf::io::source_info(filepath);
    auto builder = cudf::io::parquet_reader_options::builder(si).filter(expr_8);
    EXPECT_THROW(cudf::io::read_parquet(builder), cudf::logic_error);

    // Expected result throw to show that the filter expression is invalid,
    // not a limitation of the parquet predicate pushdown.
    auto predicate = cudf::compute_column(written_table, expr_8);
    EXPECT_THROW(cudf::apply_boolean_mask(written_table, *predicate), cudf::logic_error);
  }

  // Filtering AST - INT64(table[0] < 100) non-bool expression
  {
    auto filter_col1 = cudf::ast::column_reference(0);
    T constexpr low  = 100;
    auto lov         = cudf::numeric_scalar(low, true);
    auto low_lot     = cudf::ast::literal(lov);
    auto bool_expr   = cudf::ast::operation(cudf::ast::ast_operator::LESS, filter_col1, low_lot);
    auto cast        = cudf::ast::operation(cudf::ast::ast_operator::CAST_TO_INT64, bool_expr);

    auto builder = cudf::io::parquet_reader_options::builder(si).filter(cast);
    EXPECT_THROW(cudf::io::read_parquet(builder), cudf::logic_error);
    EXPECT_NO_THROW(cudf::compute_column(written_table, cast));
    auto predicate = cudf::compute_column(written_table, cast);
    EXPECT_NE(predicate->view().type().id(), cudf::type_id::BOOL8);
  }
}

// Filter without stats information in file.
TEST_F(ParquetReaderTest, FilterNoStats)
{
  using T                 = uint32_t;
  constexpr auto num_rows = 16000;
  auto elements =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i / 1000; });
  auto col0 = cudf::test::fixed_width_column_wrapper<T>(elements, elements + num_rows);
  auto const written_table = table_view{{col0}};
  auto const filepath      = temp_env->get_temp_filepath("FilterNoStats.parquet");
  {
    const cudf::io::parquet_writer_options out_opts =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, written_table)
        .row_group_size_rows(8000)
        .stats_level(cudf::io::statistics_freq::STATISTICS_NONE);
    cudf::io::write_parquet(out_opts);
  }
  auto si          = cudf::io::source_info(filepath);
  auto filter_col0 = cudf::ast::column_reference(0);
  auto s_value     = cudf::numeric_scalar<T>(1, true);
  auto lit_value   = cudf::ast::literal(s_value);

  // row groups min, max:
  // table[0] 0-0, 1-1, 2-2, 3-3
  // Filtering AST - table[0] > 1
  auto expr = cudf::ast::operation(cudf::ast::ast_operator::GREATER, filter_col0, lit_value);

  // Expected result
  auto predicate = cudf::compute_column(written_table, expr);
  auto expected  = cudf::apply_boolean_mask(written_table, *predicate);

  // tests
  auto builder             = cudf::io::parquet_reader_options::builder(si).filter(expr);
  auto table_with_metadata = cudf::io::read_parquet(builder);
  auto result              = table_with_metadata.tbl->view();

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), result);
}

// Filter for float column with NaN values
TEST_F(ParquetReaderTest, FilterFloatNAN)
{
  constexpr auto num_rows = 24000;
  auto elements           = cudf::detail::make_counting_transform_iterator(
    0, [num_rows](auto i) { return i > num_rows / 2 ? NAN : i; });
  auto col0 = cudf::test::fixed_width_column_wrapper<float>(elements, elements + num_rows);
  auto col1 = cudf::test::fixed_width_column_wrapper<double>(elements, elements + num_rows);

  auto const written_table = table_view{{col0, col1}};
  auto const filepath      = temp_env->get_temp_filepath("FilterFloatNAN.parquet");
  {
    const cudf::io::parquet_writer_options out_opts =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, written_table)
        .row_group_size_rows(8000);
    cudf::io::write_parquet(out_opts);
  }
  auto si          = cudf::io::source_info(filepath);
  auto filter_col0 = cudf::ast::column_reference(0);
  auto filter_col1 = cudf::ast::column_reference(1);
  auto s0_value    = cudf::numeric_scalar<float>(NAN, true);
  auto lit0_value  = cudf::ast::literal(s0_value);
  auto s1_value    = cudf::numeric_scalar<double>(NAN, true);
  auto lit1_value  = cudf::ast::literal(s1_value);

  // row groups min, max:
  // table[0] 0-0, 1-1, 2-2, 3-3
  // Filtering AST - table[0] == NAN, table[1] != NAN
  auto expr_eq  = cudf::ast::operation(cudf::ast::ast_operator::EQUAL, filter_col0, lit0_value);
  auto expr_neq = cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, filter_col1, lit1_value);

  // Expected result
  auto predicate0 = cudf::compute_column(written_table, expr_eq);
  auto expected0  = cudf::apply_boolean_mask(written_table, *predicate0);
  auto predicate1 = cudf::compute_column(written_table, expr_neq);
  auto expected1  = cudf::apply_boolean_mask(written_table, *predicate1);

  // tests
  auto builder0             = cudf::io::parquet_reader_options::builder(si).filter(expr_eq);
  auto table_with_metadata0 = cudf::io::read_parquet(builder0);
  auto result0              = table_with_metadata0.tbl->view();
  auto builder1             = cudf::io::parquet_reader_options::builder(si).filter(expr_neq);
  auto table_with_metadata1 = cudf::io::read_parquet(builder1);
  auto result1              = table_with_metadata1.tbl->view();

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected0->view(), result0);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected1->view(), result1);
}

TEST_F(ParquetReaderTest, RepeatedNoAnnotations)
{
  constexpr std::array<unsigned char, 662> repeated_bytes{
    0x50, 0x41, 0x52, 0x31, 0x15, 0x04, 0x15, 0x30, 0x15, 0x30, 0x4c, 0x15, 0x0c, 0x15, 0x00, 0x12,
    0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x04, 0x00,
    0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00, 0x15, 0x00, 0x15, 0x0a, 0x15, 0x0a,
    0x2c, 0x15, 0x0c, 0x15, 0x10, 0x15, 0x06, 0x15, 0x06, 0x00, 0x00, 0x03, 0x03, 0x88, 0xc6, 0x02,
    0x26, 0x80, 0x01, 0x1c, 0x15, 0x02, 0x19, 0x25, 0x00, 0x10, 0x19, 0x18, 0x02, 0x69, 0x64, 0x15,
    0x00, 0x16, 0x0c, 0x16, 0x78, 0x16, 0x78, 0x26, 0x54, 0x26, 0x08, 0x00, 0x00, 0x15, 0x04, 0x15,
    0x40, 0x15, 0x40, 0x4c, 0x15, 0x08, 0x15, 0x00, 0x12, 0x00, 0x00, 0xe3, 0x0c, 0x23, 0x4b, 0x01,
    0x00, 0x00, 0x00, 0xc7, 0x35, 0x3a, 0x42, 0x00, 0x00, 0x00, 0x00, 0x8e, 0x6b, 0x74, 0x84, 0x00,
    0x00, 0x00, 0x00, 0x55, 0xa1, 0xae, 0xc6, 0x00, 0x00, 0x00, 0x00, 0x15, 0x00, 0x15, 0x22, 0x15,
    0x22, 0x2c, 0x15, 0x10, 0x15, 0x10, 0x15, 0x06, 0x15, 0x06, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
    0x03, 0xc0, 0x03, 0x00, 0x00, 0x00, 0x03, 0x90, 0xaa, 0x02, 0x03, 0x94, 0x03, 0x26, 0xda, 0x02,
    0x1c, 0x15, 0x04, 0x19, 0x25, 0x00, 0x10, 0x19, 0x38, 0x0c, 0x70, 0x68, 0x6f, 0x6e, 0x65, 0x4e,
    0x75, 0x6d, 0x62, 0x65, 0x72, 0x73, 0x05, 0x70, 0x68, 0x6f, 0x6e, 0x65, 0x06, 0x6e, 0x75, 0x6d,
    0x62, 0x65, 0x72, 0x15, 0x00, 0x16, 0x10, 0x16, 0xa0, 0x01, 0x16, 0xa0, 0x01, 0x26, 0x96, 0x02,
    0x26, 0xba, 0x01, 0x00, 0x00, 0x15, 0x04, 0x15, 0x24, 0x15, 0x24, 0x4c, 0x15, 0x04, 0x15, 0x00,
    0x12, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x68, 0x6f, 0x6d, 0x65, 0x06, 0x00, 0x00, 0x00, 0x6d,
    0x6f, 0x62, 0x69, 0x6c, 0x65, 0x15, 0x00, 0x15, 0x20, 0x15, 0x20, 0x2c, 0x15, 0x10, 0x15, 0x10,
    0x15, 0x06, 0x15, 0x06, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03, 0xc0, 0x03, 0x00, 0x00, 0x00,
    0x03, 0x90, 0xef, 0x01, 0x03, 0x04, 0x26, 0xcc, 0x04, 0x1c, 0x15, 0x0c, 0x19, 0x25, 0x00, 0x10,
    0x19, 0x38, 0x0c, 0x70, 0x68, 0x6f, 0x6e, 0x65, 0x4e, 0x75, 0x6d, 0x62, 0x65, 0x72, 0x73, 0x05,
    0x70, 0x68, 0x6f, 0x6e, 0x65, 0x04, 0x6b, 0x69, 0x6e, 0x64, 0x15, 0x00, 0x16, 0x10, 0x16, 0x82,
    0x01, 0x16, 0x82, 0x01, 0x26, 0x8a, 0x04, 0x26, 0xca, 0x03, 0x00, 0x00, 0x15, 0x02, 0x19, 0x6c,
    0x48, 0x04, 0x75, 0x73, 0x65, 0x72, 0x15, 0x04, 0x00, 0x15, 0x02, 0x25, 0x00, 0x18, 0x02, 0x69,
    0x64, 0x00, 0x35, 0x02, 0x18, 0x0c, 0x70, 0x68, 0x6f, 0x6e, 0x65, 0x4e, 0x75, 0x6d, 0x62, 0x65,
    0x72, 0x73, 0x15, 0x02, 0x00, 0x35, 0x04, 0x18, 0x05, 0x70, 0x68, 0x6f, 0x6e, 0x65, 0x15, 0x04,
    0x00, 0x15, 0x04, 0x25, 0x00, 0x18, 0x06, 0x6e, 0x75, 0x6d, 0x62, 0x65, 0x72, 0x00, 0x15, 0x0c,
    0x25, 0x02, 0x18, 0x04, 0x6b, 0x69, 0x6e, 0x64, 0x25, 0x00, 0x00, 0x16, 0x00, 0x19, 0x1c, 0x19,
    0x3c, 0x26, 0x80, 0x01, 0x1c, 0x15, 0x02, 0x19, 0x25, 0x00, 0x10, 0x19, 0x18, 0x02, 0x69, 0x64,
    0x15, 0x00, 0x16, 0x0c, 0x16, 0x78, 0x16, 0x78, 0x26, 0x54, 0x26, 0x08, 0x00, 0x00, 0x26, 0xda,
    0x02, 0x1c, 0x15, 0x04, 0x19, 0x25, 0x00, 0x10, 0x19, 0x38, 0x0c, 0x70, 0x68, 0x6f, 0x6e, 0x65,
    0x4e, 0x75, 0x6d, 0x62, 0x65, 0x72, 0x73, 0x05, 0x70, 0x68, 0x6f, 0x6e, 0x65, 0x06, 0x6e, 0x75,
    0x6d, 0x62, 0x65, 0x72, 0x15, 0x00, 0x16, 0x10, 0x16, 0xa0, 0x01, 0x16, 0xa0, 0x01, 0x26, 0x96,
    0x02, 0x26, 0xba, 0x01, 0x00, 0x00, 0x26, 0xcc, 0x04, 0x1c, 0x15, 0x0c, 0x19, 0x25, 0x00, 0x10,
    0x19, 0x38, 0x0c, 0x70, 0x68, 0x6f, 0x6e, 0x65, 0x4e, 0x75, 0x6d, 0x62, 0x65, 0x72, 0x73, 0x05,
    0x70, 0x68, 0x6f, 0x6e, 0x65, 0x04, 0x6b, 0x69, 0x6e, 0x64, 0x15, 0x00, 0x16, 0x10, 0x16, 0x82,
    0x01, 0x16, 0x82, 0x01, 0x26, 0x8a, 0x04, 0x26, 0xca, 0x03, 0x00, 0x00, 0x16, 0x9a, 0x03, 0x16,
    0x0c, 0x00, 0x28, 0x49, 0x70, 0x61, 0x72, 0x71, 0x75, 0x65, 0x74, 0x2d, 0x72, 0x73, 0x20, 0x76,
    0x65, 0x72, 0x73, 0x69, 0x6f, 0x6e, 0x20, 0x30, 0x2e, 0x33, 0x2e, 0x30, 0x20, 0x28, 0x62, 0x75,
    0x69, 0x6c, 0x64, 0x20, 0x62, 0x34, 0x35, 0x63, 0x65, 0x37, 0x63, 0x62, 0x61, 0x32, 0x31, 0x39,
    0x39, 0x66, 0x32, 0x32, 0x64, 0x39, 0x33, 0x32, 0x36, 0x39, 0x63, 0x31, 0x35, 0x30, 0x64, 0x38,
    0x61, 0x38, 0x33, 0x39, 0x31, 0x36, 0x63, 0x36, 0x39, 0x62, 0x35, 0x65, 0x29, 0x00, 0x32, 0x01,
    0x00, 0x00, 0x50, 0x41, 0x52, 0x31};

  auto read_opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{
    reinterpret_cast<char const*>(repeated_bytes.data()), repeated_bytes.size()});
  auto result    = cudf::io::read_parquet(read_opts);

  EXPECT_EQ(result.tbl->view().column(0).size(), 6);
  EXPECT_EQ(result.tbl->view().num_columns(), 2);

  column_wrapper<int32_t> col0{1, 2, 3, 4, 5, 6};
  column_wrapper<int64_t> child0{{5555555555l, 1111111111l, 1111111111l, 2222222222l, 3333333333l}};
  cudf::test::strings_column_wrapper child1{{"-", "home", "home", "-", "mobile"},
                                            {false, true, true, false, true}};
  auto struct_col = cudf::test::structs_column_wrapper{{child0, child1}};

  auto list_offsets_column =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 0, 0, 0, 1, 2, 5}.release();
  auto num_list_rows = list_offsets_column->size() - 1;

  auto mask = cudf::create_null_mask(6, cudf::mask_state::ALL_VALID);
  cudf::set_null_mask(static_cast<cudf::bitmask_type*>(mask.data()), 0, 2, false);

  auto list_col = cudf::make_lists_column(
    num_list_rows, std::move(list_offsets_column), struct_col.release(), 2, std::move(mask));

  std::vector<std::unique_ptr<cudf::column>> struct_children;
  struct_children.push_back(std::move(list_col));

  auto outer_struct = cudf::test::structs_column_wrapper{{std::move(struct_children)},
                                                         {false, false, true, true, true, true}};
  table_view expected{{col0, outer_struct}};

  CUDF_TEST_EXPECT_TABLES_EQUAL(result.tbl->view(), expected);
}

TEST_F(ParquetReaderTest, DeltaSkipRowsWithNulls)
{
  using cudf::io::column_encoding;
  constexpr int num_rows = 10'000;
  constexpr auto seed    = 21337;

  std::mt19937 engine{seed};
  auto int32_list_nulls = make_parquet_list_col<int32_t>(engine, num_rows, 5, true);
  auto int32_list       = make_parquet_list_col<int32_t>(engine, num_rows, 5, false);
  auto int64_list_nulls = make_parquet_list_col<int64_t>(engine, num_rows, 5, true);
  auto int64_list       = make_parquet_list_col<int64_t>(engine, num_rows, 5, false);
  auto int16_list_nulls = make_parquet_list_col<int16_t>(engine, num_rows, 5, true);
  auto int16_list       = make_parquet_list_col<int16_t>(engine, num_rows, 5, false);
  auto int8_list_nulls  = make_parquet_list_col<int8_t>(engine, num_rows, 5, true);
  auto int8_list        = make_parquet_list_col<int8_t>(engine, num_rows, 5, false);

  auto str_list_nulls     = make_parquet_string_list_col(engine, num_rows, 5, 32, true);
  auto str_list           = make_parquet_string_list_col(engine, num_rows, 5, 32, false);
  auto big_str_list_nulls = make_parquet_string_list_col(engine, num_rows, 5, 256, true);
  auto big_str_list       = make_parquet_string_list_col(engine, num_rows, 5, 256, false);

  auto int32_data   = random_values<int32_t>(num_rows);
  auto int64_data   = random_values<int64_t>(num_rows);
  auto int16_data   = random_values<int16_t>(num_rows);
  auto int8_data    = random_values<int8_t>(num_rows);
  auto str_data     = string_values(engine, num_rows, 32);
  auto big_str_data = string_values(engine, num_rows, 256);

  auto const validity = random_validity(engine);
  auto const no_nulls = cudf::test::iterators::no_nulls();
  column_wrapper<int32_t> int32_nulls_col{int32_data.begin(), int32_data.end(), validity};
  column_wrapper<int32_t> int32_col{int32_data.begin(), int32_data.end(), no_nulls};
  column_wrapper<int64_t> int64_nulls_col{int64_data.begin(), int64_data.end(), validity};
  column_wrapper<int64_t> int64_col{int64_data.begin(), int64_data.end(), no_nulls};

  auto str_col = cudf::test::strings_column_wrapper(str_data.begin(), str_data.end(), no_nulls);
  auto str_col_nulls = cudf::purge_nonempty_nulls(
    cudf::test::strings_column_wrapper(str_data.begin(), str_data.end(), validity));
  auto big_str_col =
    cudf::test::strings_column_wrapper(big_str_data.begin(), big_str_data.end(), no_nulls);
  auto big_str_col_nulls = cudf::purge_nonempty_nulls(
    cudf::test::strings_column_wrapper(big_str_data.begin(), big_str_data.end(), validity));

  cudf::table_view tbl({int32_col,   int32_nulls_col,    *int32_list,   *int32_list_nulls,
                        int64_col,   int64_nulls_col,    *int64_list,   *int64_list_nulls,
                        *int16_list, *int16_list_nulls,  *int8_list,    *int8_list_nulls,
                        str_col,     *str_col_nulls,     *str_list,     *str_list_nulls,
                        big_str_col, *big_str_col_nulls, *big_str_list, *big_str_list_nulls,
                        str_col,     *str_col_nulls,     *str_list,     *str_list_nulls,
                        big_str_col, *big_str_col_nulls, *big_str_list, *big_str_list_nulls});

  auto const filepath = temp_env->get_temp_filepath("DeltaSkipRowsWithNulls.parquet");
  auto input_metadata = cudf::io::table_input_metadata{tbl};
  for (int i = 12; i <= 27; ++i) {
    input_metadata.column_metadata[i].set_encoding(
      i <= 19 ? column_encoding::DELTA_LENGTH_BYTE_ARRAY : column_encoding::DELTA_BYTE_ARRAY);
  }

  auto const out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, tbl)
      .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN)
      .compression(cudf::io::compression_type::NONE)
      .dictionary_policy(cudf::io::dictionary_policy::NEVER)
      .max_page_size_rows(5'000)
      .write_v2_headers(true)
      .build();
  cudf::io::write_parquet(out_opts);

  // skip_rows / num_rows
  // clang-format off
  std::vector<std::pair<int, int>> params{
    // skip and then read rest of file
    {-1, -1}, {1, -1}, {2, -1}, {32, -1}, {33, -1}, {128, -1}, {1000, -1},
    // no skip but read fewer rows
    {0, 1}, {0, 2}, {0, 31}, {0, 32}, {0, 33}, {0, 128}, {0, 129}, {0, 130},
    // skip and truncate
    {1, 32}, {1, 33}, {32, 32}, {33, 139},
    // cross page boundaries
    {3'000, 5'000}
  };

  // clang-format on
  for (auto p : params) {
    cudf::io::parquet_reader_options read_args =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
    if (p.first >= 0) { read_args.set_skip_rows(p.first); }
    if (p.second >= 0) { read_args.set_num_rows(p.second); }
    auto result = cudf::io::read_parquet(read_args);

    p.first  = p.first < 0 ? 0 : p.first;
    p.second = p.second < 0 ? num_rows - p.first : p.second;
    std::vector<cudf::size_type> slice_indices{p.first, p.first + p.second};
    std::vector<cudf::table_view> expected = cudf::slice(tbl, slice_indices);

    CUDF_TEST_EXPECT_TABLES_EQUAL(result.tbl->view(), expected[0]);

    // test writing the result back out as a further check of the delta writer's correctness
    std::vector<char> out_buffer;
    cudf::io::parquet_writer_options out_opts2 =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info{&out_buffer},
                                                result.tbl->view())
        .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN)
        .compression(cudf::io::compression_type::NONE)
        .dictionary_policy(cudf::io::dictionary_policy::NEVER)
        .max_page_size_rows(5'000)
        .write_v2_headers(true);
    cudf::io::write_parquet(out_opts2);

    cudf::io::parquet_reader_options default_in_opts = cudf::io::parquet_reader_options::builder(
      cudf::io::source_info{out_buffer.data(), out_buffer.size()});
    auto const result2 = cudf::io::read_parquet(default_in_opts);

    CUDF_TEST_EXPECT_TABLES_EQUAL(result.tbl->view(), result2.tbl->view());
  }
}

TEST_F(ParquetReaderTest, DeltaByteArraySkipAllValid)
{
  // test that the DELTA_BYTE_ARRAY decoder can handle the case where skip rows skips all valid
  // values in a page. see #15075
  constexpr int num_rows  = 500;
  constexpr int num_valid = 150;

  auto const ones = thrust::make_constant_iterator("one");

  auto valids = cudf::detail::make_counting_transform_iterator(
    0, [num_valid](auto i) { return i < num_valid; });
  auto const col      = cudf::test::strings_column_wrapper{ones, ones + num_rows, valids};
  auto const expected = table_view({col});

  auto input_metadata = cudf::io::table_input_metadata{expected};
  input_metadata.column_metadata[0].set_encoding(cudf::io::column_encoding::DELTA_BYTE_ARRAY);

  auto const filepath = temp_env->get_temp_filepath("DeltaByteArraySkipAllValid.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .write_v2_headers(true)
      .metadata(input_metadata)
      .dictionary_policy(cudf::io::dictionary_policy::NEVER);
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
      .skip_rows(num_valid + 1);
  auto result = cudf::io::read_parquet(in_opts);
  CUDF_TEST_EXPECT_TABLES_EQUAL(cudf::slice(expected, {num_valid + 1, num_rows}),
                                result.tbl->view());
}

// test that using page stats is working for full reads and various skip rows
TEST_F(ParquetReaderTest, StringsWithPageStats)
{
  constexpr int num_rows = 10'000;
  constexpr auto seed    = 21337;

  std::mt19937 engine{seed};
  auto int32_list_nulls = make_parquet_list_col<int32_t>(engine, num_rows, 5, true);
  auto int32_list       = make_parquet_list_col<int32_t>(engine, num_rows, 5, false);
  auto int64_list_nulls = make_parquet_list_col<int64_t>(engine, num_rows, 5, true);
  auto int64_list       = make_parquet_list_col<int64_t>(engine, num_rows, 5, false);
  auto int16_list_nulls = make_parquet_list_col<int16_t>(engine, num_rows, 5, true);
  auto int16_list       = make_parquet_list_col<int16_t>(engine, num_rows, 5, false);
  auto int8_list_nulls  = make_parquet_list_col<int8_t>(engine, num_rows, 5, true);
  auto int8_list        = make_parquet_list_col<int8_t>(engine, num_rows, 5, false);

  auto str_list_nulls     = make_parquet_string_list_col(engine, num_rows, 5, 32, true);
  auto str_list           = make_parquet_string_list_col(engine, num_rows, 5, 32, false);
  auto big_str_list_nulls = make_parquet_string_list_col(engine, num_rows, 5, 256, true);
  auto big_str_list       = make_parquet_string_list_col(engine, num_rows, 5, 256, false);

  auto int32_data   = random_values<int32_t>(num_rows);
  auto int64_data   = random_values<int64_t>(num_rows);
  auto int16_data   = random_values<int16_t>(num_rows);
  auto int8_data    = random_values<int8_t>(num_rows);
  auto str_data     = string_values(engine, num_rows, 32);
  auto big_str_data = string_values(engine, num_rows, 256);

  auto const validity = random_validity(engine);
  auto const no_nulls = cudf::test::iterators::no_nulls();
  column_wrapper<int32_t> int32_nulls_col{int32_data.begin(), int32_data.end(), validity};
  column_wrapper<int32_t> int32_col{int32_data.begin(), int32_data.end(), no_nulls};
  column_wrapper<int64_t> int64_nulls_col{int64_data.begin(), int64_data.end(), validity};
  column_wrapper<int64_t> int64_col{int64_data.begin(), int64_data.end(), no_nulls};

  auto str_col = cudf::test::strings_column_wrapper(str_data.begin(), str_data.end(), no_nulls);
  auto str_col_nulls = cudf::purge_nonempty_nulls(
    cudf::test::strings_column_wrapper(str_data.begin(), str_data.end(), validity));
  auto big_str_col =
    cudf::test::strings_column_wrapper(big_str_data.begin(), big_str_data.end(), no_nulls);
  auto big_str_col_nulls = cudf::purge_nonempty_nulls(
    cudf::test::strings_column_wrapper(big_str_data.begin(), big_str_data.end(), validity));

  cudf::table_view tbl({int32_col,   int32_nulls_col,    *int32_list,   *int32_list_nulls,
                        int64_col,   int64_nulls_col,    *int64_list,   *int64_list_nulls,
                        *int16_list, *int16_list_nulls,  *int8_list,    *int8_list_nulls,
                        str_col,     *str_col_nulls,     *str_list,     *str_list_nulls,
                        big_str_col, *big_str_col_nulls, *big_str_list, *big_str_list_nulls});

  auto const filepath = temp_env->get_temp_filepath("StringsWithPageStats.parquet");
  auto const out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, tbl)
      .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN)
      .max_page_size_rows(5'000)
      .build();
  cudf::io::write_parquet(out_opts);

  // skip_rows / num_rows
  // clang-format off
  std::vector<std::pair<int, int>> params{
    // skip and then read rest of file
    {-1, -1}, {1, -1}, {2, -1}, {32, -1}, {33, -1}, {128, -1}, {1'000, -1},
    // no skip but truncate
    {0, 1'000}, {0, 6'000},
    // cross page boundaries
    {3'000, 5'000}
  };

  // clang-format on
  for (auto p : params) {
    cudf::io::parquet_reader_options read_args =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
    if (p.first >= 0) { read_args.set_skip_rows(p.first); }
    if (p.second >= 0) { read_args.set_num_rows(p.second); }
    auto result = cudf::io::read_parquet(read_args);

    p.first  = p.first < 0 ? 0 : p.first;
    p.second = p.second < 0 ? num_rows - p.first : p.second;
    std::vector<cudf::size_type> slice_indices{p.first, p.first + p.second};
    std::vector<cudf::table_view> expected = cudf::slice(tbl, slice_indices);

    CUDF_TEST_EXPECT_TABLES_EQUAL(result.tbl->view(), expected[0]);
  }
}

TEST_F(ParquetReaderTest, NumRowsPerSource)
{
  int constexpr num_rows          = 10'723;  // A prime number
  int constexpr rows_in_row_group = 500;

  // Table with single col of random int64 values
  auto const int64_data = random_values<int64_t>(num_rows);
  column_wrapper<int64_t> const int64_col{
    int64_data.begin(), int64_data.end(), cudf::test::iterators::no_nulls()};
  cudf::table_view const expected({int64_col});

  // Write to Parquet
  auto const filepath = temp_env->get_temp_filepath("NumRowsPerSource.parquet");
  auto const out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .row_group_size_rows(rows_in_row_group)
      .build();
  cudf::io::write_parquet(out_opts);

  // Read single data source entirely
  {
    auto const in_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath}).build();
    auto const result = cudf::io::read_parquet(in_opts);

    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
    EXPECT_EQ(result.metadata.num_rows_per_source.size(), 1);
    EXPECT_EQ(result.metadata.num_rows_per_source[0], num_rows);
  }

  // Read rows_to_read rows skipping rows_to_skip from single data source
  {
    auto constexpr rows_to_skip = 557;  // a prime number != rows_in_row_group
    auto constexpr rows_to_read = 7'232;
    auto const in_opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                           .skip_rows(rows_to_skip)
                           .num_rows(rows_to_read)
                           .build();
    auto const result = cudf::io::read_parquet(in_opts);
    column_wrapper<int64_t> int64_col_selected{int64_data.begin() + rows_to_skip,
                                               int64_data.begin() + rows_to_skip + rows_to_read,
                                               cudf::test::iterators::no_nulls()};

    cudf::table_view const expected_selected({int64_col_selected});

    CUDF_TEST_EXPECT_TABLES_EQUAL(expected_selected, result.tbl->view());
    EXPECT_EQ(result.metadata.num_rows_per_source.size(), 1);
    EXPECT_EQ(result.metadata.num_rows_per_source[0], rows_to_read);
  }

  // Filtered read from single data source
  {
    auto constexpr max_value = 100;
    auto literal_value       = cudf::numeric_scalar<int64_t>{max_value};
    auto literal             = cudf::ast::literal{literal_value};
    auto col_ref             = cudf::ast::column_reference(0);
    auto filter_expression =
      cudf::ast::operation(cudf::ast::ast_operator::LESS_EQUAL, col_ref, literal);

    auto const in_opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
                           .filter(filter_expression)
                           .build();

    std::vector<int64_t> int64_data_filtered;
    int64_data_filtered.reserve(num_rows);
    std::copy_if(
      int64_data.begin(), int64_data.end(), std::back_inserter(int64_data_filtered), [=](auto val) {
        return val <= max_value;
      });
    column_wrapper<int64_t> int64_col_filtered{
      int64_data_filtered.begin(), int64_data_filtered.end(), cudf::test::iterators::no_nulls()};

    cudf::table_view expected_filtered({int64_col_filtered});

    auto const result = cudf::io::read_parquet(in_opts);

    CUDF_TEST_EXPECT_TABLES_EQUAL(expected_filtered, result.tbl->view());
    EXPECT_EQ(result.metadata.num_rows_per_source.size(), 0);
  }

  // Read two data sources skipping the first entire file completely
  {
    auto constexpr rows_to_skip = 15'723;
    auto constexpr nsources     = 2;
    std::vector<std::string> const datasources(nsources, filepath);

    auto const in_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{datasources})
        .skip_rows(rows_to_skip)
        .build();

    auto const result = cudf::io::read_parquet(in_opts);

    column_wrapper<int64_t> int64_col_selected{int64_data.begin() + rows_to_skip - num_rows,
                                               int64_data.end(),
                                               cudf::test::iterators::no_nulls()};

    cudf::table_view const expected_selected({int64_col_selected});

    CUDF_TEST_EXPECT_TABLES_EQUAL(expected_selected, result.tbl->view());
    EXPECT_EQ(result.metadata.num_rows_per_source.size(), 2);
    EXPECT_EQ(result.metadata.num_rows_per_source[0], 0);
    EXPECT_EQ(result.metadata.num_rows_per_source[1], nsources * num_rows - rows_to_skip);
  }

  // Read ten data sources entirely
  {
    auto constexpr nsources = 10;
    std::vector<std::string> const datasources(nsources, filepath);

    auto const in_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{datasources}).build();
    auto const result = cudf::io::read_parquet(in_opts);

    // Initialize expected_counts
    std::vector<size_t> const expected_counts(nsources, num_rows);

    EXPECT_EQ(result.metadata.num_rows_per_source.size(), nsources);
    EXPECT_TRUE(std::equal(expected_counts.cbegin(),
                           expected_counts.cend(),
                           result.metadata.num_rows_per_source.cbegin()));
  }

  // Read rows_to_read rows skipping rows_to_skip (> two sources) from ten data sources
  {
    auto constexpr rows_to_skip = 25'999;
    auto constexpr rows_to_read = 47'232;

    auto constexpr nsources = 10;
    std::vector<std::string> const datasources(nsources, filepath);

    auto const in_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{datasources})
        .skip_rows(rows_to_skip)
        .num_rows(rows_to_read)
        .build();

    auto const result = cudf::io::read_parquet(in_opts);

    // Initialize expected_counts
    std::vector<size_t> expected_counts(nsources, num_rows);

    // Adjust expected_counts for rows_to_skip
    int64_t counter = 0;
    for (auto& nrows : expected_counts) {
      if (counter < rows_to_skip) {
        counter += nrows;
        nrows = (counter >= rows_to_skip) ? counter - rows_to_skip : 0;
      } else {
        break;
      }
    }

    // Reset the counter
    counter = 0;

    // Adjust expected_counts for rows_to_read
    for (auto& nrows : expected_counts) {
      if (counter < rows_to_read) {
        counter += nrows;
        nrows = (counter >= rows_to_read) ? rows_to_read - counter + nrows : nrows;
      } else if (counter > rows_to_read) {
        nrows = 0;
      }
    }

    EXPECT_EQ(result.metadata.num_rows_per_source.size(), nsources);
    EXPECT_TRUE(std::equal(expected_counts.cbegin(),
                           expected_counts.cend(),
                           result.metadata.num_rows_per_source.cbegin()));
  }
}

TEST_F(ParquetReaderTest, NumRowsPerSourceEmptyTable)
{
  auto const nsources = 10;

  column_wrapper<int64_t> const int64_empty_col{};
  cudf::table_view const expected_empty({int64_empty_col});

  // Write to Parquet
  auto const filepath_empty = temp_env->get_temp_filepath("NumRowsPerSourceEmpty.parquet");
  auto const out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath_empty}, expected_empty)
      .build();
  cudf::io::write_parquet(out_opts);

  // Read from Parquet
  std::vector<std::string> const datasources(nsources, filepath_empty);

  auto const in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{datasources}).build();
  auto const result = cudf::io::read_parquet(in_opts);

  // Initialize expected_counts
  std::vector<size_t> const expected_counts(nsources, 0);

  EXPECT_EQ(result.metadata.num_rows_per_source.size(), nsources);
  EXPECT_TRUE(std::equal(expected_counts.cbegin(),
                         expected_counts.cend(),
                         result.metadata.num_rows_per_source.cbegin()));
}

///////////////////
// metadata tests

// Test fixture for metadata tests
struct ParquetMetadataReaderTest : public cudf::test::BaseFixture {
  std::string print(cudf::io::parquet_column_schema schema, int depth = 0)
  {
    std::string child_str;
    for (auto const& child : schema.children()) {
      child_str += print(child, depth + 1);
    }
    return std::string(depth, ' ') + schema.name() + "\n" + child_str;
  }
};

TEST_F(ParquetMetadataReaderTest, TestBasic)
{
  auto const num_rows = 1200;

  auto ints   = random_values<int>(num_rows);
  auto floats = random_values<float>(num_rows);
  column_wrapper<int> int_col(ints.begin(), ints.end());
  column_wrapper<float> float_col(floats.begin(), floats.end());

  table_view expected({int_col, float_col});

  cudf::io::table_input_metadata expected_metadata(expected);
  expected_metadata.column_metadata[0].set_name("int_col");
  expected_metadata.column_metadata[1].set_name("float_col");

  auto filepath = temp_env->get_temp_filepath("MetadataTest.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .metadata(std::move(expected_metadata));
  cudf::io::write_parquet(out_opts);

  auto meta = read_parquet_metadata(cudf::io::source_info{filepath});
  EXPECT_EQ(meta.num_rows(), num_rows);

  std::string expected_schema = R"(schema
 int_col
 float_col
)";
  EXPECT_EQ(expected_schema, print(meta.schema().root()));

  EXPECT_EQ(meta.schema().root().name(), "schema");
  EXPECT_EQ(meta.schema().root().type_kind(), cudf::io::parquet::TypeKind::UNDEFINED_TYPE);
  ASSERT_EQ(meta.schema().root().num_children(), 2);

  EXPECT_EQ(meta.schema().root().child(0).name(), "int_col");
  EXPECT_EQ(meta.schema().root().child(1).name(), "float_col");
}

TEST_F(ParquetMetadataReaderTest, TestNested)
{
  auto const num_rows       = 1200;
  auto const lists_per_row  = 4;
  auto const num_child_rows = num_rows * lists_per_row;

  auto keys = random_values<int>(num_child_rows);
  auto vals = random_values<float>(num_child_rows);
  column_wrapper<int> keys_col(keys.begin(), keys.end());
  column_wrapper<float> vals_col(vals.begin(), vals.end());
  auto s_col = cudf::test::structs_column_wrapper({keys_col, vals_col}).release();

  std::vector<int> row_offsets(num_rows + 1);
  for (int idx = 0; idx < num_rows + 1; ++idx) {
    row_offsets[idx] = idx * lists_per_row;
  }
  column_wrapper<int> offsets(row_offsets.begin(), row_offsets.end());

  auto list_col =
    cudf::make_lists_column(num_rows, offsets.release(), std::move(s_col), 0, rmm::device_buffer{});

  table_view expected({*list_col, *list_col});

  cudf::io::table_input_metadata expected_metadata(expected);
  expected_metadata.column_metadata[0].set_name("maps");
  expected_metadata.column_metadata[0].set_list_column_as_map();
  expected_metadata.column_metadata[1].set_name("lists");
  expected_metadata.column_metadata[1].child(1).child(0).set_name("int_field");
  expected_metadata.column_metadata[1].child(1).child(1).set_name("float_field");

  auto filepath = temp_env->get_temp_filepath("MetadataTest.orc");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .metadata(std::move(expected_metadata));
  cudf::io::write_parquet(out_opts);

  auto meta = read_parquet_metadata(cudf::io::source_info{filepath});
  EXPECT_EQ(meta.num_rows(), num_rows);

  std::string expected_schema = R"(schema
 maps
  key_value
   key
   value
 lists
  list
   element
    int_field
    float_field
)";
  EXPECT_EQ(expected_schema, print(meta.schema().root()));

  EXPECT_EQ(meta.schema().root().name(), "schema");
  EXPECT_EQ(meta.schema().root().type_kind(),
            cudf::io::parquet::TypeKind::UNDEFINED_TYPE);  // struct
  ASSERT_EQ(meta.schema().root().num_children(), 2);

  auto const& out_map_col = meta.schema().root().child(0);
  EXPECT_EQ(out_map_col.name(), "maps");
  EXPECT_EQ(out_map_col.type_kind(), cudf::io::parquet::TypeKind::UNDEFINED_TYPE);  // map

  ASSERT_EQ(out_map_col.num_children(), 1);
  EXPECT_EQ(out_map_col.child(0).name(), "key_value");  // key_value (named in parquet writer)
  ASSERT_EQ(out_map_col.child(0).num_children(), 2);
  EXPECT_EQ(out_map_col.child(0).child(0).name(), "key");    // key (named in parquet writer)
  EXPECT_EQ(out_map_col.child(0).child(1).name(), "value");  // value (named in parquet writer)
  EXPECT_EQ(out_map_col.child(0).child(0).type_kind(), cudf::io::parquet::TypeKind::INT32);  // int
  EXPECT_EQ(out_map_col.child(0).child(1).type_kind(),
            cudf::io::parquet::TypeKind::FLOAT);  // float

  auto const& out_list_col = meta.schema().root().child(1);
  EXPECT_EQ(out_list_col.name(), "lists");
  EXPECT_EQ(out_list_col.type_kind(), cudf::io::parquet::TypeKind::UNDEFINED_TYPE);  // list
  // TODO repetition type?
  ASSERT_EQ(out_list_col.num_children(), 1);
  EXPECT_EQ(out_list_col.child(0).name(), "list");  // list (named in parquet writer)
  ASSERT_EQ(out_list_col.child(0).num_children(), 1);

  auto const& out_list_struct_col = out_list_col.child(0).child(0);
  EXPECT_EQ(out_list_struct_col.name(), "element");  // elements (named in parquet writer)
  EXPECT_EQ(out_list_struct_col.type_kind(),
            cudf::io::parquet::TypeKind::UNDEFINED_TYPE);  // struct
  ASSERT_EQ(out_list_struct_col.num_children(), 2);

  auto const& out_int_col = out_list_struct_col.child(0);
  EXPECT_EQ(out_int_col.name(), "int_field");
  EXPECT_EQ(out_int_col.type_kind(), cudf::io::parquet::TypeKind::INT32);

  auto const& out_float_col = out_list_struct_col.child(1);
  EXPECT_EQ(out_float_col.name(), "float_field");
  EXPECT_EQ(out_float_col.type_kind(), cudf::io::parquet::TypeKind::FLOAT);
}

///////////////////////
// reader source tests

template <typename T>
struct ParquetReaderSourceTest : public ParquetReaderTest {};

TYPED_TEST_SUITE(ParquetReaderSourceTest, ByteLikeTypes);

TYPED_TEST(ParquetReaderSourceTest, BufferSourceTypes)
{
  using T = TypeParam;

  srand(31337);
  auto table = create_random_fixed_table<int>(5, 5, true);

  std::vector<char> out_buffer;
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info(&out_buffer), *table);
  cudf::io::write_parquet(out_opts);

  {
    cudf::io::parquet_reader_options in_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info(
        cudf::host_span<T>(reinterpret_cast<T*>(out_buffer.data()), out_buffer.size())));
    auto const result = cudf::io::read_parquet(in_opts);

    CUDF_TEST_EXPECT_TABLES_EQUAL(*table, result.tbl->view());
  }

  {
    cudf::io::parquet_reader_options in_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info(cudf::host_span<T const>(
        reinterpret_cast<T const*>(out_buffer.data()), out_buffer.size())));
    auto const result = cudf::io::read_parquet(in_opts);

    CUDF_TEST_EXPECT_TABLES_EQUAL(*table, result.tbl->view());
  }
}

TYPED_TEST(ParquetReaderSourceTest, BufferSourceArrayTypes)
{
  using T = TypeParam;

  srand(31337);
  auto table = create_random_fixed_table<int>(5, 5, true);

  std::vector<char> out_buffer;
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info(&out_buffer), *table);
  cudf::io::write_parquet(out_opts);

  auto full_table = cudf::concatenate(std::vector<table_view>({*table, *table}));

  {
    auto spans = std::vector<cudf::host_span<T>>{
      cudf::host_span<T>(reinterpret_cast<T*>(out_buffer.data()), out_buffer.size()),
      cudf::host_span<T>(reinterpret_cast<T*>(out_buffer.data()), out_buffer.size())};
    cudf::io::parquet_reader_options in_opts = cudf::io::parquet_reader_options::builder(
      cudf::io::source_info(cudf::host_span<cudf::host_span<T>>(spans.data(), spans.size())));
    auto const result = cudf::io::read_parquet(in_opts);

    CUDF_TEST_EXPECT_TABLES_EQUAL(*full_table, result.tbl->view());
  }

  {
    auto spans = std::vector<cudf::host_span<T const>>{
      cudf::host_span<T const>(reinterpret_cast<T const*>(out_buffer.data()), out_buffer.size()),
      cudf::host_span<T const>(reinterpret_cast<T const*>(out_buffer.data()), out_buffer.size())};
    cudf::io::parquet_reader_options in_opts = cudf::io::parquet_reader_options::builder(
      cudf::io::source_info(cudf::host_span<cudf::host_span<T const>>(spans.data(), spans.size())));
    auto const result = cudf::io::read_parquet(in_opts);

    CUDF_TEST_EXPECT_TABLES_EQUAL(*full_table, result.tbl->view());
  }
}

//////////////////////////////
// predicate pushdown tests

// Test for Types - numeric, chrono, string.
template <typename T>
struct ParquetReaderPredicatePushdownTest : public ParquetReaderTest {};

TYPED_TEST_SUITE(ParquetReaderPredicatePushdownTest, SupportedTestTypes);

TYPED_TEST(ParquetReaderPredicatePushdownTest, FilterTyped)
{
  using T = TypeParam;

  auto const [src, filepath] = create_parquet_typed_with_stats<T>("FilterTyped.parquet");
  auto const written_table   = src.view();

  // Filtering AST
  auto literal_value = []() {
    if constexpr (cudf::is_timestamp<T>()) {
      // table[0] < 10000 timestamp days/seconds/milliseconds/microseconds/nanoseconds
      return cudf::timestamp_scalar<T>(T(typename T::duration(10000)));  // i (0-20,000)
    } else if constexpr (cudf::is_duration<T>()) {
      // table[0] < 10000 day/seconds/milliseconds/microseconds/nanoseconds
      return cudf::duration_scalar<T>(T(10000));  // i (0-20,000)
    } else if constexpr (std::is_same_v<T, cudf::string_view>) {
      // table[0] < "000010000"
      return cudf::string_scalar("000010000");  // i (0-20,000)
    } else {
      // table[0] < 0 or 100u
      return cudf::numeric_scalar<T>((100 - 100 * std::is_signed_v<T>));  // i/100 (-100-100/ 0-200)
    }
  }();
  auto literal           = cudf::ast::literal(literal_value);
  auto col_name_0        = cudf::ast::column_name_reference("col0");
  auto filter_expression = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_name_0, literal);
  auto col_ref_0         = cudf::ast::column_reference(0);
  auto ref_filter        = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, literal);

  // Expected result
  auto predicate = cudf::compute_column(written_table, ref_filter);
  EXPECT_EQ(predicate->view().type().id(), cudf::type_id::BOOL8)
    << "Predicate filter should return a boolean";
  auto expected = cudf::apply_boolean_mask(written_table, *predicate);

  // Reading with Predicate Pushdown
  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
      .filter(filter_expression);
  auto result       = cudf::io::read_parquet(read_opts);
  auto result_table = result.tbl->view();

  // tests
  EXPECT_EQ(int(written_table.column(0).type().id()), int(result_table.column(0).type().id()))
    << "col0 type mismatch";
  // To make sure AST filters out some elements
  EXPECT_LT(expected->num_rows(), written_table.num_rows());
  EXPECT_EQ(result_table.num_rows(), expected->num_rows());
  EXPECT_EQ(result_table.num_columns(), expected->num_columns());
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), result_table);
}

// The test below requires several minutes to complete with memcheck, thus it is disabled by
// default.
TEST_F(ParquetReaderTest, DISABLED_ListsWideTable)
{
  auto constexpr num_rows = 2;
  auto constexpr num_cols = 26'755;  // for slightly over 2B keys
  auto constexpr seed     = 0xceed;

  std::mt19937 engine{seed};

  auto list_list       = make_parquet_list_list_col<int32_t>(0, num_rows, 1, 1, false);
  auto list_list_nulls = make_parquet_list_list_col<int32_t>(0, num_rows, 1, 1, true);

  // switch between nullable and non-nullable
  std::vector<cudf::column_view> cols(num_cols);
  bool with_nulls = false;
  std::generate_n(cols.begin(), num_cols, [&]() {
    auto const view = with_nulls ? list_list_nulls->view() : list_list->view();
    with_nulls      = not with_nulls;
    return view;
  });

  cudf::table_view expected(cols);

  // Use a host buffer for faster I/O
  std::vector<char> buffer;
  auto const out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{&buffer}, expected).build();
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options default_in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info(buffer.data(), buffer.size()));
  auto const [result, _] = cudf::io::read_parquet(default_in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result->view());
}
