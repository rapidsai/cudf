/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/io_metadata_utilities.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/io/data_sink.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_metadata.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <src/io/parquet/compact_protocol_reader.hpp>
#include <src/io/parquet/parquet.hpp>
#include <src/io/parquet/parquet_gpu.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/iterator/counting_iterator.h>

#include <fstream>
#include <random>
#include <type_traits>

// Global environment for temporary files
cudf::test::TempDirTestEnvironment* const temp_env =
  static_cast<cudf::test::TempDirTestEnvironment*>(
    ::testing::AddGlobalTestEnvironment(new cudf::test::TempDirTestEnvironment));

// Declare typed test cases
TYPED_TEST_SUITE(ParquetWriterNumericTypeTest, SupportedTypes);
TYPED_TEST_SUITE(ParquetWriterComparableTypeTest, ComparableAndFixedTypes);
TYPED_TEST_SUITE(ParquetWriterChronoTypeTest, cudf::test::ChronoTypes);
TYPED_TEST_SUITE(ParquetWriterTimestampTypeTest, SupportedTimestampTypes);
TYPED_TEST_SUITE(ParquetWriterSchemaTest, cudf::test::AllTypes);
TYPED_TEST_SUITE(ParquetReaderSourceTest, ByteLikeTypes);

// Declare typed test cases
TYPED_TEST_SUITE(ParquetChunkedWriterNumericTypeTest, SupportedTypes);

INSTANTIATE_TEST_SUITE_P(ParquetV2ReadWriteTest,
                         ParquetV2Test,
                         testing::Bool(),
                         testing::PrintToStringParamName());

TYPED_TEST(ParquetWriterNumericTypeTest, SingleColumn)
{
  auto sequence =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return TypeParam(i % 400); });
  auto validity = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return true; });

  constexpr auto num_rows = 800;
  column_wrapper<TypeParam> col(sequence, sequence + num_rows, validity);

  auto expected = table_view{{col}};

  auto filepath = temp_env->get_temp_filepath("SingleColumn.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected);
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
}

TYPED_TEST(ParquetWriterNumericTypeTest, SingleColumnWithNulls)
{
  auto sequence =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return TypeParam(i); });
  auto validity = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return (i % 2); });

  constexpr auto num_rows = 100;
  column_wrapper<TypeParam> col(sequence, sequence + num_rows, validity);

  auto expected = table_view{{col}};

  auto filepath = temp_env->get_temp_filepath("SingleColumnWithNulls.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected);
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
}

TYPED_TEST(ParquetWriterTimestampTypeTest, Timestamps)
{
  auto sequence = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return ((std::rand() / 10000) * 1000); });
  auto validity = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return true; });

  constexpr auto num_rows = 100;
  column_wrapper<TypeParam, typename decltype(sequence)::value_type> col(
    sequence, sequence + num_rows, validity);

  auto expected = table_view{{col}};

  auto filepath = temp_env->get_temp_filepath("Timestamps.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected);
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
      .timestamp_type(this->type());
  auto result = cudf::io::read_parquet(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
}

TYPED_TEST(ParquetWriterTimestampTypeTest, TimestampsWithNulls)
{
  auto sequence = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return ((std::rand() / 10000) * 1000); });
  auto validity =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return (i > 30) && (i < 60); });

  constexpr auto num_rows = 100;
  column_wrapper<TypeParam, typename decltype(sequence)::value_type> col(
    sequence, sequence + num_rows, validity);

  auto expected = table_view{{col}};

  auto filepath = temp_env->get_temp_filepath("TimestampsWithNulls.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected);
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
      .timestamp_type(this->type());
  auto result = cudf::io::read_parquet(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
}

TYPED_TEST(ParquetWriterTimestampTypeTest, TimestampOverflow)
{
  constexpr int64_t max = std::numeric_limits<int64_t>::max();
  auto sequence = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return max - i; });
  auto validity = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return true; });

  constexpr auto num_rows = 100;
  column_wrapper<TypeParam, typename decltype(sequence)::value_type> col(
    sequence, sequence + num_rows, validity);
  table_view expected({col});

  auto filepath = temp_env->get_temp_filepath("ParquetTimestampOverflow.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected);
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
      .timestamp_type(this->type());
  auto result = cudf::io::read_parquet(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
}

template <typename T>
std::string create_parquet_file(int num_cols)
{
  srand(31337);
  auto const table = create_random_fixed_table<T>(num_cols, 10, true);
  auto const filepath =
    temp_env->get_temp_filepath(typeid(T).name() + std::to_string(num_cols) + ".parquet");
  cudf::io::parquet_writer_options const out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, table->view());
  cudf::io::write_parquet(out_opts);
  return filepath;
}

TEST_F(ParquetChunkedWriterTest, SingleTable)
{
  srand(31337);
  auto table1 = create_random_fixed_table<int>(5, 5, true);

  auto filepath = temp_env->get_temp_filepath("ChunkedSingle.parquet");
  cudf::io::chunked_parquet_writer_options args =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath});
  cudf::io::parquet_chunked_writer(args).write(*table1);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *table1);
}

TEST_F(ParquetChunkedWriterTest, SimpleTable)
{
  srand(31337);
  auto table1 = create_random_fixed_table<int>(5, 5, true);
  auto table2 = create_random_fixed_table<int>(5, 5, true);

  auto full_table = cudf::concatenate(std::vector<table_view>({*table1, *table2}));

  auto filepath = temp_env->get_temp_filepath("ChunkedSimple.parquet");
  cudf::io::chunked_parquet_writer_options args =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath});
  cudf::io::parquet_chunked_writer(args).write(*table1).write(*table2);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *full_table);
}

TEST_F(ParquetChunkedWriterTest, LargeTables)
{
  srand(31337);
  auto table1 = create_random_fixed_table<int>(512, 4096, true);
  auto table2 = create_random_fixed_table<int>(512, 8192, true);

  auto full_table = cudf::concatenate(std::vector<table_view>({*table1, *table2}));

  auto filepath = temp_env->get_temp_filepath("ChunkedLarge.parquet");
  cudf::io::chunked_parquet_writer_options args =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath});
  auto md = cudf::io::parquet_chunked_writer(args).write(*table1).write(*table2).close();
  ASSERT_EQ(md, nullptr);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *full_table);
}

TEST_F(ParquetChunkedWriterTest, ManyTables)
{
  srand(31337);
  std::vector<std::unique_ptr<table>> tables;
  std::vector<table_view> table_views;
  constexpr int num_tables = 96;
  for (int idx = 0; idx < num_tables; idx++) {
    auto tbl = create_random_fixed_table<int>(16, 64, true);
    table_views.push_back(*tbl);
    tables.push_back(std::move(tbl));
  }

  auto expected = cudf::concatenate(table_views);

  auto filepath = temp_env->get_temp_filepath("ChunkedManyTables.parquet");
  cudf::io::chunked_parquet_writer_options args =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath});
  cudf::io::parquet_chunked_writer writer(args);
  std::for_each(table_views.begin(), table_views.end(), [&writer](table_view const& tbl) {
    writer.write(tbl);
  });
  auto md = writer.close({"dummy/path"});
  ASSERT_NE(md, nullptr);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *expected);
}

TEST_F(ParquetChunkedWriterTest, Strings)
{
  std::vector<std::unique_ptr<cudf::column>> cols;

  bool mask1[] = {true, true, false, true, true, true, true};
  std::vector<char const*> h_strings1{"four", "score", "and", "seven", "years", "ago", "abcdefgh"};
  cudf::test::strings_column_wrapper strings1(h_strings1.begin(), h_strings1.end(), mask1);
  cols.push_back(strings1.release());
  cudf::table tbl1(std::move(cols));

  bool mask2[] = {false, true, true, true, true, true, true};
  std::vector<char const*> h_strings2{"ooooo", "ppppppp", "fff", "j", "cccc", "bbb", "zzzzzzzzzzz"};
  cudf::test::strings_column_wrapper strings2(h_strings2.begin(), h_strings2.end(), mask2);
  cols.push_back(strings2.release());
  cudf::table tbl2(std::move(cols));

  auto expected = cudf::concatenate(std::vector<table_view>({tbl1, tbl2}));

  auto filepath = temp_env->get_temp_filepath("ChunkedStrings.parquet");
  cudf::io::chunked_parquet_writer_options args =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath});
  cudf::io::parquet_chunked_writer(args).write(tbl1).write(tbl2);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *expected);
}

TEST_F(ParquetChunkedWriterTest, ListColumn)
{
  auto valids  = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 2; });
  auto valids2 = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 3; });

  using lcw = cudf::test::lists_column_wrapper<int32_t>;

  // COL0 (Same nullability) ====================
  // [NULL, 2, NULL]
  // []
  // [4, 5]
  // NULL
  lcw col0_tbl0{{{{1, 2, 3}, valids}, {}, {4, 5}, {}}, valids2};

  // [7, 8, 9]
  // []
  // [NULL, 11]
  // NULL
  lcw col0_tbl1{{{7, 8, 9}, {}, {{10, 11}, valids}, {}}, valids2};

  // COL1 (Nullability different in different chunks, test of merging nullability in writer)
  // [NULL, 2, NULL]
  // []
  // [4, 5]
  // []
  lcw col1_tbl0{{{1, 2, 3}, valids}, {}, {4, 5}, {}};

  // [7, 8, 9]
  // []
  // [10, 11]
  // NULL
  lcw col1_tbl1{{{7, 8, 9}, {}, {10, 11}, {}}, valids2};

  // COL2 (non-nested columns to test proper schema construction)
  size_t num_rows_tbl0 = static_cast<cudf::column_view>(col0_tbl0).size();
  size_t num_rows_tbl1 = static_cast<cudf::column_view>(col0_tbl1).size();
  auto seq_col0        = random_values<int>(num_rows_tbl0);
  auto seq_col1        = random_values<int>(num_rows_tbl1);

  column_wrapper<int> col2_tbl0{seq_col0.begin(), seq_col0.end(), valids};
  column_wrapper<int> col2_tbl1{seq_col1.begin(), seq_col1.end(), valids2};

  auto tbl0 = table_view({col0_tbl0, col1_tbl0, col2_tbl0});
  auto tbl1 = table_view({col0_tbl1, col1_tbl1, col2_tbl1});

  auto expected = cudf::concatenate(std::vector<table_view>({tbl0, tbl1}));

  auto filepath = temp_env->get_temp_filepath("ChunkedLists.parquet");
  cudf::io::chunked_parquet_writer_options args =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath});
  cudf::io::parquet_chunked_writer(args).write(tbl0).write(tbl1);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *expected);
}

TEST_F(ParquetChunkedWriterTest, ListOfStruct)
{
  // Table 1
  auto weight_1   = cudf::test::fixed_width_column_wrapper<float>{{57.5, 51.1, 15.3}};
  auto ages_1     = cudf::test::fixed_width_column_wrapper<int32_t>{{30, 27, 5}};
  auto struct_1_1 = cudf::test::structs_column_wrapper{weight_1, ages_1};
  auto is_human_1 = cudf::test::fixed_width_column_wrapper<bool>{{true, true, false}};
  auto struct_2_1 = cudf::test::structs_column_wrapper{{is_human_1, struct_1_1}};

  auto list_offsets_column_1 =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 2, 3, 3}.release();
  auto num_list_rows_1 = list_offsets_column_1->size() - 1;

  auto list_col_1 = cudf::make_lists_column(
    num_list_rows_1, std::move(list_offsets_column_1), struct_2_1.release(), 0, {});

  auto table_1 = table_view({*list_col_1});

  // Table 2
  auto weight_2   = cudf::test::fixed_width_column_wrapper<float>{{1.1, -1.0, -1.0}};
  auto ages_2     = cudf::test::fixed_width_column_wrapper<int32_t>{{31, 351, 351}, {1, 1, 0}};
  auto struct_1_2 = cudf::test::structs_column_wrapper{{weight_2, ages_2}, {1, 0, 1}};
  auto is_human_2 = cudf::test::fixed_width_column_wrapper<bool>{{false, false, false}, {1, 1, 0}};
  auto struct_2_2 = cudf::test::structs_column_wrapper{{is_human_2, struct_1_2}};

  auto list_offsets_column_2 =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 1, 2, 3}.release();
  auto num_list_rows_2 = list_offsets_column_2->size() - 1;

  auto list_col_2 = cudf::make_lists_column(
    num_list_rows_2, std::move(list_offsets_column_2), struct_2_2.release(), 0, {});

  auto table_2 = table_view({*list_col_2});

  auto full_table = cudf::concatenate(std::vector<table_view>({table_1, table_2}));

  cudf::io::table_input_metadata expected_metadata(table_1);
  expected_metadata.column_metadata[0].set_name("family");
  expected_metadata.column_metadata[0].child(1).set_nullability(false);
  expected_metadata.column_metadata[0].child(1).child(0).set_name("human?");
  expected_metadata.column_metadata[0].child(1).child(1).set_name("particulars");
  expected_metadata.column_metadata[0].child(1).child(1).child(0).set_name("weight");
  expected_metadata.column_metadata[0].child(1).child(1).child(1).set_name("age");

  auto filepath = temp_env->get_temp_filepath("ChunkedListOfStruct.parquet");
  cudf::io::chunked_parquet_writer_options args =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath});
  args.set_metadata(expected_metadata);
  cudf::io::parquet_chunked_writer(args).write(table_1).write(table_2);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*result.tbl, *full_table);
  cudf::test::expect_metadata_equal(expected_metadata, result.metadata);
}

TEST_F(ParquetChunkedWriterTest, ListOfStructOfStructOfListOfList)
{
  auto valids  = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 2; });
  auto valids2 = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 3; });

  using lcw = cudf::test::lists_column_wrapper<int32_t>;

  // Table 1 ===========================

  // []
  // [NULL, 2, NULL]
  // [4, 5]
  // NULL
  lcw land_1{{{}, {{1, 2, 3}, valids}, {4, 5}, {}}, valids2};

  // []
  // [[1, 2, 3], [], [4, 5], [], [0, 6, 0]]
  // [[7, 8], []]
  // [[]]
  lcw flats_1{lcw{}, {{1, 2, 3}, {}, {4, 5}, {}, {0, 6, 0}}, {{7, 8}, {}}, lcw{lcw{}}};

  auto weight_1   = cudf::test::fixed_width_column_wrapper<float>{{57.5, 51.1, 15.3, 1.1}};
  auto ages_1     = cudf::test::fixed_width_column_wrapper<int32_t>{{30, 27, 5, 31}};
  auto struct_1_1 = cudf::test::structs_column_wrapper{weight_1, ages_1, land_1, flats_1};
  auto is_human_1 = cudf::test::fixed_width_column_wrapper<bool>{{true, true, false, false}};
  auto struct_2_1 = cudf::test::structs_column_wrapper{{is_human_1, struct_1_1}};

  auto list_offsets_column_1 =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 2, 3, 4}.release();
  auto num_list_rows_1 = list_offsets_column_1->size() - 1;

  auto list_col_1 = cudf::make_lists_column(
    num_list_rows_1, std::move(list_offsets_column_1), struct_2_1.release(), 0, {});

  auto table_1 = table_view({*list_col_1});

  // Table 2 ===========================

  // []
  // [7, 8, 9]
  lcw land_2{{}, {7, 8, 9}};

  // [[]]
  // [[], [], []]
  lcw flats_2{lcw{lcw{}}, lcw{lcw{}, lcw{}, lcw{}}};

  auto weight_2   = cudf::test::fixed_width_column_wrapper<float>{{-1.0, -1.0}};
  auto ages_2     = cudf::test::fixed_width_column_wrapper<int32_t>{{351, 351}, {1, 0}};
  auto struct_1_2 = cudf::test::structs_column_wrapper{{weight_2, ages_2, land_2, flats_2}, {0, 1}};
  auto is_human_2 = cudf::test::fixed_width_column_wrapper<bool>{{false, false}, {1, 0}};
  auto struct_2_2 = cudf::test::structs_column_wrapper{{is_human_2, struct_1_2}};

  auto list_offsets_column_2 =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 1, 2}.release();
  auto num_list_rows_2 = list_offsets_column_2->size() - 1;

  auto list_col_2 = cudf::make_lists_column(
    num_list_rows_2, std::move(list_offsets_column_2), struct_2_2.release(), 0, {});

  auto table_2 = table_view({*list_col_2});

  auto full_table = cudf::concatenate(std::vector<table_view>({table_1, table_2}));

  cudf::io::table_input_metadata expected_metadata(table_1);
  expected_metadata.column_metadata[0].set_name("family");
  expected_metadata.column_metadata[0].child(1).set_nullability(false);
  expected_metadata.column_metadata[0].child(1).child(0).set_name("human?");
  expected_metadata.column_metadata[0].child(1).child(1).set_name("particulars");
  expected_metadata.column_metadata[0].child(1).child(1).child(0).set_name("weight");
  expected_metadata.column_metadata[0].child(1).child(1).child(1).set_name("age");
  expected_metadata.column_metadata[0].child(1).child(1).child(2).set_name("land_unit");
  expected_metadata.column_metadata[0].child(1).child(1).child(3).set_name("flats");

  auto filepath = temp_env->get_temp_filepath("ListOfStructOfStructOfListOfList.parquet");
  cudf::io::chunked_parquet_writer_options args =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath});
  args.set_metadata(expected_metadata);
  cudf::io::parquet_chunked_writer(args).write(table_1).write(table_2);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*result.tbl, *full_table);
  cudf::test::expect_metadata_equal(expected_metadata, result.metadata);

  // We specifically mentioned in input schema that struct_2 is non-nullable across chunked calls.
  auto result_parent_list = result.tbl->get_column(0);
  auto result_struct_2    = result_parent_list.child(cudf::lists_column_view::child_column_index);
  EXPECT_EQ(result_struct_2.nullable(), false);
}

TEST_F(ParquetChunkedWriterTest, MismatchedTypes)
{
  srand(31337);
  auto table1 = create_random_fixed_table<int>(4, 4, true);
  auto table2 = create_random_fixed_table<float>(4, 4, true);

  auto filepath = temp_env->get_temp_filepath("ChunkedMismatchedTypes.parquet");
  cudf::io::chunked_parquet_writer_options args =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath});
  cudf::io::parquet_chunked_writer writer(args);
  writer.write(*table1);
  EXPECT_THROW(writer.write(*table2), cudf::logic_error);
  writer.close();
}

TEST_F(ParquetChunkedWriterTest, ChunkedWriteAfterClosing)
{
  srand(31337);
  auto table = create_random_fixed_table<int>(4, 4, true);

  auto filepath = temp_env->get_temp_filepath("ChunkedWriteAfterClosing.parquet");
  cudf::io::chunked_parquet_writer_options args =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath});
  cudf::io::parquet_chunked_writer writer(args);
  writer.write(*table).close();
  EXPECT_THROW(writer.write(*table), cudf::logic_error);
}

TEST_F(ParquetChunkedWriterTest, ReadingUnclosedFile)
{
  srand(31337);
  auto table = create_random_fixed_table<int>(4, 4, true);

  auto filepath = temp_env->get_temp_filepath("ReadingUnclosedFile.parquet");
  cudf::io::chunked_parquet_writer_options args =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath});
  cudf::io::parquet_chunked_writer writer(args);
  writer.write(*table);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  EXPECT_THROW(cudf::io::read_parquet(read_opts), cudf::logic_error);
}

TEST_F(ParquetChunkedWriterTest, MismatchedStructure)
{
  srand(31337);
  auto table1 = create_random_fixed_table<int>(4, 4, true);
  auto table2 = create_random_fixed_table<float>(3, 4, true);

  auto filepath = temp_env->get_temp_filepath("ChunkedMismatchedStructure.parquet");
  cudf::io::chunked_parquet_writer_options args =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath});
  cudf::io::parquet_chunked_writer writer(args);
  writer.write(*table1);
  EXPECT_THROW(writer.write(*table2), cudf::logic_error);
  writer.close();
}

TEST_F(ParquetChunkedWriterTest, MismatchedStructureList)
{
  auto valids  = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 2; });
  auto valids2 = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 3; });

  using lcw = cudf::test::lists_column_wrapper<int32_t>;

  // COL0 (mismatched depth) ====================
  // [NULL, 2, NULL]
  // []
  // [4, 5]
  // NULL
  lcw col00{{{{1, 2, 3}, valids}, {}, {4, 5}, {}}, valids2};

  // [[1, 2, 3], [], [4, 5], [], [0, 6, 0]]
  // [[7, 8]]
  // []
  // [[]]
  lcw col01{{{1, 2, 3}, {}, {4, 5}, {}, {0, 6, 0}}, {{7, 8}}, lcw{}, lcw{lcw{}}};

  // COL2 (non-nested columns to test proper schema construction)
  size_t num_rows = static_cast<cudf::column_view>(col00).size();
  auto seq_col0   = random_values<int>(num_rows);
  auto seq_col1   = random_values<int>(num_rows);

  column_wrapper<int> col10{seq_col0.begin(), seq_col0.end(), valids};
  column_wrapper<int> col11{seq_col1.begin(), seq_col1.end(), valids2};

  auto tbl0 = table_view({col00, col10});
  auto tbl1 = table_view({col01, col11});

  auto filepath = temp_env->get_temp_filepath("ChunkedLists.parquet");
  cudf::io::chunked_parquet_writer_options args =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath});
  cudf::io::parquet_chunked_writer writer(args);
  writer.write(tbl0);
  EXPECT_THROW(writer.write(tbl1), cudf::logic_error);
}

TEST_F(ParquetChunkedWriterTest, DifferentNullability)
{
  srand(31337);
  auto table1 = create_random_fixed_table<int>(5, 5, true);
  auto table2 = create_random_fixed_table<int>(5, 5, false);

  auto full_table = cudf::concatenate(std::vector<table_view>({*table1, *table2}));

  auto filepath = temp_env->get_temp_filepath("ChunkedNullable.parquet");
  cudf::io::chunked_parquet_writer_options args =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath});
  cudf::io::parquet_chunked_writer(args).write(*table1).write(*table2);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *full_table);
}

TEST_F(ParquetChunkedWriterTest, DifferentNullabilityStruct)
{
  // Struct<is_human:bool (non-nullable),
  //        Struct<weight:float>,
  //               age:int
  //              > (nullable)
  //       > (non-nullable)

  // Table 1: is_human and struct_1 are non-nullable but should be nullable when read back.
  auto weight_1   = cudf::test::fixed_width_column_wrapper<float>{{57.5, 51.1, 15.3}};
  auto ages_1     = cudf::test::fixed_width_column_wrapper<int32_t>{{30, 27, 5}};
  auto struct_1_1 = cudf::test::structs_column_wrapper{weight_1, ages_1};
  auto is_human_1 = cudf::test::fixed_width_column_wrapper<bool>{{true, true, false}};
  auto struct_2_1 = cudf::test::structs_column_wrapper{{is_human_1, struct_1_1}};
  auto table_1    = cudf::table_view({struct_2_1});

  // Table 2: struct_1 and is_human are nullable now so if we hadn't assumed worst case (nullable)
  // when writing table_1, we would have wrong pages for it.
  auto weight_2   = cudf::test::fixed_width_column_wrapper<float>{{1.1, -1.0, -1.0}};
  auto ages_2     = cudf::test::fixed_width_column_wrapper<int32_t>{{31, 351, 351}, {1, 1, 0}};
  auto struct_1_2 = cudf::test::structs_column_wrapper{{weight_2, ages_2}, {1, 0, 1}};
  auto is_human_2 = cudf::test::fixed_width_column_wrapper<bool>{{false, false, false}, {1, 1, 0}};
  auto struct_2_2 = cudf::test::structs_column_wrapper{{is_human_2, struct_1_2}};
  auto table_2    = cudf::table_view({struct_2_2});

  auto full_table = cudf::concatenate(std::vector<table_view>({table_1, table_2}));

  cudf::io::table_input_metadata expected_metadata(table_1);
  expected_metadata.column_metadata[0].set_name("being");
  expected_metadata.column_metadata[0].child(0).set_name("human?");
  expected_metadata.column_metadata[0].child(1).set_name("particulars");
  expected_metadata.column_metadata[0].child(1).child(0).set_name("weight");
  expected_metadata.column_metadata[0].child(1).child(1).set_name("age");

  auto filepath = temp_env->get_temp_filepath("ChunkedNullableStruct.parquet");
  cudf::io::chunked_parquet_writer_options args =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath});
  args.set_metadata(expected_metadata);
  cudf::io::parquet_chunked_writer(args).write(table_1).write(table_2);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*result.tbl, *full_table);
  cudf::test::expect_metadata_equal(expected_metadata, result.metadata);
}

TEST_F(ParquetChunkedWriterTest, ForcedNullability)
{
  srand(31337);
  auto table1 = create_random_fixed_table<int>(5, 5, false);
  auto table2 = create_random_fixed_table<int>(5, 5, false);

  auto full_table = cudf::concatenate(std::vector<table_view>({*table1, *table2}));

  auto filepath = temp_env->get_temp_filepath("ChunkedNoNullable.parquet");

  cudf::io::table_input_metadata metadata(*table1);

  // In the absence of prescribed per-column nullability in metadata, the writer assumes the worst
  // and considers all columns nullable. However cudf::concatenate will not force nulls in case no
  // columns are nullable. To get the expected result, we tell the writer the nullability of all
  // columns in advance.
  for (auto& col_meta : metadata.column_metadata) {
    col_meta.set_nullability(false);
  }

  cudf::io::chunked_parquet_writer_options args =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath})
      .metadata(std::move(metadata));
  cudf::io::parquet_chunked_writer(args).write(*table1).write(*table2);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *full_table);
}

TEST_F(ParquetChunkedWriterTest, ForcedNullabilityList)
{
  srand(31337);

  auto valids  = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 2; });
  auto valids2 = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 3; });

  using lcw = cudf::test::lists_column_wrapper<int32_t>;

  // COL0 ====================
  // [1, 2, 3]
  // []
  // [4, 5]
  // NULL
  lcw col00{{{1, 2, 3}, {}, {4, 5}, {}}, valids2};

  // [7]
  // []
  // [8, 9, 10, 11]
  // NULL
  lcw col01{{{7}, {}, {8, 9, 10, 11}, {}}, valids2};

  // COL1 (non-nested columns to test proper schema construction)
  size_t num_rows = static_cast<cudf::column_view>(col00).size();
  auto seq_col0   = random_values<int>(num_rows);
  auto seq_col1   = random_values<int>(num_rows);

  column_wrapper<int> col10{seq_col0.begin(), seq_col0.end(), valids};
  column_wrapper<int> col11{seq_col1.begin(), seq_col1.end(), valids2};

  auto table1 = table_view({col00, col10});
  auto table2 = table_view({col01, col11});

  auto full_table = cudf::concatenate(std::vector<table_view>({table1, table2}));

  cudf::io::table_input_metadata metadata(table1);
  metadata.column_metadata[0].set_nullability(true);  // List is nullable at first (root) level
  metadata.column_metadata[0].child(1).set_nullability(
    false);  // non-nullable at second (leaf) level
  metadata.column_metadata[1].set_nullability(true);

  auto filepath = temp_env->get_temp_filepath("ChunkedListNullable.parquet");

  cudf::io::chunked_parquet_writer_options args =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath})
      .metadata(std::move(metadata));
  cudf::io::parquet_chunked_writer(args).write(table1).write(table2);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *full_table);
}

TEST_F(ParquetChunkedWriterTest, ForcedNullabilityStruct)
{
  // Struct<is_human:bool (non-nullable),
  //        Struct<weight:float>,
  //               age:int
  //              > (nullable)
  //       > (non-nullable)

  // Table 1: is_human and struct_2 are non-nullable and should stay that way when read back.
  auto weight_1   = cudf::test::fixed_width_column_wrapper<float>{{57.5, 51.1, 15.3}};
  auto ages_1     = cudf::test::fixed_width_column_wrapper<int32_t>{{30, 27, 5}};
  auto struct_1_1 = cudf::test::structs_column_wrapper{weight_1, ages_1};
  auto is_human_1 = cudf::test::fixed_width_column_wrapper<bool>{{true, true, false}};
  auto struct_2_1 = cudf::test::structs_column_wrapper{{is_human_1, struct_1_1}};
  auto table_1    = cudf::table_view({struct_2_1});

  auto weight_2   = cudf::test::fixed_width_column_wrapper<float>{{1.1, -1.0, -1.0}};
  auto ages_2     = cudf::test::fixed_width_column_wrapper<int32_t>{{31, 351, 351}, {1, 1, 0}};
  auto struct_1_2 = cudf::test::structs_column_wrapper{{weight_2, ages_2}, {1, 0, 1}};
  auto is_human_2 = cudf::test::fixed_width_column_wrapper<bool>{{false, false, false}};
  auto struct_2_2 = cudf::test::structs_column_wrapper{{is_human_2, struct_1_2}};
  auto table_2    = cudf::table_view({struct_2_2});

  auto full_table = cudf::concatenate(std::vector<table_view>({table_1, table_2}));

  cudf::io::table_input_metadata expected_metadata(table_1);
  expected_metadata.column_metadata[0].set_name("being").set_nullability(false);
  expected_metadata.column_metadata[0].child(0).set_name("human?").set_nullability(false);
  expected_metadata.column_metadata[0].child(1).set_name("particulars");
  expected_metadata.column_metadata[0].child(1).child(0).set_name("weight");
  expected_metadata.column_metadata[0].child(1).child(1).set_name("age");

  auto filepath = temp_env->get_temp_filepath("ChunkedNullableStruct.parquet");
  cudf::io::chunked_parquet_writer_options args =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath});
  args.set_metadata(expected_metadata);
  cudf::io::parquet_chunked_writer(args).write(table_1).write(table_2);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *full_table);
  cudf::test::expect_metadata_equal(expected_metadata, result.metadata);
}

TEST_F(ParquetChunkedWriterTest, ReadRowGroups)
{
  srand(31337);
  auto table1 = create_random_fixed_table<int>(5, 5, true);
  auto table2 = create_random_fixed_table<int>(5, 5, true);

  auto full_table = cudf::concatenate(std::vector<table_view>({*table2, *table1, *table2}));

  auto filepath = temp_env->get_temp_filepath("ChunkedRowGroups.parquet");
  cudf::io::chunked_parquet_writer_options args =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath});
  {
    cudf::io::parquet_chunked_writer(args).write(*table1).write(*table2);
  }

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
      .row_groups({{1, 0, 1}});
  auto result = cudf::io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *full_table);
}

TEST_F(ParquetChunkedWriterTest, ReadRowGroupsError)
{
  srand(31337);
  auto table1 = create_random_fixed_table<int>(5, 5, true);

  auto filepath = temp_env->get_temp_filepath("ChunkedRowGroupsError.parquet");
  cudf::io::chunked_parquet_writer_options args =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath});
  cudf::io::parquet_chunked_writer(args).write(*table1);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath}).row_groups({{0, 1}});
  EXPECT_THROW(cudf::io::read_parquet(read_opts), cudf::logic_error);
  read_opts.set_row_groups({{-1}});
  EXPECT_THROW(cudf::io::read_parquet(read_opts), cudf::logic_error);
  read_opts.set_row_groups({{0}, {0}});
  EXPECT_THROW(cudf::io::read_parquet(read_opts), cudf::logic_error);
}

TYPED_TEST(ParquetChunkedWriterNumericTypeTest, UnalignedSize)
{
  // write out two 31 row tables and make sure they get
  // read back with all their validity bits in the right place

  using T = TypeParam;

  int num_els = 31;
  std::vector<std::unique_ptr<cudf::column>> cols;

  bool mask[] = {false, true, true, true, true, true, true, true, true, true, true,
                 true,  true, true, true, true, true, true, true, true, true, true,

                 true,  true, true, true, true, true, true, true, true};
  T c1a[num_els];
  std::fill(c1a, c1a + num_els, static_cast<T>(5));
  T c1b[num_els];
  std::fill(c1b, c1b + num_els, static_cast<T>(6));
  column_wrapper<T> c1a_w(c1a, c1a + num_els, mask);
  column_wrapper<T> c1b_w(c1b, c1b + num_els, mask);
  cols.push_back(c1a_w.release());
  cols.push_back(c1b_w.release());
  cudf::table tbl1(std::move(cols));

  T c2a[num_els];
  std::fill(c2a, c2a + num_els, static_cast<T>(8));
  T c2b[num_els];
  std::fill(c2b, c2b + num_els, static_cast<T>(9));
  column_wrapper<T> c2a_w(c2a, c2a + num_els, mask);
  column_wrapper<T> c2b_w(c2b, c2b + num_els, mask);
  cols.push_back(c2a_w.release());
  cols.push_back(c2b_w.release());
  cudf::table tbl2(std::move(cols));

  auto expected = cudf::concatenate(std::vector<table_view>({tbl1, tbl2}));

  auto filepath = temp_env->get_temp_filepath("ChunkedUnalignedSize.parquet");
  cudf::io::chunked_parquet_writer_options args =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath});
  cudf::io::parquet_chunked_writer(args).write(tbl1).write(tbl2);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *expected);
}

TYPED_TEST(ParquetChunkedWriterNumericTypeTest, UnalignedSize2)
{
  // write out two 33 row tables and make sure they get
  // read back with all their validity bits in the right place

  using T = TypeParam;

  int num_els = 33;
  std::vector<std::unique_ptr<cudf::column>> cols;

  bool mask[] = {false, true, true, true, true, true, true, true, true, true, true,
                 true,  true, true, true, true, true, true, true, true, true, true,
                 true,  true, true, true, true, true, true, true, true, true, true};

  T c1a[num_els];
  std::fill(c1a, c1a + num_els, static_cast<T>(5));
  T c1b[num_els];
  std::fill(c1b, c1b + num_els, static_cast<T>(6));
  column_wrapper<T> c1a_w(c1a, c1a + num_els, mask);
  column_wrapper<T> c1b_w(c1b, c1b + num_els, mask);
  cols.push_back(c1a_w.release());
  cols.push_back(c1b_w.release());
  cudf::table tbl1(std::move(cols));

  T c2a[num_els];
  std::fill(c2a, c2a + num_els, static_cast<T>(8));
  T c2b[num_els];
  std::fill(c2b, c2b + num_els, static_cast<T>(9));
  column_wrapper<T> c2a_w(c2a, c2a + num_els, mask);
  column_wrapper<T> c2b_w(c2b, c2b + num_els, mask);
  cols.push_back(c2a_w.release());
  cols.push_back(c2b_w.release());
  cudf::table tbl2(std::move(cols));

  auto expected = cudf::concatenate(std::vector<table_view>({tbl1, tbl2}));

  auto filepath = temp_env->get_temp_filepath("ChunkedUnalignedSize2.parquet");
  cudf::io::chunked_parquet_writer_options args =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath});
  cudf::io::parquet_chunked_writer(args).write(tbl1).write(tbl2);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *expected);
}

// custom mem mapped data sink that supports device writes
template <bool supports_device_writes>
class custom_test_memmap_sink : public cudf::io::data_sink {
 public:
  explicit custom_test_memmap_sink(std::vector<char>* mm_writer_buf)
  {
    mm_writer = cudf::io::data_sink::create(mm_writer_buf);
  }

  virtual ~custom_test_memmap_sink() { mm_writer->flush(); }

  void host_write(void const* data, size_t size) override { mm_writer->host_write(data, size); }

  [[nodiscard]] bool supports_device_write() const override { return supports_device_writes; }

  void device_write(void const* gpu_data, size_t size, rmm::cuda_stream_view stream) override
  {
    this->device_write_async(gpu_data, size, stream).get();
  }

  std::future<void> device_write_async(void const* gpu_data,
                                       size_t size,
                                       rmm::cuda_stream_view stream) override
  {
    return std::async(std::launch::deferred, [=] {
      char* ptr = nullptr;
      CUDF_CUDA_TRY(cudaMallocHost(&ptr, size));
      CUDF_CUDA_TRY(cudaMemcpyAsync(ptr, gpu_data, size, cudaMemcpyDefault, stream.value()));
      stream.synchronize();
      mm_writer->host_write(ptr, size);
      CUDF_CUDA_TRY(cudaFreeHost(ptr));
    });
  }

  void flush() override { mm_writer->flush(); }

  size_t bytes_written() override { return mm_writer->bytes_written(); }

 private:
  std::unique_ptr<data_sink> mm_writer;
};

TEST_F(ParquetWriterStressTest, LargeTableWeakCompression)
{
  std::vector<char> mm_buf;
  mm_buf.reserve(4 * 1024 * 1024 * 16);
  custom_test_memmap_sink<false> custom_sink(&mm_buf);

  // exercises multiple rowgroups
  srand(31337);
  auto expected = create_random_fixed_table<int>(16, 4 * 1024 * 1024, false);

  // write out using the custom sink (which uses device writes)
  cudf::io::parquet_writer_options args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{&custom_sink}, *expected);
  cudf::io::write_parquet(args);

  cudf::io::parquet_reader_options custom_args =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{mm_buf.data(), mm_buf.size()});
  auto custom_tbl = cudf::io::read_parquet(custom_args);
  CUDF_TEST_EXPECT_TABLES_EQUAL(custom_tbl.tbl->view(), expected->view());
}

TEST_F(ParquetWriterStressTest, LargeTableGoodCompression)
{
  std::vector<char> mm_buf;
  mm_buf.reserve(4 * 1024 * 1024 * 16);
  custom_test_memmap_sink<false> custom_sink(&mm_buf);

  // exercises multiple rowgroups
  srand(31337);
  auto expected = create_compressible_fixed_table<int>(16, 4 * 1024 * 1024, 128 * 1024, false);

  // write out using the custom sink (which uses device writes)
  cudf::io::parquet_writer_options args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{&custom_sink}, *expected);
  cudf::io::write_parquet(args);

  cudf::io::parquet_reader_options custom_args =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{mm_buf.data(), mm_buf.size()});
  auto custom_tbl = cudf::io::read_parquet(custom_args);
  CUDF_TEST_EXPECT_TABLES_EQUAL(custom_tbl.tbl->view(), expected->view());
}

TEST_F(ParquetWriterStressTest, LargeTableWithValids)
{
  std::vector<char> mm_buf;
  mm_buf.reserve(4 * 1024 * 1024 * 16);
  custom_test_memmap_sink<false> custom_sink(&mm_buf);

  // exercises multiple rowgroups
  srand(31337);
  auto expected = create_compressible_fixed_table<int>(16, 4 * 1024 * 1024, 6, true);

  // write out using the custom sink (which uses device writes)
  cudf::io::parquet_writer_options args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{&custom_sink}, *expected);
  cudf::io::write_parquet(args);

  cudf::io::parquet_reader_options custom_args =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{mm_buf.data(), mm_buf.size()});
  auto custom_tbl = cudf::io::read_parquet(custom_args);
  CUDF_TEST_EXPECT_TABLES_EQUAL(custom_tbl.tbl->view(), expected->view());
}

TEST_F(ParquetWriterStressTest, DeviceWriteLargeTableWeakCompression)
{
  std::vector<char> mm_buf;
  mm_buf.reserve(4 * 1024 * 1024 * 16);
  custom_test_memmap_sink<true> custom_sink(&mm_buf);

  // exercises multiple rowgroups
  srand(31337);
  auto expected = create_random_fixed_table<int>(16, 4 * 1024 * 1024, false);

  // write out using the custom sink (which uses device writes)
  cudf::io::parquet_writer_options args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{&custom_sink}, *expected);
  cudf::io::write_parquet(args);

  cudf::io::parquet_reader_options custom_args =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{mm_buf.data(), mm_buf.size()});
  auto custom_tbl = cudf::io::read_parquet(custom_args);
  CUDF_TEST_EXPECT_TABLES_EQUAL(custom_tbl.tbl->view(), expected->view());
}

TEST_F(ParquetWriterStressTest, DeviceWriteLargeTableGoodCompression)
{
  std::vector<char> mm_buf;
  mm_buf.reserve(4 * 1024 * 1024 * 16);
  custom_test_memmap_sink<true> custom_sink(&mm_buf);

  // exercises multiple rowgroups
  srand(31337);
  auto expected = create_compressible_fixed_table<int>(16, 4 * 1024 * 1024, 128 * 1024, false);

  // write out using the custom sink (which uses device writes)
  cudf::io::parquet_writer_options args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{&custom_sink}, *expected);
  cudf::io::write_parquet(args);

  cudf::io::parquet_reader_options custom_args =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{mm_buf.data(), mm_buf.size()});
  auto custom_tbl = cudf::io::read_parquet(custom_args);
  CUDF_TEST_EXPECT_TABLES_EQUAL(custom_tbl.tbl->view(), expected->view());
}

TEST_F(ParquetWriterStressTest, DeviceWriteLargeTableWithValids)
{
  std::vector<char> mm_buf;
  mm_buf.reserve(4 * 1024 * 1024 * 16);
  custom_test_memmap_sink<true> custom_sink(&mm_buf);

  // exercises multiple rowgroups
  srand(31337);
  auto expected = create_compressible_fixed_table<int>(16, 4 * 1024 * 1024, 6, true);

  // write out using the custom sink (which uses device writes)
  cudf::io::parquet_writer_options args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{&custom_sink}, *expected);
  cudf::io::write_parquet(args);

  cudf::io::parquet_reader_options custom_args =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{mm_buf.data(), mm_buf.size()});
  auto custom_tbl = cudf::io::read_parquet(custom_args);
  CUDF_TEST_EXPECT_TABLES_EQUAL(custom_tbl.tbl->view(), expected->view());
}

TEST_F(ParquetChunkedWriterTest, RowGroupPageSizeMatch)
{
  std::vector<char> out_buffer;

  auto options = cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info(&out_buffer))
                   .row_group_size_bytes(128 * 1024)
                   .max_page_size_bytes(512 * 1024)
                   .row_group_size_rows(10000)
                   .max_page_size_rows(20000)
                   .build();
  EXPECT_EQ(options.get_row_group_size_bytes(), options.get_max_page_size_bytes());
  EXPECT_EQ(options.get_row_group_size_rows(), options.get_max_page_size_rows());
}

TYPED_TEST(ParquetWriterComparableTypeTest, ThreeColumnSorted)
{
  using T = TypeParam;

  auto col0 = testdata::ascending<T>();
  auto col1 = testdata::descending<T>();
  auto col2 = testdata::unordered<T>();

  auto const expected = table_view{{col0, col1, col2}};

  auto const filepath = temp_env->get_temp_filepath("ThreeColumnSorted.parquet");
  const cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .max_page_size_rows(page_size_for_ordered_tests)
      .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN);
  cudf::io::write_parquet(out_opts);

  auto const source = cudf::io::datasource::create(filepath);
  cudf::io::parquet::detail::FileMetaData fmd;

  read_footer(source, &fmd);
  ASSERT_GT(fmd.row_groups.size(), 0);

  auto const& columns = fmd.row_groups[0].columns;
  ASSERT_EQ(columns.size(), static_cast<size_t>(expected.num_columns()));

  // now check that the boundary order for chunk 1 is ascending,
  // chunk 2 is descending, and chunk 3 is unordered
  cudf::io::parquet::detail::BoundaryOrder expected_orders[] = {
    cudf::io::parquet::detail::BoundaryOrder::ASCENDING,
    cudf::io::parquet::detail::BoundaryOrder::DESCENDING,
    cudf::io::parquet::detail::BoundaryOrder::UNORDERED};

  for (std::size_t i = 0; i < columns.size(); i++) {
    auto const ci = read_column_index(source, columns[i]);
    EXPECT_EQ(ci.boundary_order, expected_orders[i]);
  }
}

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

TEST_F(ParquetChunkedWriterTest, CompStats)
{
  auto table = create_random_fixed_table<int>(1, 100000, true);

  auto const stats = std::make_shared<cudf::io::writer_compression_statistics>();

  std::vector<char> unused_buffer;
  cudf::io::chunked_parquet_writer_options opts =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{&unused_buffer})
      .compression_statistics(stats);
  cudf::io::parquet_chunked_writer(opts).write(*table);

  EXPECT_NE(stats->num_compressed_bytes(), 0);
  EXPECT_EQ(stats->num_failed_bytes(), 0);
  EXPECT_EQ(stats->num_skipped_bytes(), 0);
  EXPECT_FALSE(std::isnan(stats->compression_ratio()));

  auto const single_table_comp_stats = *stats;
  cudf::io::parquet_chunked_writer(opts).write(*table);

  EXPECT_EQ(stats->compression_ratio(), single_table_comp_stats.compression_ratio());
  EXPECT_EQ(stats->num_compressed_bytes(), 2 * single_table_comp_stats.num_compressed_bytes());

  EXPECT_EQ(stats->num_failed_bytes(), 0);
  EXPECT_EQ(stats->num_skipped_bytes(), 0);
}

TEST_F(ParquetChunkedWriterTest, CompStatsEmptyTable)
{
  auto table_no_rows = create_random_fixed_table<int>(20, 0, false);

  auto const stats = std::make_shared<cudf::io::writer_compression_statistics>();

  std::vector<char> unused_buffer;
  cudf::io::chunked_parquet_writer_options opts =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{&unused_buffer})
      .compression_statistics(stats);
  cudf::io::parquet_chunked_writer(opts).write(*table_no_rows);

  expect_compression_stats_empty(stats);
}

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

CUDF_TEST_PROGRAM_MAIN()
