/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/io/data_sink.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <fstream>
#include <type_traits>

namespace cudf_io = cudf::io;

template <typename T, typename SourceElementT = T>
using column_wrapper =
  typename std::conditional<std::is_same<T, cudf::string_view>::value,
                            cudf::test::strings_column_wrapper,
                            cudf::test::fixed_width_column_wrapper<T, SourceElementT>>::type;
using column     = cudf::column;
using table      = cudf::table;
using table_view = cudf::table_view;

// Global environment for temporary files
auto const temp_env = static_cast<cudf::test::TempDirTestEnvironment*>(
  ::testing::AddGlobalTestEnvironment(new cudf::test::TempDirTestEnvironment));

template <typename T, typename Elements>
std::unique_ptr<cudf::table> create_fixed_table(cudf::size_type num_columns,
                                                cudf::size_type num_rows,
                                                bool include_validity,
                                                Elements elements)
{
  auto valids = cudf::test::make_counting_transform_iterator(
    0, [](auto i) { return i % 2 == 0 ? true : false; });
  std::vector<cudf::test::fixed_width_column_wrapper<T>> src_cols(num_columns);
  for (int idx = 0; idx < num_columns; idx++) {
    if (include_validity) {
      src_cols[idx] =
        cudf::test::fixed_width_column_wrapper<T>(elements, elements + num_rows, valids);
    } else {
      src_cols[idx] = cudf::test::fixed_width_column_wrapper<T>(elements, elements + num_rows);
    }
  }
  std::vector<std::unique_ptr<cudf::column>> columns(num_columns);
  std::transform(src_cols.begin(),
                 src_cols.end(),
                 columns.begin(),
                 [](cudf::test::fixed_width_column_wrapper<T>& in) {
                   auto ret = in.release();
                   ret->has_nulls();
                   return ret;
                 });
  return std::make_unique<cudf::table>(std::move(columns));
}

template <typename T>
std::unique_ptr<cudf::table> create_random_fixed_table(cudf::size_type num_columns,
                                                       cudf::size_type num_rows,
                                                       bool include_validity)
{
  auto rand_elements = cudf::test::make_counting_transform_iterator(0, [](T i) { return rand(); });
  return create_fixed_table<T>(num_columns, num_rows, include_validity, rand_elements);
}

template <typename T>
std::unique_ptr<cudf::table> create_compressible_fixed_table(cudf::size_type num_columns,
                                                             cudf::size_type num_rows,
                                                             cudf::size_type period,
                                                             bool include_validity)
{
  auto compressible_elements =
    cudf::test::make_counting_transform_iterator(0, [period](T i) { return i / period; });
  return create_fixed_table<T>(num_columns, num_rows, include_validity, compressible_elements);
}

// Base test fixture for tests
struct ParquetWriterTest : public cudf::test::BaseFixture {
};

// Base test fixture for tests
struct ParquetReaderTest : public cudf::test::BaseFixture {
};

// Base test fixture for "stress" tests
struct ParquetWriterStressTest : public cudf::test::BaseFixture {
};

// Typed test fixture for numeric type tests
template <typename T>
struct ParquetWriterNumericTypeTest : public ParquetWriterTest {
  auto type() { return cudf::data_type{cudf::type_to_id<T>()}; }
};

// Typed test fixture for timestamp type tests
template <typename T>
struct ParquetWriterChronoTypeTest : public ParquetWriterTest {
  auto type() { return cudf::data_type{cudf::type_to_id<T>()}; }
};

// Declare typed test cases
// TODO: Replace with `NumericTypes` when unsigned support is added. Issue #5352
using SupportedTypes = cudf::test::Types<int8_t, int16_t, int32_t, int64_t, bool, float, double>;
TYPED_TEST_CASE(ParquetWriterNumericTypeTest, SupportedTypes);
using SupportedChronoTypes = cudf::test::Concat<cudf::test::ChronoTypes, cudf::test::DurationTypes>;
TYPED_TEST_CASE(ParquetWriterChronoTypeTest, SupportedChronoTypes);

// Base test fixture for chunked writer tests
struct ParquetChunkedWriterTest : public cudf::test::BaseFixture {
};

// Typed test fixture for numeric type tests
template <typename T>
struct ParquetChunkedWriterNumericTypeTest : public ParquetChunkedWriterTest {
  auto type() { return cudf::data_type{cudf::type_to_id<T>()}; }
};

// Declare typed test cases
TYPED_TEST_CASE(ParquetChunkedWriterNumericTypeTest, SupportedTypes);

namespace {
// Generates a vector of uniform random values of type T
template <typename T>
inline auto random_values(size_t size)
{
  std::vector<T> values(size);

  using T1 = T;
  using uniform_distribution =
    typename std::conditional_t<std::is_same<T1, bool>::value,
                                std::bernoulli_distribution,
                                std::conditional_t<std::is_floating_point<T1>::value,
                                                   std::uniform_real_distribution<T1>,
                                                   std::uniform_int_distribution<T1>>>;

  static constexpr auto seed = 0xf00d;
  static std::mt19937 engine{seed};
  static uniform_distribution dist{};
  std::generate_n(values.begin(), size, [&]() { return T{dist(engine)}; });

  return values;
}

}  // namespace

TYPED_TEST(ParquetWriterNumericTypeTest, SingleColumn)
{
  auto sequence =
    cudf::test::make_counting_transform_iterator(0, [](auto i) { return TypeParam(i); });
  auto validity = cudf::test::make_counting_transform_iterator(0, [](auto i) { return true; });

  constexpr auto num_rows = 100;
  column_wrapper<TypeParam> col(sequence, sequence + num_rows, validity);

  std::vector<std::unique_ptr<column>> cols;
  cols.push_back(col.release());
  auto expected = std::make_unique<table>(std::move(cols));
  EXPECT_EQ(1, expected->num_columns());

  auto filepath = temp_env->get_temp_filepath("SingleColumn.parquet");
  cudf_io::parquet_writer_options out_opts =
    cudf_io::parquet_writer_options::builder(cudf_io::sink_info{filepath}, expected->view());
  cudf_io::write_parquet(out_opts);

  cudf_io::parquet_reader_options in_opts =
    cudf_io::parquet_reader_options::builder(cudf_io::source_info{filepath});
  auto result = cudf_io::read_parquet(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), result.tbl->view());
}

TYPED_TEST(ParquetWriterNumericTypeTest, SingleColumnWithNulls)
{
  auto sequence =
    cudf::test::make_counting_transform_iterator(0, [](auto i) { return TypeParam(i); });
  auto validity = cudf::test::make_counting_transform_iterator(0, [](auto i) { return (i % 2); });

  constexpr auto num_rows = 100;
  column_wrapper<TypeParam> col(sequence, sequence + num_rows, validity);

  std::vector<std::unique_ptr<column>> cols;
  cols.push_back(col.release());
  auto expected = std::make_unique<table>(std::move(cols));
  EXPECT_EQ(1, expected->num_columns());

  auto filepath = temp_env->get_temp_filepath("SingleColumnWithNulls.parquet");
  cudf_io::parquet_writer_options out_opts =
    cudf_io::parquet_writer_options::builder(cudf_io::sink_info{filepath}, expected->view());
  cudf_io::write_parquet(out_opts);

  cudf_io::parquet_reader_options in_opts =
    cudf_io::parquet_reader_options::builder(cudf_io::source_info{filepath});
  auto result = cudf_io::read_parquet(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), result.tbl->view());
}

TYPED_TEST(ParquetWriterChronoTypeTest, Chronos)
{
  auto sequence = cudf::test::make_counting_transform_iterator(
    0, [](auto i) { return ((std::rand() / 10000) * 1000); });
  auto validity = cudf::test::make_counting_transform_iterator(0, [](auto i) { return true; });

  constexpr auto num_rows = 100;
  column_wrapper<TypeParam, typename decltype(sequence)::value_type> col(
    sequence, sequence + num_rows, validity);

  std::vector<std::unique_ptr<column>> cols;
  cols.push_back(col.release());
  auto expected = std::make_unique<table>(std::move(cols));
  EXPECT_EQ(1, expected->num_columns());

  auto filepath = temp_env->get_temp_filepath("Chronos.parquet");
  cudf_io::parquet_writer_options out_opts =
    cudf_io::parquet_writer_options::builder(cudf_io::sink_info{filepath}, expected->view());
  cudf_io::write_parquet(out_opts);

  cudf_io::parquet_reader_options in_opts =
    cudf_io::parquet_reader_options::builder(cudf_io::source_info{filepath})
      .timestamp_type(this->type());
  auto result = cudf_io::read_parquet(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), result.tbl->view());
}

TYPED_TEST(ParquetWriterChronoTypeTest, ChronosWithNulls)
{
  auto sequence = cudf::test::make_counting_transform_iterator(
    0, [](auto i) { return ((std::rand() / 10000) * 1000); });
  auto validity =
    cudf::test::make_counting_transform_iterator(0, [](auto i) { return (i > 30) && (i < 60); });

  constexpr auto num_rows = 100;
  column_wrapper<TypeParam, typename decltype(sequence)::value_type> col(
    sequence, sequence + num_rows, validity);

  std::vector<std::unique_ptr<column>> cols;
  cols.push_back(col.release());
  auto expected = std::make_unique<table>(std::move(cols));
  EXPECT_EQ(1, expected->num_columns());

  auto filepath = temp_env->get_temp_filepath("ChronosWithNulls.parquet");
  cudf_io::parquet_writer_options out_opts =
    cudf_io::parquet_writer_options::builder(cudf_io::sink_info{filepath}, expected->view());
  cudf_io::write_parquet(out_opts);

  cudf_io::parquet_reader_options in_opts =
    cudf_io::parquet_reader_options::builder(cudf_io::source_info{filepath})
      .timestamp_type(this->type());
  auto result = cudf_io::read_parquet(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), result.tbl->view());
}

TEST_F(ParquetWriterTest, MultiColumn)
{
  constexpr auto num_rows = 100;

  // auto col0_data = random_values<bool>(num_rows);
  auto col1_data = random_values<int8_t>(num_rows);
  auto col2_data = random_values<int16_t>(num_rows);
  auto col3_data = random_values<int32_t>(num_rows);
  auto col4_data = random_values<float>(num_rows);
  auto col5_data = random_values<double>(num_rows);
  auto validity  = cudf::test::make_counting_transform_iterator(0, [](auto i) { return true; });

  // column_wrapper<bool> col0{
  //    col0_data.begin(), col0_data.end(), validity};
  column_wrapper<int8_t> col1{col1_data.begin(), col1_data.end(), validity};
  column_wrapper<int16_t> col2{col2_data.begin(), col2_data.end(), validity};
  column_wrapper<int32_t> col3{col3_data.begin(), col3_data.end(), validity};
  column_wrapper<float> col4{col4_data.begin(), col4_data.end(), validity};
  column_wrapper<double> col5{col5_data.begin(), col5_data.end(), validity};

  cudf_io::table_metadata expected_metadata;
  // expected_metadata.column_names.emplace_back("bools");
  expected_metadata.column_names.emplace_back("int8s");
  expected_metadata.column_names.emplace_back("int16s");
  expected_metadata.column_names.emplace_back("int32s");
  expected_metadata.column_names.emplace_back("floats");
  expected_metadata.column_names.emplace_back("doubles");

  std::vector<std::unique_ptr<column>> cols;
  // cols.push_back(col0.release());
  cols.push_back(col1.release());
  cols.push_back(col2.release());
  cols.push_back(col3.release());
  cols.push_back(col4.release());
  cols.push_back(col5.release());
  auto expected = std::make_unique<table>(std::move(cols));
  EXPECT_EQ(5, expected->num_columns());

  auto filepath = temp_env->get_temp_filepath("MultiColumn.parquet");
  cudf_io::parquet_writer_options out_opts =
    cudf_io::parquet_writer_options::builder(cudf_io::sink_info{filepath}, expected->view())
      .metadata(&expected_metadata);
  cudf_io::write_parquet(out_opts);

  cudf_io::parquet_reader_options in_opts =
    cudf_io::parquet_reader_options::builder(cudf_io::source_info{filepath});
  auto result = cudf_io::read_parquet(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), result.tbl->view());
  EXPECT_EQ(expected_metadata.column_names, result.metadata.column_names);
}

TEST_F(ParquetWriterTest, MultiColumnWithNulls)
{
  constexpr auto num_rows = 100;

  // auto col0_data = random_values<bool>(num_rows);
  auto col1_data = random_values<int8_t>(num_rows);
  auto col2_data = random_values<int16_t>(num_rows);
  auto col3_data = random_values<int32_t>(num_rows);
  auto col4_data = random_values<float>(num_rows);
  auto col5_data = random_values<double>(num_rows);
  // auto col0_mask = cudf::test::make_counting_transform_iterator(
  //    0, [](auto i) { return (i % 2); });
  auto col1_mask = cudf::test::make_counting_transform_iterator(0, [](auto i) { return (i < 10); });
  auto col2_mask = cudf::test::make_counting_transform_iterator(0, [](auto i) { return true; });
  auto col3_mask =
    cudf::test::make_counting_transform_iterator(0, [](auto i) { return (i == (num_rows - 1)); });
  auto col4_mask =
    cudf::test::make_counting_transform_iterator(0, [](auto i) { return (i >= 40 || i <= 60); });
  auto col5_mask = cudf::test::make_counting_transform_iterator(0, [](auto i) { return (i > 80); });

  // column_wrapper<bool> col0{
  //    col0_data.begin(), col0_data.end(), col0_mask};
  column_wrapper<int8_t> col1{col1_data.begin(), col1_data.end(), col1_mask};
  column_wrapper<int16_t> col2{col2_data.begin(), col2_data.end(), col2_mask};
  column_wrapper<int32_t> col3{col3_data.begin(), col3_data.end(), col3_mask};
  column_wrapper<float> col4{col4_data.begin(), col4_data.end(), col4_mask};
  column_wrapper<double> col5{col5_data.begin(), col5_data.end(), col5_mask};

  cudf_io::table_metadata expected_metadata;
  // expected_metadata.column_names.emplace_back("bools");
  expected_metadata.column_names.emplace_back("int8s");
  expected_metadata.column_names.emplace_back("int16s");
  expected_metadata.column_names.emplace_back("int32s");
  expected_metadata.column_names.emplace_back("floats");
  expected_metadata.column_names.emplace_back("doubles");

  std::vector<std::unique_ptr<column>> cols;
  // cols.push_back(col0.release());
  cols.push_back(col1.release());
  cols.push_back(col2.release());
  cols.push_back(col3.release());
  cols.push_back(col4.release());
  cols.push_back(col5.release());
  auto expected = std::make_unique<table>(std::move(cols));
  EXPECT_EQ(5, expected->num_columns());

  auto filepath = temp_env->get_temp_filepath("MultiColumnWithNulls.parquet");
  cudf_io::parquet_writer_options out_opts =
    cudf_io::parquet_writer_options::builder(cudf_io::sink_info{filepath}, expected->view())
      .metadata(&expected_metadata);
  cudf_io::write_parquet(out_opts);

  cudf_io::parquet_reader_options in_opts =
    cudf_io::parquet_reader_options::builder(cudf_io::source_info{filepath});
  auto result = cudf_io::read_parquet(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), result.tbl->view());
  EXPECT_EQ(expected_metadata.column_names, result.metadata.column_names);
}

TEST_F(ParquetWriterTest, Strings)
{
  std::vector<const char*> strings{
    "Monday", "Monday", "Friday", "Monday", "Friday", "Friday", "Friday", "Funday"};
  const auto num_rows = strings.size();

  auto seq_col0 = random_values<int>(num_rows);
  auto seq_col2 = random_values<float>(num_rows);
  auto validity = cudf::test::make_counting_transform_iterator(0, [](auto i) { return true; });

  column_wrapper<int> col0{seq_col0.begin(), seq_col0.end(), validity};
  column_wrapper<cudf::string_view> col1{strings.begin(), strings.end()};
  column_wrapper<float> col2{seq_col2.begin(), seq_col2.end(), validity};

  cudf_io::table_metadata expected_metadata;
  expected_metadata.column_names.emplace_back("col_other");
  expected_metadata.column_names.emplace_back("col_string");
  expected_metadata.column_names.emplace_back("col_another");

  std::vector<std::unique_ptr<column>> cols;
  cols.push_back(col0.release());
  cols.push_back(col1.release());
  cols.push_back(col2.release());
  auto expected = std::make_unique<table>(std::move(cols));
  EXPECT_EQ(3, expected->num_columns());

  auto filepath = temp_env->get_temp_filepath("Strings.parquet");
  cudf_io::parquet_writer_options out_opts =
    cudf_io::parquet_writer_options::builder(cudf_io::sink_info{filepath}, expected->view())
      .metadata(&expected_metadata);
  cudf_io::write_parquet(out_opts);

  cudf_io::parquet_reader_options in_opts =
    cudf_io::parquet_reader_options::builder(cudf_io::source_info{filepath});
  auto result = cudf_io::read_parquet(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), result.tbl->view());
  EXPECT_EQ(expected_metadata.column_names, result.metadata.column_names);
}

TEST_F(ParquetWriterTest, SlicedTable)
{
  // This test checks for writing zero copy, offseted views into existing cudf tables

  std::vector<const char*> strings{
    "Monday", "Monday", "Friday", "Monday", "Friday", "Friday", "Friday", "Funday"};
  const auto num_rows = strings.size();

  auto seq_col0 = random_values<int>(num_rows);
  auto seq_col2 = random_values<float>(num_rows);
  auto validity = cudf::test::make_counting_transform_iterator(0, [](auto i) { return true; });

  column_wrapper<int> col0{seq_col0.begin(), seq_col0.end(), validity};
  column_wrapper<cudf::string_view> col1{strings.begin(), strings.end()};
  column_wrapper<float> col2{seq_col2.begin(), seq_col2.end(), validity};

  using lcw = cudf::test::lists_column_wrapper<uint64_t>;
  lcw col3{{9, 8}, {7, 6, 5}, {}, {4}, {3, 2, 1, 0}, {20, 21, 22, 23, 24}, {}, {66, 666}};

  // [[[NULL,2,NULL,4]], [[NULL,6,NULL], [8,9]]]
  // [NULL, [[13],[14,15,16]],  NULL]
  // [NULL, [], NULL, [[]]]
  // NULL
  // [[[NULL,2,NULL,4]], [[NULL,6,NULL], [8,9]]]
  // [NULL, [[13],[14,15,16]],  NULL]
  // [[[]]]
  // [NULL, [], NULL, [[]]]
  auto valids  = cudf::test::make_counting_transform_iterator(0, [](auto i) { return i % 2; });
  auto valids2 = cudf::test::make_counting_transform_iterator(0, [](auto i) { return i != 3; });
  lcw col4{{
             {{{{1, 2, 3, 4}, valids}}, {{{5, 6, 7}, valids}, {8, 9}}},
             {{{{10, 11}, {12}}, {{13}, {14, 15, 16}}, {{17, 18}}}, valids},
             {{lcw{lcw{}}, lcw{}, lcw{}, lcw{lcw{}}}, valids},
             lcw{lcw{lcw{}}},
             {{{{1, 2, 3, 4}, valids}}, {{{5, 6, 7}, valids}, {8, 9}}},
             {{{{10, 11}, {12}}, {{13}, {14, 15, 16}}, {{17, 18}}}, valids},
             lcw{lcw{lcw{}}},
             {{lcw{lcw{}}, lcw{}, lcw{}, lcw{lcw{}}}, valids},
           },
           valids2};

  cudf_io::table_metadata expected_metadata;
  expected_metadata.column_names.emplace_back("col_other");
  expected_metadata.column_names.emplace_back("col_string");
  expected_metadata.column_names.emplace_back("col_another");
  expected_metadata.column_names.emplace_back("col_list");
  expected_metadata.column_names.emplace_back("col_multi_level_list");

  auto expected = table_view({col0, col1, col2, col3, col4});

  auto expected_slice = cudf::slice(expected, {2, static_cast<cudf::size_type>(num_rows) - 1});

  auto filepath = temp_env->get_temp_filepath("SlicedTable.parquet");
  cudf_io::parquet_writer_options out_opts =
    cudf_io::parquet_writer_options::builder(cudf_io::sink_info{filepath}, expected_slice)
      .metadata(&expected_metadata);
  cudf_io::write_parquet(out_opts);

  cudf_io::parquet_reader_options in_opts =
    cudf_io::parquet_reader_options::builder(cudf_io::source_info{filepath});
  auto result = cudf_io::read_parquet(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_slice, result.tbl->view());
  EXPECT_EQ(expected_metadata.column_names, result.metadata.column_names);
}

TEST_F(ParquetWriterTest, ListColumn)
{
  auto valids  = cudf::test::make_counting_transform_iterator(0, [](auto i) { return i % 2; });
  auto valids2 = cudf::test::make_counting_transform_iterator(0, [](auto i) { return i != 3; });

  using lcw = cudf::test::lists_column_wrapper<int32_t>;

  // [NULL, 2, NULL]
  // []
  // [4, 5]
  // NULL
  lcw col0{{{{1, 2, 3}, valids}, {}, {4, 5}, {}}, valids2};

  // [[1, 2, 3], [], [4, 5], [], [0, 6, 0]]
  // [[7, 8]]
  // []
  // [[]]
  lcw col1{{{1, 2, 3}, {}, {4, 5}, {}, {0, 6, 0}}, {{7, 8}}, lcw{}, lcw{lcw{}}};

  // [[1, 2, 3], [], [4, 5], NULL, [0, 6, 0]]
  // [[7, 8]]
  // []
  // [[]]
  lcw col2{{{{1, 2, 3}, {}, {4, 5}, {}, {0, 6, 0}}, valids2}, {{7, 8}}, lcw{}, lcw{lcw{}}};

  // [[1, 2, 3], [], [4, 5], NULL, [NULL, 6, NULL]]
  // [[7, 8]]
  // []
  // [[]]
  using dlcw = cudf::test::lists_column_wrapper<double>;
  dlcw col3{{{{1., 2., 3.}, {}, {4., 5.}, {}, {{0., 6., 0.}, valids}}, valids2},
            {{7., 8.}},
            dlcw{},
            dlcw{dlcw{}}};

  // TODO: uint16_t lists are not read properly in parquet reader
  // [[1, 2, 3], [], [4, 5], NULL, [0, 6, 0]]
  // [[7, 8]]
  // []
  // NULL
  // using ui16lcw = cudf::test::lists_column_wrapper<uint16_t>;
  // cudf::test::lists_column_wrapper<uint16_t> col4{
  //   {{{{1, 2, 3}, {}, {4, 5}, {}, {0, 6, 0}}, valids2}, {{7, 8}}, ui16lcw{}, ui16lcw{ui16lcw{}}},
  //   valids2};

  // [[1, 2, 3], [], [4, 5], NULL, [NULL, 6, NULL]]
  // [[7, 8]]
  // []
  // NULL
  lcw col5{
    {{{{1, 2, 3}, {}, {4, 5}, {}, {{0, 6, 0}, valids}}, valids2}, {{7, 8}}, lcw{}, lcw{lcw{}}},
    valids2};

  using strlcw = cudf::test::lists_column_wrapper<cudf::string_view>;
  cudf::test::lists_column_wrapper<cudf::string_view> col6{
    {{"Monday", "Monday", "Friday"}, {}, {"Monday", "Friday"}, {}, {"Sunday", "Funday"}},
    {{"bee", "sting"}},
    strlcw{},
    strlcw{strlcw{}}};

  // [[[NULL,2,NULL,4]], [[NULL,6,NULL], [8,9]]]
  // [NULL, [[13],[14,15,16]],  NULL]
  // [NULL, [], NULL, [[]]]
  // NULL
  lcw col7{{
             {{{{1, 2, 3, 4}, valids}}, {{{5, 6, 7}, valids}, {8, 9}}},
             {{{{10, 11}, {12}}, {{13}, {14, 15, 16}}, {{17, 18}}}, valids},
             {{lcw{lcw{}}, lcw{}, lcw{}, lcw{lcw{}}}, valids},
             lcw{lcw{lcw{}}},
           },
           valids2};

  cudf_io::table_metadata expected_metadata;
  expected_metadata.column_names.emplace_back("col_list_int_0");
  expected_metadata.column_names.emplace_back("col_list_list_int_1");
  expected_metadata.column_names.emplace_back("col_list_list_int_nullable_2");
  expected_metadata.column_names.emplace_back("col_list_list_nullable_double_nullable_3");
  // expected_metadata.column_names.emplace_back("col_list_list_uint16_4");
  expected_metadata.column_names.emplace_back("col_list_nullable_list_nullable_int_nullable_5");
  expected_metadata.column_names.emplace_back("col_list_list_string_6");
  expected_metadata.column_names.emplace_back("col_list_list_list_7");

  table_view expected({col0, col1, col2, col3, /* col4, */ col5, col6, col7});

  auto filepath = temp_env->get_temp_filepath("ListColumn.parquet");
  auto out_opts = cudf_io::parquet_writer_options::builder(cudf_io::sink_info{filepath}, expected)
                    .metadata(&expected_metadata)
                    .compression(cudf_io::compression_type::NONE);

  cudf_io::write_parquet(out_opts);

  auto in_opts = cudf_io::parquet_reader_options::builder(cudf_io::source_info{filepath});
  auto result  = cudf_io::read_parquet(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
  EXPECT_EQ(expected_metadata.column_names, result.metadata.column_names);
}

TEST_F(ParquetWriterTest, MultiIndex)
{
  constexpr auto num_rows = 100;

  auto col1_data = random_values<int8_t>(num_rows);
  auto col2_data = random_values<int16_t>(num_rows);
  auto col3_data = random_values<int32_t>(num_rows);
  auto col4_data = random_values<float>(num_rows);
  auto col5_data = random_values<double>(num_rows);
  auto validity  = cudf::test::make_counting_transform_iterator(0, [](auto i) { return true; });

  column_wrapper<int8_t> col1{col1_data.begin(), col1_data.end(), validity};
  column_wrapper<int16_t> col2{col2_data.begin(), col2_data.end(), validity};
  column_wrapper<int32_t> col3{col3_data.begin(), col3_data.end(), validity};
  column_wrapper<float> col4{col4_data.begin(), col4_data.end(), validity};
  column_wrapper<double> col5{col5_data.begin(), col5_data.end(), validity};

  cudf_io::table_metadata expected_metadata;
  expected_metadata.column_names.emplace_back("int8s");
  expected_metadata.column_names.emplace_back("int16s");
  expected_metadata.column_names.emplace_back("int32s");
  expected_metadata.column_names.emplace_back("floats");
  expected_metadata.column_names.emplace_back("doubles");
  expected_metadata.user_data.insert(
    {"pandas", "\"index_columns\": [\"floats\", \"doubles\"], \"column1\": [\"int8s\"]"});

  std::vector<std::unique_ptr<column>> cols;
  cols.push_back(col1.release());
  cols.push_back(col2.release());
  cols.push_back(col3.release());
  cols.push_back(col4.release());
  cols.push_back(col5.release());
  auto expected = std::make_unique<table>(std::move(cols));
  EXPECT_EQ(5, expected->num_columns());

  auto filepath = temp_env->get_temp_filepath("MultiIndex.parquet");
  cudf_io::parquet_writer_options out_opts =
    cudf_io::parquet_writer_options::builder(cudf_io::sink_info{filepath}, expected->view())
      .metadata(&expected_metadata);
  cudf_io::write_parquet(out_opts);

  cudf_io::parquet_reader_options in_opts =
    cudf_io::parquet_reader_options::builder(cudf_io::source_info{filepath})
      .use_pandas_metadata(true)
      .columns({"int8s", "int16s", "int32s"});
  auto result = cudf_io::read_parquet(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), result.tbl->view());
  EXPECT_EQ(expected_metadata.column_names, result.metadata.column_names);
}

TEST_F(ParquetWriterTest, HostBuffer)
{
  constexpr auto num_rows = 100 << 10;
  const auto seq_col      = random_values<int>(num_rows);
  const auto validity =
    cudf::test::make_counting_transform_iterator(0, [](auto i) { return true; });
  column_wrapper<int> col{seq_col.begin(), seq_col.end(), validity};

  cudf_io::table_metadata expected_metadata;
  expected_metadata.column_names.emplace_back("col_other");

  std::vector<std::unique_ptr<column>> cols;
  cols.push_back(col.release());
  const auto expected = std::make_unique<table>(std::move(cols));
  EXPECT_EQ(1, expected->num_columns());

  std::vector<char> out_buffer;
  cudf_io::parquet_writer_options out_opts =
    cudf_io::parquet_writer_options::builder(cudf_io::sink_info(&out_buffer), expected->view())
      .metadata(&expected_metadata);
  cudf_io::write_parquet(out_opts);
  cudf_io::parquet_reader_options in_opts = cudf_io::parquet_reader_options::builder(
    cudf_io::source_info(out_buffer.data(), out_buffer.size()));
  const auto result = cudf_io::read_parquet(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), result.tbl->view());
  EXPECT_EQ(expected_metadata.column_names, result.metadata.column_names);
}

TEST_F(ParquetWriterTest, NonNullable)
{
  srand(31337);
  auto expected = create_random_fixed_table<int>(9, 9, false);

  auto filepath = temp_env->get_temp_filepath("NonNullable.parquet");
  cudf_io::parquet_writer_options args =
    cudf_io::parquet_writer_options::builder(cudf_io::sink_info{filepath}, *expected);
  cudf_io::write_parquet(args);

  cudf_io::parquet_reader_options read_opts =
    cudf_io::parquet_reader_options::builder(cudf_io::source_info{filepath});
  auto result = cudf_io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *expected);
}

// custom data sink that supports device writes. uses plain file io.
class custom_test_data_sink : public cudf::io::data_sink {
 public:
  explicit custom_test_data_sink(std::string const& filepath)
  {
    outfile_.open(filepath, std::ios::out | std::ios::binary | std::ios::trunc);
    CUDF_EXPECTS(outfile_.is_open(), "Cannot open output file");
  }

  virtual ~custom_test_data_sink() { flush(); }

  void host_write(void const* data, size_t size) override
  {
    outfile_.write(static_cast<char const*>(data), size);
  }

  bool supports_device_write() const override { return true; }

  void device_write(void const* gpu_data, size_t size, cudaStream_t stream)
  {
    char* ptr = nullptr;
    CUDA_TRY(cudaMallocHost(&ptr, size));
    CUDA_TRY(cudaMemcpyAsync(ptr, gpu_data, size, cudaMemcpyDeviceToHost, stream));
    CUDA_TRY(cudaStreamSynchronize(stream));
    outfile_.write(ptr, size);
    CUDA_TRY(cudaFreeHost(ptr));
  }

  void flush() override { outfile_.flush(); }

  size_t bytes_written() override { return outfile_.tellp(); }

 private:
  std::ofstream outfile_;
};

TEST_F(ParquetWriterTest, CustomDataSink)
{
  auto filepath = temp_env->get_temp_filepath("CustomDataSink.parquet");
  custom_test_data_sink custom_sink(filepath);

  namespace cudf_io = cudf::io;

  srand(31337);
  auto expected = create_random_fixed_table<int>(5, 10, false);

  // write out using the custom sink
  {
    cudf_io::parquet_writer_options args =
      cudf_io::parquet_writer_options::builder(cudf_io::sink_info{&custom_sink}, *expected);
    cudf_io::write_parquet(args);
  }

  // write out using a memmapped sink
  std::vector<char> buf_sink;
  {
    cudf_io::parquet_writer_options args =
      cudf_io::parquet_writer_options::builder(cudf_io::sink_info{&buf_sink}, *expected);
    cudf_io::write_parquet(args);
  }

  // read them back in and make sure everything matches

  cudf_io::parquet_reader_options custom_args =
    cudf_io::parquet_reader_options::builder(cudf_io::source_info{filepath});
  auto custom_tbl = cudf_io::read_parquet(custom_args);
  CUDF_TEST_EXPECT_TABLES_EQUAL(custom_tbl.tbl->view(), expected->view());

  cudf_io::parquet_reader_options buf_args = cudf_io::parquet_reader_options::builder(
    cudf_io::source_info{buf_sink.data(), buf_sink.size()});
  auto buf_tbl = cudf_io::read_parquet(buf_args);
  CUDF_TEST_EXPECT_TABLES_EQUAL(buf_tbl.tbl->view(), expected->view());
}

TEST_F(ParquetWriterTest, DeviceWriteLargeishFile)
{
  auto filepath = temp_env->get_temp_filepath("DeviceWriteLargeishFile.parquet");
  custom_test_data_sink custom_sink(filepath);

  namespace cudf_io = cudf::io;

  // exercises multiple rowgroups
  srand(31337);
  auto expected = create_random_fixed_table<int>(4, 4 * 1024 * 1024, false);

  // write out using the custom sink (which uses device writes)
  cudf_io::parquet_writer_options args =
    cudf_io::parquet_writer_options::builder(cudf_io::sink_info{&custom_sink}, *expected);
  cudf_io::write_parquet(args);

  cudf_io::parquet_reader_options custom_args =
    cudf_io::parquet_reader_options::builder(cudf_io::source_info{filepath});
  auto custom_tbl = cudf_io::read_parquet(custom_args);
  CUDF_TEST_EXPECT_TABLES_EQUAL(custom_tbl.tbl->view(), expected->view());
}
template <typename T>
std::string create_parquet_file(int num_cols)
{
  srand(31337);
  auto const table = create_random_fixed_table<T>(num_cols, 10, true);
  auto const filepath =
    temp_env->get_temp_filepath(typeid(T).name() + std::to_string(num_cols) + ".parquet");
  cudf_io::parquet_writer_options const out_opts =
    cudf_io::parquet_writer_options::builder(cudf_io::sink_info{filepath}, table->view());
  cudf_io::write_parquet(out_opts);
  return filepath;
}

TEST_F(ParquetWriterTest, MultipleMismatchedSources)
{
  auto const int5file = create_parquet_file<int>(5);
  {
    auto const float5file = create_parquet_file<float>(5);
    std::vector<std::string> files{int5file, float5file};
    cudf_io::parquet_reader_options const read_opts =
      cudf_io::parquet_reader_options::builder(cudf_io::source_info{files});
    EXPECT_THROW(cudf_io::read_parquet(read_opts), cudf::logic_error);
  }
  {
    auto const int10file = create_parquet_file<int>(10);
    std::vector<std::string> files{int5file, int10file};
    cudf_io::parquet_reader_options const read_opts =
      cudf_io::parquet_reader_options::builder(cudf_io::source_info{files});
    EXPECT_THROW(cudf_io::read_parquet(read_opts), cudf::logic_error);
  }
}

TEST_F(ParquetChunkedWriterTest, SingleTable)
{
  srand(31337);
  auto table1 = create_random_fixed_table<int>(5, 5, true);

  auto filepath = temp_env->get_temp_filepath("ChunkedSingle.parquet");
  cudf_io::chunked_parquet_writer_options args =
    cudf_io::chunked_parquet_writer_options::builder(cudf_io::sink_info{filepath});
  auto state = cudf_io::write_parquet_chunked_begin(args);
  cudf_io::write_parquet_chunked(*table1, state);
  cudf_io::write_parquet_chunked_end(state);

  cudf_io::parquet_reader_options read_opts =
    cudf_io::parquet_reader_options::builder(cudf_io::source_info{filepath});
  auto result = cudf_io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *table1);
}

TEST_F(ParquetChunkedWriterTest, SimpleTable)
{
  srand(31337);
  auto table1 = create_random_fixed_table<int>(5, 5, true);
  auto table2 = create_random_fixed_table<int>(5, 5, true);

  auto full_table = cudf::concatenate({*table1, *table2});

  auto filepath = temp_env->get_temp_filepath("ChunkedSimple.parquet");
  cudf_io::chunked_parquet_writer_options args =
    cudf_io::chunked_parquet_writer_options::builder(cudf_io::sink_info{filepath});
  auto state = cudf_io::write_parquet_chunked_begin(args);
  cudf_io::write_parquet_chunked(*table1, state);
  cudf_io::write_parquet_chunked(*table2, state);
  cudf_io::write_parquet_chunked_end(state);

  cudf_io::parquet_reader_options read_opts =
    cudf_io::parquet_reader_options::builder(cudf_io::source_info{filepath});
  auto result = cudf_io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *full_table);
}

TEST_F(ParquetChunkedWriterTest, LargeTables)
{
  srand(31337);
  auto table1 = create_random_fixed_table<int>(512, 4096, true);
  auto table2 = create_random_fixed_table<int>(512, 8192, true);

  auto full_table = cudf::concatenate({*table1, *table2});

  auto filepath = temp_env->get_temp_filepath("ChunkedLarge.parquet");
  cudf_io::chunked_parquet_writer_options args =
    cudf_io::chunked_parquet_writer_options::builder(cudf_io::sink_info{filepath});
  auto state = cudf_io::write_parquet_chunked_begin(args);
  cudf_io::write_parquet_chunked(*table1, state);
  cudf_io::write_parquet_chunked(*table2, state);
  auto md = cudf_io::write_parquet_chunked_end(state);
  CUDF_EXPECTS(!md, "The return value should be null.");

  cudf_io::parquet_reader_options read_opts =
    cudf_io::parquet_reader_options::builder(cudf_io::source_info{filepath});
  auto result = cudf_io::read_parquet(read_opts);

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
  cudf_io::chunked_parquet_writer_options args =
    cudf_io::chunked_parquet_writer_options::builder(cudf_io::sink_info{filepath});
  auto state = cudf_io::write_parquet_chunked_begin(args);
  std::for_each(table_views.begin(), table_views.end(), [&state](table_view const& tbl) {
    cudf_io::write_parquet_chunked(tbl, state);
  });
  auto md = cudf_io::write_parquet_chunked_end(state, true, "dummy/path");
  CUDF_EXPECTS(md, "The returned metadata should not be null.");

  cudf_io::parquet_reader_options read_opts =
    cudf_io::parquet_reader_options::builder(cudf_io::source_info{filepath});
  auto result = cudf_io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *expected);
}

TEST_F(ParquetChunkedWriterTest, Strings)
{
  std::vector<std::unique_ptr<cudf::column>> cols;

  bool mask1[] = {1, 1, 0, 1, 1, 1, 1};
  std::vector<const char*> h_strings1{"four", "score", "and", "seven", "years", "ago", "abcdefgh"};
  cudf::test::strings_column_wrapper strings1(h_strings1.begin(), h_strings1.end(), mask1);
  cols.push_back(strings1.release());
  cudf::table tbl1(std::move(cols));

  bool mask2[] = {0, 1, 1, 1, 1, 1, 1};
  std::vector<const char*> h_strings2{"ooooo", "ppppppp", "fff", "j", "cccc", "bbb", "zzzzzzzzzzz"};
  cudf::test::strings_column_wrapper strings2(h_strings2.begin(), h_strings2.end(), mask2);
  cols.push_back(strings2.release());
  cudf::table tbl2(std::move(cols));

  auto expected = cudf::concatenate({tbl1, tbl2});

  auto filepath = temp_env->get_temp_filepath("ChunkedStrings.parquet");
  cudf_io::chunked_parquet_writer_options args =
    cudf_io::chunked_parquet_writer_options::builder(cudf_io::sink_info{filepath});
  auto state = cudf_io::write_parquet_chunked_begin(args);
  cudf_io::write_parquet_chunked(tbl1, state);
  cudf_io::write_parquet_chunked(tbl2, state);
  cudf_io::write_parquet_chunked_end(state);

  cudf_io::parquet_reader_options read_opts =
    cudf_io::parquet_reader_options::builder(cudf_io::source_info{filepath});
  auto result = cudf_io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *expected);
}

TEST_F(ParquetChunkedWriterTest, MismatchedTypes)
{
  srand(31337);
  auto table1 = create_random_fixed_table<int>(4, 4, true);
  auto table2 = create_random_fixed_table<float>(4, 4, true);

  auto filepath = temp_env->get_temp_filepath("ChunkedMismatchedTypes.parquet");
  cudf_io::chunked_parquet_writer_options args =
    cudf_io::chunked_parquet_writer_options::builder(cudf_io::sink_info{filepath});
  auto state = cudf_io::write_parquet_chunked_begin(args);
  cudf_io::write_parquet_chunked(*table1, state);
  EXPECT_THROW(cudf_io::write_parquet_chunked(*table2, state), cudf::logic_error);
  cudf_io::write_parquet_chunked_end(state);
}

TEST_F(ParquetChunkedWriterTest, MismatchedStructure)
{
  srand(31337);
  auto table1 = create_random_fixed_table<int>(4, 4, true);
  auto table2 = create_random_fixed_table<float>(3, 4, true);

  auto filepath = temp_env->get_temp_filepath("ChunkedMismatchedStructure.parquet");
  cudf_io::chunked_parquet_writer_options args =
    cudf_io::chunked_parquet_writer_options::builder(cudf_io::sink_info{filepath});
  auto state = cudf_io::write_parquet_chunked_begin(args);
  cudf_io::write_parquet_chunked(*table1, state);
  EXPECT_THROW(cudf_io::write_parquet_chunked(*table2, state), cudf::logic_error);
  cudf_io::write_parquet_chunked_end(state);
}

TEST_F(ParquetChunkedWriterTest, ReadRowGroups)
{
  srand(31337);
  auto table1 = create_random_fixed_table<int>(5, 5, true);
  auto table2 = create_random_fixed_table<int>(5, 5, true);

  auto full_table = cudf::concatenate({*table2, *table1, *table2});

  auto filepath = temp_env->get_temp_filepath("ChunkedRowGroups.parquet");
  cudf_io::chunked_parquet_writer_options args =
    cudf_io::chunked_parquet_writer_options::builder(cudf_io::sink_info{filepath});
  auto state = cudf_io::write_parquet_chunked_begin(args);
  cudf_io::write_parquet_chunked(*table1, state);
  cudf_io::write_parquet_chunked(*table2, state);
  cudf_io::write_parquet_chunked_end(state);

  cudf_io::parquet_reader_options read_opts =
    cudf_io::parquet_reader_options::builder(cudf_io::source_info{filepath})
      .row_groups({{1, 0, 1}});
  auto result = cudf_io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *full_table);
}

TEST_F(ParquetChunkedWriterTest, ReadRowGroupsError)
{
  srand(31337);
  auto table1 = create_random_fixed_table<int>(5, 5, true);

  auto filepath = temp_env->get_temp_filepath("ChunkedRowGroupsError.parquet");
  cudf_io::chunked_parquet_writer_options args =
    cudf_io::chunked_parquet_writer_options::builder(cudf_io::sink_info{filepath});
  auto state = cudf_io::write_parquet_chunked_begin(args);
  cudf_io::write_parquet_chunked(*table1, state);
  cudf_io::write_parquet_chunked_end(state);

  cudf_io::parquet_reader_options read_opts =
    cudf_io::parquet_reader_options::builder(cudf_io::source_info{filepath}).row_groups({{0, 1}});
  EXPECT_THROW(cudf_io::read_parquet(read_opts), cudf::logic_error);
  read_opts.set_row_groups({{-1}});
  EXPECT_THROW(cudf_io::read_parquet(read_opts), cudf::logic_error);
  read_opts.set_row_groups({{0}, {0}});
  EXPECT_THROW(cudf_io::read_parquet(read_opts), cudf::logic_error);
}

TYPED_TEST(ParquetChunkedWriterNumericTypeTest, UnalignedSize)
{
  // write out two 31 row tables and make sure they get
  // read back with all their validity bits in the right place

  using T = TypeParam;

  int num_els = 31;
  std::vector<std::unique_ptr<cudf::column>> cols;

  bool mask[] = {0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

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

  auto expected = cudf::concatenate({tbl1, tbl2});

  auto filepath = temp_env->get_temp_filepath("ChunkedUnalignedSize.parquet");
  cudf_io::chunked_parquet_writer_options args =
    cudf_io::chunked_parquet_writer_options::builder(cudf_io::sink_info{filepath});
  auto state = cudf_io::write_parquet_chunked_begin(args);
  cudf_io::write_parquet_chunked(tbl1, state);
  cudf_io::write_parquet_chunked(tbl2, state);
  cudf_io::write_parquet_chunked_end(state);

  cudf_io::parquet_reader_options read_opts =
    cudf_io::parquet_reader_options::builder(cudf_io::source_info{filepath});
  auto result = cudf_io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *expected);
}

TYPED_TEST(ParquetChunkedWriterNumericTypeTest, UnalignedSize2)
{
  // write out two 33 row tables and make sure they get
  // read back with all their validity bits in the right place

  using T = TypeParam;

  int num_els = 33;
  std::vector<std::unique_ptr<cudf::column>> cols;

  bool mask[] = {0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

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

  auto expected = cudf::concatenate({tbl1, tbl2});

  auto filepath = temp_env->get_temp_filepath("ChunkedUnalignedSize2.parquet");
  cudf_io::chunked_parquet_writer_options args =
    cudf_io::chunked_parquet_writer_options::builder(cudf_io::sink_info{filepath});
  auto state = cudf_io::write_parquet_chunked_begin(args);
  cudf_io::write_parquet_chunked(tbl1, state);
  cudf_io::write_parquet_chunked(tbl2, state);
  cudf_io::write_parquet_chunked_end(state);

  cudf_io::parquet_reader_options read_opts =
    cudf_io::parquet_reader_options::builder(cudf_io::source_info{filepath});
  auto result = cudf_io::read_parquet(read_opts);

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

  bool supports_device_write() const override { return supports_device_writes; }

  void device_write(void const* gpu_data, size_t size, cudaStream_t stream)
  {
    char* ptr = nullptr;
    CUDA_TRY(cudaMallocHost(&ptr, size));
    CUDA_TRY(cudaMemcpyAsync(ptr, gpu_data, size, cudaMemcpyDeviceToHost, stream));
    CUDA_TRY(cudaStreamSynchronize(stream));
    mm_writer->host_write(ptr, size);
    CUDA_TRY(cudaFreeHost(ptr));
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

  namespace cudf_io = cudf::io;

  // exercises multiple rowgroups
  srand(31337);
  auto expected = create_random_fixed_table<int>(16, 4 * 1024 * 1024, false);

  // write out using the custom sink (which uses device writes)
  cudf_io::parquet_writer_options args =
    cudf_io::parquet_writer_options::builder(cudf_io::sink_info{&custom_sink}, *expected);
  cudf_io::write_parquet(args);

  cudf_io::parquet_reader_options custom_args =
    cudf_io::parquet_reader_options::builder(cudf_io::source_info{mm_buf.data(), mm_buf.size()});
  auto custom_tbl = cudf_io::read_parquet(custom_args);
  CUDF_TEST_EXPECT_TABLES_EQUAL(custom_tbl.tbl->view(), expected->view());
}

TEST_F(ParquetWriterStressTest, LargeTableGoodCompression)
{
  std::vector<char> mm_buf;
  mm_buf.reserve(4 * 1024 * 1024 * 16);
  custom_test_memmap_sink<false> custom_sink(&mm_buf);

  namespace cudf_io = cudf::io;

  // exercises multiple rowgroups
  srand(31337);
  auto expected = create_compressible_fixed_table<int>(16, 4 * 1024 * 1024, 128 * 1024, false);

  // write out using the custom sink (which uses device writes)
  cudf_io::parquet_writer_options args =
    cudf_io::parquet_writer_options::builder(cudf_io::sink_info{&custom_sink}, *expected);
  cudf_io::write_parquet(args);

  cudf_io::parquet_reader_options custom_args =
    cudf_io::parquet_reader_options::builder(cudf_io::source_info{mm_buf.data(), mm_buf.size()});
  auto custom_tbl = cudf_io::read_parquet(custom_args);
  CUDF_TEST_EXPECT_TABLES_EQUAL(custom_tbl.tbl->view(), expected->view());
}

TEST_F(ParquetWriterStressTest, LargeTableWithValids)
{
  std::vector<char> mm_buf;
  mm_buf.reserve(4 * 1024 * 1024 * 16);
  custom_test_memmap_sink<false> custom_sink(&mm_buf);

  namespace cudf_io = cudf::io;

  // exercises multiple rowgroups
  srand(31337);
  auto expected = create_compressible_fixed_table<int>(16, 4 * 1024 * 1024, 6, true);

  // write out using the custom sink (which uses device writes)
  cudf_io::parquet_writer_options args =
    cudf_io::parquet_writer_options::builder(cudf_io::sink_info{&custom_sink}, *expected);
  cudf_io::write_parquet(args);

  cudf_io::parquet_reader_options custom_args =
    cudf_io::parquet_reader_options::builder(cudf_io::source_info{mm_buf.data(), mm_buf.size()});
  auto custom_tbl = cudf_io::read_parquet(custom_args);
  CUDF_TEST_EXPECT_TABLES_EQUAL(custom_tbl.tbl->view(), expected->view());
}

TEST_F(ParquetWriterStressTest, DeviceWriteLargeTableWeakCompression)
{
  std::vector<char> mm_buf;
  mm_buf.reserve(4 * 1024 * 1024 * 16);
  custom_test_memmap_sink<true> custom_sink(&mm_buf);

  namespace cudf_io = cudf::io;

  // exercises multiple rowgroups
  srand(31337);
  auto expected = create_random_fixed_table<int>(16, 4 * 1024 * 1024, false);

  // write out using the custom sink (which uses device writes)
  cudf_io::parquet_writer_options args =
    cudf_io::parquet_writer_options::builder(cudf_io::sink_info{&custom_sink}, *expected);
  cudf_io::write_parquet(args);

  cudf_io::parquet_reader_options custom_args =
    cudf_io::parquet_reader_options::builder(cudf_io::source_info{mm_buf.data(), mm_buf.size()});
  auto custom_tbl = cudf_io::read_parquet(custom_args);
  CUDF_TEST_EXPECT_TABLES_EQUAL(custom_tbl.tbl->view(), expected->view());
}

TEST_F(ParquetWriterStressTest, DeviceWriteLargeTableGoodCompression)
{
  std::vector<char> mm_buf;
  mm_buf.reserve(4 * 1024 * 1024 * 16);
  custom_test_memmap_sink<true> custom_sink(&mm_buf);

  namespace cudf_io = cudf::io;

  // exercises multiple rowgroups
  srand(31337);
  auto expected = create_compressible_fixed_table<int>(16, 4 * 1024 * 1024, 128 * 1024, false);

  // write out using the custom sink (which uses device writes)
  cudf_io::parquet_writer_options args =
    cudf_io::parquet_writer_options::builder(cudf_io::sink_info{&custom_sink}, *expected);
  cudf_io::write_parquet(args);

  cudf_io::parquet_reader_options custom_args =
    cudf_io::parquet_reader_options::builder(cudf_io::source_info{mm_buf.data(), mm_buf.size()});
  auto custom_tbl = cudf_io::read_parquet(custom_args);
  CUDF_TEST_EXPECT_TABLES_EQUAL(custom_tbl.tbl->view(), expected->view());
}

TEST_F(ParquetWriterStressTest, DeviceWriteLargeTableWithValids)
{
  std::vector<char> mm_buf;
  mm_buf.reserve(4 * 1024 * 1024 * 16);
  custom_test_memmap_sink<true> custom_sink(&mm_buf);

  namespace cudf_io = cudf::io;

  // exercises multiple rowgroups
  srand(31337);
  auto expected = create_compressible_fixed_table<int>(16, 4 * 1024 * 1024, 6, true);

  // write out using the custom sink (which uses device writes)
  cudf_io::parquet_writer_options args =
    cudf_io::parquet_writer_options::builder(cudf_io::sink_info{&custom_sink}, *expected);
  cudf_io::write_parquet(args);

  cudf_io::parquet_reader_options custom_args =
    cudf_io::parquet_reader_options::builder(cudf_io::source_info{mm_buf.data(), mm_buf.size()});
  auto custom_tbl = cudf_io::read_parquet(custom_args);
  CUDF_TEST_EXPECT_TABLES_EQUAL(custom_tbl.tbl->view(), expected->view());
}

TEST_F(ParquetReaderTest, UserBounds)
{
  // trying to read more rows than there are should result in
  // receiving the properly capped # of rows
  {
    srand(31337);
    auto expected = create_random_fixed_table<int>(4, 4, false);

    auto filepath = temp_env->get_temp_filepath("TooManyRows.parquet");
    cudf_io::parquet_writer_options args =
      cudf_io::parquet_writer_options::builder(cudf_io::sink_info{filepath}, *expected);
    cudf_io::write_parquet(args);

    // attempt to read more rows than there actually are
    cudf_io::parquet_reader_options read_opts =
      cudf_io::parquet_reader_options::builder(cudf_io::source_info{filepath}).num_rows(16);
    auto result = cudf_io::read_parquet(read_opts);

    // we should only get back 4 rows
    EXPECT_EQ(result.tbl->view().column(0).size(), 4);
  }

  // trying to read past the end of the # of actual rows should result
  // in empty columns.
  {
    srand(31337);
    auto expected = create_random_fixed_table<int>(4, 4, false);

    auto filepath = temp_env->get_temp_filepath("PastBounds.parquet");
    cudf_io::parquet_writer_options args =
      cudf_io::parquet_writer_options::builder(cudf_io::sink_info{filepath}, *expected);
    cudf_io::write_parquet(args);

    // attempt to read more rows than there actually are
    cudf_io::parquet_reader_options read_opts =
      cudf_io::parquet_reader_options::builder(cudf_io::source_info{filepath}).skip_rows(4);
    auto result = cudf_io::read_parquet(read_opts);

    // we should get empty columns back
    EXPECT_EQ(result.tbl->view().num_columns(), 4);
    EXPECT_EQ(result.tbl->view().column(0).size(), 0);
  }

  // trying to read 0 rows should result in reading the whole file
  // at the moment we get back 4.  when that bug gets fixed, this
  // test can be flipped.
  {
    srand(31337);
    auto expected = create_random_fixed_table<int>(4, 4, false);

    auto filepath = temp_env->get_temp_filepath("ZeroRows.parquet");
    cudf_io::parquet_writer_options args =
      cudf_io::parquet_writer_options::builder(cudf_io::sink_info{filepath}, *expected);
    cudf_io::write_parquet(args);

    // attempt to read more rows than there actually are
    cudf_io::parquet_reader_options read_opts =
      cudf_io::parquet_reader_options::builder(cudf_io::source_info{filepath}).num_rows(0);
    auto result = cudf_io::read_parquet(read_opts);

    EXPECT_EQ(result.tbl->view().num_columns(), 4);
    EXPECT_EQ(result.tbl->view().column(0).size(), 0);
  }

  // trying to read 0 rows past the end of the # of actual rows should result
  // in empty columns.
  {
    srand(31337);
    auto expected = create_random_fixed_table<int>(4, 4, false);

    auto filepath = temp_env->get_temp_filepath("ZeroRowsPastBounds.parquet");
    cudf_io::parquet_writer_options args =
      cudf_io::parquet_writer_options::builder(cudf_io::sink_info{filepath}, *expected);
    cudf_io::write_parquet(args);

    // attempt to read more rows than there actually are
    cudf_io::parquet_reader_options read_opts =
      cudf_io::parquet_reader_options::builder(cudf_io::source_info{filepath})
        .skip_rows(4)
        .num_rows(0);
    auto result = cudf_io::read_parquet(read_opts);

    // we should get empty columns back
    EXPECT_EQ(result.tbl->view().num_columns(), 4);
    EXPECT_EQ(result.tbl->view().column(0).size(), 0);
  }
}

TEST_F(ParquetReaderTest, ReorderedColumns)
{
  {
    auto a = cudf::test::strings_column_wrapper{{"a", "", "c"}, {true, false, true}};
    auto b = cudf::test::fixed_width_column_wrapper<int>{1, 2, 3};

    cudf::table_view tbl{{a, b}};
    auto filepath = temp_env->get_temp_filepath("ReorderedColumns.parquet");
    cudf_io::table_metadata md;
    md.column_names.push_back("a");
    md.column_names.push_back("b");
    cudf_io::parquet_writer_options opts =
      cudf_io::parquet_writer_options::builder(cudf_io::sink_info{filepath}, tbl).metadata(&md);
    cudf_io::write_parquet(opts);

    // read them out of order
    cudf_io::parquet_reader_options read_opts =
      cudf_io::parquet_reader_options::builder(cudf_io::source_info{filepath}).columns({"b", "a"});
    auto result = cudf_io::read_parquet(read_opts);

    cudf::test::expect_columns_equal(result.tbl->view().column(0), b);
    cudf::test::expect_columns_equal(result.tbl->view().column(1), a);
  }

  {
    auto a = cudf::test::fixed_width_column_wrapper<int>{1, 2, 3};
    auto b = cudf::test::strings_column_wrapper{{"a", "", "c"}, {true, false, true}};

    cudf::table_view tbl{{a, b}};
    auto filepath = temp_env->get_temp_filepath("ReorderedColumns2.parquet");
    cudf_io::table_metadata md;
    md.column_names.push_back("a");
    md.column_names.push_back("b");
    cudf_io::parquet_writer_options opts =
      cudf_io::parquet_writer_options::builder(cudf_io::sink_info{filepath}, tbl).metadata(&md);
    cudf_io::write_parquet(opts);

    // read them out of order
    cudf_io::parquet_reader_options read_opts =
      cudf_io::parquet_reader_options::builder(cudf_io::source_info{filepath}).columns({"b", "a"});
    auto result = cudf_io::read_parquet(read_opts);

    cudf::test::expect_columns_equal(result.tbl->view().column(0), b);
    cudf::test::expect_columns_equal(result.tbl->view().column(1), a);
  }

  auto a = cudf::test::fixed_width_column_wrapper<int>{1, 2, 3, 10, 20, 30};
  auto b = cudf::test::strings_column_wrapper{{"a", "", "c", "cats", "dogs", "owls"},
                                              {true, false, true, true, false, true}};
  auto c = cudf::test::fixed_width_column_wrapper<int>{{15, 16, 17, 25, 26, 32},
                                                       {false, true, true, true, true, false}};
  auto d = cudf::test::strings_column_wrapper{"ducks", "sheep", "cows", "fish", "birds", "ants"};

  cudf::table_view tbl{{a, b, c, d}};
  auto filepath = temp_env->get_temp_filepath("ReorderedColumns3.parquet");
  cudf_io::table_metadata md;
  md.column_names.push_back("a");
  md.column_names.push_back("b");
  md.column_names.push_back("c");
  md.column_names.push_back("d");
  cudf_io::parquet_writer_options opts =
    cudf_io::parquet_writer_options::builder(cudf_io::sink_info{filepath}, tbl).metadata(&md);
  cudf_io::write_parquet(opts);

  {
    // read them out of order
    cudf_io::parquet_reader_options read_opts =
      cudf_io::parquet_reader_options::builder(cudf_io::source_info{filepath})
        .columns({"d", "a", "b", "c"});
    auto result = cudf_io::read_parquet(read_opts);

    cudf::test::expect_columns_equal(result.tbl->view().column(0), d);
    cudf::test::expect_columns_equal(result.tbl->view().column(1), a);
    cudf::test::expect_columns_equal(result.tbl->view().column(2), b);
    cudf::test::expect_columns_equal(result.tbl->view().column(3), c);
  }

  {
    // read them out of order
    cudf_io::parquet_reader_options read_opts =
      cudf_io::parquet_reader_options::builder(cudf_io::source_info{filepath})
        .columns({"c", "d", "a", "b"});
    auto result = cudf_io::read_parquet(read_opts);

    cudf::test::expect_columns_equal(result.tbl->view().column(0), c);
    cudf::test::expect_columns_equal(result.tbl->view().column(1), d);
    cudf::test::expect_columns_equal(result.tbl->view().column(2), a);
    cudf::test::expect_columns_equal(result.tbl->view().column(3), b);
  }

  {
    // read them out of order
    cudf_io::parquet_reader_options read_opts =
      cudf_io::parquet_reader_options::builder(cudf_io::source_info{filepath})
        .columns({"d", "c", "b", "a"});
    auto result = cudf_io::read_parquet(read_opts);

    cudf::test::expect_columns_equal(result.tbl->view().column(0), d);
    cudf::test::expect_columns_equal(result.tbl->view().column(1), c);
    cudf::test::expect_columns_equal(result.tbl->view().column(2), b);
    cudf::test::expect_columns_equal(result.tbl->view().column(3), a);
  }
}

CUDF_TEST_PROGRAM_MAIN()
