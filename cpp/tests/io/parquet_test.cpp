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

#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/io/data_sink.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <rmm/cuda_stream_view.hpp>

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
  auto col6_vals = random_values<int32_t>(num_rows);
  auto col7_vals = random_values<int64_t>(num_rows);
  auto col6_data = cudf::test::make_counting_transform_iterator(0, [col6_vals](auto i) {
    return numeric::decimal32{col6_vals[i], numeric::scale_type{5}};
  });
  auto col7_data = cudf::test::make_counting_transform_iterator(0, [col7_vals](auto i) {
    return numeric::decimal64{col7_vals[i], numeric::scale_type{-5}};
  });
  auto validity  = cudf::test::make_counting_transform_iterator(0, [](auto i) { return true; });

  // column_wrapper<bool> col0{
  //    col0_data.begin(), col0_data.end(), validity};
  column_wrapper<int8_t> col1{col1_data.begin(), col1_data.end(), validity};
  column_wrapper<int16_t> col2{col2_data.begin(), col2_data.end(), validity};
  column_wrapper<int32_t> col3{col3_data.begin(), col3_data.end(), validity};
  column_wrapper<float> col4{col4_data.begin(), col4_data.end(), validity};
  column_wrapper<double> col5{col5_data.begin(), col5_data.end(), validity};
  column_wrapper<numeric::decimal32> col6{col6_data, col6_data + num_rows, validity};
  column_wrapper<numeric::decimal64> col7{col7_data, col7_data + num_rows, validity};

  cudf_io::table_metadata expected_metadata;
  // expected_metadata.column_names.emplace_back("bools");
  expected_metadata.column_names.emplace_back("int8s");
  expected_metadata.column_names.emplace_back("int16s");
  expected_metadata.column_names.emplace_back("int32s");
  expected_metadata.column_names.emplace_back("floats");
  expected_metadata.column_names.emplace_back("doubles");
  expected_metadata.column_names.emplace_back("decimal32s");
  expected_metadata.column_names.emplace_back("decimal64s");

  std::vector<std::unique_ptr<column>> cols;
  // cols.push_back(col0.release());
  cols.push_back(col1.release());
  cols.push_back(col2.release());
  cols.push_back(col3.release());
  cols.push_back(col4.release());
  cols.push_back(col5.release());
  cols.push_back(col6.release());
  cols.push_back(col7.release());
  auto expected = std::make_unique<table>(std::move(cols));
  EXPECT_EQ(7, expected->num_columns());

  auto filepath = temp_env->get_temp_filepath("MultiColumn.parquet");
  cudf_io::parquet_writer_options out_opts =
    cudf_io::parquet_writer_options::builder(cudf_io::sink_info{filepath}, expected->view())
      .metadata(&expected_metadata);
  std::vector<uint8_t> precisions = {10, 20};
  out_opts.set_decimal_precision(precisions);
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
  auto col6_vals = random_values<int32_t>(num_rows);
  auto col7_vals = random_values<int64_t>(num_rows);
  auto col6_data = cudf::test::make_counting_transform_iterator(0, [col6_vals](auto i) {
    return numeric::decimal32{col6_vals[i], numeric::scale_type{-2}};
  });
  auto col7_data = cudf::test::make_counting_transform_iterator(0, [col7_vals](auto i) {
    return numeric::decimal64{col7_vals[i], numeric::scale_type{-8}};
  });
  // auto col0_mask = cudf::test::make_counting_transform_iterator(
  //    0, [](auto i) { return (i % 2); });
  auto col1_mask = cudf::test::make_counting_transform_iterator(0, [](auto i) { return (i < 10); });
  auto col2_mask = cudf::test::make_counting_transform_iterator(0, [](auto i) { return true; });
  auto col3_mask =
    cudf::test::make_counting_transform_iterator(0, [](auto i) { return (i == (num_rows - 1)); });
  auto col4_mask =
    cudf::test::make_counting_transform_iterator(0, [](auto i) { return (i >= 40 || i <= 60); });
  auto col5_mask = cudf::test::make_counting_transform_iterator(0, [](auto i) { return (i > 80); });
  auto col6_mask = cudf::test::make_counting_transform_iterator(0, [](auto i) { return (i % 5); });
  auto col7_mask =
    cudf::test::make_counting_transform_iterator(0, [](auto i) { return (i != 55); });

  // column_wrapper<bool> col0{
  //    col0_data.begin(), col0_data.end(), col0_mask};
  column_wrapper<int8_t> col1{col1_data.begin(), col1_data.end(), col1_mask};
  column_wrapper<int16_t> col2{col2_data.begin(), col2_data.end(), col2_mask};
  column_wrapper<int32_t> col3{col3_data.begin(), col3_data.end(), col3_mask};
  column_wrapper<float> col4{col4_data.begin(), col4_data.end(), col4_mask};
  column_wrapper<double> col5{col5_data.begin(), col5_data.end(), col5_mask};
  column_wrapper<numeric::decimal32> col6{col6_data, col6_data + num_rows, col6_mask};
  column_wrapper<numeric::decimal64> col7{col7_data, col7_data + num_rows, col7_mask};

  cudf_io::table_metadata expected_metadata;
  // expected_metadata.column_names.emplace_back("bools");
  expected_metadata.column_names.emplace_back("int8s");
  expected_metadata.column_names.emplace_back("int16s");
  expected_metadata.column_names.emplace_back("int32s");
  expected_metadata.column_names.emplace_back("floats");
  expected_metadata.column_names.emplace_back("doubles");
  expected_metadata.column_names.emplace_back("decimal32s");
  expected_metadata.column_names.emplace_back("decimal64s");

  std::vector<std::unique_ptr<column>> cols;
  // cols.push_back(col0.release());
  cols.push_back(col1.release());
  cols.push_back(col2.release());
  cols.push_back(col3.release());
  cols.push_back(col4.release());
  cols.push_back(col5.release());
  cols.push_back(col6.release());
  cols.push_back(col7.release());
  auto expected = std::make_unique<table>(std::move(cols));
  EXPECT_EQ(7, expected->num_columns());

  auto filepath = temp_env->get_temp_filepath("MultiColumnWithNulls.parquet");
  cudf_io::parquet_writer_options out_opts =
    cudf_io::parquet_writer_options::builder(cudf_io::sink_info{filepath}, expected->view())
      .metadata(&expected_metadata);
  std::vector<uint8_t> precisions = {9, 20};
  out_opts.set_decimal_precision(precisions);

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

  void device_write(void const* gpu_data, size_t size, rmm::cuda_stream_view stream) override
  {
    char* ptr = nullptr;
    CUDA_TRY(cudaMallocHost(&ptr, size));
    CUDA_TRY(cudaMemcpyAsync(ptr, gpu_data, size, cudaMemcpyDeviceToHost, stream.value()));
    stream.synchronize();
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

TEST_F(ParquetWriterTest, Slice)
{
  auto col =
    cudf::test::fixed_width_column_wrapper<int>{{1, 2, 3, 4, 5}, {true, true, true, false, true}};
  std::vector<cudf::size_type> indices{2, 5};
  std::vector<cudf::column_view> result = cudf::slice(col, indices);
  cudf::table_view tbl{result};

  auto filepath = temp_env->get_temp_filepath("Slice.parquet");
  cudf_io::parquet_writer_options out_opts =
    cudf_io::parquet_writer_options::builder(cudf_io::sink_info{filepath}, tbl);
  cudf_io::write_parquet(out_opts);

  cudf_io::parquet_reader_options in_opts =
    cudf_io::parquet_reader_options::builder(cudf_io::source_info{filepath});
  auto read_table = cudf_io::read_parquet(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(read_table.tbl->view(), tbl);
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

TEST_F(ParquetChunkedWriterTest, ListColumn)
{
  auto valids  = cudf::test::make_counting_transform_iterator(0, [](auto i) { return i % 2; });
  auto valids2 = cudf::test::make_counting_transform_iterator(0, [](auto i) { return i != 3; });

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

  auto expected = cudf::concatenate({tbl0, tbl1});

  auto filepath = temp_env->get_temp_filepath("ChunkedLists.parquet");
  cudf_io::chunked_parquet_writer_options args =
    cudf_io::chunked_parquet_writer_options::builder(cudf_io::sink_info{filepath});
  auto state = cudf_io::write_parquet_chunked_begin(args);
  cudf_io::write_parquet_chunked(tbl0, state);
  cudf_io::write_parquet_chunked(tbl1, state);
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

TEST_F(ParquetChunkedWriterTest, MismatchedStructureList)
{
  auto valids  = cudf::test::make_counting_transform_iterator(0, [](auto i) { return i % 2; });
  auto valids2 = cudf::test::make_counting_transform_iterator(0, [](auto i) { return i != 3; });

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
  cudf_io::chunked_parquet_writer_options args =
    cudf_io::chunked_parquet_writer_options::builder(cudf_io::sink_info{filepath});
  auto state = cudf_io::write_parquet_chunked_begin(args);
  cudf_io::write_parquet_chunked(tbl0, state);
  CUDF_EXPECT_THROW_MESSAGE(cudf_io::write_parquet_chunked(tbl1, state),
                            "Mismatch in schema between multiple calls to write_chunk");
}

TEST_F(ParquetChunkedWriterTest, DifferentNullability)
{
  srand(31337);
  auto table1 = create_random_fixed_table<int>(5, 5, true);
  auto table2 = create_random_fixed_table<int>(5, 5, false);

  auto full_table = cudf::concatenate({*table1, *table2});

  auto filepath = temp_env->get_temp_filepath("ChunkedNullable.parquet");
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

TEST_F(ParquetChunkedWriterTest, ForcedNullability)
{
  srand(31337);
  auto table1 = create_random_fixed_table<int>(5, 5, false);
  auto table2 = create_random_fixed_table<int>(5, 5, false);

  auto full_table = cudf::concatenate({*table1, *table2});

  auto filepath = temp_env->get_temp_filepath("ChunkedNoNullable.parquet");

  cudf::io::table_metadata_with_nullability nullable_metadata;

  // In the absence of prescribed per-column nullability in metadata, the writer assumes the worst
  // and considers all columns nullable. However cudf::concatenate will not force nulls in case no
  // columns are nullable. To get the expected result, we tell the writer the nullability of all
  // columns in advance.
  nullable_metadata.column_nullable.insert(nullable_metadata.column_nullable.begin(), 5, false);

  cudf_io::chunked_parquet_writer_options args =
    cudf_io::chunked_parquet_writer_options::builder(cudf_io::sink_info{filepath})
      .nullable_metadata(&nullable_metadata);
  auto state = cudf_io::write_parquet_chunked_begin(args);
  cudf_io::write_parquet_chunked(*table1, state);
  cudf_io::write_parquet_chunked(*table2, state);
  cudf_io::write_parquet_chunked_end(state);

  cudf_io::parquet_reader_options read_opts =
    cudf_io::parquet_reader_options::builder(cudf_io::source_info{filepath});
  auto result = cudf_io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *full_table);
}

TEST_F(ParquetChunkedWriterTest, ForcedNullabilityList)
{
  srand(31337);

  auto valids  = cudf::test::make_counting_transform_iterator(0, [](auto i) { return i % 2; });
  auto valids2 = cudf::test::make_counting_transform_iterator(0, [](auto i) { return i != 3; });

  using lcw = cudf::test::lists_column_wrapper<int32_t>;

  cudf::io::table_metadata_with_nullability nullable_metadata;

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

  nullable_metadata.column_nullable.push_back(true);   // List is nullable at first (root) level
  nullable_metadata.column_nullable.push_back(false);  // non-nullable at second (leaf) level

  // COL1 (non-nested columns to test proper schema construction)
  size_t num_rows = static_cast<cudf::column_view>(col00).size();
  auto seq_col0   = random_values<int>(num_rows);
  auto seq_col1   = random_values<int>(num_rows);

  column_wrapper<int> col10{seq_col0.begin(), seq_col0.end(), valids};
  column_wrapper<int> col11{seq_col1.begin(), seq_col1.end(), valids2};

  nullable_metadata.column_nullable.push_back(true);

  auto table1 = table_view({col00, col10});
  auto table2 = table_view({col01, col11});

  auto full_table = cudf::concatenate({table1, table2});

  auto filepath = temp_env->get_temp_filepath("ChunkedListNullable.parquet");

  cudf_io::chunked_parquet_writer_options args =
    cudf_io::chunked_parquet_writer_options::builder(cudf_io::sink_info{filepath})
      .nullable_metadata(&nullable_metadata);
  auto state = cudf_io::write_parquet_chunked_begin(args);
  cudf_io::write_parquet_chunked(table1, state);
  cudf_io::write_parquet_chunked(table2, state);
  cudf_io::write_parquet_chunked_end(state);

  cudf_io::parquet_reader_options read_opts =
    cudf_io::parquet_reader_options::builder(cudf_io::source_info{filepath});
  auto result = cudf_io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *full_table);
}

TEST_F(ParquetChunkedWriterTest, WrongNullability)
{
  srand(31337);
  auto table1 = create_random_fixed_table<int>(5, 5, false);

  auto filepath = temp_env->get_temp_filepath("ChunkedWrongNullable.parquet");

  cudf::io::table_metadata_with_nullability nullable_metadata;
  nullable_metadata.column_nullable.insert(nullable_metadata.column_nullable.begin(), 6, true);
  cudf_io::chunked_parquet_writer_options args =
    cudf_io::chunked_parquet_writer_options::builder(cudf_io::sink_info{filepath})
      .nullable_metadata(&nullable_metadata);
  auto state = cudf_io::write_parquet_chunked_begin(args);
  EXPECT_THROW(cudf_io::write_parquet_chunked(*table1, state), cudf::logic_error);

  nullable_metadata.column_nullable.clear();
  nullable_metadata.column_nullable.insert(nullable_metadata.column_nullable.begin(), 4, true);
  cudf_io::chunked_parquet_writer_options args2 =
    cudf_io::chunked_parquet_writer_options::builder(cudf_io::sink_info{filepath})
      .nullable_metadata(&nullable_metadata);
  state = cudf_io::write_parquet_chunked_begin(args2);
  EXPECT_THROW(cudf_io::write_parquet_chunked(*table1, state), cudf::logic_error);
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

TEST_F(ParquetChunkedWriterTest, DecimalWrite)
{
  constexpr cudf::size_type num_rows = 500;
  auto seq_col0                      = random_values<int32_t>(num_rows);
  auto seq_col1                      = random_values<int64_t>(num_rows);

  auto valids = cudf::test::make_counting_transform_iterator(0, [](auto i) { return i % 2 == 0; });

  auto col0 = cudf::test::fixed_point_column_wrapper<int32_t>{
    seq_col0.begin(), seq_col0.end(), valids, numeric::scale_type{5}};
  auto col1 = cudf::test::fixed_point_column_wrapper<int64_t>{
    seq_col1.begin(), seq_col1.end(), valids, numeric::scale_type{-9}};

  auto table = table_view({col0, col1});

  auto filepath = temp_env->get_temp_filepath("DecimalWrite.parquet");
  cudf_io::chunked_parquet_writer_options args =
    cudf_io::chunked_parquet_writer_options::builder(cudf_io::sink_info{filepath});

  // verify failure if no decimal precision given
  auto state = cudf_io::write_parquet_chunked_begin(args);
  EXPECT_THROW(cudf_io::write_parquet_chunked(table, state), cudf::logic_error);

  // verify failure if too small a precision is given
  std::vector<uint8_t> precisions{7, 1};
  args.set_decimal_precision_data(precisions);
  state = cudf_io::write_parquet_chunked_begin(args);
  EXPECT_THROW(cudf_io::write_parquet_chunked(table, state), cudf::logic_error);

  // verify failure if too few precisions given
  precisions.pop_back();
  args.set_decimal_precision_data(precisions);
  state = cudf_io::write_parquet_chunked_begin(args);
  EXPECT_THROW(cudf_io::write_parquet_chunked(table, state), cudf::logic_error);

  // verify failure if too many precisions given
  precisions = {7, 14, 11};
  args.set_decimal_precision_data(precisions);
  state = cudf_io::write_parquet_chunked_begin(args);
  EXPECT_THROW(cudf_io::write_parquet_chunked(table, state), cudf::logic_error);

  // write correctly
  precisions.pop_back();
  args.set_decimal_precision_data(precisions);
  state = cudf_io::write_parquet_chunked_begin(args);
  cudf_io::write_parquet_chunked(table, state);
  cudf_io::write_parquet_chunked_end(state);

  cudf_io::parquet_reader_options read_opts =
    cudf_io::parquet_reader_options::builder(cudf_io::source_info{filepath});
  auto result = cudf_io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, table);
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

  void device_write(void const* gpu_data, size_t size, rmm::cuda_stream_view stream) override
  {
    char* ptr = nullptr;
    CUDA_TRY(cudaMallocHost(&ptr, size));
    CUDA_TRY(cudaMemcpyAsync(ptr, gpu_data, size, cudaMemcpyDeviceToHost, stream.value()));
    stream.synchronize();
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

TEST_F(ParquetReaderTest, DecimalRead)
{
  {
    /* We could add a dataset to include this file, but we don't want tests in cudf to have data.
       This test is a temporary test until python gains the ability to write decimal, so we're
       embedding
       a parquet file directly into the code here to prevent issues with finding the file */
    const unsigned char decimals_parquet[] = {
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
    unsigned int decimals_parquet_len = 2366;

    cudf_io::parquet_reader_options read_opts = cudf_io::parquet_reader_options::builder(
      cudf_io::source_info{reinterpret_cast<const char*>(decimals_parquet), decimals_parquet_len});
    auto result = cudf_io::read_parquet(read_opts);

    auto validity = cudf::test::make_counting_transform_iterator(0, [](auto i) { return i != 50; });

    EXPECT_EQ(result.tbl->view().num_columns(), 3);

    int32_t col0_data[] = {
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
    cudf::test::expect_columns_equal(result.tbl->view().column(0), col0);

    int64_t col1_data[] = {29274040266581,  -17210335917753, -58420730139037,
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

    EXPECT_EQ(static_cast<std::size_t>(result.tbl->view().column(1).size()),
              sizeof(col1_data) / sizeof(col1_data[0]));
    cudf::test::fixed_point_column_wrapper<int64_t> col1(
      std::begin(col1_data), std::end(col1_data), validity, numeric::scale_type{-5});
    cudf::test::expect_columns_equal(result.tbl->view().column(1), col1);

    cudf_io::parquet_reader_options read_strict_opts = read_opts;
    read_strict_opts.set_strict_decimal_types(true);
    read_strict_opts.set_columns({"dec7p4", "dec14p5"});
    EXPECT_NO_THROW(cudf_io::read_parquet(read_strict_opts));
  }
  {
    // dec7p3: Decimal(precision=7, scale=3) backed by FIXED_LENGTH_BYTE_ARRAY(length = 4)
    // dec12p11: Decimal(precision=12, scale=11) backed by FIXED_LENGTH_BYTE_ARRAY(length = 6)
    // dec20p1: Decimal(precision=20, scale=1) backed by FIXED_LENGTH_BYTE_ARRAY(length = 9)
    const unsigned char fixed_len_bytes_decimal_parquet[] = {
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

    cudf_io::parquet_reader_options read_opts =
      cudf_io::parquet_reader_options::builder(cudf_io::source_info{
        reinterpret_cast<const char*>(fixed_len_bytes_decimal_parquet), parquet_len});
    read_opts.set_strict_decimal_types(true);
    read_opts.set_columns({"dec7p3", "dec12p11"});
    auto result = cudf_io::read_parquet(read_opts);
    EXPECT_EQ(result.tbl->view().num_columns(), 2);

    auto validity_c0 =
      cudf::test::make_counting_transform_iterator(0, [](auto i) { return i != 19; });
    int64_t col0_data[] = {6361295, 698632,  7821423, 7073444, 9631892, 3021012, 5195059,
                           9913714, 901749,  7776938, 3186566, 4955569, 5131067, 98619,
                           2282579, 7521455, 4430706, 1937859, 4532040, 0};

    EXPECT_EQ(static_cast<std::size_t>(result.tbl->view().column(0).size()),
              sizeof(col0_data) / sizeof(col0_data[0]));
    cudf::test::fixed_point_column_wrapper<int64_t> col0(
      std::begin(col0_data), std::end(col0_data), validity_c0, numeric::scale_type{-3});
    cudf::test::expect_columns_equal(result.tbl->view().column(0), col0);

    auto validity_c1 =
      cudf::test::make_counting_transform_iterator(0, [](auto i) { return i != 18; });
    int64_t col1_data[] = {361378026250,
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

    EXPECT_EQ(static_cast<std::size_t>(result.tbl->view().column(1).size()),
              sizeof(col1_data) / sizeof(col1_data[0]));
    cudf::test::fixed_point_column_wrapper<int64_t> col1(
      std::begin(col1_data), std::end(col1_data), validity_c1, numeric::scale_type{-11});
    cudf::test::expect_columns_equal(result.tbl->view().column(1), col1);

    read_opts.set_columns({"dec20p1"});
    EXPECT_THROW(cudf_io::read_parquet(read_opts), cudf::logic_error);
  }
}
CUDF_TEST_PROGRAM_MAIN()
