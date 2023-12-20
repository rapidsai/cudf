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
// TODO: Replace with `NumericTypes` when unsigned support is added. Issue #5352
using SupportedTypes = cudf::test::Types<int8_t, int16_t, int32_t, int64_t, bool, float, double>;
TYPED_TEST_SUITE(ParquetWriterNumericTypeTest, SupportedTypes);
using ComparableAndFixedTypes =
  cudf::test::Concat<cudf::test::ComparableTypes, cudf::test::FixedPointTypes>;
TYPED_TEST_SUITE(ParquetWriterComparableTypeTest, ComparableAndFixedTypes);
TYPED_TEST_SUITE(ParquetWriterChronoTypeTest, cudf::test::ChronoTypes);
using SupportedTimestampTypes =
  cudf::test::Types<cudf::timestamp_ms, cudf::timestamp_us, cudf::timestamp_ns>;
TYPED_TEST_SUITE(ParquetWriterTimestampTypeTest, SupportedTimestampTypes);
TYPED_TEST_SUITE(ParquetWriterSchemaTest, cudf::test::AllTypes);
using ByteLikeTypes = cudf::test::Types<int8_t, char, uint8_t, unsigned char, std::byte>;
TYPED_TEST_SUITE(ParquetReaderSourceTest, ByteLikeTypes);

// Declare typed test cases
TYPED_TEST_SUITE(ParquetChunkedWriterNumericTypeTest, SupportedTypes);

// test the allowed bit widths for dictionary encoding
INSTANTIATE_TEST_SUITE_P(ParquetDictionaryTest,
                         ParquetSizedTest,
                         testing::Range(1, 25),
                         testing::PrintToStringParamName());

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

TEST_P(ParquetV2Test, MultiColumn)
{
  constexpr auto num_rows = 100000;
  auto const is_v2        = GetParam();

  // auto col0_data = random_values<bool>(num_rows);
  auto col1_data = random_values<int8_t>(num_rows);
  auto col2_data = random_values<int16_t>(num_rows);
  auto col3_data = random_values<int32_t>(num_rows);
  auto col4_data = random_values<float>(num_rows);
  auto col5_data = random_values<double>(num_rows);
  auto col6_vals = random_values<int16_t>(num_rows);
  auto col7_vals = random_values<int32_t>(num_rows);
  auto col8_vals = random_values<int64_t>(num_rows);
  auto col6_data = cudf::detail::make_counting_transform_iterator(0, [col6_vals](auto i) {
    return numeric::decimal32{col6_vals[i], numeric::scale_type{5}};
  });
  auto col7_data = cudf::detail::make_counting_transform_iterator(0, [col7_vals](auto i) {
    return numeric::decimal64{col7_vals[i], numeric::scale_type{-5}};
  });
  auto col8_data = cudf::detail::make_counting_transform_iterator(0, [col8_vals](auto i) {
    return numeric::decimal128{col8_vals[i], numeric::scale_type{-6}};
  });
  auto validity  = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return true; });

  // column_wrapper<bool> col0{
  //    col0_data.begin(), col0_data.end(), validity};
  column_wrapper<int8_t> col1{col1_data.begin(), col1_data.end(), validity};
  column_wrapper<int16_t> col2{col2_data.begin(), col2_data.end(), validity};
  column_wrapper<int32_t> col3{col3_data.begin(), col3_data.end(), validity};
  column_wrapper<float> col4{col4_data.begin(), col4_data.end(), validity};
  column_wrapper<double> col5{col5_data.begin(), col5_data.end(), validity};
  column_wrapper<numeric::decimal32> col6{col6_data, col6_data + num_rows, validity};
  column_wrapper<numeric::decimal64> col7{col7_data, col7_data + num_rows, validity};
  column_wrapper<numeric::decimal128> col8{col8_data, col8_data + num_rows, validity};

  auto expected = table_view{{col1, col2, col3, col4, col5, col6, col7, col8}};

  cudf::io::table_input_metadata expected_metadata(expected);
  // expected_metadata.column_metadata[0].set_name( "bools");
  expected_metadata.column_metadata[0].set_name("int8s");
  expected_metadata.column_metadata[1].set_name("int16s");
  expected_metadata.column_metadata[2].set_name("int32s");
  expected_metadata.column_metadata[3].set_name("floats");
  expected_metadata.column_metadata[4].set_name("doubles");
  expected_metadata.column_metadata[5].set_name("decimal32s").set_decimal_precision(10);
  expected_metadata.column_metadata[6].set_name("decimal64s").set_decimal_precision(20);
  expected_metadata.column_metadata[7].set_name("decimal128s").set_decimal_precision(40);

  auto filepath = temp_env->get_temp_filepath("MultiColumn.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .write_v2_headers(is_v2)
      .metadata(expected_metadata);
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
  cudf::test::expect_metadata_equal(expected_metadata, result.metadata);
}

TEST_P(ParquetV2Test, MultiColumnWithNulls)
{
  constexpr auto num_rows = 100;
  auto const is_v2        = GetParam();

  // auto col0_data = random_values<bool>(num_rows);
  auto col1_data = random_values<int8_t>(num_rows);
  auto col2_data = random_values<int16_t>(num_rows);
  auto col3_data = random_values<int32_t>(num_rows);
  auto col4_data = random_values<float>(num_rows);
  auto col5_data = random_values<double>(num_rows);
  auto col6_vals = random_values<int32_t>(num_rows);
  auto col7_vals = random_values<int64_t>(num_rows);
  auto col6_data = cudf::detail::make_counting_transform_iterator(0, [col6_vals](auto i) {
    return numeric::decimal32{col6_vals[i], numeric::scale_type{-2}};
  });
  auto col7_data = cudf::detail::make_counting_transform_iterator(0, [col7_vals](auto i) {
    return numeric::decimal64{col7_vals[i], numeric::scale_type{-8}};
  });
  // auto col0_mask = cudf::detail::make_counting_transform_iterator(
  //    0, [](auto i) { return (i % 2); });
  auto col1_mask =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return (i < 10); });
  auto col2_mask = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return true; });
  auto col3_mask =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return (i == (num_rows - 1)); });
  auto col4_mask =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return (i >= 40 && i <= 60); });
  auto col5_mask =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return (i > 80); });
  auto col6_mask =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return (i % 5); });
  auto col7_mask =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return (i != 55); });

  // column_wrapper<bool> col0{
  //    col0_data.begin(), col0_data.end(), col0_mask};
  column_wrapper<int8_t> col1{col1_data.begin(), col1_data.end(), col1_mask};
  column_wrapper<int16_t> col2{col2_data.begin(), col2_data.end(), col2_mask};
  column_wrapper<int32_t> col3{col3_data.begin(), col3_data.end(), col3_mask};
  column_wrapper<float> col4{col4_data.begin(), col4_data.end(), col4_mask};
  column_wrapper<double> col5{col5_data.begin(), col5_data.end(), col5_mask};
  column_wrapper<numeric::decimal32> col6{col6_data, col6_data + num_rows, col6_mask};
  column_wrapper<numeric::decimal64> col7{col7_data, col7_data + num_rows, col7_mask};

  auto expected = table_view{{/*col0, */ col1, col2, col3, col4, col5, col6, col7}};

  cudf::io::table_input_metadata expected_metadata(expected);
  // expected_metadata.column_names.emplace_back("bools");
  expected_metadata.column_metadata[0].set_name("int8s");
  expected_metadata.column_metadata[1].set_name("int16s");
  expected_metadata.column_metadata[2].set_name("int32s");
  expected_metadata.column_metadata[3].set_name("floats");
  expected_metadata.column_metadata[4].set_name("doubles");
  expected_metadata.column_metadata[5].set_name("decimal32s").set_decimal_precision(9);
  expected_metadata.column_metadata[6].set_name("decimal64s").set_decimal_precision(20);

  auto filepath = temp_env->get_temp_filepath("MultiColumnWithNulls.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .write_v2_headers(is_v2)
      .metadata(expected_metadata);

  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
  // TODO: Need to be able to return metadata in tree form from reader so they can be compared.
  // Unfortunately the closest thing to a hierarchical schema is column_name_info which does not
  // have any tests for it c++ or python.
  cudf::test::expect_metadata_equal(expected_metadata, result.metadata);
}

TEST_P(ParquetV2Test, Strings)
{
  auto const is_v2 = GetParam();

  std::vector<char const*> strings{
    "Monday", "Wȅdnȅsday", "Friday", "Monday", "Friday", "Friday", "Friday", "Funday"};
  auto const num_rows = strings.size();

  auto seq_col0 = random_values<int>(num_rows);
  auto seq_col2 = random_values<float>(num_rows);
  auto validity = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return true; });

  column_wrapper<int> col0{seq_col0.begin(), seq_col0.end(), validity};
  column_wrapper<cudf::string_view> col1{strings.begin(), strings.end()};
  column_wrapper<float> col2{seq_col2.begin(), seq_col2.end(), validity};

  auto expected = table_view{{col0, col1, col2}};

  cudf::io::table_input_metadata expected_metadata(expected);
  expected_metadata.column_metadata[0].set_name("col_other");
  expected_metadata.column_metadata[1].set_name("col_string");
  expected_metadata.column_metadata[2].set_name("col_another");

  auto filepath = temp_env->get_temp_filepath("Strings.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .write_v2_headers(is_v2)
      .metadata(expected_metadata);
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
  cudf::test::expect_metadata_equal(expected_metadata, result.metadata);
}

TEST_P(ParquetV2Test, StringsAsBinary)
{
  auto const is_v2 = GetParam();
  std::vector<char const*> unicode_strings{
    "Monday", "Wȅdnȅsday", "Friday", "Monday", "Friday", "Friday", "Friday", "Funday"};
  std::vector<char const*> ascii_strings{
    "Monday", "Wednesday", "Friday", "Monday", "Friday", "Friday", "Friday", "Funday"};

  column_wrapper<cudf::string_view> col0{ascii_strings.begin(), ascii_strings.end()};
  column_wrapper<cudf::string_view> col1{unicode_strings.begin(), unicode_strings.end()};
  column_wrapper<cudf::string_view> col2{ascii_strings.begin(), ascii_strings.end()};
  cudf::test::lists_column_wrapper<uint8_t> col3{{'M', 'o', 'n', 'd', 'a', 'y'},
                                                 {'W', 'e', 'd', 'n', 'e', 's', 'd', 'a', 'y'},
                                                 {'F', 'r', 'i', 'd', 'a', 'y'},
                                                 {'M', 'o', 'n', 'd', 'a', 'y'},
                                                 {'F', 'r', 'i', 'd', 'a', 'y'},
                                                 {'F', 'r', 'i', 'd', 'a', 'y'},
                                                 {'F', 'r', 'i', 'd', 'a', 'y'},
                                                 {'F', 'u', 'n', 'd', 'a', 'y'}};
  cudf::test::lists_column_wrapper<uint8_t> col4{
    {'M', 'o', 'n', 'd', 'a', 'y'},
    {'W', 200, 133, 'd', 'n', 200, 133, 's', 'd', 'a', 'y'},
    {'F', 'r', 'i', 'd', 'a', 'y'},
    {'M', 'o', 'n', 'd', 'a', 'y'},
    {'F', 'r', 'i', 'd', 'a', 'y'},
    {'F', 'r', 'i', 'd', 'a', 'y'},
    {'F', 'r', 'i', 'd', 'a', 'y'},
    {'F', 'u', 'n', 'd', 'a', 'y'}};

  auto write_tbl = table_view{{col0, col1, col2, col3, col4}};

  cudf::io::table_input_metadata expected_metadata(write_tbl);
  expected_metadata.column_metadata[0].set_name("col_single").set_output_as_binary(true);
  expected_metadata.column_metadata[1].set_name("col_string").set_output_as_binary(true);
  expected_metadata.column_metadata[2].set_name("col_another").set_output_as_binary(true);
  expected_metadata.column_metadata[3].set_name("col_binary");
  expected_metadata.column_metadata[4].set_name("col_binary2");

  auto filepath = temp_env->get_temp_filepath("BinaryStrings.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, write_tbl)
      .write_v2_headers(is_v2)
      .dictionary_policy(cudf::io::dictionary_policy::NEVER)
      .metadata(expected_metadata);
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
      .set_column_schema(
        {cudf::io::reader_column_schema().set_convert_binary_to_strings(false),
         cudf::io::reader_column_schema().set_convert_binary_to_strings(false),
         cudf::io::reader_column_schema().set_convert_binary_to_strings(false),
         cudf::io::reader_column_schema().add_child(cudf::io::reader_column_schema()),
         cudf::io::reader_column_schema().add_child(cudf::io::reader_column_schema())});
  auto result   = cudf::io::read_parquet(in_opts);
  auto expected = table_view{{col3, col4, col3, col3, col4}};

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
  cudf::test::expect_metadata_equal(expected_metadata, result.metadata);
}

TEST_P(ParquetV2Test, SlicedTable)
{
  // This test checks for writing zero copy, offsetted views into existing cudf tables

  std::vector<char const*> strings{
    "Monday", "Wȅdnȅsday", "Friday", "Monday", "Friday", "Friday", "Friday", "Funday"};
  auto const num_rows = strings.size();
  auto const is_v2    = GetParam();

  auto seq_col0 = random_values<int>(num_rows);
  auto seq_col2 = random_values<float>(num_rows);
  auto validity =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 3 != 0; });

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
  auto valids  = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 2; });
  auto valids2 = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 3; });
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

  // Struct column
  auto ages_col = cudf::test::fixed_width_column_wrapper<int32_t>{
    {48, 27, 25, 31, 351, 351, 29, 15}, {1, 1, 1, 1, 1, 0, 1, 1}};

  auto col5 = cudf::test::structs_column_wrapper{{ages_col}, {1, 1, 1, 1, 0, 1, 1, 1}};

  // Struct/List mixed column

  // []
  // [NULL, 2, NULL]
  // [4, 5]
  // NULL
  // []
  // [7, 8, 9]
  // [10]
  // [11, 12]
  lcw land{{{}, {{1, 2, 3}, valids}, {4, 5}, {}, {}, {7, 8, 9}, {10}, {11, 12}}, valids2};

  // []
  // [[1, 2, 3], [], [4, 5], [], [0, 6, 0]]
  // [[7, 8], []]
  // [[]]
  // [[]]
  // [[], [], []]
  // [[10]]
  // [[13, 14], [15]]
  lcw flats{lcw{},
            {{1, 2, 3}, {}, {4, 5}, {}, {0, 6, 0}},
            {{7, 8}, {}},
            lcw{lcw{}},
            lcw{lcw{}},
            lcw{lcw{}, lcw{}, lcw{}},
            {lcw{10}},
            {{13, 14}, {15}}};

  auto struct_1 = cudf::test::structs_column_wrapper{land, flats};
  auto is_human = cudf::test::fixed_width_column_wrapper<bool>{
    {true, true, false, false, true, false, true, false}};
  auto col6 = cudf::test::structs_column_wrapper{{is_human, struct_1}};

  auto expected = table_view({col0, col1, col2, col3, col4, col5, col6});

  // auto expected_slice = expected;
  auto expected_slice = cudf::slice(expected, {2, static_cast<cudf::size_type>(num_rows) - 1});

  cudf::io::table_input_metadata expected_metadata(expected_slice);
  expected_metadata.column_metadata[0].set_name("col_other");
  expected_metadata.column_metadata[1].set_name("col_string");
  expected_metadata.column_metadata[2].set_name("col_another");
  expected_metadata.column_metadata[3].set_name("col_list");
  expected_metadata.column_metadata[4].set_name("col_multi_level_list");
  expected_metadata.column_metadata[5].set_name("col_struct");
  expected_metadata.column_metadata[5].set_name("col_struct_list");
  expected_metadata.column_metadata[6].child(0).set_name("human?");
  expected_metadata.column_metadata[6].child(1).set_name("particulars");
  expected_metadata.column_metadata[6].child(1).child(0).set_name("land");
  expected_metadata.column_metadata[6].child(1).child(1).set_name("flats");

  auto filepath = temp_env->get_temp_filepath("SlicedTable.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected_slice)
      .write_v2_headers(is_v2)
      .metadata(expected_metadata);
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected_slice, result.tbl->view());
  cudf::test::expect_metadata_equal(expected_metadata, result.metadata);
}

TEST_P(ParquetV2Test, ListColumn)
{
  auto const is_v2 = GetParam();

  auto valids  = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 2; });
  auto valids2 = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 3; });

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

  table_view expected({col0, col1, col2, col3, /* col4, */ col5, col6, col7});

  cudf::io::table_input_metadata expected_metadata(expected);
  expected_metadata.column_metadata[0].set_name("col_list_int_0");
  expected_metadata.column_metadata[1].set_name("col_list_list_int_1");
  expected_metadata.column_metadata[2].set_name("col_list_list_int_nullable_2");
  expected_metadata.column_metadata[3].set_name("col_list_list_nullable_double_nullable_3");
  // expected_metadata.column_metadata[0].set_name("col_list_list_uint16_4");
  expected_metadata.column_metadata[4].set_name("col_list_nullable_list_nullable_int_nullable_5");
  expected_metadata.column_metadata[5].set_name("col_list_list_string_6");
  expected_metadata.column_metadata[6].set_name("col_list_list_list_7");

  auto filepath = temp_env->get_temp_filepath("ListColumn.parquet");
  auto out_opts = cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
                    .write_v2_headers(is_v2)
                    .metadata(expected_metadata)
                    .compression(cudf::io::compression_type::NONE);

  cudf::io::write_parquet(out_opts);

  auto in_opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result  = cudf::io::read_parquet(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
  cudf::test::expect_metadata_equal(expected_metadata, result.metadata);
}

TEST_P(ParquetV2Test, StructOfList)
{
  auto const is_v2 = GetParam();

  // Struct<is_human:bool,
  //        Struct<weight:float,
  //               ages:int,
  //               land_unit:List<int>>,
  //               flats:List<List<int>>
  //              >
  //       >

  auto weights_col = cudf::test::fixed_width_column_wrapper<float>{1.1, 2.4, 5.3, 8.0, 9.6, 6.9};

  auto ages_col =
    cudf::test::fixed_width_column_wrapper<int32_t>{{48, 27, 25, 31, 351, 351}, {1, 1, 1, 1, 1, 0}};

  auto valids  = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 2; });
  auto valids2 = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i != 3; });

  using lcw = cudf::test::lists_column_wrapper<int32_t>;

  // []
  // [NULL, 2, NULL]
  // [4, 5]
  // NULL
  // []
  // [7, 8, 9]
  lcw land_unit{{{}, {{1, 2, 3}, valids}, {4, 5}, {}, {}, {7, 8, 9}}, valids2};

  // []
  // [[1, 2, 3], [], [4, 5], [], [0, 6, 0]]
  // [[7, 8], []]
  // [[]]
  // [[]]
  // [[], [], []]
  lcw flats{lcw{},
            {{1, 2, 3}, {}, {4, 5}, {}, {0, 6, 0}},
            {{7, 8}, {}},
            lcw{lcw{}},
            lcw{lcw{}},
            lcw{lcw{}, lcw{}, lcw{}}};

  auto struct_1 = cudf::test::structs_column_wrapper{{weights_col, ages_col, land_unit, flats},
                                                     {1, 1, 1, 1, 0, 1}};

  auto is_human_col = cudf::test::fixed_width_column_wrapper<bool>{
    {true, true, false, false, false, false}, {1, 1, 0, 1, 1, 0}};

  auto struct_2 =
    cudf::test::structs_column_wrapper{{is_human_col, struct_1}, {0, 1, 1, 1, 1, 1}}.release();

  auto expected = table_view({*struct_2});

  cudf::io::table_input_metadata expected_metadata(expected);
  expected_metadata.column_metadata[0].set_name("being");
  expected_metadata.column_metadata[0].child(0).set_name("human?");
  expected_metadata.column_metadata[0].child(1).set_name("particulars");
  expected_metadata.column_metadata[0].child(1).child(0).set_name("weight");
  expected_metadata.column_metadata[0].child(1).child(1).set_name("age");
  expected_metadata.column_metadata[0].child(1).child(2).set_name("land_unit");
  expected_metadata.column_metadata[0].child(1).child(3).set_name("flats");

  auto filepath = temp_env->get_temp_filepath("StructOfList.parquet");
  cudf::io::parquet_writer_options args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .write_v2_headers(is_v2)
      .metadata(expected_metadata);
  cudf::io::write_parquet(args);

  cudf::io::parquet_reader_options read_args =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info(filepath));
  auto const result = cudf::io::read_parquet(read_args);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
  cudf::test::expect_metadata_equal(expected_metadata, result.metadata);
}

TEST_P(ParquetV2Test, ListOfStruct)
{
  auto const is_v2 = GetParam();

  // List<Struct<is_human:bool,
  //             Struct<weight:float,
  //                    ages:int,
  //                   >
  //            >
  //     >

  auto weight_col = cudf::test::fixed_width_column_wrapper<float>{1.1, 2.4, 5.3, 8.0, 9.6, 6.9};

  auto ages_col =
    cudf::test::fixed_width_column_wrapper<int32_t>{{48, 27, 25, 31, 351, 351}, {1, 1, 1, 1, 1, 0}};

  auto struct_1 = cudf::test::structs_column_wrapper{{weight_col, ages_col}, {1, 1, 1, 1, 0, 1}};

  auto is_human_col = cudf::test::fixed_width_column_wrapper<bool>{
    {true, true, false, false, false, false}, {1, 1, 0, 1, 1, 0}};

  auto struct_2 =
    cudf::test::structs_column_wrapper{{is_human_col, struct_1}, {0, 1, 1, 1, 1, 1}}.release();

  auto list_offsets_column =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 2, 5, 5, 6}.release();
  auto num_list_rows = list_offsets_column->size() - 1;

  auto list_col = cudf::make_lists_column(
    num_list_rows, std::move(list_offsets_column), std::move(struct_2), 0, {});

  auto expected = table_view({*list_col});

  cudf::io::table_input_metadata expected_metadata(expected);
  expected_metadata.column_metadata[0].set_name("family");
  expected_metadata.column_metadata[0].child(1).child(0).set_name("human?");
  expected_metadata.column_metadata[0].child(1).child(1).set_name("particulars");
  expected_metadata.column_metadata[0].child(1).child(1).child(0).set_name("weight");
  expected_metadata.column_metadata[0].child(1).child(1).child(1).set_name("age");

  auto filepath = temp_env->get_temp_filepath("ListOfStruct.parquet");
  cudf::io::parquet_writer_options args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .write_v2_headers(is_v2)
      .metadata(expected_metadata);
  cudf::io::write_parquet(args);

  cudf::io::parquet_reader_options read_args =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info(filepath));
  auto const result = cudf::io::read_parquet(read_args);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
  cudf::test::expect_metadata_equal(expected_metadata, result.metadata);
}

TEST_P(ParquetV2Test, PartitionedWriteEmptyPartitions)
{
  auto const is_v2 = GetParam();

  auto source = create_random_fixed_table<int>(4, 4, false);

  auto filepath1 = temp_env->get_temp_filepath("PartitionedWrite1.parquet");
  auto filepath2 = temp_env->get_temp_filepath("PartitionedWrite2.parquet");

  auto partition1 = cudf::io::partition_info{1, 0};
  auto partition2 = cudf::io::partition_info{1, 0};

  auto expected1 =
    cudf::slice(*source, {partition1.start_row, partition1.start_row + partition1.num_rows});
  auto expected2 =
    cudf::slice(*source, {partition2.start_row, partition2.start_row + partition2.num_rows});

  cudf::io::parquet_writer_options args =
    cudf::io::parquet_writer_options::builder(
      cudf::io::sink_info(std::vector<std::string>{filepath1, filepath2}), *source)
      .partitions({partition1, partition2})
      .write_v2_headers(is_v2)
      .compression(cudf::io::compression_type::NONE);
  cudf::io::write_parquet(args);

  auto result1 = cudf::io::read_parquet(
    cudf::io::parquet_reader_options::builder(cudf::io::source_info(filepath1)));
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected1, result1.tbl->view());

  auto result2 = cudf::io::read_parquet(
    cudf::io::parquet_reader_options::builder(cudf::io::source_info(filepath2)));
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected2, result2.tbl->view());
}

TEST_P(ParquetV2Test, PartitionedWriteEmptyColumns)
{
  auto const is_v2 = GetParam();

  auto source = create_random_fixed_table<int>(0, 4, false);

  auto filepath1 = temp_env->get_temp_filepath("PartitionedWrite1.parquet");
  auto filepath2 = temp_env->get_temp_filepath("PartitionedWrite2.parquet");

  auto partition1 = cudf::io::partition_info{1, 0};
  auto partition2 = cudf::io::partition_info{1, 0};

  auto expected1 =
    cudf::slice(*source, {partition1.start_row, partition1.start_row + partition1.num_rows});
  auto expected2 =
    cudf::slice(*source, {partition2.start_row, partition2.start_row + partition2.num_rows});

  cudf::io::parquet_writer_options args =
    cudf::io::parquet_writer_options::builder(
      cudf::io::sink_info(std::vector<std::string>{filepath1, filepath2}), *source)
      .partitions({partition1, partition2})
      .write_v2_headers(is_v2)
      .compression(cudf::io::compression_type::NONE);
  cudf::io::write_parquet(args);

  auto result1 = cudf::io::read_parquet(
    cudf::io::parquet_reader_options::builder(cudf::io::source_info(filepath1)));
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected1, result1.tbl->view());

  auto result2 = cudf::io::read_parquet(
    cudf::io::parquet_reader_options::builder(cudf::io::source_info(filepath2)));
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected2, result2.tbl->view());
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

TEST_P(ParquetV2Test, LargeColumnIndex)
{
  // create a file large enough to be written in 2 batches (currently 1GB per batch)
  // pick fragment size that num_rows is divisible by, so we'll get equal sized row groups
  const std::string s1(1000, 'a');
  const std::string s2(1000, 'b');
  constexpr auto num_rows  = 512 * 1024;
  constexpr auto frag_size = num_rows / 128;
  auto const is_v2         = GetParam();

  auto col0_elements = cudf::detail::make_counting_transform_iterator(
    0, [&](auto i) { return (i < num_rows) ? s1 : s2; });
  auto col0 = cudf::test::strings_column_wrapper(col0_elements, col0_elements + 2 * num_rows);

  auto const expected = table_view{{col0, col0}};

  auto const filepath = temp_env->get_temp_filepath("LargeColumnIndex.parquet");
  const cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN)
      .compression(cudf::io::compression_type::NONE)
      .dictionary_policy(cudf::io::dictionary_policy::NEVER)
      .write_v2_headers(is_v2)
      .max_page_fragment_size(frag_size)
      .row_group_size_bytes(1024 * 1024 * 1024)
      .row_group_size_rows(num_rows);
  cudf::io::write_parquet(out_opts);

  auto const source = cudf::io::datasource::create(filepath);
  cudf::io::parquet::detail::FileMetaData fmd;

  read_footer(source, &fmd);

  for (auto const& rg : fmd.row_groups) {
    for (size_t c = 0; c < rg.columns.size(); c++) {
      auto const& chunk = rg.columns[c];

      auto const ci    = read_column_index(source, chunk);
      auto const stats = get_statistics(chunk);

      // check trunc(page.min) <= stats.min && trun(page.max) >= stats.max
      auto const ptype = fmd.schema[c + 1].type;
      auto const ctype = fmd.schema[c + 1].converted_type;
      ASSERT_TRUE(stats.min_value.has_value());
      ASSERT_TRUE(stats.max_value.has_value());
      EXPECT_TRUE(compare_binary(ci.min_values[0], stats.min_value.value(), ptype, ctype) <= 0);
      EXPECT_TRUE(compare_binary(ci.max_values[0], stats.max_value.value(), ptype, ctype) >= 0);
    }
  }
}

TEST_P(ParquetV2Test, CheckColumnOffsetIndex)
{
  constexpr auto num_rows      = 100000;
  auto const is_v2             = GetParam();
  auto const expected_hdr_type = is_v2 ? cudf::io::parquet::detail::PageType::DATA_PAGE_V2
                                       : cudf::io::parquet::detail::PageType::DATA_PAGE;

  // fixed length strings
  auto str1_elements = cudf::detail::make_counting_transform_iterator(0, [](auto i) {
    char buf[30];
    sprintf(buf, "%012d", i);
    return std::string(buf);
  });
  auto col0          = cudf::test::strings_column_wrapper(str1_elements, str1_elements + num_rows);

  auto col1_data = random_values<int8_t>(num_rows);
  auto col2_data = random_values<int16_t>(num_rows);
  auto col3_data = random_values<int32_t>(num_rows);
  auto col4_data = random_values<uint64_t>(num_rows);
  auto col5_data = random_values<float>(num_rows);
  auto col6_data = random_values<double>(num_rows);

  auto col1 = cudf::test::fixed_width_column_wrapper<int8_t>(col1_data.begin(), col1_data.end());
  auto col2 = cudf::test::fixed_width_column_wrapper<int16_t>(col2_data.begin(), col2_data.end());
  auto col3 = cudf::test::fixed_width_column_wrapper<int32_t>(col3_data.begin(), col3_data.end());
  auto col4 = cudf::test::fixed_width_column_wrapper<uint64_t>(col4_data.begin(), col4_data.end());
  auto col5 = cudf::test::fixed_width_column_wrapper<float>(col5_data.begin(), col5_data.end());
  auto col6 = cudf::test::fixed_width_column_wrapper<double>(col6_data.begin(), col6_data.end());

  // mixed length strings
  auto str2_elements = cudf::detail::make_counting_transform_iterator(0, [](auto i) {
    char buf[30];
    sprintf(buf, "%d", i);
    return std::string(buf);
  });
  auto col7          = cudf::test::strings_column_wrapper(str2_elements, str2_elements + num_rows);

  auto const expected = table_view{{col0, col1, col2, col3, col4, col5, col6, col7}};

  auto const filepath = temp_env->get_temp_filepath("CheckColumnOffsetIndex.parquet");
  const cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN)
      .write_v2_headers(is_v2)
      .max_page_size_rows(20000);
  cudf::io::write_parquet(out_opts);

  auto const source = cudf::io::datasource::create(filepath);
  cudf::io::parquet::detail::FileMetaData fmd;

  read_footer(source, &fmd);

  for (size_t r = 0; r < fmd.row_groups.size(); r++) {
    auto const& rg = fmd.row_groups[r];
    for (size_t c = 0; c < rg.columns.size(); c++) {
      auto const& chunk = rg.columns[c];

      // loop over offsets, read each page header, make sure it's a data page and that
      // the first row index is correct
      auto const oi = read_offset_index(source, chunk);

      int64_t num_vals = 0;
      for (size_t o = 0; o < oi.page_locations.size(); o++) {
        auto const& page_loc = oi.page_locations[o];
        auto const ph        = read_page_header(source, page_loc);
        EXPECT_EQ(ph.type, expected_hdr_type);
        EXPECT_EQ(page_loc.first_row_index, num_vals);
        num_vals += is_v2 ? ph.data_page_header_v2.num_rows : ph.data_page_header.num_values;
      }

      // loop over page stats from the column index. check that stats.min <= page.min
      // and stats.max >= page.max for each page.
      auto const ci    = read_column_index(source, chunk);
      auto const stats = get_statistics(chunk);

      ASSERT_TRUE(stats.min_value.has_value());
      ASSERT_TRUE(stats.max_value.has_value());
      ASSERT_TRUE(ci.null_counts.has_value());

      // schema indexing starts at 1
      auto const ptype = fmd.schema[c + 1].type;
      auto const ctype = fmd.schema[c + 1].converted_type;
      for (size_t p = 0; p < ci.min_values.size(); p++) {
        // null_pages should always be false
        EXPECT_FALSE(ci.null_pages[p]);
        // null_counts should always be 0
        EXPECT_EQ(ci.null_counts.value()[p], 0);
        EXPECT_TRUE(compare_binary(stats.min_value.value(), ci.min_values[p], ptype, ctype) <= 0);
      }
      for (size_t p = 0; p < ci.max_values.size(); p++)
        EXPECT_TRUE(compare_binary(stats.max_value.value(), ci.max_values[p], ptype, ctype) >= 0);
    }
  }
}

TEST_P(ParquetV2Test, CheckColumnOffsetIndexNulls)
{
  constexpr auto num_rows      = 100000;
  auto const is_v2             = GetParam();
  auto const expected_hdr_type = is_v2 ? cudf::io::parquet::detail::PageType::DATA_PAGE_V2
                                       : cudf::io::parquet::detail::PageType::DATA_PAGE;

  // fixed length strings
  auto str1_elements = cudf::detail::make_counting_transform_iterator(0, [](auto i) {
    char buf[30];
    sprintf(buf, "%012d", i);
    return std::string(buf);
  });
  auto col0          = cudf::test::strings_column_wrapper(str1_elements, str1_elements + num_rows);

  auto col1_data = random_values<int8_t>(num_rows);
  auto col2_data = random_values<int16_t>(num_rows);
  auto col3_data = random_values<int32_t>(num_rows);
  auto col4_data = random_values<uint64_t>(num_rows);
  auto col5_data = random_values<float>(num_rows);
  auto col6_data = random_values<double>(num_rows);

  auto valids =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 2 == 0; });

  // add null values for all but first column
  auto col1 =
    cudf::test::fixed_width_column_wrapper<int8_t>(col1_data.begin(), col1_data.end(), valids);
  auto col2 =
    cudf::test::fixed_width_column_wrapper<int16_t>(col2_data.begin(), col2_data.end(), valids);
  auto col3 =
    cudf::test::fixed_width_column_wrapper<int32_t>(col3_data.begin(), col3_data.end(), valids);
  auto col4 =
    cudf::test::fixed_width_column_wrapper<uint64_t>(col4_data.begin(), col4_data.end(), valids);
  auto col5 =
    cudf::test::fixed_width_column_wrapper<float>(col5_data.begin(), col5_data.end(), valids);
  auto col6 =
    cudf::test::fixed_width_column_wrapper<double>(col6_data.begin(), col6_data.end(), valids);

  // mixed length strings
  auto str2_elements = cudf::detail::make_counting_transform_iterator(0, [](auto i) {
    char buf[30];
    sprintf(buf, "%d", i);
    return std::string(buf);
  });
  auto col7 = cudf::test::strings_column_wrapper(str2_elements, str2_elements + num_rows, valids);

  auto expected = table_view{{col0, col1, col2, col3, col4, col5, col6, col7}};

  auto const filepath = temp_env->get_temp_filepath("CheckColumnOffsetIndexNulls.parquet");
  const cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN)
      .write_v2_headers(is_v2)
      .max_page_size_rows(20000);
  cudf::io::write_parquet(out_opts);

  auto const source = cudf::io::datasource::create(filepath);
  cudf::io::parquet::detail::FileMetaData fmd;

  read_footer(source, &fmd);

  for (size_t r = 0; r < fmd.row_groups.size(); r++) {
    auto const& rg = fmd.row_groups[r];
    for (size_t c = 0; c < rg.columns.size(); c++) {
      auto const& chunk = rg.columns[c];

      // loop over offsets, read each page header, make sure it's a data page and that
      // the first row index is correct
      auto const oi = read_offset_index(source, chunk);

      int64_t num_vals = 0;
      for (size_t o = 0; o < oi.page_locations.size(); o++) {
        auto const& page_loc = oi.page_locations[o];
        auto const ph        = read_page_header(source, page_loc);
        EXPECT_EQ(ph.type, expected_hdr_type);
        EXPECT_EQ(page_loc.first_row_index, num_vals);
        num_vals += is_v2 ? ph.data_page_header_v2.num_rows : ph.data_page_header.num_values;
      }

      // loop over page stats from the column index. check that stats.min <= page.min
      // and stats.max >= page.max for each page.
      auto const ci    = read_column_index(source, chunk);
      auto const stats = get_statistics(chunk);

      // should be half nulls, except no nulls in column 0
      ASSERT_TRUE(stats.min_value.has_value());
      ASSERT_TRUE(stats.max_value.has_value());
      ASSERT_TRUE(stats.null_count.has_value());
      EXPECT_EQ(stats.null_count.value(), c == 0 ? 0 : num_rows / 2);
      ASSERT_TRUE(ci.null_counts.has_value());

      // schema indexing starts at 1
      auto const ptype = fmd.schema[c + 1].type;
      auto const ctype = fmd.schema[c + 1].converted_type;
      for (size_t p = 0; p < ci.min_values.size(); p++) {
        EXPECT_FALSE(ci.null_pages[p]);
        if (c > 0) {  // first column has no nulls
          EXPECT_GT(ci.null_counts.value()[p], 0);
        } else {
          EXPECT_EQ(ci.null_counts.value()[p], 0);
        }
        EXPECT_TRUE(compare_binary(stats.min_value.value(), ci.min_values[p], ptype, ctype) <= 0);
      }
      for (size_t p = 0; p < ci.max_values.size(); p++) {
        EXPECT_TRUE(compare_binary(stats.max_value.value(), ci.max_values[p], ptype, ctype) >= 0);
      }
    }
  }
}

TEST_P(ParquetV2Test, CheckColumnOffsetIndexNullColumn)
{
  constexpr auto num_rows      = 100000;
  auto const is_v2             = GetParam();
  auto const expected_hdr_type = is_v2 ? cudf::io::parquet::detail::PageType::DATA_PAGE_V2
                                       : cudf::io::parquet::detail::PageType::DATA_PAGE;

  // fixed length strings
  auto str1_elements = cudf::detail::make_counting_transform_iterator(0, [](auto i) {
    char buf[30];
    sprintf(buf, "%012d", i);
    return std::string(buf);
  });
  auto col0          = cudf::test::strings_column_wrapper(str1_elements, str1_elements + num_rows);

  auto col1_data = random_values<int32_t>(num_rows);
  auto col2_data = random_values<int32_t>(num_rows);

  // col1 is all nulls
  auto valids = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return false; });
  auto col1 =
    cudf::test::fixed_width_column_wrapper<int32_t>(col1_data.begin(), col1_data.end(), valids);
  auto col2 = cudf::test::fixed_width_column_wrapper<int32_t>(col2_data.begin(), col2_data.end());

  // mixed length strings
  auto str2_elements = cudf::detail::make_counting_transform_iterator(0, [](auto i) {
    char buf[30];
    sprintf(buf, "%d", i);
    return std::string(buf);
  });
  auto col3          = cudf::test::strings_column_wrapper(str2_elements, str2_elements + num_rows);

  auto expected = table_view{{col0, col1, col2, col3}};

  auto const filepath = temp_env->get_temp_filepath("CheckColumnOffsetIndexNullColumn.parquet");
  const cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN)
      .write_v2_headers(is_v2)
      .max_page_size_rows(20000);
  cudf::io::write_parquet(out_opts);

  auto const source = cudf::io::datasource::create(filepath);
  cudf::io::parquet::detail::FileMetaData fmd;

  read_footer(source, &fmd);

  for (size_t r = 0; r < fmd.row_groups.size(); r++) {
    auto const& rg = fmd.row_groups[r];
    for (size_t c = 0; c < rg.columns.size(); c++) {
      auto const& chunk = rg.columns[c];

      // loop over offsets, read each page header, make sure it's a data page and that
      // the first row index is correct
      auto const oi = read_offset_index(source, chunk);

      int64_t num_vals = 0;
      for (size_t o = 0; o < oi.page_locations.size(); o++) {
        auto const& page_loc = oi.page_locations[o];
        auto const ph        = read_page_header(source, page_loc);
        EXPECT_EQ(ph.type, expected_hdr_type);
        EXPECT_EQ(page_loc.first_row_index, num_vals);
        num_vals += is_v2 ? ph.data_page_header_v2.num_rows : ph.data_page_header.num_values;
      }

      // loop over page stats from the column index. check that stats.min <= page.min
      // and stats.max >= page.max for each non-empty page.
      auto const ci    = read_column_index(source, chunk);
      auto const stats = get_statistics(chunk);

      // there should be no nulls except column 1 which is all nulls
      if (c != 1) {
        ASSERT_TRUE(stats.min_value.has_value());
        ASSERT_TRUE(stats.max_value.has_value());
      }
      ASSERT_TRUE(stats.null_count.has_value());
      EXPECT_EQ(stats.null_count.value(), c == 1 ? num_rows : 0);
      ASSERT_TRUE(ci.null_counts.has_value());

      // schema indexing starts at 1
      auto const ptype = fmd.schema[c + 1].type;
      auto const ctype = fmd.schema[c + 1].converted_type;
      for (size_t p = 0; p < ci.min_values.size(); p++) {
        // check tnat null_pages is true for column 1
        if (c == 1) {
          EXPECT_TRUE(ci.null_pages[p]);
          EXPECT_GT(ci.null_counts.value()[p], 0);
        }
        if (not ci.null_pages[p]) {
          EXPECT_EQ(ci.null_counts.value()[p], 0);
          EXPECT_TRUE(compare_binary(stats.min_value.value(), ci.min_values[p], ptype, ctype) <= 0);
        }
      }
      for (size_t p = 0; p < ci.max_values.size(); p++) {
        if (not ci.null_pages[p]) {
          EXPECT_TRUE(compare_binary(stats.max_value.value(), ci.max_values[p], ptype, ctype) >= 0);
        }
      }
    }
  }
}

TEST_P(ParquetV2Test, CheckColumnOffsetIndexStruct)
{
  auto const is_v2             = GetParam();
  auto const expected_hdr_type = is_v2 ? cudf::io::parquet::detail::PageType::DATA_PAGE_V2
                                       : cudf::io::parquet::detail::PageType::DATA_PAGE;

  auto c0 = testdata::ascending<uint32_t>();

  auto sc0 = testdata::ascending<cudf::string_view>();
  auto sc1 = testdata::descending<int32_t>();
  auto sc2 = testdata::unordered<int64_t>();

  std::vector<std::unique_ptr<cudf::column>> struct_children;
  struct_children.push_back(sc0.release());
  struct_children.push_back(sc1.release());
  struct_children.push_back(sc2.release());
  cudf::test::structs_column_wrapper c1(std::move(struct_children));

  auto listgen = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return i % 2 == 0 ? i / 2 : num_ordered_rows - (i / 2); });
  auto list =
    cudf::test::fixed_width_column_wrapper<int32_t>(listgen, listgen + 2 * num_ordered_rows);
  auto offgen = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i * 2; });
  auto offsets =
    cudf::test::fixed_width_column_wrapper<int32_t>(offgen, offgen + num_ordered_rows + 1);

  auto c2 = cudf::make_lists_column(num_ordered_rows, offsets.release(), list.release(), 0, {});

  table_view expected({c0, c1, *c2});

  auto const filepath = temp_env->get_temp_filepath("CheckColumnOffsetIndexStruct.parquet");
  const cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN)
      .write_v2_headers(is_v2)
      .max_page_size_rows(page_size_for_ordered_tests);
  cudf::io::write_parquet(out_opts);

  auto const source = cudf::io::datasource::create(filepath);
  cudf::io::parquet::detail::FileMetaData fmd;

  read_footer(source, &fmd);

  // hard coded schema indices.
  // TODO find a way to do this without magic
  size_t const colidxs[] = {1, 3, 4, 5, 8};
  for (size_t r = 0; r < fmd.row_groups.size(); r++) {
    auto const& rg = fmd.row_groups[r];
    for (size_t c = 0; c < rg.columns.size(); c++) {
      size_t colidx     = colidxs[c];
      auto const& chunk = rg.columns[c];

      // loop over offsets, read each page header, make sure it's a data page and that
      // the first row index is correct
      auto const oi = read_offset_index(source, chunk);

      int64_t num_vals = 0;
      for (size_t o = 0; o < oi.page_locations.size(); o++) {
        auto const& page_loc = oi.page_locations[o];
        auto const ph        = read_page_header(source, page_loc);
        EXPECT_EQ(ph.type, expected_hdr_type);
        EXPECT_EQ(page_loc.first_row_index, num_vals);
        // last column has 2 values per row
        num_vals += is_v2 ? ph.data_page_header_v2.num_rows
                          : ph.data_page_header.num_values / (c == rg.columns.size() - 1 ? 2 : 1);
      }

      // loop over page stats from the column index. check that stats.min <= page.min
      // and stats.max >= page.max for each page.
      auto const ci    = read_column_index(source, chunk);
      auto const stats = get_statistics(chunk);

      ASSERT_TRUE(stats.min_value.has_value());
      ASSERT_TRUE(stats.max_value.has_value());

      auto const ptype = fmd.schema[colidx].type;
      auto const ctype = fmd.schema[colidx].converted_type;
      for (size_t p = 0; p < ci.min_values.size(); p++) {
        EXPECT_TRUE(compare_binary(stats.min_value.value(), ci.min_values[p], ptype, ctype) <= 0);
      }
      for (size_t p = 0; p < ci.max_values.size(); p++) {
        EXPECT_TRUE(compare_binary(stats.max_value.value(), ci.max_values[p], ptype, ctype) >= 0);
      }
    }
  }
}

TEST_P(ParquetV2Test, CheckColumnOffsetIndexStructNulls)
{
  auto const is_v2             = GetParam();
  auto const expected_hdr_type = is_v2 ? cudf::io::parquet::detail::PageType::DATA_PAGE_V2
                                       : cudf::io::parquet::detail::PageType::DATA_PAGE;

  auto validity2 =
    cudf::detail::make_counting_transform_iterator(0, [](cudf::size_type i) { return i % 2; });
  auto validity3 = cudf::detail::make_counting_transform_iterator(
    0, [](cudf::size_type i) { return (i % 3) != 0; });
  auto validity4 = cudf::detail::make_counting_transform_iterator(
    0, [](cudf::size_type i) { return (i % 4) != 0; });
  auto validity5 = cudf::detail::make_counting_transform_iterator(
    0, [](cudf::size_type i) { return (i % 5) != 0; });

  auto c0 = testdata::ascending<uint32_t>();

  auto col1_data = random_values<int32_t>(num_ordered_rows);
  auto col2_data = random_values<int32_t>(num_ordered_rows);
  auto col3_data = random_values<int32_t>(num_ordered_rows);

  // col1 is all nulls
  auto col1 =
    cudf::test::fixed_width_column_wrapper<int32_t>(col1_data.begin(), col1_data.end(), validity2);
  auto col2 =
    cudf::test::fixed_width_column_wrapper<int32_t>(col2_data.begin(), col2_data.end(), validity3);
  auto col3 =
    cudf::test::fixed_width_column_wrapper<int32_t>(col2_data.begin(), col2_data.end(), validity4);

  std::vector<std::unique_ptr<cudf::column>> struct_children;
  struct_children.push_back(col1.release());
  struct_children.push_back(col2.release());
  struct_children.push_back(col3.release());
  auto struct_validity = std::vector<bool>(validity5, validity5 + num_ordered_rows);
  cudf::test::structs_column_wrapper c1(std::move(struct_children), struct_validity);
  table_view expected({c0, c1});

  auto const filepath = temp_env->get_temp_filepath("CheckColumnOffsetIndexStructNulls.parquet");
  const cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN)
      .write_v2_headers(is_v2)
      .max_page_size_rows(page_size_for_ordered_tests);
  cudf::io::write_parquet(out_opts);

  auto const source = cudf::io::datasource::create(filepath);
  cudf::io::parquet::detail::FileMetaData fmd;

  read_footer(source, &fmd);

  // all struct columns will have num_ordered_rows / 5 nulls at level 0.
  // col1 will have num_ordered_rows / 2 nulls total
  // col2 will have num_ordered_rows / 3 nulls total
  // col3 will have num_ordered_rows / 4 nulls total
  int const null_mods[] = {0, 2, 3, 4};

  for (size_t r = 0; r < fmd.row_groups.size(); r++) {
    auto const& rg = fmd.row_groups[r];
    for (size_t c = 0; c < rg.columns.size(); c++) {
      auto const& chunk = rg.columns[c];

      // loop over offsets, read each page header, make sure it's a data page and that
      // the first row index is correct
      auto const oi = read_offset_index(source, chunk);
      auto const ci = read_column_index(source, chunk);

      // check definition level histogram (repetition will not be present)
      if (c != 0) {
        ASSERT_TRUE(chunk.meta_data.size_statistics.has_value());
        ASSERT_TRUE(chunk.meta_data.size_statistics->definition_level_histogram.has_value());
        // there are no lists so there should be no repetition level histogram
        EXPECT_FALSE(chunk.meta_data.size_statistics->repetition_level_histogram.has_value());
        auto const& def_hist = chunk.meta_data.size_statistics->definition_level_histogram.value();
        ASSERT_TRUE(def_hist.size() == 3L);
        auto const l0_nulls    = num_ordered_rows / 5;
        auto const l1_l0_nulls = num_ordered_rows / (5 * null_mods[c]);
        auto const l1_nulls    = num_ordered_rows / null_mods[c] - l1_l0_nulls;
        auto const l2_vals     = num_ordered_rows - l1_nulls - l0_nulls;
        EXPECT_EQ(def_hist[0], l0_nulls);
        EXPECT_EQ(def_hist[1], l1_nulls);
        EXPECT_EQ(def_hist[2], l2_vals);
      } else {
        // column 0 has no lists and no nulls and no strings, so there should be no size stats
        EXPECT_FALSE(chunk.meta_data.size_statistics.has_value());
      }

      int64_t num_vals = 0;

      if (is_v2) { ASSERT_TRUE(ci.null_counts.has_value()); }
      for (size_t o = 0; o < oi.page_locations.size(); o++) {
        auto const& page_loc = oi.page_locations[o];
        auto const ph        = read_page_header(source, page_loc);
        EXPECT_EQ(ph.type, expected_hdr_type);
        EXPECT_EQ(page_loc.first_row_index, num_vals);
        num_vals += is_v2 ? ph.data_page_header_v2.num_rows : ph.data_page_header.num_values;
        // check that null counts match
        if (is_v2) { EXPECT_EQ(ci.null_counts.value()[o], ph.data_page_header_v2.num_nulls); }
      }
    }
  }
}

TEST_P(ParquetV2Test, CheckColumnIndexListWithNulls)
{
  auto const is_v2             = GetParam();
  auto const expected_hdr_type = is_v2 ? cudf::io::parquet::detail::PageType::DATA_PAGE_V2
                                       : cudf::io::parquet::detail::PageType::DATA_PAGE;

  using cudf::test::iterators::null_at;
  using cudf::test::iterators::nulls_at;
  using lcw = cudf::test::lists_column_wrapper<int32_t>;

  // 4 nulls
  // [NULL, 2, NULL]
  // []
  // [4, 5]
  // NULL
  // def histogram [1, 1, 2, 3]
  // rep histogram [4, 3]
  lcw col0{{{{1, 2, 3}, nulls_at({0, 2})}, {}, {4, 5}, {}}, null_at(3)};

  // 4 nulls
  // [[1, 2, 3], [], [4, 5], [], [0, 6, 0]]
  // [[7, 8]]
  // []
  // [[]]
  // def histogram [1, 3, 10]
  // rep histogram [4, 4, 6]
  lcw col1{{{1, 2, 3}, {}, {4, 5}, {}, {0, 6, 0}}, {{7, 8}}, lcw{}, lcw{lcw{}}};

  // 4 nulls
  // [[1, 2, 3], [], [4, 5], NULL, [0, 6, 0]]
  // [[7, 8]]
  // []
  // [[]]
  // def histogram [1, 1, 2, 10]
  // rep histogram [4, 4, 6]
  lcw col2{{{{1, 2, 3}, {}, {4, 5}, {}, {0, 6, 0}}, null_at(3)}, {{7, 8}}, lcw{}, lcw{lcw{}}};

  // 6 nulls
  // [[1, 2, 3], [], [4, 5], NULL, [NULL, 6, NULL]]
  // [[7, 8]]
  // []
  // [[]]
  // def histogram [1, 1, 2, 2, 8]
  // rep histogram [4, 4, 6]
  using dlcw = cudf::test::lists_column_wrapper<double>;
  dlcw col3{{{{1., 2., 3.}, {}, {4., 5.}, {}, {{0., 6., 0.}, nulls_at({0, 2})}}, null_at(3)},
            {{7., 8.}},
            dlcw{},
            dlcw{dlcw{}}};

  // 4 nulls
  // [[1, 2, 3], [], [4, 5], NULL, [0, 6, 0]]
  // [[7, 8]]
  // []
  // NULL
  // def histogram [1, 1, 1, 1, 10]
  // rep histogram [4, 4, 6]
  using ui16lcw = cudf::test::lists_column_wrapper<uint16_t>;
  cudf::test::lists_column_wrapper<uint16_t> col4{
    {{{{1, 2, 3}, {}, {4, 5}, {}, {0, 6, 0}}, null_at(3)}, {{7, 8}}, ui16lcw{}, ui16lcw{ui16lcw{}}},
    null_at(3)};

  // 6 nulls
  // [[1, 2, 3], [], [4, 5], NULL, [NULL, 6, NULL]]
  // [[7, 8]]
  // []
  // NULL
  // def histogram [1, 1, 1, 1, 2, 8]
  // rep histogram [4, 4, 6]
  lcw col5{{{{{1, 2, 3}, {}, {4, 5}, {}, {{0, 6, 0}, nulls_at({0, 2})}}, null_at(3)},
            {{7, 8}},
            lcw{},
            lcw{lcw{}}},
           null_at(3)};

  // 4 nulls
  // def histogram [1, 3, 9]
  // rep histogram [4, 4, 5]
  using strlcw = cudf::test::lists_column_wrapper<cudf::string_view>;
  cudf::test::lists_column_wrapper<cudf::string_view> col6{
    {{"Monday", "Monday", "Friday"}, {}, {"Monday", "Friday"}, {}, {"Sunday", "Funday"}},
    {{"bee", "sting"}},
    strlcw{},
    strlcw{strlcw{}}};

  // 5 nulls
  // def histogram [1, 3, 1, 8]
  // rep histogram [4, 4, 5]
  using strlcw = cudf::test::lists_column_wrapper<cudf::string_view>;
  cudf::test::lists_column_wrapper<cudf::string_view> col7{{{"Monday", "Monday", "Friday"},
                                                            {},
                                                            {{"Monday", "Friday"}, null_at(1)},
                                                            {},
                                                            {"Sunday", "Funday"}},
                                                           {{"bee", "sting"}},
                                                           strlcw{},
                                                           strlcw{strlcw{}}};

  // 11 nulls
  // D   5   6   5  6        5  6  5      6 6
  // R   0   3   3  3        1  3  3      2 3
  // [[[NULL,2,NULL,4]], [[NULL,6,NULL], [8,9]]]
  // D 2      6    6   6  6      2
  // R 0      1    2   3  3      1
  // [NULL, [[13],[14,15,16]],  NULL]
  // D 2     3   2      4
  // R 0     1   1      1
  // [NULL, [], NULL, [[]]]
  // D 0
  // R 0
  // NULL
  // def histogram [1, 0, 4, 1, 1, 4, 9]
  // rep histogram [4, 6, 2, 8]
  lcw col8{{
             {{{{1, 2, 3, 4}, nulls_at({0, 2})}}, {{{5, 6, 7}, nulls_at({0, 2})}, {8, 9}}},
             {{{{10, 11}, {12}}, {{13}, {14, 15, 16}}, {{17, 18}}}, nulls_at({0, 2})},
             {{lcw{lcw{}}, lcw{}, lcw{}, lcw{lcw{}}}, nulls_at({0, 2})},
             lcw{lcw{lcw{}}},
           },
           null_at(3)};

  table_view expected({col0, col1, col2, col3, col4, col5, col6, col7});

  int64_t const expected_null_counts[]            = {4, 4, 4, 6, 4, 6, 4, 5, 11};
  std::vector<int64_t> const expected_def_hists[] = {{1, 1, 2, 3},
                                                     {1, 3, 10},
                                                     {1, 1, 2, 10},
                                                     {1, 1, 2, 2, 8},
                                                     {1, 1, 1, 1, 10},
                                                     {1, 1, 1, 1, 2, 8},
                                                     {1, 3, 9},
                                                     {1, 3, 1, 8},
                                                     {1, 0, 4, 1, 1, 4, 9}};
  std::vector<int64_t> const expected_rep_hists[] = {{4, 3},
                                                     {4, 4, 6},
                                                     {4, 4, 6},
                                                     {4, 4, 6},
                                                     {4, 4, 6},
                                                     {4, 4, 6},
                                                     {4, 4, 5},
                                                     {4, 4, 5},
                                                     {4, 6, 2, 8}};

  auto const filepath = temp_env->get_temp_filepath("ColumnIndexListWithNulls.parquet");
  auto out_opts = cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
                    .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN)
                    .write_v2_headers(is_v2)
                    .compression(cudf::io::compression_type::NONE);

  cudf::io::write_parquet(out_opts);

  auto const source = cudf::io::datasource::create(filepath);
  cudf::io::parquet::detail::FileMetaData fmd;

  read_footer(source, &fmd);

  for (size_t r = 0; r < fmd.row_groups.size(); r++) {
    auto const& rg = fmd.row_groups[r];
    for (size_t c = 0; c < rg.columns.size(); c++) {
      auto const& chunk = rg.columns[c];

      ASSERT_TRUE(chunk.meta_data.size_statistics.has_value());
      ASSERT_TRUE(chunk.meta_data.size_statistics->definition_level_histogram.has_value());
      ASSERT_TRUE(chunk.meta_data.size_statistics->repetition_level_histogram.has_value());
      // there is only one page, so chunk stats should match the page stats
      EXPECT_EQ(chunk.meta_data.size_statistics->definition_level_histogram.value(),
                expected_def_hists[c]);
      EXPECT_EQ(chunk.meta_data.size_statistics->repetition_level_histogram.value(),
                expected_rep_hists[c]);
      // only column 6 has string data
      if (c == 6) {
        ASSERT_TRUE(chunk.meta_data.size_statistics->unencoded_byte_array_data_bytes.has_value());
        EXPECT_EQ(chunk.meta_data.size_statistics->unencoded_byte_array_data_bytes.value(), 50L);
      } else if (c == 7) {
        ASSERT_TRUE(chunk.meta_data.size_statistics->unencoded_byte_array_data_bytes.has_value());
        EXPECT_EQ(chunk.meta_data.size_statistics->unencoded_byte_array_data_bytes.value(), 44L);
      } else {
        EXPECT_FALSE(chunk.meta_data.size_statistics->unencoded_byte_array_data_bytes.has_value());
      }

      // loop over offsets, read each page header, make sure it's a data page and that
      // the first row index is correct
      auto const oi = read_offset_index(source, chunk);

      for (size_t o = 0; o < oi.page_locations.size(); o++) {
        auto const& page_loc = oi.page_locations[o];
        auto const ph        = read_page_header(source, page_loc);
        EXPECT_EQ(ph.type, expected_hdr_type);
        // check null counts in V2 header
        if (is_v2) { EXPECT_EQ(ph.data_page_header_v2.num_nulls, expected_null_counts[c]); }
      }

      // check null counts in column chunk stats and page indexes
      auto const ci    = read_column_index(source, chunk);
      auto const stats = get_statistics(chunk);
      EXPECT_EQ(stats.null_count, expected_null_counts[c]);

      // should only be one page
      EXPECT_FALSE(ci.null_pages[0]);
      ASSERT_TRUE(ci.null_counts.has_value());
      EXPECT_EQ(ci.null_counts.value()[0], expected_null_counts[c]);

      ASSERT_TRUE(ci.definition_level_histogram.has_value());
      EXPECT_EQ(ci.definition_level_histogram.value(), expected_def_hists[c]);

      ASSERT_TRUE(ci.repetition_level_histogram.has_value());
      EXPECT_EQ(ci.repetition_level_histogram.value(), expected_rep_hists[c]);

      if (c == 6) {
        ASSERT_TRUE(oi.unencoded_byte_array_data_bytes.has_value());
        EXPECT_EQ(oi.unencoded_byte_array_data_bytes.value()[0], 50L);
      } else if (c == 7) {
        ASSERT_TRUE(oi.unencoded_byte_array_data_bytes.has_value());
        EXPECT_EQ(oi.unencoded_byte_array_data_bytes.value()[0], 44L);
      } else {
        EXPECT_FALSE(oi.unencoded_byte_array_data_bytes.has_value());
      }
    }
  }
}

TEST_P(ParquetSizedTest, DictionaryTest)
{
  unsigned int const cardinality = (1 << (GetParam() - 1)) + 1;
  unsigned int const nrows       = std::max(cardinality * 3 / 2, 3'000'000U);

  auto elements       = cudf::detail::make_counting_transform_iterator(0, [cardinality](auto i) {
    return "a unique string value suffixed with " + std::to_string(i % cardinality);
  });
  auto const col0     = cudf::test::strings_column_wrapper(elements, elements + nrows);
  auto const expected = table_view{{col0}};

  auto const filepath = temp_env->get_temp_filepath("DictionaryTest.parquet");
  // set row group size so that there will be only one row group
  // no compression so we can easily read page data
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .compression(cudf::io::compression_type::NONE)
      .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN)
      .dictionary_policy(cudf::io::dictionary_policy::ALWAYS)
      .row_group_size_rows(nrows)
      .row_group_size_bytes(512 * 1024 * 1024);
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options default_in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto const result = cudf::io::read_parquet(default_in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());

  // make sure dictionary was used
  auto const source = cudf::io::datasource::create(filepath);
  cudf::io::parquet::detail::FileMetaData fmd;

  read_footer(source, &fmd);
  auto used_dict = [&fmd]() {
    for (auto enc : fmd.row_groups[0].columns[0].meta_data.encodings) {
      if (enc == cudf::io::parquet::detail::Encoding::PLAIN_DICTIONARY or
          enc == cudf::io::parquet::detail::Encoding::RLE_DICTIONARY) {
        return true;
      }
    }
    return false;
  };
  EXPECT_TRUE(used_dict());

  // and check that the correct number of bits was used
  auto const oi    = read_offset_index(source, fmd.row_groups[0].columns[0]);
  auto const nbits = read_dict_bits(source, oi.page_locations[0]);
  EXPECT_EQ(nbits, GetParam());
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

// These chrono types are not supported because parquet writer does not have a type to represent
// them.
using UnsupportedChronoTypes =
  cudf::test::Types<cudf::timestamp_s, cudf::duration_D, cudf::duration_s>;
// Also fixed point types unsupported, because AST does not support them yet.
using SupportedTestTypes = cudf::test::RemoveIf<cudf::test::ContainedIn<UnsupportedChronoTypes>,
                                                cudf::test::ComparableTypes>;

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

TEST_P(ParquetV2Test, CheckEncodings)
{
  using cudf::io::parquet::detail::Encoding;
  constexpr auto num_rows = 100'000;
  auto const is_v2        = GetParam();

  auto const validity = cudf::test::iterators::no_nulls();
  // data should be PLAIN for v1, RLE for V2
  auto col0_data =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) -> bool { return i % 2 == 0; });
  // data should be PLAIN for v1, DELTA_BINARY_PACKED for v2
  auto col1_data = random_values<int32_t>(num_rows);
  // data should be PLAIN_DICTIONARY for v1, PLAIN and RLE_DICTIONARY for v2
  auto col2_data = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return 1; });

  cudf::test::fixed_width_column_wrapper<bool> col0{col0_data, col0_data + num_rows, validity};
  column_wrapper<int32_t> col1{col1_data.begin(), col1_data.end(), validity};
  column_wrapper<int32_t> col2{col2_data, col2_data + num_rows, validity};

  auto expected = table_view{{col0, col1, col2}};

  auto const filename = is_v2 ? "CheckEncodingsV2.parquet" : "CheckEncodingsV1.parquet";
  auto filepath       = temp_env->get_temp_filepath(filename);
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .max_page_size_rows(num_rows)
      .write_v2_headers(is_v2);
  cudf::io::write_parquet(out_opts);

  // make sure the expected encodings are present
  auto contains = [](auto const& vec, auto const& enc) {
    return std::find(vec.begin(), vec.end(), enc) != vec.end();
  };

  auto const source = cudf::io::datasource::create(filepath);
  cudf::io::parquet::detail::FileMetaData fmd;

  read_footer(source, &fmd);
  auto const& chunk0_enc = fmd.row_groups[0].columns[0].meta_data.encodings;
  auto const& chunk1_enc = fmd.row_groups[0].columns[1].meta_data.encodings;
  auto const& chunk2_enc = fmd.row_groups[0].columns[2].meta_data.encodings;
  if (is_v2) {
    // col0 should have RLE for rep/def and data
    EXPECT_TRUE(chunk0_enc.size() == 1);
    EXPECT_TRUE(contains(chunk0_enc, Encoding::RLE));
    // col1 should have RLE for rep/def and DELTA_BINARY_PACKED for data
    EXPECT_TRUE(chunk1_enc.size() == 2);
    EXPECT_TRUE(contains(chunk1_enc, Encoding::RLE));
    EXPECT_TRUE(contains(chunk1_enc, Encoding::DELTA_BINARY_PACKED));
    // col2 should have RLE for rep/def, PLAIN for dict, and RLE_DICTIONARY for data
    EXPECT_TRUE(chunk2_enc.size() == 3);
    EXPECT_TRUE(contains(chunk2_enc, Encoding::RLE));
    EXPECT_TRUE(contains(chunk2_enc, Encoding::PLAIN));
    EXPECT_TRUE(contains(chunk2_enc, Encoding::RLE_DICTIONARY));
  } else {
    // col0 should have RLE for rep/def and PLAIN for data
    EXPECT_TRUE(chunk0_enc.size() == 2);
    EXPECT_TRUE(contains(chunk0_enc, Encoding::RLE));
    EXPECT_TRUE(contains(chunk0_enc, Encoding::PLAIN));
    // col1 should have RLE for rep/def and PLAIN for data
    EXPECT_TRUE(chunk1_enc.size() == 2);
    EXPECT_TRUE(contains(chunk1_enc, Encoding::RLE));
    EXPECT_TRUE(contains(chunk1_enc, Encoding::PLAIN));
    // col2 should have RLE for rep/def and PLAIN_DICTIONARY for data and dict
    EXPECT_TRUE(chunk2_enc.size() == 2);
    EXPECT_TRUE(contains(chunk2_enc, Encoding::RLE));
    EXPECT_TRUE(contains(chunk2_enc, Encoding::PLAIN_DICTIONARY));
  }
}

// removing duration_D, duration_s, and timestamp_s as they don't appear to be supported properly.
// see definition of UnsupportedChronoTypes above.
using DeltaDecimalTypes = cudf::test::Types<numeric::decimal32, numeric::decimal64>;
using DeltaBinaryTypes =
  cudf::test::Concat<cudf::test::IntegralTypesNotBool, cudf::test::ChronoTypes, DeltaDecimalTypes>;
using SupportedDeltaTestTypes =
  cudf::test::RemoveIf<cudf::test::ContainedIn<UnsupportedChronoTypes>, DeltaBinaryTypes>;
TYPED_TEST_SUITE(ParquetWriterDeltaTest, SupportedDeltaTestTypes);

TYPED_TEST(ParquetWriterDeltaTest, SupportedDeltaTestTypes)
{
  using T   = TypeParam;
  auto col0 = testdata::ascending<T>();
  auto col1 = testdata::unordered<T>();

  auto const expected = table_view{{col0, col1}};

  auto const filepath = temp_env->get_temp_filepath("DeltaBinaryPacked.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .write_v2_headers(true)
      .dictionary_policy(cudf::io::dictionary_policy::NEVER);
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(in_opts);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
}

TYPED_TEST(ParquetWriterDeltaTest, SupportedDeltaTestTypesSliced)
{
  using T                = TypeParam;
  constexpr int num_rows = 4'000;
  auto col0              = testdata::ascending<T>();
  auto col1              = testdata::unordered<T>();

  auto const expected = table_view{{col0, col1}};
  auto expected_slice = cudf::slice(expected, {num_rows, 2 * num_rows});
  ASSERT_EQ(expected_slice[0].num_rows(), num_rows);

  auto const filepath = temp_env->get_temp_filepath("DeltaBinaryPackedSliced.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected_slice)
      .write_v2_headers(true)
      .dictionary_policy(cudf::io::dictionary_policy::NEVER);
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(in_opts);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_slice, result.tbl->view());
}

TYPED_TEST(ParquetWriterDeltaTest, SupportedDeltaListSliced)
{
  using T = TypeParam;

  constexpr int num_slice = 4'000;
  constexpr int num_rows  = 32 * 1024;

  std::mt19937 gen(6542);
  std::bernoulli_distribution bn(0.7f);
  auto valids =
    cudf::detail::make_counting_transform_iterator(0, [&](int index) { return bn(gen); });
  auto values = thrust::make_counting_iterator(0);

  // list<T>
  constexpr int vals_per_row = 4;
  auto c1_offset_iter        = cudf::detail::make_counting_transform_iterator(
    0, [vals_per_row](cudf::size_type idx) { return idx * vals_per_row; });
  cudf::test::fixed_width_column_wrapper<cudf::size_type> c1_offsets(c1_offset_iter,
                                                                     c1_offset_iter + num_rows + 1);
  cudf::test::fixed_width_column_wrapper<T> c1_vals(
    values, values + (num_rows * vals_per_row), valids);
  auto [null_mask, null_count] = cudf::test::detail::make_null_mask(valids, valids + num_rows);

  auto _c1 = cudf::make_lists_column(
    num_rows, c1_offsets.release(), c1_vals.release(), null_count, std::move(null_mask));
  auto c1 = cudf::purge_nonempty_nulls(*_c1);

  auto const expected = table_view{{*c1}};
  auto expected_slice = cudf::slice(expected, {num_slice, 2 * num_slice});
  ASSERT_EQ(expected_slice[0].num_rows(), num_slice);

  auto const filepath = temp_env->get_temp_filepath("DeltaBinaryPackedListSliced.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected_slice)
      .write_v2_headers(true)
      .dictionary_policy(cudf::io::dictionary_policy::NEVER);
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(in_opts);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_slice, result.tbl->view());
}

CUDF_TEST_PROGRAM_MAIN()
