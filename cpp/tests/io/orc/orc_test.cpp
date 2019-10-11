/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <sstream>
#include <tuple>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <tests/utilities/cudf_test_fixtures.h>
#include <tests/io/io_test_utils.hpp>
#include <tests/utilities/column_wrapper.cuh>

TempDirTestEnvironment *const temp_env = static_cast<TempDirTestEnvironment *>(
    ::testing::AddGlobalTestEnvironment(new TempDirTestEnvironment));

/**
 * @brief Base test fixture for ORC reader/writer tests
 **/
struct OrcWriterTest : public GdfTest {};

/**
 * @brief Typed test fixture for type-parameterized tests
 **/
template <typename T>
struct OrcWriterTypedParamTest : public OrcWriterTest {};

/**
 * @brief Test fixture for time-unit value-parameterized tests
 **/
struct OrcWriterValueParamTest
    : public OrcWriterTest,
      public testing::WithParamInterface<gdf_time_unit> {};

/**
 * @brief cuDF types that can be written to ORC types
 **/
using test_types = ::testing::Types<cudf::bool8, int8_t, int16_t, int32_t,
                                    int64_t, float, double>;
TYPED_TEST_CASE(OrcWriterTypedParamTest, test_types);

namespace {

/**
 * @brief Helper function to set column name
 **/
void column_set_name(gdf_column *col, const std::string &name) {
  col->col_name = static_cast<char *>(malloc(name.length() + 1));
  std::copy(name.begin(), name.end(), col->col_name);
  col->col_name[name.length()] = '\0';
}

/**
 * @brief Helper function to compare columns and raise GTest failure if required
 **/
auto columns_are_equal(const gdf_column &left, const gdf_column &right) {
  if (gdf_equal_columns(left, right)) {
    return ::testing::AssertionSuccess();
  } else {
    std::ostringstream buffer;
    buffer << std::endl;
    buffer << "    left data: " << left.col_name << std::endl;
    print_gdf_column(&left, 10, buffer);
    buffer << "    left valid:" << std::endl;
    print_valid_data(left.valid, left.size, buffer);
    buffer << "    right data: " << right.col_name << std::endl;
    print_gdf_column(&right, 10, buffer);
    buffer << "    right valid:" << std::endl;
    print_valid_data(right.valid, right.size, buffer);

    return ::testing::AssertionFailure() << buffer.str();
  }
}

/**
 * @brief Helper function to compare two cudf::tables
 **/
void tables_are_equal(const cudf::table &left, const cudf::table &right) {
  EXPECT_EQ(left.num_columns(), right.num_columns());
  auto expected = left.begin();
  auto result = right.begin();
  while (result != right.end()) {
    EXPECT_TRUE(columns_are_equal(*(*expected++), *(*result++)));
  }
}

}  // namespace

TYPED_TEST(OrcWriterTypedParamTest, SingleColumn) {
  constexpr auto num_rows = 100;
  cudf::test::column_wrapper<TypeParam> col{random_values<TypeParam>(num_rows),
                                            [](size_t row) { return true; }};
  column_set_name(col.get(), "col_" + std::string{typeid(TypeParam).name()});

  auto gdf_col = col.get();
  auto expected = cudf::table{&gdf_col, 1};
  EXPECT_EQ(1, expected.num_columns());

  auto filepath =
      temp_env->get_temp_filepath("OrcWriterSingleColumn") + gdf_col->col_name;

  cudf::orc_write_arg out_args{cudf::sink_info{filepath}};
  out_args.table = expected;
  cudf::write_orc(out_args);

  cudf::orc_read_arg in_args{cudf::source_info{filepath}};
  in_args.use_index = false;
  auto result = cudf::read_orc(in_args);

  tables_are_equal(expected, result);
}

TYPED_TEST(OrcWriterTypedParamTest, SingleColumnWithNulls) {
  constexpr auto num_rows = 100;
  auto nulls_threshold = 20;
  auto valids_func = [=](size_t row) {
    return (row < nulls_threshold || row > (num_rows - nulls_threshold));
  };
  cudf::test::column_wrapper<TypeParam> col{random_values<TypeParam>(num_rows),
                                            valids_func};
  column_set_name(col.get(), "col_" + std::string{typeid(TypeParam).name()});

  auto gdf_col = col.get();
  auto expected = cudf::table{&gdf_col, 1};
  EXPECT_EQ(1, expected.num_columns());

  auto filepath =
      temp_env->get_temp_filepath("OrcWriterSingleColumnWithNulls") +
      gdf_col->col_name;

  cudf::orc_write_arg out_args{cudf::sink_info{filepath}};
  out_args.table = expected;
  cudf::write_orc(out_args);

  cudf::orc_read_arg in_args{cudf::source_info{filepath}};
  in_args.use_index = false;
  auto result = cudf::read_orc(in_args);

  tables_are_equal(expected, result);
}

TEST_F(OrcWriterTest, MultiColumn) {
  constexpr auto num_rows = 100;
  auto valids_func = [](size_t row) { return true; };
  cudf::test::column_wrapper<cudf::bool8> col0{
      random_values<cudf::bool8>(num_rows), valids_func};
  cudf::test::column_wrapper<int8_t> col1{random_values<int8_t>(num_rows),
                                          valids_func};
  cudf::test::column_wrapper<int16_t> col2{random_values<int16_t>(num_rows),
                                           valids_func};
  cudf::test::column_wrapper<int32_t> col3{random_values<int32_t>(num_rows),
                                           valids_func};
  cudf::test::column_wrapper<float> col4{random_values<float>(num_rows),
                                         valids_func};
  cudf::test::column_wrapper<double> col5{random_values<double>(num_rows),
                                          valids_func};
  column_set_name(col0.get(), "bools");
  column_set_name(col1.get(), "int8s");
  column_set_name(col2.get(), "int16s");
  column_set_name(col3.get(), "int32s");
  column_set_name(col4.get(), "floats");
  column_set_name(col5.get(), "doubles");

  std::vector<gdf_column *> columns;
  columns.push_back(col0.get());
  columns.push_back(col1.get());
  columns.push_back(col2.get());
  columns.push_back(col3.get());
  columns.push_back(col4.get());
  columns.push_back(col5.get());
  auto expected = cudf::table{columns.data(), 6};
  EXPECT_EQ(columns.size(), expected.num_columns());

  auto filepath = temp_env->get_temp_filepath("OrcWriterMultiColumn.orc");

  cudf::orc_write_arg out_args{cudf::sink_info{filepath}};
  out_args.table = expected;
  cudf::write_orc(out_args);

  cudf::orc_read_arg in_args{cudf::source_info{filepath}};
  in_args.use_index = false;
  auto result = cudf::read_orc(in_args);

  tables_are_equal(expected, result);
}

TEST_F(OrcWriterTest, MultiColumnWithNulls) {
  constexpr auto num_rows = 100;

  // Boolean column with valids only on every other row
  cudf::test::column_wrapper<cudf::bool8> col0{
      random_values<cudf::bool8>(num_rows),
      [=](size_t row) { return (row % 2); }};
  // Bytes column with valids only before row 10
  cudf::test::column_wrapper<int8_t> col1{
      random_values<int8_t>(num_rows), [=](size_t row) { return (row < 10); }};
  // Shorts column with all valids
  cudf::test::column_wrapper<int16_t> col2{random_values<int16_t>(num_rows),
                                           [=](size_t row) { return true; }};
  // Integers column with only last row valid
  cudf::test::column_wrapper<int32_t> col3{
      random_values<int32_t>(num_rows),
      [=](size_t row) { return (row == (num_rows - 1)); }};
  // Floats column with valids only within rows 40 to 60
  cudf::test::column_wrapper<float> col4{
      random_values<float>(num_rows),
      [=](size_t row) { return (row >= 40 || row <= 60); }};
  // Doubles column with valids only after row 80
  cudf::test::column_wrapper<double> col5{
      random_values<double>(num_rows), [=](size_t row) { return (row > 80); }};

  column_set_name(col0.get(), "bools");
  column_set_name(col1.get(), "int8s");
  column_set_name(col2.get(), "int16s");
  column_set_name(col3.get(), "int32s");
  column_set_name(col4.get(), "floats");
  column_set_name(col5.get(), "doubles");

  std::vector<gdf_column *> columns;
  columns.push_back(col0.get());
  columns.push_back(col1.get());
  columns.push_back(col2.get());
  columns.push_back(col3.get());
  columns.push_back(col4.get());
  columns.push_back(col5.get());
  auto expected = cudf::table{columns.data(), 6};
  EXPECT_EQ(columns.size(), expected.num_columns());

  auto filepath =
      temp_env->get_temp_filepath("OrcWriterMultiColumnWithNulls.orc");

  cudf::orc_write_arg out_args{cudf::sink_info{filepath}};
  out_args.table = expected;
  cudf::write_orc(out_args);

  cudf::orc_read_arg in_args{cudf::source_info{filepath}};
  in_args.use_index = false;
  auto result = cudf::read_orc(in_args);

  tables_are_equal(expected, result);
}

TEST_P(OrcWriterValueParamTest, Timestamps) {
  constexpr auto num_rows = 100;
  auto values_fn = [](size_t row) { return cudf::timestamp{std::rand() / 10}; };
  auto valids_fn = [](size_t row) { return true; };

  cudf::test::column_wrapper<cudf::timestamp> col{num_rows, values_fn,
                                                  valids_fn};
  column_set_name(col.get(), "col_timestamp");
  col.get()->dtype_info.time_unit = GetParam();

  auto gdf_col = col.get();
  auto expected = cudf::table{&gdf_col, 1};
  EXPECT_EQ(1, expected.num_columns());

  auto filepath = temp_env->get_temp_filepath("OrcWriterTimestamps");

  cudf::orc_write_arg out_args{cudf::sink_info{filepath}};
  out_args.table = expected;
  cudf::write_orc(out_args);

  cudf::orc_read_arg in_args{cudf::source_info{filepath}};
  in_args.use_index = false;
  in_args.timestamp_unit = GetParam();
  auto result = cudf::read_orc(in_args);

  tables_are_equal(expected, result);
}
INSTANTIATE_TEST_CASE_P(OrcWriter, OrcWriterValueParamTest,
                        testing::Values(TIME_UNIT_s, TIME_UNIT_ms, TIME_UNIT_us,
                                        TIME_UNIT_ns));

TEST_F(OrcWriterTest, Strings) {
  std::vector<const char *> data{"Monday", "Monday", "Friday", "Monday",
                                 "Friday", "Friday", "Friday", "Funday"};
  gdf_size_type column_size = data.size();

  cudf::test::column_wrapper<int> col0{random_values<int>(column_size),
                                       [](size_t row) { return true; }};
  cudf::test::column_wrapper<cudf::nvstring_category> col1{column_size,
                                                           data.data()};
  cudf::test::column_wrapper<float> col2{random_values<float>(column_size),
                                         [](size_t row) { return true; }};

  column_set_name(col0.get(), "col_other");
  column_set_name(col1.get(), "col_string");
  column_set_name(col2.get(), "col_another");

  std::vector<gdf_column *> columns;
  columns.push_back(col0.get());
  columns.push_back(col1.get());
  columns.push_back(col2.get());
  auto expected = cudf::table{columns.data(), 3};
  EXPECT_EQ(3, expected.num_columns());

  auto filepath = temp_env->get_temp_filepath("OrcWriterStrings");

  cudf::orc_write_arg out_args{cudf::sink_info{filepath}};
  out_args.table = expected;
  cudf::write_orc(out_args);

  cudf::orc_read_arg in_args{cudf::source_info{filepath}};
  in_args.use_index = false;
  auto result = cudf::read_orc(in_args);

  // Need to compare the string data themselves as the column types are
  // different (GDF_STRING_CATEGORY vs GDF_STRING)
  auto expect_strs = nvcategory_to_strings(
      static_cast<NVCategory *>(expected.get_column(1)->dtype_info.category));
  auto result_strs = nvstrings_to_strings(
      static_cast<NVStrings *>(result.get_column(1)->data));

  ASSERT_THAT(expect_strs, ::testing::ContainerEq(result_strs));
}
