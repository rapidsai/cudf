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
 * @brief Base test fixture for ORC writer
 **/
struct orc_writer_test : GdfTest {};

/**
 * @brief Typed test fixture for type-parameterized ORC writer tests
 **/
template <typename T>
struct orc_writer_typed_test : orc_writer_test {};

/**
 * @brief cuDF types that can be written to ORC types
 **/
using test_types = ::testing::Types<int8_t, float, double>;
TYPED_TEST_CASE(orc_writer_typed_test, test_types);

namespace {

/**
 * @brief Helper function to set column name
 **/
void column_set_name(gdf_column *col, const std::string &name) {
  if (col->col_name) {
    free(col->col_name);
  }
  col->col_name = static_cast<char *>(malloc(name.length() + 1));
  std::snprintf(col->col_name, name.length() + 1, "%s", name.c_str());
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

TYPED_TEST(orc_writer_typed_test, SingleColumn) {
  constexpr auto num_rows = 100;
  cudf::test::column_wrapper<TypeParam> col{random_values<TypeParam>(num_rows),
                                            [](size_t row) { return true; }};
  column_set_name(col.get(), "col_" + std::string(typeid(TypeParam).name()));

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

TYPED_TEST(orc_writer_typed_test, SingleColumnWithNulls) {
  constexpr auto num_rows = 100;
  auto nulls_threshold = 20;
  auto valids_func = [=](size_t row) {
    return (row < nulls_threshold || row > (num_rows - nulls_threshold));
  };
  cudf::test::column_wrapper<TypeParam> col{random_values<TypeParam>(num_rows),
                                            valids_func};
  column_set_name(col.get(), "col_" + std::string(typeid(TypeParam).name()));

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

TEST_F(orc_writer_test, MultiColumn) {
  constexpr auto num_rows = 100;
  auto valids_func = [](size_t row) { return true; };
  cudf::test::column_wrapper<int8_t> col0{random_values<int8_t>(num_rows),
                                          valids_func};
  cudf::test::column_wrapper<float> col2{random_values<float>(num_rows),
                                         valids_func};
  cudf::test::column_wrapper<double> col3{random_values<double>(num_rows),
                                          valids_func};
  column_set_name(col0.get(), "int8s");
  column_set_name(col2.get(), "floats");
  column_set_name(col3.get(), "doubles");

  std::vector<gdf_column *> cols;
  cols.push_back(col0.get());
  // cols.push_back(col1.get());
  cols.push_back(col2.get());
  cols.push_back(col3.get());
  auto expected = cudf::table{cols.data(), 3};
  EXPECT_EQ(cols.size(), expected.num_columns());

  auto filepath = temp_env->get_temp_filepath("OrcWriterMultiColumn.orc");

  cudf::orc_write_arg out_args{cudf::sink_info{filepath}};
  out_args.table = expected;
  cudf::write_orc(out_args);

  cudf::orc_read_arg in_args{cudf::source_info{filepath}};
  in_args.use_index = false;
  auto result = cudf::read_orc(in_args);

  tables_are_equal(expected, result);
}

TEST_F(orc_writer_test, MultiColumnWithNulls) {
  constexpr auto num_rows = 100;
  cudf::test::column_wrapper<int8_t> col0{
      random_values<int8_t>(num_rows), [=](size_t row) { return (row < 10); }};
  cudf::test::column_wrapper<float> col2{
      random_values<float>(num_rows),
      [=](size_t row) { return (row >= 40 || row <= 60); }};
  cudf::test::column_wrapper<double> col3{
      random_values<double>(num_rows), [=](size_t row) { return (row > 80); }};
  column_set_name(col0.get(), "int8s");
  column_set_name(col2.get(), "floats");
  column_set_name(col3.get(), "doubles");

  std::vector<gdf_column *> cols;
  cols.push_back(col0.get());
  // cols.push_back(col1.get());
  cols.push_back(col2.get());
  cols.push_back(col3.get());
  auto expected = cudf::table{cols.data(), 3};
  EXPECT_EQ(cols.size(), expected.num_columns());

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
