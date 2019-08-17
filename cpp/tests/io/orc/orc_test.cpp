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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <tests/utilities/cudf_test_fixtures.h>
#include <tests/io/io_test_utils.hpp>
#include <tests/utilities/column_wrapper.cuh>

TempDirTestEnvironment *const temp_env = static_cast<TempDirTestEnvironment *>(
    ::testing::AddGlobalTestEnvironment(new TempDirTestEnvironment));
struct orc_writer_test : GdfTest {};

namespace {

void column_set_name(gdf_column &col, const std::string &name) {
  if (col.col_name) {
    free(col.col_name);
  }
  col.col_name = (char *)malloc(name.length() + 1);
  std::strcpy(col.col_name, name.c_str());
}

auto columns_are_equal(gdf_column const &lhs, gdf_column const &rhs) {
  if (gdf_equal_columns(lhs, rhs)) {
    return ::testing::AssertionSuccess();
  } else {
    std::ostringstream buffer;
    buffer << std::endl;
    buffer << "    lhs:" << std::endl;
    print_gdf_column(&lhs, 10, buffer);
    buffer << "    rhs:" << std::endl;
    print_gdf_column(&rhs, 10, buffer);

    return ::testing::AssertionFailure() << buffer.str();
  }
}

void tables_are_equal(cudf::table &lhs, cudf::table &rhs) {
  EXPECT_EQ(lhs.num_columns(), rhs.num_columns());
  auto expected = lhs.begin();
  auto result = rhs.begin();
  while (result != rhs.end()) {
    EXPECT_TRUE(columns_are_equal(*(*expected++), *(*result++)));
  }
}

}  // namespace

TEST_F(orc_writer_test, Basic) {
  constexpr auto num_rows = 100;
  // cudf::test::column_wrapper<int8_t> col0{random_values<int8_t>(rows)};
  // cudf::test::column_wrapper<int32_t> col1{random_values<int32_t>(rows)};
  cudf::test::column_wrapper<float> col2{
      random_values<float>(num_rows), [](size_t row) { return true; }};
  column_set_name(col2, "floats");
  cudf::test::column_wrapper<double> col3{
      random_values<double>(num_rows), [](size_t row) { return true; }};
  column_set_name(col3, "doubles");

  std::vector<gdf_column *> cols;
  // cols.push_back(col0.get());
  // cols.push_back(col1.get());
  cols.push_back(col2.get());
  cols.push_back(col3.get());
  auto expected = cudf::table{cols.data(), 2};
  EXPECT_EQ(cols.size(), expected.num_columns());

  const std::string file_path = temp_env->get_temp_dir() + "OrcWriterBasic.orc";

  cudf::orc_write_arg out_args{cudf::sink_info{file_path}};
  out_args.table = expected;
  cudf::write_orc(out_args);

  cudf::orc_read_arg in_args{cudf::source_info{file_path}};
  in_args.use_index = false;
  auto result = cudf::read_orc(in_args);

  tables_are_equal(expected, result);
}
