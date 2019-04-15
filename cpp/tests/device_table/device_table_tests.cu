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

#include <thrust/device_vector.h>
#include <dataframe/device_table.cuh>
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tests/utilities/column_wrapper.cuh"
#include "tests/utilities/cudf_test_fixtures.h"
#include "tests/utilities/cudf_test_utils.cuh"
#include "types.hpp"

struct DeviceTableTest : GdfTest {};

TEST_F(DeviceTableTest, First) {
  constexpr int size{1000};

  auto all_zeros = [](auto index) { return 0; };
  auto all_valid = [](auto index) { return true; };

  cudf::test::column_wrapper<int32_t> col0(size, all_zeros, all_valid);
  cudf::test::column_wrapper<float> col1(size, all_zeros, all_valid);
  cudf::test::column_wrapper<double> col2(size, all_zeros, all_valid);
  cudf::test::column_wrapper<int8_t> col3(size, all_zeros, all_valid);

  std::vector<gdf_column*> gdf_cols{col0, col1, col2, col3};

  device_table table(4, gdf_cols.data());

  // Table attributes such as number of rows/columns should
  // match expected
  EXPECT_EQ(size, table.num_rows());
  EXPECT_EQ(4, table.num_columns());

  // Pointers to the `gdf_column` should be identical
  EXPECT_EQ(col0.get(), table.get_column(0));
  EXPECT_EQ(col1.get(), table.get_column(1));
  EXPECT_EQ(col2.get(), table.get_column(2));
  EXPECT_EQ(col3.get(), table.get_column(3));

  gdf_column** cols = table.columns();
  EXPECT_EQ(col0.get(), cols[0]);
  EXPECT_EQ(col1.get(), cols[1]);
  EXPECT_EQ(col2.get(), cols[2]);
  EXPECT_EQ(col3.get(), cols[3]);

  // gdf_columns should equal the column_wrappers
  EXPECT_TRUE(col0 == *table.get_column(0));
  EXPECT_TRUE(col1 == *table.get_column(1));
  EXPECT_TRUE(col2 == *table.get_column(2));
  EXPECT_TRUE(col3 == *table.get_column(3));

  int const expected_row_byte_size =
      sizeof(int32_t) + sizeof(float) + sizeof(double) + sizeof(int8_t);
  EXPECT_EQ(expected_row_byte_size, table.get_row_size_bytes());
}