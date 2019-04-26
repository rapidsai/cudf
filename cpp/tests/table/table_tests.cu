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

#include <tests/utilities/column_wrapper.cuh>
#include <tests/utilities/cudf_test_fixtures.h>
#include <types.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <random>

template <typename T>
struct TableTest : public GdfTest {
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution{1000, 10000};
  int random_size() { return distribution(generator); }
};

using TestingTypes = ::testing::Types<int8_t, int16_t, int32_t, int64_t, float,
                                      double, cudf::date32, cudf::date64,
                                      cudf::timestamp, cudf::category>;

TYPED_TEST_CASE(TableTest, TestingTypes);

namespace {
void columns_are_equal(gdf_column const* lhs, gdf_column const* rhs) {
  EXPECT_EQ(lhs->data, rhs->data);
  EXPECT_EQ(lhs->valid, rhs->valid);
  EXPECT_EQ(lhs->dtype, rhs->dtype);
  EXPECT_EQ(lhs->size, rhs->size);
  EXPECT_EQ(lhs->null_count, rhs->null_count);
}
}  // namespace

TYPED_TEST(TableTest, SingleColumn) {
  const auto size = this->random_size();
  cudf::test::column_wrapper<TypeParam> col(size);
  gdf_column* gdf_col = col.get();
  cudf::table table(&gdf_col, 1);
  EXPECT_EQ(1, table.num_columns());
  columns_are_equal(gdf_col, *table.begin());
  columns_are_equal(gdf_col, table.get_column(0));
}

TYPED_TEST(TableTest, MultiColumn) {
  const auto size = this->random_size();
  cudf::test::column_wrapper<TypeParam> col0(size);
  cudf::test::column_wrapper<double> col1(size);
  cudf::test::column_wrapper<cudf::date32> col2(size);

  std::vector<gdf_column*> cols;
  cols.push_back(col0.get());
  cols.push_back(col1.get());
  cols.push_back(col2.get());
  cudf::table table(cols.data(), 3);
  EXPECT_EQ(3, table.num_columns());

  auto expected = cols.begin();
  auto result = table.begin();
  while(result != table.end()){
    columns_are_equal(*expected++, *result++);
  }

  columns_are_equal(cols[0], table.get_column(0));
  columns_are_equal(cols[1], table.get_column(1));
  columns_are_equal(cols[2], table.get_column(2));
}
