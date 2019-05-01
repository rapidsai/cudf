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

#include <tests/utilities/cudf_test_fixtures.h>
#include <tests/utilities/column_wrapper.cuh>
#include <table/table.hpp>
#include <utilities/type_dispatcher.hpp>

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
  while (result != table.end()) {
    columns_are_equal(*expected++, *result++);
  }

  columns_are_equal(cols[0], table.get_column(0));
  columns_are_equal(cols[1], table.get_column(1));
  columns_are_equal(cols[2], table.get_column(2));
}

TYPED_TEST(TableTest, ConstructColumns) {
  const auto size = this->random_size();
  std::vector<gdf_dtype> dtypes{GDF_INT8, GDF_INT32, GDF_FLOAT32,
                                cudf::gdf_dtype_of<TypeParam>()};

  if (GDF_TIMESTAMP == cudf::gdf_dtype_of<TypeParam>()) {
    // Can't invoke a constructor with mutliple arguments in the body of a macro
    // because the comma confuses the macro. Use a lambda wrapper as a
    // workaround
    auto constructor = [size, dtypes]() { cudf::table{size, dtypes}; };
    EXPECT_THROW(constructor(), cudf::logic_error);
  } else {
    // Construct columns, no bitmask allocation
    cudf::table t{size, dtypes};

    for (gdf_size_type i = 0; i < t.num_columns(); ++i) {
      gdf_column* col = t.get_column(i);
      EXPECT_NE(nullptr, col->data);
      EXPECT_EQ(nullptr, col->valid);
      EXPECT_EQ(size, col->size);
      EXPECT_EQ(0, col->null_count);
      EXPECT_EQ(dtypes[i], col->dtype);
    }

    // User responsible for freeing columns...
    std::for_each(t.begin(), t.end(), [](gdf_column* col) {
      RMM_FREE(col->data, 0);
      RMM_FREE(col->valid, 0);
      delete col;
    });
    delete[] t.begin();
  }
}

TYPED_TEST(TableTest, ConstructColumnsWithBitmasksNulls) {
  const auto size = this->random_size();
  std::vector<gdf_dtype> dtypes{GDF_INT64, GDF_FLOAT64, GDF_INT8,
                                cudf::gdf_dtype_of<TypeParam>()};

  if (GDF_TIMESTAMP == cudf::gdf_dtype_of<TypeParam>()) {
    // Can't invoke a constructor with multiple arguments in the body of a macro
    // because the comma confuses the macro. Use a lambda wrapper as a
    // workaround
    auto constructor = [size, dtypes]() { cudf::table{size, dtypes}; };
    EXPECT_THROW(constructor(), cudf::logic_error);
  } else {
    // Construct columns, each with a bitmask allocation indicating all values
    // are null
    cudf::table t{size, dtypes, true, false};

    for (gdf_size_type i = 0; i < t.num_columns(); ++i) {
      gdf_column* col = t.get_column(i);
      EXPECT_NE(nullptr, col->data);
      EXPECT_NE(nullptr, col->valid);
      EXPECT_EQ(size, col->size);
      EXPECT_EQ(0, col->null_count);
      EXPECT_EQ(dtypes[i], col->dtype);

      gdf_size_type valid_count{-1};
      gdf_count_nonzero_mask(col->valid, col->size, &valid_count);
      EXPECT_EQ(0, valid_count);
    }

    // User responsible for freeing columns...
    std::for_each(t.begin(), t.end(), [](gdf_column* col) {
      RMM_FREE(col->data, 0);
      RMM_FREE(col->valid, 0);
      delete col;
    });
    delete[] t.begin();
  }
}

TYPED_TEST(TableTest, ConstructColumnsWithBitmasksValid) {
  const auto size = this->random_size();
  std::vector<gdf_dtype> dtypes{GDF_INT64, GDF_FLOAT64, GDF_INT8,
                                cudf::gdf_dtype_of<TypeParam>()};

  if (GDF_TIMESTAMP == cudf::gdf_dtype_of<TypeParam>()) {
    // Can't invoke a constructor with multiple arguments in the body of a macro
    // because the comma confuses the macro. Use a lambda wrapper as a
    // workaround
    auto constructor = [size, dtypes]() { cudf::table{size, dtypes}; };
    EXPECT_THROW(constructor(), cudf::logic_error);
  } else {
    // Construct columns, each with a bitmask allocation indicating all values
    // are null
    cudf::table t{size, dtypes, true, true};

    for (gdf_size_type i = 0; i < t.num_columns(); ++i) {
      gdf_column* col = t.get_column(i);
      EXPECT_NE(nullptr, col->data);
      EXPECT_NE(nullptr, col->valid);
      EXPECT_EQ(size, col->size);
      EXPECT_EQ(0, col->null_count);
      EXPECT_EQ(dtypes[i], col->dtype);

      gdf_size_type valid_count{-1};
      gdf_count_nonzero_mask(col->valid, col->size, &valid_count);
      EXPECT_EQ(size, valid_count);
    }

    // User responsible for freeing columns...
    std::for_each(t.begin(), t.end(), [](gdf_column* col) {
      RMM_FREE(col->data, 0);
      RMM_FREE(col->valid, 0);
      delete col;
    });
    delete[] t.begin();
  }
}