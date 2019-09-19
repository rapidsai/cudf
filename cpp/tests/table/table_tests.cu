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
#include <cudf/legacy/table.hpp>
#include <tests/utilities/column_wrapper.cuh>
#include <cudf/utilities/legacy/type_dispatcher.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <random>

template <typename T>
using column_wrapper = cudf::test::column_wrapper<T>;

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

  std::vector<gdf_dtype_extra_info> dtype_infos{4, {TIME_UNIT_NONE}};

  if (GDF_TIMESTAMP == cudf::gdf_dtype_of<TypeParam>()) {
    dtype_infos[3].time_unit = TIME_UNIT_ns;
  }

  // Construct columns, no bitmask allocation
  cudf::table t{size, dtypes, dtype_infos};

  for (gdf_size_type i = 0; i < t.num_columns(); ++i) {
    gdf_column* col = t.get_column(i);
    EXPECT_NE(nullptr, col->data);
    EXPECT_EQ(nullptr, col->valid);
    EXPECT_EQ(size, col->size);
    EXPECT_EQ(0, col->null_count);
    EXPECT_EQ(dtypes[i], col->dtype);
    EXPECT_EQ(dtype_infos[i].time_unit, col->dtype_info.time_unit);
  }

  // User responsible for freeing columns...
  std::for_each(t.begin(), t.end(), [](gdf_column* col) {
    RMM_FREE(col->data, 0);
    RMM_FREE(col->valid, 0);
    delete col;
  });
}

TYPED_TEST(TableTest, ConstructColumnsWithBitmasksNulls) {
  const auto size = this->random_size();
  std::vector<gdf_dtype> dtypes{GDF_INT64, GDF_FLOAT64, GDF_INT8,
                                cudf::gdf_dtype_of<TypeParam>()};

  std::vector<gdf_dtype_extra_info> dtype_infos{4, {TIME_UNIT_NONE}};

  if (GDF_TIMESTAMP == cudf::gdf_dtype_of<TypeParam>()) {
    dtype_infos[3].time_unit = TIME_UNIT_ns;
  }

  // Construct columns, each with a bitmask allocation indicating all values
  // are null
  cudf::table t{size, dtypes, dtype_infos, true, false};

  for (gdf_size_type i = 0; i < t.num_columns(); ++i) {
    gdf_column* col = t.get_column(i);
    EXPECT_NE(nullptr, col->data);
    EXPECT_NE(nullptr, col->valid);
    EXPECT_EQ(size, col->size);
    EXPECT_EQ(0, col->null_count);
    EXPECT_EQ(dtypes[i], col->dtype);
    EXPECT_EQ(dtype_infos[i].time_unit, col->dtype_info.time_unit);

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
}

TYPED_TEST(TableTest, ConstructColumnsWithBitmasksValid) {
  const auto size = this->random_size();
  std::vector<gdf_dtype> dtypes{GDF_INT64, GDF_FLOAT64, GDF_INT8,
                                cudf::gdf_dtype_of<TypeParam>()};


  std::vector<gdf_dtype_extra_info> dtype_infos{4, {TIME_UNIT_NONE}};

  if (GDF_TIMESTAMP == cudf::gdf_dtype_of<TypeParam>()) {
    dtype_infos[3].time_unit = TIME_UNIT_ns;
  }

  // Construct columns, each with a bitmask allocation indicating all values
  // are null
  cudf::table t{size, dtypes, dtype_infos, true, true};

  for (gdf_size_type i = 0; i < t.num_columns(); ++i) {
    gdf_column* col = t.get_column(i);
    EXPECT_NE(nullptr, col->data);
    EXPECT_NE(nullptr, col->valid);
    EXPECT_EQ(size, col->size);
    EXPECT_EQ(0, col->null_count);
    EXPECT_EQ(dtypes[i], col->dtype);
    EXPECT_EQ(dtype_infos[i].time_unit, col->dtype_info.time_unit);

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
}

TYPED_TEST(TableTest, GetTableWithSelectedColumns)
{
    column_wrapper <int8_t> col1 = column_wrapper<int8_t>({1,2,3,4},[](auto row) { return true; });
    column_wrapper <int16_t> col2 = column_wrapper<int16_t>({1,2,3,4},[](auto row) { return true; });
    column_wrapper <int32_t> col3 = column_wrapper<int32_t>({4,5,6,7},[](auto row) { return true; });
    column_wrapper <int64_t> col4 = column_wrapper<int64_t>({4,5,6,7},[](auto row) { return true; });

    std::vector<gdf_column*> cols;
    cols.push_back(col1.get());
    cols.push_back(col2.get());
    cols.push_back(col3.get());
    cols.push_back(col4.get());

    cudf::table table(cols);
    cudf::table selected_table = table.select(std::vector<gdf_size_type>{2,3});
    columns_are_equal(table.get_column(2), selected_table.get_column(0));
    columns_are_equal(table.get_column(3), selected_table.get_column(1));
}

TYPED_TEST(TableTest, SelectingMoreThanNumberOfColumns)
{
    column_wrapper <int8_t> col1 = column_wrapper<int8_t>({1,2,3,4},[](auto row) { return true; });
    column_wrapper <int16_t> col2 = column_wrapper<int16_t>({1,2,3,4},[](auto row) { return true; });

    std::vector<gdf_column*> cols;
    cols.push_back(col1.get());
    cols.push_back(col2.get());

    cudf::table table(cols);
    CUDF_EXPECT_THROW_MESSAGE (table.select(std::vector<gdf_size_type>{0,1,2}), "Requested too many columns.");
}

TYPED_TEST(TableTest, SelectingNoColumns)
{
    column_wrapper <int8_t> col1 = column_wrapper<int8_t>({1,2,3,4},[](auto row) { return true; });
    column_wrapper <int16_t> col2 = column_wrapper<int16_t>({1,2,3,4},[](auto row) { return true; });

    std::vector<gdf_column*> cols;
    cols.push_back(col1.get());
    cols.push_back(col2.get());

    cudf::table table(cols);
    cudf::table selected_table = table.select(std::vector<gdf_size_type>{});

    EXPECT_EQ(selected_table.num_columns(), 0);
}

TYPED_TEST(TableTest, ConcatTables)
{
    column_wrapper <int8_t> col1 = column_wrapper<int8_t>({1,2,3,4},[](auto row) { return true; });
    column_wrapper <int16_t> col2 = column_wrapper<int16_t>({1,2,3,4},[](auto row) { return true; });

    std::vector<gdf_column*> cols;
    cols.push_back(col1.get());
    cols.push_back(col2.get());

    cudf::table table1{std::vector<gdf_column*>{col1.get()}};
    cudf::table table2{std::vector<gdf_column*>{col2.get()}};

    cudf::table concated_table = cudf::concat (table1, table2);
    columns_are_equal(concated_table.get_column(0), col1.get());
    columns_are_equal(concated_table.get_column(1), col2.get());
}

TYPED_TEST(TableTest, ConcatTablesRowsMismatch)
{
    column_wrapper <int8_t> col1 = column_wrapper<int8_t>({1,2,3,4},[](auto row) { return true; });
    column_wrapper <int16_t> col2 = column_wrapper<int16_t>({1,2,3},[](auto row) { return true; });

    std::vector<gdf_column*> cols;
    cols.push_back(col1.get());
    cols.push_back(col2.get());

    cudf::table table1{std::vector<gdf_column*>{col1.get()}};
    cudf::table table2{std::vector<gdf_column*>{col2.get()}};

    CUDF_EXPECT_THROW_MESSAGE(cudf::concat (table1, table2), "Number of rows mismatch");
}

TYPED_TEST(TableTest, ConcatEmptyTables)
{
    cudf::table table1 = cudf::table{};
    cudf::table table2 = cudf::table{};

    cudf::table concated_table = cudf::concat (table1, table2);
    EXPECT_EQ(concated_table.num_columns(), 0);
}
