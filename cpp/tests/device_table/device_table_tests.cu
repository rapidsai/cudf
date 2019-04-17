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

#include <dataframe/device_table.cuh>
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tests/utilities/column_wrapper.cuh"
#include "tests/utilities/cudf_test_fixtures.h"
#include "tests/utilities/cudf_test_utils.cuh"
#include "types.hpp"

#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/logical.h>

struct DeviceTableTest : GdfTest {
  gdf_size_type const size{1000};
};

struct row_has_nulls {
  device_table* t;

  row_has_nulls(device_table* _t) : t{_t} {}

  __device__ bool operator()(int row_index) {
    return t->row_has_nulls(row_index);
  }
};

/**---------------------------------------------------------------------------*
 * @brief Compares if a row in one table is equal to all rows in another table.
 *
 *---------------------------------------------------------------------------**/
struct all_rows_equal {
  device_table* lhs;
  device_table* rhs;
  bool nulls_are_equal;

  all_rows_equal(device_table* _lhs, device_table* _rhs,
                 bool _nulls_are_equal = false)
      : lhs{_lhs}, rhs{_rhs}, nulls_are_equal{_nulls_are_equal} {}

  /**---------------------------------------------------------------------------*
   * @brief Returns true if row `lhs_index` in the `lhs` table is equal to every
   * row in the `rhs` table.
   *
   *---------------------------------------------------------------------------**/
  __device__ bool operator()(int lhs_index) {
    auto row_equality = [this, lhs_index](gdf_size_type rhs_index) {
      return lhs->rows_equal(*rhs, lhs_index, rhs_index, nulls_are_equal);
    };
    return thrust::all_of(thrust::seq, thrust::make_counting_iterator(0),
                          thrust::make_counting_iterator(rhs->num_rows()),
                          row_equality);
  }
};

struct row_comparison {
  device_table* lhs;
  device_table* rhs;
  bool nulls_are_equal;

  row_comparison(device_table* _lhs, device_table* _rhs,
                 bool _nulls_are_equal = false)
      : lhs{_lhs}, rhs{_rhs}, nulls_are_equal{_nulls_are_equal} {}

  __device__ bool operator()(
      thrust::pair<gdf_size_type, gdf_size_type> indices) {
    return lhs->rows_equal(*rhs, indices.first, indices.second,
                           nulls_are_equal);
  }
};

TEST_F(DeviceTableTest, HostFunctions) {
  const int val{42};
  auto init_values = [val](auto index) { return val; };
  auto all_valid = [](auto index) { return true; };

  cudf::test::column_wrapper<int32_t> col0(size, init_values, all_valid);
  cudf::test::column_wrapper<float> col1(size, init_values, all_valid);
  cudf::test::column_wrapper<double> col2(size, init_values, all_valid);
  cudf::test::column_wrapper<int8_t> col3(size, init_values, all_valid);

  std::vector<gdf_column*> gdf_cols{col0, col1, col2, col3};

  auto table = device_table::create(gdf_cols.size(), gdf_cols.data());

  // Table attributes such as number of rows/columns should
  // match expected
  EXPECT_EQ(size, table->num_rows());
  EXPECT_EQ(4, table->num_columns());

  // Pointers to the `gdf_column` should be identical
  EXPECT_EQ(col0.get(), table->get_column(0));
  EXPECT_EQ(col1.get(), table->get_column(1));
  EXPECT_EQ(col2.get(), table->get_column(2));
  EXPECT_EQ(col3.get(), table->get_column(3));

  gdf_column** cols = table->columns();
  EXPECT_EQ(col0.get(), cols[0]);
  EXPECT_EQ(col1.get(), cols[1]);
  EXPECT_EQ(col2.get(), cols[2]);
  EXPECT_EQ(col3.get(), cols[3]);

  // gdf_columns should equal the column_wrappers
  EXPECT_TRUE(col0 == *table->get_column(0));
  EXPECT_TRUE(col1 == *table->get_column(1));
  EXPECT_TRUE(col2 == *table->get_column(2));
  EXPECT_TRUE(col3 == *table->get_column(3));

  int const expected_row_byte_size =
      sizeof(int32_t) + sizeof(float) + sizeof(double) + sizeof(int8_t);
  EXPECT_EQ(expected_row_byte_size, table->get_row_size_bytes());
}

TEST_F(DeviceTableTest, AllRowsEqualNoNulls) {
  const int val{42};
  auto init_values = [val](auto index) { return val; };
  auto all_valid = [](auto index) { return true; };

  // 4 columns will all rows equal, no nulls
  cudf::test::column_wrapper<int32_t> col0(size, init_values, all_valid);
  cudf::test::column_wrapper<float> col1(size, init_values, all_valid);
  cudf::test::column_wrapper<double> col2(size, init_values, all_valid);
  cudf::test::column_wrapper<int8_t> col3(size, init_values, all_valid);

  std::vector<gdf_column*> gdf_cols{col0, col1, col2, col3};

  auto table = device_table::create(gdf_cols.size(), gdf_cols.data());

  // Every row should be valid
  EXPECT_FALSE(thrust::all_of(
      rmm::exec_policy()->on(0), thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(size), row_has_nulls(table.get())));

  // Every row should be equal to every other row regardless of NULL ?= NULL
  EXPECT_TRUE(thrust::all_of(rmm::exec_policy()->on(0),
                             thrust::make_counting_iterator(0),
                             thrust::make_counting_iterator(size),
                             all_rows_equal(table.get(), table.get(), true)));
  EXPECT_TRUE(thrust::all_of(rmm::exec_policy()->on(0),
                             thrust::make_counting_iterator(0),
                             thrust::make_counting_iterator(size),
                             all_rows_equal(table.get(), table.get(), false)));
}

TEST_F(DeviceTableTest, AllRowsEqualWithNulls) {
  const int val{42};
  auto init_values = [val](auto index) { return val; };
  auto all_valid = [](auto index) { return true; };
  auto all_null = [](auto index) { return false; };

  // 4 columns with all rows equal, last column is all nulls
  cudf::test::column_wrapper<int32_t> col0(size, init_values, all_valid);
  cudf::test::column_wrapper<float> col1(size, init_values, all_valid);
  cudf::test::column_wrapper<double> col2(size, init_values, all_valid);
  cudf::test::column_wrapper<int8_t> col3(size, init_values, all_null);

  std::vector<gdf_column*> gdf_cols{col0, col1, col2, col3};

  auto table = device_table::create(gdf_cols.size(), gdf_cols.data());

  // Every row should contain nulls
  EXPECT_TRUE(thrust::all_of(
      rmm::exec_policy()->on(0), thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(size - 1), row_has_nulls(table.get())));

  // If NULL != NULL, no row can equal any other row
  EXPECT_FALSE(thrust::all_of(rmm::exec_policy()->on(0),
                              thrust::make_counting_iterator(0),
                              thrust::make_counting_iterator(size),
                              all_rows_equal(table.get(), table.get(), false)));

  // If NULL == NULL, all rows should be equal
  EXPECT_TRUE(thrust::all_of(rmm::exec_policy()->on(0),
                             thrust::make_counting_iterator(0),
                             thrust::make_counting_iterator(size),
                             all_rows_equal(table.get(), table.get(), true)));
}

TEST_F(DeviceTableTest, AllRowsDifferentWithNulls) {
  int const val{42};
  auto init_values = [val](auto index) { return index; };
  auto all_valid = [](auto index) { return true; };
  auto all_null = [](auto index) { return false; };

  // 4 columns with all rows different, last column is all nulls
  cudf::test::column_wrapper<int32_t> col0(size, init_values, all_valid);
  cudf::test::column_wrapper<float> col1(size, init_values, all_valid);
  cudf::test::column_wrapper<double> col2(size, init_values, all_valid);
  cudf::test::column_wrapper<int8_t> col3(size, init_values, all_null);

  std::vector<gdf_column*> gdf_cols{col0, col1, col2, col3};

  auto table = device_table::create(gdf_cols.size(), gdf_cols.data());

  // Every row should contain nulls
  EXPECT_TRUE(thrust::all_of(
      rmm::exec_policy()->on(0), thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(size - 1), row_has_nulls(table.get())));

  // Every row should
  EXPECT_FALSE(thrust::all_of(rmm::exec_policy()->on(0),
                              thrust::make_counting_iterator(0),
                              thrust::make_counting_iterator(size),
                              all_rows_equal(table.get(), table.get(), false)));
  EXPECT_FALSE(thrust::all_of(rmm::exec_policy()->on(0),
                              thrust::make_counting_iterator(0),
                              thrust::make_counting_iterator(size),
                              all_rows_equal(table.get(), table.get(), true)));
}

// Test where a single column has every other value null,
// should mean that every other row is null

// Test where one table is identical to the other

// test where one table is the reverse of the other
