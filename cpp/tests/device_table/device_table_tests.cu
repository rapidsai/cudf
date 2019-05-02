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

#include <table/device_table.cuh>
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tests/utilities/column_wrapper.cuh"
#include "tests/utilities/cudf_test_fixtures.h"
#include "tests/utilities/cudf_test_utils.cuh"
#include "types.hpp"

#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/logical.h>

#include <numeric>
#include <random>

struct DeviceTableTest : GdfTest {
  gdf_size_type const size{1000};
};

/**---------------------------------------------------------------------------*
 * @brief Compares if a row in one table is equal to all rows in another table.
 *
 *---------------------------------------------------------------------------**/
struct all_rows_equal {
  device_table lhs;
  device_table rhs;
  bool nulls_are_equal;

  all_rows_equal(device_table _lhs, device_table _rhs,
                 bool _nulls_are_equal = false)
      : lhs{_lhs}, rhs{_rhs}, nulls_are_equal{_nulls_are_equal} {}

  /**---------------------------------------------------------------------------*
   * @brief Returns true if row `lhs_index` in the `lhs` table is equal to every
   * row in the `rhs` table.
   *
   *---------------------------------------------------------------------------**/
  __device__ bool operator()(int lhs_index) {
    auto row_equality = [this, lhs_index](gdf_size_type rhs_index) {
      return rows_equal(lhs, lhs_index, rhs, rhs_index, nulls_are_equal);
    };
    return thrust::all_of(thrust::seq, thrust::make_counting_iterator(0),
                          thrust::make_counting_iterator(rhs.num_rows()),
                          row_equality);
  }
};

struct row_comparison {
  device_table lhs;
  device_table rhs;
  bool nulls_are_equal;

  using index_pair = thrust::tuple<gdf_size_type, gdf_size_type>;

  row_comparison(device_table _lhs, device_table _rhs,
                 bool _nulls_are_equal = false)
      : lhs{_lhs}, rhs{_rhs}, nulls_are_equal{_nulls_are_equal} {}

  __device__ bool operator()(index_pair const& indices) {
    return rows_equal(lhs, thrust::get<0>(indices), rhs,
                      thrust::get<1>(indices), nulls_are_equal);
  }
};

struct row_hasher {
  device_table t;
  row_hasher(device_table _t) : t{_t} {}
  __device__ hash_value_type operator()(gdf_size_type row_index) {
    return hash_row(t, row_index);
  }
};

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

  // Every row should be equal to every other row regardless of NULL ?= NULL
  EXPECT_TRUE(thrust::all_of(rmm::exec_policy()->on(0),
                             thrust::make_counting_iterator(0),
                             thrust::make_counting_iterator(size),
                             all_rows_equal(*table, *table, true)));
  EXPECT_TRUE(thrust::all_of(rmm::exec_policy()->on(0),
                             thrust::make_counting_iterator(0),
                             thrust::make_counting_iterator(size),
                             all_rows_equal(*table, *table, false)));

  // Compute hash value of every row
  thrust::device_vector<hash_value_type> row_hashes(table->num_rows());
  thrust::tabulate(row_hashes.begin(), row_hashes.end(), row_hasher{*table});

  // All hash values should be equal
  EXPECT_TRUE(thrust::equal(row_hashes.begin() + 1, row_hashes.end(),
                            row_hashes.begin()));
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

  // If NULL != NULL, no row can equal any other row
  EXPECT_FALSE(thrust::all_of(rmm::exec_policy()->on(0),
                              thrust::make_counting_iterator(0),
                              thrust::make_counting_iterator(size),
                              all_rows_equal(*table, *table, false)));

  // If NULL == NULL, all rows should be equal
  EXPECT_TRUE(thrust::all_of(rmm::exec_policy()->on(0),
                             thrust::make_counting_iterator(0),
                             thrust::make_counting_iterator(size),
                             all_rows_equal(*table, *table, true)));

  // Compute hash value of every row
  thrust::device_vector<hash_value_type> row_hashes(table->num_rows());
  thrust::tabulate(row_hashes.begin(), row_hashes.end(), row_hasher{*table});

  // All hash values should be equal because hash_row should ignore nulls
  EXPECT_TRUE(thrust::equal(row_hashes.begin() + 1, row_hashes.end(),
                            row_hashes.begin()));
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

  // If NULL==NULL, every row should be equal to itself
  thrust::device_vector<gdf_size_type> indices(table->num_rows());
  thrust::sequence(indices.begin(), indices.end());
  EXPECT_TRUE(thrust::all_of(rmm::exec_policy()->on(0),
                             thrust::make_zip_iterator(thrust::make_tuple(
                                 indices.begin(), indices.begin())),
                             thrust::make_zip_iterator(thrust::make_tuple(
                                 indices.end(), indices.end())),
                             row_comparison{*table, *table, true}));

  // If NULL!=NULL, every row should *not* be equal to itself
  EXPECT_FALSE(thrust::all_of(rmm::exec_policy()->on(0),
                              thrust::make_zip_iterator(thrust::make_tuple(
                                  indices.begin(), indices.begin())),
                              thrust::make_zip_iterator(thrust::make_tuple(
                                  indices.end(), indices.end())),
                              row_comparison{*table, *table, false}));

  // Compute hash value of every row
  thrust::device_vector<hash_value_type> row_hashes(table->num_rows());
  thrust::tabulate(row_hashes.begin(), row_hashes.end(), row_hasher{*table});

  // All hash values should be NOT be equal
  EXPECT_FALSE(thrust::equal(row_hashes.begin() + 1, row_hashes.end(),
                             row_hashes.begin()));

  // Every row should be different from every other row other than itself
  for (gdf_size_type i = 0; i < table->num_rows(); ++i) {
    thrust::device_vector<gdf_size_type> left_indices(table->num_rows(), i);
    thrust::device_vector<gdf_size_type> right_indices(table->num_rows());
    thrust::sequence(right_indices.begin(), right_indices.end());

    // Remove indices comparing a row against itself
    left_indices.erase(left_indices.begin() + i);
    right_indices.erase(right_indices.begin() + i);

    // Ensure row `i` is not equal to every other row `j`, `i != j`
    EXPECT_FALSE(thrust::all_of(
        rmm::exec_policy()->on(0),
        thrust::make_zip_iterator(
            thrust::make_tuple(left_indices.begin(), right_indices.begin())),
        thrust::make_zip_iterator(
            thrust::make_tuple(left_indices.end(), right_indices.end())),
        row_comparison{*table, *table, true}));
  }
}

TEST_F(DeviceTableTest, TwoTablesAllRowsEqual) {
  int const val{42};
  auto init_values = [val](auto index) { return index; };
  auto random_values = [](auto index) {
    return std::default_random_engine{}();
  };
  auto all_valid = [](auto index) { return true; };
  auto all_null = [](auto index) { return false; };

  cudf::test::column_wrapper<int32_t> left_col0(size, init_values, all_valid);
  cudf::test::column_wrapper<float> left_col1(size, init_values, all_valid);
  cudf::test::column_wrapper<double> left_col2(size, init_values, all_valid);
  cudf::test::column_wrapper<int8_t> left_col3(size, random_values, all_null);
  std::vector<gdf_column*> left_cols{left_col0, left_col1, left_col2,
                                     left_col3};
  auto left_table = device_table::create(left_cols.size(), left_cols.data());

  cudf::test::column_wrapper<int32_t> right_col0(size, init_values, all_valid);
  cudf::test::column_wrapper<float> right_col1(size, init_values, all_valid);
  cudf::test::column_wrapper<double> right_col2(size, init_values, all_valid);
  cudf::test::column_wrapper<int8_t> right_col3(size, random_values, all_null);
  std::vector<gdf_column*> right_cols{right_col0, right_col1, right_col2,
                                      right_col3};
  auto right_table = device_table::create(right_cols.size(), right_cols.data());

  // If NULL==NULL, left_table row @ i should equal right_table row @ i
  thrust::device_vector<gdf_size_type> indices(left_table->num_rows());
  thrust::sequence(indices.begin(), indices.end());
  EXPECT_TRUE(thrust::all_of(rmm::exec_policy()->on(0),
                             thrust::make_zip_iterator(thrust::make_tuple(
                                 indices.begin(), indices.begin())),
                             thrust::make_zip_iterator(thrust::make_tuple(
                                 indices.end(), indices.end())),
                             row_comparison{*left_table, *right_table, true}));

  // If NULL!=NULL, left_table row @ i should NOT equal right_table row @ i
  EXPECT_FALSE(
      thrust::all_of(rmm::exec_policy()->on(0),
                     thrust::make_zip_iterator(
                         thrust::make_tuple(indices.begin(), indices.begin())),
                     thrust::make_zip_iterator(
                         thrust::make_tuple(indices.end(), indices.end())),
                     row_comparison{*left_table, *right_table, false}));
}

struct row_copier {
  device_table target;
  device_table source;

  using index_pair = thrust::tuple<gdf_size_type, gdf_size_type>;

  row_copier(device_table _target, device_table _source)
      : target{_target}, source{_source} {}

  __device__ void operator()(index_pair const& indices) {
    copy_row(target, thrust::get<0>(indices), source, thrust::get<1>(indices));
  }
};

TEST_F(DeviceTableTest, CopyRowsNoNulls) {
  int const val{42};
  auto init_values = [val](auto index) { return index; };
  auto all_valid = [](auto index) { return true; };

  cudf::test::column_wrapper<int32_t> source_col0(size, init_values, all_valid);
  cudf::test::column_wrapper<float> source_col1(size, init_values, all_valid);
  cudf::test::column_wrapper<double> source_col2(size, init_values, all_valid);
  cudf::test::column_wrapper<int8_t> source_col3(size, init_values, all_valid);
  std::vector<gdf_column*> source_cols{source_col0, source_col1, source_col2,
                                       source_col3};
  auto source_table =
      device_table::create(source_cols.size(), source_cols.data());

  cudf::test::column_wrapper<int32_t> target_col0(size);
  cudf::test::column_wrapper<float> target_col1(size);
  cudf::test::column_wrapper<double> target_col2(size);
  cudf::test::column_wrapper<int8_t> target_col3(size);
  std::vector<gdf_column*> target_cols{target_col0, target_col1, target_col2,
                                       target_col3};
  auto target_table =
      device_table::create(target_cols.size(), target_cols.data());

  // Copy a random row from the source table to a random row in the target table
  // Thrust doesn't have a `shuffle` algorithm, so we've got to do it on the
  // host
  std::vector<gdf_size_type> indices(source_table->num_rows());
  std::iota(indices.begin(), indices.end(), 0);
  std::shuffle(indices.begin(), indices.end(), std::default_random_engine{});
  thrust::device_vector<gdf_size_type> target_indices(indices);

  std::shuffle(indices.begin(), indices.end(), std::default_random_engine{});
  thrust::device_vector<gdf_size_type> source_indices(indices);

  // Copy source_table row @ source_indices[i] to target_table @
  // target_indices[i]
  EXPECT_NO_THROW(thrust::for_each(
      rmm::exec_policy()->on(0),
      thrust::make_zip_iterator(
          thrust::make_tuple(target_indices.begin(), source_indices.begin())),
      thrust::make_zip_iterator(
          thrust::make_tuple(target_indices.end(), source_indices.end())),
      row_copier{*target_table, *source_table}));

  // ensure source_table row @ source_indices[i] == target_table row @
  // target_indices[i]
  EXPECT_TRUE(thrust::all_of(
      rmm::exec_policy()->on(0),
      thrust::make_zip_iterator(
          thrust::make_tuple(source_indices.begin(), target_indices.begin())),
      thrust::make_zip_iterator(
          thrust::make_tuple(source_indices.end(), target_indices.end())),
      row_comparison{*source_table, *target_table}));
}