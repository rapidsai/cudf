
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

#include <cudf/groupby.hpp>
#include <cudf/legacy/table.hpp>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <tests/utilities/column_wrapper.cuh>
#include <tests/utilities/compare_column_wrappers.cuh>
#include <tests/utilities/cudf_test_fixtures.h>

#include <random>

template <typename T> using column_wrapper = cudf::test::column_wrapper<T>;

template <typename KeyType> struct SingleColumnGroupby : public GdfTest {
  using Key = KeyType;

  struct column_equality {
    template <typename T>
    bool operator()(gdf_column lhs, gdf_column rhs) const {
      std::unique_ptr<column_wrapper<T>> lhs_col;
      std::unique_ptr<column_wrapper<T>> rhs_col;
      lhs_col.reset(new column_wrapper<T>(lhs));
      rhs_col.reset(new column_wrapper<T>(rhs));
      expect_columns_are_equal(*lhs_col, *rhs_col);
      return true;
    }
  };

  void expect_tables_are_equal(cudf::table const &lhs, cudf::table const &rhs) {
    EXPECT_EQ(lhs.num_columns(), rhs.num_columns());
    EXPECT_EQ(lhs.num_rows(), rhs.num_rows());
    EXPECT_TRUE(
        std::equal(lhs.begin(), lhs.end(), rhs.begin(),
                   [](gdf_column const *lhs_col, gdf_column const *rhs_col) {
                     return cudf::type_dispatcher(
                         lhs_col->dtype, column_equality{}, *lhs_col, *rhs_col);
                   }));
  }

  std::pair<cudf::table, gdf_column> gdf_solution(cudf::table const &input_keys,
                                                  bool ignore_null_keys) {
    gdf_context context;
    if (not ignore_null_keys) { // SQL
      context.flag_groupby_include_nulls = true;
      context.flag_null_sort_behavior = GDF_NULL_AS_LARGEST;
    } else { // PANDAS
      context.flag_groupby_include_nulls = false;
      context.flag_null_sort_behavior = GDF_NULL_AS_LARGEST;
    }
    std::vector<int> groupby_col_indices;
    for (gdf_size_type i = 0; i < input_keys.num_columns(); i++)
      groupby_col_indices.push_back(i);

    return gdf_group_by_without_aggregations(
        input_keys, groupby_col_indices.size(), groupby_col_indices.data(),
        &context);
  }

  inline void destroy_table(cudf::table *t) {
    std::for_each(t->begin(), t->end(), [](gdf_column *col) {
      gdf_column_free(col);
      delete col;
    });
  }

  void evaluate_test(column_wrapper<KeyType> keys,
                     column_wrapper<KeyType> sorted_keys,
                     column_wrapper<gdf_size_type> column_offsets,
                     bool ignore_null_keys = true) {
    using namespace cudf::test;

    cudf::table input_keys{keys.get()};

    cudf::table actual_keys_table;
    gdf_column column_offsets_output;
    std::tie(actual_keys_table, column_offsets_output) =
        gdf_solution(input_keys, ignore_null_keys);

    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());

    cudf::table sorted_expected_keys{sorted_keys.get()};
    cudf::table expected_column_offsets{column_offsets.get()};
    cudf::table actual_column_offsets({&column_offsets_output}); 
    
    CUDF_EXPECT_NO_THROW(
        expect_tables_are_equal(actual_keys_table, sorted_expected_keys));

    CUDF_EXPECT_NO_THROW(expect_tables_are_equal(actual_column_offsets,
                                                 expected_column_offsets));

    destroy_table(&actual_keys_table);
    gdf_column_free(&column_offsets_output);
  }
};

using TestingTypes =
    ::testing::Types<int8_t, int32_t, int64_t, float, double, cudf::category,
                     cudf::date32, cudf::date64>;

TYPED_TEST_CASE(SingleColumnGroupby, TestingTypes);

TYPED_TEST(SingleColumnGroupby, OneGroupNoNullsPandasStyle) {
  constexpr int size{10};
  using T = typename SingleColumnGroupby<TypeParam>::Key;

  T key{42};
  bool ignore_null_keys = true;
  this->evaluate_test(
      column_wrapper<T>(size, [key](auto index) { return key; }),
      column_wrapper<T>(size, [key](auto index) { return key; }),
      column_wrapper<gdf_size_type>({0}), ignore_null_keys);
}

TYPED_TEST(SingleColumnGroupby, OneGroupNoNullsSqlStyle) {
  constexpr int size{10};
  using T = typename SingleColumnGroupby<TypeParam>::Key;

  T key{42};
  bool ignore_null_keys = false;
  this->evaluate_test(
      column_wrapper<T>(size, [key](auto index) { return key; }),
      column_wrapper<T>(size, [key](auto index) { return key; }),
      column_wrapper<gdf_size_type>({0}), ignore_null_keys);
}

TYPED_TEST(SingleColumnGroupby, OneGroupEvenNullKeysPandasStyle) {
  constexpr int size{10};
  using T = typename SingleColumnGroupby<TypeParam>::Key;

  T key{42};
  bool ignore_null_keys = true;
  this->evaluate_test(
      column_wrapper<T>(size, [key](auto index) { return key; },
                        [](auto index) { return index % 2; }),
      column_wrapper<T>({T(key), T(key), T(key), T(key), T(key)},
                        [&](auto index) { return true; }),
      column_wrapper<gdf_size_type>({0}), ignore_null_keys);
}

TYPED_TEST(SingleColumnGroupby, OneGroupEvenNullKeysSqlStyle) {
  constexpr int size{10};
  using T = typename SingleColumnGroupby<TypeParam>::Key;

  T key{42};
  bool ignore_null_keys = false;
  this->evaluate_test(column_wrapper<T>(size, [key](auto index) { return key; },
                                        [](auto index) { return index % 2; }),
                      column_wrapper<T>({T(key), T(key), T(key), T(key), T(key),
                                         T(0), T(0), T(0), T(0), T(0)},
                                        [&](auto index) { return index < 5; }),
                      column_wrapper<gdf_size_type>({0, 5}), ignore_null_keys);
}

TYPED_TEST(SingleColumnGroupby, EightKeysAllUniquePandasStyle) {
  using T = typename SingleColumnGroupby<TypeParam>::Key;
  bool ignore_null_keys = true;

  this->evaluate_test(
      column_wrapper<T>({T(0), T(1), T(2), T(3), T(4), T(5), T(6), T(7)}),
      column_wrapper<T>({T(0), T(1), T(2), T(3), T(4), T(5), T(6), T(7)}),
      column_wrapper<gdf_size_type>({0, 1, 2, 3, 4, 5, 6, 7}),
      ignore_null_keys);
}

TYPED_TEST(SingleColumnGroupby, EightKeysAllUniqueSqlStyle) {
  using T = typename SingleColumnGroupby<TypeParam>::Key;
  bool ignore_null_keys = false;

  this->evaluate_test(
      column_wrapper<T>({T(0), T(1), T(2), T(3), T(4), T(5), T(6), T(7)}),
      column_wrapper<T>({T(0), T(1), T(2), T(3), T(4), T(5), T(6), T(7)}),
      column_wrapper<gdf_size_type>({0, 1, 2, 3, 4, 5, 6, 7}),
      ignore_null_keys);
}

TYPED_TEST(SingleColumnGroupby, EightKeysAllUniqueEvenKeysNullPandasStyle) {
  using T = typename SingleColumnGroupby<TypeParam>::Key;
  bool ignore_null_keys = true;

  this->evaluate_test(
      column_wrapper<T>({T(0), T(1), T(2), T(3), T(4), T(5), T(6), T(7)},
                        [](auto index) { return index % 2; }),
      column_wrapper<T>({T(1), T(3), T(5), T(7)},
                        [](auto index) { return true; }),
      column_wrapper<gdf_size_type>({0, 1, 2, 3}), ignore_null_keys);
}

TYPED_TEST(SingleColumnGroupby, EightKeysAllUniqueEvenKeysNullSqlStyle) {
  using T = typename SingleColumnGroupby<TypeParam>::Key;
  bool ignore_null_keys = false;

  this->evaluate_test(
      column_wrapper<T>({T(0), T(1), T(2), T(3), T(4), T(5), T(6), T(7)},
                        [](auto index) { return index % 2; }),
      column_wrapper<T>({T(1), T(3), T(5), T(7), T(0), T(0), T(0), T(0)},
                        [](auto index) { return index < 4; }), /*  */
      column_wrapper<gdf_size_type>({0, 1, 2, 3, 4}), ignore_null_keys);
}
