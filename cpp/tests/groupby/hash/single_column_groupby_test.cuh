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

#ifndef _SINGLE_COLUMN_GROUPBY_TEST_CUH
#define _SINGLE_COLUMN_GROUPBY_TEST_CUH

#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/legacy/table.hpp>
#include <tests/utilities/column_wrapper.cuh>
#include <tests/utilities/compare_column_wrappers.cuh>
#include <cudf/utilities/legacy/type_dispatcher.hpp>
#include "type_info.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <utility>

namespace cudf {
namespace test {
namespace detail {

/**---------------------------------------------------------------------------*
 * @brief Performs a sort-by-key on tables.
 *
 * Sorts the rows of `keys` into ascending order and reorders the corresponding
 * rows of `values` to match the sorted order.
 *
 * @param keys The keys to sort
 * @param values The values to reorder
 * @return std::pair<table, table> A pair whose first element contains the
 * sorted keys and second element contains reordered values.
 *---------------------------------------------------------------------------**/
inline std::pair<table, table> sort_by_key(cudf::table const& keys,
                                           cudf::table const& values) {
  CUDF_EXPECTS(keys.num_rows() == values.num_rows(),
               "Size mismatch between keys and values");
  rmm::device_vector<gdf_index_type> sorted_indices(keys.num_rows());
  gdf_column gdf_sorted_indices;
  gdf_column_view(&gdf_sorted_indices, sorted_indices.data().get(), nullptr,
                  sorted_indices.size(), GDF_INT32);
  gdf_context context;
  context.flag_null_sort_behavior = GDF_NULL_AS_LARGEST;
  gdf_order_by(keys.begin(), nullptr, keys.num_columns(), &gdf_sorted_indices,
               &context);

  cudf::table sorted_output_keys = cudf::allocate_like(keys, true);
  cudf::table sorted_output_values = cudf::allocate_like(values, true);

  cudf::gather(&keys, sorted_indices.data().get(), &sorted_output_keys);
  cudf::gather(&values, sorted_indices.data().get(), &sorted_output_values);

  EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());

  return std::make_pair(sorted_output_keys, sorted_output_values);
}

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

/**---------------------------------------------------------------------------*
 * @brief Verifies the equality of two tables
 *
 * @param lhs The first table
 * @param rhs The second table
 *---------------------------------------------------------------------------**/
inline void expect_tables_are_equal(cudf::table const& lhs,
                                    cudf::table const& rhs) {
  EXPECT_EQ(lhs.num_columns(), rhs.num_columns());
  EXPECT_EQ(lhs.num_rows(), rhs.num_rows());
  EXPECT_TRUE(
      std::equal(lhs.begin(), lhs.end(), rhs.begin(),
                 [](gdf_column const* lhs_col, gdf_column const* rhs_col) {
                   return cudf::type_dispatcher(
                       lhs_col->dtype, column_equality{}, *lhs_col, *rhs_col);
                 }));
}

inline void destroy_table(table* t) {
  std::for_each(t->begin(), t->end(), [](gdf_column* col) {
    gdf_column_free(col);
    delete col;
  });
}
}  // namespace detail

template <cudf::groupby::hash::operators op, typename Key, typename Value,
          typename ResultValue>
void single_column_groupby_test(column_wrapper<Key> keys,
                                column_wrapper<Value> values,
                                column_wrapper<Key> expected_keys,
                                column_wrapper<ResultValue> expected_values) {
  using namespace cudf::test;
  using namespace cudf::groupby::hash;

  static_assert(std::is_same<ResultValue, expected_result_t<Value, op>>::value,
                "Incorrect type for expected_values.");
  ASSERT_EQ(keys.size(), values.size())
      << "Number of keys must be equal to number of values.";
  ASSERT_EQ(expected_keys.size(), expected_values.size())
      << "Number of keys must be equal to number of values.";

  cudf::table input_keys{keys.get()};
  cudf::table input_values{values.get()};

  cudf::table actual_keys_table;
  cudf::table actual_values_table;
  std::tie(actual_keys_table, actual_values_table) =
      cudf::groupby::hash::groupby(input_keys, input_values, {op});

  EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());

  cudf::table sorted_actual_keys;
  cudf::table sorted_actual_values;
  std::tie(sorted_actual_keys, sorted_actual_values) =
      detail::sort_by_key(actual_keys_table, actual_values_table);

  cudf::table sorted_expected_keys;
  cudf::table sorted_expected_values;
  std::tie(sorted_expected_keys, sorted_expected_values) =
      detail::sort_by_key({expected_keys.get()}, {expected_values.get()});

  CUDF_EXPECT_NO_THROW(detail::expect_tables_are_equal(sorted_actual_keys,
                                                       sorted_expected_keys));
  CUDF_EXPECT_NO_THROW(detail::expect_tables_are_equal(sorted_actual_values,
                                                       sorted_expected_values));

  detail::destroy_table(&sorted_actual_keys);
  detail::destroy_table(&sorted_actual_values);
  detail::destroy_table(&sorted_expected_keys);
  detail::destroy_table(&sorted_expected_values);
}

inline void multi_column_groupby_test(
    cudf::table const& keys, cudf::table const& values,
    std::vector<cudf::groupby::hash::operators> const& ops,
    cudf::table const& expected_keys, cudf::table const& expected_values) {
  using namespace cudf::test;
  using namespace cudf::groupby::hash;

  cudf::table actual_keys_table;
  cudf::table actual_values_table;
  std::tie(actual_keys_table, actual_values_table) =
      cudf::groupby::hash::groupby(keys, values, ops);

  EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());

  cudf::table sorted_actual_keys;
  cudf::table sorted_actual_values;
  std::tie(sorted_actual_keys, sorted_actual_values) =
      detail::sort_by_key(actual_keys_table, actual_values_table);

  cudf::table sorted_expected_keys;
  cudf::table sorted_expected_values;
  std::tie(sorted_expected_keys, sorted_expected_values) =
      detail::sort_by_key(expected_keys, expected_values);

  CUDF_EXPECT_NO_THROW(detail::expect_tables_are_equal(sorted_actual_keys,
                                                       sorted_expected_keys));
  CUDF_EXPECT_NO_THROW(detail::expect_tables_are_equal(sorted_actual_values,
                                                       sorted_expected_values));

  detail::destroy_table(&sorted_actual_keys);
  detail::destroy_table(&sorted_actual_values);
  detail::destroy_table(&sorted_expected_keys);
  detail::destroy_table(&sorted_expected_values);
}

}  // namespace test
}  // namespace cudf
#endif
