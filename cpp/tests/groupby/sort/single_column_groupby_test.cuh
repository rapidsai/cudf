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

#ifndef _SORT_SINGLE_COLUMN_GROUPBY_TEST_CUH
#define _SORT_SINGLE_COLUMN_GROUPBY_TEST_CUH

#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/legacy/table.hpp>
#include <tests/utilities/column_wrapper.cuh>
#include <tests/utilities/compare_column_wrappers.cuh>
#include <cudf/utilities/legacy/type_dispatcher.hpp>
#include "../common/groupby_test.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <utility>

namespace cudf {
namespace test {

template <cudf::groupby::operators const_expr_op,  typename Key, typename Value,
          typename ResultValue>
void single_column_groupby_test(cudf::groupby::sort::operation && op,
                                column_wrapper<Key> keys,
                                column_wrapper<Value> values,
                                column_wrapper<Key> expected_keys,
                                column_wrapper<ResultValue> expected_values,
                                bool ignore_null_keys = true
                                ) {
  using namespace cudf::test;
  using namespace cudf::groupby::sort;
   
  static_assert(std::is_same<ResultValue, expected_result_t<Value, const_expr_op>>::value,
                "Incorrect type for expected_values.");
  
  ASSERT_EQ(keys.size(), values.size())
      << "Number of keys must be equal to number of values.";
 
  cudf::table input_keys{keys.get()};
  cudf::table input_values{values.get()};

  cudf::table sorted_actual_keys;
  std::vector<gdf_column*> sorted_actual_values;
  cudf::groupby::sort::Options options{ignore_null_keys};

  std::vector<operation> ops;
  ops.emplace_back(std::move(op));
  std::tie(sorted_actual_keys, sorted_actual_values) =
      cudf::groupby::sort::groupby(input_keys, input_values, ops, options); 
  
  EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());

  cudf::table sorted_expected_keys{expected_keys.get()};
  std::vector<gdf_column*> sorted_expected_values{expected_values.get()}; 
  
  CUDF_EXPECT_NO_THROW(detail::expect_tables_are_equal(sorted_actual_keys,
                                                       sorted_expected_keys));

  CUDF_EXPECT_NO_THROW(detail::expect_values_are_equal(sorted_actual_values,
                                                       sorted_expected_values));
  detail::destroy_columns(&sorted_actual_keys);
  detail::destroy_columns(&sorted_actual_values); 
}

inline void multi_column_groupby_test(
    cudf::table const& keys, cudf::table const& values,
    std::vector<cudf::groupby::operators> const& ops,
    cudf::table const& expected_keys, cudf::table const& expected_values,
    bool ignore_null_keys = true) {
  using namespace cudf::test;
  using namespace cudf::groupby::sort;

  cudf::table sorted_actual_keys;
  std::vector<gdf_column*>  sorted_actual_values;

  std::vector<cudf::groupby::sort::operation> ops_with_args(ops.size());
  for (size_t i = 0; i < ops.size(); i++) {
    ops_with_args[i] = cudf::groupby::sort::operation{ops[i], nullptr};
  }
  std::tie(sorted_actual_keys, sorted_actual_values) =
      cudf::groupby::sort::groupby(keys, values, ops_with_args);

  EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());

  cudf::table sorted_expected_keys = expected_keys;
  std::vector<gdf_column*> sorted_expected_values(expected_values.num_columns()); 
  std::transform(expected_values.begin(), expected_values.end(), 
    sorted_expected_values.begin(), [](const gdf_column *col){
      return const_cast<gdf_column*>(col);
    });

  CUDF_EXPECT_NO_THROW(detail::expect_tables_are_equal(sorted_actual_keys,
                                                       sorted_expected_keys));

  CUDF_EXPECT_NO_THROW(detail::expect_values_are_equal(sorted_actual_values,
                                                       sorted_expected_values));

  detail::destroy_columns(&sorted_actual_keys);
  detail::destroy_columns(&sorted_actual_values); 
}

}  // namespace test
}  // namespace cudf
#endif
