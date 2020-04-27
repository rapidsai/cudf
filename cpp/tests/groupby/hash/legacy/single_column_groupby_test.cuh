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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <cudf/legacy/copying.hpp>
#include <cudf/legacy/groupby.hpp>
#include <cudf/legacy/table.hpp>
#include <cudf/utilities/legacy/type_dispatcher.hpp>
#include <tests/utilities/legacy/column_wrapper.cuh>
#include <tests/utilities/legacy/compare_column_wrappers.cuh>
#include <utility>
#include "../../common/legacy/groupby_test.hpp"

namespace cudf {
namespace test {
template <cudf::groupby::operators op, typename Key, typename Value, typename ResultValue>
void single_column_groupby_test(column_wrapper<Key> keys,
                                column_wrapper<Value> values,
                                column_wrapper<Key> expected_keys,
                                column_wrapper<ResultValue> expected_values)
{
  using namespace cudf::test;
  using namespace cudf::groupby::hash;
  using namespace cudf::groupby;

  static_assert(std::is_same<ResultValue, expected_result_t<Value, op>>::value,
                "Incorrect type for expected_values.");
  ASSERT_EQ(keys.size(), values.size()) << "Number of keys must be equal to number of values.";
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

  CUDF_EXPECT_NO_THROW(detail::expect_tables_are_equal(sorted_actual_keys, sorted_expected_keys));
  CUDF_EXPECT_NO_THROW(
    detail::expect_tables_are_equal(sorted_actual_values, sorted_expected_values));

  detail::destroy_columns(&sorted_actual_keys);
  detail::destroy_columns(&sorted_actual_values);
  detail::destroy_columns(&sorted_expected_keys);
  detail::destroy_columns(&sorted_expected_values);
}

inline void multi_column_groupby_test(cudf::table const& keys,
                                      cudf::table const& values,
                                      std::vector<cudf::groupby::operators> const& ops,
                                      cudf::table const& expected_keys,
                                      cudf::table const& expected_values)
{
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

  CUDF_EXPECT_NO_THROW(detail::expect_tables_are_equal(sorted_actual_keys, sorted_expected_keys));
  CUDF_EXPECT_NO_THROW(
    detail::expect_tables_are_equal(sorted_actual_values, sorted_expected_values));

  detail::destroy_columns(&sorted_actual_keys);
  detail::destroy_columns(&sorted_actual_values);
  detail::destroy_columns(&sorted_expected_keys);
  detail::destroy_columns(&sorted_expected_values);
}

}  // namespace test
}  // namespace cudf
#endif
