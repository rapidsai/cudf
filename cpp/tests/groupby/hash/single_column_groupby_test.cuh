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

#include <cudf/groupby.hpp>
#include <cudf/table.hpp>
#include <tests/utilities/column_wrapper.cuh>
#include <tests/utilities/compare_column_wrappers.cuh>
#include <utilities/type_dispatcher.hpp>
#include "type_info.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace cudf {
namespace test {

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

  cudf::table output_keys_table;
  cudf::table output_values_table;
  std::tie(output_keys_table, output_values_table) =
      cudf::groupby::hash::groupby(input_keys, input_values, {op});

  EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());

  ASSERT_EQ(cudf::gdf_dtype_of<Key>(), output_keys_table.get_column(0)->dtype);
  ASSERT_EQ(cudf::gdf_dtype_of<ResultValue>(),
            output_values_table.get_column(0)->dtype);

  // TODO Is there a better way to test that these don't throw other than
  // doing the construction twice?
  CUDF_EXPECT_NO_THROW(
      column_wrapper<Key> output_keys(*output_keys_table.get_column(0)));
  CUDF_EXPECT_NO_THROW(column_wrapper<ResultValue> output_values(
      *output_values_table.get_column(0)));

  column_wrapper<Key> output_keys(*output_keys_table.get_column(0));
  column_wrapper<ResultValue> output_values(*output_values_table.get_column(0));

  // Sort-by-key the expected and actual data to make them directly comparable
  // TODO: Need to do a sort by key on the indices to generate a gather map, and
  // then do a gather in order to make sure the output null values are
  // rearranged correctly
  EXPECT_NO_THROW(thrust::stable_sort_by_key(
      rmm::exec_policy()->on(0), expected_keys.get_data().begin(),
      expected_keys.get_data().end(), expected_values.get_data().end()));

  EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());

  EXPECT_NO_THROW(thrust::stable_sort_by_key(
      rmm::exec_policy()->on(0), output_keys.get_data().begin(),
      output_keys.get_data().end(), output_values.get_data().begin()));
  EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());


  bool const print_all_unequal_pairs{true};
  CUDF_EXPECT_NO_THROW(expect_columns_are_equal(output_keys, "Actual Keys",
                                                expected_keys, "Expected Keys",
                                                print_all_unequal_pairs));
  CUDF_EXPECT_NO_THROW(
      expect_columns_are_equal(output_values, "Actual Values", expected_values,
                               "Expected Values", print_all_unequal_pairs));
}

}  // namespace test
}  // namespace cudf
#endif
