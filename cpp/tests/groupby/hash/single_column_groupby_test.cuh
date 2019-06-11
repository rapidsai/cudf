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
#include <cudf/table.hpp>
#include <tests/utilities/column_wrapper.cuh>
#include <tests/utilities/compare_column_wrappers.cuh>
#include <utilities/type_dispatcher.hpp>
#include "type_info.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>


namespace cudf {
namespace test {

inline auto sort_by_key(cudf::table const& keys, cudf::table const& values) {
  rmm::device_vector<gdf_index_type> sorted_indices(keys.num_rows());
  gdf_column gdf_sorted_indices;
  gdf_column_view(&gdf_sorted_indices, sorted_indices.data().get(), nullptr,
                  sorted_indices.size(), GDF_INT32);
  gdf_context context;
  context.flag_null_sort_behavior = GDF_NULL_AS_LARGEST;
  gdf_order_by(keys.begin(), nullptr, keys.num_columns(), &gdf_sorted_indices,
               &context);

  cudf::table sorted_output_keys = cudf::allocate_like(keys);
  cudf::table sorted_output_values = cudf::allocate_like(values);

  cudf::gather(&keys, sorted_indices.data().get(), &sorted_output_keys);
  cudf::gather(&values, sorted_indices.data().get(), &sorted_output_values);

  EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());

  return std::make_pair(sorted_output_keys, sorted_output_values);
}

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

  ASSERT_EQ(cudf::gdf_dtype_of<Key>(), actual_keys_table.get_column(0)->dtype);
  ASSERT_EQ(cudf::gdf_dtype_of<ResultValue>(),
            actual_values_table.get_column(0)->dtype);

  cudf::table sorted_actual_keys;
  cudf::table sorted_actual_values;
  std::tie(sorted_actual_keys, sorted_actual_values) =
      sort_by_key(actual_keys_table, actual_values_table);

  cudf::table sorted_expected_keys;
  cudf::table sorted_expected_values;
  std::tie(sorted_expected_keys, sorted_expected_values) =
      sort_by_key({expected_keys.get()}, {expected_values.get()});

  // TODO Is there a better way to test that these don't throw other than
  // doing the construction twice?
  // CUDF_EXPECT_NO_THROW(
  //    column_wrapper<Key> output_keys(*output_keys_table.get_column(0)));
  // CUDF_EXPECT_NO_THROW(column_wrapper<ResultValue> output_values(
  //    *output_values_table.get_column(0)));

  // column_wrapper<Key> output_keys(*output_keys_table.get_column(0));
  // column_wrapper<ResultValue>
  // output_values(*output_values_table.get_column(0));

  // Sort-by-key the expected and actual data to make them directly
  // comparable
  // TODO: Need to do a sort by key on the indices to generate a gather map,
  // and then do a gather in order to make sure the output null values are
  // rearranged correctly
  // thrust::device_vector<gdf_index_type> expected_gather_map(
  //    expected_keys.size());
  // thrust::sequence(thrust::device, expected_gather_map.begin(),
  //                 expected_gather_map.end());
  // EXPECT_NO_THROW(thrust::stable_sort_by_key(
  //    rmm::exec_policy()->on(0), expected_keys.get_data().begin(),
  //    expected_keys.get_data().end(), expected_gather_map.begin()));
  // gdf_column gathered_expected_values =
  //    cudf::allocate_like(*expected_values.get());

  /*
    CUDF_EXPECT_NO_THROW(cudf::gather({expected_values.get()},
                                      expected_gather_map.data().get(),
                                      {&gathered_expected_values}));
    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());
    */

  /*
    thrust::device_vector<gdf_index_type>
    output_gather_map(output_keys.size()); thrust::sequence(thrust::device,
    output_gather_map.begin(), output_gather_map.end());
    EXPECT_NO_THROW(thrust::stable_sort_by_key(
        rmm::exec_policy()->on(0), output_keys.get_data().begin(),
        output_keys.get_data().end(), output_gather_map.begin()));
    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());
    */

  // gdf_column auto gathered_output_values =
  // cudf::allocate_like(expected_values.get());

  // bool const print_all_unequal_pairs{true};
  // CUDF_EXPECT_NO_THROW(expect_columns_are_equal(output_keys, "Actual Keys",
  //                                              expected_keys, "Expected
  //                                              Keys",
  //                                              print_all_unequal_pairs));
  // CUDF_EXPECT_NO_THROW(
  //    expect_columns_are_equal(output_values, "Actual Values",
  //    expected_values,
  //                             "Expected Values", print_all_unequal_pairs));
}

}  // namespace test
}  // namespace cudf
#endif
