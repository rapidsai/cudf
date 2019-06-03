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
#include <groupby.hpp>
#include <table.hpp>
#include <tests/utilities/column_wrapper.cuh>
#include <utilities/type_dispatcher.hpp>
#include "type_info.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <random>

template <typename T>
struct SingleColumnGroupbyTest : public GdfTest {
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution{1000, 10000};
  int random_size() { return distribution(generator); }
};

using TestingTypes =
    ::testing::Types<int32_t, int8_t, int16_t, int32_t, int64_t, float, double,
                     cudf::date32, cudf::date64, cudf::category, cudf::bool8>;

TYPED_TEST_CASE(SingleColumnGroupbyTest, TestingTypes);

using namespace cudf::test;
using namespace cudf::groupby::hash;

template <typename ColType, operators op,
          typename ResultType = expected_result_t<ColType, op> >
struct compute_reference_solution {
  compute_reference_solution(column_wrapper<ColType>& _keys,
                             column_wrapper<ColType>& _values)
      : keys{_keys}, values{_values} {}

  auto operator()() {
    rmm::device_vector<ColType> output_keys(keys.size());
    rmm::device_vector<ResultType> output_values(values.size());

    // This wont work because corresponding_functor_t needs lhs and rhs types to
    // be identical
    // auto end = thrust::reduce_by_key(
    //    keys.get_data().begin(), keys.get_data().end(),
    //    values.get_data().begin(), output_keys.begin(), output_values.begin(),
    //    thrust::equal_to<ColType>{}, corresponding_functor_t<op>{});

    // output_keys.erase(end.first, output_keys.end());
    // output_values.erase(end.second, output_values.end());

    return std::make_tuple(output_keys, output_values);
  }

  column_wrapper<ColType>& keys;
  column_wrapper<ColType>& values;
};

template <typename ColType, operators op>
void single_column_groupby_test(column_wrapper<ColType> keys,
                                column_wrapper<ColType> values) {
  ASSERT_EQ(keys.size(), values.size())
      << "Number of keys must be equal to number of values for this test";

  cudf::table input_keys{keys.get()};
  cudf::table input_values{values.get()};

  cudf::table output_keys;
  cudf::table output_values;
  std::tie(output_keys, output_values) =
      groupby(input_keys, input_values, {op});

  // Sort by key to prepare for computing reference solution
  thrust::stable_sort_by_key(thrust::device, keys.get_data().begin(),
                             keys.get_data().end(), values.get_data().begin());

  // Get the number of unique keys
  rmm::device_vector<ColType> unique_keys(keys.size());
  auto last = thrust::unique_copy(thrust::device, keys.get_data().begin(),
                                  keys.get_data().end(), unique_keys.begin());
  unique_keys.erase(last, unique_keys.end());
  gdf_size_type const num_groups = unique_keys.size();

  // Verify output size is correct
  EXPECT_EQ(num_groups, output_keys.num_rows());
  EXPECT_EQ(num_groups, output_values.num_rows());

  // Verify output types are correct
  using expected_output_type = expected_result_t<ColType, op>;
  //static_assert(not std::is_same<expected_output_type, void>::value,
  //              "Invalid ColType and Op combination.");
  gdf_dtype expected_dtype = cudf::gdf_dtype_of<expected_output_type>();
  EXPECT_EQ(input_keys.get_column(0)->dtype, output_keys.get_column(0)->dtype);
  EXPECT_EQ(expected_dtype, output_values.get_column(0)->dtype);

  column_wrapper<ColType> output_keys_column(*output_keys.get_column(0));

 //column_wrapper<expected_output_type> output_values_column(
 //    *output_values.get_column(0));

  // Sort by key on output values to make them comparable to reference
  // solution
  /*
  thrust::stable_sort_by_key(thrust::device,
                             output_keys_column.get_data().begin(),
                             output_keys_column.get_data().end(),
                             output_values_column.get_data().begin());

  EXPECT_TRUE(thrust::equal(unique_keys.begin(), last,
                            output_keys_column.get_data().begin()));

  // compute reference solution
  rmm::device_vector<ColType> expected_keys;
  rmm::device_vector<expected_output_type> expected_values;

  std::tie(expected_keys, expected_values) =
      compute_reference_solution<ColType, op>(keys, values)();

  // Sort by key to make them comparable to computed solution
  thrust::stable_sort_by_key(thrust::device, expected_keys.begin(),
                             expected_keys.end(), expected_values.begin());

  EXPECT_TRUE(thrust::equal(thrust::device, expected_keys.begin(),
                            expected_keys.end(),
                            output_keys_column.get_data().begin()));

  EXPECT_TRUE(thrust::equal(thrust::device, expected_values.begin(),
                            expected_values.end(),
                            output_values_column.get_data().begin()));
                            */
}

TYPED_TEST(SingleColumnGroupbyTest, OneGroupNoNullsSum) {
  constexpr int size{10};
  single_column_groupby_test<TypeParam, SUM>(
      column_wrapper<TypeParam>(size, [](auto index) { return TypeParam(42); }),
      column_wrapper<TypeParam>(size,
                                [](auto index) { return TypeParam(index); }));
}
