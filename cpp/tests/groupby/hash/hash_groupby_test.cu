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
          typename ResultType = expected_result_t<ColType, op>>
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

template <typename ColType, operators op, typename ResultType>
void single_column_groupby_test(column_wrapper<ColType> keys,
                                column_wrapper<ColType> values,
                                column_wrapper<ColType> expected_keys,
                                column_wrapper<ResultType> expected_values) {
  static_assert(std::is_same<ResultType, expected_result_t<ColType, op>>::value,
                "Incorrect type for expected_values.");

  ASSERT_EQ(keys.size(), values.size())
      << "Number of keys must be equal to number of values.";

  cudf::table input_keys{keys.get()};
  cudf::table input_values{values.get()};

  cudf::table output_keys_table;
  cudf::table output_values_table;
  std::tie(output_keys_table, output_values_table) =
      groupby(input_keys, input_values, {op});

  ASSERT_EQ(cudf::gdf_dtype_of<ColType>(),
            output_keys_table.get_column(0)->dtype);
  ASSERT_EQ(cudf::gdf_dtype_of<ResultType>(),
            output_values_table.get_column(0)->dtype);

  column_wrapper<ColType> output_keys(*output_keys_table.get_column(0));
  column_wrapper<ResultType> output_values(*output_values_table.get_column(0));

  // Sort-by-key the expected and actual data to make them directly comparable
  thrust::stable_sort_by_key(thrust::device, expected_keys.get_data().begin(),
                             expected_keys.get_data().end(),
                             expected_values.get_data().begin());
  thrust::stable_sort_by_key(thrust::device, output_keys.get_data().begin(),
                             output_keys.get_data().end(),
                             output_values.get_data().begin());

  EXPECT_TRUE(expected_keys == output_keys);
  EXPECT_TRUE(expected_values == output_values);
}

TYPED_TEST(SingleColumnGroupbyTest, OneGroupNoNullsCount) {
  constexpr int size{10};
  constexpr operators op{COUNT};
  using ResultType = expected_result_t<TypeParam, op>;
  single_column_groupby_test<TypeParam, op>(
      column_wrapper<TypeParam>(size, [](auto index) { return TypeParam(42); }),
      column_wrapper<TypeParam>(size,
                                [](auto index) { return TypeParam(index); }),
      column_wrapper<TypeParam>{TypeParam(42)},
      column_wrapper<ResultType>{size});
}
