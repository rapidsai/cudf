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
#include <tests/utilities/compare_column_wrappers.cuh>
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

// TODO: tests for cudf::bool8
using TestingTypes =
    ::testing::Types<int32_t, int8_t, int16_t, int32_t, int64_t, float, double,
                     cudf::date32, cudf::date64, cudf::category>;

TYPED_TEST_CASE(SingleColumnGroupbyTest, TestingTypes);

using namespace cudf::test;
using namespace cudf::groupby::hash;

template <operators op, typename Key, typename Value, typename ResultValue>
void single_column_groupby_test(column_wrapper<Key> keys,
                                column_wrapper<Value> values,
                                column_wrapper<Key> expected_keys,
                                column_wrapper<ResultValue> expected_values) {
  static_assert(std::is_same<ResultValue, expected_result_t<Value, op>>::value,
                "Incorrect type for expected_values.");

  ASSERT_EQ(keys.size(), values.size())
      << "Number of keys must be equal to number of values.";

  cudf::table input_keys{keys.get()};
  cudf::table input_values{values.get()};

  cudf::table output_keys_table;
  cudf::table output_values_table;
  std::tie(output_keys_table, output_values_table) =
      groupby(input_keys, input_values, {op});

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
  thrust::stable_sort_by_key(thrust::device, expected_keys.get_data().begin(),
                             expected_keys.get_data().end(),
                             expected_values.get_data().begin());
  thrust::stable_sort_by_key(thrust::device, output_keys.get_data().begin(),
                             output_keys.get_data().end(),
                             output_values.get_data().begin());

  bool const print_all_unequal_pairs{true};
  CUDF_EXPECT_NO_THROW(expect_columns_are_equal(output_keys, "Actual Keys",
                                                expected_keys, "Expected Keys",
                                                print_all_unequal_pairs));
  CUDF_EXPECT_NO_THROW(
      expect_columns_are_equal(output_values, "Actual Values", expected_values,
                               "Expected Values", print_all_unequal_pairs));
}

TYPED_TEST(SingleColumnGroupbyTest, OneGroupNoNullsCount) {
  constexpr int size{10};
  constexpr operators op{COUNT};
  using ResultValue = expected_result_t<int, op>;
  single_column_groupby_test<op>(
      column_wrapper<TypeParam>(size, [](auto index) { return TypeParam(42); }),
      column_wrapper<int>(size, [](auto index) { return int(index); }),
      column_wrapper<TypeParam>{TypeParam(42)},
      column_wrapper<ResultValue>{size});
}

TYPED_TEST(SingleColumnGroupbyTest, FourGroupsNoNullsCount) {
  constexpr int size{10};
  constexpr operators op{COUNT};
  using ResultValue = expected_result_t<int, op>;
  using T = TypeParam;
  using R = ResultValue;

  single_column_groupby_test<op>(
      column_wrapper<TypeParam>{T(1), T(2), T(2), T(3), T(3), T(3), T(4), T(4),
                                T(4), T(4)},
      column_wrapper<int>(size, [](auto index) { return int(index); }),
      column_wrapper<TypeParam>{T(1), T(2), T(3), T(4)},
      column_wrapper<ResultValue>{R(1), R(2), R(3), R(4)});
}

TYPED_TEST(SingleColumnGroupbyTest, OneGroupAllNullKeysCount) {
  constexpr int size{10};
  constexpr operators op{COUNT};
  using ResultValue = expected_result_t<int, op>;

  // If all keys are null, then there should be no output
  single_column_groupby_test<op>(
      column_wrapper<TypeParam>(size, [](auto index) { return TypeParam(42); },
                                [](auto index) { return false; }),
      column_wrapper<int>(size, [](auto index) { return int(index); }),
      column_wrapper<TypeParam>{}, column_wrapper<ResultValue>{});
}
