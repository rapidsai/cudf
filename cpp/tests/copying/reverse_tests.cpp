/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/copying.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include <algorithm>
#include <numeric>
#include <random>

template <typename T>
class ReverseTypedTestFixture : public cudf::test::BaseFixture,
                                cudf::test::UniformRandomGenerator<cudf::size_type> {
 public:
  ReverseTypedTestFixture() : cudf::test::UniformRandomGenerator<cudf::size_type>{0, 10} {}

  cudf::size_type repeat_count() { return this->generate(); }
};

TYPED_TEST_CASE(ReverseTypedTestFixture, cudf::test::FixedWidthTypes);
TYPED_TEST(ReverseTypedTestFixture, ReverseTable)
{
  using T = TypeParam;
  static_assert(cudf::is_fixed_width<T>() == true, "this code assumes fixed-width types.");

  constexpr cudf::size_type num_values{10};

  auto input = cudf::test::fixed_width_column_wrapper<T, int32_t>(
    thrust::make_counting_iterator(0), thrust::make_counting_iterator(0) + num_values);

  auto expected_elements = cudf::detail::make_counting_transform_iterator(
    0, [num_values] __device__(auto i) { return num_values - i - 1; });

  auto expected =
    cudf::test::fixed_width_column_wrapper<T, typename decltype(expected_elements)::value_type>(
      expected_elements, expected_elements + num_values);

  auto input_table = cudf::table_view{{input}};
  auto const p_ret = cudf::reverse(input_table);

  EXPECT_EQ(p_ret->num_columns(), 1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(p_ret->view().column(0), expected, true);
}

TYPED_TEST(ReverseTypedTestFixture, ReverseColumn)
{
  using T = TypeParam;
  static_assert(cudf::is_fixed_width<T>() == true, "this code assumes fixed-width types.");

  constexpr cudf::size_type num_values{10};

  auto input = cudf::test::fixed_width_column_wrapper<T, int32_t>(
    thrust::make_counting_iterator(0), thrust::make_counting_iterator(0) + num_values);

  auto expected_elements = cudf::detail::make_counting_transform_iterator(
    0, [num_values] __device__(auto i) { return num_values - i - 1; });

  auto expected =
    cudf::test::fixed_width_column_wrapper<T, typename decltype(expected_elements)::value_type>(
      expected_elements, expected_elements + num_values);

  auto const column_ret = cudf::reverse(input);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(column_ret->view(), expected, true);
}

TYPED_TEST(ReverseTypedTestFixture, Nullable)
{
  using T = TypeParam;
  static_assert(cudf::is_fixed_width<T>() == true, "this code assumes fixed-width types.");

  constexpr cudf::size_type num_values{20};

  // we first generate the array for column
  std::vector<int64_t> input_values(num_values);
  std::iota(input_values.begin(), input_values.end(), 1);

  // we then generate the null indexes
  std::vector<bool> input_valids(num_values);
  for (size_t i{0}; i < input_valids.size(); i++) { input_valids[i] = (i % 2) == 0 ? true : false; }

  // we then generate the expected values
  std::vector<T> expected_values;
  std::vector<bool> expected_valids;
  for (int i = 19; i > -1; i--) {
    expected_values.push_back(cudf::test::make_type_param_scalar<T>(input_values[i]));
    expected_valids.push_back(input_valids[i]);
  }

  // we then create the necessary tables
  cudf::test::fixed_width_column_wrapper<T, int64_t> input(
    input_values.begin(), input_values.end(), input_valids.begin());

  cudf::test::fixed_width_column_wrapper<T> expected(
    expected_values.begin(), expected_values.end(), expected_valids.begin());

  cudf::table_view input_table{{input}};
  auto p_ret = cudf::reverse(input_table);

  EXPECT_EQ(p_ret->num_columns(), 1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(p_ret->view().column(0), expected);
}
