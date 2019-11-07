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

#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/cudf_gtest.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/type_lists.hpp>

#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <thrust/iterator/counting_iterator.h>

#include <algorithm>
#include <numeric>
#include <random>

template <typename T>
T random_int(T min, T max)
{
  static auto seed = unsigned{13377331};
  static auto engine = std::mt19937{seed};
  static_assert(std::is_integral<T>::value,
                "std::uniform_int_distribution works on integral types.");
  static auto uniform = std::uniform_int_distribution<T>{min, max};

  return uniform(engine);
}

template <typename T>
class RepeatTypedTestFixture : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(RepeatTypedTestFixture, cudf::test::FixedWidthTypes);

TYPED_TEST(RepeatTypedTestFixture, RepeatScalarCount)
{
  using T = TypeParam;
  static_assert(cudf::is_fixed_width<T>() == true,
                "this code assumes fixed-width types.");

  constexpr auto num_values = cudf::size_type{10};
  constexpr auto repeat_count = cudf::size_type{10};

  auto input = cudf::test::fixed_width_column_wrapper<T>(
                 thrust::make_counting_iterator(0),
                 thrust::make_counting_iterator(0) + num_values);

  static_assert(repeat_count > 0, "repeat_count should be larger than 0.");
  auto expected_elements = cudf::test::make_counting_transform_iterator(
                             0,
                             [repeat_count](auto i) {
                               return i / repeat_count;
                             });
  auto expected = cudf::test::fixed_width_column_wrapper<T>(
                    expected_elements,
                    expected_elements + num_values * repeat_count);

  auto input_table = cudf::table_view{{input}};

  auto p_count = cudf::make_numeric_scalar(cudf::data_type(cudf::INT32));
  using T_int = cudf::experimental::id_to_type<cudf::INT32>;
  using ScalarType = cudf::experimental::scalar_type_t<T_int>;
  static_cast<ScalarType*>(p_count.get())->set_value(repeat_count);

  auto ret = cudf::experimental::repeat(input_table, *p_count);

  EXPECT_EQ(ret->num_columns(), 1);
  cudf::test::expect_columns_equal(ret->view().column(0), expected);
}

TYPED_TEST(RepeatTypedTestFixture, RepeatColumnCount)
{
  using T = TypeParam;
  static_assert(cudf::is_fixed_width<T>() == true,
                "this code assumes fixed-width types.");

  constexpr auto num_values = cudf::size_type{10};
  constexpr auto max_repeat_count = cudf::size_type{10};

  auto inputs = std::vector<T>(num_values);
  std::iota(inputs.begin(), inputs.end(), 0);

  auto counts = std::vector<cudf::size_type>(num_values);
  std::transform(counts.begin(), counts.end(), counts.begin(),
    [&](cudf::size_type count) { return random_int(0, max_repeat_count); });

  auto expected_values = std::vector<T>();
  for (auto i = size_t{0}; i < counts.size(); i++) {
    for (auto j = cudf::size_type{0}; j < counts[i]; j++) {
      expected_values.push_back(inputs[i]);
    }
  }

  auto input = cudf::test::fixed_width_column_wrapper<T>(
                  inputs.begin(), inputs.end());

  auto count = cudf::test::fixed_width_column_wrapper<cudf::size_type>(
                 counts.begin(), counts.end());

  auto expected = cudf::test::fixed_width_column_wrapper<T>(
                    expected_values.begin(), expected_values.end());

  auto input_table = cudf::table_view{{input}};
  auto ret = cudf::experimental::repeat(input_table, count);

  EXPECT_EQ(ret->num_columns(), 1);
  cudf::test::expect_columns_equal(ret->view().column(0), expected);
}

TYPED_TEST(RepeatTypedTestFixture, RepeatNullable)
{
  using T = TypeParam;
  static_assert(cudf::is_fixed_width<T>() == true,
                "this code assumes fixed-width types.");

  constexpr auto num_values = cudf::size_type{10};
  constexpr auto max_repeat_count = cudf::size_type{10};

  auto input_values = std::vector<T>(num_values);
  std::iota(input_values.begin(), input_values.end(), 0);
  auto input_valids = std::vector<bool>(num_values);
  for (auto i = size_t{0}; i < input_valids.size(); i++) {
    input_valids[i] = (i % 2) == 0 ? true : false;
  }

  auto counts = std::vector<cudf::size_type>(num_values);
  std::transform(counts.begin(), counts.end(), counts.begin(),
    [&](cudf::size_type count) { return random_int(0, max_repeat_count); });

  auto expected_values = std::vector<T>();
  auto expected_valids = std::vector<bool>();
  for (auto i = size_t{0}; i < counts.size(); i++) {
    for (auto j = cudf::size_type{0}; j < counts[i]; j++) {
      expected_values.push_back(input_values[i]);
      expected_valids.push_back(input_valids[i]);
    }
  }

  auto input = cudf::test::fixed_width_column_wrapper<T>(
                  input_values.begin(), input_values.end(),
                  input_valids.begin());

  auto count = cudf::test::fixed_width_column_wrapper<cudf::size_type>(
                 counts.begin(), counts.end());

  auto expected = cudf::test::fixed_width_column_wrapper<T>(
                    expected_values.begin(), expected_values.end(),
                    expected_valids.begin());

  auto input_table = cudf::table_view{{input}};
  auto ret = cudf::experimental::repeat(input_table, count);

  EXPECT_EQ(ret->num_columns(), 1);
  cudf::test::expect_columns_equal(ret->view().column(0), expected);
}

TYPED_TEST(RepeatTypedTestFixture, ZeroSizeInput)
{
  using T = TypeParam;
  static_assert(cudf::is_fixed_width<T>() == true,
                "this code assumes fixed-width types.");

  auto input = cudf::test::fixed_width_column_wrapper<T>(
                  thrust::make_counting_iterator(0),
                  thrust::make_counting_iterator(0));

  auto count = cudf::test::fixed_width_column_wrapper<cudf::size_type>(
                  thrust::make_counting_iterator(0),
                  thrust::make_counting_iterator(0));

  auto expected = cudf::test::fixed_width_column_wrapper<T>(
                    thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(0));

  auto input_table = cudf::table_view{{input}};
  auto ret = cudf::experimental::repeat(input_table, count);

  EXPECT_EQ(ret->num_columns(), 1);
  cudf::test::expect_columns_equal(ret->view().column(0), expected);
}

class RepeatErrorTestFixture : public cudf::test::BaseFixture {};

TEST_F(RepeatErrorTestFixture, LengthMismatch)
{
  auto input = cudf::test::fixed_width_column_wrapper<int32_t>(
                  thrust::make_counting_iterator(0),
                  thrust::make_counting_iterator(0) + 100);

  auto count = cudf::test::fixed_width_column_wrapper<cudf::size_type>(
                  thrust::make_counting_iterator(0),
                  thrust::make_counting_iterator(0) + 200);

  auto input_table = cudf::table_view{{input}};

  // input_table.num_rows() != count.size()
  EXPECT_THROW(auto ret = cudf::experimental::repeat(input_table, count),
               cudf::logic_error);
}

TEST_F(RepeatErrorTestFixture, CountHasNulls)
{
  auto input = cudf::test::fixed_width_column_wrapper<int32_t>(
                  thrust::make_counting_iterator(0),
                  thrust::make_counting_iterator(0) + 100);

  auto count = cudf::test::fixed_width_column_wrapper<cudf::size_type>(
                  thrust::make_counting_iterator(0),
                  thrust::make_counting_iterator(0) + 100,
                  thrust::make_constant_iterator(false));

  auto input_table = cudf::table_view{{input}};

  // input_table.has_nulls() == true
  EXPECT_THROW(auto ret = cudf::experimental::repeat(input_table, count),
               cudf::logic_error);
}

TEST_F(RepeatErrorTestFixture, NegativeCountOrOverflow)
{
  auto input = cudf::test::fixed_width_column_wrapper<int32_t>(
                  thrust::make_counting_iterator(0),
                  thrust::make_counting_iterator(0) + 100);

  auto count_neg = cudf::test::fixed_width_column_wrapper<cudf::size_type>(
                     thrust::make_constant_iterator(-1, 0),
                     thrust::make_constant_iterator(-1, 100));

  auto value = std::numeric_limits<cudf::size_type>::max() / 10;
  auto count_overflow = cudf::test::fixed_width_column_wrapper<cudf::size_type>(
                          thrust::make_constant_iterator(value, 0),
                          thrust::make_constant_iterator(value, 100));

  auto input_table = cudf::table_view{{input}};

  // negative
  EXPECT_THROW(auto ret =
                 cudf::experimental::repeat(input_table, count_neg, true),
               cudf::logic_error);

  // overflow
  EXPECT_THROW(auto ret =
                 cudf::experimental::repeat(input_table, count_overflow, true),
               cudf::logic_error);
}
