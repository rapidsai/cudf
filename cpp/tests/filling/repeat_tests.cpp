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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/filling.hpp>
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
class RepeatTypedTestFixture : public cudf::test::BaseFixture,
                               cudf::test::UniformRandomGenerator<cudf::size_type> {
 public:
  RepeatTypedTestFixture() : cudf::test::UniformRandomGenerator<cudf::size_type>{0, 10} {}

  cudf::size_type repeat_count() { return this->generate(); }
};

TYPED_TEST_CASE(RepeatTypedTestFixture, cudf::test::FixedWidthTypes);

TYPED_TEST(RepeatTypedTestFixture, RepeatScalarCount)
{
  using T = TypeParam;
  static_assert(cudf::is_fixed_width<T>() == true, "this code assumes fixed-width types.");

  constexpr cudf::size_type num_values{10};
  constexpr cudf::size_type repeat_count{10};

  auto input = cudf::test::fixed_width_column_wrapper<T, int32_t>(
    thrust::make_counting_iterator(0), thrust::make_counting_iterator(0) + num_values);

  static_assert(repeat_count > 0, "repeat_count should be larger than 0.");
  auto expected_elements = cudf::test::make_counting_transform_iterator(
    0, [repeat_count](auto i) { return i / repeat_count; });
  auto expected =
    cudf::test::fixed_width_column_wrapper<T, typename decltype(expected_elements)::value_type>(
      expected_elements, expected_elements + num_values * repeat_count);

  auto input_table = cudf::table_view{{input}};
  auto const p_ret = cudf::repeat(input_table, repeat_count);

  EXPECT_EQ(p_ret->num_columns(), 1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(p_ret->view().column(0), expected, true);
}

TYPED_TEST(RepeatTypedTestFixture, RepeatColumnCount)
{
  using T = TypeParam;
  static_assert(cudf::is_fixed_width<T>() == true, "this code assumes fixed-width types.");

  constexpr cudf::size_type num_values{10};

  std::vector<int64_t> inputs(num_values);
  std::iota(inputs.begin(), inputs.end(), 0);

  std::vector<cudf::size_type> counts(num_values);
  std::transform(counts.begin(), counts.end(), counts.begin(), [&](cudf::size_type count) {
    return this->repeat_count();
  });

  std::vector<T> expected_values;
  for (size_t i{0}; i < counts.size(); i++) {
    for (cudf::size_type j{0}; j < counts[i]; j++) {
      expected_values.push_back(cudf::test::make_type_param_scalar<T>(inputs[i]));
    }
  }

  cudf::test::fixed_width_column_wrapper<T, int64_t> input(inputs.begin(), inputs.end());

  auto count =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>(counts.begin(), counts.end());

  cudf::test::fixed_width_column_wrapper<T> expected(expected_values.begin(),
                                                     expected_values.end());

  cudf::table_view input_table{{input}};
  auto p_ret = cudf::repeat(input_table, count);

  EXPECT_EQ(p_ret->num_columns(), 1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(p_ret->view().column(0), expected);
}

TYPED_TEST(RepeatTypedTestFixture, RepeatNullable)
{
  using T = TypeParam;
  static_assert(cudf::is_fixed_width<T>() == true, "this code assumes fixed-width types.");

  constexpr cudf::size_type num_values{10};

  std::vector<int64_t> input_values(num_values);
  std::iota(input_values.begin(), input_values.end(), 0);
  std::vector<bool> input_valids(num_values);
  for (size_t i{0}; i < input_valids.size(); i++) { input_valids[i] = (i % 2) == 0 ? true : false; }

  std::vector<cudf::size_type> counts(num_values);
  std::transform(counts.begin(), counts.end(), counts.begin(), [&](cudf::size_type count) {
    return this->repeat_count();
  });

  std::vector<T> expected_values;
  std::vector<bool> expected_valids;
  for (size_t i{0}; i < counts.size(); i++) {
    for (cudf::size_type j{0}; j < counts[i]; j++) {
      expected_values.push_back(cudf::test::make_type_param_scalar<T>(input_values[i]));
      expected_valids.push_back(input_valids[i]);
    }
  }

  cudf::test::fixed_width_column_wrapper<T, int64_t> input(
    input_values.begin(), input_values.end(), input_valids.begin());

  auto count =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>(counts.begin(), counts.end());

  cudf::test::fixed_width_column_wrapper<T> expected(
    expected_values.begin(), expected_values.end(), expected_valids.begin());

  cudf::table_view input_table{{input}};
  auto p_ret = cudf::repeat(input_table, count);

  EXPECT_EQ(p_ret->num_columns(), 1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(p_ret->view().column(0), expected);
}

TYPED_TEST(RepeatTypedTestFixture, ZeroSizeInput)
{
  using T = TypeParam;
  static_assert(cudf::is_fixed_width<T>() == true, "this code assumes fixed-width types.");

  cudf::test::fixed_width_column_wrapper<T, int32_t> input(thrust::make_counting_iterator(0),
                                                           thrust::make_counting_iterator(0));

  auto count = cudf::test::fixed_width_column_wrapper<cudf::size_type>(
    thrust::make_counting_iterator(0), thrust::make_counting_iterator(0));

  cudf::test::fixed_width_column_wrapper<T, int32_t> expected(thrust::make_counting_iterator(0),
                                                              thrust::make_counting_iterator(0));

  cudf::table_view input_table{{input}};
  auto p_ret = cudf::repeat(input_table, count);

  EXPECT_EQ(p_ret->num_columns(), 1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(p_ret->view().column(0), expected);
}

class RepeatStringTestFixture : public cudf::test::BaseFixture,
                                cudf::test::UniformRandomGenerator<cudf::size_type> {
 public:
  RepeatStringTestFixture() : cudf::test::UniformRandomGenerator<cudf::size_type>{0, 10} {}

  cudf::size_type repeat_count() { return this->generate(); }
};

TEST_F(RepeatStringTestFixture, RepeatNullable)
{
  constexpr cudf::size_type num_values{10};

  std::vector<std::string> input_values(num_values);
  std::vector<bool> input_valids(num_values);
  for (size_t i{0}; i < num_values; i++) {
    input_values[i] = "#" + std::to_string(i);
    input_valids[i] = (i % 2) == 0 ? true : false;
  }

  std::vector<cudf::size_type> counts(num_values);
  std::transform(counts.begin(), counts.end(), counts.begin(), [&](cudf::size_type count) {
    return this->repeat_count();
  });

  std::vector<std::string> expected_values;
  std::vector<bool> expected_valids;
  for (size_t i{0}; i < counts.size(); i++) {
    for (cudf::size_type j{0}; j < counts[i]; j++) {
      expected_values.push_back(input_values[i]);
      expected_valids.push_back(input_valids[i]);
    }
  }

  auto input = cudf::test::strings_column_wrapper(
    input_values.begin(), input_values.end(), input_valids.begin());

  auto count =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>(counts.begin(), counts.end());

  auto expected = cudf::test::strings_column_wrapper(
    expected_values.begin(), expected_values.end(), expected_valids.begin());

  cudf::table_view input_table{{input}};
  auto p_ret = cudf::repeat(input_table, count);

  EXPECT_EQ(p_ret->num_columns(), 1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(p_ret->view().column(0), expected);
}

TEST_F(RepeatStringTestFixture, ZeroSizeInput)
{
  std::vector<std::string> input_values{};
  auto input = cudf::test::strings_column_wrapper(input_values.begin(), input_values.end());

  auto count = cudf::test::fixed_width_column_wrapper<cudf::size_type>(
    thrust::make_counting_iterator(0), thrust::make_counting_iterator(0));

  auto expected = cudf::test::strings_column_wrapper(input_values.begin(), input_values.end());

  cudf::table_view input_table{{input}};
  auto p_ret = cudf::repeat(input_table, count);

  EXPECT_EQ(p_ret->num_columns(), 1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(p_ret->view().column(0), expected);
}

class RepeatErrorTestFixture : public cudf::test::BaseFixture {
};

TEST_F(RepeatErrorTestFixture, LengthMismatch)
{
  auto input = cudf::test::fixed_width_column_wrapper<int32_t>(
    thrust::make_counting_iterator(0), thrust::make_counting_iterator(0) + 100);

  auto count = cudf::test::fixed_width_column_wrapper<cudf::size_type>(
    thrust::make_counting_iterator(0), thrust::make_counting_iterator(0) + 200);

  cudf::table_view input_table{{input}};

  // input_table.num_rows() != count.size()
  EXPECT_THROW(auto p_ret = cudf::repeat(input_table, count), cudf::logic_error);
}

TEST_F(RepeatErrorTestFixture, CountHasNulls)
{
  auto input = cudf::test::fixed_width_column_wrapper<int32_t>(
    thrust::make_counting_iterator(0), thrust::make_counting_iterator(0) + 100);

  auto count =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>(thrust::make_counting_iterator(0),
                                                            thrust::make_counting_iterator(0) + 100,
                                                            thrust::make_constant_iterator(false));

  cudf::table_view input_table{{input}};

  // input_table.has_nulls() == true
  EXPECT_THROW(auto ret = cudf::repeat(input_table, count), cudf::logic_error);
}

TEST_F(RepeatErrorTestFixture, NegativeCountOrOverflow)
{
  auto input = cudf::test::fixed_width_column_wrapper<int32_t>(
    thrust::make_counting_iterator(0), thrust::make_counting_iterator(0) + 100);

  auto count_neg = cudf::test::fixed_width_column_wrapper<cudf::size_type>(
    thrust::make_constant_iterator(-1, 0), thrust::make_constant_iterator(-1, 100));

  auto value          = std::numeric_limits<cudf::size_type>::max() / 10;
  auto count_overflow = cudf::test::fixed_width_column_wrapper<cudf::size_type>(
    thrust::make_constant_iterator(value, 0), thrust::make_constant_iterator(value, 100));

  cudf::table_view input_table{{input}};

  // negative
  EXPECT_THROW(auto p_ret = cudf::repeat(input_table, count_neg, true), cudf::logic_error);

  // overflow
  EXPECT_THROW(auto p_ret = cudf::repeat(input_table, count_overflow, true), cudf::logic_error);
}
