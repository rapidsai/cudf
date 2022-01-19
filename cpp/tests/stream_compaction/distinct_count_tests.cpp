/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <cudf/copying.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <algorithm>
#include <cmath>

using cudf::nan_policy;
using cudf::null_equality;
using cudf::null_policy;

constexpr int32_t XXX{70};  // Mark for null elements
constexpr int32_t YYY{3};   // Mark for null elements

template <typename T>
struct DistinctCountCommon : public cudf::test::BaseFixture {
};

TYPED_TEST_SUITE(DistinctCountCommon, cudf::test::NumericTypes);

TYPED_TEST(DistinctCountCommon, NoNull)
{
  using T = TypeParam;

  auto const input = cudf::test::make_type_param_vector<T>(
    {1, 3, 3, 4, 31, 1, 8, 2, 0, 4, 1, 4, 10, 40, 31, 42, 0, 42, 8, 5, 4});

  cudf::test::fixed_width_column_wrapper<T> input_col(input.begin(), input.end());

  // explicit instantiation to one particular type (`double`) to reduce build time
  cudf::size_type const expected = std::set<double>(input.begin(), input.end()).size();
  EXPECT_EQ(
    expected,
    cudf::unordered_distinct_count(input_col, null_policy::INCLUDE, nan_policy::NAN_IS_VALID));

  // explicit instantiation to one particular type (`double`) to reduce build time
  std::vector<double> input_data(input.begin(), input.end());
  auto const new_end      = std::unique(input_data.begin(), input_data.end());
  auto const gold_ordered = new_end - input_data.begin();
  EXPECT_EQ(gold_ordered,
            cudf::distinct_count(input_col, null_policy::INCLUDE, nan_policy::NAN_IS_VALID));
}

TYPED_TEST(DistinctCountCommon, TableNoNull)
{
  using T = TypeParam;

  auto const input1 = cudf::test::make_type_param_vector<T>(
    {1, 3, 3, 3, 4, 31, 1, 8, 2, 0, 4, 1, 4, 10, 40, 31, 42, 0, 42, 8, 5, 4});
  auto const input2 = cudf::test::make_type_param_vector<T>(
    {3, 3, 3, 4, 31, 1, 8, 5, 0, 4, 1, 4, 10, 40, 31, 42, 0, 42, 8, 5, 4, 1});

  std::vector<std::pair<T, T>> pair_input;
  std::transform(
    input1.begin(), input1.end(), input2.begin(), std::back_inserter(pair_input), [](T a, T b) {
      return std::make_pair(a, b);
    });

  cudf::test::fixed_width_column_wrapper<T> input_col1(input1.begin(), input1.end());
  cudf::test::fixed_width_column_wrapper<T> input_col2(input2.begin(), input2.end());

  std::vector<cudf::column_view> cols{input_col1, input_col2};
  cudf::table_view input_table(cols);

  cudf::size_type const expected =
    std::set<std::pair<T, T>>(pair_input.begin(), pair_input.end()).size();
  EXPECT_EQ(expected, cudf::unordered_distinct_count(input_table, null_equality::EQUAL));

  auto const new_end      = std::unique(pair_input.begin(), pair_input.end());
  auto const gold_ordered = new_end - pair_input.begin();
  EXPECT_EQ(gold_ordered, cudf::distinct_count(input_table, null_equality::EQUAL));
}

struct DistinctCount : public cudf::test::BaseFixture {
};

TEST_F(DistinctCount, WithNull)
{
  using T = int32_t;

  std::vector<T> input               = {1,   3,  3,  XXX, 31, 1, 8,  2, 0, XXX, XXX,
                          XXX, 10, 40, 31,  42, 0, 42, 8, 5, XXX};
  std::vector<cudf::size_type> valid = {1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0,
                                        0, 1, 1, 1, 1, 1, 1, 1, 1, 0};

  cudf::test::fixed_width_column_wrapper<T> input_col(input.begin(), input.end(), valid.begin());

  // explicit instantiation to one particular type (`double`) to reduce build time
  cudf::size_type const expected = std::set<double>(input.begin(), input.end()).size();
  EXPECT_EQ(
    expected,
    cudf::unordered_distinct_count(input_col, null_policy::INCLUDE, nan_policy::NAN_IS_VALID));

  auto const new_end      = std::unique(input.begin(), input.end());
  auto const gold_ordered = new_end - input.begin() - 3;
  EXPECT_EQ(gold_ordered,
            cudf::distinct_count(input_col, null_policy::EXCLUDE, nan_policy::NAN_IS_VALID));
}

TEST_F(DistinctCount, IgnoringNull)
{
  using T = int32_t;

  std::vector<T> input               = {1,   YYY, YYY, XXX, 31, 1, 8,  2, 0, XXX, 1,
                          XXX, 10,  40,  31,  42, 0, 42, 8, 5, XXX};
  std::vector<cudf::size_type> valid = {1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1,
                                        0, 1, 1, 1, 1, 1, 1, 1, 1, 0};

  cudf::test::fixed_width_column_wrapper<T> input_col(input.begin(), input.end(), valid.begin());

  cudf::size_type const expected = std::set<T>(input.begin(), input.end()).size();
  // Removing 2 from expected to remove count for `XXX` and `YYY`
  EXPECT_EQ(
    expected - 2,
    cudf::unordered_distinct_count(input_col, null_policy::EXCLUDE, nan_policy::NAN_IS_VALID));

  auto const new_end      = std::unique(input.begin(), input.end());
  auto const gold_ordered = new_end - input.begin() - 5;
  EXPECT_EQ(gold_ordered,
            cudf::distinct_count(input_col, null_policy::EXCLUDE, nan_policy::NAN_IS_VALID));
}

TEST_F(DistinctCount, WithNansAndNull)
{
  using T = float;

  std::vector<T> input               = {1,   3,  NAN, XXX, 31,  1, 8,   2, 0, XXX, 1,
                          XXX, 10, 40,  31,  NAN, 0, NAN, 8, 5, XXX};
  std::vector<cudf::size_type> valid = {1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1,
                                        0, 1, 1, 1, 1, 1, 1, 1, 1, 0};

  cudf::test::fixed_width_column_wrapper<T> input_col{input.begin(), input.end(), valid.begin()};

  cudf::size_type const expected = std::set<T>(input.begin(), input.end()).size();
  EXPECT_EQ(
    expected + 1,  // +1 since `NAN` is not in std::set
    cudf::unordered_distinct_count(input_col, null_policy::INCLUDE, nan_policy::NAN_IS_VALID));

  auto const new_end      = std::unique(input.begin(), input.end());
  auto const gold_ordered = new_end - input.begin();
  EXPECT_EQ(gold_ordered,
            cudf::distinct_count(input_col, null_policy::INCLUDE, nan_policy::NAN_IS_VALID));
}

TEST_F(DistinctCount, WithNansOnly)
{
  using T = float;

  std::vector<T> input               = {1, 3, NAN, 70, 31};
  std::vector<cudf::size_type> valid = {1, 1, 1, 1, 1};

  cudf::test::fixed_width_column_wrapper<T> input_col{input.begin(), input.end(), valid.begin()};

  constexpr auto expected = 5;
  EXPECT_EQ(
    expected,
    cudf::unordered_distinct_count(input_col, null_policy::INCLUDE, nan_policy::NAN_IS_VALID));
  EXPECT_EQ(expected,
            cudf::distinct_count(input_col, null_policy::INCLUDE, nan_policy::NAN_IS_VALID));
}

TEST_F(DistinctCount, NansAsNullWithNoNull)
{
  using T = float;

  std::vector<T> input               = {1, 3, NAN, 70, 31};
  std::vector<cudf::size_type> valid = {1, 1, 1, 1, 1};

  cudf::test::fixed_width_column_wrapper<T> input_col{input.begin(), input.end(), valid.begin()};

  constexpr auto expected = 5;
  EXPECT_EQ(
    expected,
    cudf::unordered_distinct_count(input_col, null_policy::INCLUDE, nan_policy::NAN_IS_NULL));
  EXPECT_EQ(expected,
            cudf::distinct_count(input_col, null_policy::INCLUDE, nan_policy::NAN_IS_NULL));
}

TEST_F(DistinctCount, NansAsNullWithNull)
{
  using T = float;

  std::vector<T> input               = {1, 3, NAN, XXX, 31};
  std::vector<cudf::size_type> valid = {1, 1, 1, 0, 1};

  cudf::test::fixed_width_column_wrapper<T> input_col{input.begin(), input.end(), valid.begin()};

  constexpr auto expected = 4;
  EXPECT_EQ(
    expected,
    cudf::unordered_distinct_count(input_col, null_policy::INCLUDE, nan_policy::NAN_IS_NULL));
  EXPECT_EQ(expected,
            cudf::distinct_count(input_col, null_policy::INCLUDE, nan_policy::NAN_IS_NULL));
}

TEST_F(DistinctCount, NansAsNullWithIgnoreNull)
{
  using T = float;

  std::vector<T> input               = {1, 3, NAN, XXX, 31};
  std::vector<cudf::size_type> valid = {1, 1, 1, 0, 1};

  cudf::test::fixed_width_column_wrapper<T> input_col{input.begin(), input.end(), valid.begin()};

  constexpr auto expected = 3;
  EXPECT_EQ(
    expected,
    cudf::unordered_distinct_count(input_col, null_policy::EXCLUDE, nan_policy::NAN_IS_NULL));
  EXPECT_EQ(expected,
            cudf::distinct_count(input_col, null_policy::EXCLUDE, nan_policy::NAN_IS_NULL));
}

TEST_F(DistinctCount, EmptyColumn)
{
  using T = float;

  cudf::test::fixed_width_column_wrapper<T> input_col{};

  constexpr auto expected = 0;
  EXPECT_EQ(
    expected,
    cudf::unordered_distinct_count(input_col, null_policy::EXCLUDE, nan_policy::NAN_IS_NULL));
  EXPECT_EQ(expected,
            cudf::distinct_count(input_col, null_policy::EXCLUDE, nan_policy::NAN_IS_NULL));
}

TEST_F(DistinctCount, StringColumnWithNull)
{
  cudf::test::strings_column_wrapper input_col{
    {"", "this", "is", "this", "This", "a", "column", "of", "the", "strings"},
    {1, 1, 1, 1, 1, 1, 1, 1, 0, 1}};

  cudf::size_type const expected =
    (std::vector<std::string>{"", "this", "is", "This", "a", "column", "of", "strings"}).size();
  EXPECT_EQ(
    expected,
    cudf::unordered_distinct_count(input_col, null_policy::EXCLUDE, nan_policy::NAN_IS_VALID));
}

TEST_F(DistinctCount, TableWithNull)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col1{{5, 4, 3, 5, 8, 1, 4, 5, 0, 9, -1},
                                                       {1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0}};
  cudf::test::fixed_width_column_wrapper<int32_t> col2{{2, 2, 2, -1, 2, 1, 2, 0, 0, 9, -1},
                                                       {1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0}};
  cudf::table_view input{{col1, col2}};

  EXPECT_EQ(8, cudf::unordered_distinct_count(input, null_equality::EQUAL));
  EXPECT_EQ(10, cudf::unordered_distinct_count(input, null_equality::UNEQUAL));
}

TEST_F(DistinctCount, EmptyColumnedTable)
{
  std::vector<cudf::column_view> cols{};

  cudf::table_view input(cols);

  EXPECT_EQ(0, cudf::unordered_distinct_count(input, null_equality::EQUAL));
  EXPECT_EQ(0, cudf::unordered_distinct_count(input, null_equality::UNEQUAL));
}

TEST_F(DistinctCount, TableMixedTypes)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col1{{5, 4, 3, 5, 8, 1, 4, 5, 0, 9, -1},
                                                       {1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0}};
  cudf::test::fixed_width_column_wrapper<double> col2{{2, 2, 2, -1, 2, 1, 2, 0, 0, 9, -1},
                                                      {1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0}};
  cudf::test::fixed_width_column_wrapper<uint32_t> col3{{2, 2, 2, -1, 2, 1, 2, 0, 0, 9, -1},
                                                        {1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0}};
  cudf::table_view input{{col1, col2, col3}};

  EXPECT_EQ(9, cudf::unordered_distinct_count(input, null_equality::EQUAL));
  EXPECT_EQ(10, cudf::unordered_distinct_count(input, null_equality::UNEQUAL));
}

TEST_F(DistinctCount, TableWithStringColumnWithNull)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col1{{0, 9, 8, 9, 6, 5, 4, 3, 2, 1, 0},
                                                       {1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0}};
  cudf::test::strings_column_wrapper col2{
    {"", "this", "is", "this", "this", "a", "column", "of", "the", "strings", ""},
    {1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0}};

  cudf::table_view input{{col1, col2}};
  EXPECT_EQ(9, cudf::unordered_distinct_count(input, null_equality::EQUAL));
  EXPECT_EQ(10, cudf::unordered_distinct_count(input, null_equality::UNEQUAL));
}
