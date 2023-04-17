/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/quantiles.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

template <typename T>
struct QuantilesTest : public cudf::test::BaseFixture {};

using TestTypes = cudf::test::AllTypes;

TYPED_TEST_SUITE(QuantilesTest, TestTypes);

TYPED_TEST(QuantilesTest, TestZeroColumns)
{
  auto input = cudf::table_view(std::vector<cudf::column_view>{});

  EXPECT_THROW(cudf::quantiles(input, {0.0f}), cudf::logic_error);
}

TYPED_TEST(QuantilesTest, TestMultiColumnZeroRows)
{
  using T = TypeParam;

  cudf::test::fixed_width_column_wrapper<T> input_a({});
  auto input = cudf::table_view({input_a});

  EXPECT_THROW(cudf::quantiles(input, {0.0f}), cudf::logic_error);
}

TYPED_TEST(QuantilesTest, TestZeroRequestedQuantiles)
{
  using T = TypeParam;

  cudf::test::fixed_width_column_wrapper<T, int32_t> input_a({1}, {1});
  auto input = cudf::table_view(std::vector<cudf::column_view>{input_a});

  auto actual   = cudf::quantiles(input, {});
  auto expected = cudf::empty_like(input);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), actual->view());
}

TYPED_TEST(QuantilesTest, TestMultiColumnOrderCountMismatch)
{
  using T = TypeParam;

  cudf::test::fixed_width_column_wrapper<T> input_a({});
  cudf::test::fixed_width_column_wrapper<T> input_b({});
  auto input = cudf::table_view({input_a});

  EXPECT_THROW(cudf::quantiles(input,
                               {0.0f},
                               cudf::interpolation::NEAREST,
                               cudf::sorted::NO,
                               {cudf::order::ASCENDING},
                               {cudf::null_order::AFTER, cudf::null_order::AFTER}),
               cudf::logic_error);
}

TYPED_TEST(QuantilesTest, TestMultiColumnNullOrderCountMismatch)
{
  using T = TypeParam;

  cudf::test::fixed_width_column_wrapper<T> input_a({});
  cudf::test::fixed_width_column_wrapper<T> input_b({});
  auto input = cudf::table_view({input_a});

  EXPECT_THROW(cudf::quantiles(input,
                               {0.0f},
                               cudf::interpolation::NEAREST,
                               cudf::sorted::NO,
                               {cudf::order::ASCENDING, cudf::order::ASCENDING},
                               {cudf::null_order::AFTER}),
               cudf::logic_error);
}

TYPED_TEST(QuantilesTest, TestMultiColumnArithmeticInterpolation)
{
  using T = TypeParam;

  cudf::test::fixed_width_column_wrapper<T> input_a({});
  cudf::test::fixed_width_column_wrapper<T> input_b({});
  auto input = cudf::table_view({input_a});

  EXPECT_THROW(cudf::quantiles(input, {0.0f}, cudf::interpolation::LINEAR), cudf::logic_error);

  EXPECT_THROW(cudf::quantiles(input, {0.0f}, cudf::interpolation::MIDPOINT), cudf::logic_error);
}

TYPED_TEST(QuantilesTest, TestMultiColumnUnsorted)
{
  using T = TypeParam;

  auto input_a = cudf::test::strings_column_wrapper(
    {"C", "B", "A", "A", "D", "B", "D", "B", "D", "C", "C", "C",
     "D", "B", "D", "B", "C", "C", "A", "D", "B", "A", "A", "A"},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});

  cudf::test::fixed_width_column_wrapper<T, int32_t> input_b(
    {4, 3, 5, 0, 1, 0, 4, 1, 5, 3, 0, 5, 2, 4, 3, 2, 1, 2, 3, 0, 5, 1, 4, 2},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});

  auto input = cudf::table_view({input_a, input_b});

  auto actual = cudf::quantiles(input,
                                {0.0f, 0.5f, 0.7f, 0.25f, 1.0f},
                                cudf::interpolation::NEAREST,
                                cudf::sorted::NO,
                                {cudf::order::ASCENDING, cudf::order::DESCENDING});

  auto expected_a = cudf::test::strings_column_wrapper({"A", "C", "C", "B", "D"}, {1, 1, 1, 1, 1});

  cudf::test::fixed_width_column_wrapper<T, int32_t> expected_b({5, 5, 1, 5, 0}, {1, 1, 1, 1, 1});

  auto expected = cudf::table_view({expected_a, expected_b});

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, actual->view());
}

TYPED_TEST(QuantilesTest, TestMultiColumnAssumedSorted)
{
  using T = TypeParam;

  auto input_a = cudf::test::strings_column_wrapper(
    {"C", "B", "A", "A", "D", "B", "D", "B", "D", "C", "C", "C",
     "D", "B", "D", "B", "C", "C", "A", "D", "B", "A", "A", "A"},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});

  cudf::test::fixed_width_column_wrapper<T, int32_t> input_b(
    {4, 3, 5, 0, 1, 0, 4, 1, 5, 3, 0, 5, 2, 4, 3, 2, 1, 2, 3, 0, 5, 1, 4, 2},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});

  auto input = cudf::table_view({input_a, input_b});

  auto actual = cudf::quantiles(
    input, {0.0f, 0.5f, 0.7f, 0.25f, 1.0f}, cudf::interpolation::NEAREST, cudf::sorted::YES);

  auto expected_a = cudf::test::strings_column_wrapper({"C", "D", "C", "D", "A"}, {1, 1, 1, 1, 1});

  cudf::test::fixed_width_column_wrapper<T, int32_t> expected_b({4, 2, 1, 4, 2}, {1, 1, 1, 1, 1});

  auto expected = cudf::table_view({expected_a, expected_b});

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, actual->view());
}
