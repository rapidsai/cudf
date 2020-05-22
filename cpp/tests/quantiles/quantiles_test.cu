/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/cudf_gtest.hpp>
#include <tests/utilities/table_utilities.hpp>
#include <tests/utilities/type_lists.hpp>

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/quantiles.hpp>
#include <cudf/utilities/error.hpp>

using namespace cudf;
using namespace test;

template <typename T>
struct QuantilesTest : public BaseFixture {
};

using TestTypes = AllTypes;

TYPED_TEST_CASE(QuantilesTest, TestTypes);

TYPED_TEST(QuantilesTest, TestZeroColumns)
{
  auto input = table_view(std::vector<column_view>{});

  EXPECT_THROW(quantiles(input, {0.0f}), logic_error);
}

TYPED_TEST(QuantilesTest, TestMultiColumnZeroRows)
{
  using T = TypeParam;

  auto input_a = fixed_width_column_wrapper<T>({});
  auto input   = table_view({input_a});

  EXPECT_THROW(quantiles(input, {0.0f}), logic_error);
}

TYPED_TEST(QuantilesTest, TestZeroRequestedQuantiles)
{
  using T = TypeParam;

  auto input_a = fixed_width_column_wrapper<T>({1}, {1});
  auto input   = table_view(std::vector<column_view>{input_a});

  auto actual   = quantiles(input, {});
  auto expected = empty_like(input);

  expect_tables_equal(expected->view(), actual->view());
}

TYPED_TEST(QuantilesTest, TestMultiColumnOrderCountMismatch)
{
  using T = TypeParam;

  auto input_a = fixed_width_column_wrapper<T>({});
  auto input_b = fixed_width_column_wrapper<T>({});
  auto input   = table_view({input_a});

  EXPECT_THROW(quantiles(input,
                         {0.0f},
                         interpolation::NEAREST,
                         sorted::NO,
                         {order::ASCENDING},
                         {null_order::AFTER, null_order::AFTER}),
               logic_error);
}

TYPED_TEST(QuantilesTest, TestMultiColumnNullOrderCountMismatch)
{
  using T = TypeParam;

  auto input_a = fixed_width_column_wrapper<T>({});
  auto input_b = fixed_width_column_wrapper<T>({});
  auto input   = table_view({input_a});

  EXPECT_THROW(quantiles(input,
                         {0.0f},
                         interpolation::NEAREST,
                         sorted::NO,
                         {order::ASCENDING, order::ASCENDING},
                         {null_order::AFTER}),
               logic_error);
}

TYPED_TEST(QuantilesTest, TestMultiColumnArithmeticInterpolation)
{
  using T = TypeParam;

  auto input_a = fixed_width_column_wrapper<T>({});
  auto input_b = fixed_width_column_wrapper<T>({});
  auto input   = table_view({input_a});

  EXPECT_THROW(quantiles(input, {0.0f}, interpolation::LINEAR), logic_error);

  EXPECT_THROW(quantiles(input, {0.0f}, interpolation::MIDPOINT), logic_error);
}

TYPED_TEST(QuantilesTest, TestMultiColumnUnsorted)
{
  using T = TypeParam;

  auto input_a = strings_column_wrapper(
    {"C", "B", "A", "A", "D", "B", "D", "B", "D", "C", "C", "C",
     "D", "B", "D", "B", "C", "C", "A", "D", "B", "A", "A", "A"},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});

  auto input_b = fixed_width_column_wrapper<T>(
    {4, 3, 5, 0, 1, 0, 4, 1, 5, 3, 0, 5, 2, 4, 3, 2, 1, 2, 3, 0, 5, 1, 4, 2},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});

  auto input = table_view({input_a, input_b});

  auto actual = quantiles(input,
                          {0.0f, 0.5f, 0.7f, 0.25f, 1.0f},
                          interpolation::NEAREST,
                          sorted::NO,
                          {order::ASCENDING, order::DESCENDING});

  auto expected_a = strings_column_wrapper({"A", "C", "C", "B", "D"}, {1, 1, 1, 1, 1});

  auto expected_b = fixed_width_column_wrapper<T>({5, 5, 1, 5, 0}, {1, 1, 1, 1, 1});

  auto expected = table_view({expected_a, expected_b});

  expect_tables_equal(expected, actual->view());
}

TYPED_TEST(QuantilesTest, TestMultiColumnAssumedSorted)
{
  using T = TypeParam;

  auto input_a = strings_column_wrapper(
    {"C", "B", "A", "A", "D", "B", "D", "B", "D", "C", "C", "C",
     "D", "B", "D", "B", "C", "C", "A", "D", "B", "A", "A", "A"},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});

  auto input_b = fixed_width_column_wrapper<T>(
    {4, 3, 5, 0, 1, 0, 4, 1, 5, 3, 0, 5, 2, 4, 3, 2, 1, 2, 3, 0, 5, 1, 4, 2},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});

  auto input = table_view({input_a, input_b});

  auto actual =
    quantiles(input, {0.0f, 0.5f, 0.7f, 0.25f, 1.0f}, interpolation::NEAREST, sorted::YES);

  auto expected_a = strings_column_wrapper({"C", "D", "C", "D", "A"}, {1, 1, 1, 1, 1});

  auto expected_b = fixed_width_column_wrapper<T>({4, 2, 1, 4, 2}, {1, 1, 1, 1, 1});

  auto expected = table_view({expected_a, expected_b});

  expect_tables_equal(expected, actual->view());
}
