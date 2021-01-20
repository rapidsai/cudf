/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <cudf/aggregation.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/rolling.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/bit.hpp>
#include <src/rolling/rolling_detail.hpp>

#include <thrust/iterator/constant_iterator.h>

#include <algorithm>
#include <vector>

struct CollectListTest : public cudf::test::BaseFixture {};

template <typename T>
struct TypedCollectListTest : public CollectListTest {};

using TypesForTest = cudf::test::Concat<cudf::test::IntegralTypes,
                                        cudf::test::FloatingPointTypes,
                                        cudf::test::DurationTypes>;

TYPED_TEST_CASE(TypedCollectListTest, TypesForTest);

TYPED_TEST(TypedCollectListTest, BasicRollingWindowNoNulls)
{
  using namespace cudf;
  using namespace cudf::test;

  using T = TypeParam;

  auto input_column = fixed_width_column_wrapper<T, int32_t>{10,11,12,13,14}; 

  auto prev_column = fixed_width_column_wrapper<size_type>{1,2,2,2,2};
  auto foll_column = fixed_width_column_wrapper<size_type>{1,1,1,1,0};

  EXPECT_EQ(static_cast<column_view>(prev_column).size(), static_cast<column_view>(foll_column).size());

  auto result_column_based_window = rolling_window(input_column, prev_column, foll_column, 1, make_collect_aggregation());

  auto expected_result = lists_column_wrapper<T, int32_t>{
    {10, 11},
    {10, 11, 12},
    {11, 12, 13},
    {12, 13, 14},
    {13, 14},
  }.release();

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result->view(), result_column_based_window->view());

  auto result_fixed_window = rolling_window(input_column, 2, 1, 1, make_collect_aggregation());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result->view(), result_fixed_window->view());
}

TYPED_TEST(TypedCollectListTest, BasicGroupedRollingWindowNoNulls)
{
  using namespace cudf;
  using namespace cudf::test;

  using T = TypeParam;

  auto group_column = fixed_width_column_wrapper<int32_t>{    1, 1, 1, 1, 1,  2, 2, 2, 2};
  auto input_column = fixed_width_column_wrapper<T, int32_t>{10,11,12,13,14, 20,21,22,23}; 

  auto result = grouped_rolling_window(table_view{std::vector<column_view>{group_column}}, input_column, 2, 1, 1, make_collect_aggregation());

  auto expected_result = lists_column_wrapper<T, int32_t>{
    {10, 11},
    {10, 11, 12},
    {11, 12, 13},
    {12, 13, 14},
    {13, 14},
    {20, 21},
    {20, 21, 22},
    {21, 22, 23},
    {22, 23}
   }.release();

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result->view(), result->view());
}

CUDF_TEST_PROGRAM_MAIN()