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

using TypesForTest = cudf::test::Concat<cudf::test::IntegralTypesNotBool,
                                        cudf::test::FloatingPointTypes>;

TYPED_TEST_CASE(TypedCollectListTest, TypesForTest);

TYPED_TEST(TypedCollectListTest, NoNulls)
{
  using namespace cudf;
  using namespace cudf::test;

  using T = TypeParam;

  auto ints_column = fixed_width_column_wrapper<T, int32_t>{70,71,72,73,74}; 

  auto prev_column = fixed_width_column_wrapper<size_type>{1,2,2,2,2};
  auto foll_column = fixed_width_column_wrapper<size_type>{1,1,1,1,0};

  EXPECT_EQ(static_cast<column_view>(prev_column).size(), static_cast<column_view>(foll_column).size());

  auto result = cudf::rolling_window(ints_column, prev_column, foll_column, 1, make_collect_aggregation());

  auto expected_result = lists_column_wrapper<T>{
    {70, 71},
    {70, 71, 72},
    {71, 72, 73},
    {72, 73, 74},
    {73, 74},
  }.release();

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result->view(), result->view());
}

CUDF_TEST_PROGRAM_MAIN()