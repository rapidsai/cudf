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

#include <cudf/aggregation.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/rolling.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/bit.hpp>
#include <src/rolling/rolling_detail.hpp>

#include <thrust/iterator/constant_iterator.h>

#include <algorithm>
#include <vector>

struct CollectListTest : public cudf::test::BaseFixture {
};

template <typename T>
struct TypedCollectListTest : public CollectListTest {
};

using TypesForTest = cudf::test::
  Concat<cudf::test::IntegralTypes, cudf::test::FloatingPointTypes, cudf::test::DurationTypes>;

TYPED_TEST_CASE(TypedCollectListTest, TypesForTest);

TYPED_TEST(TypedCollectListTest, BasicRollingWindow)
{
  using namespace cudf;
  using namespace cudf::test;

  using T = TypeParam;

  auto const input_column = fixed_width_column_wrapper<T, int32_t>{10, 11, 12, 13, 14};

  auto const prev_column = fixed_width_column_wrapper<size_type>{1, 2, 2, 2, 2};
  auto const foll_column = fixed_width_column_wrapper<size_type>{1, 1, 1, 1, 0};

  EXPECT_EQ(static_cast<column_view>(prev_column).size(),
            static_cast<column_view>(foll_column).size());

  auto const result_column_based_window =
    rolling_window(input_column, prev_column, foll_column, 1, make_collect_aggregation());

  auto const expected_result =
    lists_column_wrapper<T, int32_t>{
      {10, 11},
      {10, 11, 12},
      {11, 12, 13},
      {12, 13, 14},
      {13, 14},
    }
      .release();

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result->view(), result_column_based_window->view());

  auto const result_fixed_window =
    rolling_window(input_column, 2, 1, 1, make_collect_aggregation());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result->view(), result_fixed_window->view());
}

TYPED_TEST(TypedCollectListTest, RollingWindowWithEmptyOutputLists)
{
  using namespace cudf;
  using namespace cudf::test;

  using T = TypeParam;

  auto const input_column = fixed_width_column_wrapper<T, int32_t>{10, 11, 12, 13, 14, 15};

  auto const prev_column = fixed_width_column_wrapper<size_type>{1, 2, 2, 0, 2, 2};
  auto const foll_column = fixed_width_column_wrapper<size_type>{1, 1, 1, 0, 1, 0};

  EXPECT_EQ(static_cast<column_view>(prev_column).size(),
            static_cast<column_view>(foll_column).size());

  auto const result_column_based_window =
    rolling_window(input_column, prev_column, foll_column, 0, make_collect_aggregation());

  auto const expected_result =
    lists_column_wrapper<T, int32_t>{
      {10, 11},
      {10, 11, 12},
      {11, 12, 13},
      {},
      {13, 14, 15},
      {14, 15},
    }
      .release();

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result->view(), result_column_based_window->view());
}

TYPED_TEST(TypedCollectListTest, RollingWindowWithEmptyOutputListsAtEnds)
{
  using namespace cudf;
  using namespace cudf::test;

  using T = TypeParam;

  auto const input_column = fixed_width_column_wrapper<T, int32_t>{0, 1, 2, 3, 4, 5};

  auto const prev_column = fixed_width_column_wrapper<size_type>{0, 2, 2, 2, 2, 0};
  auto foll_column       = fixed_width_column_wrapper<size_type>{0, 1, 1, 1, 1, 0};

  auto const result =
    rolling_window(input_column, prev_column, foll_column, 0, make_collect_aggregation());

  auto const expected_result =
    lists_column_wrapper<T, int32_t>{{}, {0, 1, 2}, {1, 2, 3}, {2, 3, 4}, {3, 4, 5}, {}}.release();

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result->view(), result->view());
}

TYPED_TEST(TypedCollectListTest, RollingWindowHonoursMinPeriods)
{
  // Test that when the number of observations is fewer than min_periods,
  // the result is null.

  using namespace cudf;
  using namespace cudf::test;

  using T = TypeParam;

  auto const input_column = fixed_width_column_wrapper<T, int32_t>{0, 1, 2, 3, 4, 5};
  auto const num_elements = static_cast<column_view>(input_column).size();

  auto preceding   = 2;
  auto following   = 1;
  auto min_periods = 3;
  auto const result =
    rolling_window(input_column, preceding, following, min_periods, make_collect_aggregation());

  auto const expected_result = lists_column_wrapper<T, int32_t>{
    {{}, {0, 1, 2}, {1, 2, 3}, {2, 3, 4}, {3, 4, 5}, {}},
    make_counting_transform_iterator(0, [num_elements](auto i) {
      return i != 0 && i != (num_elements - 1);
    })}.release();

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result->view(), result->view());

  preceding   = 2;
  following   = 2;
  min_periods = 4;

  auto result_2 =
    rolling_window(input_column, preceding, following, min_periods, make_collect_aggregation());
  auto expected_result_2 = lists_column_wrapper<T, int32_t>{
    {{}, {0, 1, 2, 3}, {1, 2, 3, 4}, {2, 3, 4, 5}, {}, {}},
    make_counting_transform_iterator(0, [num_elements](auto i) {
      return i != 0 && i < 4;
    })}.release();

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result_2->view(), result_2->view());
}

TYPED_TEST(TypedCollectListTest, RollingWindowWithNullInputsHonoursMinPeriods)
{
  // Test that when the number of observations is fewer than min_periods,
  // the result is null.
  // Input column has null inputs.

  using namespace cudf;
  using namespace cudf::test;

  using T = TypeParam;

  auto const input_column =
    fixed_width_column_wrapper<T, int32_t>{{0, 1, 2, 3, 4, 5}, {1, 0, 1, 1, 0, 1}};
  // auto const num_elements = static_cast<column_view>(input_column).size();

  {
    // One result row at each end should be null.
    auto preceding   = 2;
    auto following   = 1;
    auto min_periods = 3;
    auto const result =
      rolling_window(input_column, preceding, following, min_periods, make_collect_aggregation());

    auto expected_result_child_values   = std::vector<int32_t>{0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5};
    auto expected_result_child_validity = std::vector<bool>{1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1};
    auto expected_result_child =
      fixed_width_column_wrapper<T, int32_t>(expected_result_child_values.begin(),
                                             expected_result_child_values.end(),
                                             expected_result_child_validity.begin());
    auto expected_offsets  = fixed_width_column_wrapper<size_type>{0, 0, 3, 6, 9, 12, 12}.release();
    auto expected_num_rows = expected_offsets->size() - 1;
    auto null_mask_iter    = make_counting_transform_iterator(
      size_type{0}, [expected_num_rows](auto i) { return i != 0 && i != (expected_num_rows - 1); });

    auto expected_result = make_lists_column(
      expected_num_rows,
      std::move(expected_offsets),
      expected_result_child.release(),
      2,
      cudf::test::detail::make_null_mask(null_mask_iter, null_mask_iter + expected_num_rows));

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result->view(), result->view());
  }

  {
    // First result row, and the last two result rows should be null.
    auto preceding   = 2;
    auto following   = 2;
    auto min_periods = 4;
    auto const result =
      rolling_window(input_column, preceding, following, min_periods, make_collect_aggregation());

    auto expected_result_child_values   = std::vector<int32_t>{0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5};
    auto expected_result_child_validity = std::vector<bool>{1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1};
    auto expected_result_child =
      fixed_width_column_wrapper<T, int32_t>(expected_result_child_values.begin(),
                                             expected_result_child_values.end(),
                                             expected_result_child_validity.begin());

    auto expected_offsets = fixed_width_column_wrapper<size_type>{0, 0, 4, 8, 12, 12, 12}.release();
    auto expected_num_rows = expected_offsets->size() - 1;
    auto null_mask_iter    = make_counting_transform_iterator(
      size_type{0}, [expected_num_rows](auto i) { return i > 0 && i < 4; });

    auto expected_result = make_lists_column(
      expected_num_rows,
      std::move(expected_offsets),
      expected_result_child.release(),
      3,
      cudf::test::detail::make_null_mask(null_mask_iter, null_mask_iter + expected_num_rows));

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result->view(), result->view());
  }
}

TEST_F(CollectListTest, RollingWindowHonoursMinPeriodsOnStrings)
{
  // Test that when the number of observations is fewer than min_periods,
  // the result is null.

  using namespace cudf;
  using namespace cudf::test;

  auto const input_column = strings_column_wrapper{"0", "1", "2", "3", "4", "5"};
  auto const num_elements = static_cast<column_view>(input_column).size();

  auto preceding   = 2;
  auto following   = 1;
  auto min_periods = 3;
  auto const result =
    rolling_window(input_column, preceding, following, min_periods, make_collect_aggregation());

  auto const expected_result = lists_column_wrapper<string_view>{
    {{}, {"0", "1", "2"}, {"1", "2", "3"}, {"2", "3", "4"}, {"3", "4", "5"}, {}},
    make_counting_transform_iterator(0, [num_elements](auto i) {
      return i != 0 && i != (num_elements - 1);
    })}.release();

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result->view(), result->view());

  preceding   = 2;
  following   = 2;
  min_periods = 4;

  auto result_2 =
    rolling_window(input_column, preceding, following, min_periods, make_collect_aggregation());
  auto expected_result_2 = lists_column_wrapper<string_view>{
    {{}, {"0", "1", "2", "3"}, {"1", "2", "3", "4"}, {"2", "3", "4", "5"}, {}, {}},
    make_counting_transform_iterator(0, [num_elements](auto i) {
      return i != 0 && i < 4;
    })}.release();

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result_2->view(), result_2->view());
}

TEST_F(CollectListTest, RollingWindowHonoursMinPeriodsWithDecimal)
{
  // Test that when the number of observations is fewer than min_periods,
  // the result is null.

  using namespace cudf;
  using namespace cudf::test;

  auto const input_iter = make_counting_transform_iterator(0, thrust::identity<int32_t>{});
  auto const input_column =
    fixed_point_column_wrapper<int32_t>{input_iter, input_iter + 6, numeric::scale_type{0}};

  {
    // One result row at each end should be null.
    auto preceding   = 2;
    auto following   = 1;
    auto min_periods = 3;
    auto const result =
      rolling_window(input_column, preceding, following, min_periods, make_collect_aggregation());

    auto expected_result_child_values = std::vector<int32_t>{0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5};
    auto expected_result_child =
      fixed_point_column_wrapper<int32_t>{expected_result_child_values.begin(),
                                          expected_result_child_values.end(),
                                          numeric::scale_type{0}};
    auto expected_offsets  = fixed_width_column_wrapper<size_type>{0, 0, 3, 6, 9, 12, 12}.release();
    auto expected_num_rows = expected_offsets->size() - 1;
    auto null_mask_iter    = make_counting_transform_iterator(
      size_type{0}, [expected_num_rows](auto i) { return i != 0 && i != (expected_num_rows - 1); });

    auto expected_result = make_lists_column(
      expected_num_rows,
      std::move(expected_offsets),
      expected_result_child.release(),
      2,
      cudf::test::detail::make_null_mask(null_mask_iter, null_mask_iter + expected_num_rows));

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result->view(), result->view());
  }

  {
    // First result row, and the last two result rows should be null.
    auto preceding   = 2;
    auto following   = 2;
    auto min_periods = 4;
    auto const result =
      rolling_window(input_column, preceding, following, min_periods, make_collect_aggregation());

    auto expected_result_child_values = std::vector<int32_t>{0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5};
    auto expected_result_child =
      fixed_point_column_wrapper<int32_t>{expected_result_child_values.begin(),
                                          expected_result_child_values.end(),
                                          numeric::scale_type{0}};
    auto expected_offsets = fixed_width_column_wrapper<size_type>{0, 0, 4, 8, 12, 12, 12}.release();
    auto expected_num_rows = expected_offsets->size() - 1;
    auto null_mask_iter    = make_counting_transform_iterator(
      size_type{0}, [expected_num_rows](auto i) { return i > 0 && i < 4; });

    auto expected_result = make_lists_column(
      expected_num_rows,
      std::move(expected_offsets),
      expected_result_child.release(),
      3,
      cudf::test::detail::make_null_mask(null_mask_iter, null_mask_iter + expected_num_rows));

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result->view(), result->view());
  }
}

TYPED_TEST(TypedCollectListTest, BasicGroupedRollingWindow)
{
  using namespace cudf;
  using namespace cudf::test;

  using T = TypeParam;

  auto const group_column = fixed_width_column_wrapper<int32_t>{1, 1, 1, 1, 1, 2, 2, 2, 2};
  auto const input_column =
    fixed_width_column_wrapper<T, int32_t>{10, 11, 12, 13, 14, 20, 21, 22, 23};

  auto const preceding   = 2;
  auto const following   = 1;
  auto const min_periods = 1;
  auto const result = grouped_rolling_window(table_view{std::vector<column_view>{group_column}},
                                             input_column,
                                             preceding,
                                             following,
                                             min_periods,
                                             make_collect_aggregation());

  auto const expected_result = lists_column_wrapper<T, int32_t>{
    {10, 11},
    {10, 11, 12},
    {11, 12, 13},
    {12, 13, 14},
    {13, 14},
    {20, 21},
    {20, 21, 22},
    {21, 22, 23},
    {22, 23}}.release();

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result->view(), result->view());
}

TYPED_TEST(TypedCollectListTest, BasicGroupedRollingWindowWithNulls)
{
  using namespace cudf;
  using namespace cudf::test;

  using T = TypeParam;

  auto const group_column = fixed_width_column_wrapper<int32_t>{1, 1, 1, 1, 1, 2, 2, 2, 2};
  auto const input_column = fixed_width_column_wrapper<T, int32_t>{
    {10, 11, 12, 13, 14, 20, 21, 22, 23}, {1, 0, 1, 1, 1, 1, 0, 1, 1}};

  auto const preceding   = 2;
  auto const following   = 1;
  auto const min_periods = 1;
  auto const result = grouped_rolling_window(table_view{std::vector<column_view>{group_column}},
                                             input_column,
                                             preceding,
                                             following,
                                             min_periods,
                                             make_collect_aggregation());

  auto expected_child = fixed_width_column_wrapper<T, int32_t>{
    {10, 11, 10, 11, 12, 11, 12, 13, 12, 13, 14, 13, 14, 20, 21, 20, 21, 22, 21, 22, 23, 22, 23},
    {1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1}};

  auto expected_offsets = fixed_width_column_wrapper<int32_t>{0, 2, 5, 8, 11, 13, 15, 18, 21, 23};

  auto expected_result = make_lists_column(static_cast<column_view>(group_column).size(),
                                           expected_offsets.release(),
                                           expected_child.release(),
                                           0,
                                           {});

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result->view(), result->view());
}

TYPED_TEST(TypedCollectListTest, BasicGroupedTimeRangeRollingWindow)
{
  using namespace cudf;
  using namespace cudf::test;

  using T = TypeParam;

  auto const time_column = fixed_width_column_wrapper<cudf::timestamp_D, cudf::timestamp_D::rep>{
    1, 1, 2, 2, 3, 1, 4, 5, 6};
  auto const group_column = fixed_width_column_wrapper<int32_t>{1, 1, 1, 1, 1, 2, 2, 2, 2};
  auto const input_column =
    fixed_width_column_wrapper<T, int32_t>{10, 11, 12, 13, 14, 20, 21, 22, 23};
  auto const preceding   = 2;
  auto const following   = 1;
  auto const min_periods = 1;
  auto const result =
    grouped_time_range_rolling_window(table_view{std::vector<column_view>{group_column}},
                                      time_column,
                                      cudf::order::ASCENDING,
                                      input_column,
                                      preceding,
                                      following,
                                      min_periods,
                                      make_collect_aggregation());

  auto const expected_result = lists_column_wrapper<T, int32_t>{
    {10, 11, 12, 13},
    {10, 11, 12, 13},
    {10, 11, 12, 13, 14},
    {10, 11, 12, 13, 14},
    {10, 11, 12, 13, 14},
    {20},
    {21, 22},
    {21, 22, 23},
    {21, 22, 23}}.release();

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result->view(), result->view());
}

TEST_F(CollectListTest, BasicGroupedTimeRangeRollingWindowOnStrings)
{
  using namespace cudf;
  using namespace cudf::test;

  auto const time_column = fixed_width_column_wrapper<cudf::timestamp_D, cudf::timestamp_D::rep>{
    1, 1, 2, 2, 3, 1, 4, 5, 6};
  auto const group_column = fixed_width_column_wrapper<int32_t>{1, 1, 1, 1, 1, 2, 2, 2, 2};
  auto const input_column =
    strings_column_wrapper{"10", "11", "12", "13", "14", "20", "21", "22", "23"};
  auto const preceding   = 2;
  auto const following   = 1;
  auto const min_periods = 1;
  auto const result =
    grouped_time_range_rolling_window(table_view{std::vector<column_view>{group_column}},
                                      time_column,
                                      cudf::order::ASCENDING,
                                      input_column,
                                      preceding,
                                      following,
                                      min_periods,
                                      make_collect_aggregation());

  auto const expected_result = lists_column_wrapper<cudf::string_view>{
    {"10", "11", "12", "13"},
    {"10", "11", "12", "13"},
    {"10", "11", "12", "13", "14"},
    {"10", "11", "12", "13", "14"},
    {"10", "11", "12", "13", "14"},
    {"20"},
    {"21", "22"},
    {"21", "22", "23"},
    {"21", "22", "23"}}.release();

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result->view(), result->view());
}

TYPED_TEST(TypedCollectListTest, BasicGroupedTimeRangeRollingWindowOnStructs)
{
  using namespace cudf;
  using namespace cudf::test;

  using T = TypeParam;

  auto const time_column = fixed_width_column_wrapper<cudf::timestamp_D, cudf::timestamp_D::rep>{
    1, 1, 2, 2, 3, 1, 4, 5, 6};
  auto const group_column = fixed_width_column_wrapper<int32_t>{1, 1, 1, 1, 1, 2, 2, 2, 2};
  auto numeric_member_column =
    fixed_width_column_wrapper<T, int32_t>{10, 11, 12, 13, 14, 20, 21, 22, 23};
  auto string_member_column =
    strings_column_wrapper{"10", "11", "12", "13", "14", "20", "21", "22", "23"};
  auto struct_members = std::vector<std::unique_ptr<cudf::column>>{};
  struct_members.emplace_back(numeric_member_column.release());
  struct_members.emplace_back(string_member_column.release());
  auto const struct_column = make_structs_column(9, std::move(struct_members), 0, {});
  auto const preceding     = 2;
  auto const following     = 1;
  auto const min_periods   = 1;
  auto const result =
    grouped_time_range_rolling_window(table_view{std::vector<column_view>{group_column}},
                                      time_column,
                                      cudf::order::ASCENDING,
                                      struct_column->view(),
                                      preceding,
                                      following,
                                      min_periods,
                                      make_collect_aggregation());

  auto expected_numeric_column = fixed_width_column_wrapper<T, int32_t>{
    10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 14, 10, 11, 12,
    13, 14, 10, 11, 12, 13, 14, 20, 21, 22, 21, 22, 23, 21, 22, 23};

  auto expected_string_column = strings_column_wrapper{
    "10", "11", "12", "13", "10", "11", "12", "13", "10", "11", "12", "13", "14", "10", "11", "12",
    "13", "14", "10", "11", "12", "13", "14", "20", "21", "22", "21", "22", "23", "21", "22", "23"};

  auto expected_struct_members = std::vector<std::unique_ptr<cudf::column>>{};
  expected_struct_members.emplace_back(expected_numeric_column.release());
  expected_struct_members.emplace_back(expected_string_column.release());

  auto expected_structs_column = make_structs_column(32, std::move(expected_struct_members), 0, {});
  auto expected_offsets_column =
    fixed_width_column_wrapper<size_type>{0, 4, 8, 13, 18, 23, 24, 26, 29, 32}.release();
  auto expected_result = make_lists_column(
    9, std::move(expected_offsets_column), std::move(expected_structs_column), 0, {});

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result->view(), result->view());
}

TYPED_TEST(TypedCollectListTest, GroupedTimeRangeRollingWindowWithMinPeriods)
{
  // Test that min_periods is honoured.
  // i.e. output row is null when min_periods exceeds number of observations.
  using namespace cudf;
  using namespace cudf::test;

  using T = TypeParam;

  auto const time_column = fixed_width_column_wrapper<cudf::timestamp_D, cudf::timestamp_D::rep>{
    1, 1, 2, 2, 3, 1, 4, 5, 6};
  auto const group_column = fixed_width_column_wrapper<int32_t>{1, 1, 1, 1, 1, 2, 2, 2, 2};
  auto const input_column =
    fixed_width_column_wrapper<T, int32_t>{10, 11, 12, 13, 14, 20, 21, 22, 23};
  auto const preceding   = 2;
  auto const following   = 1;
  auto const min_periods = 4;
  auto const result =
    grouped_time_range_rolling_window(table_view{std::vector<column_view>{group_column}},
                                      time_column,
                                      cudf::order::ASCENDING,
                                      input_column,
                                      preceding,
                                      following,
                                      min_periods,
                                      make_collect_aggregation());

  auto const expected_result = lists_column_wrapper<T, int32_t>{
    {{10, 11, 12, 13},
     {10, 11, 12, 13},
     {10, 11, 12, 13, 14},
     {10, 11, 12, 13, 14},
     {10, 11, 12, 13, 14},
     {},
     {},
     {},
     {}},
    make_counting_transform_iterator(0, [](auto i) {
      return i < 5;
    })}.release();

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result->view(), result->view());
}

TEST_F(CollectListTest, GroupedTimeRangeRollingWindowOnStringsWithMinPeriods)
{
  // Test that min_periods is honoured.
  // i.e. output row is null when min_periods exceeds number of observations.
  using namespace cudf;
  using namespace cudf::test;

  auto const time_column = fixed_width_column_wrapper<cudf::timestamp_D, cudf::timestamp_D::rep>{
    1, 1, 2, 2, 3, 1, 4, 5, 6};
  auto const group_column = fixed_width_column_wrapper<int32_t>{1, 1, 1, 1, 1, 2, 2, 2, 2};
  auto const input_column =
    strings_column_wrapper{"10", "11", "12", "13", "14", "20", "21", "22", "23"};
  auto const preceding   = 2;
  auto const following   = 1;
  auto const min_periods = 4;
  auto const result =
    grouped_time_range_rolling_window(table_view{std::vector<column_view>{group_column}},
                                      time_column,
                                      cudf::order::ASCENDING,
                                      input_column,
                                      preceding,
                                      following,
                                      min_periods,
                                      make_collect_aggregation());

  auto const expected_result = lists_column_wrapper<cudf::string_view>{
    {{"10", "11", "12", "13"},
     {"10", "11", "12", "13"},
     {"10", "11", "12", "13", "14"},
     {"10", "11", "12", "13", "14"},
     {"10", "11", "12", "13", "14"},
     {},
     {},
     {},
     {}},
    make_counting_transform_iterator(0, [](auto i) {
      return i < 5;
    })}.release();

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result->view(), result->view());
}

TYPED_TEST(TypedCollectListTest, GroupedTimeRangeRollingWindowOnStructsWithMinPeriods)
{
  // Test that min_periods is honoured.
  // i.e. output row is null when min_periods exceeds number of observations.
  using namespace cudf;
  using namespace cudf::test;

  using T = TypeParam;

  auto const time_column = fixed_width_column_wrapper<cudf::timestamp_D, cudf::timestamp_D::rep>{
    1, 1, 2, 2, 3, 1, 4, 5, 6};
  auto const group_column = fixed_width_column_wrapper<int32_t>{1, 1, 1, 1, 1, 2, 2, 2, 2};
  auto numeric_member_column =
    fixed_width_column_wrapper<T, int32_t>{10, 11, 12, 13, 14, 20, 21, 22, 23};
  auto string_member_column =
    strings_column_wrapper{"10", "11", "12", "13", "14", "20", "21", "22", "23"};
  auto struct_members = std::vector<std::unique_ptr<cudf::column>>{};
  struct_members.emplace_back(numeric_member_column.release());
  struct_members.emplace_back(string_member_column.release());
  auto const struct_column = make_structs_column(9, std::move(struct_members), 0, {});
  auto const preceding     = 2;
  auto const following     = 1;
  auto const min_periods   = 4;
  auto const result =
    grouped_time_range_rolling_window(table_view{std::vector<column_view>{group_column}},
                                      time_column,
                                      cudf::order::ASCENDING,
                                      struct_column->view(),
                                      preceding,
                                      following,
                                      min_periods,
                                      make_collect_aggregation());

  auto expected_numeric_column = fixed_width_column_wrapper<T, int32_t>{
    10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 14, 10, 11, 12, 13, 14, 10, 11, 12, 13, 14};

  auto expected_string_column =
    strings_column_wrapper{"10", "11", "12", "13", "10", "11", "12", "13", "10", "11", "12", "13",
                           "14", "10", "11", "12", "13", "14", "10", "11", "12", "13", "14"};

  auto expected_struct_members = std::vector<std::unique_ptr<cudf::column>>{};
  expected_struct_members.emplace_back(expected_numeric_column.release());
  expected_struct_members.emplace_back(expected_string_column.release());

  auto expected_structs_column = make_structs_column(23, std::move(expected_struct_members), 0, {});
  auto expected_offsets_column =
    fixed_width_column_wrapper<size_type>{0, 4, 8, 13, 18, 23, 23, 23, 23, 23}.release();
  auto expected_validity_iter = make_counting_transform_iterator(0, [](auto i) { return i < 5; });
  auto expected_null_mask =
    cudf::test::detail::make_null_mask(expected_validity_iter, expected_validity_iter + 9);
  auto expected_result = make_lists_column(9,
                                           std::move(expected_offsets_column),
                                           std::move(expected_structs_column),
                                           4,
                                           std::move(expected_null_mask));

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result->view(), result->view());
}

CUDF_TEST_PROGRAM_MAIN()
