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
//#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/rolling.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/bit.hpp>
#include <src/rolling/rolling_detail.hpp>

#include <thrust/iterator/constant_iterator.h>

#include <algorithm>
#include <vector>

namespace cudf {
namespace test {

#define COL_V cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>
#define COL_S cudf::test::fixed_width_column_wrapper<cudf::size_type>
#define LCL_V cudf::test::lists_column_wrapper<TypeParam, int32_t>
#define COLLECT_SET cudf::make_collect_list_aggregation()
#define COLLECT_SET_NULLS_EXCLUDED cudf::make_collect_list_aggregation(cudf::null_policy::EXCLUDE)

void test_equivalent(std::unique_ptr<cudf::column> const& lhs,
                     std::unique_ptr<cudf::column> const& rhs)
{
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(lhs->view(), rhs->view());
}

void test_equivalent(cudf::column_view const& lhs, cudf::column_view const& rhs)
{
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(lhs, rhs);
}

struct CollectSetTest : public cudf::test::BaseFixture {
};

template <typename T>
struct TypedCollectListTest : public CollectSetTest {
};

using TypesForTest = cudf::test::Concat<cudf::test::IntegralTypes,
                                        cudf::test::FloatingPointTypes,
                                        cudf::test::DurationTypes,
                                        cudf::test::FixedPointTypes>;

TYPED_TEST_CASE(TypedCollectListTest, TypesForTest);

TYPED_TEST(TypedCollectListTest, BasicRollingWindow)
{
  auto const input = COL_V{10, 11, 12, 13, 14};
  auto const expected_result =
    LCL_V{
      {10, 11},
      {10, 11, 12},
      {11, 12, 13},
      {12, 13, 14},
      {13, 14},
    }
      .release();

  // Rolling window with variable window sizes
  test_equivalent(
    expected_result,
    rolling_window(input, COL_S{1, 2, 2, 2, 2}, COL_S{1, 1, 1, 1, 0}, 1, COLLECT_SET));

  // Rolling window with fixed window size
  test_equivalent(expected_result, rolling_window(input, 2, 1, 1, COLLECT_SET));

  // Rolling window with nulls excluded
  test_equivalent(expected_result, rolling_window(input, 2, 1, 1, COLLECT_SET_NULLS_EXCLUDED));
}

TYPED_TEST(TypedCollectListTest, EmptyOutputLists)
{
  auto const input = COL_V{10, 11, 12, 13, 14, 15};
  auto const expected_result =
    LCL_V{
      {10, 11},
      {10, 11, 12},
      {11, 12, 13},
      {},
      {13, 14, 15},
      {14, 15},
    }
      .release();
  auto const prev = COL_S{1, 2, 2, 0, 2, 2};
  auto const next = COL_S{1, 1, 1, 0, 1, 0};

  // Rolling window with variable window sizes
  test_equivalent(expected_result, rolling_window(input, prev, next, 0, COLLECT_SET));

  // Rolling window with nulls excluded
  test_equivalent(expected_result,
                  rolling_window(input, prev, next, 0, COLLECT_SET_NULLS_EXCLUDED));
}

TYPED_TEST(TypedCollectListTest, EmptyOutputListsAtEnds)
{
  auto const input           = COL_V{0, 1, 2, 3, 4, 5};
  auto const expected_result = LCL_V{{}, {0, 1, 2}, {1, 2, 3}, {2, 3, 4}, {3, 4, 5}, {}}.release();
  auto const prev            = COL_S{0, 2, 2, 2, 2, 0};
  auto const next            = COL_S{0, 1, 1, 1, 1, 0};

  // Rolling window with variable window sizes
  test_equivalent(expected_result, rolling_window(input, prev, next, 0, COLLECT_SET));

  // Rolling window with nulls excluded
  test_equivalent(expected_result,
                  rolling_window(input, prev, next, 0, COLLECT_SET_NULLS_EXCLUDED));
}

#if 0
TEST_F(CollectSetTest, RollingWindowHonoursMinPeriodsOnStrings)
{
  // Test that when the number of observations is fewer than min_periods,
  // the result is null.

  using namespace cudf;
  using namespace cudf::test;

  auto const input        = strings_column_wrapper{"0", "1", "2", "3", "4", "5"};
  auto const num_elements = static_cast<column_view>(input).size();

  auto preceding    = 2;
  auto following    = 1;
  auto min_periods  = 3;
  auto const result = rolling_window(input, preceding, following, min_periods, COLLECT_SET);

  auto const expected_result = lists_column_wrapper<string_view>{
    {{}, {"0", "1", "2"}, {"1", "2", "3"}, {"2", "3", "4"}, {"3", "4", "5"}, {}},
    cudf::detail::make_counting_transform_iterator(0, [num_elements](auto i) {
      return i != 0 && i != (num_elements - 1);
    })}.release();

  test_equivalent(expected_result->view(), result->view());

  auto const result_with_nulls_excluded =
    rolling_window(input, preceding, following, min_periods, COLLECT_SET_NULLS_EXCLUDED);

  test_equivalent(expected_result->view(), result_with_nulls_excluded->view());

  preceding   = 2;
  following   = 2;
  min_periods = 4;

  auto result_2          = rolling_window(input, preceding, following, min_periods, COLLECT_SET);
  auto expected_result_2 = lists_column_wrapper<string_view>{
    {{}, {"0", "1", "2", "3"}, {"1", "2", "3", "4"}, {"2", "3", "4", "5"}, {}, {}},
    cudf::detail::make_counting_transform_iterator(0, [num_elements](auto i) {
      return i != 0 && i < 4;
    })}.release();

  test_equivalent(expected_result_2->view(), result_2->view());

  auto result_2_with_nulls_excluded =
    rolling_window(input, preceding, following, min_periods, COLLECT_SET_NULLS_EXCLUDED);

  test_equivalent(expected_result_2->view(), result_2_with_nulls_excluded->view());
}

TEST_F(CollectSetTest, RollingWindowHonoursMinPeriodsWithDecimal)
{
  // Test that when the number of observations is fewer than min_periods,
  // the result is null.

  using namespace cudf;
  using namespace cudf::test;

  auto const input_iter =
    cudf::detail::make_counting_transform_iterator(0, thrust::identity<int32_t>{});
  auto const input =
    fixed_point_column_wrapper<int32_t>{input_iter, input_iter + 6, numeric::scale_type{0}};

  {
    // One result row at each end should be null.
    auto preceding    = 2;
    auto following    = 1;
    auto min_periods  = 3;
    auto const result = rolling_window(input, preceding, following, min_periods, COLLECT_SET);

    auto expected_result_child_values = std::vector<int32_t>{0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5};
    auto expected_result_child =
      fixed_point_column_wrapper<int32_t>{expected_result_child_values.begin(),
                                          expected_result_child_values.end(),
                                          numeric::scale_type{0}};
    auto expected_offsets  = COL_S{0, 0, 3, 6, 9, 12, 12}.release();
    auto expected_num_rows = expected_offsets->size() - 1;
    auto null_mask_iter    = cudf::detail::make_counting_transform_iterator(
      size_type{0}, [expected_num_rows](auto i) { return i != 0 && i != (expected_num_rows - 1); });

    auto expected_result = make_lists_column(
      expected_num_rows,
      std::move(expected_offsets),
      expected_result_child.release(),
      2,
      cudf::test::detail::make_null_mask(null_mask_iter, null_mask_iter + expected_num_rows));

    test_equivalent(expected_result->view(), result->view());

    auto const result_with_nulls_excluded =
      rolling_window(input,
                     preceding,
                     following,
                     min_periods,
                     COLLECT_SET_NULLS_EXCLUDED);

    test_equivalent(expected_result->view(), result_with_nulls_excluded->view());
  }

  {
    // First result row, and the last two result rows should be null.
    auto preceding    = 2;
    auto following    = 2;
    auto min_periods  = 4;
    auto const result = rolling_window(input, preceding, following, min_periods, COLLECT_SET);

    auto expected_result_child_values = std::vector<int32_t>{0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5};
    auto expected_result_child =
      fixed_point_column_wrapper<int32_t>{expected_result_child_values.begin(),
                                          expected_result_child_values.end(),
                                          numeric::scale_type{0}};
    auto expected_offsets  = COL_S{0, 0, 4, 8, 12, 12, 12}.release();
    auto expected_num_rows = expected_offsets->size() - 1;
    auto null_mask_iter    = cudf::detail::make_counting_transform_iterator(
      size_type{0}, [expected_num_rows](auto i) { return i > 0 && i < 4; });

    auto expected_result = make_lists_column(
      expected_num_rows,
      std::move(expected_offsets),
      expected_result_child.release(),
      3,
      cudf::test::detail::make_null_mask(null_mask_iter, null_mask_iter + expected_num_rows));

    test_equivalent(expected_result->view(), result->view());

    auto const result_with_nulls_excluded =
      rolling_window(input,
                     preceding,
                     following,
                     min_periods,
                     COLLECT_SET_NULLS_EXCLUDED);

    test_equivalent(expected_result->view(), result_with_nulls_excluded->view());
  }
}

TYPED_TEST(TypedCollectListTest, BasicGroupedRollingWindow)
{
  using namespace cudf;
  using namespace cudf::test;

  auto const group_column = fixed_width_column_wrapper<int32_t>{1, 1, 1, 1, 1, 2, 2, 2, 2};
  auto const input        = COL_V{10, 11, 12, 13, 14, 20, 21, 22, 23};

  auto const preceding   = 2;
  auto const following   = 1;
  auto const min_periods = 1;
  auto const result = grouped_rolling_window(table_view{std::vector<column_view>{group_column}},
                                             input,
                                             preceding,
                                             following,
                                             min_periods,
                                             COLLECT_SET);

  auto const expected_result = LCL_V{
    {10, 11},
    {10, 11, 12},
    {11, 12, 13},
    {12, 13, 14},
    {13, 14},
    {20, 21},
    {20, 21, 22},
    {21, 22, 23},
    {22, 23}}.release();

  test_equivalent(expected_result->view(), result->view());

  auto const result_with_nulls_excluded =
    grouped_rolling_window(table_view{std::vector<column_view>{group_column}},
                           input,
                           preceding,
                           following,
                           min_periods,
                           COLLECT_SET_NULLS_EXCLUDED);

  test_equivalent(expected_result->view(), result_with_nulls_excluded->view());
}

TYPED_TEST(TypedCollectListTest, BasicGroupedRollingWindowWithNulls)
{
  using namespace cudf;
  using namespace cudf::test;

  auto const group_column = fixed_width_column_wrapper<int32_t>{1, 1, 1, 1, 1, 2, 2, 2, 2};
  auto const input = COL_V{{10, 11, 12, 13, 14, 20, 21, 22, 23}, {1, 0, 1, 1, 1, 1, 0, 1, 1}};

  auto const preceding   = 2;
  auto const following   = 1;
  auto const min_periods = 1;

  {
    // Nulls included.
    auto const result = grouped_rolling_window(table_view{std::vector<column_view>{group_column}},
                                               input,
                                               preceding,
                                               following,
                                               min_periods,
                                               COLLECT_SET);

    auto expected_child = COL_V{
      {10, 11, 10, 11, 12, 11, 12, 13, 12, 13, 14, 13, 14, 20, 21, 20, 21, 22, 21, 22, 23, 22, 23},
      {1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1}};

    auto expected_offsets = fixed_width_column_wrapper<int32_t>{0, 2, 5, 8, 11, 13, 15, 18, 21, 23};

    auto expected_result = make_lists_column(static_cast<column_view>(group_column).size(),
                                             expected_offsets.release(),
                                             expected_child.release(),
                                             0,
                                             {});

    test_equivalent(expected_result->view(), result->view());
  }

  {
    // Nulls excluded.
    auto const result = grouped_rolling_window(table_view{std::vector<column_view>{group_column}},
                                               input,
                                               preceding,
                                               following,
                                               min_periods,
                                               COLLECT_SET_NULLS_EXCLUDED);

    auto expected_child = COL_V{10, 10, 12, 12, 13, 12, 13, 14, 13, 14, 20, 20, 22, 22, 23, 22, 23};

    auto expected_offsets = fixed_width_column_wrapper<int32_t>{0, 1, 3, 5, 8, 10, 11, 13, 15, 17};

    auto expected_result = make_lists_column(static_cast<column_view>(group_column).size(),
                                             expected_offsets.release(),
                                             expected_child.release(),
                                             0,
                                             {});

    test_equivalent(expected_result->view(), result->view());
  }
}

TYPED_TEST(TypedCollectListTest, BasicGroupedTimeRangeRollingWindow)
{
  using namespace cudf;
  using namespace cudf::test;

  auto const time_column = fixed_width_column_wrapper<cudf::timestamp_D, cudf::timestamp_D::rep>{
    1, 1, 2, 2, 3, 1, 4, 5, 6};
  auto const group_column = fixed_width_column_wrapper<int32_t>{1, 1, 1, 1, 1, 2, 2, 2, 2};
  auto const input        = COL_V{10, 11, 12, 13, 14, 20, 21, 22, 23};
  auto const preceding    = 2;
  auto const following    = 1;
  auto const min_periods  = 1;
  auto const result =
    grouped_time_range_rolling_window(table_view{std::vector<column_view>{group_column}},
                                      time_column,
                                      cudf::order::ASCENDING,
                                      input,
                                      preceding,
                                      following,
                                      min_periods,
                                      COLLECT_SET);

  auto const expected_result = LCL_V{
    {10, 11, 12, 13},
    {10, 11, 12, 13},
    {10, 11, 12, 13, 14},
    {10, 11, 12, 13, 14},
    {10, 11, 12, 13, 14},
    {20},
    {21, 22},
    {21, 22, 23},
    {21, 22, 23}}.release();

  test_equivalent(expected_result->view(), result->view());

  auto const result_with_nulls_excluded =
    grouped_time_range_rolling_window(table_view{std::vector<column_view>{group_column}},
                                      time_column,
                                      cudf::order::ASCENDING,
                                      input,
                                      preceding,
                                      following,
                                      min_periods,
                                      COLLECT_SET_NULLS_EXCLUDED);

  test_equivalent(expected_result->view(), result_with_nulls_excluded->view());
}

TYPED_TEST(TypedCollectListTest, GroupedTimeRangeRollingWindowWithNulls)
{
  using namespace cudf;
  using namespace cudf::test;

  auto const time_column = fixed_width_column_wrapper<cudf::timestamp_D, cudf::timestamp_D::rep>{
    1, 1, 2, 2, 3, 1, 4, 5, 6};
  auto const group_column = fixed_width_column_wrapper<int32_t>{1, 1, 1, 1, 1, 2, 2, 2, 2};
  auto const input       = COL_V{{10, 11, 12, 13, 14, 20, 21, 22, 23}, {1, 0, 1, 1, 1, 1, 0, 1, 1}};
  auto const preceding   = 2;
  auto const following   = 1;
  auto const min_periods = 1;
  auto const result =
    grouped_time_range_rolling_window(table_view{std::vector<column_view>{group_column}},
                                      time_column,
                                      cudf::order::ASCENDING,
                                      input,
                                      preceding,
                                      following,
                                      min_periods,
                                      COLLECT_SET);

  auto null_at_0 = iterator_with_null_at(0);
  auto null_at_1 = iterator_with_null_at(1);

  // In the results, `11` and `21` should be nulls.
  auto const expected_result = LCL_V{
    {{10, 11, 12, 13}, null_at_1},
    {{10, 11, 12, 13}, null_at_1},
    {{10, 11, 12, 13, 14}, null_at_1},
    {{10, 11, 12, 13, 14}, null_at_1},
    {{10, 11, 12, 13, 14}, null_at_1},
    {{20}, null_at_1},
    {{21, 22}, null_at_0},
    {{21, 22, 23}, null_at_0},
    {{21, 22, 23}, null_at_0}}.release();

  test_equivalent(expected_result->view(), result->view());

  auto const result_with_nulls_excluded =
    grouped_time_range_rolling_window(table_view{std::vector<column_view>{group_column}},
                                      time_column,
                                      cudf::order::ASCENDING,
                                      input,
                                      preceding,
                                      following,
                                      min_periods,
                                      COLLECT_SET_NULLS_EXCLUDED);

  // After null exclusion, `11`, `21`, and `null` should not appear.
  auto const expected_result_with_nulls_excluded = LCL_V{
    {10, 12, 13},
    {10, 12, 13},
    {10, 12, 13, 14},
    {10, 12, 13, 14},
    {10, 12, 13, 14},
    {20},
    {22},
    {22, 23},
    {22, 23}}.release();

  test_equivalent(expected_result_with_nulls_excluded->view(), result_with_nulls_excluded->view());
}

TEST_F(CollectSetTest, BasicGroupedTimeRangeRollingWindowOnStrings)
{
  using namespace cudf;
  using namespace cudf::test;

  auto const time_column = fixed_width_column_wrapper<cudf::timestamp_D, cudf::timestamp_D::rep>{
    1, 1, 2, 2, 3, 1, 4, 5, 6};
  auto const group_column = fixed_width_column_wrapper<int32_t>{1, 1, 1, 1, 1, 2, 2, 2, 2};
  auto const input = strings_column_wrapper{"10", "11", "12", "13", "14", "20", "21", "22", "23"};
  auto const preceding   = 2;
  auto const following   = 1;
  auto const min_periods = 1;
  auto const result =
    grouped_time_range_rolling_window(table_view{std::vector<column_view>{group_column}},
                                      time_column,
                                      cudf::order::ASCENDING,
                                      input,
                                      preceding,
                                      following,
                                      min_periods,
                                      COLLECT_SET);

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

  test_equivalent(expected_result->view(), result->view());

  auto const result_with_nulls_excluded =
    grouped_time_range_rolling_window(table_view{std::vector<column_view>{group_column}},
                                      time_column,
                                      cudf::order::ASCENDING,
                                      input,
                                      preceding,
                                      following,
                                      min_periods,
                                      COLLECT_SET_NULLS_EXCLUDED);

  test_equivalent(expected_result->view(), result_with_nulls_excluded->view());
}

TEST_F(CollectSetTest, GroupedTimeRangeRollingWindowOnStringsWithNulls)
{
  using namespace cudf;
  using namespace cudf::test;

  auto const time_column = fixed_width_column_wrapper<cudf::timestamp_D, cudf::timestamp_D::rep>{
    1, 1, 2, 2, 3, 1, 4, 5, 6};
  auto const group_column = fixed_width_column_wrapper<int32_t>{1, 1, 1, 1, 1, 2, 2, 2, 2};
  auto const input = strings_column_wrapper{{"10", "11", "12", "13", "14", "20", "21", "22", "23"},
                                            {1, 0, 1, 1, 1, 1, 0, 1, 1}};
  auto const preceding   = 2;
  auto const following   = 1;
  auto const min_periods = 1;
  auto const result =
    grouped_time_range_rolling_window(table_view{std::vector<column_view>{group_column}},
                                      time_column,
                                      cudf::order::ASCENDING,
                                      input,
                                      preceding,
                                      following,
                                      min_periods,
                                      COLLECT_SET);

  auto null_at_0 = iterator_with_null_at(0);
  auto null_at_1 = iterator_with_null_at(1);

  // In the results, `11` and `21` should be nulls.
  auto const expected_result = lists_column_wrapper<cudf::string_view>{
    {{"10", "11", "12", "13"}, null_at_1},
    {{"10", "11", "12", "13"}, null_at_1},
    {{"10", "11", "12", "13", "14"}, null_at_1},
    {{"10", "11", "12", "13", "14"}, null_at_1},
    {{"10", "11", "12", "13", "14"}, null_at_1},
    {"20"},
    {{"21", "22"}, null_at_0},
    {{"21", "22", "23"}, null_at_0},
    {{"21", "22", "23"},
     null_at_0}}.release();

  test_equivalent(expected_result->view(), result->view());

  auto const result_with_nulls_excluded =
    grouped_time_range_rolling_window(table_view{std::vector<column_view>{group_column}},
                                      time_column,
                                      cudf::order::ASCENDING,
                                      input,
                                      preceding,
                                      following,
                                      min_periods,
                                      COLLECT_SET_NULLS_EXCLUDED);

  // After null exclusion, `11`, `21`, and `null` should not appear.
  auto const expected_result_with_nulls_excluded = lists_column_wrapper<cudf::string_view>{
    {"10", "12", "13"},
    {"10", "12", "13"},
    {"10", "12", "13", "14"},
    {"10", "12", "13", "14"},
    {"10", "12", "13", "14"},
    {"20"},
    {"22"},
    {"22", "23"},
    {"22", "23"}}.release();

  test_equivalent(expected_result_with_nulls_excluded->view(), result_with_nulls_excluded->view());
}

TYPED_TEST(TypedCollectListTest, BasicGroupedTimeRangeRollingWindowOnStructs)
{
  using namespace cudf;
  using namespace cudf::test;

  auto const time_column = fixed_width_column_wrapper<cudf::timestamp_D, cudf::timestamp_D::rep>{
    1, 1, 2, 2, 3, 1, 4, 5, 6};
  auto const group_column    = fixed_width_column_wrapper<int32_t>{1, 1, 1, 1, 1, 2, 2, 2, 2};
  auto numeric_member_column = COL_V{10, 11, 12, 13, 14, 20, 21, 22, 23};
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
                                      COLLECT_SET);

  auto expected_numeric_column =
    COL_V{10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 14, 10, 11, 12,
          13, 14, 10, 11, 12, 13, 14, 20, 21, 22, 21, 22, 23, 21, 22, 23};

  auto expected_string_column = strings_column_wrapper{
    "10", "11", "12", "13", "10", "11", "12", "13", "10", "11", "12", "13", "14", "10", "11", "12",
    "13", "14", "10", "11", "12", "13", "14", "20", "21", "22", "21", "22", "23", "21", "22", "23"};

  auto expected_struct_members = std::vector<std::unique_ptr<cudf::column>>{};
  expected_struct_members.emplace_back(expected_numeric_column.release());
  expected_struct_members.emplace_back(expected_string_column.release());

  auto expected_structs_column = make_structs_column(32, std::move(expected_struct_members), 0, {});
  auto expected_offsets_column = COL_S{0, 4, 8, 13, 18, 23, 24, 26, 29, 32}.release();
  auto expected_result         = make_lists_column(
    9, std::move(expected_offsets_column), std::move(expected_structs_column), 0, {});

  test_equivalent(expected_result->view(), result->view());

  auto const result_with_nulls_excluded =
    grouped_time_range_rolling_window(table_view{std::vector<column_view>{group_column}},
                                      time_column,
                                      cudf::order::ASCENDING,
                                      struct_column->view(),
                                      preceding,
                                      following,
                                      min_periods,
                                      COLLECT_SET_NULLS_EXCLUDED);

  test_equivalent(expected_result->view(), result_with_nulls_excluded->view());
}

TYPED_TEST(TypedCollectListTest, GroupedTimeRangeRollingWindowWithMinPeriods)
{
  // Test that min_periods is honoured.
  // i.e. output row is null when min_periods exceeds number of observations.
  using namespace cudf;
  using namespace cudf::test;

  auto const time_column = fixed_width_column_wrapper<cudf::timestamp_D, cudf::timestamp_D::rep>{
    1, 1, 2, 2, 3, 1, 4, 5, 6};
  auto const group_column = fixed_width_column_wrapper<int32_t>{1, 1, 1, 1, 1, 2, 2, 2, 2};
  auto const input        = COL_V{10, 11, 12, 13, 14, 20, 21, 22, 23};
  auto const preceding    = 2;
  auto const following    = 1;
  auto const min_periods  = 4;
  auto const result =
    grouped_time_range_rolling_window(table_view{std::vector<column_view>{group_column}},
                                      time_column,
                                      cudf::order::ASCENDING,
                                      input,
                                      preceding,
                                      following,
                                      min_periods,
                                      COLLECT_SET);

  auto const expected_result = LCL_V{
    {{10, 11, 12, 13},
     {10, 11, 12, 13},
     {10, 11, 12, 13, 14},
     {10, 11, 12, 13, 14},
     {10, 11, 12, 13, 14},
     {},
     {},
     {},
     {}},
    cudf::detail::make_counting_transform_iterator(0, [](auto i) {
      return i < 5;
    })}.release();

  test_equivalent(expected_result->view(), result->view());

  auto const result_with_nulls_excluded =
    grouped_time_range_rolling_window(table_view{std::vector<column_view>{group_column}},
                                      time_column,
                                      cudf::order::ASCENDING,
                                      input,
                                      preceding,
                                      following,
                                      min_periods,
                                      COLLECT_SET_NULLS_EXCLUDED);

  test_equivalent(expected_result->view(), result_with_nulls_excluded->view());
}

TYPED_TEST(TypedCollectListTest, GroupedTimeRangeRollingWindowWithNullsAndMinPeriods)
{
  // Test that min_periods is honoured.
  // i.e. output row is null when min_periods exceeds number of observations.
  using namespace cudf;
  using namespace cudf::test;

  auto const time_column = fixed_width_column_wrapper<cudf::timestamp_D, cudf::timestamp_D::rep>{
    1, 1, 2, 2, 3, 1, 4, 5, 6};
  auto const group_column = fixed_width_column_wrapper<int32_t>{1, 1, 1, 1, 1, 2, 2, 2, 2};
  auto const input       = COL_V{{10, 11, 12, 13, 14, 20, 21, 22, 23}, {1, 0, 1, 1, 1, 1, 0, 1, 1}};
  auto const preceding   = 2;
  auto const following   = 1;
  auto const min_periods = 4;
  auto const result =
    grouped_time_range_rolling_window(table_view{std::vector<column_view>{group_column}},
                                      time_column,
                                      cudf::order::ASCENDING,
                                      input,
                                      preceding,
                                      following,
                                      min_periods,
                                      COLLECT_SET);

  auto null_at_1 = iterator_with_null_at(1);

  // In the results, `11` and `21` should be nulls.
  auto const expected_result = LCL_V{
    {{{10, 11, 12, 13}, null_at_1},
     {{10, 11, 12, 13}, null_at_1},
     {{10, 11, 12, 13, 14}, null_at_1},
     {{10, 11, 12, 13, 14}, null_at_1},
     {{10, 11, 12, 13, 14}, null_at_1},
     {},
     {},
     {},
     {}},
    cudf::detail::make_counting_transform_iterator(0, [](auto i) {
      return i < 5;
    })}.release();

  test_equivalent(expected_result->view(), result->view());

  auto const result_with_nulls_excluded =
    grouped_time_range_rolling_window(table_view{std::vector<column_view>{group_column}},
                                      time_column,
                                      cudf::order::ASCENDING,
                                      input,
                                      preceding,
                                      following,
                                      min_periods,
                                      COLLECT_SET_NULLS_EXCLUDED);

  // After null exclusion, `11`, `21`, and `null` should not appear.
  auto const expected_result_with_nulls_excluded = LCL_V{
    {{10, 12, 13},
     {10, 12, 13},
     {10, 12, 13, 14},
     {10, 12, 13, 14},
     {10, 12, 13, 14},
     {},
     {},
     {},
     {}},
    cudf::detail::make_counting_transform_iterator(
      0, [](auto i) { return i < 5; })}.release();

  test_equivalent(expected_result_with_nulls_excluded->view(), result_with_nulls_excluded->view());
}

TEST_F(CollectSetTest, GroupedTimeRangeRollingWindowOnStringsWithMinPeriods)
{
  // Test that min_periods is honoured.
  // i.e. output row is null when min_periods exceeds number of observations.
  using namespace cudf;
  using namespace cudf::test;

  auto const time_column = fixed_width_column_wrapper<cudf::timestamp_D, cudf::timestamp_D::rep>{
    1, 1, 2, 2, 3, 1, 4, 5, 6};
  auto const group_column = fixed_width_column_wrapper<int32_t>{1, 1, 1, 1, 1, 2, 2, 2, 2};
  auto const input = strings_column_wrapper{"10", "11", "12", "13", "14", "20", "21", "22", "23"};
  auto const preceding   = 2;
  auto const following   = 1;
  auto const min_periods = 4;
  auto const result =
    grouped_time_range_rolling_window(table_view{std::vector<column_view>{group_column}},
                                      time_column,
                                      cudf::order::ASCENDING,
                                      input,
                                      preceding,
                                      following,
                                      min_periods,
                                      COLLECT_SET);

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
    cudf::detail::make_counting_transform_iterator(0, [](auto i) {
      return i < 5;
    })}.release();

  test_equivalent(expected_result->view(), result->view());

  auto const result_with_nulls_excluded =
    grouped_time_range_rolling_window(table_view{std::vector<column_view>{group_column}},
                                      time_column,
                                      cudf::order::ASCENDING,
                                      input,
                                      preceding,
                                      following,
                                      min_periods,
                                      COLLECT_SET_NULLS_EXCLUDED);

  test_equivalent(expected_result->view(), result_with_nulls_excluded->view());
}

TEST_F(CollectSetTest, GroupedTimeRangeRollingWindowOnStringsWithNullsAndMinPeriods)
{
  // Test that min_periods is honoured.
  // i.e. output row is null when min_periods exceeds number of observations.
  using namespace cudf;
  using namespace cudf::test;

  auto const time_column = fixed_width_column_wrapper<cudf::timestamp_D, cudf::timestamp_D::rep>{
    1, 1, 2, 2, 3, 1, 4, 5, 6};
  auto const group_column = fixed_width_column_wrapper<int32_t>{1, 1, 1, 1, 1, 2, 2, 2, 2};
  auto const input = strings_column_wrapper{{"10", "11", "12", "13", "14", "20", "21", "22", "23"},
                                            {1, 0, 1, 1, 1, 1, 0, 1, 1}};
  auto const preceding   = 2;
  auto const following   = 1;
  auto const min_periods = 4;
  auto const result =
    grouped_time_range_rolling_window(table_view{std::vector<column_view>{group_column}},
                                      time_column,
                                      cudf::order::ASCENDING,
                                      input,
                                      preceding,
                                      following,
                                      min_periods,
                                      COLLECT_SET);

  auto null_at_1 = iterator_with_null_at(1);

  // In the results, `11` and `21` should be nulls.
  auto const expected_result = lists_column_wrapper<cudf::string_view>{
    {{{"10", "11", "12", "13"}, null_at_1},
     {{"10", "11", "12", "13"}, null_at_1},
     {{"10", "11", "12", "13", "14"}, null_at_1},
     {{"10", "11", "12", "13", "14"}, null_at_1},
     {{"10", "11", "12", "13", "14"}, null_at_1},
     {},
     {},
     {},
     {}},
    cudf::detail::make_counting_transform_iterator(0, [](auto i) {
      return i < 5;
    })}.release();

  test_equivalent(expected_result->view(), result->view());

  auto const result_with_nulls_excluded =
    grouped_time_range_rolling_window(table_view{std::vector<column_view>{group_column}},
                                      time_column,
                                      cudf::order::ASCENDING,
                                      input,
                                      preceding,
                                      following,
                                      min_periods,
                                      COLLECT_SET_NULLS_EXCLUDED);

  // After null exclusion, `11`, `21`, and `null` should not appear.
  auto const expected_result_with_nulls_excluded = lists_column_wrapper<cudf::string_view>{
    {{"10", "12", "13"},
     {"10", "12", "13"},
     {"10", "12", "13", "14"},
     {"10", "12", "13", "14"},
     {"10", "12", "13", "14"},
     {},
     {},
     {},
     {}},
    cudf::detail::make_counting_transform_iterator(
      0, [](auto i) { return i < 5; })}.release();

  test_equivalent(expected_result_with_nulls_excluded->view(), result_with_nulls_excluded->view());
}

TYPED_TEST(TypedCollectListTest, GroupedTimeRangeRollingWindowOnStructsWithMinPeriods)
{
  // Test that min_periods is honoured.
  // i.e. output row is null when min_periods exceeds number of observations.
  using namespace cudf;
  using namespace cudf::test;

  auto const time_column = fixed_width_column_wrapper<cudf::timestamp_D, cudf::timestamp_D::rep>{
    1, 1, 2, 2, 3, 1, 4, 5, 6};
  auto const group_column    = fixed_width_column_wrapper<int32_t>{1, 1, 1, 1, 1, 2, 2, 2, 2};
  auto numeric_member_column = COL_V{10, 11, 12, 13, 14, 20, 21, 22, 23};
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
                                      COLLECT_SET);

  auto expected_numeric_column = COL_V{10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13,
                                       14, 10, 11, 12, 13, 14, 10, 11, 12, 13, 14};

  auto expected_string_column =
    strings_column_wrapper{"10", "11", "12", "13", "10", "11", "12", "13", "10", "11", "12", "13",
                           "14", "10", "11", "12", "13", "14", "10", "11", "12", "13", "14"};

  auto expected_struct_members = std::vector<std::unique_ptr<cudf::column>>{};
  expected_struct_members.emplace_back(expected_numeric_column.release());
  expected_struct_members.emplace_back(expected_string_column.release());

  auto expected_structs_column = make_structs_column(23, std::move(expected_struct_members), 0, {});
  auto expected_offsets_column = COL_S{0, 4, 8, 13, 18, 23, 23, 23, 23, 23}.release();
  auto expected_validity_iter =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i < 5; });
  auto expected_null_mask =
    cudf::test::detail::make_null_mask(expected_validity_iter, expected_validity_iter + 9);
  auto expected_result = make_lists_column(9,
                                           std::move(expected_offsets_column),
                                           std::move(expected_structs_column),
                                           4,
                                           std::move(expected_null_mask));

  test_equivalent(expected_result->view(), result->view());

  auto const result_with_nulls_excluded =
    grouped_time_range_rolling_window(table_view{std::vector<column_view>{group_column}},
                                      time_column,
                                      cudf::order::ASCENDING,
                                      struct_column->view(),
                                      preceding,
                                      following,
                                      min_periods,
                                      COLLECT_SET_NULLS_EXCLUDED);

  test_equivalent(expected_result->view(), result_with_nulls_excluded->view());
}

#endif
}  // namespace test
}  // namespace cudf
CUDF_TEST_PROGRAM_MAIN()
