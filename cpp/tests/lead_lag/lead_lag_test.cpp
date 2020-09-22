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

#include <thrust/iterator/counting_iterator.h>
#include <algorithm>
#include <cudf/aggregation.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/utilities/device_operators.cuh>
#include <cudf/null_mask.hpp>
#include <cudf/rolling.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <rmm/device_buffer.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/cudf_gtest.hpp>
#include <tests/utilities/type_lists.hpp>
#include "cudf/scalar/scalar_factories.hpp"
#include "thrust/iterator/transform_iterator.h"

using cudf::size_type;
using namespace cudf::test;

struct LeadLagWindowTest : public cudf::test::BaseFixture {
};

template <typename T>
struct TypedLeadLagWindowTest : public cudf::test::BaseFixture {
};

using TypesForTest = cudf::test::Concat<cudf::test::IntegralTypes,
                                        cudf::test::FloatingPointTypes,
                                        cudf::test::DurationTypes,
                                        cudf::test::TimestampTypes>;

TYPED_TEST_CASE(TypedLeadLagWindowTest, TypesForTest);

TYPED_TEST(TypedLeadLagWindowTest, LeadLagBasics)
{
  using T = int32_t;

  auto const input_col =
    fixed_width_column_wrapper<T>{0, 1, 2, 3, 4, 5, 0, 10, 20, 30, 40, 50}.release();
  auto const input_size   = input_col->size();
  auto const grouping_key = fixed_width_column_wrapper<int32_t>{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
  auto const grouping_keys = cudf::table_view{std::vector<cudf::column_view>{grouping_key}};

  auto lead_3_output_col = cudf::grouped_rolling_window(
    grouping_keys, input_col->view(), 4, 3, 1, cudf::make_lead_aggregation(3));

  expect_columns_equivalent(
    *lead_3_output_col,
    fixed_width_column_wrapper<T>{{3, 4, 5, -1, -1, -1, 30, 40, 50, -1, -1, -1},
                                  {1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0}}
      .release()
      ->view());

  auto lag_2_output_col = cudf::grouped_rolling_window(
    grouping_keys, input_col->view(), 3, 2, 1, cudf::make_lag_aggregation(2));

  expect_columns_equivalent(
    *lag_2_output_col,
    fixed_width_column_wrapper<T>{{-1, -1, 0, 1, 2, 3, -1, -1, 0, 10, 20, 30},
                                  {0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1}}
      .release()
      ->view());
}

TYPED_TEST(TypedLeadLagWindowTest, LeadLagWithNulls)
{
  using T = TypeParam;

  auto const input_col = fixed_width_column_wrapper<T>{{0, 1, 2, 3, 4, 5, 0, 10, 20, 30, 40, 50},
                                                       {1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1}}
                           .release();
  auto const input_size   = input_col->size();
  auto const grouping_key = fixed_width_column_wrapper<int32_t>{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
  auto const grouping_keys = cudf::table_view{std::vector<cudf::column_view>{grouping_key}};

  auto lead_3_output_col = cudf::grouped_rolling_window(
    grouping_keys, input_col->view(), 4, 3, 1, cudf::make_lead_aggregation(3));

  expect_columns_equivalent(
    *lead_3_output_col,
    fixed_width_column_wrapper<T>{{3, 4, 5, -1, -1, -1, 30, 40, 50, -1, -1, -1},
                                  {1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0}}
      .release()
      ->view());

  auto const lag_2_output_col = cudf::grouped_rolling_window(
    grouping_keys, input_col->view(), 3, 2, 1, cudf::make_lag_aggregation(2));

  expect_columns_equivalent(
    *lag_2_output_col,
    fixed_width_column_wrapper<T>{{-1, -1, 0, 1, -1, 3, -1, -1, 0, 10, -1, 30},
                                  {0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1}}
      .release()
      ->view());
}

TYPED_TEST(TypedLeadLagWindowTest, TestLeadLagWithDefaults)
{
  using T = TypeParam;

  auto const input_col = fixed_width_column_wrapper<T>{{0, 1, 2, 3, 4, 5, 0, 10, 20, 30, 40, 50},
                                                       {1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1}}
                           .release();
  auto const input_size   = input_col->size();
  auto const grouping_key = fixed_width_column_wrapper<int32_t>{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
  auto const grouping_keys = cudf::table_view{std::vector<cudf::column_view>{grouping_key}};

  auto const default_value =
    cudf::make_fixed_width_scalar(detail::fixed_width_type_converter<int32_t, T>{}(99));
  auto const default_outputs = cudf::make_column_from_scalar(*default_value, input_col->size());

  auto lead_3_output_col = cudf::grouped_rolling_window(
    grouping_keys, input_col->view(), *default_outputs, 4, 3, 1, cudf::make_lead_aggregation(3));
  expect_columns_equivalent(
    *lead_3_output_col,
    fixed_width_column_wrapper<T>{{3, 4, 5, 99, 99, 99, 30, 40, 50, 99, 99, 99},
                                  {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}}
      .release()
      ->view());

  auto const lag_2_output_col = cudf::grouped_rolling_window(
    grouping_keys, input_col->view(), *default_outputs, 3, 2, 1, cudf::make_lag_aggregation(2));

  expect_columns_equivalent(
    *lag_2_output_col,
    fixed_width_column_wrapper<T>{{99, 99, 0, 1, -1, 3, 99, 99, 0, 10, -1, 30},
                                  {1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1}}
      .release()
      ->view());
}

TYPED_TEST(TypedLeadLagWindowTest, TestLeadLagWithOutOfRangeOffsets)
{
  using T = TypeParam;

  auto const input_col = fixed_width_column_wrapper<T>{{0, 1, 2, 3, 4, 5, 0, 10, 20, 30, 40, 50},
                                                       {1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1}}
                           .release();
  auto const input_size   = input_col->size();
  auto const grouping_key = fixed_width_column_wrapper<int32_t>{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
  auto const grouping_keys = cudf::table_view{std::vector<cudf::column_view>{grouping_key}};

  auto const default_value =
    cudf::make_fixed_width_scalar(detail::fixed_width_type_converter<int32_t, T>{}(99));
  auto const default_outputs = cudf::make_column_from_scalar(*default_value, input_col->size());

  auto lead_30_output_col = cudf::grouped_rolling_window(
    grouping_keys, input_col->view(), 4, 3, 1, cudf::make_lead_aggregation(30));

  expect_columns_equivalent(
    *lead_30_output_col,
    fixed_width_column_wrapper<T>{{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}
      .release()
      ->view());

  auto const lag_20_output_col = cudf::grouped_rolling_window(
    grouping_keys, input_col->view(), *default_outputs, 3, 2, 1, cudf::make_lag_aggregation(20));

  expect_columns_equivalent(
    *lag_20_output_col,
    fixed_width_column_wrapper<T>{{99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99},
                                  {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}}
      .release()
      ->view());
}

TYPED_TEST(TypedLeadLagWindowTest, TestLeadLagWithZeroOffsets)
{
  using T = TypeParam;

  auto const input_col = fixed_width_column_wrapper<T>{{0, 1, 2, 3, 4, 5, 0, 10, 20, 30, 40, 50},
                                                       {1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1}}
                           .release();
  auto const input_size   = input_col->size();
  auto const grouping_key = fixed_width_column_wrapper<int32_t>{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
  auto const grouping_keys = cudf::table_view{std::vector<cudf::column_view>{grouping_key}};

  auto lead_0_output_col = cudf::grouped_rolling_window(
    grouping_keys, input_col->view(), 4, 3, 1, cudf::make_lead_aggregation(0));

  expect_columns_equivalent(*lead_0_output_col, *input_col);

  auto const lag_0_output_col = cudf::grouped_rolling_window(
    grouping_keys, input_col->view(), 3, 2, 1, cudf::make_lag_aggregation(0));
  ;

  expect_columns_equivalent(*lag_0_output_col, *input_col);
}

TYPED_TEST(TypedLeadLagWindowTest, TestLeadLagWithNegativeOffsets)
{
  using T = TypeParam;

  auto const input_col = fixed_width_column_wrapper<T>{{0, 1, 2, 3, 4, 5, 0, 10, 20, 30, 40, 50},
                                                       {1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1}}
                           .release();
  auto const input_size   = input_col->size();
  auto const grouping_key = fixed_width_column_wrapper<int32_t>{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
  auto const grouping_keys = cudf::table_view{std::vector<cudf::column_view>{grouping_key}};

  auto const default_value =
    cudf::make_fixed_width_scalar(detail::fixed_width_type_converter<int32_t, T>{}(99));
  auto const default_outputs = cudf::make_column_from_scalar(*default_value, input_col->size());

  auto lag_minus_3_output_col = cudf::grouped_rolling_window(
    grouping_keys, input_col->view(), *default_outputs, 4, 3, 1, cudf::make_lag_aggregation(-3));
  ;

  expect_columns_equivalent(
    *lag_minus_3_output_col,
    fixed_width_column_wrapper<T>{{3, 4, 5, 99, 99, 99, 30, 40, 50, 99, 99, 99},
                                  {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}}
      .release()
      ->view());

  auto const lead_minus_2_output_col = cudf::grouped_rolling_window(
    grouping_keys, input_col->view(), *default_outputs, 3, 2, 1, cudf::make_lead_aggregation(-2));

  expect_columns_equivalent(
    *lead_minus_2_output_col,
    fixed_width_column_wrapper<T>{{99, 99, 0, 1, -1, 3, 99, 99, 0, 10, -1, 30},
                                  {1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1}}
      .release()
      ->view());
}

TYPED_TEST(TypedLeadLagWindowTest, TestLeadLagWithNoGrouping)
{
  using T = TypeParam;

  auto const input_col =
    fixed_width_column_wrapper<T>{{0, 1, 2, 3, 4, 5}, {1, 1, 0, 1, 1, 1}}.release();
  auto const input_size    = input_col->size();
  auto const grouping_keys = cudf::table_view{std::vector<cudf::column_view>{}};

  auto const default_value =
    cudf::make_fixed_width_scalar(detail::fixed_width_type_converter<int32_t, T>{}(99));
  auto const default_outputs = cudf::make_column_from_scalar(*default_value, input_col->size());

  auto lead_3_output_col = cudf::grouped_rolling_window(
    grouping_keys, input_col->view(), *default_outputs, 4, 3, 1, cudf::make_lead_aggregation(3));
  ;

  expect_columns_equivalent(
    *lead_3_output_col,
    fixed_width_column_wrapper<T>{{3, 4, 5, 99, 99, 99}, {1, 1, 1, 1, 1, 1}}.release()->view());

  auto const lag_2_output_col = cudf::grouped_rolling_window(
    grouping_keys, input_col->view(), *default_outputs, 3, 2, 1, cudf::make_lag_aggregation(2));

  expect_columns_equivalent(
    *lag_2_output_col,
    fixed_width_column_wrapper<T>{{99, 99, 0, 1, -1, 3}, {1, 1, 1, 1, 0, 1}}.release()->view());
}

TEST_F(LeadLagWindowTest, TestLeadLagWithAllNullInput)
{
  using T = int32_t;

  auto const input_col = fixed_width_column_wrapper<T>{
    {0, 1, 2, 3, 4, 5, 0, 10, 20, 30, 40, 50}, make_counting_transform_iterator(0, [](auto i) {
      return false;
    })}.release();
  auto const input_size   = input_col->size();
  auto const grouping_key = fixed_width_column_wrapper<int32_t>{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
  auto const grouping_keys = cudf::table_view{std::vector<cudf::column_view>{grouping_key}};

  auto const default_value =
    cudf::make_fixed_width_scalar(detail::fixed_width_type_converter<int32_t, T>{}(99));
  auto const default_outputs = cudf::make_column_from_scalar(*default_value, input_col->size());

  auto lead_3_output_col = cudf::grouped_rolling_window(
    grouping_keys, input_col->view(), *default_outputs, 4, 3, 1, cudf::make_lead_aggregation(3));
  expect_columns_equivalent(
    *lead_3_output_col,
    fixed_width_column_wrapper<T>{{-1, -1, -1, 99, 99, 99, -1, -1, -1, 99, 99, 99},
                                  {0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1}}
      .release()
      ->view());

  auto const lag_2_output_col = cudf::grouped_rolling_window(
    grouping_keys, input_col->view(), *default_outputs, 3, 2, 1, cudf::make_lag_aggregation(2));

  expect_columns_equivalent(
    *lag_2_output_col,
    fixed_width_column_wrapper<T>{{99, 99, -1, -1, -1, -1, 99, 99, -1, -1, -1, -1},
                                  {1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0}}
      .release()
      ->view());
}

CUDF_TEST_PROGRAM_MAIN()
