/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <cudf/aggregation.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/utilities/device_operators.cuh>
#include <cudf/null_mask.hpp>
#include <cudf/rolling.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/type_lists.hpp>

#include <rmm/device_buffer.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <algorithm>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <memory>

using cudf::size_type;
using namespace cudf::test;

struct RankWindowTest : public cudf::test::BaseFixture {
};

template <typename T>
struct TypedRankWindowTest : public cudf::test::BaseFixture {
};

using TypesForTest = cudf::test::Concat<cudf::test::IntegralTypesNotBool,
                                        cudf::test::FloatingPointTypes,
                                        cudf::test::DurationTypes,
                                        cudf::test::TimestampTypes>;

TYPED_TEST_CASE(TypedRankWindowTest, TypesForTest);

TYPED_TEST(TypedRankWindowTest, RankBasics)
{
  using T = int32_t;

  auto const input_col1 =
    fixed_width_column_wrapper<T>{0, 1, 2, 3, 4, 5, 0, 10, 20, 30, 40, 50}.release();
  auto const input_col2 =
    fixed_width_column_wrapper<T>{1, 1, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5};

  auto const grouping_key = fixed_width_column_wrapper<int32_t>{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
  auto const grouping_keys = cudf::table_view{std::vector<cudf::column_view>{grouping_key}};
  auto const order_by_cols = cudf::table_view{std::vector<cudf::column_view>{input_col2}};

  auto const min_periods = 1;
  auto const preceding   = 4;
  auto const following   = 3;
  auto const preceding_col =
    fixed_width_column_wrapper<size_type>{4, 4, 4, 4, 4, 4, 1, 2, 3, 4, 4, 4};
  auto const following_col =
    fixed_width_column_wrapper<size_type>{1, 2, 3, 4, 5, 6, 4, 3, 2, 1, 1, 1};

  auto rank_output_col = cudf::grouped_rolling_window(grouping_keys,
                                                        input_col1->view(),
                                                        order_by_cols,
                                                        preceding,
                                                        following,
                                                        min_periods,
                                                        cudf::make_rank_aggregation());

  auto rank_output_col2 = cudf::rolling_window(input_col1->view(),
                                               order_by_cols,
                                               preceding_col,
                                               following_col,
                                               min_periods,
                                               cudf::make_rank_aggregation());

  expect_columns_equivalent(
    *rank_output_col,
    fixed_width_column_wrapper<size_type>{{1, 1, 3, 3, 4, 3, 1, 1, 1, 4, 3, 2}}
      .release()
      ->view());

    expect_columns_equivalent(
    *rank_output_col,
    fixed_width_column_wrapper<size_type>{{1, 1, 3, 3, 4, 3, 1, 1, 1, 4, 3, 2}}
      .release()
      ->view());

  auto dense_rank_outpput_col = cudf::grouped_rolling_window(grouping_keys,
                                                       input_col1->view(),
                                                       order_by_cols,
                                                       preceding,
                                                       following,
                                                       min_periods,
                                                       cudf::make_dense_rank_aggregation());

  auto dense_rank_outpput_col2 = cudf::rolling_window(input_col1->view(),
                                                    order_by_cols,
                                                    preceding_col,
                                                    following_col,
                                                    min_periods,
                                                    cudf::make_dense_rank_aggregation());

  expect_columns_equivalent(
    *dense_rank_outpput_col,
    fixed_width_column_wrapper<size_type>{{1, 1, 2, 2, 3, 2, 1, 1, 1, 2, 2, 2}}
      .release()
      ->view());

  expect_columns_equivalent(
    *dense_rank_outpput_col2,
    fixed_width_column_wrapper<size_type>{{1, 1, 2, 2, 3, 2, 1, 1, 1, 2, 2, 2}}
      .release()
      ->view());
}

TYPED_TEST(TypedRankWindowTest, RankWithNulls)
{
  using T = TypeParam;

  auto const input_col = fixed_width_column_wrapper<T>{{0, 1, 2, 3, 4, 5, 0, 10, 20, 30, 40, 50},
                                                       {1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1}}
                           .release();
  auto const input_col2 =
    // fixed_width_column_wrapper<T>{{1, 1, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5}, {1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1}};
    fixed_width_column_wrapper<T>{{-5, -5, -4, -4, -3, -3, -2, -2, -2, -1, -1, -1}, {1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1}};

  auto const grouping_key = fixed_width_column_wrapper<int32_t>{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
  auto const grouping_keys = cudf::table_view{std::vector<cudf::column_view>{grouping_key}};
  auto const order_by_cols = cudf::table_view{std::vector<cudf::column_view>{input_col2}};

  auto const preceding   = 4;
  auto const following   = 3;
  auto const min_periods = 1;

  auto rank_output_col = cudf::grouped_rolling_window(grouping_keys,
                                                      input_col->view(),
                                                      order_by_cols,
                                                      preceding,
                                                      following,
                                                      min_periods,
                                                      cudf::make_rank_aggregation());

  expect_columns_equivalent(
    *rank_output_col,
    fixed_width_column_wrapper<size_type>{{1, 1, 3, 4, 4, 3, 1, 1, 3, 4, 3, 2}}

      .release()
      ->view());

  auto const dense_rank_output_col = cudf::grouped_rolling_window(grouping_keys,
                                                             input_col->view(),
                                                             order_by_cols,
                                                             preceding,
                                                             following,
                                                             min_periods,
                                                             cudf::make_dense_rank_aggregation());

  expect_columns_equivalent(
    *dense_rank_output_col,
    fixed_width_column_wrapper<size_type>{{1, 1, 2, 3, 4, 3, 1, 1, 2, 3, 3, 2}}
      .release()
      ->view());
}

TYPED_TEST(TypedRankWindowTest, TestRankWithNoGrouping)
{
  using T = TypeParam;

  auto const input_col =
    fixed_width_column_wrapper<T>{{10, 12, 12, 14, 21, 21}, {1, 1, 0, 1, 1, 1}}.release();
  auto const input_col2 =
    fixed_width_column_wrapper<T>{1, 1, 2, 2, 3, 3};

  auto const grouping_keys = cudf::table_view{std::vector<cudf::column_view>{}};
  auto const order_by_cols = cudf::table_view{std::vector<cudf::column_view>{input_col2}};

  auto const preceding   = 4;
  auto const following   = 3;
  auto const min_periods = 1;

  auto rank_output_col = cudf::grouped_rolling_window(grouping_keys,
                                                      input_col->view(),
                                                      order_by_cols,
                                                      preceding,
                                                      following,
                                                      min_periods,
                                                      cudf::make_rank_aggregation());

  expect_columns_equivalent(
    *rank_output_col,
    fixed_width_column_wrapper<size_type>{{1, 1, 3, 3, 4, 3}}.release()->view());

  auto const dense_rank_output_col = cudf::grouped_rolling_window(grouping_keys,
                                                             input_col->view(),
                                                             order_by_cols,
                                                             preceding,
                                                             following,
                                                             min_periods,
                                                             cudf::make_dense_rank_aggregation());

  expect_columns_equivalent(
    *dense_rank_output_col,
    fixed_width_column_wrapper<size_type>{{1, 1, 2, 2, 3, 2}}.release()->view());
}

TYPED_TEST(TypedRankWindowTest, TestRankWithAllNullInput)
{
  using T = TypeParam;

  auto const input_col = fixed_width_column_wrapper<T>{
    {0, 1, 2, 3, 4, 5, 0, 10, 20, 30, 40, 50},
    cudf::detail::make_counting_transform_iterator(0, [](auto i) {
      return false;
    })}.release();

  auto const input_col2 =
    fixed_width_column_wrapper<T>{5, 5, 5, 4, 4, 4, 3, 3, 2, 2, 1, 1};

  auto const grouping_key = fixed_width_column_wrapper<int32_t>{0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
  auto const grouping_keys = cudf::table_view{std::vector<cudf::column_view>{grouping_key}};
  auto const order_by_cols = cudf::table_view{std::vector<cudf::column_view>{input_col2}};

  auto const preceding   = 4;
  auto const following   = 3;
  auto const min_periods = 1;

  auto rank_output_col = cudf::grouped_rolling_window(grouping_keys,
                                                        input_col->view(),
                                                        order_by_cols,
                                                        preceding,
                                                        following,
                                                        min_periods,
                                                        cudf::make_rank_aggregation());
  expect_columns_equivalent(
    *rank_output_col,
    fixed_width_column_wrapper<size_type>{{1, 1, 1, 4, 3, 2, 4, 1, 2, 2, 4, 3}}
      .release()
      ->view());

  auto const dense_rank_output_col = cudf::grouped_rolling_window(grouping_keys,
                                                             input_col->view(),
                                                             order_by_cols,
                                                             preceding,
                                                             following,
                                                             min_periods,
                                                             cudf::make_dense_rank_aggregation());

  expect_columns_equivalent(
    *dense_rank_output_col,
    fixed_width_column_wrapper<size_type>{{1, 1, 1, 2, 2, 2, 2, 1, 2, 2, 3, 2}}
      .release()
      ->view());
}

TEST_F(RankWindowTest, RankWithoutGroupby)
{

  auto const input_col = strings_column_wrapper{
    {"0", "1", "2", "3", "4", "5"}, cudf::detail::make_counting_transform_iterator(0, [](auto i) {
      return false;
    })}.release();

  auto const grouping_key  = fixed_width_column_wrapper<int32_t>{0, 0, 0, 0, 0, 0};
  auto const grouping_keys = cudf::table_view{std::vector<cudf::column_view>{grouping_key}};

  auto const preceding   = 4;
  auto const following   = 3;
  auto const min_periods = 1;

  EXPECT_THROW(cudf::grouped_rolling_window(grouping_keys,
                                            input_col->view(),
                                            preceding,
                                            following,
                                            min_periods,
                                            cudf::make_rank_aggregation()),
               cudf::logic_error);
  EXPECT_THROW(cudf::grouped_rolling_window(grouping_keys,
                                            input_col->view(),
                                            preceding,
                                            following,
                                            min_periods,
                                            cudf::make_dense_rank_aggregation()),
               cudf::logic_error);
}
