/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
#include <cudf_test/default_stream.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/rolling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>

class RollingTest : public cudf::test::BaseFixture {};

TEST_F(RollingTest, FixedSize)
{
  cudf::test::fixed_width_column_wrapper<cudf::size_type> input({1, 2, 3, 4, 5, 6, 7, 8, 9});

  cudf::rolling_window(input,
                       2,
                       3,
                       1,
                       *cudf::make_min_aggregation<cudf::rolling_aggregation>(),
                       cudf::test::get_default_stream());
}

TEST_F(RollingTest, FixedSizeDefault)
{
  cudf::test::fixed_width_column_wrapper<cudf::size_type> input({1, 2, 3, 4, 5, 6, 7, 8, 9});
  cudf::test::fixed_width_column_wrapper<cudf::size_type> defaults({42, 42, 42, 42, 9, 9, 7, 1, 1});

  cudf::rolling_window(input,
                       defaults,
                       2,
                       3,
                       1,
                       *cudf::make_lead_aggregation<cudf::rolling_aggregation>(1),
                       cudf::test::get_default_stream());
}

TEST_F(RollingTest, VariableSize)
{
  cudf::test::fixed_width_column_wrapper<cudf::size_type> input({1, 2, 3, 4, 5, 6, 7, 8, 9});
  cudf::test::fixed_width_column_wrapper<cudf::size_type> preceding({2, 2, 2, 2, 3, 3, 3, 3, 3});
  cudf::test::fixed_width_column_wrapper<cudf::size_type> following({3, 3, 3, 3, 3, 2, 2, 2, 2});

  cudf::rolling_window(input,
                       preceding,
                       following,
                       1,
                       *cudf::make_min_aggregation<cudf::rolling_aggregation>(),
                       cudf::test::get_default_stream());
}

class GroupedRollingTest : public cudf::test::BaseFixture {};

TEST_F(GroupedRollingTest, FixedSize)
{
  cudf::test::fixed_width_column_wrapper<cudf::size_type> input({1, 2, 3, 4, 5, 6, 7, 8, 9});

  cudf::test::fixed_width_column_wrapper<cudf::size_type> key_0({1, 1, 1, 2, 2, 2, 3, 3, 3});

  cudf::test::fixed_width_column_wrapper<cudf::size_type> key_1({4, 4, 4, 5, 5, 5, 6, 6, 6});

  cudf::table_view grouping_keys{std::vector<cudf::column_view>{key_0, key_1}};

  cudf::grouped_rolling_window(grouping_keys,
                               input,
                               2,
                               3,
                               1,
                               *cudf::make_min_aggregation<cudf::rolling_aggregation>(),
                               cudf::test::get_default_stream());
}

TEST_F(GroupedRollingTest, FixedSizeDefault)
{
  cudf::test::fixed_width_column_wrapper<cudf::size_type> input({1, 2, 3, 4, 5, 6, 7, 8, 9});

  cudf::test::fixed_width_column_wrapper<cudf::size_type> key_0({1, 1, 1, 2, 2, 2, 3, 3, 3});

  cudf::test::fixed_width_column_wrapper<cudf::size_type> key_1({4, 4, 4, 5, 5, 5, 6, 6, 6});

  cudf::test::fixed_width_column_wrapper<cudf::size_type> defaults({42, 42, 42, 42, 9, 9, 7, 1, 1});

  cudf::table_view grouping_keys{std::vector<cudf::column_view>{key_0, key_1}};

  cudf::grouped_rolling_window(grouping_keys,
                               input,
                               defaults,
                               2,
                               3,
                               1,
                               *cudf::make_lead_aggregation<cudf::rolling_aggregation>(1),
                               cudf::test::get_default_stream());
}

TEST_F(GroupedRollingTest, WindowBounds)
{
  cudf::test::fixed_width_column_wrapper<cudf::size_type> input({1, 2, 3, 4, 5, 6, 7, 8, 9});

  cudf::test::fixed_width_column_wrapper<cudf::size_type> key_0({1, 1, 1, 2, 2, 2, 3, 3, 3});

  cudf::test::fixed_width_column_wrapper<cudf::size_type> key_1({4, 4, 4, 5, 5, 5, 6, 6, 6});

  auto const unbounded_preceding = cudf::window_bounds::unbounded();
  auto const following           = cudf::window_bounds::get(1L);

  cudf::table_view grouping_keys{std::vector<cudf::column_view>{key_0, key_1}};

  cudf::grouped_rolling_window(grouping_keys,
                               input,
                               unbounded_preceding,
                               following,
                               1,
                               *cudf::make_min_aggregation<cudf::rolling_aggregation>(),
                               cudf::test::get_default_stream());
}

TEST_F(GroupedRollingTest, WindowBoundsDefault)
{
  cudf::test::fixed_width_column_wrapper<cudf::size_type> input({1, 2, 3, 4, 5, 6, 7, 8, 9});

  cudf::test::fixed_width_column_wrapper<cudf::size_type> key_0({1, 1, 1, 2, 2, 2, 3, 3, 3});

  cudf::test::fixed_width_column_wrapper<cudf::size_type> key_1({4, 4, 4, 5, 5, 5, 6, 6, 6});

  cudf::test::fixed_width_column_wrapper<cudf::size_type> defaults({42, 42, 42, 42, 9, 9, 7, 1, 1});

  auto const unbounded_preceding = cudf::window_bounds::unbounded();
  auto const following           = cudf::window_bounds::get(1L);

  cudf::table_view grouping_keys{std::vector<cudf::column_view>{key_0, key_1}};

  cudf::grouped_rolling_window(grouping_keys,
                               input,
                               defaults,
                               unbounded_preceding,
                               following,
                               1,
                               *cudf::make_lead_aggregation<cudf::rolling_aggregation>(1),
                               cudf::test::get_default_stream());
}

class GroupedTimeRollingTest : public cudf::test::BaseFixture {};

TEST_F(GroupedTimeRollingTest, FixedSize)
{
  auto const grp_col =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  auto const agg_col = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, {1, 1, 1, 1, 1, 0, 1, 1, 1, 1}};
  auto const time_col =
    cudf::test::fixed_width_column_wrapper<cudf::timestamp_D, cudf::timestamp_D::rep>{
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, {0, 0, 0, 0, 1, 1, 1, 1, 1, 1}};

  auto const grouping_keys = cudf::table_view{std::vector<cudf::column_view>{grp_col}};
  auto const preceding     = 1L;
  auto const following     = 1L;
  auto const min_periods   = 1L;
  cudf::grouped_time_range_rolling_window(
    grouping_keys,
    time_col,
    cudf::order::ASCENDING,
    agg_col,
    preceding,
    following,
    min_periods,
    *cudf::make_count_aggregation<cudf::rolling_aggregation>(),
    cudf::test::get_default_stream());
}

TEST_F(GroupedTimeRollingTest, WindowBounds)
{
  auto const grp_col =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  auto const agg_col = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, {1, 1, 1, 1, 1, 0, 1, 1, 1, 1}};
  auto const time_col =
    cudf::test::fixed_width_column_wrapper<cudf::timestamp_D, cudf::timestamp_D::rep>{
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, {0, 0, 0, 0, 1, 1, 1, 1, 1, 1}};

  auto const grouping_keys       = cudf::table_view{std::vector<cudf::column_view>{grp_col}};
  auto const unbounded_preceding = cudf::window_bounds::unbounded();
  auto const following           = cudf::window_bounds::get(1L);

  auto const min_periods = 1L;
  cudf::grouped_time_range_rolling_window(
    grouping_keys,
    time_col,
    cudf::order::ASCENDING,
    agg_col,
    unbounded_preceding,
    following,
    min_periods,
    *cudf::make_count_aggregation<cudf::rolling_aggregation>(),
    cudf::test::get_default_stream());
}

class GroupedRangeRollingTest : public cudf::test::BaseFixture {};

TEST_F(GroupedRangeRollingTest, RangeWindowBounds)
{
  auto const grp_col = cudf::test::fixed_width_column_wrapper<int>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  auto const agg_col = cudf::test::fixed_width_column_wrapper<int>{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                                                                   {1, 1, 1, 1, 1, 0, 1, 1, 1, 1}};

  auto const order_by = cudf::test::fixed_width_column_wrapper<int>{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                                                                    {0, 0, 0, 0, 1, 1, 1, 1, 1, 1}};

  cudf::range_window_bounds preceding = cudf::range_window_bounds::get(
    cudf::numeric_scalar<int>{int{1}, true, cudf::test::get_default_stream()});

  cudf::range_window_bounds following = cudf::range_window_bounds::get(
    cudf::numeric_scalar<int>{int{1}, true, cudf::test::get_default_stream()});

  auto const min_periods = cudf::size_type{1};

  auto const grouping_keys = cudf::table_view{std::vector<cudf::column_view>{grp_col}};

  cudf::grouped_range_rolling_window(grouping_keys,
                                     order_by,
                                     cudf::order::ASCENDING,
                                     agg_col,
                                     preceding,
                                     following,
                                     min_periods,
                                     *cudf::make_count_aggregation<cudf::rolling_aggregation>(),
                                     cudf::test::get_default_stream());
}
