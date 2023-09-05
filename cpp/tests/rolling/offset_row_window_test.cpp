/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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
#include <cudf_test/iterator_utilities.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/groupby.hpp>
#include <cudf/lists/explode.hpp>
#include <cudf/rolling.hpp>
#include <cudf/utilities/default_stream.hpp>

template <typename T>
using fwcw = cudf::test::fixed_width_column_wrapper<T>;
template <typename T>
using decimals_column = cudf::test::fixed_point_column_wrapper<T>;
using ints_column     = fwcw<int32_t>;
using bigints_column  = fwcw<int64_t>;
using strings_column  = cudf::test::strings_column_wrapper;
using lists_column    = cudf::test::lists_column_wrapper<int32_t>;
using column_ptr      = std::unique_ptr<cudf::column>;
using cudf::test::iterators::no_nulls;
using cudf::test::iterators::nulls_at;

auto constexpr null = int32_t{0};  // NULL representation for int32_t;

struct OffsetRowWindowTest : public cudf::test::BaseFixture {
  static ints_column const _keys;    // {0, 0, 0, 0, 0, 0, 1, 1, 1, 1};
  static ints_column const _values;  // {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  struct rolling_runner {
    cudf::window_bounds _preceding, _following;
    cudf::size_type _min_periods;
    bool _grouped = true;

    rolling_runner(cudf::window_bounds const& preceding,
                   cudf::window_bounds const& following,
                   cudf::size_type min_periods_ = 1)
      : _preceding{preceding}, _following{following}, _min_periods{min_periods_}
    {
    }

    rolling_runner& min_periods(cudf::size_type min_periods_)
    {
      _min_periods = min_periods_;
      return *this;
    }

    rolling_runner& grouped(bool grouped_)
    {
      _grouped = grouped_;
      return *this;
    }

    std::unique_ptr<cudf::column> operator()(cudf::rolling_aggregation const& agg) const
    {
      auto const grouping_keys =
        _grouped ? std::vector<cudf::column_view>{_keys} : std::vector<cudf::column_view>{};
      return cudf::grouped_rolling_window(
        cudf::table_view{grouping_keys}, _values, _preceding, _following, _min_periods, agg);
    }
  };
};

ints_column const OffsetRowWindowTest::_keys{0, 0, 0, 0, 0, 0, 1, 1, 1, 1};
ints_column const OffsetRowWindowTest::_values{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

auto const AGG_COUNT_NON_NULL =
  cudf::make_count_aggregation<cudf::rolling_aggregation>(cudf::null_policy::EXCLUDE);
auto const AGG_COUNT_ALL =
  cudf::make_count_aggregation<cudf::rolling_aggregation>(cudf::null_policy::INCLUDE);
auto const AGG_MIN          = cudf::make_min_aggregation<cudf::rolling_aggregation>();
auto const AGG_MAX          = cudf::make_max_aggregation<cudf::rolling_aggregation>();
auto const AGG_SUM          = cudf::make_sum_aggregation<cudf::rolling_aggregation>();
auto const AGG_COLLECT_LIST = cudf::make_collect_list_aggregation<cudf::rolling_aggregation>();

TEST_F(OffsetRowWindowTest, OffsetRowWindow_3_to_Minus_1)
{
  auto const preceding = cudf::window_bounds::get(3);
  auto const following = cudf::window_bounds::get(-1);
  auto run_rolling     = rolling_runner{preceding, following}.min_periods(1).grouped(true);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*run_rolling(*AGG_COUNT_NON_NULL),
                                 ints_column{{0, 1, 2, 2, 2, 2, 0, 1, 2, 2}, nulls_at({0, 6})});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*run_rolling(*AGG_COUNT_ALL),
                                 ints_column{{0, 1, 2, 2, 2, 2, 0, 1, 2, 2}, nulls_at({0, 6})});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *run_rolling(*AGG_MIN), ints_column{{null, 0, 0, 1, 2, 3, null, 6, 6, 7}, nulls_at({0, 6})});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *run_rolling(*AGG_MAX), ints_column{{null, 0, 1, 2, 3, 4, null, 6, 7, 8}, nulls_at({0, 6})});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *run_rolling(*AGG_SUM),
    bigints_column{{null, 0, 1, 3, 5, 7, null, 6, 13, 15}, nulls_at({0, 6})});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *run_rolling(*AGG_COLLECT_LIST),
    lists_column{{{}, {0}, {0, 1}, {1, 2}, {2, 3}, {3, 4}, {}, {6}, {6, 7}, {7, 8}},
                 nulls_at({0, 6})});

  run_rolling.min_periods(0);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*run_rolling(*AGG_COUNT_NON_NULL),
                                 ints_column{{0, 1, 2, 2, 2, 2, 0, 1, 2, 2}, no_nulls()});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*run_rolling(*AGG_COUNT_ALL),
                                 ints_column{{0, 1, 2, 2, 2, 2, 0, 1, 2, 2}, no_nulls()});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *run_rolling(*AGG_COLLECT_LIST),
    lists_column{{{}, {0}, {0, 1}, {1, 2}, {2, 3}, {3, 4}, {}, {6}, {6, 7}, {7, 8}}, no_nulls()});
}

TEST_F(OffsetRowWindowTest, OffsetRowWindow_0_to_2)
{
  auto const preceding = cudf::window_bounds::get(0);
  auto const following = cudf::window_bounds::get(2);
  auto run_rolling     = rolling_runner{preceding, following}.min_periods(1).grouped(true);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *run_rolling(*AGG_COUNT_NON_NULL),
    ints_column{{2, 2, 2, 2, 1, null, 2, 2, 1, null}, nulls_at({5, 9})});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *run_rolling(*AGG_COUNT_ALL),
    ints_column{{2, 2, 2, 2, 1, null, 2, 2, 1, null}, nulls_at({5, 9})});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *run_rolling(*AGG_MIN), ints_column{{1, 2, 3, 4, 5, null, 7, 8, 9, null}, nulls_at({5, 9})});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *run_rolling(*AGG_MAX), ints_column{{2, 3, 4, 5, 5, null, 8, 9, 9, null}, nulls_at({5, 9})});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *run_rolling(*AGG_SUM),
    bigints_column{{3, 5, 7, 9, 5, null, 15, 17, 9, null}, nulls_at({5, 9})});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *run_rolling(*AGG_COLLECT_LIST),
    lists_column{{{1, 2}, {2, 3}, {3, 4}, {4, 5}, {5}, {}, {7, 8}, {8, 9}, {9}, {}},
                 nulls_at({5, 9})});

  run_rolling.min_periods(0);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*run_rolling(*AGG_COUNT_NON_NULL),
                                 ints_column{{2, 2, 2, 2, 1, 0, 2, 2, 1, 0}, no_nulls()});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*run_rolling(*AGG_COUNT_ALL),
                                 ints_column{{2, 2, 2, 2, 1, 0, 2, 2, 1, 0}, no_nulls()});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *run_rolling(*AGG_COLLECT_LIST),
    lists_column{{{1, 2}, {2, 3}, {3, 4}, {4, 5}, {5}, {}, {7, 8}, {8, 9}, {9}, {}}, no_nulls});
}
