/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
#include <cudf_test/type_lists.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/column/column.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/rolling.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/bit.hpp>

#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>

#include <algorithm>
#include <vector>

template <typename T>
using fwcw = cudf::test::fixed_width_column_wrapper<T>;
template <typename T>
using decimals_column = cudf::test::fixed_point_column_wrapper<T>;
using ints_column     = fwcw<int32_t>;
using bigints_column  = fwcw<int64_t>;
using strings_column  = cudf::test::strings_column_wrapper;
using column_ptr      = std::unique_ptr<cudf::column>;

struct BaseGroupedRollingRangeOrderByDecimalTest : public cudf::test::BaseFixture {
  // Stand-in for std::pow(10, n), but for integral return.
  static constexpr std::array<int32_t, 6> pow10{1, 10, 100, 1000, 10000, 100000};
  // Test data.
  column_ptr const grouping_keys = ints_column{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2}.release();
  column_ptr const agg_values    = ints_column{1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3}.release();
  cudf::size_type const num_rows = grouping_keys->size();
};

using base = BaseGroupedRollingRangeOrderByDecimalTest;  // Shortcut to base test class.

template <typename DecimalT>
struct GroupedRollingRangeOrderByDecimalTypedTest : BaseGroupedRollingRangeOrderByDecimalTest {
  using Rep = typename DecimalT::rep;

  [[nodiscard]] auto make_fixed_point_range_bounds(typename DecimalT::rep value,
                                                   numeric::scale_type scale) const
  {
    return cudf::range_window_bounds::get(*cudf::make_fixed_point_scalar<DecimalT>(value, scale));
  }

  [[nodiscard]] auto make_unbounded_fixed_point_range_bounds() const
  {
    return cudf::range_window_bounds::unbounded(cudf::data_type{cudf::type_to_id<DecimalT>()});
  }

  /// For different scales, generate order_by column with
  /// the same effective values:           [0, 100,   200,   300,   ... 1100,   1200,   1300]
  /// For scale == -2, the rep values are: [0, 10000, 20000, 30000, ... 110000, 120000, 130000]
  /// For scale ==  2, the rep values are: [0, 1,     2,     3,     ... 11,     12,     13]
  [[nodiscard]] column_ptr generate_order_by_column(numeric::scale_type scale) const
  {
    auto const begin = thrust::make_transform_iterator(
      thrust::make_counting_iterator<Rep>(0),
      [&](auto i) -> Rep { return (i * 10000) / base::pow10[scale + 2]; });

    return decimals_column<Rep>{begin, begin + num_rows, numeric::scale_type{scale}}.release();
  }

  /**
   * @brief Scale the range bounds value to new scale, so that effective
   * value remains identical.
   *
   * Keeping the effective range bounds value identical ensures that
   * the expected result from grouped_rolling remains the same.
   */
  [[nodiscard]] Rep rescale_range_value(Rep const& value_at_scale_0,
                                        numeric::scale_type new_scale) const
  {
    // Scale  ->   Rep (for value == 200)
    //  -2    ->       20000
    //  -1    ->       2000
    //   0    ->       200
    //   1    ->       20
    //   2    ->       2
    return (value_at_scale_0 * 100) / base::pow10[new_scale + 2];
  }

  /**
   * @brief Get grouped rolling results for specified order-by column and range scale
   *
   */
  column_ptr get_grouped_range_rolling_result(cudf::column_view const& order_by_column,
                                              numeric::scale_type const& range_scale) const
  {
    auto const preceding =
      this->make_fixed_point_range_bounds(rescale_range_value(Rep{200}, range_scale), range_scale);
    auto const following =
      this->make_fixed_point_range_bounds(rescale_range_value(Rep{100}, range_scale), range_scale);

    return cudf::grouped_range_rolling_window(
      cudf::table_view{{grouping_keys->view()}},
      order_by_column,
      cudf::order::ASCENDING,
      agg_values->view(),
      preceding,
      following,
      1,  // min_periods
      *cudf::make_sum_aggregation<cudf::rolling_aggregation>());
  }

  /**
   * @brief Run grouped_rolling test for specified order-by column scale with
   * no nulls in the order-by column
   *
   */
  void run_test_no_null_oby(numeric::scale_type const& order_by_column_scale) const
  {
    auto const order_by = generate_order_by_column(order_by_column_scale);
    // Run tests for range bounds generated for all scales >= oby_column_scale.
    for (int32_t range_scale = order_by_column_scale; range_scale <= 2; ++range_scale) {
      auto const results =
        get_grouped_range_rolling_result(*order_by, numeric::scale_type{range_scale});
      auto const expected_results = bigints_column{{2, 3, 4, 4, 4, 3, 4, 6, 8, 6, 6, 9, 12, 9},
                                                   cudf::test::iterators::no_nulls()};
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_results);
    }
  }

  /**
   * @brief Run grouped_rolling test for specified order-by column scale with
   * nulls in the order-by column (i.e. 2 nulls at the beginning of each group)
   *
   */
  void run_test_nulls_in_oby(numeric::scale_type const& order_by_column_scale) const
  {
    // Nullify the first two rows of each group in the order_by column.
    auto const nulled_order_by = [&] {
      auto col           = generate_order_by_column(order_by_column_scale);
      auto new_null_mask = create_null_mask(col->size(), cudf::mask_state::ALL_VALID);
      cudf::set_null_mask(static_cast<cudf::bitmask_type*>(new_null_mask.data()),
                          0,
                          2,
                          false);  // Nulls in first group.
      cudf::set_null_mask(static_cast<cudf::bitmask_type*>(new_null_mask.data()),
                          6,
                          8,
                          false);  // Nulls in second group.
      cudf::set_null_mask(static_cast<cudf::bitmask_type*>(new_null_mask.data()),
                          10,
                          12,
                          false);  // Nulls in third group.
      col->set_null_mask(std::move(new_null_mask), 6);
      return col;
    }();

    // Run tests for range bounds generated for all scales >= oby_column_scale.
    for (auto range_scale = int32_t{order_by_column_scale}; range_scale <= 2; ++range_scale) {
      auto const results =
        get_grouped_range_rolling_result(*nulled_order_by, numeric::scale_type{range_scale});
      auto const expected_results = bigints_column{{2, 2, 2, 3, 4, 3, 4, 4, 4, 4, 6, 6, 6, 6},
                                                   cudf::test::iterators::no_nulls()};
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_results);
    }
  }

  /**
   * @brief Run grouped_rolling test for specified order-by column scale with
   * unbounded preceding and unbounded following.
   *
   */
  void run_test_unbounded_preceding_to_unbounded_following(numeric::scale_type oby_column_scale)
  {
    auto const order_by  = generate_order_by_column(oby_column_scale);
    auto const preceding = make_unbounded_fixed_point_range_bounds();
    auto const following = make_unbounded_fixed_point_range_bounds();
    auto results =
      cudf::grouped_range_rolling_window(cudf::table_view{{grouping_keys->view()}},
                                         order_by->view(),
                                         cudf::order::ASCENDING,
                                         agg_values->view(),
                                         preceding,
                                         following,
                                         1,  // min_periods
                                         *cudf::make_sum_aggregation<cudf::rolling_aggregation>());

    auto expected_results = bigints_column{{6, 6, 6, 6, 6, 6, 8, 8, 8, 8, 12, 12, 12, 12},
                                           cudf::test::iterators::no_nulls()};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_results);
  }

  /**
   * @brief Run grouped_rolling test for specified order-by column scale with
   * unbounded preceding and unbounded following.
   *
   */
  void run_test_unbounded_preceding_to_current_row(numeric::scale_type oby_column_scale)
  {
    auto const order_by            = generate_order_by_column(oby_column_scale);
    auto const unbounded_preceding = make_unbounded_fixed_point_range_bounds();

    for (int32_t range_scale = oby_column_scale; range_scale <= 2; ++range_scale) {
      auto const current_row =
        make_fixed_point_range_bounds(rescale_range_value(Rep{0}, numeric::scale_type{range_scale}),
                                      numeric::scale_type{range_scale});
      auto const results = cudf::grouped_range_rolling_window(
        cudf::table_view{{grouping_keys->view()}},
        order_by->view(),
        cudf::order::ASCENDING,
        agg_values->view(),
        unbounded_preceding,
        current_row,
        1,  // min_periods
        *cudf::make_sum_aggregation<cudf::rolling_aggregation>());

      auto expected_results = bigints_column{{1, 2, 3, 4, 5, 6, 2, 4, 6, 8, 3, 6, 9, 12},
                                             cudf::test::iterators::no_nulls()};
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_results);
    }
  }

  /**
   * @brief Run grouped_rolling test for specified order-by column scale with
   * unbounded preceding and unbounded following.
   *
   */
  void run_test_current_row_to_unbounded_following(numeric::scale_type oby_column_scale)
  {
    auto const order_by            = generate_order_by_column(oby_column_scale);
    auto const unbounded_following = make_unbounded_fixed_point_range_bounds();

    for (int32_t range_scale = oby_column_scale; range_scale <= 2; ++range_scale) {
      auto const current_row =
        make_fixed_point_range_bounds(rescale_range_value(Rep{0}, numeric::scale_type{range_scale}),
                                      numeric::scale_type{range_scale});
      auto const results = cudf::grouped_range_rolling_window(
        cudf::table_view{{grouping_keys->view()}},
        order_by->view(),
        cudf::order::ASCENDING,
        agg_values->view(),
        current_row,
        unbounded_following,
        1,  // min_periods
        *cudf::make_sum_aggregation<cudf::rolling_aggregation>());

      auto expected_results = bigints_column{{6, 5, 4, 3, 2, 1, 8, 6, 4, 2, 12, 9, 6, 3},
                                             cudf::test::iterators::no_nulls()};
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_results);
    }
  }
};

TYPED_TEST_SUITE(GroupedRollingRangeOrderByDecimalTypedTest, cudf::test::FixedPointTypes);

TYPED_TEST(GroupedRollingRangeOrderByDecimalTypedTest, BoundedRanges)
{
  for (auto const order_by_column_scale : {-2, -1, 0, 1, 2}) {
    auto const oby_scale = numeric::scale_type{order_by_column_scale};
    this->run_test_no_null_oby(oby_scale);
    this->run_test_nulls_in_oby(oby_scale);
  }
}

TYPED_TEST(GroupedRollingRangeOrderByDecimalTypedTest, UnboundedRanges)
{
  for (auto const order_by_scale : {-2, -1, 0, 1, 2}) {
    auto const order_by_column_scale = numeric::scale_type{order_by_scale};
    this->run_test_unbounded_preceding_to_unbounded_following(order_by_column_scale);
    this->run_test_unbounded_preceding_to_current_row(order_by_column_scale);
    this->run_test_current_row_to_unbounded_following(order_by_column_scale);
  }
}

struct GroupedRollingRangeOrderByStringTest : public cudf::test::BaseFixture {
  // Test data.
  column_ptr const grouping_keys = ints_column{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2}.release();
  column_ptr const agg_values    = ints_column{1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3}.release();
  cudf::size_type const num_rows = grouping_keys->size();
  cudf::size_type const min_periods = 1;

  cudf::data_type const string_type{cudf::type_id::STRING};
  cudf::range_window_bounds const unbounded_preceding =
    cudf::range_window_bounds::unbounded(string_type);
  cudf::range_window_bounds const unbounded_following =
    cudf::range_window_bounds::unbounded(string_type);
  cudf::range_window_bounds const current_row = cudf::range_window_bounds::current_row(string_type);

  [[nodiscard]] static auto nullable_ints_column(std::initializer_list<int> const& ints)
  {
    return ints_column{ints, cudf::test::iterators::no_nulls()};
  }

  [[nodiscard]] auto get_count_over_partitioned_window(
    cudf::column_view const& order_by,
    cudf::order const& order,
    cudf::range_window_bounds const& preceding,
    cudf::range_window_bounds const& following) const
  {
    return cudf::grouped_range_rolling_window(
      cudf::table_view{{*grouping_keys}},
      order_by,
      order,
      *agg_values,
      preceding,
      following,
      min_periods,
      *cudf::make_count_aggregation<cudf::rolling_aggregation>());
  }

  [[nodiscard]] auto get_count_over_unpartitioned_window(
    cudf::column_view const& order_by,
    cudf::order const& order,
    cudf::range_window_bounds const& preceding,
    cudf::range_window_bounds const& following) const
  {
    return cudf::grouped_range_rolling_window(
      cudf::table_view{std::vector<cudf::column_view>{}},
      order_by,
      order,
      *agg_values,
      preceding,
      following,
      min_periods,
      *cudf::make_count_aggregation<cudf::rolling_aggregation>());
  }
};

TEST_F(GroupedRollingRangeOrderByStringTest, Ascending_Partitioned_NoNulls)
{
  // clang-format off
  auto const orderby =
      strings_column{
          "A", "A", "A", "B", "B", "B", // Group 0.
          "C", "C", "C", "C",           // Group 1.
          "D", "D", "E", "E"            // Group 2.
      }.release();
  // clang-format on

  // Partitioned cases.
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *get_count_over_partitioned_window(
      *orderby, cudf::order::ASCENDING, unbounded_preceding, current_row),
    nullable_ints_column({3, 3, 3, 6, 6, 6, 4, 4, 4, 4, 2, 2, 4, 4}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *get_count_over_partitioned_window(
      *orderby, cudf::order::ASCENDING, current_row, unbounded_following),
    nullable_ints_column({6, 6, 6, 3, 3, 3, 4, 4, 4, 4, 4, 4, 2, 2}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *get_count_over_partitioned_window(
      *orderby, cudf::order::ASCENDING, unbounded_preceding, unbounded_following),
    nullable_ints_column({6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *get_count_over_partitioned_window(*orderby, cudf::order::ASCENDING, current_row, current_row),
    nullable_ints_column({3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 2, 2, 2, 2}));
}

TEST_F(GroupedRollingRangeOrderByStringTest, Ascending_NoParts_NoNulls)
{
  // clang-format off
  auto const orderby =
      strings_column{
          "A", "A", "A", "B", "B", "B",
          "C", "C", "C", "C",
          "D", "D", "E", "E"
      }.release();
  // clang-format on

  // Un-partitioned cases.
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *get_count_over_unpartitioned_window(
      *orderby, cudf::order::ASCENDING, unbounded_preceding, current_row),
    nullable_ints_column({3, 3, 3, 6, 6, 6, 10, 10, 10, 10, 12, 12, 14, 14}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *get_count_over_unpartitioned_window(
      *orderby, cudf::order::ASCENDING, current_row, unbounded_following),
    nullable_ints_column({14, 14, 14, 11, 11, 11, 8, 8, 8, 8, 4, 4, 2, 2}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *get_count_over_unpartitioned_window(
      *orderby, cudf::order::ASCENDING, unbounded_preceding, unbounded_following),
    nullable_ints_column({14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*get_count_over_unpartitioned_window(
                                   *orderby, cudf::order::ASCENDING, current_row, current_row),
                                 nullable_ints_column({3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 2, 2, 2, 2}));
}

TEST_F(GroupedRollingRangeOrderByStringTest, Ascending_Partitioned_WithNulls)
{
  // clang-format off
  auto const orderby =
      strings_column{
          {
              "X", "X", "X", "B", "B", "B", // Group 0.
              "C", "C", "C", "C",           // Group 1.
              "X", "X", "E", "E"            // Group 2.
          },
          cudf::test::iterators::nulls_at({0, 1, 2, 10, 11})
      }.release();
  // clang-format on

  // Partitioned cases.
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *get_count_over_partitioned_window(
      *orderby, cudf::order::ASCENDING, unbounded_preceding, current_row),
    nullable_ints_column({3, 3, 3, 6, 6, 6, 4, 4, 4, 4, 2, 2, 4, 4}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *get_count_over_partitioned_window(
      *orderby, cudf::order::ASCENDING, current_row, unbounded_following),
    nullable_ints_column({6, 6, 6, 3, 3, 3, 4, 4, 4, 4, 4, 4, 2, 2}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *get_count_over_partitioned_window(
      *orderby, cudf::order::ASCENDING, unbounded_preceding, unbounded_following),
    nullable_ints_column({6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *get_count_over_partitioned_window(*orderby, cudf::order::ASCENDING, current_row, current_row),
    nullable_ints_column({3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 2, 2, 2, 2}));
}

TEST_F(GroupedRollingRangeOrderByStringTest, Ascending_NoParts_WithNulls)
{
  // Un-partitioned cases. Null values have to be clustered together.
  // clang-format off
  auto const orderby =
      strings_column{
          {
              "X", "X", "X", "B", "B", "B",
              "C", "C", "C", "C",
              "D", "D", "E", "E"
          },
          cudf::test::iterators::nulls_at({0, 1, 2})
      }.release();
  // clang-format on

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *get_count_over_unpartitioned_window(
      *orderby, cudf::order::ASCENDING, unbounded_preceding, current_row),
    nullable_ints_column({3, 3, 3, 6, 6, 6, 10, 10, 10, 10, 12, 12, 14, 14}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *get_count_over_unpartitioned_window(
      *orderby, cudf::order::ASCENDING, current_row, unbounded_following),
    nullable_ints_column({14, 14, 14, 11, 11, 11, 8, 8, 8, 8, 4, 4, 2, 2}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *get_count_over_unpartitioned_window(
      *orderby, cudf::order::ASCENDING, unbounded_preceding, unbounded_following),
    nullable_ints_column({14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*get_count_over_unpartitioned_window(
                                   *orderby, cudf::order::ASCENDING, current_row, current_row),
                                 nullable_ints_column({3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 2, 2, 2, 2}));
}

TEST_F(GroupedRollingRangeOrderByStringTest, Descending_Partitioned_NoNulls)
{
  // clang-format off
  auto const orderby =
      strings_column{
          "B", "B", "B", "A", "A", "A", // Group 0.
          "C", "C", "C", "C",           // Group 1.
          "E", "E", "D", "D"            // Group 2.
      }.release();
  // clang-format on

  // Partitioned cases.
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *get_count_over_partitioned_window(
      *orderby, cudf::order::DESCENDING, unbounded_preceding, current_row),
    nullable_ints_column({3, 3, 3, 6, 6, 6, 4, 4, 4, 4, 2, 2, 4, 4}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *get_count_over_partitioned_window(
      *orderby, cudf::order::DESCENDING, current_row, unbounded_following),
    nullable_ints_column({6, 6, 6, 3, 3, 3, 4, 4, 4, 4, 4, 4, 2, 2}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *get_count_over_partitioned_window(
      *orderby, cudf::order::DESCENDING, unbounded_preceding, unbounded_following),
    nullable_ints_column({6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *get_count_over_partitioned_window(*orderby, cudf::order::DESCENDING, current_row, current_row),
    nullable_ints_column({3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 2, 2, 2, 2}));
}

TEST_F(GroupedRollingRangeOrderByStringTest, Descending_NoParts_NoNulls)
{
  // Un-partitioned cases. Order-by column must be entirely in descending order.
  // clang-format off
  auto const orderby =
      strings_column{
          "E", "E", "E", "D", "D", "D",
          "C", "C", "C", "C",
          "B", "B", "A", "A"
      }.release();
  // clang-format on

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *get_count_over_unpartitioned_window(
      *orderby, cudf::order::DESCENDING, unbounded_preceding, current_row),
    nullable_ints_column({3, 3, 3, 6, 6, 6, 10, 10, 10, 10, 12, 12, 14, 14}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *get_count_over_unpartitioned_window(
      *orderby, cudf::order::DESCENDING, current_row, unbounded_following),
    nullable_ints_column({14, 14, 14, 11, 11, 11, 8, 8, 8, 8, 4, 4, 2, 2}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *get_count_over_unpartitioned_window(
      *orderby, cudf::order::DESCENDING, unbounded_preceding, unbounded_following),
    nullable_ints_column({14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*get_count_over_unpartitioned_window(
                                   *orderby, cudf::order::DESCENDING, current_row, current_row),
                                 nullable_ints_column({3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 2, 2, 2, 2}));
}

TEST_F(GroupedRollingRangeOrderByStringTest, Descending_Partitioned_WithNulls)
{
  // clang-format off
  auto const orderby =
      strings_column{
          {
              "X", "X", "X", "A", "A", "A", // Group 0.
              "C", "C", "C", "C",           // Group 1.
              "X", "X", "D", "D"            // Group 2.
          },
          cudf::test::iterators::nulls_at({0, 1, 2, 10, 11})
      }.release();
  // clang-format on

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *get_count_over_partitioned_window(
      *orderby, cudf::order::DESCENDING, unbounded_preceding, current_row),
    nullable_ints_column({3, 3, 3, 6, 6, 6, 4, 4, 4, 4, 2, 2, 4, 4}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *get_count_over_partitioned_window(
      *orderby, cudf::order::DESCENDING, current_row, unbounded_following),
    nullable_ints_column({6, 6, 6, 3, 3, 3, 4, 4, 4, 4, 4, 4, 2, 2}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *get_count_over_partitioned_window(
      *orderby, cudf::order::DESCENDING, unbounded_preceding, unbounded_following),
    nullable_ints_column({6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *get_count_over_partitioned_window(*orderby, cudf::order::DESCENDING, current_row, current_row),
    nullable_ints_column({3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 2, 2, 2, 2}));
}

TEST_F(GroupedRollingRangeOrderByStringTest, Descending_NoParts_WithNulls)
{
  // Order-by column must be ordered completely in descending.
  // clang-format off
  auto const orderby =
      strings_column{
          {
              "X", "X", "X", "D", "D", "D",
              "C", "C", "C", "C",
              "B", "B", "A", "A"
          },
          cudf::test::iterators::nulls_at({0, 1, 2})
      }.release();
  // clang-format on

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *get_count_over_unpartitioned_window(
      *orderby, cudf::order::DESCENDING, unbounded_preceding, current_row),
    nullable_ints_column({3, 3, 3, 6, 6, 6, 10, 10, 10, 10, 12, 12, 14, 14}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *get_count_over_unpartitioned_window(
      *orderby, cudf::order::DESCENDING, current_row, unbounded_following),
    nullable_ints_column({14, 14, 14, 11, 11, 11, 8, 8, 8, 8, 4, 4, 2, 2}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *get_count_over_unpartitioned_window(
      *orderby, cudf::order::DESCENDING, unbounded_preceding, unbounded_following),
    nullable_ints_column({14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*get_count_over_unpartitioned_window(
                                   *orderby, cudf::order::DESCENDING, current_row, current_row),
                                 nullable_ints_column({3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 2, 2, 2, 2}));
}
