/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
#include <src/rolling/detail/rolling.hpp>

#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include <algorithm>
#include <vector>

namespace cudf::test::rolling {

template <typename T>
using fwcw = cudf::test::fixed_width_column_wrapper<T>;
template <typename T>
using decimals   = cudf::test::fixed_point_column_wrapper<T>;
using ints       = fwcw<int32_t>;
using bigints    = fwcw<int64_t>;
using column_ptr = std::unique_ptr<cudf::column>;
using namespace numeric;
using namespace cudf::test::iterators;

struct BaseGroupedRollingRangeOrderByDecimalTest : public BaseFixture {
  // Stand-in for std::pow(10, n), but for integral return.
  static constexpr std::array<int32_t, 6> pow10{1, 10, 100, 1000, 10000, 100000};
  // Test data.
  column_ptr const grouping_keys = ints{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2}.release();
  column_ptr const agg_values    = ints{1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3}.release();
  cudf::size_type const num_rows = grouping_keys->size();
};

using base = BaseGroupedRollingRangeOrderByDecimalTest;  // Shortcut to base test class.

template <typename DecimalT>
struct GroupedRollingRangeOrderByDecimalTypedTest : BaseGroupedRollingRangeOrderByDecimalTest {

  using Rep = typename DecimalT::rep;

  auto make_fixed_point_range_bounds(typename DecimalT::rep value, scale_type scale) const
  {
    return cudf::range_window_bounds::get(*cudf::make_fixed_point_scalar<DecimalT>(value, scale));
  }

  auto make_unbounded_decimal_range_bounds() const
  {
    return cudf::range_window_bounds::unbounded(data_type{type_to_id<DecimalT>()});
  }

  /// For different scales, generate order_by column with
  /// the same effective values:           [0, 100,   200,   300,   ... 1100,   1200,   1300]
  /// For scale == -2, the rep values are: [0, 10000, 20000, 30000, ... 110000, 120000, 130000]
  /// For scale ==  2, the rep values are: [0, 1,     2,     3,     ... 11,     12,     13]
  column_ptr generate_order_by_column(scale_type scale) const
  {
    auto const begin = thrust::make_transform_iterator(
      thrust::make_counting_iterator<Rep>(0),
      [&](auto i) -> Rep { return (i * 10000) / base::pow10[scale + 2]; });

    return decimals<Rep>{begin, begin + num_rows, scale_type{scale}}.release();
  }

  void run_test_no_null_oby(column_view const& order_by,
                            range_window_bounds preceding,
                            range_window_bounds following) const
  {
    auto const results =
      cudf::grouped_range_rolling_window(cudf::table_view{{grouping_keys->view()}},
                                         order_by,
                                         cudf::order::ASCENDING,
                                         agg_values->view(),
                                         preceding,
                                         following,
                                         1,  // min_periods
                                         *cudf::make_sum_aggregation<rolling_aggregation>());
    auto const expected_results = bigints{{2, 3, 4, 4, 4, 3, 4, 6, 8, 6, 6, 9, 12, 9}, no_nulls()};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_results);
  }

  void run_test_nulls_in_oby(column_view const& order_by,
                             range_window_bounds preceding,
                             range_window_bounds following) const
  {
    // Nullify the first two rows of each group in the order_by column.
    auto const nulled_order_by = [&] {
      auto col           = cudf::column{order_by};
      auto new_null_mask = create_null_mask(col.size(), mask_state::ALL_VALID);
      set_null_mask(
        static_cast<bitmask_type*>(new_null_mask.data()), 0, 2, false);  // Nulls in first group.
      set_null_mask(
        static_cast<bitmask_type*>(new_null_mask.data()), 6, 8, false);  // Nulls in second group.
      set_null_mask(
        static_cast<bitmask_type*>(new_null_mask.data()), 10, 12, false);  // Nulls in third group.
      col.set_null_mask(std::move(new_null_mask));
      return col;
    }();

    auto const results =
      cudf::grouped_range_rolling_window(cudf::table_view{{grouping_keys->view()}},
                                         nulled_order_by.view(),
                                         cudf::order::ASCENDING,
                                         agg_values->view(),
                                         preceding,
                                         following,
                                         1,  // min_periods
                                         *cudf::make_sum_aggregation<rolling_aggregation>());
    auto const expected_results = bigints{{2, 2, 2, 3, 4, 3, 4, 4, 4, 4, 6, 6, 6, 6}, no_nulls()};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_results);
  }

  void run_test_unbounded_preceding_to_unbounded_following(scale_type oby_column_scale)
  {
    auto const order_by = generate_order_by_column(oby_column_scale);
    auto const preceding = make_unbounded_decimal_range_bounds();
    auto const following = make_unbounded_decimal_range_bounds();
    auto results = cudf::grouped_range_rolling_window(
      cudf::table_view{{grouping_keys->view()}},
      order_by->view(),
      cudf::order::ASCENDING,
      agg_values->view(),
      preceding,
      following,
      1,  // min_periods
      *cudf::make_sum_aggregation<rolling_aggregation>());
    
    auto expected_results = bigints{{6, 6, 6, 6, 6, 6, 8, 8, 8, 8, 12, 12, 12, 12}, no_nulls()};
  }
};

using RepresentationTypes =
  ::testing::Types<numeric::decimal32, numeric::decimal64, numeric::decimal128>;

TYPED_TEST_SUITE(GroupedRollingRangeOrderByDecimalTypedTest, RepresentationTypes);

TYPED_TEST(GroupedRollingRangeOrderByDecimalTypedTest, BoundedRanges)
{
  using DecimalT = TypeParam;               // Decimal type for order_by column.
  using Rep      = typename DecimalT::rep;  // Representation type for order_by column.

  // For different scales, generate order_by column with
  // the same effective values:           [0, 100,   200,   300,   ... 1100,   1200,   1300]
  // For scale == -2, the rep values are: [0, 10000, 20000, 30000, ... 110000, 120000, 130000]
  // For scale ==  2, the rep values are: [0, 1,     2,     3,     ... 11,     12,     13]
  for (auto oby_column_scale : {-2, -1, 0, 1, 2}) {
    auto const order_by = [num_rows = this->num_rows, oby_column_scale] {
      auto const begin = thrust::make_transform_iterator(
        thrust::make_counting_iterator<Rep>(0),
        [&](auto i) -> Rep { return (i * 10000) / base::pow10[oby_column_scale + 2]; });
      return decimals<Rep>{begin, begin + num_rows, scale_type{oby_column_scale}}.release();
    }();

    // Run tests for range bounds generated for all scales >= oby_column_scale.
    for (auto range_scale = oby_column_scale; range_scale <= 2; ++range_scale) {
      // Scale the range bounds value, depending on the scale,
      // so that the effective range bounds value is the same.
      // This keeps the expected results the same for all scales.
      auto rescale_range_value = [&](auto value, auto scale) {
        // Scale  ->   Rep (for value == 200)
        //  -2    ->       20000
        //  -1    ->       2000
        //   0    ->       200
        //   1    ->       20
        //   2    ->       2
        return (value * 100) / base::pow10[scale + 2];
      };
      auto const preceding = this->make_fixed_point_range_bounds(
        rescale_range_value(Rep{200}, range_scale), scale_type{range_scale});
      auto const following = this->make_fixed_point_range_bounds(
        rescale_range_value(Rep{100}, range_scale), scale_type{range_scale});

      this->run_test_no_null_oby(order_by->view(), preceding, following);
      this->run_test_nulls_in_oby(order_by->view(), preceding, following);
    }
  }
}

TYPED_TEST(GroupedRollingRangeOrderByDecimalTypedTest, Unbounded)
{
  for (auto oby_column_scale : {-2, -1, 0, 1, 2}) {
    std::cout << "Running for column scale: " << oby_column_scale << std::endl;
    this->run_test_unbounded_preceding_to_unbounded_following(scale_type{oby_column_scale});
  }
}

}  // namespace cudf::test::rolling
