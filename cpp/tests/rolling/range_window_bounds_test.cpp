/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include <chrono>
#include <cuda/std/ratio>
#include <vector>

#include <cudf/rolling/range_window_bounds.hpp>
#include <src/rolling/detail/range_window_bounds.hpp>

namespace cudf {
namespace test {

struct RangeWindowBoundsTest : public BaseFixture {
};

template <typename Timestamp>
struct TimestampRangeWindowBoundsTest : RangeWindowBoundsTest {
};

TYPED_TEST_SUITE(TimestampRangeWindowBoundsTest, cudf::test::TimestampTypes);

TEST_F(RangeWindowBoundsTest, TestBasicTimestampRangeTypeMapping)
{
  // Explicitly check that the programmatic mapping of orderby column types
  // to their respective range and range_rep types is accurate.

  using namespace cudf::detail;

  static_assert(std::is_same_v<range_type<timestamp_D>, duration_D>);
  static_assert(std::is_same_v<range_type<timestamp_s>, duration_s>);
  static_assert(std::is_same_v<range_type<timestamp_ms>, duration_ms>);
  static_assert(std::is_same_v<range_type<timestamp_us>, duration_us>);
  static_assert(std::is_same_v<range_type<timestamp_ns>, duration_ns>);

  static_assert(std::is_same_v<range_rep_type<timestamp_D>, int32_t>);
  static_assert(std::is_same_v<range_rep_type<timestamp_s>, int64_t>);
  static_assert(std::is_same_v<range_rep_type<timestamp_ms>, int64_t>);
  static_assert(std::is_same_v<range_rep_type<timestamp_us>, int64_t>);
  static_assert(std::is_same_v<range_rep_type<timestamp_ns>, int64_t>);
}

TYPED_TEST(TimestampRangeWindowBoundsTest, BoundsConstruction)
{
  using OrderByType = TypeParam;
  using range_type  = cudf::detail::range_type<OrderByType>;
  using rep_type    = cudf::detail::range_rep_type<OrderByType>;

  using range_window_bounds = cudf::range_window_bounds;

  static_assert(cudf::is_duration<range_type>());
  auto range_3 = range_window_bounds::get(duration_scalar<range_type>{3, true});
  EXPECT_FALSE(range_3.is_unbounded() &&
               "range_window_bounds constructed from scalar cannot be unbounded.");
  EXPECT_EQ(cudf::detail::range_comparable_value<OrderByType>(range_3), rep_type{3});

  auto range_unbounded = range_window_bounds::unbounded(data_type{type_to_id<range_type>()});
  EXPECT_TRUE(range_unbounded.is_unbounded() &&
              "range_window_bounds::unbounded() must return an unbounded range.");
  EXPECT_EQ(cudf::detail::range_comparable_value<OrderByType>(range_unbounded), rep_type{});
}

TYPED_TEST(TimestampRangeWindowBoundsTest, WrongRangeType)
{
  using OrderByType = TypeParam;

  using wrong_range_type =
    std::conditional_t<std::is_same_v<OrderByType, timestamp_D>, duration_ns, duration_D>;
  auto range_3 = cudf::range_window_bounds::get(duration_scalar<wrong_range_type>{3, true});

  EXPECT_THROW(cudf::detail::range_comparable_value<OrderByType>(range_3), cudf::logic_error);

  auto range_unbounded = range_window_bounds::unbounded(data_type{type_to_id<wrong_range_type>()});
  EXPECT_THROW(cudf::detail::range_comparable_value<OrderByType>(range_unbounded),
               cudf::logic_error);
}

template <typename T>
struct NumericRangeWindowBoundsTest : RangeWindowBoundsTest {
};

using TypesForTest = cudf::test::IntegralTypesNotBool;

TYPED_TEST_SUITE(NumericRangeWindowBoundsTest, TypesForTest);

TYPED_TEST(NumericRangeWindowBoundsTest, BasicNumericRangeTypeMapping)
{
  using T = TypeParam;

  using range_type     = cudf::detail::range_type<T>;
  using range_rep_type = cudf::detail::range_rep_type<T>;

  static_assert(std::is_same_v<T, range_type>);
  static_assert(std::is_same_v<T, range_rep_type>);
}

TYPED_TEST(NumericRangeWindowBoundsTest, BoundsConstruction)
{
  using OrderByType = TypeParam;
  using range_type  = cudf::detail::range_type<OrderByType>;
  using rep_type    = cudf::detail::range_rep_type<OrderByType>;

  using range_window_bounds = cudf::range_window_bounds;

  static_assert(std::is_integral_v<range_type>);
  auto range_3 = range_window_bounds::get(numeric_scalar<range_type>{3, true});
  EXPECT_FALSE(range_3.is_unbounded() &&
               "range_window_bounds constructed from scalar cannot be unbounded.");
  EXPECT_EQ(cudf::detail::range_comparable_value<OrderByType>(range_3), rep_type{3});

  auto range_unbounded = range_window_bounds::unbounded(data_type{type_to_id<range_type>()});
  EXPECT_TRUE(range_unbounded.is_unbounded() &&
              "range_window_bounds::unbounded() must return an unbounded range.");
  EXPECT_EQ(cudf::detail::range_comparable_value<OrderByType>(range_unbounded), rep_type{});
}

TYPED_TEST(NumericRangeWindowBoundsTest, WrongRangeType)
{
  using OrderByType = TypeParam;

  using wrong_range_type =
    std::conditional_t<std::is_same_v<OrderByType, int32_t>, int16_t, int32_t>;
  auto range_3 = cudf::range_window_bounds::get(numeric_scalar<wrong_range_type>{3, true});

  EXPECT_THROW(cudf::detail::range_comparable_value<OrderByType>(range_3), cudf::logic_error);

  auto range_unbounded = range_window_bounds::unbounded(data_type{type_to_id<wrong_range_type>()});
  EXPECT_THROW(cudf::detail::range_comparable_value<OrderByType>(range_unbounded),
               cudf::logic_error);
}

}  // namespace test
}  // namespace cudf
