/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/type_lists.hpp>

#include <src/rolling/detail/range_comparators.cuh>

struct RangeComparatorTest : cudf::test::BaseFixture {};

template <typename T>
struct RangeComparatorTypedTest : RangeComparatorTest {};

using TestTypes =
  cudf::test::Concat<cudf::test::IntegralTypesNotBool, cudf::test::FloatingPointTypes>;

TYPED_TEST_SUITE(RangeComparatorTypedTest, TestTypes);

TYPED_TEST(RangeComparatorTypedTest, TestLessComparator)
{
  auto const less     = cudf::detail::nan_aware_less{};
  auto constexpr nine = TypeParam{9};
  auto constexpr ten  = TypeParam{10};

  EXPECT_TRUE(less(nine, ten));
  EXPECT_FALSE(less(ten, nine));
  EXPECT_FALSE(less(ten, ten));

  if constexpr (std::is_floating_point_v<TypeParam>) {
    auto constexpr NaN = std::numeric_limits<TypeParam>::quiet_NaN();
    auto constexpr Inf = std::numeric_limits<TypeParam>::infinity();
    // NaN.
    EXPECT_FALSE(less(NaN, ten));
    EXPECT_FALSE(less(NaN, NaN));
    EXPECT_FALSE(less(NaN, Inf));
    EXPECT_FALSE(less(NaN, -Inf));
    // Infinity.
    EXPECT_TRUE(less(Inf, NaN));
    EXPECT_FALSE(less(Inf, Inf));
    EXPECT_FALSE(less(Inf, ten));
    EXPECT_FALSE(less(Inf, -Inf));
    // -Infinity.
    EXPECT_TRUE(less(-Inf, NaN));
    EXPECT_TRUE(less(-Inf, Inf));
    EXPECT_TRUE(less(-Inf, ten));
    EXPECT_FALSE(less(-Inf, -Inf));
    // Finite.
    EXPECT_TRUE(less(ten, NaN));
    EXPECT_TRUE(less(ten, Inf));
    EXPECT_FALSE(less(ten, -Inf));
  }
}

TYPED_TEST(RangeComparatorTypedTest, TestGreaterComparator)
{
  auto const greater  = cudf::detail::nan_aware_greater{};
  auto constexpr nine = TypeParam{9};
  auto constexpr ten  = TypeParam{10};

  EXPECT_FALSE(greater(nine, ten));
  EXPECT_TRUE(greater(ten, nine));
  EXPECT_FALSE(greater(ten, ten));

  if constexpr (std::is_floating_point_v<TypeParam>) {
    auto constexpr NaN = std::numeric_limits<TypeParam>::quiet_NaN();
    auto constexpr Inf = std::numeric_limits<TypeParam>::infinity();
    // NaN.
    EXPECT_TRUE(greater(NaN, ten));
    EXPECT_FALSE(greater(NaN, NaN));
    EXPECT_TRUE(greater(NaN, Inf));
    EXPECT_TRUE(greater(NaN, -Inf));
    // Infinity.
    EXPECT_FALSE(greater(Inf, NaN));
    EXPECT_FALSE(greater(Inf, Inf));
    EXPECT_TRUE(greater(Inf, ten));
    EXPECT_TRUE(greater(Inf, -Inf));
    // -Infinity.
    EXPECT_FALSE(greater(-Inf, NaN));
    EXPECT_FALSE(greater(-Inf, Inf));
    EXPECT_FALSE(greater(-Inf, ten));
    EXPECT_FALSE(greater(-Inf, -Inf));
    // Finite.
    EXPECT_FALSE(greater(ten, NaN));
    EXPECT_FALSE(greater(ten, Inf));
    EXPECT_TRUE(greater(ten, -Inf));
  }
}
