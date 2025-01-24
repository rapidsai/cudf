/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/random.hpp>
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/rolling.hpp>
#include <cudf/scalar/scalar_factories.hpp>

#include <limits>

struct Base : cudf::test::BaseFixture {};

template <typename T>
struct UngroupedIntegralRangeWindows : cudf::test::BaseFixture {};
TYPED_TEST_SUITE(UngroupedIntegralRangeWindows, cudf::test::IntegralTypesNotBool);

template <typename T>
struct UngroupedSignedIntegralRangeWindows : cudf::test::BaseFixture {};
using SignedIntegralTypes = cudf::test::Types<int8_t, int16_t, int32_t, int64_t>;
TYPED_TEST_SUITE(UngroupedSignedIntegralRangeWindows, SignedIntegralTypes);

template <typename T>
struct GroupedIntegralRangeWindows : cudf::test::BaseFixture {};
TYPED_TEST_SUITE(GroupedIntegralRangeWindows, cudf::test::IntegralTypesNotBool);

template <typename T>
struct GroupedSignedIntegralRangeWindows : cudf::test::BaseFixture {};
TYPED_TEST_SUITE(GroupedSignedIntegralRangeWindows, SignedIntegralTypes);

TYPED_TEST(UngroupedIntegralRangeWindows, AscendingNoNullsZeroPrecedingFollowing)
{
  using T         = TypeParam;
  constexpr T min = std::numeric_limits<T>::lowest();
  constexpr T max = std::numeric_limits<T>::max();
  auto orderby    = cudf::test::fixed_width_column_wrapper<T>{
    {min, T{5}, T{5}, T{6}, T{7}, T{9}, T{9}, T{12}, T{13}, T{17}, T{22}, T{22}, max}};

  auto prec = cudf::make_fixed_width_scalar<T>(0);
  auto foll = cudf::make_fixed_width_scalar<T>(0);

  auto result = cudf::make_range_windows(cudf::table_view{},
                                         orderby,
                                         cudf::order::ASCENDING,
                                         cudf::null_order::BEFORE,
                                         cudf::bounded_closed{*prec},
                                         cudf::bounded_closed{*foll});

  auto expect_preceding = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    {1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1}};
  auto expect_following = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    {0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<0>(result)->view(), expect_preceding, cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<1>(result)->view(), expect_following, cudf::test::debug_output_level::ALL_ERRORS);

  result = cudf::make_range_windows(cudf::table_view{},
                                    orderby,
                                    cudf::order::ASCENDING,
                                    cudf::null_order::BEFORE,
                                    cudf::bounded_open{*prec},
                                    cudf::bounded_open{*foll});

  expect_preceding = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    {0, -1, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0}};
  expect_following = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    {-1, -1, -2, -1, -1, -1, -2, -1, -1, -1, -1, -2, -1}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<0>(result)->view(), expect_preceding, cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<1>(result)->view(), expect_following, cudf::test::debug_output_level::ALL_ERRORS);
}

TYPED_TEST(UngroupedIntegralRangeWindows, AscendingNoNullsPositivePrecedingFollowing)
{
  using T         = TypeParam;
  constexpr T min = std::numeric_limits<T>::lowest();
  constexpr T max = std::numeric_limits<T>::max();
  auto orderby    = cudf::test::fixed_width_column_wrapper<T>{
    {min, T{5}, T{5}, T{6}, T{7}, T{9}, T{9}, T{12}, T{13}, T{17}, T{22}, T{22}, max}};

  auto prec = cudf::make_fixed_width_scalar<T>(2);
  auto foll = cudf::make_fixed_width_scalar<T>(1);

  auto result = cudf::make_range_windows(cudf::table_view{},
                                         orderby,
                                         cudf::order::ASCENDING,
                                         cudf::null_order::BEFORE,
                                         cudf::bounded_closed{*prec},
                                         cudf::bounded_closed{*foll});

  auto expect_preceding = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    {1, 1, 2, 3, 4, 2, 3, 1, 2, 1, 1, 2, 1}};
  auto expect_following = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    {0, 2, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<0>(result)->view(), expect_preceding, cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<1>(result)->view(), expect_following, cudf::test::debug_output_level::ALL_ERRORS);

  result = cudf::make_range_windows(cudf::table_view{},
                                    orderby,
                                    cudf::order::ASCENDING,
                                    cudf::null_order::BEFORE,
                                    cudf::bounded_open{*prec},
                                    cudf::bounded_open{*foll});

  expect_preceding = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    {1, 1, 2, 3, 2, 1, 2, 1, 2, 1, 1, 2, 1}};
  expect_following = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    {0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<0>(result)->view(), expect_preceding, cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<1>(result)->view(), expect_following, cudf::test::debug_output_level::ALL_ERRORS);
}

TYPED_TEST(UngroupedSignedIntegralRangeWindows, AscendingNoNullsNegativePreceding)
{
  using T         = TypeParam;
  constexpr T min = std::numeric_limits<T>::lowest();
  constexpr T max = std::numeric_limits<T>::max();
  auto orderby    = cudf::test::fixed_width_column_wrapper<T>{
    {min, T{5}, T{5}, T{6}, T{7}, T{9}, T{9}, T{12}, T{13}, T{17}, T{22}, T{22}, max}};

  auto prec = cudf::make_fixed_width_scalar<T>(-1);
  auto foll = cudf::make_fixed_width_scalar<T>(2);

  auto result = cudf::make_range_windows(cudf::table_view{},
                                         orderby,
                                         cudf::order::ASCENDING,
                                         cudf::null_order::BEFORE,
                                         cudf::bounded_closed{*prec},
                                         cudf::bounded_closed{*foll});

  auto expect_preceding = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    {0, -1, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0}};
  auto expect_following = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    {0, 3, 2, 1, 2, 1, 0, 1, 0, 0, 1, 0, 0}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<0>(result)->view(), expect_preceding, cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<1>(result)->view(), expect_following, cudf::test::debug_output_level::ALL_ERRORS);

  result = cudf::make_range_windows(cudf::table_view{},
                                    orderby,
                                    cudf::order::ASCENDING,
                                    cudf::null_order::BEFORE,
                                    cudf::bounded_open{*prec},
                                    cudf::bounded_open{*foll});

  expect_preceding = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    {0, -2, -1, -1, 0, -1, 0, -1, 0, 0, -1, 0, 0}};
  expect_following = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    {0, 2, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<0>(result)->view(), expect_preceding, cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<1>(result)->view(), expect_following, cudf::test::debug_output_level::ALL_ERRORS);
}

TYPED_TEST(UngroupedSignedIntegralRangeWindows, AscendingNoNullsNegativeFollowing)
{
  using T         = TypeParam;
  constexpr T min = std::numeric_limits<T>::lowest();
  constexpr T max = std::numeric_limits<T>::max();
  auto orderby    = cudf::test::fixed_width_column_wrapper<T>{
    {min, T{5}, T{5}, T{6}, T{7}, T{9}, T{9}, T{12}, T{13}, T{17}, T{22}, T{22}, max}};

  auto prec = cudf::make_fixed_width_scalar<T>(4);
  auto foll = cudf::make_fixed_width_scalar<T>(-2);

  auto result = cudf::make_range_windows(cudf::table_view{},
                                         orderby,
                                         cudf::order::ASCENDING,
                                         cudf::null_order::BEFORE,
                                         cudf::bounded_closed{*prec},
                                         cudf::bounded_closed{*foll});

  auto expect_preceding = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    {1, 1, 2, 3, 4, 5, 6, 3, 4, 2, 1, 2, 1}};
  auto expect_following = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    {-1, -1, -2, -3, -2, -1, -2, -1, -2, -1, -1, -2, -1}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<0>(result)->view(), expect_preceding, cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<1>(result)->view(), expect_following, cudf::test::debug_output_level::ALL_ERRORS);

  result = cudf::make_range_windows(cudf::table_view{},
                                    orderby,
                                    cudf::order::ASCENDING,
                                    cudf::null_order::BEFORE,
                                    cudf::bounded_open{*prec},
                                    cudf::bounded_open{*foll});

  expect_preceding = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    {1, 1, 2, 3, 4, 3, 4, 3, 2, 1, 1, 2, 1}};
  expect_following = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    {-1, -1, -2, -3, -4, -2, -3, -1, -2, -1, -1, -2, -1}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<0>(result)->view(), expect_preceding, cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<1>(result)->view(), expect_following, cudf::test::debug_output_level::ALL_ERRORS);
}

TYPED_TEST(UngroupedIntegralRangeWindows, DescendingNoNullsZeroPrecedingFollowing)
{
  using T         = TypeParam;
  constexpr T min = std::numeric_limits<T>::lowest();
  constexpr T max = std::numeric_limits<T>::max();
  auto orderby    = cudf::test::fixed_width_column_wrapper<T>{
    {max, T{22}, T{22}, T{17}, T{13}, T{12}, T{9}, T{9}, T{7}, T{6}, T{5}, T{5}, min}};

  auto prec = cudf::make_fixed_width_scalar<T>(0);
  auto foll = cudf::make_fixed_width_scalar<T>(0);

  auto result = cudf::make_range_windows(cudf::table_view{},
                                         orderby,
                                         cudf::order::DESCENDING,
                                         cudf::null_order::BEFORE,
                                         cudf::bounded_closed{*prec},
                                         cudf::bounded_closed{*foll});

  auto expect_preceding = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    {1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1}};
  auto expect_following = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    {0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<0>(result)->view(), expect_preceding, cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<1>(result)->view(), expect_following, cudf::test::debug_output_level::ALL_ERRORS);

  result = cudf::make_range_windows(cudf::table_view{},
                                    orderby,
                                    cudf::order::DESCENDING,
                                    cudf::null_order::BEFORE,
                                    cudf::bounded_open{*prec},
                                    cudf::bounded_open{*foll});

  expect_preceding = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    {0, -1, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0}};
  expect_following = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    {-1, -1, -2, -1, -1, -1, -1, -2, -1, -1, -1, -2, -1}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<0>(result)->view(), expect_preceding, cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<1>(result)->view(), expect_following, cudf::test::debug_output_level::ALL_ERRORS);
}

TYPED_TEST(UngroupedIntegralRangeWindows, DescendingNoNullsPositivePrecedingFollowing)
{
  using T         = TypeParam;
  constexpr T min = std::numeric_limits<T>::lowest();
  constexpr T max = std::numeric_limits<T>::max();
  auto orderby    = cudf::test::fixed_width_column_wrapper<T>{
    {max, T{22}, T{22}, T{17}, T{13}, T{12}, T{9}, T{9}, T{7}, T{6}, T{5}, T{5}, min}};

  auto prec = cudf::make_fixed_width_scalar<T>(2);
  auto foll = cudf::make_fixed_width_scalar<T>(1);

  auto result = cudf::make_range_windows(cudf::table_view{},
                                         orderby,
                                         cudf::order::DESCENDING,
                                         cudf::null_order::BEFORE,
                                         cudf::bounded_closed{*prec},
                                         cudf::bounded_closed{*foll});

  auto expect_preceding = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    {1, 1, 2, 1, 1, 2, 1, 2, 3, 2, 3, 4, 1}};
  auto expect_following = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    {0, 1, 0, 0, 1, 0, 1, 0, 1, 2, 1, 0, 0}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<0>(result)->view(), expect_preceding, cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<1>(result)->view(), expect_following, cudf::test::debug_output_level::ALL_ERRORS);

  result = cudf::make_range_windows(cudf::table_view{},
                                    orderby,
                                    cudf::order::DESCENDING,
                                    cudf::null_order::BEFORE,
                                    cudf::bounded_open{*prec},
                                    cudf::bounded_open{*foll});

  expect_preceding = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    {1, 1, 2, 1, 1, 2, 1, 2, 1, 2, 2, 3, 1}};
  expect_following = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    {0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<0>(result)->view(), expect_preceding, cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<1>(result)->view(), expect_following, cudf::test::debug_output_level::ALL_ERRORS);
}

TYPED_TEST(UngroupedSignedIntegralRangeWindows, DescendingNoNullsNegativePreceding)
{
  using T         = TypeParam;
  constexpr T min = std::numeric_limits<T>::lowest();
  constexpr T max = std::numeric_limits<T>::max();
  auto orderby    = cudf::test::fixed_width_column_wrapper<T>{
    {max, T{22}, T{22}, T{17}, T{13}, T{12}, T{9}, T{9}, T{7}, T{6}, T{5}, T{5}, min}};

  auto prec = cudf::make_fixed_width_scalar<T>(-1);
  auto foll = cudf::make_fixed_width_scalar<T>(2);

  auto result = cudf::make_range_windows(cudf::table_view{},
                                         orderby,
                                         cudf::order::DESCENDING,
                                         cudf::null_order::BEFORE,
                                         cudf::bounded_closed{*prec},
                                         cudf::bounded_closed{*foll});

  auto expect_preceding = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    {0, -1, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0}};
  auto expect_following = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    {0, 1, 0, 0, 1, 0, 2, 1, 3, 2, 1, 0, 0}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<0>(result)->view(), expect_preceding, cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<1>(result)->view(), expect_following, cudf::test::debug_output_level::ALL_ERRORS);

  result = cudf::make_range_windows(cudf::table_view{},
                                    orderby,
                                    cudf::order::DESCENDING,
                                    cudf::null_order::BEFORE,
                                    cudf::bounded_open{*prec},
                                    cudf::bounded_open{*foll});

  expect_preceding = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    {0, -1, 0, 0, -1, 0, -1, 0, -1, -2, -1, 0, 0}};
  expect_following = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    {0, 1, 0, 0, 1, 0, 1, 0, 1, 2, 1, 0, 0}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<0>(result)->view(), expect_preceding, cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<1>(result)->view(), expect_following, cudf::test::debug_output_level::ALL_ERRORS);
}

TYPED_TEST(UngroupedSignedIntegralRangeWindows, DescendingNoNullsNegativeFollowing)
{
  using T         = TypeParam;
  constexpr T min = std::numeric_limits<T>::lowest();
  constexpr T max = std::numeric_limits<T>::max();
  auto orderby    = cudf::test::fixed_width_column_wrapper<T>{
    {max, T{22}, T{22}, T{17}, T{13}, T{12}, T{9}, T{9}, T{7}, T{6}, T{5}, T{5}, min}};

  auto prec = cudf::make_fixed_width_scalar<T>(4);
  auto foll = cudf::make_fixed_width_scalar<T>(-2);

  auto result = cudf::make_range_windows(cudf::table_view{},
                                         orderby,
                                         cudf::order::DESCENDING,
                                         cudf::null_order::BEFORE,
                                         cudf::bounded_closed{*prec},
                                         cudf::bounded_closed{*foll});

  auto expect_preceding = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    {1, 1, 2, 1, 2, 2, 3, 4, 3, 4, 5, 6, 1}};
  auto expect_following = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    {-1, -1, -2, -1, -1, -2, -1, -2, -1, -2, -2, -3, -1}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<0>(result)->view(), expect_preceding, cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<1>(result)->view(), expect_following, cudf::test::debug_output_level::ALL_ERRORS);

  result = cudf::make_range_windows(cudf::table_view{},
                                    orderby,
                                    cudf::order::DESCENDING,
                                    cudf::null_order::BEFORE,
                                    cudf::bounded_open{*prec},
                                    cudf::bounded_open{*foll});

  expect_preceding = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    {1, 1, 2, 1, 1, 2, 2, 3, 3, 4, 3, 4, 1}};
  expect_following = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    {-1, -1, -2, -1, -1, -2, -1, -2, -3, -2, -3, -4, -1}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<0>(result)->view(), expect_preceding, cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<1>(result)->view(), expect_following, cudf::test::debug_output_level::ALL_ERRORS);
}

TYPED_TEST(UngroupedIntegralRangeWindows, AscendingNullsZeroPrecedingFollowing)
{
  using T         = TypeParam;
  constexpr T min = std::numeric_limits<T>::lowest();
  constexpr T max = std::numeric_limits<T>::max();
  auto orderby    = cudf::test::fixed_width_column_wrapper<T>{
    {min, T{5}, T{5}, T{6}, T{7}, T{9}, T{9}, T{12}, T{13}, T{17}, T{22}, T{22}, max},
    {false, false, false, false, true, true, true, true, true, true, true, true, true}};

  auto prec = cudf::make_fixed_width_scalar<T>(0);
  auto foll = cudf::make_fixed_width_scalar<T>(0);

  auto result = cudf::make_range_windows(cudf::table_view{},
                                         orderby,
                                         cudf::order::ASCENDING,
                                         cudf::null_order::BEFORE,
                                         cudf::bounded_closed{*prec},
                                         cudf::bounded_closed{*foll});

  auto expect_preceding = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    {1, 2, 3, 4, 1, 1, 2, 1, 1, 1, 1, 2, 1}};
  auto expect_following = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    {3, 2, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<0>(result)->view(), expect_preceding, cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<1>(result)->view(), expect_following, cudf::test::debug_output_level::ALL_ERRORS);

  result = cudf::make_range_windows(cudf::table_view{},
                                    orderby,
                                    cudf::order::ASCENDING,
                                    cudf::null_order::BEFORE,
                                    cudf::bounded_open{*prec},
                                    cudf::bounded_open{*foll});

  // TODO: What does it mean to be a BOUNDED_OPEN window in presence of nulls?
  // Should the group of nulls be an empty window?
  expect_preceding = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    {1, 2, 3, 4, 0, -1, 0, 0, 0, 0, -1, 0, 0}};
  expect_following = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    {3, 2, 1, 0, -1, -1, -2, -1, -1, -1, -1, -2, -1}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<0>(result)->view(), expect_preceding, cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<1>(result)->view(), expect_following, cudf::test::debug_output_level::ALL_ERRORS);
}

TYPED_TEST(UngroupedIntegralRangeWindows, AscendingNullsPositivePrecedingFollowing)
{
  using T         = TypeParam;
  constexpr T min = std::numeric_limits<T>::lowest();
  constexpr T max = std::numeric_limits<T>::max();
  auto orderby    = cudf::test::fixed_width_column_wrapper<T>{
    {min, T{5}, T{5}, T{6}, T{7}, T{9}, T{9}, T{12}, T{13}, T{17}, T{22}, T{22}, max},
    {false, false, false, false, true, true, true, true, true, true, true, true, true}};

  auto prec = cudf::make_fixed_width_scalar<T>(2);
  auto foll = cudf::make_fixed_width_scalar<T>(1);

  auto result = cudf::make_range_windows(cudf::table_view{},
                                         orderby,
                                         cudf::order::ASCENDING,
                                         cudf::null_order::BEFORE,
                                         cudf::bounded_closed{*prec},
                                         cudf::bounded_closed{*foll});

  auto expect_preceding = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    {1, 2, 3, 4, 1, 2, 3, 1, 2, 1, 1, 2, 1}};
  auto expect_following = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    {3, 2, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<0>(result)->view(), expect_preceding, cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<1>(result)->view(), expect_following, cudf::test::debug_output_level::ALL_ERRORS);

  result = cudf::make_range_windows(cudf::table_view{},
                                    orderby,
                                    cudf::order::ASCENDING,
                                    cudf::null_order::BEFORE,
                                    cudf::bounded_open{*prec},
                                    cudf::bounded_open{*foll});

  expect_preceding = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    {1, 2, 3, 4, 1, 1, 2, 1, 2, 1, 1, 2, 1}};
  expect_following = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    {3, 2, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<0>(result)->view(), expect_preceding, cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<1>(result)->view(), expect_following, cudf::test::debug_output_level::ALL_ERRORS);
}

TYPED_TEST(UngroupedSignedIntegralRangeWindows, AscendingNullsNegativePreceding)
{
  using T         = TypeParam;
  constexpr T min = std::numeric_limits<T>::lowest();
  constexpr T max = std::numeric_limits<T>::max();
  auto orderby    = cudf::test::fixed_width_column_wrapper<T>{
    {min, T{5}, T{5}, T{6}, T{7}, T{9}, T{9}, T{12}, T{13}, T{17}, T{22}, T{22}, max},
    {true, true, true, true, true, true, false, false, false, false, false, false, false}};

  auto prec = cudf::make_fixed_width_scalar<T>(-1);
  auto foll = cudf::make_fixed_width_scalar<T>(2);

  auto result = cudf::make_range_windows(cudf::table_view{},
                                         orderby,
                                         cudf::order::ASCENDING,
                                         cudf::null_order::AFTER,
                                         cudf::bounded_closed{*prec},
                                         cudf::bounded_closed{*foll});

  auto expect_preceding = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    {0, -1, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7}};
  auto expect_following = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    {0, 3, 2, 1, 1, 0, 6, 5, 4, 3, 2, 1, 0}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<0>(result)->view(), expect_preceding, cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<1>(result)->view(), expect_following, cudf::test::debug_output_level::ALL_ERRORS);

  result = cudf::make_range_windows(cudf::table_view{},
                                    orderby,
                                    cudf::order::ASCENDING,
                                    cudf::null_order::AFTER,
                                    cudf::bounded_open{*prec},
                                    cudf::bounded_open{*foll});

  expect_preceding = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    {0, -2, -1, -1, 0, 0, 1, 2, 3, 4, 5, 6, 7}};
  expect_following = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    {0, 2, 1, 1, 0, 0, 6, 5, 4, 3, 2, 1, 0}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<0>(result)->view(), expect_preceding, cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<1>(result)->view(), expect_following, cudf::test::debug_output_level::ALL_ERRORS);
}

TYPED_TEST(UngroupedSignedIntegralRangeWindows, AscendingNullsNegativeFollowing)
{
  using T         = TypeParam;
  constexpr T min = std::numeric_limits<T>::lowest();
  constexpr T max = std::numeric_limits<T>::max();
  auto orderby    = cudf::test::fixed_width_column_wrapper<T>{
    {min, T{5}, T{5}, T{6}, T{7}, T{9}, T{9}, T{12}, T{13}, T{17}, T{22}, T{22}, max},
    {true, true, true, true, true, true, false, false, false, false, false, false, false}};

  auto prec = cudf::make_fixed_width_scalar<T>(4);
  auto foll = cudf::make_fixed_width_scalar<T>(-2);

  auto result = cudf::make_range_windows(cudf::table_view{},
                                         orderby,
                                         cudf::order::ASCENDING,
                                         cudf::null_order::AFTER,
                                         cudf::bounded_closed{*prec},
                                         cudf::bounded_closed{*foll});

  auto expect_preceding = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    {1, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6, 7}};
  auto expect_following = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    {-1, -1, -2, -3, -2, -1, 6, 5, 4, 3, 2, 1, 0}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<0>(result)->view(), expect_preceding, cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<1>(result)->view(), expect_following, cudf::test::debug_output_level::ALL_ERRORS);

  result = cudf::make_range_windows(cudf::table_view{},
                                    orderby,
                                    cudf::order::ASCENDING,
                                    cudf::null_order::AFTER,
                                    cudf::bounded_open{*prec},
                                    cudf::bounded_open{*foll});

  expect_preceding = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    {1, 1, 2, 3, 4, 3, 1, 2, 3, 4, 5, 6, 7}};
  expect_following = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    {-1, -1, -2, -3, -4, -2, 6, 5, 4, 3, 2, 1, 0}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<0>(result)->view(), expect_preceding, cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<1>(result)->view(), expect_following, cudf::test::debug_output_level::ALL_ERRORS);
}

TYPED_TEST(UngroupedIntegralRangeWindows, DescendingNullsZeroPrecedingFollowing)
{
  using T         = TypeParam;
  constexpr T min = std::numeric_limits<T>::lowest();
  constexpr T max = std::numeric_limits<T>::max();
  auto orderby    = cudf::test::fixed_width_column_wrapper<T>{
    {max, T{22}, T{22}, T{17}, T{13}, T{12}, T{9}, T{9}, T{7}, T{6}, T{5}, T{5}, min},
    {true, true, true, true, true, true, true, false, false, false, false, false, false}};

  auto prec = cudf::make_fixed_width_scalar<T>(0);
  auto foll = cudf::make_fixed_width_scalar<T>(0);

  auto result = cudf::make_range_windows(cudf::table_view{},
                                         orderby,
                                         cudf::order::DESCENDING,
                                         cudf::null_order::BEFORE,
                                         cudf::bounded_closed{*prec},
                                         cudf::bounded_closed{*foll});

  auto expect_preceding = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    {1, 1, 2, 1, 1, 1, 1, 1, 2, 3, 4, 5, 6}};
  auto expect_following = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    {0, 1, 0, 0, 0, 0, 0, 5, 4, 3, 2, 1, 0}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<0>(result)->view(), expect_preceding, cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<1>(result)->view(), expect_following, cudf::test::debug_output_level::ALL_ERRORS);

  result = cudf::make_range_windows(cudf::table_view{},
                                    orderby,
                                    cudf::order::DESCENDING,
                                    cudf::null_order::BEFORE,
                                    cudf::bounded_open{*prec},
                                    cudf::bounded_open{*foll});

  expect_preceding = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    {0, -1, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6}};
  expect_following = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    {-1, -1, -2, -1, -1, -1, -1, 5, 4, 3, 2, 1, 0}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<0>(result)->view(), expect_preceding, cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<1>(result)->view(), expect_following, cudf::test::debug_output_level::ALL_ERRORS);
}

TYPED_TEST(UngroupedIntegralRangeWindows, DescendingNullsPositivePrecedingFollowing)
{
  using T         = TypeParam;
  constexpr T min = std::numeric_limits<T>::lowest();
  constexpr T max = std::numeric_limits<T>::max();
  auto orderby    = cudf::test::fixed_width_column_wrapper<T>{
    {max, T{22}, T{22}, T{17}, T{13}, T{12}, T{9}, T{9}, T{7}, T{6}, T{5}, T{5}, min},
    {true, true, true, true, true, true, true, false, false, false, false, false, false}};

  auto prec = cudf::make_fixed_width_scalar<T>(2);
  auto foll = cudf::make_fixed_width_scalar<T>(1);

  auto result = cudf::make_range_windows(cudf::table_view{},
                                         orderby,
                                         cudf::order::DESCENDING,
                                         cudf::null_order::BEFORE,
                                         cudf::bounded_closed{*prec},
                                         cudf::bounded_closed{*foll});

  auto expect_preceding = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    {1, 1, 2, 1, 1, 2, 1, 1, 2, 3, 4, 5, 6}};
  auto expect_following = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    {0, 1, 0, 0, 1, 0, 0, 5, 4, 3, 2, 1, 0}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<0>(result)->view(), expect_preceding, cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<1>(result)->view(), expect_following, cudf::test::debug_output_level::ALL_ERRORS);

  result = cudf::make_range_windows(cudf::table_view{},
                                    orderby,
                                    cudf::order::DESCENDING,
                                    cudf::null_order::BEFORE,
                                    cudf::bounded_open{*prec},
                                    cudf::bounded_open{*foll});

  expect_preceding = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    {1, 1, 2, 1, 1, 2, 1, 1, 2, 3, 4, 5, 6}};
  expect_following = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    {0, 1, 0, 0, 0, 0, 0, 5, 4, 3, 2, 1, 0}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<0>(result)->view(), expect_preceding, cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<1>(result)->view(), expect_following, cudf::test::debug_output_level::ALL_ERRORS);
}

TYPED_TEST(UngroupedSignedIntegralRangeWindows, DescendingNullsNegativePreceding)
{
  using T         = TypeParam;
  constexpr T min = std::numeric_limits<T>::lowest();
  constexpr T max = std::numeric_limits<T>::max();
  auto orderby    = cudf::test::fixed_width_column_wrapper<T>{
    {max, T{22}, T{22}, T{17}, T{13}, T{12}, T{9}, T{9}, T{7}, T{6}, T{5}, T{5}, min},
    {false, false, false, false, false, true, true, true, true, true, true, true, true}};

  auto prec = cudf::make_fixed_width_scalar<T>(-1);
  auto foll = cudf::make_fixed_width_scalar<T>(2);

  auto result = cudf::make_range_windows(cudf::table_view{},
                                         orderby,
                                         cudf::order::DESCENDING,
                                         cudf::null_order::AFTER,
                                         cudf::bounded_closed{*prec},
                                         cudf::bounded_closed{*foll});

  auto expect_preceding = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    {1, 2, 3, 4, 5, 0, -1, 0, 0, 0, -1, 0, 0}};
  auto expect_following = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    {4, 3, 2, 1, 0, 0, 2, 1, 3, 2, 1, 0, 0}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<0>(result)->view(), expect_preceding, cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<1>(result)->view(), expect_following, cudf::test::debug_output_level::ALL_ERRORS);

  result = cudf::make_range_windows(cudf::table_view{},
                                    orderby,
                                    cudf::order::DESCENDING,
                                    cudf::null_order::AFTER,
                                    cudf::bounded_open{*prec},
                                    cudf::bounded_open{*foll});

  expect_preceding = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    {1, 2, 3, 4, 5, 0, -1, 0, -1, -2, -1, 0, 0}};
  expect_following = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    {4, 3, 2, 1, 0, 0, 1, 0, 1, 2, 1, 0, 0}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<0>(result)->view(), expect_preceding, cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<1>(result)->view(), expect_following, cudf::test::debug_output_level::ALL_ERRORS);
}

TYPED_TEST(UngroupedSignedIntegralRangeWindows, DescendingNullsNegativeFollowing)
{
  using T         = TypeParam;
  constexpr T min = std::numeric_limits<T>::lowest();
  constexpr T max = std::numeric_limits<T>::max();
  auto orderby    = cudf::test::fixed_width_column_wrapper<T>{
    {max, T{22}, T{22}, T{17}, T{13}, T{12}, T{9}, T{9}, T{7}, T{6}, T{5}, T{5}, min},
    {false, false, false, false, false, true, true, true, true, true, true, true, true}};

  auto prec = cudf::make_fixed_width_scalar<T>(4);
  auto foll = cudf::make_fixed_width_scalar<T>(-2);

  auto result = cudf::make_range_windows(cudf::table_view{},
                                         orderby,
                                         cudf::order::DESCENDING,
                                         cudf::null_order::AFTER,
                                         cudf::bounded_closed{*prec},
                                         cudf::bounded_closed{*foll});

  auto expect_preceding = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    {1, 2, 3, 4, 5, 1, 2, 3, 3, 4, 5, 6, 1}};
  auto expect_following = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    {4, 3, 2, 1, 0, -1, -1, -2, -1, -2, -2, -3, -1}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<0>(result)->view(), expect_preceding, cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<1>(result)->view(), expect_following, cudf::test::debug_output_level::ALL_ERRORS);

  result = cudf::make_range_windows(cudf::table_view{},
                                    orderby,
                                    cudf::order::DESCENDING,
                                    cudf::null_order::AFTER,
                                    cudf::bounded_open{*prec},
                                    cudf::bounded_open{*foll});

  expect_preceding = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    {1, 2, 3, 4, 5, 1, 2, 3, 3, 4, 3, 4, 1}};
  expect_following = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    {4, 3, 2, 1, 0, -1, -1, -2, -3, -2, -3, -4, -1}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<0>(result)->view(), expect_preceding, cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<1>(result)->view(), expect_following, cudf::test::debug_output_level::ALL_ERRORS);
}

TYPED_TEST(GroupedIntegralRangeWindows, AscendingNoNullsZeroPrecedingFollowing)
{
  using T         = TypeParam;
  constexpr T min = std::numeric_limits<T>::lowest();
  constexpr T max = std::numeric_limits<T>::max();
  auto orderby    = cudf::test::fixed_width_column_wrapper<T>{{
    // clang-format off
    // Group-1
    min, T{5}, T{5}, T{6}, T{7}, T{9}, T{9}, T{12}, T{13}, T{17}, T{22}, T{22}, max,
    // Group-2
    min, T{5}, T{5}, T{6}, T{7}, T{9}, T{9}, T{12}, T{13}, T{17}, T{22}, T{22}, max,
    // Group-3
    min, T{5}, T{5}, T{6}, T{7}, T{9}, T{9}, T{12}, T{13}, T{17}, T{22}, T{22}, max,
    // clang-format on
  }};

  auto group_keys = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // Group-2
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    // Group-3
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    // clang-format on
  }};

  auto prec = cudf::make_fixed_width_scalar<T>(0);
  auto foll = cudf::make_fixed_width_scalar<T>(0);

  auto result = cudf::make_range_windows(cudf::table_view{{group_keys}},
                                         orderby,
                                         cudf::order::ASCENDING,
                                         cudf::null_order::BEFORE,
                                         cudf::bounded_closed{*prec},
                                         cudf::bounded_closed{*foll});

  auto expect_preceding = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1,
    // Group-2
    1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1,
    // Group-3
    1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1,
    // clang-format on
  }};
  auto expect_following = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
    // Group-2
    0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
    // Group-3
    0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
    // clang-format on
  }};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<0>(result)->view(), expect_preceding, cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<1>(result)->view(), expect_following, cudf::test::debug_output_level::ALL_ERRORS);

  result = cudf::make_range_windows(cudf::table_view{{group_keys}},
                                    orderby,
                                    cudf::order::ASCENDING,
                                    cudf::null_order::BEFORE,
                                    cudf::bounded_open{*prec},
                                    cudf::bounded_open{*foll});

  expect_preceding = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    0, -1, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0,
    // Group-2
    0, -1, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0,
    // Group-3
    0, -1, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0,
    // clang-format on
  }};
  expect_following = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    -1, -1, -2, -1, -1, -1, -2, -1, -1, -1, -1, -2, -1,
    // Group-2
    -1, -1, -2, -1, -1, -1, -2, -1, -1, -1, -1, -2, -1,
    // Group-3
    -1, -1, -2, -1, -1, -1, -2, -1, -1, -1, -1, -2, -1,
    // clang-format on
  }};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<0>(result)->view(), expect_preceding, cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<1>(result)->view(), expect_following, cudf::test::debug_output_level::ALL_ERRORS);
}

TYPED_TEST(GroupedIntegralRangeWindows, AscendingNoNullsPositivePrecedingFollowing)
{
  using T         = TypeParam;
  constexpr T min = std::numeric_limits<T>::lowest();
  constexpr T max = std::numeric_limits<T>::max();
  auto orderby    = cudf::test::fixed_width_column_wrapper<T>{{
    // clang-format off
    // Group-1
    min, T{5}, T{5}, T{6}, T{7}, T{9}, T{9}, T{12}, T{13}, T{17}, T{22}, T{22}, max,
    // Group-2
    min, T{5}, T{5}, T{6}, T{7}, T{9}, T{9}, T{12}, T{13}, T{17}, T{22}, T{22}, max,
    // Group-3
    min, T{5}, T{5}, T{6}, T{7}, T{9}, T{9}, T{12}, T{13}, T{17}, T{22}, T{22}, max,
    // clang-format on
  }};

  auto group_keys = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // Group-2
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    // Group-3
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    // clang-format on
  }};

  auto prec = cudf::make_fixed_width_scalar<T>(2);
  auto foll = cudf::make_fixed_width_scalar<T>(1);

  auto result = cudf::make_range_windows(cudf::table_view{{group_keys}},
                                         orderby,
                                         cudf::order::ASCENDING,
                                         cudf::null_order::BEFORE,
                                         cudf::bounded_closed{*prec},
                                         cudf::bounded_closed{*foll});

  auto expect_preceding = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    1, 1, 2, 3, 4, 2, 3, 1, 2, 1, 1, 2, 1,
    // Group-2
    1, 1, 2, 3, 4, 2, 3, 1, 2, 1, 1, 2, 1,
    // Group-3
    1, 1, 2, 3, 4, 2, 3, 1, 2, 1, 1, 2, 1,
    // clang-format on
  }};
  auto expect_following = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    0, 2, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0,
    // Group-2
    0, 2, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0,
    // Group-3
    0, 2, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0,
    // clang-format on
  }};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<0>(result)->view(), expect_preceding, cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<1>(result)->view(), expect_following, cudf::test::debug_output_level::ALL_ERRORS);

  result = cudf::make_range_windows(cudf::table_view{{group_keys}},
                                    orderby,
                                    cudf::order::ASCENDING,
                                    cudf::null_order::BEFORE,
                                    cudf::bounded_open{*prec},
                                    cudf::bounded_open{*foll});

  expect_preceding = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    1, 1, 2, 3, 2, 1, 2, 1, 2, 1, 1, 2, 1,
    // Group-2
    1, 1, 2, 3, 2, 1, 2, 1, 2, 1, 1, 2, 1,
    // Group-3
    1, 1, 2, 3, 2, 1, 2, 1, 2, 1, 1, 2, 1,
    // clang-format on
  }};
  expect_following = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
    // Group-2
    0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
    // Group-3
    0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
    // clang-format on
  }};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<0>(result)->view(), expect_preceding, cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<1>(result)->view(), expect_following, cudf::test::debug_output_level::ALL_ERRORS);
}

TYPED_TEST(GroupedSignedIntegralRangeWindows, AscendingNoNullsNegativePreceding)
{
  using T         = TypeParam;
  constexpr T min = std::numeric_limits<T>::lowest();
  constexpr T max = std::numeric_limits<T>::max();
  auto orderby    = cudf::test::fixed_width_column_wrapper<T>{{
    // clang-format off
    // Group-1
    min, T{5}, T{5}, T{6}, T{7}, T{9}, T{9}, T{12}, T{13}, T{17}, T{22}, T{22}, max,
    // Group-2
    min, T{5}, T{5}, T{6}, T{7}, T{9}, T{9}, T{12}, T{13}, T{17}, T{22}, T{22}, max,
    // Group-3
    min, T{5}, T{5}, T{6}, T{7}, T{9}, T{9}, T{12}, T{13}, T{17}, T{22}, T{22}, max,
    // clang-format on
  }};

  auto group_keys = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // Group-2
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    // Group-3
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    // clang-format on
  }};

  auto prec = cudf::make_fixed_width_scalar<T>(-1);
  auto foll = cudf::make_fixed_width_scalar<T>(2);

  auto result = cudf::make_range_windows(cudf::table_view{{group_keys}},
                                         orderby,
                                         cudf::order::ASCENDING,
                                         cudf::null_order::BEFORE,
                                         cudf::bounded_closed{*prec},
                                         cudf::bounded_closed{*foll});

  auto expect_preceding = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    0, -1, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0,
    // Group-2
    0, -1, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0,
    // Group-3
    0, -1, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0,
    // clang-format on
  }};
  auto expect_following = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    0, 3, 2, 1, 2, 1, 0, 1, 0, 0, 1, 0, 0,
    // Group-2
    0, 3, 2, 1, 2, 1, 0, 1, 0, 0, 1, 0, 0,
    // Group-3
    0, 3, 2, 1, 2, 1, 0, 1, 0, 0, 1, 0, 0,
    // clang-format on
  }};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<0>(result)->view(), expect_preceding, cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<1>(result)->view(), expect_following, cudf::test::debug_output_level::ALL_ERRORS);

  result = cudf::make_range_windows(cudf::table_view{{group_keys}},
                                    orderby,
                                    cudf::order::ASCENDING,
                                    cudf::null_order::BEFORE,
                                    cudf::bounded_open{*prec},
                                    cudf::bounded_open{*foll});

  expect_preceding = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    0, -2, -1, -1, 0, -1, 0, -1, 0, 0, -1, 0, 0,
    // Group-2
    0, -2, -1, -1, 0, -1, 0, -1, 0, 0, -1, 0, 0,
    // Group-3
    0, -2, -1, -1, 0, -1, 0, -1, 0, 0, -1, 0, 0,
    // clang-format on
  }};
  expect_following = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    0, 2, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0,
    // Group-2
    0, 2, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0,
    // Group-3
    0, 2, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0,
    // clang-format on
  }};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<0>(result)->view(), expect_preceding, cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<1>(result)->view(), expect_following, cudf::test::debug_output_level::ALL_ERRORS);
}

TYPED_TEST(GroupedSignedIntegralRangeWindows, AscendingNoNullsNegativeFollowing)
{
  using T         = TypeParam;
  constexpr T min = std::numeric_limits<T>::lowest();
  constexpr T max = std::numeric_limits<T>::max();
  auto orderby    = cudf::test::fixed_width_column_wrapper<T>{{
    // clang-format off
    // Group-1
    min, T{5}, T{5}, T{6}, T{7}, T{9}, T{9}, T{12}, T{13}, T{17}, T{22}, T{22}, max,
    // Group-2
    min, T{5}, T{5}, T{6}, T{7}, T{9}, T{9}, T{12}, T{13}, T{17}, T{22}, T{22}, max,
    // Group-3
    min, T{5}, T{5}, T{6}, T{7}, T{9}, T{9}, T{12}, T{13}, T{17}, T{22}, T{22}, max,
    // clang-format on
  }};

  auto group_keys = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // Group-2
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    // Group-3
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    // clang-format on
  }};

  auto prec = cudf::make_fixed_width_scalar<T>(4);
  auto foll = cudf::make_fixed_width_scalar<T>(-2);

  auto result = cudf::make_range_windows(cudf::table_view{{group_keys}},
                                         orderby,
                                         cudf::order::ASCENDING,
                                         cudf::null_order::BEFORE,
                                         cudf::bounded_closed{*prec},
                                         cudf::bounded_closed{*foll});

  auto expect_preceding = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    1, 1, 2, 3, 4, 5, 6, 3, 4, 2, 1, 2, 1,
    // Group-2
    1, 1, 2, 3, 4, 5, 6, 3, 4, 2, 1, 2, 1,
    // Group-3
    1, 1, 2, 3, 4, 5, 6, 3, 4, 2, 1, 2, 1,
    // clang-format on
  }};
  auto expect_following = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    -1, -1, -2, -3, -2, -1, -2, -1, -2, -1, -1, -2, -1,
    // Group-2
    -1, -1, -2, -3, -2, -1, -2, -1, -2, -1, -1, -2, -1,
    // Group-3
    -1, -1, -2, -3, -2, -1, -2, -1, -2, -1, -1, -2, -1,
    // clang-format on
  }};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<0>(result)->view(), expect_preceding, cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<1>(result)->view(), expect_following, cudf::test::debug_output_level::ALL_ERRORS);

  result = cudf::make_range_windows(cudf::table_view{{group_keys}},
                                    orderby,
                                    cudf::order::ASCENDING,
                                    cudf::null_order::BEFORE,
                                    cudf::bounded_open{*prec},
                                    cudf::bounded_open{*foll});

  expect_preceding = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    1, 1, 2, 3, 4, 3, 4, 3, 2, 1, 1, 2, 1,
    // Group-2
    1, 1, 2, 3, 4, 3, 4, 3, 2, 1, 1, 2, 1,
    // Group-3
    1, 1, 2, 3, 4, 3, 4, 3, 2, 1, 1, 2, 1,
    // clang-format on
  }};
  expect_following = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    -1, -1, -2, -3, -4, -2, -3, -1, -2, -1, -1, -2, -1,
    // Group-2
    -1, -1, -2, -3, -4, -2, -3, -1, -2, -1, -1, -2, -1,
    // Group-3
    -1, -1, -2, -3, -4, -2, -3, -1, -2, -1, -1, -2, -1,
    // clang-format on
  }};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<0>(result)->view(), expect_preceding, cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<1>(result)->view(), expect_following, cudf::test::debug_output_level::ALL_ERRORS);
}

TYPED_TEST(GroupedIntegralRangeWindows, DescendingNoNullsZeroPrecedingFollowing)
{
  using T         = TypeParam;
  constexpr T min = std::numeric_limits<T>::lowest();
  constexpr T max = std::numeric_limits<T>::max();
  auto orderby    = cudf::test::fixed_width_column_wrapper<T>{{
    // clang-format off
    // Group-1
    max, T{22}, T{22}, T{17}, T{13}, T{12}, T{9}, T{9}, T{7}, T{6}, T{5}, T{5}, min,
    // Group-2
    max, T{22}, T{22}, T{17}, T{13}, T{12}, T{9}, T{9}, T{7}, T{6}, T{5}, T{5}, min,
    // Group-3
    max, T{22}, T{22}, T{17}, T{13}, T{12}, T{9}, T{9}, T{7}, T{6}, T{5}, T{5}, min,
    // clang-format on
  }};

  auto group_keys = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // Group-2
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    // Group-3
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    // clang-format on
  }};

  auto prec = cudf::make_fixed_width_scalar<T>(0);
  auto foll = cudf::make_fixed_width_scalar<T>(0);

  auto result = cudf::make_range_windows(cudf::table_view{{group_keys}},
                                         orderby,
                                         cudf::order::DESCENDING,
                                         cudf::null_order::BEFORE,
                                         cudf::bounded_closed{*prec},
                                         cudf::bounded_closed{*foll});

  auto expect_preceding = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1,
    // Group-2
    1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1,
    // Group-3
    1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1,
    // clang-format on
  }};
  auto expect_following = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
    // Group-2
    0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
    // Group-3
    0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
    // clang-format on
  }};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<0>(result)->view(), expect_preceding, cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<1>(result)->view(), expect_following, cudf::test::debug_output_level::ALL_ERRORS);

  result = cudf::make_range_windows(cudf::table_view{{group_keys}},
                                    orderby,
                                    cudf::order::DESCENDING,
                                    cudf::null_order::BEFORE,
                                    cudf::bounded_open{*prec},
                                    cudf::bounded_open{*foll});

  expect_preceding = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    0, -1, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0,
    // Group-2
    0, -1, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0,
    // Group-3
    0, -1, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0,
    // clang-format on
  }};
  expect_following = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    -1, -1, -2, -1, -1, -1, -1, -2, -1, -1, -1, -2, -1,
    // Group-2
    -1, -1, -2, -1, -1, -1, -1, -2, -1, -1, -1, -2, -1,
    // Group-3
    -1, -1, -2, -1, -1, -1, -1, -2, -1, -1, -1, -2, -1,
    // clang-format on
  }};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<0>(result)->view(), expect_preceding, cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<1>(result)->view(), expect_following, cudf::test::debug_output_level::ALL_ERRORS);
}

TYPED_TEST(GroupedIntegralRangeWindows, DescendingNoNullsPositivePrecedingFollowing)
{
  using T         = TypeParam;
  constexpr T min = std::numeric_limits<T>::lowest();
  constexpr T max = std::numeric_limits<T>::max();
  auto orderby    = cudf::test::fixed_width_column_wrapper<T>{{
    // clang-format off
    // Group-1
    max, T{22}, T{22}, T{17}, T{13}, T{12}, T{9}, T{9}, T{7}, T{6}, T{5}, T{5}, min,
    // Group-2
    max, T{22}, T{22}, T{17}, T{13}, T{12}, T{9}, T{9}, T{7}, T{6}, T{5}, T{5}, min,
    // Group-3
    max, T{22}, T{22}, T{17}, T{13}, T{12}, T{9}, T{9}, T{7}, T{6}, T{5}, T{5}, min,
    // clang-format on
  }};

  auto group_keys = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // Group-2
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    // Group-3
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    // clang-format on
  }};

  auto prec = cudf::make_fixed_width_scalar<T>(2);
  auto foll = cudf::make_fixed_width_scalar<T>(1);

  auto result = cudf::make_range_windows(cudf::table_view{{group_keys}},
                                         orderby,
                                         cudf::order::DESCENDING,
                                         cudf::null_order::BEFORE,
                                         cudf::bounded_closed{*prec},
                                         cudf::bounded_closed{*foll});

  auto expect_preceding = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    1, 1, 2, 1, 1, 2, 1, 2, 3, 2, 3, 4, 1,
    // Group-2
    1, 1, 2, 1, 1, 2, 1, 2, 3, 2, 3, 4, 1,
    // Group-3
    1, 1, 2, 1, 1, 2, 1, 2, 3, 2, 3, 4, 1,
    // clang-format on
  }};
  auto expect_following = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    0, 1, 0, 0, 1, 0, 1, 0, 1, 2, 1, 0, 0,
    // Group-2
    0, 1, 0, 0, 1, 0, 1, 0, 1, 2, 1, 0, 0,
    // Group-3
    0, 1, 0, 0, 1, 0, 1, 0, 1, 2, 1, 0, 0,
    // clang-format on
  }};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<0>(result)->view(), expect_preceding, cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<1>(result)->view(), expect_following, cudf::test::debug_output_level::ALL_ERRORS);

  result = cudf::make_range_windows(cudf::table_view{{group_keys}},
                                    orderby,
                                    cudf::order::DESCENDING,
                                    cudf::null_order::BEFORE,
                                    cudf::bounded_open{*prec},
                                    cudf::bounded_open{*foll});

  expect_preceding = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    1, 1, 2, 1, 1, 2, 1, 2, 1, 2, 2, 3, 1,
    // Group-2
    1, 1, 2, 1, 1, 2, 1, 2, 1, 2, 2, 3, 1,
    // Group-3
    1, 1, 2, 1, 1, 2, 1, 2, 1, 2, 2, 3, 1,
    // clang-format on
  }};
  expect_following = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
    // Group-2
    0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
    // Group-3
    0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
    // clang-format on
  }};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<0>(result)->view(), expect_preceding, cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<1>(result)->view(), expect_following, cudf::test::debug_output_level::ALL_ERRORS);
}

TYPED_TEST(GroupedSignedIntegralRangeWindows, DescendingNoNullsNegativePreceding)
{
  using T         = TypeParam;
  constexpr T min = std::numeric_limits<T>::lowest();
  constexpr T max = std::numeric_limits<T>::max();
  auto orderby    = cudf::test::fixed_width_column_wrapper<T>{{
    // clang-format off
    // Group-1
    max, T{22}, T{22}, T{17}, T{13}, T{12}, T{9}, T{9}, T{7}, T{6}, T{5}, T{5}, min,
    // Group-2
    max, T{22}, T{22}, T{17}, T{13}, T{12}, T{9}, T{9}, T{7}, T{6}, T{5}, T{5}, min,
    // Group-3
    max, T{22}, T{22}, T{17}, T{13}, T{12}, T{9}, T{9}, T{7}, T{6}, T{5}, T{5}, min,
    // clang-format on
  }};

  auto group_keys = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // Group-2
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    // Group-3
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    // clang-format on
  }};

  auto prec = cudf::make_fixed_width_scalar<T>(-1);
  auto foll = cudf::make_fixed_width_scalar<T>(2);

  auto result = cudf::make_range_windows(cudf::table_view{{group_keys}},
                                         orderby,
                                         cudf::order::DESCENDING,
                                         cudf::null_order::BEFORE,
                                         cudf::bounded_closed{*prec},
                                         cudf::bounded_closed{*foll});

  auto expect_preceding = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    0, -1, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0,
    // Group-2
    0, -1, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0,
    // Group-3
    0, -1, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0,
    // clang-format on
  }};
  auto expect_following = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    0, 1, 0, 0, 1, 0, 2, 1, 3, 2, 1, 0, 0,
    // Group-2
    0, 1, 0, 0, 1, 0, 2, 1, 3, 2, 1, 0, 0,
    // Group-3
    0, 1, 0, 0, 1, 0, 2, 1, 3, 2, 1, 0, 0,
    // clang-format on
  }};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<0>(result)->view(), expect_preceding, cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<1>(result)->view(), expect_following, cudf::test::debug_output_level::ALL_ERRORS);

  result = cudf::make_range_windows(cudf::table_view{{group_keys}},
                                    orderby,
                                    cudf::order::DESCENDING,
                                    cudf::null_order::BEFORE,
                                    cudf::bounded_open{*prec},
                                    cudf::bounded_open{*foll});

  expect_preceding = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    0, -1, 0, 0, -1, 0, -1, 0, -1, -2, -1, 0, 0,
    // Group-2
    0, -1, 0, 0, -1, 0, -1, 0, -1, -2, -1, 0, 0,
    // Group-3
    0, -1, 0, 0, -1, 0, -1, 0, -1, -2, -1, 0, 0,
    // clang-format on
  }};
  expect_following = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    0, 1, 0, 0, 1, 0, 1, 0, 1, 2, 1, 0, 0,
    // Group-2
    0, 1, 0, 0, 1, 0, 1, 0, 1, 2, 1, 0, 0,
    // Group-3
    0, 1, 0, 0, 1, 0, 1, 0, 1, 2, 1, 0, 0,
    // clang-format on
  }};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<0>(result)->view(), expect_preceding, cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<1>(result)->view(), expect_following, cudf::test::debug_output_level::ALL_ERRORS);
}

TYPED_TEST(GroupedSignedIntegralRangeWindows, DescendingNoNullsNegativeFollowing)
{
  using T         = TypeParam;
  constexpr T min = std::numeric_limits<T>::lowest();
  constexpr T max = std::numeric_limits<T>::max();
  auto orderby    = cudf::test::fixed_width_column_wrapper<T>{{
    // clang-format off
    // Group-1
    max, T{22}, T{22}, T{17}, T{13}, T{12}, T{9}, T{9}, T{7}, T{6}, T{5}, T{5}, min,
    // Group-2
    max, T{22}, T{22}, T{17}, T{13}, T{12}, T{9}, T{9}, T{7}, T{6}, T{5}, T{5}, min,
    // Group-3
    max, T{22}, T{22}, T{17}, T{13}, T{12}, T{9}, T{9}, T{7}, T{6}, T{5}, T{5}, min,
    // clang-format on
  }};

  auto group_keys = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // Group-2
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    // Group-3
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    // clang-format on
  }};

  auto prec = cudf::make_fixed_width_scalar<T>(4);
  auto foll = cudf::make_fixed_width_scalar<T>(-2);

  auto result = cudf::make_range_windows(cudf::table_view{{group_keys}},
                                         orderby,
                                         cudf::order::DESCENDING,
                                         cudf::null_order::BEFORE,
                                         cudf::bounded_closed{*prec},
                                         cudf::bounded_closed{*foll});

  auto expect_preceding = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    1, 1, 2, 1, 2, 2, 3, 4, 3, 4, 5, 6, 1,
    // Group-2
    1, 1, 2, 1, 2, 2, 3, 4, 3, 4, 5, 6, 1,
    // Group-3
    1, 1, 2, 1, 2, 2, 3, 4, 3, 4, 5, 6, 1,
    // clang-format on
  }};
  auto expect_following = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    -1, -1, -2, -1, -1, -2, -1, -2, -1, -2, -2, -3, -1,
    // Group-2
    -1, -1, -2, -1, -1, -2, -1, -2, -1, -2, -2, -3, -1,
    // Group-3
    -1, -1, -2, -1, -1, -2, -1, -2, -1, -2, -2, -3, -1,
    // clang-format on
  }};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<0>(result)->view(), expect_preceding, cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<1>(result)->view(), expect_following, cudf::test::debug_output_level::ALL_ERRORS);

  result = cudf::make_range_windows(cudf::table_view{{group_keys}},
                                    orderby,
                                    cudf::order::DESCENDING,
                                    cudf::null_order::BEFORE,
                                    cudf::bounded_open{*prec},
                                    cudf::bounded_open{*foll});

  expect_preceding = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    1, 1, 2, 1, 1, 2, 2, 3, 3, 4, 3, 4, 1,
    // Group-2
    1, 1, 2, 1, 1, 2, 2, 3, 3, 4, 3, 4, 1,
    // Group-3
    1, 1, 2, 1, 1, 2, 2, 3, 3, 4, 3, 4, 1,
    // clang-format on
  }};
  expect_following = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    -1, -1, -2, -1, -1, -2, -1, -2, -3, -2, -3, -4, -1,
    // Group-2
    -1, -1, -2, -1, -1, -2, -1, -2, -3, -2, -3, -4, -1,
    // Group-3
    -1, -1, -2, -1, -1, -2, -1, -2, -3, -2, -3, -4, -1,
    // clang-format on
  }};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<0>(result)->view(), expect_preceding, cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<1>(result)->view(), expect_following, cudf::test::debug_output_level::ALL_ERRORS);
}

TYPED_TEST(GroupedIntegralRangeWindows, AscendingNullsZeroPrecedingFollowing)
{
  using T         = TypeParam;
  constexpr T min = std::numeric_limits<T>::lowest();
  constexpr T max = std::numeric_limits<T>::max();
  // clang-format off
  auto orderby    = cudf::test::fixed_width_column_wrapper<T>{{
    // Group-1
    min, T{5}, T{5}, T{6}, T{7}, T{9}, T{9}, T{12}, T{13}, T{17}, T{22}, T{22}, max,
    // Group-2
    min, T{5}, T{5}, T{6}, T{7}, T{9}, T{9}, T{12}, T{13}, T{17}, T{22}, T{22}, max,
    // Group-3,
    min, T{5}, T{5}, T{6}, T{7}, T{9}, T{9}, T{12}, T{13}, T{17}, T{22}, T{22}, max,
  }, {
    // Group-1
    false, false, false, false, true, true, true, true, true, true, true, true, true,
    // Group-2
    true, true, true, true, true, true, true, true, true, true, true, true, true,
    // Group-3
    false, false, true, true, true, true, true, true, true, true, true, true, true,
  }};
  // clang-format on

  auto group_keys = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // Group-2
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    // Group-3
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    // clang-format on
  }};

  auto prec = cudf::make_fixed_width_scalar<T>(0);
  auto foll = cudf::make_fixed_width_scalar<T>(0);

  auto result = cudf::make_range_windows(cudf::table_view{{group_keys}},
                                         orderby,
                                         cudf::order::ASCENDING,
                                         cudf::null_order::BEFORE,
                                         cudf::bounded_closed{*prec},
                                         cudf::bounded_closed{*foll});

  auto expect_preceding = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    1, 2, 3, 4, 1, 1, 2, 1, 1, 1, 1, 2, 1,
    // Group-2
    1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1,
    // Group-3
    1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1,
    // clang-format on
  }};
  auto expect_following = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    3, 2, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
    // Group-2
    0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
    // Group-3
    1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
    // clang-format on
  }};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<0>(result)->view(), expect_preceding, cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<1>(result)->view(), expect_following, cudf::test::debug_output_level::ALL_ERRORS);

  result = cudf::make_range_windows(cudf::table_view{{group_keys}},
                                    orderby,
                                    cudf::order::ASCENDING,
                                    cudf::null_order::BEFORE,
                                    cudf::bounded_open{*prec},
                                    cudf::bounded_open{*foll});

  // TODO: What does it mean to be a BOUNDED_OPEN window in presence of nulls?
  // Should the group of nulls be an empty window?
  expect_preceding = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    1, 2, 3, 4, 0, -1, 0, 0, 0, 0, -1, 0, 0,
    // Group-2
    0, -1, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0,
    // Group-3
    1, 2, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0,
    // clang-format off
  }};
  expect_following = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    3, 2, 1, 0, -1, -1, -2, -1, -1, -1, -1, -2, -1,
    // Group-2
    -1, -1, -2, -1, -1, -1, -2, -1, -1, -1, -1, -2, -1,
    // Group-3
    1, 0, -1, -1, -1, -1, -2, -1, -1, -1, -1, -2, -1,
    // clang-format off
  }};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<0>(result)->view(), expect_preceding, cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<1>(result)->view(), expect_following, cudf::test::debug_output_level::ALL_ERRORS);
}


TYPED_TEST(GroupedIntegralRangeWindows, AscendingNullsPositivePrecedingFollowing)
{
  using T         = TypeParam;
  constexpr T min = std::numeric_limits<T>::lowest();
  constexpr T max = std::numeric_limits<T>::max();
  // clang-format off
  auto orderby    = cudf::test::fixed_width_column_wrapper<T>{{
    // Group-1
    min, T{5}, T{5}, T{6}, T{7}, T{9}, T{9}, T{12}, T{13}, T{17}, T{22}, T{22}, max,
    // Group-2
    min, T{5}, T{5}, T{6}, T{7}, T{9}, T{9}, T{12}, T{13}, T{17}, T{22}, T{22}, max,
    // Group-3,
    min, T{5}, T{5}, T{6}, T{7}, T{9}, T{9}, T{12}, T{13}, T{17}, T{22}, T{22}, max,
  }, {
    // Group-1
    false, false, false, false, true, true, true, true, true, true, true, true, true,
    // Group-2
    true, true, true, true, true, true, true, true, true, true, true, true, true,
    // Group-3
    false, false, true, true, true, true, true, true, true, true, true, true, true,
  }};
  // clang-format on

  auto group_keys = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // Group-2
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    // Group-3
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    // clang-format on
  }};

  auto prec = cudf::make_fixed_width_scalar<T>(2);
  auto foll = cudf::make_fixed_width_scalar<T>(1);

  auto result = cudf::make_range_windows(cudf::table_view{{group_keys}},
                                         orderby,
                                         cudf::order::ASCENDING,
                                         cudf::null_order::BEFORE,
                                         cudf::bounded_closed{*prec},
                                         cudf::bounded_closed{*foll});

  auto expect_preceding = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    1, 2, 3, 4, 1, 2, 3, 1, 2, 1, 1, 2, 1,
    // Group-2
    1, 1, 2, 3, 4, 2, 3, 1, 2, 1, 1, 2, 1,
    // Group-3
    1, 2, 1, 2, 3, 2, 3, 1, 2, 1, 1, 2, 1,
    // clang-format off
  }};
  auto expect_following = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    3, 2, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0,
    // Group-2
    0, 2, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0,
    // Group-3
    1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0,
    // clang-format off
  }};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<0>(result)->view(), expect_preceding, cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<1>(result)->view(), expect_following, cudf::test::debug_output_level::ALL_ERRORS);

  result = cudf::make_range_windows(cudf::table_view{{group_keys}},
                                          orderby,
                                          cudf::order::ASCENDING,
                                          cudf::null_order::BEFORE,
                                          cudf::bounded_open{*prec},
                                          cudf::bounded_open{*foll});

  expect_preceding = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    1, 2, 3, 4, 1, 1, 2, 1, 2, 1, 1, 2, 1,
    // Group-2
    1, 1, 2, 3, 2, 1, 2, 1, 2, 1, 1, 2, 1,
    // Group-3
    1, 2, 1, 2, 2, 1, 2, 1, 2, 1, 1, 2, 1
    // clang-format off
  }};
  expect_following = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    3, 2, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
    // Group-2
    0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
    // Group-3
    1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
    // clang-format off
  }};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<0>(result)->view(), expect_preceding, cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<1>(result)->view(), expect_following, cudf::test::debug_output_level::ALL_ERRORS);
}

TYPED_TEST(GroupedSignedIntegralRangeWindows, AscendingNullsNegativePreceding)
{
  using T         = TypeParam;
  constexpr T min = std::numeric_limits<T>::lowest();
  constexpr T max = std::numeric_limits<T>::max();
  // clang-format off
  auto orderby    = cudf::test::fixed_width_column_wrapper<T>{{
    // Group-1
    min, T{5}, T{5}, T{6}, T{7}, T{9}, T{9}, T{12}, T{13}, T{17}, T{22}, T{22}, max,
    // Group-2
    min, T{5}, T{5}, T{6}, T{7}, T{9}, T{9}, T{12}, T{13}, T{17}, T{22}, T{22}, max,
    // Group-3
    min, T{5}, T{5}, T{6}, T{7}, T{9}, T{9}, T{12}, T{13}, T{17}, T{22}, T{22}, max,
  }, {
    // Group-1
    true, true, true, true, true, true, false, false, false, false, false, false, false,
    // Group-2
    true, true, true, true, true, true, true, true, true, true, true, true, true,
    // Group-3
    true, true, true, true, true, true, true, true, true, true, false, false, false,
  }};
  // clang-format on

  auto group_keys = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // Group-2
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    // Group-3
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    // clang-format on
  }};

  auto prec = cudf::make_fixed_width_scalar<T>(-1);
  auto foll = cudf::make_fixed_width_scalar<T>(2);

  auto result = cudf::make_range_windows(cudf::table_view{{group_keys}},
                                         orderby,
                                         cudf::order::ASCENDING,
                                         cudf::null_order::AFTER,
                                         cudf::bounded_closed{*prec},
                                         cudf::bounded_closed{*foll});

  auto expect_preceding = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    0, -1, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7,
    // Group-2
    0, -1, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0,
    // Group-3
    0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 1, 2, 3,
    // clang-format on
  }};
  auto expect_following = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    0, 3, 2, 1, 1, 0, 6, 5, 4, 3, 2, 1, 0,
    // Group-2
    0, 3, 2, 1, 2, 1, 0, 1, 0, 0, 1, 0, 0,
    // Group-3
    0, 3, 2, 1, 2, 1, 0, 1, 0, 0, 2, 1, 0,
    // clang-format on
  }};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<0>(result)->view(), expect_preceding, cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<1>(result)->view(), expect_following, cudf::test::debug_output_level::ALL_ERRORS);

  result = cudf::make_range_windows(cudf::table_view{{group_keys}},
                                    orderby,
                                    cudf::order::ASCENDING,
                                    cudf::null_order::AFTER,
                                    cudf::bounded_open{*prec},
                                    cudf::bounded_open{*foll});

  expect_preceding = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    0, -2, -1, -1, 0, 0, 1, 2, 3, 4, 5, 6, 7,
    // Group-2
    0, -2, -1, -1, 0, -1, 0, -1, 0, 0, -1, 0, 0,
    // Group-3
    0, -2, -1, -1, 0, -1, 0, -1, 0, 0, 1, 2, 3,
    // clang-format on
  }};
  expect_following = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    0, 2, 1, 1, 0, 0, 6, 5, 4, 3, 2, 1, 0,
    // Group-2
    0, 2, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0,
    // Group-3
    0, 2, 1, 1, 0, 1, 0, 1, 0, 0, 2, 1, 0,
    // clang-format on
  }};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<0>(result)->view(), expect_preceding, cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<1>(result)->view(), expect_following, cudf::test::debug_output_level::ALL_ERRORS);
}

TYPED_TEST(GroupedSignedIntegralRangeWindows, AscendingNullsNegativeFollowing)
{
  using T         = TypeParam;
  constexpr T min = std::numeric_limits<T>::lowest();
  constexpr T max = std::numeric_limits<T>::max();
  // clang-format off
  auto orderby    = cudf::test::fixed_width_column_wrapper<T>{{
    // Group-1
    min, T{5}, T{5}, T{6}, T{7}, T{9}, T{9}, T{12}, T{13}, T{17}, T{22}, T{22}, max,
    // Group-2
    min, T{5}, T{5}, T{6}, T{7}, T{9}, T{9}, T{12}, T{13}, T{17}, T{22}, T{22}, max,
    // Group-3
    min, T{5}, T{5}, T{6}, T{7}, T{9}, T{9}, T{12}, T{13}, T{17}, T{22}, T{22}, max,
  }, {
    // Group-1
    true, true, true, true, true, true, false, false, false, false, false, false, false,
    // Group-2
    true, true, true, true, true, true, true, true, true, true, true, true, true,
    // Group-3
    true, true, true, true, true, true, true, true, true, true, false, false, false,
  }};
  // clang-format on

  auto group_keys = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // Group-2
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    // Group-3
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    // clang-format on
  }};

  auto prec = cudf::make_fixed_width_scalar<T>(4);
  auto foll = cudf::make_fixed_width_scalar<T>(-2);

  auto result = cudf::make_range_windows(cudf::table_view{{group_keys}},
                                         orderby,
                                         cudf::order::ASCENDING,
                                         cudf::null_order::AFTER,
                                         cudf::bounded_closed{*prec},
                                         cudf::bounded_closed{*foll});

  auto expect_preceding = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    1, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6, 7,
    // Group-2
    1, 1, 2, 3, 4, 5, 6, 3, 4, 2, 1, 2, 1,
    // Group-3
    1, 1, 2, 3, 4, 5, 6, 3, 4, 2, 1, 2, 3,
    // clang-format on
  }};
  auto expect_following = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    -1, -1, -2, -3, -2, -1, 6, 5, 4, 3, 2, 1, 0,
    // Group-2
    -1, -1, -2, -3, -2, -1, -2, -1, -2, -1, -1, -2, -1,
    // Group-3
    -1, -1, -2, -3, -2, -1, -2, -1, -2, -1, 2, 1, 0,
    // clang-format on
  }};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<0>(result)->view(), expect_preceding, cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<1>(result)->view(), expect_following, cudf::test::debug_output_level::ALL_ERRORS);

  result = cudf::make_range_windows(cudf::table_view{{group_keys}},
                                    orderby,
                                    cudf::order::ASCENDING,
                                    cudf::null_order::AFTER,
                                    cudf::bounded_open{*prec},
                                    cudf::bounded_open{*foll});

  expect_preceding = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    1, 1, 2, 3, 4, 3, 1, 2, 3, 4, 5, 6, 7,
    // Group-2
    1, 1, 2, 3, 4, 3, 4, 3, 2, 1, 1, 2, 1,
    // Group-3
    1, 1, 2, 3, 4, 3, 4, 3, 2, 1, 1, 2, 3,
    // clang-format on
  }};
  expect_following = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    -1, -1, -2, -3, -4, -2, 6, 5, 4, 3, 2, 1, 0,
    // Group-2
    -1, -1, -2, -3, -4, -2, -3, -1, -2, -1, -1, -2, -1,
    // Group-3
    -1, -1, -2, -3, -4, -2, -3, -1, -2, -1, 2, 1, 0,
    // clang-format on
  }};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<0>(result)->view(), expect_preceding, cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<1>(result)->view(), expect_following, cudf::test::debug_output_level::ALL_ERRORS);
}

TYPED_TEST(GroupedIntegralRangeWindows, DescendingNullsZeroPrecedingFollowing)
{
  using T         = TypeParam;
  constexpr T min = std::numeric_limits<T>::lowest();
  constexpr T max = std::numeric_limits<T>::max();
  // clang-format off
  auto orderby    = cudf::test::fixed_width_column_wrapper<T>{{
    // Group-1
    max, T{22}, T{22}, T{17}, T{13}, T{12}, T{9}, T{9}, T{7}, T{6}, T{5}, T{5}, min,
    // Group-2
    max, T{22}, T{22}, T{17}, T{13}, T{12}, T{9}, T{9}, T{7}, T{6}, T{5}, T{5}, min,
    // Group-3
    max, T{22}, T{22}, T{17}, T{13}, T{12}, T{9}, T{9}, T{7}, T{6}, T{5}, T{5}, min,
  }, {
    // Group-1
    true, true, true, true, true, true, true, false, false, false, false, false, false,
    // Group-2
    true, true, true, true, true, true, true, true, true, true, true, true, true,
    // Group-3
    true, true, true, true, true, true, true, true, true, true, false, false, false,
  }};
  // clang-format on

  auto group_keys = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // Group-2
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    // Group-3
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    // clang-format on
  }};

  auto prec = cudf::make_fixed_width_scalar<T>(0);
  auto foll = cudf::make_fixed_width_scalar<T>(0);

  auto result = cudf::make_range_windows(cudf::table_view{{group_keys}},
                                         orderby,
                                         cudf::order::DESCENDING,
                                         cudf::null_order::BEFORE,
                                         cudf::bounded_closed{*prec},
                                         cudf::bounded_closed{*foll});

  auto expect_preceding = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    1, 1, 2, 1, 1, 1, 1, 1, 2, 3, 4, 5, 6,
    // Group-2
    1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1,
    // Group-3
    1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 3,
    // clang-format on
  }};
  auto expect_following = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    0, 1, 0, 0, 0, 0, 0, 5, 4, 3, 2, 1, 0,
    // Group-2
    0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
    // Group-3
    0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 2, 1, 0,
    // clang-format on
  }};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<0>(result)->view(), expect_preceding, cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<1>(result)->view(), expect_following, cudf::test::debug_output_level::ALL_ERRORS);

  result = cudf::make_range_windows(cudf::table_view{{group_keys}},
                                    orderby,
                                    cudf::order::DESCENDING,
                                    cudf::null_order::BEFORE,
                                    cudf::bounded_open{*prec},
                                    cudf::bounded_open{*foll});

  expect_preceding = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    0, -1, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6,
    // Group-2
    0, -1, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0,
    // Group-3
    0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 1, 2, 3,
    // clang-format on
  }};
  expect_following = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    -1, -1, -2, -1, -1, -1, -1, 5, 4, 3, 2, 1, 0,
    // Group-2
    -1, -1, -2, -1, -1, -1, -1, -2, -1, -1, -1, -2, -1,
    // Group-3
    -1, -1, -2, -1, -1, -1, -1, -2, -1, -1, 2, 1, 0,
    // clang-format on
  }};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<0>(result)->view(), expect_preceding, cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<1>(result)->view(), expect_following, cudf::test::debug_output_level::ALL_ERRORS);
}

TYPED_TEST(GroupedIntegralRangeWindows, DescendingNullsPositivePrecedingFollowing)
{
  using T         = TypeParam;
  constexpr T min = std::numeric_limits<T>::lowest();
  constexpr T max = std::numeric_limits<T>::max();
  // clang-format off
  auto orderby    = cudf::test::fixed_width_column_wrapper<T>{{
    // Group-1
    max, T{22}, T{22}, T{17}, T{13}, T{12}, T{9}, T{9}, T{7}, T{6}, T{5}, T{5}, min,
    // Group-2
    max, T{22}, T{22}, T{17}, T{13}, T{12}, T{9}, T{9}, T{7}, T{6}, T{5}, T{5}, min,
    // Group-3
    max, T{22}, T{22}, T{17}, T{13}, T{12}, T{9}, T{9}, T{7}, T{6}, T{5}, T{5}, min,
  }, {
    // Group-1
    true, true, true, true, true, true, true, false, false, false, false, false, false,
    // Group-2
    true, true, true, true, true, true, true, true, true, true, true, true, true,
    // Group-3
    true, true, true, true, true, true, true, true, true, true, false, false, false,
  }};
  // clang-format on

  auto group_keys = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // Group-2
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    // Group-3
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    // clang-format on
  }};

  auto prec = cudf::make_fixed_width_scalar<T>(2);
  auto foll = cudf::make_fixed_width_scalar<T>(1);

  auto result = cudf::make_range_windows(cudf::table_view{{group_keys}},
                                         orderby,
                                         cudf::order::DESCENDING,
                                         cudf::null_order::BEFORE,
                                         cudf::bounded_closed{*prec},
                                         cudf::bounded_closed{*foll});

  auto expect_preceding = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    1, 1, 2, 1, 1, 2, 1, 1, 2, 3, 4, 5, 6,
    // Group-2
    1, 1, 2, 1, 1, 2, 1, 2, 3, 2, 3, 4, 1,
    // Group-3
    1, 1, 2, 1, 1, 2, 1, 2, 3, 2, 1, 2, 3,
    // clang-format on
  }};
  auto expect_following = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    0, 1, 0, 0, 1, 0, 0, 5, 4, 3, 2, 1, 0,
    // Group-2
    0, 1, 0, 0, 1, 0, 1, 0, 1, 2, 1, 0, 0,
    // Group-3
    0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 2, 1, 0,
    // clang-format on
  }};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<0>(result)->view(), expect_preceding, cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<1>(result)->view(), expect_following, cudf::test::debug_output_level::ALL_ERRORS);

  result = cudf::make_range_windows(cudf::table_view{{group_keys}},
                                    orderby,
                                    cudf::order::DESCENDING,
                                    cudf::null_order::BEFORE,
                                    cudf::bounded_open{*prec},
                                    cudf::bounded_open{*foll});

  expect_preceding = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    1, 1, 2, 1, 1, 2, 1, 1, 2, 3, 4, 5, 6,
    // Group-2
    1, 1, 2, 1, 1, 2, 1, 2, 1, 2, 2, 3, 1,
    // Group-3
    1, 1, 2, 1, 1, 2, 1, 2, 1, 2, 1, 2, 3,
    // clang-format on
  }};
  expect_following = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    0, 1, 0, 0, 0, 0, 0, 5, 4, 3, 2, 1, 0,
    // Group-2
    0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
    // Group-3
    0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 2, 1, 0,
    // clang-format on
  }};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<0>(result)->view(), expect_preceding, cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<1>(result)->view(), expect_following, cudf::test::debug_output_level::ALL_ERRORS);
}

TYPED_TEST(GroupedSignedIntegralRangeWindows, DescendingNullsNegativePreceding)
{
  using T         = TypeParam;
  constexpr T min = std::numeric_limits<T>::lowest();
  constexpr T max = std::numeric_limits<T>::max();
  // clang-format off
  auto orderby    = cudf::test::fixed_width_column_wrapper<T>{{
    // Group-1
    max, T{22}, T{22}, T{17}, T{13}, T{12}, T{9}, T{9}, T{7}, T{6}, T{5}, T{5}, min,
    // Group-2
    max, T{22}, T{22}, T{17}, T{13}, T{12}, T{9}, T{9}, T{7}, T{6}, T{5}, T{5}, min,
    // Group-3
    max, T{22}, T{22}, T{17}, T{13}, T{12}, T{9}, T{9}, T{7}, T{6}, T{5}, T{5}, min,
  }, {
    // Group-1
    false, false, false, false, false, true, true, true, true, true, true, true, true,
    // Group-2
    true, true, true, true, true, true, true, true, true, true, true, true, true,
    // Group-3
    false, false, false, false, true, true, true, true, true, true, true, true, true,
  }};
  // clang-format on

  auto group_keys = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // Group-2
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    // Group-3
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    // clang-format on
  }};

  auto prec = cudf::make_fixed_width_scalar<T>(-1);
  auto foll = cudf::make_fixed_width_scalar<T>(2);

  auto result = cudf::make_range_windows(cudf::table_view{{group_keys}},
                                         orderby,
                                         cudf::order::DESCENDING,
                                         cudf::null_order::AFTER,
                                         cudf::bounded_closed{*prec},
                                         cudf::bounded_closed{*foll});

  auto expect_preceding = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    1, 2, 3, 4, 5, 0, -1, 0, 0, 0, -1, 0, 0,
    // Group-2
    0, -1, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0,
    // Group-3
    1, 2, 3, 4, 0, 0, -1, 0, 0, 0, -1, 0, 0,
    // clang-format on
  }};
  auto expect_following = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    4, 3, 2, 1, 0, 0, 2, 1, 3, 2, 1, 0, 0,
    // Group-2
    0, 1, 0, 0, 1, 0, 2, 1, 3, 2, 1, 0, 0,
    // Group-3
    3, 2, 1, 0, 1, 0, 2, 1, 3, 2, 1, 0, 0,
    // clang-format on
  }};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<0>(result)->view(), expect_preceding, cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<1>(result)->view(), expect_following, cudf::test::debug_output_level::ALL_ERRORS);

  result = cudf::make_range_windows(cudf::table_view{{group_keys}},
                                    orderby,
                                    cudf::order::DESCENDING,
                                    cudf::null_order::AFTER,
                                    cudf::bounded_open{*prec},
                                    cudf::bounded_open{*foll});

  expect_preceding = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    1, 2, 3, 4, 5, 0, -1, 0, -1, -2, -1, 0, 0,
    // Group-2
    0, -1, 0, 0, -1, 0, -1, 0, -1, -2, -1, 0, 0,
    // Group-3
    1, 2, 3, 4, -1, 0, -1, 0, -1, -2, -1, 0, 0,
    // clang-format on
  }};
  expect_following = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    4, 3, 2, 1, 0, 0, 1, 0, 1, 2, 1, 0, 0,
    // Group-2
    0, 1, 0, 0, 1, 0, 1, 0, 1, 2, 1, 0, 0,
    // Group-3
    3, 2, 1, 0, 1, 0, 1, 0, 1, 2, 1, 0, 0,
    // clang-format on
  }};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<0>(result)->view(), expect_preceding, cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<1>(result)->view(), expect_following, cudf::test::debug_output_level::ALL_ERRORS);
}

TYPED_TEST(GroupedSignedIntegralRangeWindows, DescendingNullsNegativeFollowing)
{
  using T         = TypeParam;
  constexpr T min = std::numeric_limits<T>::lowest();
  constexpr T max = std::numeric_limits<T>::max();
  // clang-format off
  auto orderby    = cudf::test::fixed_width_column_wrapper<T>{{
    // Group-1
    max, T{22}, T{22}, T{17}, T{13}, T{12}, T{9}, T{9}, T{7}, T{6}, T{5}, T{5}, min,
    // Group-2
    max, T{22}, T{22}, T{17}, T{13}, T{12}, T{9}, T{9}, T{7}, T{6}, T{5}, T{5}, min,
    // Group-3
    max, T{22}, T{22}, T{17}, T{13}, T{12}, T{9}, T{9}, T{7}, T{6}, T{5}, T{5}, min,
  }, {
    // Group-1
    false, false, false, false, false, true, true, true, true, true, true, true, true,
    // Group-2
    true, true, true, true, true, true, true, true, true, true, true, true, true,
    // Group-3
    false, false, false, false, true, true, true, true, true, true, true, true, true,
  }};
  // clang-format on

  auto group_keys = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // Group-2
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    // Group-3
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    // clang-format on
  }};

  auto prec = cudf::make_fixed_width_scalar<T>(4);
  auto foll = cudf::make_fixed_width_scalar<T>(-2);

  auto result = cudf::make_range_windows(cudf::table_view{{group_keys}},
                                         orderby,
                                         cudf::order::DESCENDING,
                                         cudf::null_order::AFTER,
                                         cudf::bounded_closed{*prec},
                                         cudf::bounded_closed{*foll});

  auto expect_preceding = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    1, 2, 3, 4, 5, 1, 2, 3, 3, 4, 5, 6, 1,
    // Group-2
    1, 1, 2, 1, 2, 2, 3, 4, 3, 4, 5, 6, 1,
    // Group-3
    1, 2, 3, 4, 1, 2, 3, 4, 3, 4, 5, 6, 1,
    // clang-format on
  }};
  auto expect_following = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    4, 3, 2, 1, 0, -1, -1, -2, -1, -2, -2, -3, -1,
    // Group-2
    -1, -1, -2, -1, -1, -2, -1, -2, -1, -2, -2, -3, -1,
    // Group-3
    3, 2, 1, 0, -1, -2, -1, -2, -1, -2, -2, -3, -1,
    // clang-format on
  }};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<0>(result)->view(), expect_preceding, cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<1>(result)->view(), expect_following, cudf::test::debug_output_level::ALL_ERRORS);

  result = cudf::make_range_windows(cudf::table_view{{group_keys}},
                                    orderby,
                                    cudf::order::DESCENDING,
                                    cudf::null_order::AFTER,
                                    cudf::bounded_open{*prec},
                                    cudf::bounded_open{*foll});

  expect_preceding = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    1, 2, 3, 4, 5, 1, 2, 3, 3, 4, 3, 4, 1,
    // Group-2
    1, 1, 2, 1, 1, 2, 2, 3, 3, 4, 3, 4, 1,
    // Group-3
    1, 2, 3, 4, 1, 2, 2, 3, 3, 4, 3, 4, 1,
    // clang-format on
  }};
  expect_following = cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
    // clang-format off
    // Group-1
    4, 3, 2, 1, 0, -1, -1, -2, -3, -2, -3, -4, -1,
    // Group-2
    -1, -1, -2, -1, -1, -2, -1, -2, -3, -2, -3, -4, -1,
    // Group-3
    3, 2, 1, 0, -1, -2, -1, -2, -3, -2, -3, -4, -1,
    // clang-format on
  }};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<0>(result)->view(), expect_preceding, cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<1>(result)->view(), expect_following, cudf::test::debug_output_level::ALL_ERRORS);
}
