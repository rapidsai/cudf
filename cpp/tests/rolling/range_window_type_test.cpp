/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/random.hpp>
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/rolling.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/table/table_view.hpp>

#include <limits>

template <typename T>
struct UngroupedBase : cudf::test::BaseFixture {
  static constexpr T min{std::numeric_limits<T>::lowest()};
  static constexpr T max{std::numeric_limits<T>::max()};
  using cw_t = cudf::test::fixed_width_column_wrapper<T>;
  cw_t ascending_no_nulls{};
  cw_t ascending_nulls_before{};
  cw_t ascending_nulls_after{};
  cw_t descending_no_nulls{};
  cw_t descending_nulls_before{};
  cw_t descending_nulls_after{};
  UngroupedBase()
    : ascending_no_nulls{
        // clang-format off
        {min, T{5}, T{5}, T{6}, T{7}, T{9}, T{9}, T{12}, T{13}, T{17}, T{22}, T{22}, max}},
        // clang-format on
      ascending_nulls_before{
        {min, T{5}, T{5}, T{6}, T{7}, T{9}, T{9}, T{12}, T{13}, T{17}, T{22}, T{22}, max},
        {false, false, false, false, true, true, true, true, true, true, true, true, true}},
      ascending_nulls_after{
        {min, T{5}, T{5}, T{6}, T{7}, T{9}, T{9}, T{12}, T{13}, T{17}, T{22}, T{22}, max},
        {true, true, true, true, true, true, false, false, false, false, false, false, false}},
      descending_no_nulls{
        {max, T{22}, T{22}, T{17}, T{13}, T{12}, T{9}, T{9}, T{7}, T{6}, T{5}, T{5}, min}},
      descending_nulls_before{
        {max, T{22}, T{22}, T{17}, T{13}, T{12}, T{9}, T{9}, T{7}, T{6}, T{5}, T{5}, min},
        {true, true, true, true, true, true, true, false, false, false, false, false, false}},
      descending_nulls_after{
        {max, T{22}, T{22}, T{17}, T{13}, T{12}, T{9}, T{9}, T{7}, T{6}, T{5}, T{5}, min},
        {false, false, false, false, false, true, true, true, true, true, true, true, true}}
  {
  }

  void run_test(cudf::column_view const& orderby,
                cudf::order order,
                cudf::null_order null_order,
                cudf::range_window_type preceding,
                cudf::range_window_type following,
                cudf::column_view const& expect_preceding,
                cudf::column_view const& expect_following)
  {
    auto result = cudf::make_range_windows(
      cudf::table_view{}, orderby, order, null_order, preceding, following);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(
      std::get<0>(result)->view(), expect_preceding, cudf::test::debug_output_level::ALL_ERRORS);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(
      std::get<1>(result)->view(), expect_following, cudf::test::debug_output_level::ALL_ERRORS);
  }
};
template <typename T>
struct UngroupedIntegralRangeWindows : UngroupedBase<T> {};
TYPED_TEST_SUITE(UngroupedIntegralRangeWindows, cudf::test::IntegralTypesNotBool);

template <typename T>
struct UngroupedSignedIntegralRangeWindows : UngroupedBase<T> {};
using SignedIntegralTypes = cudf::test::Types<int8_t, int16_t, int32_t, int64_t>;
TYPED_TEST_SUITE(UngroupedSignedIntegralRangeWindows, SignedIntegralTypes);

TYPED_TEST(UngroupedIntegralRangeWindows, AscendingNoNullsZeroPrecedingFollowing)
{
  using T   = TypeParam;
  auto prec = cudf::make_fixed_width_scalar<T>(0);
  auto foll = cudf::make_fixed_width_scalar<T>(0);

  this->run_test(this->ascending_no_nulls,
                 cudf::order::ASCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_closed{*prec},
                 cudf::bounded_closed{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1}},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0}});
  this->run_test(this->ascending_no_nulls,
                 cudf::order::ASCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_open{*prec},
                 cudf::bounded_open{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {0, -1, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0}},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {-1, -1, -2, -1, -1, -1, -2, -1, -1, -1, -1, -2, -1}});
}

TYPED_TEST(UngroupedIntegralRangeWindows, AscendingNoNullsPositivePrecedingFollowing)
{
  using T   = TypeParam;
  auto prec = cudf::make_fixed_width_scalar<T>(2);
  auto foll = cudf::make_fixed_width_scalar<T>(1);

  this->run_test(this->ascending_no_nulls,
                 cudf::order::ASCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_closed{*prec},
                 cudf::bounded_closed{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {1, 1, 2, 3, 4, 2, 3, 1, 2, 1, 1, 2, 1}},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {0, 2, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0}});
  this->run_test(this->ascending_no_nulls,
                 cudf::order::ASCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_open{*prec},
                 cudf::bounded_open{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {1, 1, 2, 3, 2, 1, 2, 1, 2, 1, 1, 2, 1}},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0}});
}

TYPED_TEST(UngroupedSignedIntegralRangeWindows, AscendingNoNullsNegativePreceding)
{
  using T   = TypeParam;
  auto prec = cudf::make_fixed_width_scalar<T>(-1);
  auto foll = cudf::make_fixed_width_scalar<T>(2);

  this->run_test(this->ascending_no_nulls,
                 cudf::order::ASCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_closed{*prec},
                 cudf::bounded_closed{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {0, -1, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0}},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {0, 3, 2, 1, 2, 1, 0, 1, 0, 0, 1, 0, 0}});
  this->run_test(this->ascending_no_nulls,
                 cudf::order::ASCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_open{*prec},
                 cudf::bounded_open{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {0, -2, -1, -1, 0, -1, 0, -1, 0, 0, -1, 0, 0}},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {0, 2, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0}});
}

TYPED_TEST(UngroupedSignedIntegralRangeWindows, AscendingNoNullsNegativeFollowing)
{
  using T   = TypeParam;
  auto prec = cudf::make_fixed_width_scalar<T>(4);
  auto foll = cudf::make_fixed_width_scalar<T>(-2);

  this->run_test(this->ascending_no_nulls,
                 cudf::order::ASCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_closed{*prec},
                 cudf::bounded_closed{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {1, 1, 2, 3, 4, 5, 6, 3, 4, 2, 1, 2, 1}},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {-1, -1, -2, -3, -2, -1, -2, -1, -2, -1, -1, -2, -1}});
  this->run_test(this->ascending_no_nulls,
                 cudf::order::ASCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_open{*prec},
                 cudf::bounded_open{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {1, 1, 2, 3, 4, 3, 4, 3, 2, 1, 1, 2, 1}},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {-1, -1, -2, -3, -4, -2, -3, -1, -2, -1, -1, -2, -1}});
}

TYPED_TEST(UngroupedIntegralRangeWindows, DescendingNoNullsZeroPrecedingFollowing)
{
  using T   = TypeParam;
  auto prec = cudf::make_fixed_width_scalar<T>(0);
  auto foll = cudf::make_fixed_width_scalar<T>(0);

  this->run_test(this->descending_no_nulls,
                 cudf::order::DESCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_closed{*prec},
                 cudf::bounded_closed{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1}},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0}});
  this->run_test(this->descending_no_nulls,
                 cudf::order::DESCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_open{*prec},
                 cudf::bounded_open{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {0, -1, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0}},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {-1, -1, -2, -1, -1, -1, -1, -2, -1, -1, -1, -2, -1}});
}

TYPED_TEST(UngroupedIntegralRangeWindows, DescendingNoNullsPositivePrecedingFollowing)
{
  using T   = TypeParam;
  auto prec = cudf::make_fixed_width_scalar<T>(2);
  auto foll = cudf::make_fixed_width_scalar<T>(1);

  this->run_test(this->descending_no_nulls,
                 cudf::order::DESCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_closed{*prec},
                 cudf::bounded_closed{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {1, 1, 2, 1, 1, 2, 1, 2, 3, 2, 3, 4, 1}},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {0, 1, 0, 0, 1, 0, 1, 0, 1, 2, 1, 0, 0}});
  this->run_test(this->descending_no_nulls,
                 cudf::order::DESCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_open{*prec},
                 cudf::bounded_open{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {1, 1, 2, 1, 1, 2, 1, 2, 1, 2, 2, 3, 1}},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0}});
}

TYPED_TEST(UngroupedSignedIntegralRangeWindows, DescendingNoNullsNegativePreceding)
{
  using T   = TypeParam;
  auto prec = cudf::make_fixed_width_scalar<T>(-1);
  auto foll = cudf::make_fixed_width_scalar<T>(2);

  this->run_test(this->descending_no_nulls,
                 cudf::order::DESCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_closed{*prec},
                 cudf::bounded_closed{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {0, -1, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0}},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {0, 1, 0, 0, 1, 0, 2, 1, 3, 2, 1, 0, 0}});
  this->run_test(this->descending_no_nulls,
                 cudf::order::DESCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_open{*prec},
                 cudf::bounded_open{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {0, -1, 0, 0, -1, 0, -1, 0, -1, -2, -1, 0, 0}},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {0, 1, 0, 0, 1, 0, 1, 0, 1, 2, 1, 0, 0}});
}

TYPED_TEST(UngroupedSignedIntegralRangeWindows, DescendingNoNullsNegativeFollowing)
{
  using T   = TypeParam;
  auto prec = cudf::make_fixed_width_scalar<T>(4);
  auto foll = cudf::make_fixed_width_scalar<T>(-2);

  this->run_test(this->descending_no_nulls,
                 cudf::order::DESCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_closed{*prec},
                 cudf::bounded_closed{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {1, 1, 2, 1, 2, 2, 3, 4, 3, 4, 5, 6, 1}},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {-1, -1, -2, -1, -1, -2, -1, -2, -1, -2, -2, -3, -1}});
  this->run_test(this->descending_no_nulls,
                 cudf::order::DESCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_open{*prec},
                 cudf::bounded_open{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {1, 1, 2, 1, 1, 2, 2, 3, 3, 4, 3, 4, 1}},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {-1, -1, -2, -1, -1, -2, -1, -2, -3, -2, -3, -4, -1}});
}

TYPED_TEST(UngroupedIntegralRangeWindows, AscendingNullsZeroPrecedingFollowing)
{
  using T   = TypeParam;
  auto prec = cudf::make_fixed_width_scalar<T>(0);
  auto foll = cudf::make_fixed_width_scalar<T>(0);

  this->run_test(this->ascending_nulls_before,
                 cudf::order::ASCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_closed{*prec},
                 cudf::bounded_closed{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {1, 2, 3, 4, 1, 1, 2, 1, 1, 1, 1, 2, 1}},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {3, 2, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0}});
  this->run_test(this->ascending_nulls_before,
                 cudf::order::ASCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_open{*prec},
                 cudf::bounded_open{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {1, 2, 3, 4, 0, -1, 0, 0, 0, 0, -1, 0, 0}},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {3, 2, 1, 0, -1, -1, -2, -1, -1, -1, -1, -2, -1}});
}

TYPED_TEST(UngroupedIntegralRangeWindows, AscendingNullsPositivePrecedingFollowing)
{
  using T   = TypeParam;
  auto prec = cudf::make_fixed_width_scalar<T>(2);
  auto foll = cudf::make_fixed_width_scalar<T>(1);

  this->run_test(this->ascending_nulls_before,
                 cudf::order::ASCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_closed{*prec},
                 cudf::bounded_closed{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {1, 2, 3, 4, 1, 2, 3, 1, 2, 1, 1, 2, 1}},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {3, 2, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0}});
  this->run_test(this->ascending_nulls_before,
                 cudf::order::ASCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_open{*prec},
                 cudf::bounded_open{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {1, 2, 3, 4, 1, 1, 2, 1, 2, 1, 1, 2, 1}},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {3, 2, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0}});
}

TYPED_TEST(UngroupedSignedIntegralRangeWindows, AscendingNullsNegativePreceding)
{
  using T   = TypeParam;
  auto prec = cudf::make_fixed_width_scalar<T>(-1);
  auto foll = cudf::make_fixed_width_scalar<T>(2);

  this->run_test(this->ascending_nulls_after,
                 cudf::order::ASCENDING,
                 cudf::null_order::AFTER,
                 cudf::bounded_closed{*prec},
                 cudf::bounded_closed{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {0, -1, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7}},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {0, 3, 2, 1, 1, 0, 6, 5, 4, 3, 2, 1, 0}});
  this->run_test(this->ascending_nulls_after,
                 cudf::order::ASCENDING,
                 cudf::null_order::AFTER,
                 cudf::bounded_open{*prec},
                 cudf::bounded_open{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {0, -2, -1, -1, 0, 0, 1, 2, 3, 4, 5, 6, 7}},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {0, 2, 1, 1, 0, 0, 6, 5, 4, 3, 2, 1, 0}});
}

TYPED_TEST(UngroupedSignedIntegralRangeWindows, AscendingNullsNegativeFollowing)
{
  using T   = TypeParam;
  auto prec = cudf::make_fixed_width_scalar<T>(4);
  auto foll = cudf::make_fixed_width_scalar<T>(-2);

  this->run_test(this->ascending_nulls_after,
                 cudf::order::ASCENDING,
                 cudf::null_order::AFTER,
                 cudf::bounded_closed{*prec},
                 cudf::bounded_closed{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {1, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6, 7}},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {-1, -1, -2, -3, -2, -1, 6, 5, 4, 3, 2, 1, 0}});
  this->run_test(this->ascending_nulls_after,
                 cudf::order::ASCENDING,
                 cudf::null_order::AFTER,
                 cudf::bounded_open{*prec},
                 cudf::bounded_open{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {1, 1, 2, 3, 4, 3, 1, 2, 3, 4, 5, 6, 7}},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {-1, -1, -2, -3, -4, -2, 6, 5, 4, 3, 2, 1, 0}});
}

TYPED_TEST(UngroupedIntegralRangeWindows, DescendingNullsZeroPrecedingFollowing)
{
  using T   = TypeParam;
  auto prec = cudf::make_fixed_width_scalar<T>(0);
  auto foll = cudf::make_fixed_width_scalar<T>(0);

  this->run_test(this->descending_nulls_before,
                 cudf::order::DESCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_closed{*prec},
                 cudf::bounded_closed{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {1, 1, 2, 1, 1, 1, 1, 1, 2, 3, 4, 5, 6}},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {0, 1, 0, 0, 0, 0, 0, 5, 4, 3, 2, 1, 0}});
  this->run_test(this->descending_nulls_before,
                 cudf::order::DESCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_open{*prec},
                 cudf::bounded_open{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {0, -1, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6}},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {-1, -1, -2, -1, -1, -1, -1, 5, 4, 3, 2, 1, 0}});
}

TYPED_TEST(UngroupedIntegralRangeWindows, DescendingNullsPositivePrecedingFollowing)
{
  using T   = TypeParam;
  auto prec = cudf::make_fixed_width_scalar<T>(2);
  auto foll = cudf::make_fixed_width_scalar<T>(1);

  this->run_test(this->descending_nulls_before,
                 cudf::order::DESCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_closed{*prec},
                 cudf::bounded_closed{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {1, 1, 2, 1, 1, 2, 1, 1, 2, 3, 4, 5, 6}},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {0, 1, 0, 0, 1, 0, 0, 5, 4, 3, 2, 1, 0}});
  this->run_test(this->descending_nulls_before,
                 cudf::order::DESCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_open{*prec},
                 cudf::bounded_open{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {1, 1, 2, 1, 1, 2, 1, 1, 2, 3, 4, 5, 6}},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {0, 1, 0, 0, 0, 0, 0, 5, 4, 3, 2, 1, 0}});
}

TYPED_TEST(UngroupedSignedIntegralRangeWindows, DescendingNullsNegativePreceding)
{
  using T   = TypeParam;
  auto prec = cudf::make_fixed_width_scalar<T>(-1);
  auto foll = cudf::make_fixed_width_scalar<T>(2);

  this->run_test(this->descending_nulls_after,
                 cudf::order::DESCENDING,
                 cudf::null_order::AFTER,
                 cudf::bounded_closed{*prec},
                 cudf::bounded_closed{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {1, 2, 3, 4, 5, 0, -1, 0, 0, 0, -1, 0, 0}},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {4, 3, 2, 1, 0, 0, 2, 1, 3, 2, 1, 0, 0}});
  this->run_test(this->descending_nulls_after,
                 cudf::order::DESCENDING,
                 cudf::null_order::AFTER,
                 cudf::bounded_open{*prec},
                 cudf::bounded_open{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {1, 2, 3, 4, 5, 0, -1, 0, -1, -2, -1, 0, 0}},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {4, 3, 2, 1, 0, 0, 1, 0, 1, 2, 1, 0, 0}});
}

TYPED_TEST(UngroupedSignedIntegralRangeWindows, DescendingNullsNegativeFollowing)
{
  using T   = TypeParam;
  auto prec = cudf::make_fixed_width_scalar<T>(4);
  auto foll = cudf::make_fixed_width_scalar<T>(-2);

  this->run_test(this->descending_nulls_after,
                 cudf::order::DESCENDING,
                 cudf::null_order::AFTER,
                 cudf::bounded_closed{*prec},
                 cudf::bounded_closed{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {1, 2, 3, 4, 5, 1, 2, 3, 3, 4, 5, 6, 1}},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {4, 3, 2, 1, 0, -1, -1, -2, -1, -2, -2, -3, -1}});
  this->run_test(this->descending_nulls_after,
                 cudf::order::DESCENDING,
                 cudf::null_order::AFTER,
                 cudf::bounded_open{*prec},
                 cudf::bounded_open{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {1, 2, 3, 4, 5, 1, 2, 3, 3, 4, 3, 4, 1}},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {4, 3, 2, 1, 0, -1, -1, -2, -3, -2, -3, -4, -1}});
}

template <typename T>
struct GroupedBase : cudf::test::BaseFixture {
  static constexpr T min{std::numeric_limits<T>::lowest()};
  static constexpr T max{std::numeric_limits<T>::max()};
  using cw_t = cudf::test::fixed_width_column_wrapper<T>;
  cw_t group_keys{};
  cw_t ascending_no_nulls{};
  cw_t ascending_nulls_before{};
  cw_t ascending_nulls_after{};
  cw_t descending_no_nulls{};
  cw_t descending_nulls_before{};
  cw_t descending_nulls_after{};
  GroupedBase()
    // clang-format off
    : group_keys{{
        // Group-1
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // Group-2
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        // Group-3
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
      }},
      ascending_no_nulls{{
        // Group-1
        min, T{5}, T{5}, T{6}, T{7}, T{9}, T{9}, T{12}, T{13}, T{17}, T{22}, T{22}, max,
        // Group-2
        min, T{5}, T{5}, T{6}, T{7}, T{9}, T{9}, T{12}, T{13}, T{17}, T{22}, T{22}, max,
        // Group-3
        min, T{5}, T{5}, T{6}, T{7}, T{9}, T{9}, T{12}, T{13}, T{17}, T{22}, T{22}, max,
      }},
      ascending_nulls_before{{
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
      }},
      ascending_nulls_after{{
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
      }},
      descending_no_nulls{{
        // Group-1
        max, T{22}, T{22}, T{17}, T{13}, T{12}, T{9}, T{9}, T{7}, T{6}, T{5}, T{5}, min,
        // Group-2
        max, T{22}, T{22}, T{17}, T{13}, T{12}, T{9}, T{9}, T{7}, T{6}, T{5}, T{5}, min,
        // Group-3
        max, T{22}, T{22}, T{17}, T{13}, T{12}, T{9}, T{9}, T{7}, T{6}, T{5}, T{5}, min,
      }},
      descending_nulls_before{{
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
      }},
      descending_nulls_after{{
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
      }}
  // clang-format on
  {
  }

  void run_test(cudf::column_view const& orderby,
                cudf::order order,
                cudf::null_order null_order,
                cudf::range_window_type preceding,
                cudf::range_window_type following,
                cudf::column_view const& expect_preceding,
                cudf::column_view const& expect_following)
  {
    auto result = cudf::make_range_windows(
      cudf::table_view{{this->group_keys}}, orderby, order, null_order, preceding, following);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(
      std::get<0>(result)->view(), expect_preceding, cudf::test::debug_output_level::ALL_ERRORS);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(
      std::get<1>(result)->view(), expect_following, cudf::test::debug_output_level::ALL_ERRORS);
  }
};

template <typename T>
struct GroupedIntegralRangeWindows : GroupedBase<T> {};
TYPED_TEST_SUITE(GroupedIntegralRangeWindows, cudf::test::IntegralTypesNotBool);

template <typename T>
struct GroupedSignedIntegralRangeWindows : GroupedBase<T> {};
TYPED_TEST_SUITE(GroupedSignedIntegralRangeWindows, SignedIntegralTypes);

TYPED_TEST(GroupedIntegralRangeWindows, AscendingNoNullsZeroPrecedingFollowing)
{
  using T   = TypeParam;
  auto prec = cudf::make_fixed_width_scalar<T>(0);
  auto foll = cudf::make_fixed_width_scalar<T>(0);

  this->run_test(this->ascending_no_nulls,
                 cudf::order::ASCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_closed{*prec},
                 cudf::bounded_closed{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1,
                   // Group-2
                   1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1,
                   // Group-3
                   1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1,
                   // clang-format on
                 }},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
                   // Group-2
                   0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
                   // Group-3
                   0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
                   // clang-format on
                 }});
  this->run_test(this->ascending_no_nulls,
                 cudf::order::ASCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_open{*prec},
                 cudf::bounded_open{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   0, -1, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0,
                   // Group-2
                   0, -1, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0,
                   // Group-3
                   0, -1, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0,
                   // clang-format on
                 }},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   -1, -1, -2, -1, -1, -1, -2, -1, -1, -1, -1, -2, -1,
                   // Group-2
                   -1, -1, -2, -1, -1, -1, -2, -1, -1, -1, -1, -2, -1,
                   // Group-3
                   -1, -1, -2, -1, -1, -1, -2, -1, -1, -1, -1, -2, -1,
                   // clang-format on
                 }});
}

TYPED_TEST(GroupedIntegralRangeWindows, AscendingNoNullsPositivePrecedingFollowing)
{
  using T   = TypeParam;
  auto prec = cudf::make_fixed_width_scalar<T>(2);
  auto foll = cudf::make_fixed_width_scalar<T>(1);

  this->run_test(this->ascending_no_nulls,
                 cudf::order::ASCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_closed{*prec},
                 cudf::bounded_closed{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   1, 1, 2, 3, 4, 2, 3, 1, 2, 1, 1, 2, 1,
                   // Group-2
                   1, 1, 2, 3, 4, 2, 3, 1, 2, 1, 1, 2, 1,
                   // Group-3
                   1, 1, 2, 3, 4, 2, 3, 1, 2, 1, 1, 2, 1,
                   // clang-format on
                 }},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   0, 2, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0,
                   // Group-2
                   0, 2, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0,
                   // Group-3
                   0, 2, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0,
                   // clang-format on
                 }});
  this->run_test(this->ascending_no_nulls,
                 cudf::order::ASCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_open{*prec},
                 cudf::bounded_open{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   1, 1, 2, 3, 2, 1, 2, 1, 2, 1, 1, 2, 1,
                   // Group-2
                   1, 1, 2, 3, 2, 1, 2, 1, 2, 1, 1, 2, 1,
                   // Group-3
                   1, 1, 2, 3, 2, 1, 2, 1, 2, 1, 1, 2, 1,
                   // clang-format on
                 }},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
                   // Group-2
                   0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
                   // Group-3
                   0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
                   // clang-format on
                 }});
}

TYPED_TEST(GroupedSignedIntegralRangeWindows, AscendingNoNullsNegativePreceding)
{
  using T   = TypeParam;
  auto prec = cudf::make_fixed_width_scalar<T>(-1);
  auto foll = cudf::make_fixed_width_scalar<T>(2);

  this->run_test(this->ascending_no_nulls,
                 cudf::order::ASCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_closed{*prec},
                 cudf::bounded_closed{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   0, -1, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0,
                   // Group-2
                   0, -1, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0,
                   // Group-3
                   0, -1, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0,
                   // clang-format on
                 }},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   0, 3, 2, 1, 2, 1, 0, 1, 0, 0, 1, 0, 0,
                   // Group-2
                   0, 3, 2, 1, 2, 1, 0, 1, 0, 0, 1, 0, 0,
                   // Group-3
                   0, 3, 2, 1, 2, 1, 0, 1, 0, 0, 1, 0, 0,
                   // clang-format on
                 }});
  this->run_test(this->ascending_no_nulls,
                 cudf::order::ASCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_open{*prec},
                 cudf::bounded_open{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   0, -2, -1, -1, 0, -1, 0, -1, 0, 0, -1, 0, 0,
                   // Group-2
                   0, -2, -1, -1, 0, -1, 0, -1, 0, 0, -1, 0, 0,
                   // Group-3
                   0, -2, -1, -1, 0, -1, 0, -1, 0, 0, -1, 0, 0,
                   // clang-format on
                 }},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   0, 2, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0,
                   // Group-2
                   0, 2, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0,
                   // Group-3
                   0, 2, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0,
                   // clang-format on
                 }});
}

TYPED_TEST(GroupedSignedIntegralRangeWindows, AscendingNoNullsNegativeFollowing)
{
  using T   = TypeParam;
  auto prec = cudf::make_fixed_width_scalar<T>(4);
  auto foll = cudf::make_fixed_width_scalar<T>(-2);

  this->run_test(this->ascending_no_nulls,
                 cudf::order::ASCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_closed{*prec},
                 cudf::bounded_closed{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   1, 1, 2, 3, 4, 5, 6, 3, 4, 2, 1, 2, 1,
                   // Group-2
                   1, 1, 2, 3, 4, 5, 6, 3, 4, 2, 1, 2, 1,
                   // Group-3
                   1, 1, 2, 3, 4, 5, 6, 3, 4, 2, 1, 2, 1,
                   // clang-format on
                 }},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   -1, -1, -2, -3, -2, -1, -2, -1, -2, -1, -1, -2, -1,
                   // Group-2
                   -1, -1, -2, -3, -2, -1, -2, -1, -2, -1, -1, -2, -1,
                   // Group-3
                   -1, -1, -2, -3, -2, -1, -2, -1, -2, -1, -1, -2, -1,
                   // clang-format on
                 }});
  this->run_test(this->ascending_no_nulls,
                 cudf::order::ASCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_open{*prec},
                 cudf::bounded_open{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   1, 1, 2, 3, 4, 3, 4, 3, 2, 1, 1, 2, 1,
                   // Group-2
                   1, 1, 2, 3, 4, 3, 4, 3, 2, 1, 1, 2, 1,
                   // Group-3
                   1, 1, 2, 3, 4, 3, 4, 3, 2, 1, 1, 2, 1,
                   // clang-format on
                 }},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   -1, -1, -2, -3, -4, -2, -3, -1, -2, -1, -1, -2, -1,
                   // Group-2
                   -1, -1, -2, -3, -4, -2, -3, -1, -2, -1, -1, -2, -1,
                   // Group-3
                   -1, -1, -2, -3, -4, -2, -3, -1, -2, -1, -1, -2, -1,
                   // clang-format on
                 }});
}

TYPED_TEST(GroupedIntegralRangeWindows, DescendingNoNullsZeroPrecedingFollowing)
{
  using T   = TypeParam;
  auto prec = cudf::make_fixed_width_scalar<T>(0);
  auto foll = cudf::make_fixed_width_scalar<T>(0);

  this->run_test(this->descending_no_nulls,
                 cudf::order::DESCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_closed{*prec},
                 cudf::bounded_closed{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1,
                   // Group-2
                   1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1,
                   // Group-3
                   1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1,
                   // clang-format on
                 }},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
                   // Group-2
                   0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
                   // Group-3
                   0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
                   // clang-format on
                 }});
  this->run_test(this->descending_no_nulls,
                 cudf::order::DESCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_open{*prec},
                 cudf::bounded_open{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   0, -1, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0,
                   // Group-2
                   0, -1, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0,
                   // Group-3
                   0, -1, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0,
                   // clang-format on
                 }},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   -1, -1, -2, -1, -1, -1, -1, -2, -1, -1, -1, -2, -1,
                   // Group-2
                   -1, -1, -2, -1, -1, -1, -1, -2, -1, -1, -1, -2, -1,
                   // Group-3
                   -1, -1, -2, -1, -1, -1, -1, -2, -1, -1, -1, -2, -1,
                   // clang-format on
                 }});
}

TYPED_TEST(GroupedIntegralRangeWindows, DescendingNoNullsPositivePrecedingFollowing)
{
  using T   = TypeParam;
  auto prec = cudf::make_fixed_width_scalar<T>(2);
  auto foll = cudf::make_fixed_width_scalar<T>(1);

  this->run_test(this->descending_no_nulls,
                 cudf::order::DESCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_closed{*prec},
                 cudf::bounded_closed{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   1, 1, 2, 1, 1, 2, 1, 2, 3, 2, 3, 4, 1,
                   // Group-2
                   1, 1, 2, 1, 1, 2, 1, 2, 3, 2, 3, 4, 1,
                   // Group-3
                   1, 1, 2, 1, 1, 2, 1, 2, 3, 2, 3, 4, 1,
                   // clang-format on
                 }},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   0, 1, 0, 0, 1, 0, 1, 0, 1, 2, 1, 0, 0,
                   // Group-2
                   0, 1, 0, 0, 1, 0, 1, 0, 1, 2, 1, 0, 0,
                   // Group-3
                   0, 1, 0, 0, 1, 0, 1, 0, 1, 2, 1, 0, 0,
                   // clang-format on
                 }});
  this->run_test(this->descending_no_nulls,
                 cudf::order::DESCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_open{*prec},
                 cudf::bounded_open{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   1, 1, 2, 1, 1, 2, 1, 2, 1, 2, 2, 3, 1,
                   // Group-2
                   1, 1, 2, 1, 1, 2, 1, 2, 1, 2, 2, 3, 1,
                   // Group-3
                   1, 1, 2, 1, 1, 2, 1, 2, 1, 2, 2, 3, 1,
                   // clang-format on
                 }},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
                   // Group-2
                   0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
                   // Group-3
                   0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
                   // clang-format on
                 }});
}

TYPED_TEST(GroupedSignedIntegralRangeWindows, DescendingNoNullsNegativePreceding)
{
  using T   = TypeParam;
  auto prec = cudf::make_fixed_width_scalar<T>(-1);
  auto foll = cudf::make_fixed_width_scalar<T>(2);

  this->run_test(this->descending_no_nulls,
                 cudf::order::DESCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_closed{*prec},
                 cudf::bounded_closed{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   0, -1, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0,
                   // Group-2
                   0, -1, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0,
                   // Group-3
                   0, -1, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0,
                   // clang-format on
                 }},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   0, 1, 0, 0, 1, 0, 2, 1, 3, 2, 1, 0, 0,
                   // Group-2
                   0, 1, 0, 0, 1, 0, 2, 1, 3, 2, 1, 0, 0,
                   // Group-3
                   0, 1, 0, 0, 1, 0, 2, 1, 3, 2, 1, 0, 0,
                   // clang-format on
                 }});
  this->run_test(this->descending_no_nulls,
                 cudf::order::DESCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_open{*prec},
                 cudf::bounded_open{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   0, -1, 0, 0, -1, 0, -1, 0, -1, -2, -1, 0, 0,
                   // Group-2
                   0, -1, 0, 0, -1, 0, -1, 0, -1, -2, -1, 0, 0,
                   // Group-3
                   0, -1, 0, 0, -1, 0, -1, 0, -1, -2, -1, 0, 0,
                   // clang-format on
                 }},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   0, 1, 0, 0, 1, 0, 1, 0, 1, 2, 1, 0, 0,
                   // Group-2
                   0, 1, 0, 0, 1, 0, 1, 0, 1, 2, 1, 0, 0,
                   // Group-3
                   0, 1, 0, 0, 1, 0, 1, 0, 1, 2, 1, 0, 0,
                   // clang-format on
                 }});
}

TYPED_TEST(GroupedSignedIntegralRangeWindows, DescendingNoNullsNegativeFollowing)
{
  using T   = TypeParam;
  auto prec = cudf::make_fixed_width_scalar<T>(4);
  auto foll = cudf::make_fixed_width_scalar<T>(-2);

  this->run_test(this->descending_no_nulls,
                 cudf::order::DESCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_closed{*prec},
                 cudf::bounded_closed{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   1, 1, 2, 1, 2, 2, 3, 4, 3, 4, 5, 6, 1,
                   // Group-2
                   1, 1, 2, 1, 2, 2, 3, 4, 3, 4, 5, 6, 1,
                   // Group-3
                   1, 1, 2, 1, 2, 2, 3, 4, 3, 4, 5, 6, 1,
                   // clang-format on
                 }},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   -1, -1, -2, -1, -1, -2, -1, -2, -1, -2, -2, -3, -1,
                   // Group-2
                   -1, -1, -2, -1, -1, -2, -1, -2, -1, -2, -2, -3, -1,
                   // Group-3
                   -1, -1, -2, -1, -1, -2, -1, -2, -1, -2, -2, -3, -1,
                   // clang-format on
                 }});
  this->run_test(this->descending_no_nulls,
                 cudf::order::DESCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_open{*prec},
                 cudf::bounded_open{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   1, 1, 2, 1, 1, 2, 2, 3, 3, 4, 3, 4, 1,
                   // Group-2
                   1, 1, 2, 1, 1, 2, 2, 3, 3, 4, 3, 4, 1,
                   // Group-3
                   1, 1, 2, 1, 1, 2, 2, 3, 3, 4, 3, 4, 1,
                   // clang-format on
                 }},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   -1, -1, -2, -1, -1, -2, -1, -2, -3, -2, -3, -4, -1,
                   // Group-2
                   -1, -1, -2, -1, -1, -2, -1, -2, -3, -2, -3, -4, -1,
                   // Group-3
                   -1, -1, -2, -1, -1, -2, -1, -2, -3, -2, -3, -4, -1,
                   // clang-format on
                 }});
}

TYPED_TEST(GroupedIntegralRangeWindows, AscendingNullsZeroPrecedingFollowing)
{
  using T   = TypeParam;
  auto prec = cudf::make_fixed_width_scalar<T>(0);
  auto foll = cudf::make_fixed_width_scalar<T>(0);

  this->run_test(this->ascending_nulls_before,
                 cudf::order::ASCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_closed{*prec},
                 cudf::bounded_closed{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   1, 2, 3, 4, 1, 1, 2, 1, 1, 1, 1, 2, 1,
                   // Group-2
                   1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1,
                   // Group-3
                   1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1,
                   // clang-format on
                 }},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   3, 2, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
                   // Group-2
                   0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
                   // Group-3
                   1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
                   // clang-format on
                 }});
  this->run_test(this->ascending_nulls_before,
                 cudf::order::ASCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_open{*prec},
                 cudf::bounded_open{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   1, 2, 3, 4, 0, -1, 0, 0, 0, 0, -1, 0, 0,
                   // Group-2
                   0, -1, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0,
                   // Group-3
                   1, 2, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0,
                   // clang-format on
                 }},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   3, 2, 1, 0, -1, -1, -2, -1, -1, -1, -1, -2, -1,
                   // Group-2
                   -1, -1, -2, -1, -1, -1, -2, -1, -1, -1, -1, -2, -1,
                   // Group-3
                   1, 0, -1, -1, -1, -1, -2, -1, -1, -1, -1, -2, -1,
                   // clang-format on
                 }});
}

TYPED_TEST(GroupedIntegralRangeWindows, AscendingNullsPositivePrecedingFollowing)
{
  using T   = TypeParam;
  auto prec = cudf::make_fixed_width_scalar<T>(2);
  auto foll = cudf::make_fixed_width_scalar<T>(1);

  this->run_test(this->ascending_nulls_before,
                 cudf::order::ASCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_closed{*prec},
                 cudf::bounded_closed{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   1, 2, 3, 4, 1, 2, 3, 1, 2, 1, 1, 2, 1,
                   // Group-2
                   1, 1, 2, 3, 4, 2, 3, 1, 2, 1, 1, 2, 1,
                   // Group-3
                   1, 2, 1, 2, 3, 2, 3, 1, 2, 1, 1, 2, 1,
                   // clang-format on
                 }},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   3, 2, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0,
                   // Group-2
                   0, 2, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0,
                   // Group-3
                   1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0,
                   // clang-format on
                 }});
  this->run_test(this->ascending_nulls_before,
                 cudf::order::ASCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_open{*prec},
                 cudf::bounded_open{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   1, 2, 3, 4, 1, 1, 2, 1, 2, 1, 1, 2, 1,
                   // Group-2
                   1, 1, 2, 3, 2, 1, 2, 1, 2, 1, 1, 2, 1,
                   // Group-3
                   1, 2, 1, 2, 2, 1, 2, 1, 2, 1, 1, 2, 1
                   // clang-format on
                 }},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   3, 2, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
                   // Group-2
                   0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
                   // Group-3
                   1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
                   // clang-format on
                 }});
}

TYPED_TEST(GroupedSignedIntegralRangeWindows, AscendingNullsNegativePreceding)
{
  using T   = TypeParam;
  auto prec = cudf::make_fixed_width_scalar<T>(-1);
  auto foll = cudf::make_fixed_width_scalar<T>(2);

  this->run_test(this->ascending_nulls_after,
                 cudf::order::ASCENDING,
                 cudf::null_order::AFTER,
                 cudf::bounded_closed{*prec},
                 cudf::bounded_closed{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   0, -1, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7,
                   // Group-2
                   0, -1, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0,
                   // Group-3
                   0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 1, 2, 3,
                   // clang-format on
                 }},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   0, 3, 2, 1, 1, 0, 6, 5, 4, 3, 2, 1, 0,
                   // Group-2
                   0, 3, 2, 1, 2, 1, 0, 1, 0, 0, 1, 0, 0,
                   // Group-3
                   0, 3, 2, 1, 2, 1, 0, 1, 0, 0, 2, 1, 0,
                   // clang-format on
                 }});
  this->run_test(this->ascending_nulls_after,
                 cudf::order::ASCENDING,
                 cudf::null_order::AFTER,
                 cudf::bounded_open{*prec},
                 cudf::bounded_open{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   0, -2, -1, -1, 0, 0, 1, 2, 3, 4, 5, 6, 7,
                   // Group-2
                   0, -2, -1, -1, 0, -1, 0, -1, 0, 0, -1, 0, 0,
                   // Group-3
                   0, -2, -1, -1, 0, -1, 0, -1, 0, 0, 1, 2, 3,
                   // clang-format on
                 }},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   0, 2, 1, 1, 0, 0, 6, 5, 4, 3, 2, 1, 0,
                   // Group-2
                   0, 2, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0,
                   // Group-3
                   0, 2, 1, 1, 0, 1, 0, 1, 0, 0, 2, 1, 0,
                   // clang-format on
                 }});
}

TYPED_TEST(GroupedSignedIntegralRangeWindows, AscendingNullsNegativeFollowing)
{
  using T   = TypeParam;
  auto prec = cudf::make_fixed_width_scalar<T>(4);
  auto foll = cudf::make_fixed_width_scalar<T>(-2);

  this->run_test(this->ascending_nulls_after,
                 cudf::order::ASCENDING,
                 cudf::null_order::AFTER,
                 cudf::bounded_closed{*prec},
                 cudf::bounded_closed{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   1, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6, 7,
                   // Group-2
                   1, 1, 2, 3, 4, 5, 6, 3, 4, 2, 1, 2, 1,
                   // Group-3
                   1, 1, 2, 3, 4, 5, 6, 3, 4, 2, 1, 2, 3,
                   // clang-format on
                 }},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   -1, -1, -2, -3, -2, -1, 6, 5, 4, 3, 2, 1, 0,
                   // Group-2
                   -1, -1, -2, -3, -2, -1, -2, -1, -2, -1, -1, -2, -1,
                   // Group-3
                   -1, -1, -2, -3, -2, -1, -2, -1, -2, -1, 2, 1, 0,
                   // clang-format on
                 }});
  this->run_test(this->ascending_nulls_after,
                 cudf::order::ASCENDING,
                 cudf::null_order::AFTER,
                 cudf::bounded_open{*prec},
                 cudf::bounded_open{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   1, 1, 2, 3, 4, 3, 1, 2, 3, 4, 5, 6, 7,
                   // Group-2
                   1, 1, 2, 3, 4, 3, 4, 3, 2, 1, 1, 2, 1,
                   // Group-3
                   1, 1, 2, 3, 4, 3, 4, 3, 2, 1, 1, 2, 3,
                   // clang-format on
                 }},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   -1, -1, -2, -3, -4, -2, 6, 5, 4, 3, 2, 1, 0,
                   // Group-2
                   -1, -1, -2, -3, -4, -2, -3, -1, -2, -1, -1, -2, -1,
                   // Group-3
                   -1, -1, -2, -3, -4, -2, -3, -1, -2, -1, 2, 1, 0,
                   // clang-format on
                 }});
}

TYPED_TEST(GroupedIntegralRangeWindows, DescendingNullsZeroPrecedingFollowing)
{
  using T   = TypeParam;
  auto prec = cudf::make_fixed_width_scalar<T>(0);
  auto foll = cudf::make_fixed_width_scalar<T>(0);

  this->run_test(this->descending_nulls_before,
                 cudf::order::DESCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_closed{*prec},
                 cudf::bounded_closed{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   1, 1, 2, 1, 1, 1, 1, 1, 2, 3, 4, 5, 6,
                   // Group-2
                   1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1,
                   // Group-3
                   1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 3,
                   // clang-format on
                 }},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   0, 1, 0, 0, 0, 0, 0, 5, 4, 3, 2, 1, 0,
                   // Group-2
                   0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
                   // Group-3
                   0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 2, 1, 0,
                   // clang-format on
                 }});
  this->run_test(this->descending_nulls_before,
                 cudf::order::DESCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_open{*prec},
                 cudf::bounded_open{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   0, -1, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6,
                   // Group-2
                   0, -1, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0,
                   // Group-3
                   0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 1, 2, 3,
                   // clang-format on
                 }},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   -1, -1, -2, -1, -1, -1, -1, 5, 4, 3, 2, 1, 0,
                   // Group-2
                   -1, -1, -2, -1, -1, -1, -1, -2, -1, -1, -1, -2, -1,
                    // Group-3
                   -1, -1, -2, -1, -1, -1, -1, -2, -1, -1, 2, 1, 0,
                   // clang-format on
                 }});
}

TYPED_TEST(GroupedIntegralRangeWindows, DescendingNullsPositivePrecedingFollowing)
{
  using T   = TypeParam;
  auto prec = cudf::make_fixed_width_scalar<T>(2);
  auto foll = cudf::make_fixed_width_scalar<T>(1);

  this->run_test(this->descending_nulls_before,
                 cudf::order::DESCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_closed{*prec},
                 cudf::bounded_closed{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   1, 1, 2, 1, 1, 2, 1, 1, 2, 3, 4, 5, 6,
                   // Group-2
                   1, 1, 2, 1, 1, 2, 1, 2, 3, 2, 3, 4, 1,
                   // Group-3
                   1, 1, 2, 1, 1, 2, 1, 2, 3, 2, 1, 2, 3,
                   // clang-format on
                 }},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   0, 1, 0, 0, 1, 0, 0, 5, 4, 3, 2, 1, 0,
                   // Group-2
                   0, 1, 0, 0, 1, 0, 1, 0, 1, 2, 1, 0, 0,
                   // Group-3
                   0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 2, 1, 0,
                   // clang-format on
                 }});
  this->run_test(this->descending_nulls_before,
                 cudf::order::DESCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_open{*prec},
                 cudf::bounded_open{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   1, 1, 2, 1, 1, 2, 1, 1, 2, 3, 4, 5, 6,
                   // Group-2
                   1, 1, 2, 1, 1, 2, 1, 2, 1, 2, 2, 3, 1,
                   // Group-3
                   1, 1, 2, 1, 1, 2, 1, 2, 1, 2, 1, 2, 3,
                   // clang-format on
                 }},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   0, 1, 0, 0, 0, 0, 0, 5, 4, 3, 2, 1, 0,
                   // Group-2
                   0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
                   // Group-3
                   0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 2, 1, 0,
                   // clang-format on
                 }});
}

TYPED_TEST(GroupedSignedIntegralRangeWindows, DescendingNullsNegativePreceding)
{
  using T   = TypeParam;
  auto prec = cudf::make_fixed_width_scalar<T>(-1);
  auto foll = cudf::make_fixed_width_scalar<T>(2);

  this->run_test(this->descending_nulls_after,
                 cudf::order::DESCENDING,
                 cudf::null_order::AFTER,
                 cudf::bounded_closed{*prec},
                 cudf::bounded_closed{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   1, 2, 3, 4, 5, 0, -1, 0, 0, 0, -1, 0, 0,
                   // Group-2
                   0, -1, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0,
                   // Group-3
                   1, 2, 3, 4, 0, 0, -1, 0, 0, 0, -1, 0, 0,
                   // clang-format on
                 }},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   4, 3, 2, 1, 0, 0, 2, 1, 3, 2, 1, 0, 0,
                   // Group-2
                   0, 1, 0, 0, 1, 0, 2, 1, 3, 2, 1, 0, 0,
                   // Group-3
                   3, 2, 1, 0, 1, 0, 2, 1, 3, 2, 1, 0, 0,
                   // clang-format on
                 }});
  this->run_test(this->descending_nulls_after,
                 cudf::order::DESCENDING,
                 cudf::null_order::AFTER,
                 cudf::bounded_open{*prec},
                 cudf::bounded_open{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   1, 2, 3, 4, 5, 0, -1, 0, -1, -2, -1, 0, 0,
                   // Group-2
                   0, -1, 0, 0, -1, 0, -1, 0, -1, -2, -1, 0, 0,
                   // Group-3
                   1, 2, 3, 4, -1, 0, -1, 0, -1, -2, -1, 0, 0,
                   // clang-format on
                 }},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   4, 3, 2, 1, 0, 0, 1, 0, 1, 2, 1, 0, 0,
                   // Group-2
                   0, 1, 0, 0, 1, 0, 1, 0, 1, 2, 1, 0, 0,
                   // Group-3
                   3, 2, 1, 0, 1, 0, 1, 0, 1, 2, 1, 0, 0,
                   // clang-format on
                 }});
}

TYPED_TEST(GroupedSignedIntegralRangeWindows, DescendingNullsNegativeFollowing)
{
  using T   = TypeParam;
  auto prec = cudf::make_fixed_width_scalar<T>(4);
  auto foll = cudf::make_fixed_width_scalar<T>(-2);

  this->run_test(this->descending_nulls_after,
                 cudf::order::DESCENDING,
                 cudf::null_order::AFTER,
                 cudf::bounded_closed{*prec},
                 cudf::bounded_closed{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   1, 2, 3, 4, 5, 1, 2, 3, 3, 4, 5, 6, 1,
                   // Group-2
                   1, 1, 2, 1, 2, 2, 3, 4, 3, 4, 5, 6, 1,
                   // Group-3
                   1, 2, 3, 4, 1, 2, 3, 4, 3, 4, 5, 6, 1,
                   // clang-format on
                 }},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   4, 3, 2, 1, 0, -1, -1, -2, -1, -2, -2, -3, -1,
                   // Group-2
                   -1, -1, -2, -1, -1, -2, -1, -2, -1, -2, -2, -3, -1,
                   // Group-3
                   3, 2, 1, 0, -1, -2, -1, -2, -1, -2, -2, -3, -1,
                   // clang-format on
                 }});
  this->run_test(this->descending_nulls_after,
                 cudf::order::DESCENDING,
                 cudf::null_order::AFTER,
                 cudf::bounded_open{*prec},
                 cudf::bounded_open{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   1, 2, 3, 4, 5, 1, 2, 3, 3, 4, 3, 4, 1,
                   // Group-2
                   1, 1, 2, 1, 1, 2, 2, 3, 3, 4, 3, 4, 1,
                   // Group-3
                   1, 2, 3, 4, 1, 2, 2, 3, 3, 4, 3, 4, 1,
                   // clang-format on
                 }},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   4, 3, 2, 1, 0, -1, -1, -2, -3, -2, -3, -4, -1,
                   // Group-2
                   -1, -1, -2, -1, -1, -2, -1, -2, -3, -2, -3, -4, -1,
                   // Group-3
                   3, 2, 1, 0, -1, -2, -1, -2, -3, -2, -3, -4, -1,
                   // clang-format on
                 }});
}
