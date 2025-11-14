/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/copying.hpp>
#include <cudf/sorting.hpp>
#include <cudf/types.hpp>

#include <type_traits>
#include <vector>

using TestTypes = cudf::test::
  Concat<cudf::test::IntegralTypesNotBool, cudf::test::FloatingPointTypes, cudf::test::ChronoTypes>;

template <typename T>
struct TopKTypes : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(TopKTypes, TestTypes);

TYPED_TEST(TopKTypes, TopK)
{
  using T = TypeParam;

  auto itr   = thrust::counting_iterator<int32_t>(0);
  auto input = cudf::test::fixed_width_column_wrapper<T, int32_t>(
    itr, itr + 100, cudf::test::iterators::null_at(4));
  auto expected =
    cudf::test::fixed_width_column_wrapper<T, int32_t>({99, 98, 97, 96, 95, 94, 93, 92, 91, 90});
  auto result = cudf::top_k(input, 10);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result->view());
  auto expected_order = cudf::test::fixed_width_column_wrapper<cudf::size_type>(
    {99, 98, 97, 96, 95, 94, 93, 92, 91, 90});
  result = cudf::top_k_order(input, 10);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_order, result->view());

  result   = cudf::top_k(input, 10, cudf::order::ASCENDING);
  expected = cudf::test::fixed_width_column_wrapper<T, int32_t>({0, 1, 2, 3, 5, 6, 7, 8, 9, 10});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result->view());
  expected_order =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>({0, 1, 2, 3, 5, 6, 7, 8, 9, 10});
  result = cudf::top_k_order(input, 10, cudf::order::ASCENDING);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_order, result->view());
}

TYPED_TEST(TopKTypes, TopKSegmented)
{
  using T    = TypeParam;
  using LCW  = cudf::test::lists_column_wrapper<T, int32_t>;
  using LCWO = cudf::test::lists_column_wrapper<cudf::size_type>;

  auto itr   = thrust::counting_iterator<int32_t>(0);
  auto input = cudf::test::fixed_width_column_wrapper<T, int32_t>(
    itr, itr + 100, cudf::test::iterators::null_at(4));
  auto offsets =
    cudf::test::fixed_width_column_wrapper<int32_t>({0, 15, 20, 23, 40, 42, 60, 70, 80, 90, 100});
  {
    // clang-format off
    LCW expected({
      {14, 13, 12}, {19, 18, 17}, {22, 21, 20}, {39, 38, 37}, {41, 40},
      {59, 58, 57}, {69, 68, 67}, {79, 78, 77}, {89, 88, 87}, {99, 98, 97}});
    LCWO expected_order({
      {14, 13, 12}, {19, 18, 17}, {22, 21, 20}, {39, 38, 37}, {41, 40},
      {59, 58, 57}, {69, 68, 67}, {79, 78, 77}, {89, 88, 87}, {99, 98, 97}});
    // clang-format on
    auto result = cudf::segmented_top_k(input, offsets, 3);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result->view());
    result = cudf::segmented_top_k_order(input, offsets, 3);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_order, result->view());
  }

  {
    // clang-format off
    LCW expected({
      {0,  1,  2},  {15, 16, 17}, {20, 21, 22}, {23, 24, 25}, {40, 41},
      {42, 43, 44}, {60, 61, 62}, {70, 71, 72}, {80, 81, 82}, {90, 91, 92}});
     LCWO expected_order({
      {0,  1,  2},  {15, 16, 17}, {20, 21, 22}, {23, 24, 25}, {40, 41},
      {42, 43, 44}, {60, 61, 62}, {70, 71, 72}, {80, 81, 82}, {90, 91, 92}});
    // clang-format on
    auto result = cudf::segmented_top_k(input, offsets, 3, cudf::order::ASCENDING);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result->view());
    result = cudf::segmented_top_k_order(input, offsets, 3, cudf::order::ASCENDING);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_order, result->view());
  }
}

struct TopK : public cudf::test::BaseFixture {};

TEST_F(TopK, Empty)
{
  auto input = cudf::test::fixed_width_column_wrapper<int32_t>({0, 1, 2, 3});

  auto result = cudf::top_k(input, 0);
  EXPECT_EQ(result->size(), 0);
  result = cudf::top_k_order(input, 0);
  EXPECT_EQ(result->size(), 0);
  result = cudf::segmented_top_k(input, input, 0);
  EXPECT_EQ(result->size(), 0);
  result = cudf::segmented_top_k_order(input, input, 0);
  EXPECT_EQ(result->size(), 0);
}

TEST_F(TopK, Errors)
{
  auto itr   = thrust::counting_iterator<int64_t>(0);
  auto input = cudf::test::fixed_width_column_wrapper<int64_t>(itr, itr + 100);

  EXPECT_THROW(cudf::top_k(input, -1), std::invalid_argument);
  EXPECT_THROW(cudf::top_k_order(input, -1), std::invalid_argument);

  auto offsets = cudf::test::fixed_width_column_wrapper<int32_t>({0, 15, 20, 23, 40, 42});
  EXPECT_THROW(cudf::segmented_top_k(input, offsets, -1), std::invalid_argument);
  EXPECT_THROW(cudf::segmented_top_k_order(input, offsets, -1), std::invalid_argument);
  offsets = cudf::test::fixed_width_column_wrapper<int32_t>({});
  EXPECT_THROW(cudf::segmented_top_k(input, offsets, 10), std::invalid_argument);
  EXPECT_THROW(cudf::segmented_top_k_order(input, offsets, 10), std::invalid_argument);
  offsets = cudf::test::fixed_width_column_wrapper<int32_t>({0, 15}, {1, 0});
  EXPECT_THROW(cudf::segmented_top_k(input, offsets, 10), std::invalid_argument);
  EXPECT_THROW(cudf::segmented_top_k_order(input, offsets, 10), std::invalid_argument);

  EXPECT_THROW(cudf::segmented_top_k(input, input, 10), cudf::data_type_error);
  EXPECT_THROW(cudf::segmented_top_k_order(input, input, 10), cudf::data_type_error);
}
