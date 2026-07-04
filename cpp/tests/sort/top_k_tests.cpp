/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/copying.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <cuda/iterator>

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

  auto itr   = cuda::counting_iterator<int32_t>{0};
  auto input = cudf::test::fixed_width_column_wrapper<T, int32_t>(itr, itr + 100);
  auto expected =
    cudf::test::fixed_width_column_wrapper<T, int32_t>({90, 91, 92, 93, 94, 95, 96, 97, 98, 99});
  auto result = cudf::top_k(input, 10);
  result      = std::move(cudf::sort(cudf::table_view({result->view()}))->release().front());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result->view());
  result = cudf::top_k_order(input, 10);
  result = std::move(cudf::sort(cudf::table_view({result->view()}))->release().front());
  auto expected_order = cudf::test::fixed_width_column_wrapper<cudf::size_type>(
    {90, 91, 92, 93, 94, 95, 96, 97, 98, 99});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_order, result->view());

  result   = cudf::top_k(input, 10, cudf::order::ASCENDING);
  result   = std::move(cudf::sort(cudf::table_view({result->view()}))->release().front());
  expected = cudf::test::fixed_width_column_wrapper<T, int32_t>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result->view());
  result = cudf::top_k_order(input, 10, cudf::order::ASCENDING);
  result = std::move(cudf::sort(cudf::table_view({result->view()}))->release().front());
  expected_order =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_order, result->view());
}

TYPED_TEST(TopKTypes, TopK_Nulls)
{
  using T = TypeParam;

  auto itr   = cuda::counting_iterator<int32_t>{0};
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

  auto itr   = cuda::counting_iterator<int32_t>{0};
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

TEST_F(TopK, SegmentedUncoveredTail)
{
  // rows 8-9 hold the largest values but lie past the last offset: they are in no segment
  auto input =
    cudf::test::fixed_width_column_wrapper<int32_t>({40, 10, 20, 30, 50, 15, 25, 5, 90, 80});
  auto offsets = cudf::test::fixed_width_column_wrapper<int32_t>({0, 5, 8});

  // segment [0,5)={40,10,20,30,50} -> top-2 {50,40}; segment [5,8)={15,25,5} -> top-2 {25,15}
  auto expected = cudf::test::lists_column_wrapper<int32_t>({{50, 40}, {25, 15}});
  auto result   = cudf::segmented_top_k(input, offsets, 2);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result->view());

  auto expected_order = cudf::test::lists_column_wrapper<cudf::size_type>({{4, 0}, {6, 5}});
  result              = cudf::segmented_top_k_order(input, offsets, 2);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_order, result->view());
}

TEST_F(TopK, SegmentedUncoveredHead)
{
  // rows 0-1 hold the largest values but precede the first offset: they are in no segment
  auto input   = cudf::test::fixed_width_column_wrapper<int32_t>({100, 90, 7, 9, 8});
  auto offsets = cudf::test::fixed_width_column_wrapper<int32_t>({2, 5});

  // single segment [2,5)={7,9,8} -> top-2 {9,8} at rows {3,4}
  auto expected = cudf::test::lists_column_wrapper<int32_t>({{9, 8}});
  auto result   = cudf::segmented_top_k(input, offsets, 2);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result->view());

  auto expected_order = cudf::test::lists_column_wrapper<cudf::size_type>({{3, 4}});
  result              = cudf::segmented_top_k_order(input, offsets, 2);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_order, result->view());

  {
    // head-uncovered layout with multiple real segments: offsets {2,5,8} over 8 rows leave
    // rows 0-1 (the largest values) uncovered, then segments [2,5) and [5,8)
    auto input   = cudf::test::fixed_width_column_wrapper<int32_t>({100, 90, 7, 9, 8, 30, 10, 20});
    auto offsets = cudf::test::fixed_width_column_wrapper<int32_t>({2, 5, 8});

    // segment [2,5)={7,9,8} -> top-2 {9,8} at rows {3,4};
    // segment [5,8)={30,10,20} -> top-2 {30,20} at rows {5,7}
    auto expected = cudf::test::lists_column_wrapper<int32_t>({{9, 8}, {30, 20}});
    auto result   = cudf::segmented_top_k(input, offsets, 2);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result->view());

    auto expected_order = cudf::test::lists_column_wrapper<cudf::size_type>({{3, 4}, {5, 7}});
    result              = cudf::segmented_top_k_order(input, offsets, 2);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_order, result->view());
  }
}

TEST_F(TopK, SegmentedUncoveredBoth)
{
  // same shape as the segmented_sorted_order doc example: offsets {3,7} over 10 rows leave
  // rows 0-2 and 7-9 uncovered on both sides of the single segment
  auto input   = cudf::test::fixed_width_column_wrapper<int32_t>({9, 8, 7, 6, 5, 4, 3, 2, 1, 0});
  auto offsets = cudf::test::fixed_width_column_wrapper<int32_t>({3, 7});

  {
    // segment [3,7)={6,5,4,3} -> top-3 {6,5,4} at rows {3,4,5}
    auto expected = cudf::test::lists_column_wrapper<int32_t>({{6, 5, 4}});
    auto result   = cudf::segmented_top_k(input, offsets, 3);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result->view());

    auto expected_order = cudf::test::lists_column_wrapper<cudf::size_type>({{3, 4, 5}});
    result              = cudf::segmented_top_k_order(input, offsets, 3);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_order, result->view());
  }
  {
    // ASCENDING: smallest-3 of segment [3,7)={6,5,4,3} -> {3,4,5} at rows {6,5,4}
    auto expected = cudf::test::lists_column_wrapper<int32_t>({{3, 4, 5}});
    auto result   = cudf::segmented_top_k(input, offsets, 3, cudf::order::ASCENDING);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result->view());

    auto expected_order = cudf::test::lists_column_wrapper<cudf::size_type>({{6, 5, 4}});
    result              = cudf::segmented_top_k_order(input, offsets, 3, cudf::order::ASCENDING);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_order, result->view());
  }
}

TEST_F(TopK, SegmentedUncoveredAll)
{
  // a single offset defines zero segments: every row is uncovered so the result has no rows
  auto input   = cudf::test::fixed_width_column_wrapper<int32_t>({1, 2, 3, 4, 5, 6, 7, 8});
  auto offsets = cudf::test::fixed_width_column_wrapper<int32_t>({5});

  // both APIs must return a well-formed empty LIST column: zero rows, no nulls, and a child of the
  // expected element type (values for segmented_top_k, size_type indices for segmented_top_k_order)
  auto result = cudf::segmented_top_k(input, offsets, 2);
  EXPECT_EQ(result->size(), 0);
  EXPECT_EQ(result->type().id(), cudf::type_id::LIST);
  EXPECT_EQ(result->null_count(), 0);
  EXPECT_EQ(cudf::lists_column_view(result->view()).child().type().id(), cudf::type_id::INT32);

  result = cudf::segmented_top_k_order(input, offsets, 2);
  EXPECT_EQ(result->size(), 0);
  EXPECT_EQ(result->type().id(), cudf::type_id::LIST);
  EXPECT_EQ(result->null_count(), 0);
  EXPECT_EQ(cudf::lists_column_view(result->view()).child().type().id(), cudf::type_id::INT32);
}

TEST_F(TopK, SegmentedUncoveredNull)
{
  // a NULL occupies an uncovered tail row (row 8): coverage exclusion must happen before any
  // null ordering, so the null never appears in the result. row 9 is uncovered too. In DESCENDING
  // a null would otherwise sort first (null_order::BEFORE), so this sharply checks coverage wins.
  auto input = cudf::test::fixed_width_column_wrapper<int32_t>(
    {40, 10, 20, 30, 50, 15, 25, 5, 0, 80}, cudf::test::iterators::null_at(8));
  auto offsets = cudf::test::fixed_width_column_wrapper<int32_t>({0, 5, 8});

  // segment [0,5)={40,10,20,30,50} -> top-2 {50,40} at rows {4,0};
  // segment [5,8)={15,25,5} -> top-2 {25,15} at rows {6,5}; rows 8(null) and 9 excluded
  auto expected = cudf::test::lists_column_wrapper<int32_t>({{50, 40}, {25, 15}});
  auto result   = cudf::segmented_top_k(input, offsets, 2);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result->view());

  auto expected_order = cudf::test::lists_column_wrapper<cudf::size_type>({{4, 0}, {6, 5}});
  result              = cudf::segmented_top_k_order(input, offsets, 2);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_order, result->view());
}

TEST_F(TopK, SegmentedUncoveredSmallSegment)
{
  // a covered segment with fewer than k elements must return ALL of its elements, while the
  // uncovered head (row 0) and tail (row 7) stay excluded. k=3 with a size-2 segment [1,3)
  auto input   = cudf::test::fixed_width_column_wrapper<int32_t>({50, 8, 4, 9, 1, 6, 7, 99});
  auto offsets = cudf::test::fixed_width_column_wrapper<int32_t>({1, 3, 7});

  // segment [1,3)={8,4} (size 2 < k) -> all {8,4} at rows {1,2};
  // segment [3,7)={9,1,6,7} -> top-3 {9,7,6} at rows {3,6,5}; rows 0 and 7 excluded
  auto expected = cudf::test::lists_column_wrapper<int32_t>({{8, 4}, {9, 7, 6}});
  auto result   = cudf::segmented_top_k(input, offsets, 3);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result->view());

  auto expected_order = cudf::test::lists_column_wrapper<cudf::size_type>({{1, 2}, {3, 6, 5}});
  result              = cudf::segmented_top_k_order(input, offsets, 3);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_order, result->view());
}

TEST_F(TopK, Errors)
{
  auto itr   = cuda::counting_iterator<int64_t>{0};
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
