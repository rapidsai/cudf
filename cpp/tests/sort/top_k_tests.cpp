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
#include <cudf_test/debug_utilities.hpp>
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
struct TopK : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(TopK, TestTypes);

TYPED_TEST(TopK, TopK)
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

  EXPECT_THROW(cudf::top_k(input, 101), std::invalid_argument);
  EXPECT_THROW(cudf::top_k_order(input, 101), std::invalid_argument);
}

TYPED_TEST(TopK, TopKSegmented)
{
  using T    = TypeParam;
  using LCW  = cudf::test::lists_column_wrapper<T, int32_t>;
  using LCWO = cudf::test::lists_column_wrapper<cudf::size_type>;

  auto itr   = thrust::counting_iterator<int32_t>(0);
  auto input = cudf::test::fixed_width_column_wrapper<T, int32_t>(
    itr, itr + 100, cudf::test::iterators::null_at(4));
  auto offsets =
    cudf::test::fixed_width_column_wrapper<int32_t>({0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100});
  {
    // clang-format off
    LCW expected({
      { 9,  8,  7}, {19, 18, 17}, {29, 28, 27}, {39, 38, 37}, {49, 48, 47},
      {59, 58, 57}, {69, 68, 67}, {79, 78, 77}, {89, 88, 87}, {99, 98, 97}});
    LCWO expected_order({
      { 9,  8,  7}, {19, 18, 17}, {29, 28, 27}, {39, 38, 37}, {49, 48, 47},
      {59, 58, 57}, {69, 68, 67}, {79, 78, 77}, {89, 88, 87}, {99, 98, 97}});
    // clang-format on
    auto result = cudf::top_k_segmented(input, offsets, 3);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result->view());
    result = cudf::top_k_segmented_order(input, offsets, 3);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_order, result->view());
  }

  {
    // clang-format off
    LCW expected({
      {0,  1,  2},  {10, 11, 12}, {20, 21, 22}, {30, 31, 32}, {40, 41, 42},
      {50, 51, 52}, {60, 61, 62}, {70, 71, 72}, {80, 81, 82}, {90, 91, 92}});
     LCWO expected_order({
      {0,  1,  2},  {10, 11, 12}, {20, 21, 22}, {30, 31, 32}, {40, 41, 42},
      {50, 51, 52}, {60, 61, 62}, {70, 71, 72}, {80, 81, 82}, {90, 91, 92}});
    // clang-format on
    auto result = cudf::top_k_segmented(input, offsets, 3, cudf::order::ASCENDING);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result->view());
    result = cudf::top_k_segmented_order(input, offsets, 3, cudf::order::ASCENDING);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_order, result->view());
  }
}
