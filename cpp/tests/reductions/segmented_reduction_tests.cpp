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

#include "cudf_test/column_utilities.hpp"
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/reduction.hpp>
#include <cudf/types.hpp>

#include <limits>

namespace cudf {
namespace test {

#define XXX 0  // null placeholder

template <typename T>
struct SegmentedReductionTest : public cudf::test::BaseFixture {
};

struct SegmentedReductionTestUntyped : public cudf::test::BaseFixture {
};

TYPED_TEST_SUITE(SegmentedReductionTest, NumericTypes);

TYPED_TEST(SegmentedReductionTest, SumExcludeNulls)
{
  // [1, 2, 3], [1, null, 3], [1], [null], [null, null], []
  // values:   {1, 2, 3, 1, XXX, 3, 1, XXX, XXX, XXX}
  // offsets:  {0, 3, 6, 7, 8, 10, 10}
  // nullmask: {1, 1, 1, 1, 0, 1, 1, 0, 0, 0}
  // outputs:  {6, 4, 1, XXX, XXX, XXX}
  // output nullmask: {1, 1, 1, 0, 0, 0}
  auto input     = fixed_width_column_wrapper<TypeParam>{{1, 2, 3, 1, XXX, 3, 1, XXX, XXX, XXX},
                                                     {1, 1, 1, 1, 0, 1, 1, 0, 0, 0}};
  auto offsets   = std::vector<size_type>{0, 3, 6, 7, 8, 10, 10};
  auto d_offsets = thrust::device_vector<size_type>(offsets);
  auto expect = fixed_width_column_wrapper<TypeParam>{{6, 4, 1, XXX, XXX, XXX}, {1, 1, 1, 0, 0, 0}};

  auto res = segmented_reduce(input,
                              d_offsets,
                              *make_sum_aggregation<segmented_reduce_aggregation>(),
                              data_type{type_to_id<TypeParam>()},
                              null_policy::EXCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);
}

TYPED_TEST(SegmentedReductionTest, ProductExcludeNulls)
{
  // [1, 3, 5], [null, 3, 5], [1], [null], [null, null], []
  // values:    {1, 3, 5, XXX, 3, 5, 1, XXX, XXX, XXX}
  // offsets:   {0, 3, 6, 7, 8, 10, 10}
  // nullmask:  {1, 1, 1, 0, 1, 1, 1, 0, 0, 0}
  // outputs:   {15, 15, 1, XXX, XXX, XXX}
  // output nullmask: {1, 1, 1, 0, 0, 0}
  auto input     = fixed_width_column_wrapper<TypeParam>{{1, 3, 5, XXX, 3, 5, 1, XXX, XXX, XXX},
                                                     {1, 1, 1, 0, 1, 1, 1, 0, 0, 0}};
  auto offsets   = std::vector<size_type>{0, 3, 6, 7, 8, 10, 10};
  auto d_offsets = thrust::device_vector<size_type>(offsets);
  auto expect =
    fixed_width_column_wrapper<TypeParam>{{15, 15, 1, XXX, XXX, XXX}, {1, 1, 1, 0, 0, 0}};

  auto res = segmented_reduce(input,
                              d_offsets,
                              *make_product_aggregation<segmented_reduce_aggregation>(),
                              data_type{type_to_id<TypeParam>()},
                              null_policy::EXCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);
}

TYPED_TEST(SegmentedReductionTest, MaxExcludeNulls)
{
  // [1, 2, 3], [1, null, 3], [1], [null], [null, null], []
  // values:    {1, 2, 3, 1, XXX, 3, 1, XXX, XXX, XXX}
  // offsets:   {0, 3, 6, 7, 8, 10, 10}
  // nullmask:  {1, 1, 1, 1, 0, 1, 1, 0, 0, 0}
  // outputs:   {3, 3, 1, XXX, XXX, XXX}
  // output nullmask: {1, 1, 1, 0, 0, 0}
  auto input     = fixed_width_column_wrapper<TypeParam>{{1, 2, 3, 1, XXX, 3, 1, XXX, XXX, XXX},
                                                     {1, 1, 1, 1, 0, 1, 1, 0, 0, 0}};
  auto offsets   = std::vector<size_type>{0, 3, 6, 7, 8, 10, 10};
  auto d_offsets = thrust::device_vector<size_type>(offsets);
  auto expect = fixed_width_column_wrapper<TypeParam>{{3, 3, 1, XXX, XXX, XXX}, {1, 1, 1, 0, 0, 0}};

  auto res = segmented_reduce(input,
                              d_offsets,
                              *make_max_aggregation<segmented_reduce_aggregation>(),
                              data_type{type_to_id<TypeParam>()},
                              null_policy::EXCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);
}

TYPED_TEST(SegmentedReductionTest, MinExcludeNulls)
{
  // [1, 2, 3], [1, null, 3], [1], [null], [null, null], []
  // values:   {1, 2, 3, 1, XXX, 3, 1, XXX, XXX, XXX}
  // offsets:  {0, 3, 6, 7, 8, 10, 10}
  // nullmask: {1, 1, 1, 1, 0, 1, 1, 0, 0, 0}
  // outputs:  {1, 1, 1, XXX, XXX, XXX}
  // output nullmask: {1, 1, 1, 0, 0, 0}
  auto input     = fixed_width_column_wrapper<TypeParam>{{1, 2, 3, 1, XXX, 3, 1, XXX, XXX, XXX},
                                                     {1, 1, 1, 1, 0, 1, 1, 0, 0, 0}};
  auto offsets   = std::vector<size_type>{0, 3, 6, 7, 8, 10, 10};
  auto d_offsets = thrust::device_vector<size_type>(offsets);
  auto expect = fixed_width_column_wrapper<TypeParam>{{1, 1, 1, XXX, XXX, XXX}, {1, 1, 1, 0, 0, 0}};

  auto res = segmented_reduce(input,
                              d_offsets,
                              *make_min_aggregation<segmented_reduce_aggregation>(),
                              data_type{type_to_id<TypeParam>()},
                              null_policy::EXCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);
}

TYPED_TEST(SegmentedReductionTest, AnyExcludeNulls)
{
  // [0, 0, 0], [0, null, 0], [0, 1, 0], [1, null, 0], [], [0], [1], [null], [null, null]
  // values:  {0, 0, 0, 0, XXX, 0, 0, 1, 0, 1, XXX, 0, 0, 1, XXX, XXX, XXX}
  // offsets: {0, 3, 6, 9, 12, 12, 13, 14, 15, 17}
  // nullmask:{1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0}
  // outputs: {0, 0, 1, 1, XXX, 0, 1, XXX, XXX}
  // output nullmask: {1, 1, 1, 1, 0, 1, 1, 0, 0}
  auto input = fixed_width_column_wrapper<TypeParam>{
    {0, 0, 0, 0, XXX, 0, 0, 1, 0, 1, XXX, 0, 0, 1, XXX, XXX, XXX},
    {1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0}};
  auto offsets   = std::vector<size_type>{0, 3, 6, 9, 12, 12, 13, 14, 15, 17};
  auto d_offsets = thrust::device_vector<size_type>(offsets);
  auto expect    = fixed_width_column_wrapper<bool>{
    {false, false, true, true, bool{XXX}, false, true, bool{XXX}, bool{XXX}},
    {true, true, true, true, false, true, true, false, false}};

  auto res = segmented_reduce(input,
                              d_offsets,
                              *make_any_aggregation<segmented_reduce_aggregation>(),
                              data_type{type_id::BOOL8},
                              null_policy::EXCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);
}

TYPED_TEST(SegmentedReductionTest, AllExcludeNulls)
{
  // [1, 2, 3], [1, null, 3], [], [1], [null], [null, null], [1, 0, 3], [1, null, 0], [0]
  // values: {1, 2, 3, 1, XXX, 3, 1, XXX, XXX, XXX, 1, 0, 3, 1, XXX, 0, 0}
  // offsets: {0, 3, 6, 6, 7, 8, 10, 13, 16, 17}
  // nullmask: {1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1}
  // outputs: {true, true, XXX, true, XXX, XXX, false, false, false}
  // output nullmask: {1, 1, 0, 1, 0, 0, 1, 1, 1}
  auto input = fixed_width_column_wrapper<TypeParam>{
    {1, 2, 3, 1, XXX, 3, 1, XXX, XXX, XXX, 1, 0, 3, 1, XXX, 0, 0},
    {1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1}};
  auto offsets   = std::vector<size_type>{0, 3, 6, 6, 7, 8, 10, 13, 16, 17};
  auto d_offsets = thrust::device_vector<size_type>(offsets);
  auto expect    = fixed_width_column_wrapper<bool>{
    {true, true, bool{XXX}, true, bool{XXX}, bool{XXX}, false, false, false},
    {true, true, false, true, false, false, true, true, true}};

  auto res = segmented_reduce(input,
                              d_offsets,
                              *make_all_aggregation<segmented_reduce_aggregation>(),
                              data_type{type_id::BOOL8},
                              null_policy::EXCLUDE);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);
}

TYPED_TEST(SegmentedReductionTest, SumIncludeNulls)
{
  // [1, 2, 3], [1, null, 3], [1], [null], [null, null], []
  // values:   {1, 2, 3, 1, XXX, 3, 1, XXX, XXX, XXX}
  // offsets:  {0, 3, 6, 7, 8, 10, 10}
  // nullmask: {1, 1, 1, 1, 0, 1, 1, 0, 0, 0}
  // outputs:  {6, XXX, 1, XXX, XXX, XXX}
  // output nullmask: {1, 0, 1, 0, 0, 0}
  auto input     = fixed_width_column_wrapper<TypeParam>{{1, 2, 3, 1, XXX, 3, 1, XXX, XXX, XXX},
                                                     {1, 1, 1, 1, 0, 1, 1, 0, 0, 0}};
  auto offsets   = std::vector<size_type>{0, 3, 6, 7, 8, 10, 10};
  auto d_offsets = thrust::device_vector<size_type>(offsets);
  auto expect =
    fixed_width_column_wrapper<TypeParam>{{6, XXX, 1, XXX, XXX, XXX}, {1, 0, 1, 0, 0, 0}};

  auto res = segmented_reduce(input,
                              d_offsets,
                              *make_sum_aggregation<segmented_reduce_aggregation>(),
                              data_type{type_to_id<TypeParam>()},
                              null_policy::INCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);
}

TYPED_TEST(SegmentedReductionTest, ProductIncludeNulls)
{
  // [1, 3, 5], [null, 3, 5], [1], [null], [null, null], []
  // values:    {1, 3, 5, XXX, 3, 5, 1, XXX, XXX, XXX}
  // offsets:   {0, 3, 6, 7, 8, 10, 10}
  // nullmask:  {1, 1, 1, 0, 1, 1, 1, 0, 0, 0}
  // outputs:   {15, XXX, 1, XXX, XXX, XXX}
  // output nullmask: {1, 0, 1, 0, 0, 0}
  auto input     = fixed_width_column_wrapper<TypeParam>{{1, 3, 5, XXX, 3, 5, 1, XXX, XXX, XXX},
                                                     {1, 1, 1, 0, 1, 1, 1, 0, 0, 0}};
  auto offsets   = std::vector<size_type>{0, 3, 6, 7, 8, 10, 10};
  auto d_offsets = thrust::device_vector<size_type>(offsets);
  auto expect =
    fixed_width_column_wrapper<TypeParam>{{15, XXX, 1, XXX, XXX, XXX}, {1, 0, 1, 0, 0, 0}};

  auto res = segmented_reduce(input,
                              d_offsets,
                              *make_product_aggregation<segmented_reduce_aggregation>(),
                              data_type{type_to_id<TypeParam>()},
                              null_policy::INCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);
}

TYPED_TEST(SegmentedReductionTest, MaxIncludeNulls)
{
  // [1, 2, 3], [1, null, 3], [1], [null], [null, null], []
  // values:    {1, 2, 3, 1, XXX, 3, 1, XXX, XXX, XXX}
  // offsets:   {0, 3, 6, 7, 8, 10, 10}
  // nullmask:  {1, 1, 1, 1, 0, 1, 1, 0, 0, 0}
  // outputs:   {3, XXX, 1, XXX, XXX, XXX}
  // output nullmask: {1, 0, 1, 0, 0, 0}
  auto input     = fixed_width_column_wrapper<TypeParam>{{1, 2, 3, 1, XXX, 3, 1, XXX, XXX, XXX},
                                                     {1, 1, 1, 1, 0, 1, 1, 0, 0, 0}};
  auto offsets   = std::vector<size_type>{0, 3, 6, 7, 8, 10, 10};
  auto d_offsets = thrust::device_vector<size_type>(offsets);
  auto expect =
    fixed_width_column_wrapper<TypeParam>{{3, XXX, 1, XXX, XXX, XXX}, {1, 0, 1, 0, 0, 0}};

  auto res = segmented_reduce(input,
                              d_offsets,
                              *make_max_aggregation<segmented_reduce_aggregation>(),
                              data_type{type_to_id<TypeParam>()},
                              null_policy::INCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);
}

TYPED_TEST(SegmentedReductionTest, MinIncludeNulls)
{
  // [1, 2, 3], [1, null, 3], [1], [null], [null, null], []
  // values:   {1, 2, 3, 1, XXX, 3, 1, XXX, XXX}
  // offsets:  {0, 3, 6, 7, 8, 10, 10}
  // nullmask: {1, 1, 1, 1, 0, 1, 1, 0, 0}
  // outputs:  {1, XXX, 1, XXX, XXX, XXX}
  // output nullmask: {1, 0, 1, 0, 0, 0}
  auto input     = fixed_width_column_wrapper<TypeParam>{{1, 2, 3, 1, XXX, 3, 1, XXX, XXX, XXX},
                                                     {1, 1, 1, 1, 0, 1, 1, 0, 0}};
  auto offsets   = std::vector<size_type>{0, 3, 6, 7, 8, 10, 10};
  auto d_offsets = thrust::device_vector<size_type>(offsets);
  auto expect =
    fixed_width_column_wrapper<TypeParam>{{1, XXX, 1, XXX, XXX, XXX}, {1, 0, 1, 0, 0, 0}};

  auto res = segmented_reduce(input,
                              d_offsets,
                              *make_min_aggregation<segmented_reduce_aggregation>(),
                              data_type{type_to_id<TypeParam>()},
                              null_policy::INCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);
}

TYPED_TEST(SegmentedReductionTest, AnyIncludeNulls)
{
  // [0, 0, 0], [0, null, 0], [0, 1, 0], [1, null, 0], [], [0], [1], [null], [null, null]
  // values:  {0, 0, 0, 0, XXX, 0, 0, 1, 0, 1, XXX, 0, 0, 1, XXX, XXX, XXX}
  // offsets: {0, 3, 6, 9, 12, 12, 13, 14, 15, 17}
  // nullmask:{1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0}
  // outputs: {0, XXX, 1, XXX, XXX, 0, 1, XXX, XXX}
  // output nullmask: {1, 0, 1, 0, 0, 1, 1, 0, 0}
  auto input = fixed_width_column_wrapper<TypeParam>{
    {0, 0, 0, 0, XXX, 0, 0, 1, 0, 1, XXX, 0, 0, 1, XXX, XXX, XXX},
    {1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0}};
  auto offsets   = std::vector<size_type>{0, 3, 6, 9, 12, 12, 13, 14, 15, 17};
  auto d_offsets = thrust::device_vector<size_type>(offsets);
  auto expect    = fixed_width_column_wrapper<bool>{
    {false, bool{XXX}, true, bool{XXX}, bool{XXX}, false, true, bool{XXX}, bool{XXX}},
    {true, false, true, false, false, true, true, false, false}};

  auto res = segmented_reduce(input,
                              d_offsets,
                              *make_any_aggregation<segmented_reduce_aggregation>(),
                              data_type{type_id::BOOL8},
                              null_policy::INCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);
}

TYPED_TEST(SegmentedReductionTest, AllIncludeNulls)
{
  // [1, 2, 3], [1, null, 3], [], [1], [null], [null, null], [1, 0, 3], [1, null, 0], [0]
  // values: {1, 2, 3, 1, XXX, 3, 1, XXX, XXX, XXX, 1, 0, 3, 1, XXX, 0, 0}
  // offsets: {0, 3, 6, 6, 7, 8, 10, 13, 16, 17}
  // nullmask: {1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1}
  // outputs: {true, XXX, XXX, true, XXX, XXX, false, XXX, false}
  // output nullmask: {1, 0, 0, 1, 0, 0, 1, 0, 1}
  auto input = fixed_width_column_wrapper<TypeParam>{
    {1, 2, 3, 1, XXX, 3, 1, XXX, XXX, XXX, 1, 0, 3, 1, XXX, 0, 0},
    {1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1}};
  auto offsets   = std::vector<size_type>{0, 3, 6, 6, 7, 8, 10, 13, 16, 17};
  auto d_offsets = thrust::device_vector<size_type>(offsets);
  auto expect    = fixed_width_column_wrapper<bool>{
    {true, bool{XXX}, bool{XXX}, true, bool{XXX}, bool{XXX}, false, bool{XXX}, false},
    {true, false, false, true, false, false, true, false, true}};

  auto res = segmented_reduce(input,
                              d_offsets,
                              *make_all_aggregation<segmented_reduce_aggregation>(),
                              data_type{type_id::BOOL8},
                              null_policy::INCLUDE);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);
}

TEST_F(SegmentedReductionTestUntyped, PartialSegmentReudction)
{
  // Segmented reduction allows offsets only specify part of the input columns.
  // [1], [2, 3], [4]
  // values: {1, 2, 3, 4, 5, 6, 7}
  // offsets: {0, 1, 3, 4}
  // nullmask: {1, 1, 1, 1, 1, 1, 1}
  // outputs: {1, 5, 4}
  // output nullmask: {1, 1, 1}

  auto input     = fixed_width_column_wrapper<int32_t>{{1, 2, 3, 4, 5, 6, 7},
                                                   {true, true, true, true, true, true, true}};
  auto offsets   = std::vector<size_type>{0, 1, 3, 4};
  auto d_offsets = thrust::device_vector<size_type>(offsets);
  auto expect    = fixed_width_column_wrapper<int32_t>{{1, 5, 4}, {true, true, true}};

  auto res = segmented_reduce(input,
                              d_offsets,
                              *make_sum_aggregation<segmented_reduce_aggregation>(),
                              data_type{type_id::INT32},
                              null_policy::INCLUDE);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);
}

TEST_F(SegmentedReductionTestUntyped, NonNullableInput)
{
  // Segmented reduction allows offsets only specify part of the input columns.
  // [1], [], [2, 3], [4, 5, 6, 7]
  // values: {1, 2, 3, 4, 5, 6, 7}
  // offsets: {0, 1, 1, 3, 7}
  // nullmask: nullptr
  // outputs: {1, 5, 4}
  // output nullmask: {1, 1, 1}

  auto input     = fixed_width_column_wrapper<int32_t>{1, 2, 3, 4, 5, 6, 7};
  auto offsets   = std::vector<size_type>{0, 1, 1, 3, 7};
  auto d_offsets = thrust::device_vector<size_type>(offsets);
  auto expect    = fixed_width_column_wrapper<int32_t>{{1, XXX, 5, 22}, {true, false, true, true}};

  auto res = segmented_reduce(input,
                              d_offsets,
                              *make_sum_aggregation<segmented_reduce_aggregation>(),
                              data_type{type_id::INT32},
                              null_policy::INCLUDE);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);
}

TEST_F(SegmentedReductionTestUntyped, ReduceEmptyColumn)
{
  auto input     = fixed_width_column_wrapper<int32_t>{};
  auto offsets   = std::vector<size_type>{0};
  auto d_offsets = thrust::device_vector<size_type>(offsets);
  auto expect    = fixed_width_column_wrapper<int32_t>{};

  auto res = segmented_reduce(input,
                              d_offsets,
                              *make_sum_aggregation<segmented_reduce_aggregation>(),
                              data_type{type_to_id<int32_t>()},
                              null_policy::EXCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);
}

int32_t pow10(int32_t exponent) { return exponent == 0 ? 1 : 10 * pow10(exponent - 1); }

template <typename T>
struct SegmentedReductionFixedPointTest : public cudf::test::BaseFixture {
 public:
  std::vector<typename T::rep> scale_list_by_pow10(std::vector<typename T::rep> input,
                                                   int32_t exponent)
  {
    std::vector<typename T::rep> result(input.size());
    std::transform(input.begin(), input.end(), result.begin(), [&exponent](auto x) {
      return exponent >= 0 ? x * pow10(exponent) : x / pow10(-exponent);
    });
    return result;
  }
};

TYPED_TEST_SUITE(SegmentedReductionFixedPointTest, cudf::test::FixedPointTypes);

TYPED_TEST(SegmentedReductionFixedPointTest, MaxIncludeNullsScaleZero)
{
  // [1, 2, 3], [1], [], [2, NULL, 3], [NULL], [NULL, NULL] | scale: 0
  // values:  {1, 2, 3, 1, 2, XXX, 3, XXX, XXX, XXX}
  // nullmask:{1, 1, 1, 1, 1, 0, 1, 0, 0, 0}
  // offsets: {0, 3, 4, 4, 7, 8, 10}
  // output_dtype: decimalXX, scale: -1, 0, 1
  // outputs: {3, 1, XXX, XXX, XXX, XXX}
  // output nullmask: {1, 1, 0, 0, 0, 0}

  using DecimalXX = TypeParam;

  for (int output_scale : {-1, 0, 1}) {
    fixed_point_column_wrapper<typename DecimalXX::rep> input{
      {1, 2, 3, 1, 2, XXX, 3, XXX, XXX, XXX},
      {true, true, true, true, true, false, true, false, false, false},
      numeric::scale_type(0)};
    fixed_width_column_wrapper<size_type> offsets{0, 3, 4, 4, 7, 8, 10};

    data_type output_dtype{type_to_id<DecimalXX>(), numeric::scale_type{output_scale}};

    auto result_rep = this->scale_list_by_pow10({3, 1, XXX, XXX, XXX, XXX}, -output_scale);
    fixed_point_column_wrapper<typename DecimalXX::rep> expect{
      result_rep.begin(),
      result_rep.end(),
      {true, true, false, false, false, false},
      numeric::scale_type(output_scale)};

    auto res = segmented_reduce(input,
                                column_view(offsets),
                                *make_max_aggregation<segmented_reduce_aggregation>(),
                                output_dtype,
                                null_policy::INCLUDE);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);
  }
}

struct SegmentedReductionStringTest : public cudf::test::BaseFixture {
};

TEST_F(SegmentedReductionStringTest, MaxIncludeNulls)
{
  // ['world'], ['cudf', NULL, 'cuml'], ['hello', 'rapids', 'ai'], [], [NULL], [NULL, NULL]
  // values:  {"world", "cudf", XXX, "cuml", "hello", "rapids", "ai", XXX, XXX, XXX}
  // nullmask:{1, 1, 0, 1, 1, 1, 1, 0, 0, 0}
  // offsets: {0, 1, 4, 7, 7, 8, 10}
  // output_dtype: string dtype
  // outputs: {"world", XXX, "rapids", XXX, XXX, XXX}
  // output nullmask: {1, 0, 1, 0, 0, 0}

  strings_column_wrapper input{
    {"world", "cudf", XXX, "cuml", "hello", "rapids", "ai", XXX, XXX, XXX},
    {true, true, false, true, true, true, true, false, false, false}};
  fixed_width_column_wrapper<size_type> offsets{0, 1, 4, 7, 7, 8, 10};
  data_type output_dtype{type_id::STRING};

  strings_column_wrapper expect{{"world", XXX, "rapids", XXX, XXX, XXX},
                                {true, false, true, false, false, false}};

  auto res = segmented_reduce(input,
                              column_view(offsets),
                              *make_max_aggregation<segmented_reduce_aggregation>(),
                              output_dtype,
                              null_policy::INCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);
}
}

#undef XXX

}  // namespace test
}  // namespace cudf
