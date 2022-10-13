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
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/types.hpp>

#include <thrust/device_vector.h>

#include <limits>

namespace cudf {
namespace test {

#define XXX 0  // null placeholder

template <typename T>
struct SegmentedReductionTest : public cudf::test::BaseFixture {
};

struct SegmentedReductionTestUntyped : public cudf::test::BaseFixture {
};

TYPED_TEST_CASE(SegmentedReductionTest, NumericTypes);

TYPED_TEST(SegmentedReductionTest, SumExcludeNulls)
{
  // [1, 2, 3], [1, null, 3], [1], [null], [null, null], []
  // values:   {1, 2, 3, 1, XXX, 3, 1, XXX, XXX, XXX}
  // offsets:  {0, 3, 6, 7, 8, 10, 10}
  // nullmask: {1, 1, 1, 1, 0, 1, 1, 0, 0, 0}
  // outputs:  {6, 4, 1, XXX, XXX, XXX}
  // output nullmask: {1, 1, 1, 0, 0, 0}
  auto const input   = fixed_width_column_wrapper<TypeParam>{{1, 2, 3, 1, XXX, 3, 1, XXX, XXX, XXX},
                                                           {1, 1, 1, 1, 0, 1, 1, 0, 0, 0}};
  auto const offsets = std::vector<size_type>{0, 3, 6, 7, 8, 10, 10};
  auto const d_offsets = thrust::device_vector<size_type>(offsets);
  auto const expect =
    fixed_width_column_wrapper<TypeParam>{{6, 4, 1, XXX, XXX, XXX}, {1, 1, 1, 0, 0, 0}};

  auto res = segmented_reduce(input,
                              d_offsets,
                              *make_sum_aggregation<segmented_reduce_aggregation>(),
                              data_type{type_to_id<TypeParam>()},
                              null_policy::EXCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);

  // Test with initial value
  auto const init_scalar = cudf::make_fixed_width_scalar<TypeParam>(3);
  auto const init_expect =
    fixed_width_column_wrapper<TypeParam>{{9, 7, 4, 3, 3, 3}, {1, 1, 1, 1, 1, 1}};

  res = segmented_reduce(input,
                         d_offsets,
                         *make_sum_aggregation<segmented_reduce_aggregation>(),
                         data_type{type_to_id<TypeParam>()},
                         null_policy::EXCLUDE,
                         *init_scalar);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, init_expect);

  // Test with null initial value
  init_scalar->set_valid_async(false);
  res = segmented_reduce(input,
                         d_offsets,
                         *make_sum_aggregation<segmented_reduce_aggregation>(),
                         data_type{type_to_id<TypeParam>()},
                         null_policy::EXCLUDE,
                         *init_scalar);
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
  auto const input   = fixed_width_column_wrapper<TypeParam>{{1, 3, 5, XXX, 3, 5, 1, XXX, XXX, XXX},
                                                           {1, 1, 1, 0, 1, 1, 1, 0, 0, 0}};
  auto const offsets = std::vector<size_type>{0, 3, 6, 7, 8, 10, 10};
  auto const d_offsets = thrust::device_vector<size_type>(offsets);
  auto const expect =
    fixed_width_column_wrapper<TypeParam>{{15, 15, 1, XXX, XXX, XXX}, {1, 1, 1, 0, 0, 0}};

  auto res = segmented_reduce(input,
                              d_offsets,
                              *make_product_aggregation<segmented_reduce_aggregation>(),
                              data_type{type_to_id<TypeParam>()},
                              null_policy::EXCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);

  // Test with initial value
  auto const init_scalar = cudf::make_fixed_width_scalar<TypeParam>(3);
  auto const init_expect =
    fixed_width_column_wrapper<TypeParam>{{45, 45, 3, 3, 3, 3}, {1, 1, 1, 1, 1, 1}};

  res = segmented_reduce(input,
                         d_offsets,
                         *make_product_aggregation<segmented_reduce_aggregation>(),
                         data_type{type_to_id<TypeParam>()},
                         null_policy::EXCLUDE,
                         *init_scalar);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, init_expect);

  // Test with null initial value
  init_scalar->set_valid_async(false);
  res = segmented_reduce(input,
                         d_offsets,
                         *make_product_aggregation<segmented_reduce_aggregation>(),
                         data_type{type_to_id<TypeParam>()},
                         null_policy::EXCLUDE,
                         *init_scalar);
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
  auto const input   = fixed_width_column_wrapper<TypeParam>{{1, 2, 3, 1, XXX, 3, 1, XXX, XXX, XXX},
                                                           {1, 1, 1, 1, 0, 1, 1, 0, 0, 0}};
  auto const offsets = std::vector<size_type>{0, 3, 6, 7, 8, 10, 10};
  auto const d_offsets = thrust::device_vector<size_type>(offsets);
  auto const expect =
    fixed_width_column_wrapper<TypeParam>{{3, 3, 1, XXX, XXX, XXX}, {1, 1, 1, 0, 0, 0}};

  auto res = segmented_reduce(input,
                              d_offsets,
                              *make_max_aggregation<segmented_reduce_aggregation>(),
                              data_type{type_to_id<TypeParam>()},
                              null_policy::EXCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);

  // Test with initial value
  auto const init_scalar = cudf::make_fixed_width_scalar<TypeParam>(2);
  auto const init_expect =
    fixed_width_column_wrapper<TypeParam>{{3, 3, 2, 2, 2, 2}, {1, 1, 1, 1, 1, 1}};

  res = segmented_reduce(input,
                         d_offsets,
                         *make_max_aggregation<segmented_reduce_aggregation>(),
                         data_type{type_to_id<TypeParam>()},
                         null_policy::EXCLUDE,
                         *init_scalar);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, init_expect);

  // Test with null initial value
  init_scalar->set_valid_async(false);
  res = segmented_reduce(input,
                         d_offsets,
                         *make_max_aggregation<segmented_reduce_aggregation>(),
                         data_type{type_to_id<TypeParam>()},
                         null_policy::EXCLUDE,
                         *init_scalar);
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
  auto const input   = fixed_width_column_wrapper<TypeParam>{{1, 2, 3, 1, XXX, 3, 1, XXX, XXX, XXX},
                                                           {1, 1, 1, 1, 0, 1, 1, 0, 0, 0}};
  auto const offsets = std::vector<size_type>{0, 3, 6, 7, 8, 10, 10};
  auto const d_offsets = thrust::device_vector<size_type>(offsets);
  auto const expect =
    fixed_width_column_wrapper<TypeParam>{{1, 1, 1, XXX, XXX, XXX}, {1, 1, 1, 0, 0, 0}};

  auto res = segmented_reduce(input,
                              d_offsets,
                              *make_min_aggregation<segmented_reduce_aggregation>(),
                              data_type{type_to_id<TypeParam>()},
                              null_policy::EXCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);

  // Test with initial value
  auto const init_scalar = cudf::make_fixed_width_scalar<TypeParam>(2);
  auto const init_expect =
    fixed_width_column_wrapper<TypeParam>{{1, 1, 1, 2, 2, 2}, {1, 1, 1, 1, 1, 1}};

  res = segmented_reduce(input,
                         d_offsets,
                         *make_min_aggregation<segmented_reduce_aggregation>(),
                         data_type{type_to_id<TypeParam>()},
                         null_policy::EXCLUDE,
                         *init_scalar);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, init_expect);

  // Test with null initial value
  init_scalar->set_valid_async(false);
  res = segmented_reduce(input,
                         d_offsets,
                         *make_min_aggregation<segmented_reduce_aggregation>(),
                         data_type{type_to_id<TypeParam>()},
                         null_policy::EXCLUDE,
                         *init_scalar);
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
  auto const input = fixed_width_column_wrapper<TypeParam>{
    {0, 0, 0, 0, XXX, 0, 0, 1, 0, 1, XXX, 0, 0, 1, XXX, XXX, XXX},
    {1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0}};
  auto const offsets   = std::vector<size_type>{0, 3, 6, 9, 12, 12, 13, 14, 15, 17};
  auto const d_offsets = thrust::device_vector<size_type>(offsets);
  auto const expect    = fixed_width_column_wrapper<bool>{
    {false, false, true, true, bool{XXX}, false, true, bool{XXX}, bool{XXX}},
    {true, true, true, true, false, true, true, false, false}};

  auto res = segmented_reduce(input,
                              d_offsets,
                              *make_any_aggregation<segmented_reduce_aggregation>(),
                              data_type{type_id::BOOL8},
                              null_policy::EXCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);

  // Test with initial value
  auto const init_scalar = cudf::make_fixed_width_scalar<TypeParam>(1);
  auto const init_expect =
    fixed_width_column_wrapper<bool>{{true, true, true, true, true, true, true, true, true},
                                     {true, true, true, true, true, true, true, true, true}};

  res = segmented_reduce(input,
                         d_offsets,
                         *make_any_aggregation<segmented_reduce_aggregation>(),
                         data_type{type_id::BOOL8},
                         null_policy::EXCLUDE,
                         *init_scalar);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, init_expect);

  // Test with null initial value
  init_scalar->set_valid_async(false);
  res = segmented_reduce(input,
                         d_offsets,
                         *make_any_aggregation<segmented_reduce_aggregation>(),
                         data_type{type_id::BOOL8},
                         null_policy::EXCLUDE,
                         *init_scalar);
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
  auto const input = fixed_width_column_wrapper<TypeParam>{
    {1, 2, 3, 1, XXX, 3, 1, XXX, XXX, XXX, 1, 0, 3, 1, XXX, 0, 0},
    {1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1}};
  auto const offsets   = std::vector<size_type>{0, 3, 6, 6, 7, 8, 10, 13, 16, 17};
  auto const d_offsets = thrust::device_vector<size_type>(offsets);
  auto const expect    = fixed_width_column_wrapper<bool>{
    {true, true, bool{XXX}, true, bool{XXX}, bool{XXX}, false, false, false},
    {true, true, false, true, false, false, true, true, true}};

  auto res = segmented_reduce(input,
                              d_offsets,
                              *make_all_aggregation<segmented_reduce_aggregation>(),
                              data_type{type_id::BOOL8},
                              null_policy::EXCLUDE);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);

  // Test with initial value
  auto const init_scalar = cudf::make_fixed_width_scalar<TypeParam>(1);
  auto const init_expect =
    fixed_width_column_wrapper<bool>{{true, true, true, true, true, true, false, false, false},
                                     {true, true, true, true, true, true, true, true, true}};

  res = segmented_reduce(input,
                         d_offsets,
                         *make_all_aggregation<segmented_reduce_aggregation>(),
                         data_type{type_id::BOOL8},
                         null_policy::EXCLUDE,
                         *init_scalar);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, init_expect);

  // Test with null initial value
  init_scalar->set_valid_async(false);
  res = segmented_reduce(input,
                         d_offsets,
                         *make_all_aggregation<segmented_reduce_aggregation>(),
                         data_type{type_id::BOOL8},
                         null_policy::EXCLUDE,
                         *init_scalar);
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
  auto const input   = fixed_width_column_wrapper<TypeParam>{{1, 2, 3, 1, XXX, 3, 1, XXX, XXX, XXX},
                                                           {1, 1, 1, 1, 0, 1, 1, 0, 0, 0}};
  auto const offsets = std::vector<size_type>{0, 3, 6, 7, 8, 10, 10};
  auto const d_offsets = thrust::device_vector<size_type>(offsets);
  auto const expect =
    fixed_width_column_wrapper<TypeParam>{{6, XXX, 1, XXX, XXX, XXX}, {1, 0, 1, 0, 0, 0}};

  auto res = segmented_reduce(input,
                              d_offsets,
                              *make_sum_aggregation<segmented_reduce_aggregation>(),
                              data_type{type_to_id<TypeParam>()},
                              null_policy::INCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);

  // Test with initial value
  auto const init_scalar = cudf::make_fixed_width_scalar<TypeParam>(3);
  auto const init_expect =
    fixed_width_column_wrapper<TypeParam>{{9, XXX, 4, XXX, XXX, 3}, {1, 0, 1, 0, 0, 1}};

  res = segmented_reduce(input,
                         d_offsets,
                         *make_sum_aggregation<segmented_reduce_aggregation>(),
                         data_type{type_to_id<TypeParam>()},
                         null_policy::INCLUDE,
                         *init_scalar);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, init_expect);

  // Test with null initial value
  init_scalar->set_valid_async(false);
  auto null_init_expect =
    fixed_width_column_wrapper<TypeParam>{{XXX, XXX, XXX, XXX, XXX, XXX}, {0, 0, 0, 0, 0, 0}};

  res = segmented_reduce(input,
                         d_offsets,
                         *make_sum_aggregation<segmented_reduce_aggregation>(),
                         data_type{type_to_id<TypeParam>()},
                         null_policy::INCLUDE,
                         *init_scalar);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, null_init_expect);
}

TYPED_TEST(SegmentedReductionTest, ProductIncludeNulls)
{
  // [1, 3, 5], [null, 3, 5], [1], [null], [null, null], []
  // values:    {1, 3, 5, XXX, 3, 5, 1, XXX, XXX, XXX}
  // offsets:   {0, 3, 6, 7, 8, 10, 10}
  // nullmask:  {1, 1, 1, 0, 1, 1, 1, 0, 0, 0}
  // outputs:   {15, XXX, 1, XXX, XXX, XXX}
  // output nullmask: {1, 0, 1, 0, 0, 0}
  auto const input   = fixed_width_column_wrapper<TypeParam>{{1, 3, 5, XXX, 3, 5, 1, XXX, XXX, XXX},
                                                           {1, 1, 1, 0, 1, 1, 1, 0, 0, 0}};
  auto const offsets = std::vector<size_type>{0, 3, 6, 7, 8, 10, 10};
  auto const d_offsets = thrust::device_vector<size_type>(offsets);
  auto const expect =
    fixed_width_column_wrapper<TypeParam>{{15, XXX, 1, XXX, XXX, XXX}, {1, 0, 1, 0, 0, 0}};

  auto res = segmented_reduce(input,
                              d_offsets,
                              *make_product_aggregation<segmented_reduce_aggregation>(),
                              data_type{type_to_id<TypeParam>()},
                              null_policy::INCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);

  // Test with initial value
  auto const init_scalar = cudf::make_fixed_width_scalar<TypeParam>(3);
  auto const init_expect =
    fixed_width_column_wrapper<TypeParam>{{45, XXX, 3, XXX, XXX, 3}, {1, 0, 1, 0, 0, 1}};

  res = segmented_reduce(input,
                         d_offsets,
                         *make_product_aggregation<segmented_reduce_aggregation>(),
                         data_type{type_to_id<TypeParam>()},
                         null_policy::INCLUDE,
                         *init_scalar);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, init_expect);

  // Test with null initial value
  init_scalar->set_valid_async(false);
  auto null_init_expect =
    fixed_width_column_wrapper<TypeParam>{{XXX, XXX, XXX, XXX, XXX, XXX}, {0, 0, 0, 0, 0, 0}};

  res = segmented_reduce(input,
                         d_offsets,
                         *make_product_aggregation<segmented_reduce_aggregation>(),
                         data_type{type_to_id<TypeParam>()},
                         null_policy::INCLUDE,
                         *init_scalar);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, null_init_expect);
}

TYPED_TEST(SegmentedReductionTest, MaxIncludeNulls)
{
  // [1, 2, 3], [1, null, 3], [1], [null], [null, null], []
  // values:    {1, 2, 3, 1, XXX, 3, 1, XXX, XXX, XXX}
  // offsets:   {0, 3, 6, 7, 8, 10, 10}
  // nullmask:  {1, 1, 1, 1, 0, 1, 1, 0, 0, 0}
  // outputs:   {3, XXX, 1, XXX, XXX, XXX}
  // output nullmask: {1, 0, 1, 0, 0, 0}
  auto const input   = fixed_width_column_wrapper<TypeParam>{{1, 2, 3, 1, XXX, 3, 1, XXX, XXX, XXX},
                                                           {1, 1, 1, 1, 0, 1, 1, 0, 0, 0}};
  auto const offsets = std::vector<size_type>{0, 3, 6, 7, 8, 10, 10};
  auto const d_offsets = thrust::device_vector<size_type>(offsets);
  auto const expect =
    fixed_width_column_wrapper<TypeParam>{{3, XXX, 1, XXX, XXX, XXX}, {1, 0, 1, 0, 0, 0}};

  auto res = segmented_reduce(input,
                              d_offsets,
                              *make_max_aggregation<segmented_reduce_aggregation>(),
                              data_type{type_to_id<TypeParam>()},
                              null_policy::INCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);

  // Test with initial value
  auto const init_scalar = cudf::make_fixed_width_scalar<TypeParam>(2);
  auto const init_expect =
    fixed_width_column_wrapper<TypeParam>{{3, XXX, 2, XXX, XXX, 2}, {1, 0, 1, 0, 0, 1}};

  res = segmented_reduce(input,
                         d_offsets,
                         *make_max_aggregation<segmented_reduce_aggregation>(),
                         data_type{type_to_id<TypeParam>()},
                         null_policy::INCLUDE,
                         *init_scalar);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, init_expect);

  // Test with null initial value
  init_scalar->set_valid_async(false);
  auto null_init_expect =
    fixed_width_column_wrapper<TypeParam>{{XXX, XXX, XXX, XXX, XXX, XXX}, {0, 0, 0, 0, 0, 0}};

  res = segmented_reduce(input,
                         d_offsets,
                         *make_max_aggregation<segmented_reduce_aggregation>(),
                         data_type{type_to_id<TypeParam>()},
                         null_policy::INCLUDE,
                         *init_scalar);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, null_init_expect);
}

TYPED_TEST(SegmentedReductionTest, MinIncludeNulls)
{
  // [1, 2, 3], [1, null, 3], [1], [null], [null, null], []
  // values:   {1, 2, 3, 1, XXX, 3, 1, XXX, XXX}
  // offsets:  {0, 3, 6, 7, 8, 10, 10}
  // nullmask: {1, 1, 1, 1, 0, 1, 1, 0, 0}
  // outputs:  {1, XXX, 1, XXX, XXX, XXX}
  // output nullmask: {1, 0, 1, 0, 0, 0}
  auto const input   = fixed_width_column_wrapper<TypeParam>{{1, 2, 3, 1, XXX, 3, 1, XXX, XXX, XXX},
                                                           {1, 1, 1, 1, 0, 1, 1, 0, 0}};
  auto const offsets = std::vector<size_type>{0, 3, 6, 7, 8, 10, 10};
  auto const d_offsets = thrust::device_vector<size_type>(offsets);
  auto const expect =
    fixed_width_column_wrapper<TypeParam>{{1, XXX, 1, XXX, XXX, XXX}, {1, 0, 1, 0, 0, 0}};

  auto res = segmented_reduce(input,
                              d_offsets,
                              *make_min_aggregation<segmented_reduce_aggregation>(),
                              data_type{type_to_id<TypeParam>()},
                              null_policy::INCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);

  // Test with initial value
  auto const init_scalar = cudf::make_fixed_width_scalar<TypeParam>(2);
  auto const init_expect =
    fixed_width_column_wrapper<TypeParam>{{1, XXX, 1, XXX, XXX, 2}, {1, 0, 1, 0, 0, 1}};

  res = segmented_reduce(input,
                         d_offsets,
                         *make_min_aggregation<segmented_reduce_aggregation>(),
                         data_type{type_to_id<TypeParam>()},
                         null_policy::INCLUDE,
                         *init_scalar);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, init_expect);

  // Test with null initial value
  init_scalar->set_valid_async(false);
  auto null_init_expect =
    fixed_width_column_wrapper<TypeParam>{{XXX, XXX, XXX, XXX, XXX, XXX}, {0, 0, 0, 0, 0, 0}};

  res = segmented_reduce(input,
                         d_offsets,
                         *make_min_aggregation<segmented_reduce_aggregation>(),
                         data_type{type_to_id<TypeParam>()},
                         null_policy::INCLUDE,
                         *init_scalar);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, null_init_expect);
}

TYPED_TEST(SegmentedReductionTest, AnyIncludeNulls)
{
  // [0, 0, 0], [0, null, 0], [0, 1, 0], [1, null, 0], [], [0], [1], [null], [null, null]
  // values:  {0, 0, 0, 0, XXX, 0, 0, 1, 0, 1, XXX, 0, 0, 1, XXX, XXX, XXX}
  // offsets: {0, 3, 6, 9, 12, 12, 13, 14, 15, 17}
  // nullmask:{1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0}
  // outputs: {0, XXX, 1, XXX, XXX, 0, 1, XXX, XXX}
  // output nullmask: {1, 0, 1, 0, 0, 1, 1, 0, 0}
  auto const input = fixed_width_column_wrapper<TypeParam>{
    {0, 0, 0, 0, XXX, 0, 0, 1, 0, 1, XXX, 0, 0, 1, XXX, XXX, XXX},
    {1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0}};
  auto const offsets   = std::vector<size_type>{0, 3, 6, 9, 12, 12, 13, 14, 15, 17};
  auto const d_offsets = thrust::device_vector<size_type>(offsets);
  auto const expect    = fixed_width_column_wrapper<bool>{
    {false, bool{XXX}, true, bool{XXX}, bool{XXX}, false, true, bool{XXX}, bool{XXX}},
    {true, false, true, false, false, true, true, false, false}};

  auto res = segmented_reduce(input,
                              d_offsets,
                              *make_any_aggregation<segmented_reduce_aggregation>(),
                              data_type{type_id::BOOL8},
                              null_policy::INCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);

  // Test with initial value
  auto const init_scalar = cudf::make_fixed_width_scalar<TypeParam>(1);
  auto const init_expect = fixed_width_column_wrapper<bool>{
    {true, bool{XXX}, true, bool{XXX}, true, true, true, bool{XXX}, bool{XXX}},
    {true, false, true, false, true, true, true, false, false}};

  res = segmented_reduce(input,
                         d_offsets,
                         *make_any_aggregation<segmented_reduce_aggregation>(),
                         data_type{type_id::BOOL8},
                         null_policy::INCLUDE,
                         *init_scalar);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, init_expect);

  // Test with null initial value
  init_scalar->set_valid_async(false);
  auto null_init_expect = fixed_width_column_wrapper<bool>{
    {bool{XXX},
     bool{XXX},
     bool{XXX},
     bool{XXX},
     bool{XXX},
     bool{XXX},
     bool{XXX},
     bool{XXX},
     bool{XXX}},
    {false, false, false, false, false, false, false, false, false}};

  res = segmented_reduce(input,
                         d_offsets,
                         *make_any_aggregation<segmented_reduce_aggregation>(),
                         data_type{type_id::BOOL8},
                         null_policy::INCLUDE,
                         *init_scalar);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, null_init_expect);
}

TYPED_TEST(SegmentedReductionTest, AllIncludeNulls)
{
  // [1, 2, 3], [1, null, 3], [], [1], [null], [null, null], [1, 0, 3], [1, null, 0], [0]
  // values: {1, 2, 3, 1, XXX, 3, 1, XXX, XXX, XXX, 1, 0, 3, 1, XXX, 0, 0}
  // offsets: {0, 3, 6, 6, 7, 8, 10, 13, 16, 17}
  // nullmask: {1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1}
  // outputs: {true, XXX, XXX, true, XXX, XXX, false, XXX, false}
  // output nullmask: {1, 0, 0, 1, 0, 0, 1, 0, 1}
  auto const input = fixed_width_column_wrapper<TypeParam>{
    {1, 2, 3, 1, XXX, 3, 1, XXX, XXX, XXX, 1, 0, 3, 1, XXX, 0, 0},
    {1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1}};
  auto const offsets   = std::vector<size_type>{0, 3, 6, 6, 7, 8, 10, 13, 16, 17};
  auto const d_offsets = thrust::device_vector<size_type>(offsets);
  auto const expect    = fixed_width_column_wrapper<bool>{
    {true, bool{XXX}, bool{XXX}, true, bool{XXX}, bool{XXX}, false, bool{XXX}, false},
    {true, false, false, true, false, false, true, false, true}};

  auto res = segmented_reduce(input,
                              d_offsets,
                              *make_all_aggregation<segmented_reduce_aggregation>(),
                              data_type{type_id::BOOL8},
                              null_policy::INCLUDE);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);

  // Test with initial value
  auto const init_scalar = cudf::make_fixed_width_scalar<TypeParam>(1);
  auto const init_expect = fixed_width_column_wrapper<bool>{
    {true, bool{XXX}, true, true, bool{XXX}, bool{XXX}, false, bool{XXX}, false},
    {true, false, true, true, false, false, true, false, true}};

  res = segmented_reduce(input,
                         d_offsets,
                         *make_all_aggregation<segmented_reduce_aggregation>(),
                         data_type{type_id::BOOL8},
                         null_policy::INCLUDE,
                         *init_scalar);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, init_expect);

  // Test with null initial value
  init_scalar->set_valid_async(false);
  auto null_init_expect = fixed_width_column_wrapper<bool>{
    {bool{XXX},
     bool{XXX},
     bool{XXX},
     bool{XXX},
     bool{XXX},
     bool{XXX},
     bool{XXX},
     bool{XXX},
     bool{XXX}},
    {false, false, false, false, false, false, false, false, false}};

  res = segmented_reduce(input,
                         d_offsets,
                         *make_all_aggregation<segmented_reduce_aggregation>(),
                         data_type{type_id::BOOL8},
                         null_policy::INCLUDE,
                         *init_scalar);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, null_init_expect);
}

TEST_F(SegmentedReductionTestUntyped, PartialSegmentReduction)
{
  // Segmented reduction allows offsets only specify part of the input columns.
  // [1], [2, 3], [4]
  // values: {1, 2, 3, 4, 5, 6, 7}
  // offsets: {0, 1, 3, 4}
  // nullmask: {1, 1, 1, 1, 1, 1, 1}
  // outputs: {1, 5, 4}
  // output nullmask: {1, 1, 1}

  auto const input = fixed_width_column_wrapper<int32_t>{
    {1, 2, 3, 4, 5, 6, 7}, {true, true, true, true, true, true, true}};
  auto const offsets   = std::vector<size_type>{1, 3, 4};
  auto const d_offsets = thrust::device_vector<size_type>(offsets);
  auto const expect    = fixed_width_column_wrapper<int32_t>{{5, 4}, {true, true}};

  auto res = segmented_reduce(input,
                              d_offsets,
                              *make_sum_aggregation<segmented_reduce_aggregation>(),
                              data_type{type_id::INT32},
                              null_policy::INCLUDE);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);

  // Test with initial value
  auto const init_scalar = cudf::make_fixed_width_scalar<int32_t>(3);
  auto const init_expect = fixed_width_column_wrapper<int32_t>{{8, 7}, {true, true}};

  res = segmented_reduce(input,
                         d_offsets,
                         *make_sum_aggregation<segmented_reduce_aggregation>(),
                         data_type{type_id::INT32},
                         null_policy::INCLUDE,
                         *init_scalar);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, init_expect);

  // Test with null initial value
  init_scalar->set_valid_async(false);
  auto null_init_expect = fixed_width_column_wrapper<int32_t>{{XXX, XXX}, {false, false}};

  res = segmented_reduce(input,
                         d_offsets,
                         *make_sum_aggregation<segmented_reduce_aggregation>(),
                         data_type{type_id::INT32},
                         null_policy::INCLUDE,
                         *init_scalar);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, null_init_expect);
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

  auto const input     = fixed_width_column_wrapper<int32_t>{1, 2, 3, 4, 5, 6, 7};
  auto const offsets   = std::vector<size_type>{0, 1, 1, 3, 7};
  auto const d_offsets = thrust::device_vector<size_type>(offsets);
  auto const expect =
    fixed_width_column_wrapper<int32_t>{{1, XXX, 5, 22}, {true, false, true, true}};

  auto res = segmented_reduce(input,
                              d_offsets,
                              *make_sum_aggregation<segmented_reduce_aggregation>(),
                              data_type{type_id::INT32},
                              null_policy::INCLUDE);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);

  // Test with initial value
  auto const init_scalar = cudf::make_fixed_width_scalar<int32_t>(3);
  auto const init_expect =
    fixed_width_column_wrapper<int32_t>{{4, 3, 8, 25}, {true, true, true, true}};

  res = segmented_reduce(input,
                         d_offsets,
                         *make_sum_aggregation<segmented_reduce_aggregation>(),
                         data_type{type_id::INT32},
                         null_policy::INCLUDE,
                         *init_scalar);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, init_expect);

  // Test with null initial value
  init_scalar->set_valid_async(false);
  auto null_init_expect =
    fixed_width_column_wrapper<int32_t>{{XXX, XXX, XXX, XXX}, {false, false, false, false}};

  res = segmented_reduce(input,
                         d_offsets,
                         *make_sum_aggregation<segmented_reduce_aggregation>(),
                         data_type{type_id::INT32},
                         null_policy::INCLUDE,
                         *init_scalar);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, null_init_expect);
}

TEST_F(SegmentedReductionTestUntyped, ReduceEmptyColumn)
{
  auto const input     = fixed_width_column_wrapper<int32_t>{};
  auto const offsets   = std::vector<size_type>{0};
  auto const d_offsets = thrust::device_vector<size_type>(offsets);
  auto const expect    = fixed_width_column_wrapper<int32_t>{};

  auto res = segmented_reduce(input,
                              d_offsets,
                              *make_sum_aggregation<segmented_reduce_aggregation>(),
                              data_type{type_to_id<int32_t>()},
                              null_policy::EXCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);

  // Test with initial value
  auto const init_scalar = cudf::make_fixed_width_scalar<int32_t>(3);
  res                    = segmented_reduce(input,
                         d_offsets,
                         *make_sum_aggregation<segmented_reduce_aggregation>(),
                         data_type{type_to_id<int32_t>()},
                         null_policy::EXCLUDE,
                         *init_scalar);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);

  // Test with null initial value
  init_scalar->set_valid_async(false);
  res = segmented_reduce(input,
                         d_offsets,
                         *make_sum_aggregation<segmented_reduce_aggregation>(),
                         data_type{type_to_id<int32_t>()},
                         null_policy::EXCLUDE,
                         *init_scalar);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);
}

TEST_F(SegmentedReductionTestUntyped, EmptyInputWithOffsets)
{
  auto const input     = fixed_width_column_wrapper<int32_t>{};
  auto const offsets   = std::vector<size_type>{0, 0, 0, 0, 0, 0};
  auto const d_offsets = thrust::device_vector<size_type>(offsets);
  auto const expect =
    fixed_width_column_wrapper<int32_t>{{XXX, XXX, XXX, XXX, XXX}, {0, 0, 0, 0, 0}};

  auto aggregates =
    std::vector<std::unique_ptr<cudf::segmented_reduce_aggregation,
                                std::default_delete<cudf::segmented_reduce_aggregation>>>();
  aggregates.push_back(std::move(make_max_aggregation<segmented_reduce_aggregation>()));
  aggregates.push_back(std::move(make_min_aggregation<segmented_reduce_aggregation>()));
  aggregates.push_back(std::move(make_sum_aggregation<segmented_reduce_aggregation>()));
  aggregates.push_back(std::move(make_product_aggregation<segmented_reduce_aggregation>()));

  auto output_type = data_type{type_to_id<int32_t>()};
  for (auto&& agg : aggregates) {
    auto result = segmented_reduce(input, d_offsets, *agg, output_type, null_policy::EXCLUDE);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
  }

  auto const expect_bool =
    fixed_width_column_wrapper<bool>{{XXX, XXX, XXX, XXX, XXX}, {0, 0, 0, 0, 0}};

  auto result = segmented_reduce(input,
                                 d_offsets,
                                 *make_any_aggregation<segmented_reduce_aggregation>(),
                                 data_type{type_id::BOOL8},
                                 null_policy::INCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect_bool);
  result = segmented_reduce(input,
                            d_offsets,
                            *make_all_aggregation<segmented_reduce_aggregation>(),
                            data_type{type_id::BOOL8},
                            null_policy::INCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect_bool);
}

template <typename T>
struct SegmentedReductionFixedPointTest : public cudf::test::BaseFixture {
};

TYPED_TEST_SUITE(SegmentedReductionFixedPointTest, cudf::test::FixedPointTypes);

TYPED_TEST(SegmentedReductionFixedPointTest, MaxIncludeNulls)
{
  // scale: -2, 0, 5
  // [1, 2, 3], [1, null, 3], [1], [null], [null, null], []
  // values:    {1, 2, 3, 1, XXX, 3, 1, XXX, XXX, XXX}
  // offsets:   {0, 3, 6, 7, 8, 10, 10}
  // nullmask:  {1, 1, 1, 1, 0, 1, 1, 0, 0, 0}
  // outputs:   {3, XXX, 1, XXX, XXX, XXX}
  // output nullmask: {1, 0, 1, 0, 0, 0}

  using RepType = device_storage_type_t<TypeParam>;

  for (auto scale : {-2, 0, 5}) {
    auto const input   = fixed_point_column_wrapper<RepType>({1, 2, 3, 1, XXX, 3, 1, XXX, XXX, XXX},
                                                           {1, 1, 1, 1, 0, 1, 1, 0, 0, 0},
                                                           numeric::scale_type{scale});
    auto const offsets = std::vector<size_type>{0, 3, 6, 7, 8, 10, 10};
    auto const d_offsets = thrust::device_vector<size_type>(offsets);
    auto out_type        = column_view(input).type();
    auto const expect    = fixed_point_column_wrapper<RepType>(
      {3, XXX, 1, XXX, XXX, XXX}, {1, 0, 1, 0, 0, 0}, numeric::scale_type{scale});

    auto res = segmented_reduce(input,
                                d_offsets,
                                *make_max_aggregation<segmented_reduce_aggregation>(),
                                out_type,
                                null_policy::INCLUDE);

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);
  }
}

TYPED_TEST(SegmentedReductionFixedPointTest, MaxExcludeNulls)
{
  // scale: -2, 0, 5
  // [1, 2, 3], [1, null, 3], [1], [null], [null, null], []
  // values:    {1, 2, 3, 1, XXX, 3, 1, XXX, XXX, XXX}
  // offsets:   {0, 3, 6, 7, 8, 10, 10}
  // nullmask:  {1, 1, 1, 1, 0, 1, 1, 0, 0, 0}
  // outputs:   {3, 3, 1, XXX, XXX, XXX}
  // output nullmask: {1, 1, 1, 0, 0, 0}

  using RepType = device_storage_type_t<TypeParam>;

  for (auto scale : {-2, 0, 5}) {
    auto const input   = fixed_point_column_wrapper<RepType>({1, 2, 3, 1, XXX, 3, 1, XXX, XXX, XXX},
                                                           {1, 1, 1, 1, 0, 1, 1, 0, 0, 0},
                                                           numeric::scale_type{scale});
    auto const offsets = std::vector<size_type>{0, 3, 6, 7, 8, 10, 10};
    auto const d_offsets = thrust::device_vector<size_type>(offsets);
    auto out_type        = column_view(input).type();
    auto const expect    = fixed_point_column_wrapper<RepType>(
      {3, 3, 1, XXX, XXX, XXX}, {1, 1, 1, 0, 0, 0}, numeric::scale_type{scale});

    auto res = segmented_reduce(input,
                                d_offsets,
                                *make_max_aggregation<segmented_reduce_aggregation>(),
                                out_type,
                                null_policy::EXCLUDE);

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);
  }
}

TYPED_TEST(SegmentedReductionFixedPointTest, MinIncludeNulls)
{
  // scale: -2, 0, 5
  // [1, 2, 3], [1, null, 3], [1], [null], [null, null], []
  // values:    {1, 2, 3, 1, XXX, 3, 1, XXX, XXX, XXX}
  // offsets:   {0, 3, 6, 7, 8, 10, 10}
  // nullmask:  {1, 1, 1, 1, 0, 1, 1, 0, 0, 0}
  // outputs:   {1, XXX, 1, XXX, XXX, XXX}
  // output nullmask: {1, 0, 1, 0, 0, 0}

  using RepType = device_storage_type_t<TypeParam>;

  for (auto scale : {-2, 0, 5}) {
    auto const input   = fixed_point_column_wrapper<RepType>({1, 2, 3, 1, XXX, 3, 1, XXX, XXX, XXX},
                                                           {1, 1, 1, 1, 0, 1, 1, 0, 0, 0},
                                                           numeric::scale_type{scale});
    auto const offsets = std::vector<size_type>{0, 3, 6, 7, 8, 10, 10};
    auto const d_offsets = thrust::device_vector<size_type>(offsets);
    auto out_type        = column_view(input).type();
    auto const expect    = fixed_point_column_wrapper<RepType>(
      {1, XXX, 1, XXX, XXX, XXX}, {1, 0, 1, 0, 0, 0}, numeric::scale_type{scale});

    auto res = segmented_reduce(input,
                                d_offsets,
                                *make_min_aggregation<segmented_reduce_aggregation>(),
                                out_type,
                                null_policy::INCLUDE);

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);
  }
}

TYPED_TEST(SegmentedReductionFixedPointTest, MinExcludeNulls)
{
  // scale: -2, 0, 5
  // [1, 2, 3], [1, null, 3], [1], [null], [null, null], []
  // values:    {1, 2, 3, 1, XXX, 3, 1, XXX, XXX, XXX}
  // offsets:   {0, 3, 6, 7, 8, 10, 10}
  // nullmask:  {1, 1, 1, 1, 0, 1, 1, 0, 0, 0}
  // outputs:   {1, 1, 1, XXX, XXX, XXX}
  // output nullmask: {1, 1, 1, 0, 0, 0}

  using RepType = device_storage_type_t<TypeParam>;

  for (auto scale : {-2, 0, 5}) {
    auto const input   = fixed_point_column_wrapper<RepType>({1, 2, 3, 1, XXX, 3, 1, XXX, XXX, XXX},
                                                           {1, 1, 1, 1, 0, 1, 1, 0, 0, 0},
                                                           numeric::scale_type{scale});
    auto const offsets = std::vector<size_type>{0, 3, 6, 7, 8, 10, 10};
    auto const d_offsets = thrust::device_vector<size_type>(offsets);
    auto out_type        = column_view(input).type();
    auto const expect    = fixed_point_column_wrapper<RepType>(
      {1, 1, 1, XXX, XXX, XXX}, {1, 1, 1, 0, 0, 0}, numeric::scale_type{scale});

    auto res = segmented_reduce(input,
                                d_offsets,
                                *make_min_aggregation<segmented_reduce_aggregation>(),
                                out_type,
                                null_policy::EXCLUDE);

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);
  }
}

TYPED_TEST(SegmentedReductionFixedPointTest, MaxNonNullableInput)
{
  // scale: -2, 0, 5
  // [1, 2, 3], [1], []
  // values:    {1, 2, 3, 1}
  // offsets:   {0, 3, 4}
  // outputs:   {3, 1, XXX}
  // output nullmask: {1, 1, 0}

  using RepType = device_storage_type_t<TypeParam>;

  for (auto scale : {-2, 0, 5}) {
    auto const input =
      fixed_point_column_wrapper<RepType>({1, 2, 3, 1}, numeric::scale_type{scale});
    auto const offsets   = std::vector<size_type>{0, 3, 4, 4};
    auto const d_offsets = thrust::device_vector<size_type>(offsets);
    auto out_type        = column_view(input).type();
    auto const expect =
      fixed_point_column_wrapper<RepType>({3, 1, XXX}, {1, 1, 0}, numeric::scale_type{scale});

    auto include_null_res = segmented_reduce(input,
                                             d_offsets,
                                             *make_max_aggregation<segmented_reduce_aggregation>(),
                                             out_type,
                                             null_policy::INCLUDE);

    auto exclude_null_res = segmented_reduce(input,
                                             d_offsets,
                                             *make_max_aggregation<segmented_reduce_aggregation>(),
                                             out_type,
                                             null_policy::EXCLUDE);

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*include_null_res, expect);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*exclude_null_res, expect);
  }
}

TYPED_TEST(SegmentedReductionFixedPointTest, MinNonNullableInput)
{
  // scale: -2, 0, 5
  // [1, 2, 3], [1], []
  // values:    {1, 2, 3, 1}
  // offsets:   {0, 3, 4}
  // outputs:   {1, 1, XXX}
  // output nullmask: {1, 1, 0}

  using RepType = device_storage_type_t<TypeParam>;

  for (auto scale : {-2, 0, 5}) {
    auto const input =
      fixed_point_column_wrapper<RepType>({1, 2, 3, 1}, numeric::scale_type{scale});
    auto const offsets   = std::vector<size_type>{0, 3, 4, 4};
    auto const d_offsets = thrust::device_vector<size_type>(offsets);
    auto out_type        = column_view(input).type();
    auto const expect =
      fixed_point_column_wrapper<RepType>({1, 1, XXX}, {1, 1, 0}, numeric::scale_type{scale});

    auto include_null_res = segmented_reduce(input,
                                             d_offsets,
                                             *make_min_aggregation<segmented_reduce_aggregation>(),
                                             out_type,
                                             null_policy::INCLUDE);

    auto exclude_null_res = segmented_reduce(input,
                                             d_offsets,
                                             *make_min_aggregation<segmented_reduce_aggregation>(),
                                             out_type,
                                             null_policy::EXCLUDE);

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*include_null_res, expect);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*exclude_null_res, expect);
  }
}

// String min/max test grid
// Segment: Length 0, length 1, length 2
// Element nulls: No nulls, all nulls, some nulls
// String: Empty string,
// Position of the min/max: start of segment, end of segment
// Include null, exclude null

#undef XXX
#define XXX ""  // null placeholder

struct SegmentedReductionStringTest : public cudf::test::BaseFixture {
  std::pair<strings_column_wrapper, fixed_width_column_wrapper<size_type>> input()
  {
    return std::pair(
      strings_column_wrapper{
        {"world", "cudf", XXX, "", "rapids", "i am", "ai", "apples", "zebras", XXX, XXX, XXX},
        {true, true, false, true, true, true, true, true, true, false, false, false}},
      fixed_width_column_wrapper<size_type>{0, 1, 4, 7, 9, 9, 10, 12});
  }
};

TEST_F(SegmentedReductionStringTest, MaxIncludeNulls)
{
  // data: ['world'], ['cudf', NULL, ''], ['rapids', 'i am', 'ai'], ['apples', 'zebras'],
  //       [], [NULL], [NULL, NULL]
  // values:  {"world", "cudf", XXX, "", "rapids", "i am", "ai", "apples", "zebras", XXX, XXX, XXX}
  // nullmask:{1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0}
  // offsets: {0, 1, 4, 7, 9, 9, 10, 12}
  // output_dtype: string dtype
  // outputs: {"world", XXX, "rapids", "zebras", XXX, XXX, XXX}
  // output nullmask: {1, 0, 1, 1, 0, 0, 0}

  auto const [input, offsets] = this->input();
  data_type output_dtype{type_id::STRING};

  strings_column_wrapper expect{{"world", XXX, "rapids", "zebras", XXX, XXX, XXX},
                                {true, false, true, true, false, false, false}};

  auto res = segmented_reduce(input,
                              column_view(offsets),
                              *make_max_aggregation<segmented_reduce_aggregation>(),
                              output_dtype,
                              null_policy::INCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);
}

TEST_F(SegmentedReductionStringTest, MaxExcludeNulls)
{
  // data: ['world'], ['cudf', NULL, ''], ['rapids', 'i am', 'ai'], ['apples', 'zebras'],
  //       [], [NULL], [NULL, NULL]
  // values:  {"world", "cudf", XXX, "", "rapids", "i am", "ai", "apples", "zebras", XXX, XXX, XXX}
  // nullmask:{1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0}
  // offsets: {0, 1, 4, 7, 9, 9, 10, 12}
  // output_dtype: string dtype
  // outputs: {"world", "cudf", "rapids", "zebras", XXX, XXX, XXX}
  // output nullmask: {1, 1, 1, 1, 0, 0, 0}

  auto const [input, offsets] = this->input();
  data_type output_dtype{type_id::STRING};

  strings_column_wrapper expect{{"world", "cudf", "rapids", "zebras", XXX, XXX, XXX},
                                {true, true, true, true, false, false, false}};

  auto res = segmented_reduce(input,
                              column_view(offsets),
                              *make_max_aggregation<segmented_reduce_aggregation>(),
                              output_dtype,
                              null_policy::EXCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);
}

TEST_F(SegmentedReductionStringTest, MinIncludeNulls)
{
  // data: ['world'], ['cudf', NULL, ''], ['rapids', 'i am', 'ai'], ['apples', 'zebras'],
  //       [], [NULL], [NULL, NULL]
  // values:  {"world", "cudf", XXX, "", "rapids", "i am", "ai", "apples", "zebras", XXX, XXX, XXX}
  // nullmask:{1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0}
  // offsets: {0, 1, 4, 7, 9, 9, 10, 12}
  // output_dtype: string dtype
  // outputs: {"world", XXX, "ai", "apples", XXX, XXX, XXX}
  // output nullmask: {1, 0, 1, 1, 0, 0, 0}

  auto const [input, offsets] = this->input();
  data_type output_dtype{type_id::STRING};

  strings_column_wrapper expect{{"world", XXX, "ai", "apples", XXX, XXX, XXX},
                                {true, false, true, true, false, false, false}};

  auto res = segmented_reduce(input,
                              column_view(offsets),
                              *make_min_aggregation<segmented_reduce_aggregation>(),
                              output_dtype,
                              null_policy::INCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);
}

TEST_F(SegmentedReductionStringTest, MinExcludeNulls)
{
  // data: ['world'], ['cudf', NULL, ''], ['rapids', 'i am', 'ai'], ['apples', 'zebras'],
  //       [], [NULL], [NULL, NULL]
  // values:  {"world", "cudf", XXX, "", "rapids", "i am", "ai", "apples", "zebras", XXX, XXX, XXX}
  // nullmask:{1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0}
  // offsets: {0, 1, 4, 7, 9, 9, 10, 12}
  // output_dtype: string dtype
  // outputs: {"world", "", "ai", "apples", XXX, XXX, XXX}
  // output nullmask: {1, 1, 1, 1, 0, 0, 0}

  auto const [input, offsets] = this->input();
  data_type output_dtype{type_id::STRING};

  strings_column_wrapper expect{{"world", "", "ai", "apples", XXX, XXX, XXX},
                                {true, true, true, true, false, false, false}};

  auto res = segmented_reduce(input,
                              column_view(offsets),
                              *make_min_aggregation<segmented_reduce_aggregation>(),
                              output_dtype,
                              null_policy::EXCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);
}

TEST_F(SegmentedReductionStringTest, EmptyInputWithOffsets)
{
  auto const input     = strings_column_wrapper{};
  auto const offsets   = std::vector<size_type>{0, 0, 0, 0};
  auto const d_offsets = thrust::device_vector<size_type>(offsets);
  auto const expect    = strings_column_wrapper({XXX, XXX, XXX}, {0, 0, 0});

  auto result = segmented_reduce(input,
                                 d_offsets,
                                 *make_max_aggregation<segmented_reduce_aggregation>(),
                                 data_type{type_id::STRING},
                                 null_policy::EXCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
  result = segmented_reduce(input,
                            d_offsets,
                            *make_min_aggregation<segmented_reduce_aggregation>(),
                            data_type{type_id::STRING},
                            null_policy::INCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
}

#undef XXX

}  // namespace test
}  // namespace cudf
