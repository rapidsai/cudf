/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
#include <cudf_test/default_stream.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>

class ReductionTest : public cudf::test::BaseFixture {};

TEST_F(ReductionTest, ReductionSum)
{
  cudf::test::fixed_width_column_wrapper<int> input({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  cudf::reduce(input,
               *cudf::make_sum_aggregation<cudf::reduce_aggregation>(),
               cudf::data_type(cudf::type_id::INT32),
               cudf::test::get_default_stream());
}

TEST_F(ReductionTest, ReductionSumScalarInit)
{
  cudf::test::fixed_width_column_wrapper<int> input({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  auto const init_scalar = cudf::make_fixed_width_scalar<int>(3, cudf::test::get_default_stream());
  cudf::reduce(input,
               *cudf::make_sum_aggregation<cudf::reduce_aggregation>(),
               cudf::data_type(cudf::type_id::INT32),
               *init_scalar,
               cudf::test::get_default_stream());
}

TEST_F(ReductionTest, SegmentedReductionSum)
{
  auto const input     = cudf::test::fixed_width_column_wrapper<int>{{1, 2, 3, 1, 0, 3, 1, 0, 0, 0},
                                                                     {1, 1, 1, 1, 0, 1, 1, 0, 0, 0}};
  auto const offsets   = std::vector<cudf::size_type>{0, 3, 6, 7, 8, 10, 10};
  auto const d_offsets = cudf::detail::make_device_uvector_async(
    offsets, cudf::test::get_default_stream(), rmm::mr::get_current_device_resource());

  auto res =
    cudf::segmented_reduce(input,
                           d_offsets,
                           *cudf::make_sum_aggregation<cudf::segmented_reduce_aggregation>(),
                           cudf::data_type(cudf::type_id::INT32),
                           cudf::null_policy::EXCLUDE,
                           cudf::test::get_default_stream());
}

TEST_F(ReductionTest, SegmentedReductionSumScalarInit)
{
  auto const input     = cudf::test::fixed_width_column_wrapper<int>{{1, 2, 3, 1, 0, 3, 1, 0, 0, 0},
                                                                     {1, 1, 1, 1, 0, 1, 1, 0, 0, 0}};
  auto const offsets   = std::vector<cudf::size_type>{0, 3, 6, 7, 8, 10, 10};
  auto const d_offsets = cudf::detail::make_device_uvector_async(
    offsets, cudf::test::get_default_stream(), rmm::mr::get_current_device_resource());
  auto const init_scalar = cudf::make_fixed_width_scalar<int>(3, cudf::test::get_default_stream());
  auto res =
    cudf::segmented_reduce(input,
                           d_offsets,
                           *cudf::make_sum_aggregation<cudf::segmented_reduce_aggregation>(),
                           cudf::data_type(cudf::type_id::INT32),
                           cudf::null_policy::EXCLUDE,
                           *init_scalar,
                           cudf::test::get_default_stream());
}

TEST_F(ReductionTest, ScanMin)
{
  auto const input = cudf::test::fixed_width_column_wrapper<int>{
    {123, 64, 63, 99, -5, 123, -16, -120, -111}, {1, 0, 1, 1, 1, 1, 0, 0, 1}};

  cudf::scan(input,
             *cudf::make_min_aggregation<cudf::scan_aggregation>(),
             cudf::scan_type::INCLUSIVE,
             cudf::null_policy::EXCLUDE,
             cudf::test::get_default_stream());
}

TEST_F(ReductionTest, MinMax)
{
  auto const input = cudf::test::fixed_width_column_wrapper<int>{
    {123, 64, 63, 99, -5, 123, -16, -120, -111}, {1, 0, 1, 1, 1, 1, 0, 0, 1}};

  cudf::minmax(input, cudf::test::get_default_stream());
}
