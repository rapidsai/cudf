/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS,  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/scan_reduce_iterator.cuh>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/host_vector.h>
#include <thrust/scan.h>

#include <algorithm>

using TestingTypes = cudf::test::IntegralTypesNotBool;

template <typename T>
struct ScanReduceIteratorTest : public cudf::test::BaseFixture {
};

TYPED_TEST_SUITE(ScanReduceIteratorTest, TestingTypes);

TYPED_TEST(ScanReduceIteratorTest, ScanReduce)
{
  using T  = TypeParam;
  using RT = int64_t;

  auto stream = cudf::get_default_stream();

  auto host_values = cudf::test::make_type_param_vector<T>({0, 6, 0, -14, 13, 64, -13, -20, 45});

  auto d_col  = cudf::test::fixed_width_column_wrapper<T>(host_values.begin(), host_values.end());
  auto d_view = cudf::column_view(d_col);

  auto reduction = rmm::device_scalar<RT>(0, stream);
  auto result    = rmm::device_uvector<T>(d_view.size(), stream);
  auto output_itr =
    cudf::detail::make_scan_reduce_output_iterator(result.begin(), result.end(), reduction.data());

  thrust::exclusive_scan(
    rmm::exec_policy(stream), d_view.begin<T>(), d_view.end<T>(), output_itr, RT{0});

  auto expected_values = thrust::host_vector<T>(host_values.size());
  std::exclusive_scan(host_values.begin(), host_values.end(), expected_values.begin(), T{0});
  auto expected_reduce = static_cast<RT>(
    std::reduce(host_values.begin(), host_values.begin() + host_values.size() - 1, T{0}));

  auto expected =
    cudf::test::fixed_width_column_wrapper<T>(expected_values.begin(), expected_values.end());
  auto result_col =
    cudf::column_view(cudf::data_type(cudf::type_to_id<T>()), d_view.size(), result.data());

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result_col, expected);
  EXPECT_EQ(reduction.value(stream), expected_reduce);
}

struct ScanReduceIteratorIntTest : public cudf::test::BaseFixture {
};

TEST_F(ScanReduceIteratorIntTest, ScanWithOverflow)
{
  auto stream = cudf::get_default_stream();

  std::vector<int32_t> host_values(30000, 100000);
  auto d_col =
    cudf::test::fixed_width_column_wrapper<int32_t>(host_values.begin(), host_values.end());
  auto d_view = cudf::column_view(d_col);

  auto reduction = rmm::device_scalar<int64_t>(0, stream);
  auto result    = rmm::device_uvector<int32_t>(d_view.size(), stream);
  auto output_itr =
    cudf::detail::make_scan_reduce_output_iterator(result.begin(), result.end(), reduction.data());

  thrust::exclusive_scan(rmm::exec_policy(stream),
                         d_view.begin<int32_t>(),
                         d_view.end<int32_t>(),
                         output_itr,
                         int64_t{0});

  auto expected = static_cast<int64_t>(
    std::reduce(host_values.begin(), host_values.begin() + host_values.size() - 1, int64_t{0}));
  EXPECT_EQ(reduction.value(stream), expected);
}
