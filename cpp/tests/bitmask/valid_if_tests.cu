/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include <cudf/detail/iterator.cuh>
#include <cudf/detail/valid_if.cuh>
#include <cudf/types.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>

struct ValidIfTest : public cudf::test::BaseFixture {
};

struct odds_valid {
  __host__ __device__ bool operator()(cudf::size_type i) { return i % 2; }
};
struct all_valid {
  __host__ __device__ bool operator()(cudf::size_type i) { return true; }
};
struct all_null {
  __host__ __device__ bool operator()(cudf::size_type i) { return false; }
};

TEST_F(ValidIfTest, EmptyRange)
{
  auto actual = cudf::detail::valid_if(
    thrust::make_counting_iterator(0), thrust::make_counting_iterator(0), odds_valid{});
  auto buffer = actual.first;
  EXPECT_EQ(0u, buffer.size());
  EXPECT_EQ(nullptr, buffer.data());
  EXPECT_EQ(0, actual.second);
}

TEST_F(ValidIfTest, InvalidRange)
{
  EXPECT_THROW(
    cudf::detail::valid_if(
      thrust::make_counting_iterator(1), thrust::make_counting_iterator(0), odds_valid{}),
    cudf::logic_error);
}

TEST_F(ValidIfTest, OddsValid)
{
  auto iter     = cudf::detail::make_counting_transform_iterator(0, odds_valid{});
  auto expected = cudf::test::detail::make_null_mask(iter, iter + 10000);
  auto actual   = cudf::detail::valid_if(
    thrust::make_counting_iterator(0), thrust::make_counting_iterator(10000), odds_valid{});
  CUDF_TEST_EXPECT_EQUAL_BUFFERS(expected.data(), actual.first.data(), expected.size());
  EXPECT_EQ(5000, actual.second);
}

TEST_F(ValidIfTest, AllValid)
{
  auto iter     = cudf::detail::make_counting_transform_iterator(0, all_valid{});
  auto expected = cudf::test::detail::make_null_mask(iter, iter + 10000);
  auto actual   = cudf::detail::valid_if(
    thrust::make_counting_iterator(0), thrust::make_counting_iterator(10000), all_valid{});
  CUDF_TEST_EXPECT_EQUAL_BUFFERS(expected.data(), actual.first.data(), expected.size());
  EXPECT_EQ(0, actual.second);
}

TEST_F(ValidIfTest, AllNull)
{
  auto iter     = cudf::detail::make_counting_transform_iterator(0, all_null{});
  auto expected = cudf::test::detail::make_null_mask(iter, iter + 10000);
  auto actual   = cudf::detail::valid_if(
    thrust::make_counting_iterator(0), thrust::make_counting_iterator(10000), all_null{});
  CUDF_TEST_EXPECT_EQUAL_BUFFERS(expected.data(), actual.first.data(), expected.size());
  EXPECT_EQ(10000, actual.second);
}
