/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/cudf_gtest.hpp>
#include <utilities/error_utils.hpp>

#include <gmock/gmock.h>

#include <thrust/device_vector.h>

struct CountBitmaskTest : public cudf::test::BaseFixture {};

TEST_F(CountBitmaskTest, NullMask) {
  EXPECT_EQ(0, cudf::count_set_bits(nullptr, 0, 32));
}

TEST_F(CountBitmaskTest, NegativeStart) {
  thrust::device_vector<cudf::bitmask_type> mask(1, 0);
  EXPECT_THROW(cudf::count_set_bits(mask.data().get(), -1, 32),
               cudf::logic_error);
}

TEST_F(CountBitmaskTest, StartLargerThanStop) {
  thrust::device_vector<cudf::bitmask_type> mask(1, 0);
  EXPECT_THROW(cudf::count_set_bits(mask.data().get(), 32, 31),
               cudf::logic_error);
}

TEST_F(CountBitmaskTest, EmptyRange) {
  thrust::device_vector<cudf::bitmask_type> mask(1, 0);
  EXPECT_EQ(0, cudf::count_set_bits(mask.data().get(), 17, 17));
}

TEST_F(CountBitmaskTest, SingleWordAllZero) {
  thrust::device_vector<cudf::bitmask_type> mask(1, 0);
  EXPECT_EQ(0, cudf::count_set_bits(mask.data().get(), 0, 32));
}

TEST_F(CountBitmaskTest, SingleBitAllZero) {
  thrust::device_vector<cudf::bitmask_type> mask(1, 0);
  EXPECT_EQ(0, cudf::count_set_bits(mask.data().get(), 17, 18));
}

TEST_F(CountBitmaskTest, SingleWordAllBitsSet) {
  thrust::device_vector<cudf::bitmask_type> mask(1, ~cudf::bitmask_type{0});
  EXPECT_EQ(32, cudf::count_set_bits(mask.data().get(), 0, 32));
}

TEST_F(CountBitmaskTest, SingleWordPreSlack) {
  thrust::device_vector<cudf::bitmask_type> mask(1, ~cudf::bitmask_type{0});
  EXPECT_EQ(25, cudf::count_set_bits(mask.data().get(), 7, 32));
}

TEST_F(CountBitmaskTest, SingleWordPostSlack) {
  thrust::device_vector<cudf::bitmask_type> mask(1, ~cudf::bitmask_type{0});
  EXPECT_EQ(17, cudf::count_set_bits(mask.data().get(), 0, 17));
}

TEST_F(CountBitmaskTest, SingleWordSubset) {
  thrust::device_vector<cudf::bitmask_type> mask(1, ~cudf::bitmask_type{0});
  EXPECT_EQ(30, cudf::count_set_bits(mask.data().get(), 1, 31));
}

TEST_F(CountBitmaskTest, SingleWordSubset2) {
  thrust::device_vector<cudf::bitmask_type> mask(1, ~cudf::bitmask_type{0});
  EXPECT_EQ(28, cudf::count_set_bits(mask.data().get(), 2, 30));
}

TEST_F(CountBitmaskTest, CountSingleBit) {
  thrust::device_vector<cudf::bitmask_type> mask(1, ~cudf::bitmask_type{0});
  EXPECT_EQ(1, cudf::count_set_bits(mask.data().get(), 13, 14));
}