/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
#include <algorithm>
#include <iostream>

#include <cudf_test/base_fixture.hpp>

#include <cudf/null_mask.hpp>

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <cudf/utilities/bit.hpp>

struct valid_bit_functor {
  cudf::bitmask_type const* _null_mask;
  __device__ bool operator()(cudf::size_type element_index) const noexcept
  {
    return cudf::bit_is_set(_null_mask, element_index);
  }
};

std::ostream& operator<<(std::ostream& stream, thrust::host_vector<bool> const& bits)
{
  for (auto _bit : bits) stream << int(_bit);
  return stream;
}

struct SetBitmaskTest : public cudf::test::BaseFixture {
  void expect_bitmask_equal(cudf::bitmask_type const* bitmask,  // Device Ptr
                            cudf::size_type start_bit,
                            thrust::host_vector<bool> const& expect)
  {
    auto itb_dev = thrust::make_transform_iterator(thrust::counting_iterator<cudf::size_type>{0},
                                                   valid_bit_functor{bitmask});
    thrust::device_vector<bool> result(itb_dev + start_bit, itb_dev + start_bit + expect.size());
    thrust::host_vector<bool> host_result(result);
    EXPECT_THAT(host_result, testing::ElementsAreArray(expect));
  }

  void test_set_null_range(cudf::size_type size,
                           cudf::size_type begin,
                           cudf::size_type end,
                           bool valid)
  {
    thrust::host_vector<bool> expected(end - begin, valid);
    // TEST
    rmm::device_buffer mask = create_null_mask(size, cudf::mask_state::UNINITIALIZED);
    // valid ? cudf::mask_state::ALL_NULL : cudf::mask_state::ALL_VALID);
    cudf::set_null_mask(static_cast<cudf::bitmask_type*>(mask.data()), begin, end, valid);
    expect_bitmask_equal(static_cast<cudf::bitmask_type*>(mask.data()), begin, expected);
  }

  void test_null_partition(cudf::size_type size, cudf::size_type middle, bool valid)
  {
    thrust::host_vector<bool> expected(size);
    std::generate(expected.begin(), expected.end(), [n = 0, middle, valid]() mutable {
      auto i = n++;
      return (!valid) ^ (i < middle);
    });
    // TEST
    rmm::device_buffer mask = create_null_mask(size, cudf::mask_state::UNINITIALIZED);
    cudf::set_null_mask(static_cast<cudf::bitmask_type*>(mask.data()), 0, middle, valid);
    cudf::set_null_mask(static_cast<cudf::bitmask_type*>(mask.data()), middle, size, !valid);
    expect_bitmask_equal(static_cast<cudf::bitmask_type*>(mask.data()), 0, expected);
  }
};

// tests for set_null_mask
TEST_F(SetBitmaskTest, fill_range)
{
  cudf::size_type size = 121;
  for (auto begin = 0; begin < size; begin += 5)
    for (auto end = begin + 1; end <= size; end += 7) {
      this->test_set_null_range(size, begin, end, true);
      this->test_set_null_range(size, begin, end, false);
    }
}

TEST_F(SetBitmaskTest, null_mask_partition)
{
  cudf::size_type size = 64;
  for (auto middle = 1; middle < size; middle++) {
    this->test_null_partition(size, middle, true);
    this->test_null_partition(size, middle, false);
  }
}

TEST_F(SetBitmaskTest, error_range)
{
  cudf::size_type size = 121;
  using size_pair      = std::pair<cudf::size_type, cudf::size_type>;
  std::vector<size_pair> begin_end_fail{
    {-1, size},  // begin>=0
    {-2, -1},    // begin>=0
    {8, 8},      // begin<end
    {9, 8},      // begin<end
  };
  for (auto begin_end : begin_end_fail) {
    auto begin = begin_end.first, end = begin_end.second;
    EXPECT_ANY_THROW(this->test_set_null_range(size, begin, end, true));
    EXPECT_ANY_THROW(this->test_set_null_range(size, begin, end, false));
  }
  std::vector<size_pair> begin_end_pass{
    {0, size},         // begin>=0
    {0, 1},            // begin>=0
    {8, 9},            // begin<end
    {size - 1, size},  // begin<end
  };
  for (auto begin_end : begin_end_pass) {
    auto begin = begin_end.first, end = begin_end.second;
    EXPECT_NO_THROW(this->test_set_null_range(size, begin, end, true));
    EXPECT_NO_THROW(this->test_set_null_range(size, begin, end, false));
  }
}
