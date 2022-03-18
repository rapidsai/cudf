/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <cudf/copying.hpp>
#include <cudf/detail/is_element_valid.hpp>
#include <cudf/detail/iterator.cuh>

#include <thrust/iterator/counting_iterator.h>

namespace cudf {
namespace test {

struct IsElementValidTest : public BaseFixture {
};

TEST_F(IsElementValidTest, IsElementValidBasic)
{
  fixed_width_column_wrapper<int32_t> col({1, 1, 1, 1, 1}, {1, 0, 0, 0, 1});
  EXPECT_TRUE(cudf::detail::is_element_valid_sync(col, 0));
  EXPECT_FALSE(cudf::detail::is_element_valid_sync(col, 1));
  EXPECT_FALSE(cudf::detail::is_element_valid_sync(col, 2));
  EXPECT_FALSE(cudf::detail::is_element_valid_sync(col, 3));
  EXPECT_TRUE(cudf::detail::is_element_valid_sync(col, 4));
}

TEST_F(IsElementValidTest, IsElementValidLarge)
{
  auto filter        = [](auto i) { return static_cast<bool>(i % 3); };
  auto val           = thrust::make_counting_iterator(0);
  auto valid         = cudf::detail::make_counting_transform_iterator(0, filter);
  size_type num_rows = 1000;

  fixed_width_column_wrapper<int32_t> col(val, val + num_rows, valid);

  for (int i = 0; i < num_rows; i++) {
    EXPECT_EQ(cudf::detail::is_element_valid_sync(col, i), filter(i));
  }
}

TEST_F(IsElementValidTest, IsElementValidOffset)
{
  fixed_width_column_wrapper<int32_t> col({1, 1, 1, 1, 1}, {1, 0, 0, 0, 1});
  {
    auto offset_col = slice(col, {1, 5}).front();
    EXPECT_FALSE(cudf::detail::is_element_valid_sync(offset_col, 0));
    EXPECT_FALSE(cudf::detail::is_element_valid_sync(offset_col, 1));
    EXPECT_FALSE(cudf::detail::is_element_valid_sync(offset_col, 2));
    EXPECT_TRUE(cudf::detail::is_element_valid_sync(offset_col, 3));
  }
  {
    auto offset_col = slice(col, {2, 5}).front();
    EXPECT_FALSE(cudf::detail::is_element_valid_sync(offset_col, 0));
    EXPECT_FALSE(cudf::detail::is_element_valid_sync(offset_col, 1));
    EXPECT_TRUE(cudf::detail::is_element_valid_sync(offset_col, 2));
  }
}

TEST_F(IsElementValidTest, IsElementValidOffsetLarge)
{
  auto filter        = [](auto i) { return static_cast<bool>(i % 3); };
  size_type offset   = 37;
  auto val           = thrust::make_counting_iterator(0);
  auto valid         = cudf::detail::make_counting_transform_iterator(0, filter);
  size_type num_rows = 1000;

  fixed_width_column_wrapper<int32_t> col(val, val + num_rows, valid);
  auto offset_col = slice(col, {offset, num_rows}).front();

  for (int i = 0; i < offset_col.size(); i++) {
    EXPECT_EQ(cudf::detail::is_element_valid_sync(offset_col, i), filter(i + offset));
  }
}

}  // namespace test

}  // namespace cudf
