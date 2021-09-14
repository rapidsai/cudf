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
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/reshape.hpp>
#include <cudf/table/table_view.hpp>

#include <limits>

namespace cudf {
namespace test {

// template <typename T>
struct OneHotEncodingTest : public BaseFixture {
};

// TYPED_TEST_CASE(OneHotEncodingTest, cudf::test::NumericTypes);

// TYPED_TEST(OneHotEncodingTest, Basic) {
TEST_F(OneHotEncodingTest, Basic)
{
  auto input    = fixed_width_column_wrapper<int32_t>{8, 8, 8, 9, 9};
  auto category = fixed_width_column_wrapper<int32_t>{8, 9};

  auto col0 = fixed_width_column_wrapper<bool>{1, 1, 1, 0, 0};
  auto col1 = fixed_width_column_wrapper<bool>{0, 0, 0, 1, 1};

  auto expected = table_view{{col0, col1}};

  auto [_, got] = one_hot_encoding(input, category);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, got);
}

TEST_F(OneHotEncodingTest, Nulls)
{
  auto input    = fixed_width_column_wrapper<int32_t>{{8, 8, 8, 9, 9}, {1, 1, 0, 1, 1}};
  auto category = fixed_width_column_wrapper<int32_t>({8, 9, -1}, {1, 1, 0});

  auto col0 = fixed_width_column_wrapper<bool>{1, 1, 0, 0, 0};
  auto col1 = fixed_width_column_wrapper<bool>{0, 0, 0, 1, 1};
  auto col2 = fixed_width_column_wrapper<bool>{0, 0, 1, 0, 0};

  auto expected = table_view{{col0, col1, col2}};

  auto [_, got] = one_hot_encoding(input, category);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, got);
}

TEST_F(OneHotEncodingTest, NaNs)
{
  auto const nan = std::numeric_limits<float>::signaling_NaN();

  auto input    = fixed_width_column_wrapper<float>{8.f, 8.f, 8.f, 9.f, nan};
  auto category = fixed_width_column_wrapper<float>{8.f, 9.f, nan};

  auto col0 = fixed_width_column_wrapper<bool>{1, 1, 1, 0, 0};
  auto col1 = fixed_width_column_wrapper<bool>{0, 0, 0, 1, 0};
  auto col2 = fixed_width_column_wrapper<bool>{0, 0, 0, 0, 1};

  auto expected = table_view{{col0, col1, col2}};

  auto [_, got] = one_hot_encoding(input, category);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, got);
}

}  // namespace test
}  // namespace cudf
