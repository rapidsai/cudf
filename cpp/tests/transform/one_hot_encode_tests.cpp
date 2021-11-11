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

#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>

#include <limits>

namespace cudf {
namespace test {

template <typename T>
struct OneHotEncodingTestTyped : public BaseFixture {
};

struct OneHotEncodingTest : public BaseFixture {
};

TYPED_TEST_SUITE(OneHotEncodingTestTyped, cudf::test::NumericTypes);

TYPED_TEST(OneHotEncodingTestTyped, Basic)
{
  auto input    = fixed_width_column_wrapper<int32_t>{8, 8, 8, 9, 9};
  auto category = fixed_width_column_wrapper<int32_t>{8, 9};

  auto col0 = fixed_width_column_wrapper<bool>{1, 1, 1, 0, 0};
  auto col1 = fixed_width_column_wrapper<bool>{0, 0, 0, 1, 1};

  auto expected = table_view{{col0, col1}};

  [[maybe_unused]] auto [res_ptr, got] = one_hot_encode(input, category);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, got);
}

TYPED_TEST(OneHotEncodingTestTyped, Nulls)
{
  auto input    = fixed_width_column_wrapper<int32_t>{{8, 8, 8, 9, 9}, {1, 1, 0, 1, 1}};
  auto category = fixed_width_column_wrapper<int32_t>({8, 9, -1}, {1, 1, 0});

  auto col0 = fixed_width_column_wrapper<bool>{1, 1, 0, 0, 0};
  auto col1 = fixed_width_column_wrapper<bool>{0, 0, 0, 1, 1};
  auto col2 = fixed_width_column_wrapper<bool>{0, 0, 1, 0, 0};

  auto expected = table_view{{col0, col1, col2}};

  [[maybe_unused]] auto [res_ptr, got] = one_hot_encode(input, category);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, got);
}

TEST_F(OneHotEncodingTest, Diagonal)
{
  auto input    = fixed_width_column_wrapper<int32_t>{1, 2, 3, 4, 5};
  auto category = fixed_width_column_wrapper<int32_t>{1, 2, 3, 4, 5};

  auto col0 = fixed_width_column_wrapper<bool>{1, 0, 0, 0, 0};
  auto col1 = fixed_width_column_wrapper<bool>{0, 1, 0, 0, 0};
  auto col2 = fixed_width_column_wrapper<bool>{0, 0, 1, 0, 0};
  auto col3 = fixed_width_column_wrapper<bool>{0, 0, 0, 1, 0};
  auto col4 = fixed_width_column_wrapper<bool>{0, 0, 0, 0, 1};

  auto expected = table_view{{col0, col1, col2, col3, col4}};

  [[maybe_unused]] auto [res_ptr, got] = one_hot_encode(input, category);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, got);
}

TEST_F(OneHotEncodingTest, ZeroInput)
{
  auto input    = strings_column_wrapper{};
  auto category = strings_column_wrapper{"rapids", "cudf"};

  auto col0 = fixed_width_column_wrapper<bool>{};
  auto col1 = fixed_width_column_wrapper<bool>{};

  auto expected = table_view{{col0, col1}};

  [[maybe_unused]] auto [res_ptr, got] = one_hot_encode(input, category);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, got);
}

TEST_F(OneHotEncodingTest, ZeroCat)
{
  auto input    = strings_column_wrapper{"ji", "ji", "ji"};
  auto category = strings_column_wrapper{};

  auto expected = table_view{};

  [[maybe_unused]] auto [res_ptr, got] = one_hot_encode(input, category);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, got);
}

TEST_F(OneHotEncodingTest, ZeroInputCat)
{
  auto input    = strings_column_wrapper{};
  auto category = strings_column_wrapper{};

  auto expected = table_view{};

  [[maybe_unused]] auto [res_ptr, got] = one_hot_encode(input, category);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, got);
}

TEST_F(OneHotEncodingTest, OneCat)
{
  auto input    = strings_column_wrapper{"ji", "ji", "ji"};
  auto category = strings_column_wrapper{"ji"};

  auto col0 = fixed_width_column_wrapper<bool>{1, 1, 1};

  auto expected = table_view{{col0}};

  [[maybe_unused]] auto [res_ptr, got] = one_hot_encode(input, category);

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

  [[maybe_unused]] auto [res_ptr, got] = one_hot_encode(input, category);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, got);
}

TEST_F(OneHotEncodingTest, Strings)
{
  auto input = strings_column_wrapper{
    {"hello", "rapidsai", "cudf", "hello", "cuspatial", "hello", "world", "!"},
    {1, 1, 1, 1, 0, 1, 1, 0}};
  auto category = strings_column_wrapper{{"hello", "world", ""}, {1, 1, 0}};

  auto col0 = fixed_width_column_wrapper<bool>{1, 0, 0, 1, 0, 1, 0, 0};
  auto col1 = fixed_width_column_wrapper<bool>{0, 0, 0, 0, 0, 0, 1, 0};
  auto col2 = fixed_width_column_wrapper<bool>{0, 0, 0, 0, 1, 0, 0, 1};

  auto expected = table_view{{col0, col1, col2}};

  [[maybe_unused]] auto [res_ptr, got] = one_hot_encode(input, category);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, got);
}

TEST_F(OneHotEncodingTest, Dictionary)
{
  auto input    = dictionary_column_wrapper<std::string>{"aa", "xx", "aa", "aa", "yy", "ef"};
  auto category = dictionary_column_wrapper<std::string>{"aa", "ef"};

  auto col0 = fixed_width_column_wrapper<bool>{1, 0, 1, 1, 0, 0};
  auto col1 = fixed_width_column_wrapper<bool>{0, 0, 0, 0, 0, 1};

  auto expected = table_view{{col0, col1}};

  [[maybe_unused]] auto [res_ptr, got] = one_hot_encode(input, category);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, got);
}

TEST_F(OneHotEncodingTest, MismatchTypes)
{
  auto input    = strings_column_wrapper{"xx", "yy", "xx"};
  auto category = fixed_width_column_wrapper<int64_t>{1};

  EXPECT_THROW(one_hot_encode(input, category), cudf::logic_error);
}

}  // namespace test
}  // namespace cudf
