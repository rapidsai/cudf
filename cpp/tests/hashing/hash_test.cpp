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
#include <cudf/hashing.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/type_lists.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>

using cudf::test::fixed_width_column_wrapper;
using cudf::test::strings_column_wrapper;
using cudf::test::expect_columns_equal;
using cudf::test::expect_column_properties_equal;
using cudf::experimental::bool8;

class HashTest : public cudf::test::BaseFixture {};

TEST_F(HashTest, MultiValue)
{
  strings_column_wrapper const strings_col(
    {"",
    "The quick brown fox",
    "jumps over the lazy dog.",
    "All work and no play makes Jack a dull boy",
    "!\"#$%&\'()*+,-./0123456789:;<=>?@[\\]^_`{|}~"});

  using limits = std::numeric_limits<int32_t>;
  fixed_width_column_wrapper<int32_t> const ints_col(
    {0, 100, -100, limits::min(), limits::max()});

  // Different truthy values should be equal
  fixed_width_column_wrapper<bool8> const bools_col1({0, 1, 1, 1, 0});
  fixed_width_column_wrapper<bool8> const bools_col2({0, 1, 2, 255, 0});

  using ts = cudf::timestamp_s;
  fixed_width_column_wrapper<ts> const secs_col(
    {ts::duration::zero(), 100, -100, ts::duration::min(), ts::duration::max()});

  auto const input1 = cudf::table_view(
    {strings_col, ints_col, bools_col1, secs_col});
  auto const input2 = cudf::table_view(
    {strings_col, ints_col, bools_col2, secs_col});

  auto const output1 = cudf::hash(input1);
  auto const output2 = cudf::hash(input2);

  EXPECT_EQ(input1.num_rows(), output1->size());
  expect_columns_equal(output1->view(), output2->view());
}

TEST_F(HashTest, MultiValueNulls)
{
  // Nulls with different values should be equal
  strings_column_wrapper const strings_col1(
    {"",
    "The quick brown fox",
    "jumps over the lazy dog.",
    "All work and no play makes Jack a dull boy",
    "!\"#$%&\'()*+,-./0123456789:;<=>?@[\\]^_`{|}~"},
    {0, 1, 1, 0, 1});
  strings_column_wrapper const strings_col2(
    {"different but null",
    "The quick brown fox",
    "jumps over the lazy dog.",
    "I am Jack's complete lack of null value",
    "!\"#$%&\'()*+,-./0123456789:;<=>?@[\\]^_`{|}~"},
    {0, 1, 1, 0, 1});

  // Nulls with different values should be equal
  using limits = std::numeric_limits<int32_t>;
  fixed_width_column_wrapper<int32_t> const ints_col1(
    {0, 100, -100, limits::min(), limits::max()}, {1, 0, 0, 1, 1});
  fixed_width_column_wrapper<int32_t> const ints_col2(
    {0, -200, 200, limits::min(), limits::max()}, {1, 0, 0, 1, 1});

  // Nulls with different values should be equal
  // Different truthy values should be equal
  fixed_width_column_wrapper<bool8> const bools_col1(
    {0, 1, 0, 1, 1}, {1, 1, 0, 0, 1});
  fixed_width_column_wrapper<bool8> const bools_col2(
    {0, 2, 1, 0, 255}, {1, 1, 0, 0, 1});

  // Nulls with different values should be equal
  using ts = cudf::timestamp_s;
  fixed_width_column_wrapper<ts> const secs_col1(
    {ts::duration::zero(), 100, -100, ts::duration::min(), ts::duration::max()},
    {1, 0, 0, 1, 1});
  fixed_width_column_wrapper<ts> const secs_col2(
    {ts::duration::zero(), -200, 200, ts::duration::min(), ts::duration::max()},
    {1, 0, 0, 1, 1});

  auto const input1 = cudf::table_view(
    {strings_col1, ints_col1, bools_col1, secs_col1});
  auto const input2 = cudf::table_view(
    {strings_col2, ints_col2, bools_col2, secs_col2});

  auto const output1 = cudf::hash(input1);
  auto const output2 = cudf::hash(input2);

  EXPECT_EQ(input1.num_rows(), output1->size());
  expect_columns_equal(output1->view(), output2->view());
}

template <typename T>
class HashTestTyped : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(HashTestTyped, cudf::test::FixedWidthTypes);

TYPED_TEST(HashTestTyped, Equality)
{
  fixed_width_column_wrapper<TypeParam> const col(
    {0, 127, 1, 2, 8});
  auto const input = cudf::table_view({col});

  // Hash of same input should be equal
  auto const output1 = cudf::hash(input);
  auto const output2 = cudf::hash(input);

  EXPECT_EQ(input.num_rows(), output1->size());
  expect_columns_equal(output1->view(), output2->view());
}

TYPED_TEST(HashTestTyped, EqualityNulls)
{
  // Nulls with different values should be equal
  fixed_width_column_wrapper<TypeParam> const col1(
    {0, 127, 1, 2, 8}, {0, 1, 1, 1, 1});
  fixed_width_column_wrapper<TypeParam> const col2(
    {1, 127, 1, 2, 8}, {0, 1, 1, 1, 1});

  auto const input1 = cudf::table_view({col1});
  auto const input2 = cudf::table_view({col2});

  auto const output1 = cudf::hash(input1);
  auto const output2 = cudf::hash(input2);

  EXPECT_EQ(input1.num_rows(), output1->size());
  expect_columns_equal(output1->view(), output2->view());
}

template <typename T>
class HashTestFloatTyped : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(HashTestFloatTyped, cudf::test::FloatingPointTypes);

TYPED_TEST(HashTestFloatTyped, TestExtremes)
{
  TypeParam min = std::numeric_limits<TypeParam>::min();
  TypeParam max = std::numeric_limits<TypeParam>::max();
  TypeParam nan = std::numeric_limits<TypeParam>::quiet_NaN();
  TypeParam inf = std::numeric_limits<TypeParam>::infinity();
  
  fixed_width_column_wrapper<TypeParam> const col1( {  0.0, 100.0, -100.0, min, max,  nan, inf, -inf } );
  fixed_width_column_wrapper<TypeParam> const col2( { -0.0, 100.0, -100.0, min, max, -nan, inf, -inf } );

  auto const input1 = cudf::table_view({col1});
  auto const input2 = cudf::table_view({col2});

  auto const output1 = cudf::hash(input1);
  auto const output2 = cudf::hash(input2);

  expect_columns_equal(output1->view(), output2->view(), true);
}

