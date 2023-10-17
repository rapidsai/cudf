/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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
#include <cudf/hashing.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_lists.hpp>

constexpr cudf::test::debug_output_level verbosity{cudf::test::debug_output_level::ALL_ERRORS};

class SHA1HashTest : public cudf::test::BaseFixture {};

TEST_F(SHA1HashTest, EmptyTable)
{
  auto const empty_table        = cudf::table_view{};
  auto const empty_column       = cudf::make_empty_column(cudf::data_type(cudf::type_id::STRING));
  auto const output_empty_table = cudf::hash(empty_table, cudf::hash_id::HASH_SHA1);
  EXPECT_EQ(empty_column->size(), output_empty_table->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(empty_column->view(), output_empty_table->view());

  auto const table_one_empty_column  = cudf::table_view{{empty_column->view()}};
  auto const output_one_empty_column = cudf::hash(empty_table, cudf::hash_id::HASH_SHA1);
  EXPECT_EQ(empty_column->size(), output_one_empty_column->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(empty_column->view(), output_one_empty_column->view());
}

TEST_F(SHA1HashTest, MultiValue)
{
  cudf::test::strings_column_wrapper const strings_col(
    {"",
     "0",
     "A 56 character string to test message padding algorithm.",
     "A 63 character string to test message padding algorithm, again.",
     "A 64 character string to test message padding algorithm, again!!",
     "A very long (greater than 128 bytes/char string) to execute a multi hash-step data point in "
     "the hash function being tested. This string needed to be longer.",
     "All work and no play makes Jack a dull boy",
     "!\"#$%&\'()*+,-./0123456789:;<=>?@[\\]^_`{|}~"});

  cudf::test::strings_column_wrapper const sha1_string_results1(
    {"da39a3ee5e6b4b0d3255bfef95601890afd80709",
     "b6589fc6ab0dc82cf12099d1c2d40ab994e8410c",
     "cb73203438ab46ea54491c53e288a2703c440c4a",
     "c595ebd13a785c1c2659e010a42e2ff9987ef51f",
     "4ffaf61804c55b8c2171be548bef2e1d0baca17a",
     "595965dd18f38087186162c788485fe249242131",
     "a62ca720fbab830c8890044eacbeac216f1ca2e4",
     "11e16c52273b5669a41d17ec7c187475193f88b3"});

  cudf::test::strings_column_wrapper const sha1_string_results2(
    {"da39a3ee5e6b4b0d3255bfef95601890afd80709",
     "fb96549631c835eb239cd614cc6b5cb7d295121a",
     "e3977ee0ea7f238134ec93c79988fa84b7c5d79e",
     "f6f75b6fa3c3d8d86b44fcb2c98c9ad4b37dcdd0",
     "c7abd431a775c604edf41a62f7f215e7258dc16a",
     "153fdf20d2bd8ae76241197314d6e0be7fe10f50",
     "8c3656f7cb37898f9296c1965000d6da13fed64e",
     "b4a848399375ec842c2cb445d98b5f80a4dce94f"});

  using limits = std::numeric_limits<int32_t>;
  cudf::test::fixed_width_column_wrapper<int32_t> const ints_col(
    {0, 100, -100, limits::min(), limits::max(), 1, 2, 3});

  // Different truth values should be equal
  cudf::test::fixed_width_column_wrapper<bool> const bools_col1({0, 1, 1, 1, 0, 1, 1, 1});
  cudf::test::fixed_width_column_wrapper<bool> const bools_col2({0, 1, 2, 255, 0, 1, 2, 255});

  auto const string_input1       = cudf::table_view({strings_col});
  auto const string_input2       = cudf::table_view({strings_col, strings_col});
  auto const sha1_string_output1 = cudf::hash(string_input1, cudf::hash_id::HASH_SHA1);
  auto const sha1_string_output2 = cudf::hash(string_input2, cudf::hash_id::HASH_SHA1);
  EXPECT_EQ(string_input1.num_rows(), sha1_string_output1->size());
  EXPECT_EQ(string_input2.num_rows(), sha1_string_output2->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sha1_string_output1->view(), sha1_string_results1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sha1_string_output2->view(), sha1_string_results2);

  auto const input1       = cudf::table_view({strings_col, ints_col, bools_col1});
  auto const input2       = cudf::table_view({strings_col, ints_col, bools_col2});
  auto const sha1_output1 = cudf::hash(input1, cudf::hash_id::HASH_SHA1);
  auto const sha1_output2 = cudf::hash(input2, cudf::hash_id::HASH_SHA1);
  EXPECT_EQ(input1.num_rows(), sha1_output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sha1_output1->view(), sha1_output2->view());
}

TEST_F(SHA1HashTest, MultiValueNulls)
{
  // Nulls with different values should be equal
  cudf::test::strings_column_wrapper const strings_col1(
    {"",
     "Different but null!",
     "A very long (greater than 128 bytes/char string) to execute a multi hash-step data point in "
     "the hash function being tested. This string needed to be longer.",
     "All work and no play makes Jack a dull boy",
     "!\"#$%&\'()*+,-./0123456789:;<=>?@[\\]^_`{|}~"},
    {1, 0, 0, 1, 0});
  cudf::test::strings_column_wrapper const strings_col2(
    {"",
     "Another string that is null.",
     "Very different... but null",
     "All work and no play makes Jack a dull boy",
     ""},
    {1, 0, 0, 1, 0});

  // Nulls with different values should be equal
  using limits = std::numeric_limits<int32_t>;
  cudf::test::fixed_width_column_wrapper<int32_t> const ints_col1(
    {0, 100, -100, limits::min(), limits::max()}, {1, 0, 0, 1, 0});
  cudf::test::fixed_width_column_wrapper<int32_t> const ints_col2(
    {0, -200, 200, limits::min(), limits::max()}, {1, 0, 0, 0, 1});

  // Nulls with different values should be equal
  // Different truthy values should be equal
  cudf::test::fixed_width_column_wrapper<bool> const bools_col1({0, 1, 0, 1, 1}, {1, 1, 0, 0, 1});
  cudf::test::fixed_width_column_wrapper<bool> const bools_col2({0, 2, 1, 0, 255}, {1, 1, 0, 1, 0});

  auto const input1 = cudf::table_view({strings_col1, ints_col1, bools_col1});
  auto const input2 = cudf::table_view({strings_col2, ints_col2, bools_col2});

  auto const output1 = cudf::hash(input1, cudf::hash_id::HASH_SHA1);
  auto const output2 = cudf::hash(input2, cudf::hash_id::HASH_SHA1);

  EXPECT_EQ(input1.num_rows(), output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

template <typename T>
class SHA1HashTestTyped : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(SHA1HashTestTyped, cudf::test::NumericTypes);

TYPED_TEST(SHA1HashTestTyped, Equality)
{
  cudf::test::fixed_width_column_wrapper<TypeParam> const col({0, 127, 1, 2, 8});
  auto const input = cudf::table_view({col});

  // Hash of same input should be equal
  auto const output1 = cudf::hash(input, cudf::hash_id::HASH_SHA1);
  auto const output2 = cudf::hash(input, cudf::hash_id::HASH_SHA1);

  EXPECT_EQ(input.num_rows(), output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

TYPED_TEST(SHA1HashTestTyped, EqualityNulls)
{
  using T = TypeParam;

  // Nulls with different values should be equal
  cudf::test::fixed_width_column_wrapper<T> const col1({0, 127, 1, 2, 8}, {0, 1, 1, 1, 1});
  cudf::test::fixed_width_column_wrapper<T> const col2({1, 127, 1, 2, 8}, {0, 1, 1, 1, 1});

  auto const input1 = cudf::table_view({col1});
  auto const input2 = cudf::table_view({col2});

  auto const output1 = cudf::hash(input1, cudf::hash_id::HASH_SHA1);
  auto const output2 = cudf::hash(input2, cudf::hash_id::HASH_SHA1);

  EXPECT_EQ(input1.num_rows(), output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

template <typename T>
class SHA1HashTestFloatTyped : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(SHA1HashTestFloatTyped, cudf::test::FloatingPointTypes);

TYPED_TEST(SHA1HashTestFloatTyped, TestExtremes)
{
  using T = TypeParam;
  T min   = std::numeric_limits<T>::min();
  T max   = std::numeric_limits<T>::max();
  T nan   = std::numeric_limits<T>::quiet_NaN();
  T inf   = std::numeric_limits<T>::infinity();

  cudf::test::fixed_width_column_wrapper<T> const col1(
    {T(0.0), T(100.0), T(-100.0), min, max, nan, inf, -inf});
  cudf::test::fixed_width_column_wrapper<T> const col2(
    {T(-0.0), T(100.0), T(-100.0), min, max, -nan, inf, -inf});

  auto const input1 = cudf::table_view({col1});
  auto const input2 = cudf::table_view({col2});

  auto const output1 = cudf::hash(input1, cudf::hash_id::HASH_SHA1);
  auto const output2 = cudf::hash(input2, cudf::hash_id::HASH_SHA1);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}
