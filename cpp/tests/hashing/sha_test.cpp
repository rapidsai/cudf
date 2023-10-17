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

class SHA224HashTest : public cudf::test::BaseFixture {};

TEST_F(SHA224HashTest, EmptyTable)
{
  auto const empty_table        = cudf::table_view{};
  auto const empty_column       = cudf::make_empty_column(cudf::data_type(cudf::type_id::STRING));
  auto const output_empty_table = cudf::hash(empty_table, cudf::hash_id::HASH_SHA224);
  EXPECT_EQ(empty_column->size(), output_empty_table->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(empty_column->view(), output_empty_table->view());

  auto const table_one_empty_column  = cudf::table_view{{empty_column->view()}};
  auto const output_one_empty_column = cudf::hash(empty_table, cudf::hash_id::HASH_SHA224);
  EXPECT_EQ(empty_column->size(), output_one_empty_column->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(empty_column->view(), output_one_empty_column->view());
}

TEST_F(SHA224HashTest, MultiValue)
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

  cudf::test::strings_column_wrapper const sha224_string_results1(
    {"d14a028c2a3a2bc9476102bb288234c415a2b01f828ea62ac5b3e42f",
     "dfd5f9139a820075df69d7895015360b76d0360f3d4b77a845689614",
     "5d1ed8373987e403482cefe1662a63fa3076c0a5331d141f41654bbe",
     "0662c91000b99de7a20c89097dd62f59120398d52499497489ccff95",
     "f9ea303770699483f3e53263b32a3b3c876d1b8808ce84df4b8ca1c4",
     "2da6cd4bdaa0a99fd7236cd5507c52e12328e71192e83b32d2f110f9",
     "e7d0adb165079efc6c6343112f8b154aa3644ca6326f658aaa0f8e4a",
     "309cc09eaa051beea7d0b0159daca9b4e8a533cb554e8f382c82709e"});

  cudf::test::strings_column_wrapper const sha224_string_results2(
    {"d14a028c2a3a2bc9476102bb288234c415a2b01f828ea62ac5b3e42f",
     "5538ae2b02d4ae0b7090dc908ca69cd11a2ffad43c7435f1dbad5e6a",
     "8e1955a473a149368dc0a931f99379b44b0bb752f206dbdf68629232",
     "8581001e08295b7884428c022378cfdd643c977aefe4512f0252dc30",
     "d5854dfe3c32996345b103a6a16c7bdfa924723d620b150737e77370",
     "dd56deac5f2caa579a440ee814fc04a3afaf805d567087ac3317beb3",
     "14fb559f6309604bedd89183f585f3b433932b5b0e675848feebf8ec",
     "d219eefea538491efcb69bc5bbef4177ad991d1b6e1367b5981b8c31"});

  using limits = std::numeric_limits<int32_t>;
  cudf::test::fixed_width_column_wrapper<int32_t> const ints_col(
    {0, 100, -100, limits::min(), limits::max(), 1, 2, 3});

  // Different truth values should be equal
  cudf::test::fixed_width_column_wrapper<bool> const bools_col1({0, 1, 1, 1, 0, 1, 1, 1});
  cudf::test::fixed_width_column_wrapper<bool> const bools_col2({0, 1, 2, 255, 0, 1, 2, 255});

  auto const string_input1         = cudf::table_view({strings_col});
  auto const string_input2         = cudf::table_view({strings_col, strings_col});
  auto const sha224_string_output1 = cudf::hash(string_input1, cudf::hash_id::HASH_SHA224);
  auto const sha224_string_output2 = cudf::hash(string_input2, cudf::hash_id::HASH_SHA224);
  EXPECT_EQ(string_input1.num_rows(), sha224_string_output1->size());
  EXPECT_EQ(string_input2.num_rows(), sha224_string_output2->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sha224_string_output1->view(), sha224_string_results1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sha224_string_output2->view(), sha224_string_results2);

  auto const input1         = cudf::table_view({strings_col, ints_col, bools_col1});
  auto const input2         = cudf::table_view({strings_col, ints_col, bools_col2});
  auto const sha224_output1 = cudf::hash(input1, cudf::hash_id::HASH_SHA224);
  auto const sha224_output2 = cudf::hash(input2, cudf::hash_id::HASH_SHA224);
  EXPECT_EQ(input1.num_rows(), sha224_output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sha224_output1->view(), sha224_output2->view());
}

TEST_F(SHA224HashTest, MultiValueNulls)
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

  auto const output1 = cudf::hash(input1, cudf::hash_id::HASH_SHA224);
  auto const output2 = cudf::hash(input2, cudf::hash_id::HASH_SHA224);

  EXPECT_EQ(input1.num_rows(), output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

template <typename T>
class SHA224HashTestTyped : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(SHA224HashTestTyped, cudf::test::NumericTypes);

TYPED_TEST(SHA224HashTestTyped, Equality)
{
  cudf::test::fixed_width_column_wrapper<TypeParam> const col({0, 127, 1, 2, 8});
  auto const input = cudf::table_view({col});

  // Hash of same input should be equal
  auto const output1 = cudf::hash(input, cudf::hash_id::HASH_SHA224);
  auto const output2 = cudf::hash(input, cudf::hash_id::HASH_SHA224);

  EXPECT_EQ(input.num_rows(), output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

TYPED_TEST(SHA224HashTestTyped, EqualityNulls)
{
  using T = TypeParam;

  // Nulls with different values should be equal
  cudf::test::fixed_width_column_wrapper<T> const col1({0, 127, 1, 2, 8}, {0, 1, 1, 1, 1});
  cudf::test::fixed_width_column_wrapper<T> const col2({1, 127, 1, 2, 8}, {0, 1, 1, 1, 1});

  auto const input1 = cudf::table_view({col1});
  auto const input2 = cudf::table_view({col2});

  auto const output1 = cudf::hash(input1, cudf::hash_id::HASH_SHA224);
  auto const output2 = cudf::hash(input2, cudf::hash_id::HASH_SHA224);

  EXPECT_EQ(input1.num_rows(), output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

template <typename T>
class SHA224HashTestFloatTyped : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(SHA224HashTestFloatTyped, cudf::test::FloatingPointTypes);

TYPED_TEST(SHA224HashTestFloatTyped, TestExtremes)
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

  auto const output1 = cudf::hash(input1, cudf::hash_id::HASH_SHA224);
  auto const output2 = cudf::hash(input2, cudf::hash_id::HASH_SHA224);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

class SHA256HashTest : public cudf::test::BaseFixture {};

TEST_F(SHA256HashTest, EmptyTable)
{
  auto const empty_table        = cudf::table_view{};
  auto const empty_column       = cudf::make_empty_column(cudf::data_type(cudf::type_id::STRING));
  auto const output_empty_table = cudf::hash(empty_table, cudf::hash_id::HASH_SHA256);
  EXPECT_EQ(empty_column->size(), output_empty_table->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(empty_column->view(), output_empty_table->view());

  auto const table_one_empty_column  = cudf::table_view{{empty_column->view()}};
  auto const output_one_empty_column = cudf::hash(empty_table, cudf::hash_id::HASH_SHA256);
  EXPECT_EQ(empty_column->size(), output_one_empty_column->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(empty_column->view(), output_one_empty_column->view());
}

TEST_F(SHA256HashTest, MultiValue)
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

  cudf::test::strings_column_wrapper const sha256_string_results1(
    {"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
     "5feceb66ffc86f38d952786c6d696c79c2dbc239dd4e91b46729d73a27fb57e9",
     "d16883c666112142c1d72c9080b41161be7563250539e3f6ab6e2fdf2210074b",
     "11174fa180460f5d683c2e63fcdd897dcbf10c28a9225d3ced9a8bbc3774415d",
     "10a7d211e692c6f71bb9f7524ba1437588c2797356f05fc585340f002fe7015e",
     "339d610dcb030bb4222bcf18c8ab82d911bfe7fb95b2cd9f6785fd4562b02401",
     "2ce9936a4a2234bf8a76c37d92e01d549d03949792242e7f8a1ad68575e4e4a8",
     "255fdd4d80a72f67921eb36f3e1157ea3e995068cee80e430c034e0d3692f614"});

  cudf::test::strings_column_wrapper const sha256_string_results2(
    {"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
     "f1534392279bddbf9d43dde8701cb5be14b82f76ec6607bf8d6ad557f60f304e",
     "96c204fa5d44b2487abfec105a05f8ae634551604f6596202ca99e3724e3953a",
     "2e7be264f3ecbb2930e7c54bf6c5fc1f310a8c63c50916bb713f34699ed11719",
     "224e4dce71d5dbd5e79ba65aaced7ad9c4f45dda146278087b2b61d164f056f0",
     "91f3108d4e9c696fdb37ae49fdc6a2237f1d1f977b7216406cc8a6365355f43b",
     "490be480afe271685e9c1fdf46daac0b9bf7f25602e153ca92a0ddb0e4b662ef",
     "4ddc45855d7ce3ab09efacff1fbafb33502f7dd468dc5a62826689c1c658dbce"});

  using limits = std::numeric_limits<int32_t>;
  cudf::test::fixed_width_column_wrapper<int32_t> const ints_col(
    {0, 100, -100, limits::min(), limits::max(), 1, 2, 3});

  // Different truth values should be equal
  cudf::test::fixed_width_column_wrapper<bool> const bools_col1({0, 1, 1, 1, 0, 1, 1, 1});
  cudf::test::fixed_width_column_wrapper<bool> const bools_col2({0, 1, 2, 255, 0, 1, 2, 255});

  auto const string_input1         = cudf::table_view({strings_col});
  auto const string_input2         = cudf::table_view({strings_col, strings_col});
  auto const sha256_string_output1 = cudf::hash(string_input1, cudf::hash_id::HASH_SHA256);
  auto const sha256_string_output2 = cudf::hash(string_input2, cudf::hash_id::HASH_SHA256);
  EXPECT_EQ(string_input1.num_rows(), sha256_string_output1->size());
  EXPECT_EQ(string_input2.num_rows(), sha256_string_output2->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sha256_string_output1->view(), sha256_string_results1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sha256_string_output2->view(), sha256_string_results2);

  auto const input1         = cudf::table_view({strings_col, ints_col, bools_col1});
  auto const input2         = cudf::table_view({strings_col, ints_col, bools_col2});
  auto const sha256_output1 = cudf::hash(input1, cudf::hash_id::HASH_SHA256);
  auto const sha256_output2 = cudf::hash(input2, cudf::hash_id::HASH_SHA256);
  EXPECT_EQ(input1.num_rows(), sha256_output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sha256_output1->view(), sha256_output2->view());
}

TEST_F(SHA256HashTest, MultiValueNulls)
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

  auto const output1 = cudf::hash(input1, cudf::hash_id::HASH_SHA256);
  auto const output2 = cudf::hash(input2, cudf::hash_id::HASH_SHA256);

  EXPECT_EQ(input1.num_rows(), output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

template <typename T>
class SHA256HashTestTyped : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(SHA256HashTestTyped, cudf::test::NumericTypes);

TYPED_TEST(SHA256HashTestTyped, Equality)
{
  cudf::test::fixed_width_column_wrapper<TypeParam> const col({0, 127, 1, 2, 8});
  auto const input = cudf::table_view({col});

  // Hash of same input should be equal
  auto const output1 = cudf::hash(input, cudf::hash_id::HASH_SHA256);
  auto const output2 = cudf::hash(input, cudf::hash_id::HASH_SHA256);

  EXPECT_EQ(input.num_rows(), output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

TYPED_TEST(SHA256HashTestTyped, EqualityNulls)
{
  using T = TypeParam;

  // Nulls with different values should be equal
  cudf::test::fixed_width_column_wrapper<T> const col1({0, 127, 1, 2, 8}, {0, 1, 1, 1, 1});
  cudf::test::fixed_width_column_wrapper<T> const col2({1, 127, 1, 2, 8}, {0, 1, 1, 1, 1});

  auto const input1 = cudf::table_view({col1});
  auto const input2 = cudf::table_view({col2});

  auto const output1 = cudf::hash(input1, cudf::hash_id::HASH_SHA256);
  auto const output2 = cudf::hash(input2, cudf::hash_id::HASH_SHA256);

  EXPECT_EQ(input1.num_rows(), output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

template <typename T>
class SHA256HashTestFloatTyped : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(SHA256HashTestFloatTyped, cudf::test::FloatingPointTypes);

TYPED_TEST(SHA256HashTestFloatTyped, TestExtremes)
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

  auto const output1 = cudf::hash(input1, cudf::hash_id::HASH_SHA256);
  auto const output2 = cudf::hash(input2, cudf::hash_id::HASH_SHA256);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

class SHA384HashTest : public cudf::test::BaseFixture {};

TEST_F(SHA384HashTest, EmptyTable)
{
  auto const empty_table        = cudf::table_view{};
  auto const empty_column       = cudf::make_empty_column(cudf::data_type(cudf::type_id::STRING));
  auto const output_empty_table = cudf::hash(empty_table, cudf::hash_id::HASH_SHA384);
  EXPECT_EQ(empty_column->size(), output_empty_table->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(empty_column->view(), output_empty_table->view());

  auto const table_one_empty_column  = cudf::table_view{{empty_column->view()}};
  auto const output_one_empty_column = cudf::hash(empty_table, cudf::hash_id::HASH_SHA384);
  EXPECT_EQ(empty_column->size(), output_one_empty_column->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(empty_column->view(), output_one_empty_column->view());
}

TEST_F(SHA384HashTest, MultiValue)
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

  cudf::test::strings_column_wrapper const sha384_string_results1(
    {"38b060a751ac96384cd9327eb1b1e36a21fdb71114be07434c0cc7bf63f6e1da274edebfe76f65fbd51ad2f14898b"
     "95b",
     "5f91550edb03f0bb8917da57f0f8818976f5da971307b7ee4886bb951c4891a1f16f840dae8f655aa5df718884ebc"
     "15b",
     "982000cce895dc439edbcb7ba5b908cb5b7e939fe913d58506a486735a914b0dfbcebb02c33c428287baa0bfc7fe0"
     "948",
     "c3ea54e4d6d97c2a84dac9ac48ed9dd1a49118be880d8466044720cfdcd23427bf556f12204bb34ede29dbf207033"
     "78c",
     "5d7a853a18138fa90feac07c896dfca65a0f1eb2ed40f1fd7be6238dd7ef429bb1aeb0236735500eb954c9b4ba923"
     "254",
     "c72bcaf3a4b01986711cd5d2614aa8f9d7fad61455613eac4561b1468f9a25dd26566c8ad1190dec7567be4f6fc1d"
     "b29",
     "281826f23bebb3f835d2f15edcb0cdb3078ae2d7dc516f3a366af172dff4db6dd5833bc1e5ee411d52c598773e939"
     "7b6",
     "3a9d1a870a5f6a4c04df1daf1808163d33852897ebc757a5b028a1214fbc758485a392159b11bc360cfadc79f9512"
     "822"});

  cudf::test::strings_column_wrapper const sha384_string_results2(
    {"38b060a751ac96384cd9327eb1b1e36a21fdb71114be07434c0cc7bf63f6e1da274edebfe76f65fbd51ad2f14898b"
     "95b",
     "34ae2cd40efabf896d8d4173e500278d10671b2d914efb5480e8349190bc7e8e1d532ad568d00a8295ea536a9b42b"
     "bc6",
     "e80c25efd8032ea94dad1509a68f9bf745ce1184b8a148714c28c7e0fae1100ab14057417394f83118eaa151e014d"
     "917",
     "69eaddc4ef2ed967fc6a86d3ed3777b2c2015df4cf8bbbf65681556f451a4a0ae805a89c2d56641b4422b5f248c56"
     "77d",
     "112a6f9c74741d490747db90f5e901a88b7a32f637c030d6d96e5f89a70a5f1ee209e018648842c0e1d32002f95fd"
     "d07",
     "dc6f24bb0eb2c96fb53c52c402f073de089f3aeae9594be0c4f4cb31b13bd48769b80aa97d83a25ece1edf0c83373"
     "f56",
     "781a33adfdcdcbb514318728c074fbb59d44002995825642e0c9bfef8a2ccf3fb637b39ff3dd265df8cd93c86e945"
     "ce9",
     "d2efb1591c4503f23c34ddb4da6bb1017d3d4d7c9f23ee6aa52e71c98d41060bc35eb22f41b6130d5c42a6e717fb3"
     "edf"});

  using limits = std::numeric_limits<int32_t>;
  cudf::test::fixed_width_column_wrapper<int32_t> const ints_col(
    {0, 100, -100, limits::min(), limits::max(), 1, 2, 3});

  // Different truth values should be equal
  cudf::test::fixed_width_column_wrapper<bool> const bools_col1({0, 1, 1, 1, 0, 1, 1, 1});
  cudf::test::fixed_width_column_wrapper<bool> const bools_col2({0, 1, 2, 255, 0, 1, 2, 255});

  auto const string_input1         = cudf::table_view({strings_col});
  auto const string_input2         = cudf::table_view({strings_col, strings_col});
  auto const sha384_string_output1 = cudf::hash(string_input1, cudf::hash_id::HASH_SHA384);
  auto const sha384_string_output2 = cudf::hash(string_input2, cudf::hash_id::HASH_SHA384);
  EXPECT_EQ(string_input1.num_rows(), sha384_string_output1->size());
  EXPECT_EQ(string_input2.num_rows(), sha384_string_output2->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sha384_string_output1->view(), sha384_string_results1, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sha384_string_output2->view(), sha384_string_results2, verbosity);

  auto const input1         = cudf::table_view({strings_col, ints_col, bools_col1});
  auto const input2         = cudf::table_view({strings_col, ints_col, bools_col2});
  auto const sha384_output1 = cudf::hash(input1, cudf::hash_id::HASH_SHA384);
  auto const sha384_output2 = cudf::hash(input2, cudf::hash_id::HASH_SHA384);
  EXPECT_EQ(input1.num_rows(), sha384_output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sha384_output1->view(), sha384_output2->view(), verbosity);
}

TEST_F(SHA384HashTest, MultiValueNulls)
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

  auto const output1 = cudf::hash(input1, cudf::hash_id::HASH_SHA384);
  auto const output2 = cudf::hash(input2, cudf::hash_id::HASH_SHA384);

  EXPECT_EQ(input1.num_rows(), output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

template <typename T>
class SHA384HashTestTyped : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(SHA384HashTestTyped, cudf::test::NumericTypes);

TYPED_TEST(SHA384HashTestTyped, Equality)
{
  cudf::test::fixed_width_column_wrapper<TypeParam> const col({0, 127, 1, 2, 8});
  auto const input = cudf::table_view({col});

  // Hash of same input should be equal
  auto const output1 = cudf::hash(input, cudf::hash_id::HASH_SHA384);
  auto const output2 = cudf::hash(input, cudf::hash_id::HASH_SHA384);

  EXPECT_EQ(input.num_rows(), output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

TYPED_TEST(SHA384HashTestTyped, EqualityNulls)
{
  using T = TypeParam;

  // Nulls with different values should be equal
  cudf::test::fixed_width_column_wrapper<T> const col1({0, 127, 1, 2, 8}, {0, 1, 1, 1, 1});
  cudf::test::fixed_width_column_wrapper<T> const col2({1, 127, 1, 2, 8}, {0, 1, 1, 1, 1});

  auto const input1 = cudf::table_view({col1});
  auto const input2 = cudf::table_view({col2});

  auto const output1 = cudf::hash(input1, cudf::hash_id::HASH_SHA384);
  auto const output2 = cudf::hash(input2, cudf::hash_id::HASH_SHA384);

  EXPECT_EQ(input1.num_rows(), output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

template <typename T>
class SHA384HashTestFloatTyped : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(SHA384HashTestFloatTyped, cudf::test::FloatingPointTypes);

TYPED_TEST(SHA384HashTestFloatTyped, TestExtremes)
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

  auto const output1 = cudf::hash(input1, cudf::hash_id::HASH_SHA384);
  auto const output2 = cudf::hash(input2, cudf::hash_id::HASH_SHA384);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

class SHA512HashTest : public cudf::test::BaseFixture {};

TEST_F(SHA512HashTest, EmptyTable)
{
  auto const empty_table        = cudf::table_view{};
  auto const empty_column       = cudf::make_empty_column(cudf::data_type(cudf::type_id::STRING));
  auto const output_empty_table = cudf::hash(empty_table, cudf::hash_id::HASH_SHA512);
  EXPECT_EQ(empty_column->size(), output_empty_table->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(empty_column->view(), output_empty_table->view());

  auto const table_one_empty_column  = cudf::table_view{{empty_column->view()}};
  auto const output_one_empty_column = cudf::hash(empty_table, cudf::hash_id::HASH_SHA512);
  EXPECT_EQ(empty_column->size(), output_one_empty_column->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(empty_column->view(), output_one_empty_column->view());
}

TEST_F(SHA512HashTest, MultiValue)
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

  cudf::test::strings_column_wrapper const sha512_string_results1(
    {"cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce47d0d13c5d85f2b0ff8318d2877ee"
     "c2f63b931bd47417a81a538327af927da3e",
     "31bca02094eb78126a517b206a88c73cfa9ec6f704c7030d18212cace820f025f00bf0ea68dbf3f3a5436ca63b53b"
     "f7bf80ad8d5de7d8359d0b7fed9dbc3ab99",
     "1d8b355dbe0c4ad81c9815a1490f0b6a6fa710e42ca60767ffd6d845acd116defe307c9496a80c4a67653873af6ed"
     "83e2e04c2102f55f9cd402677b246832e4c",
     "8ac8ae9de5597aa630f071f81fcb94dc93b6a8f92d8f2cdd5a469764a5daf6ef387b6465ae097dcd6e0c64286260d"
     "cc3d2c789d2cf5960df648c78a765e6c27c",
     "9c436e24be60e17425a1a829642d97e7180b57485cf95db007cf5b32bbae1f2325b6874b3377e37806b15b739bffa"
     "412ea6d095b726487d70e7b50e92d56c750",
     "6a25ca1f20f6e79faea2a0770075e4262beb66b40f59c22d3e8abdb6188ef8d8914faf5dbf6df76165bb61b81dfda"
     "46643f0d6366a39f7bd3d270312f9d3cf87",
     "bae9eb4b5c05a4c5f85750b70b2f0ce78e387f992f0927a017eb40bd180a13004f6252a6bbf9816f195fb7d86668c"
     "393dc0985aaf7168f48e8b905f3b9b02df2",
     "05a4ca1c523dcab32edb7d8793934a4cdf41a9062b229d711f5326e297bda83fa965118b9d7636172b43688e8e149"
     "008b3f967f1a969962b7e959af894a8a315"});

  cudf::test::strings_column_wrapper const sha512_string_results2(
    {"cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce47d0d13c5d85f2b0ff8318d2877ee"
     "c2f63b931bd47417a81a538327af927da3e",
     "8ab3361c051a97ddc3c665d29f2762f8ac4240d08995f8724b6d07d8cbedd32c28f589ccdae514f20a6c8eea6f755"
     "408dd3dd6837d66932ca2352eaeab594427",
     "338b22eb841420affff9904f903ed14c91bf8f4d1b10f25c145a31018367607a2cf562121ba7eaa2d08db3382cc82"
     "149805198c1fa3e7dc714fc2782e0f6ebd8",
     "d3045ecde16ea036d2f2ff3fa685beb46d5fcb73de71f0aee653265f18b22e4c131255e6eb5ad3be2f32914408ec6"
     "67911b49d951714decbdbfca1957be8ba10",
     "da7706221f8861ef522ab9555f57306382fb18c337536545d839e431dede4ff9f9affafb82ab5588734a8fc6631e6"
     "a0cd864634b62e24a42755c863c5d5c5848",
     "04dadc8fdf205fe535c8eb38f20882fc2a0e308081052d7588e74f6620aa207749039468c126db7407050def80415"
     "1d037cb188d5d4d459015032972a9e9f001",
     "aae2e742074847889a029a8d3170f9e17177d48ec0b9dabe572aa68dd3001af0c512f164ba84aa75b13950948170a"
     "0912912d16c98d2f05cb633c0d5b6a9105e",
     "77f46e99a7a51ac04b4380ebca70c0782381629f711169a3b9dad3fc9aa6221a9c0cdaa9b9ea4329773e773e2987c"
     "d1eebe0661386909684927d67819a2cf736"});

  using limits = std::numeric_limits<int32_t>;
  cudf::test::fixed_width_column_wrapper<int32_t> const ints_col(
    {0, 100, -100, limits::min(), limits::max(), 1, 2, 3});

  // Different truth values should be equal
  cudf::test::fixed_width_column_wrapper<bool> const bools_col1({0, 1, 1, 1, 0, 1, 1, 1});
  cudf::test::fixed_width_column_wrapper<bool> const bools_col2({0, 1, 2, 255, 0, 1, 2, 255});

  auto const string_input1         = cudf::table_view({strings_col});
  auto const string_input2         = cudf::table_view({strings_col, strings_col});
  auto const sha512_string_output1 = cudf::hash(string_input1, cudf::hash_id::HASH_SHA512);
  auto const sha512_string_output2 = cudf::hash(string_input2, cudf::hash_id::HASH_SHA512);
  EXPECT_EQ(string_input1.num_rows(), sha512_string_output1->size());
  EXPECT_EQ(string_input2.num_rows(), sha512_string_output2->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sha512_string_output1->view(), sha512_string_results1, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sha512_string_output2->view(), sha512_string_results2, verbosity);

  auto const input1         = cudf::table_view({strings_col, ints_col, bools_col1});
  auto const input2         = cudf::table_view({strings_col, ints_col, bools_col2});
  auto const sha512_output1 = cudf::hash(input1, cudf::hash_id::HASH_SHA512);
  auto const sha512_output2 = cudf::hash(input2, cudf::hash_id::HASH_SHA512);
  EXPECT_EQ(input1.num_rows(), sha512_output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sha512_output1->view(), sha512_output2->view(), verbosity);
}

TEST_F(SHA512HashTest, MultiValueNulls)
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

  auto const output1 = cudf::hash(input1, cudf::hash_id::HASH_SHA512);
  auto const output2 = cudf::hash(input2, cudf::hash_id::HASH_SHA512);

  EXPECT_EQ(input1.num_rows(), output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

template <typename T>
class SHA512HashTestTyped : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(SHA512HashTestTyped, cudf::test::NumericTypes);

TYPED_TEST(SHA512HashTestTyped, Equality)
{
  cudf::test::fixed_width_column_wrapper<TypeParam> const col({0, 127, 1, 2, 8});
  auto const input = cudf::table_view({col});

  // Hash of same input should be equal
  auto const output1 = cudf::hash(input, cudf::hash_id::HASH_SHA512);
  auto const output2 = cudf::hash(input, cudf::hash_id::HASH_SHA512);

  EXPECT_EQ(input.num_rows(), output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

TYPED_TEST(SHA512HashTestTyped, EqualityNulls)
{
  using T = TypeParam;

  // Nulls with different values should be equal
  cudf::test::fixed_width_column_wrapper<T> const col1({0, 127, 1, 2, 8}, {0, 1, 1, 1, 1});
  cudf::test::fixed_width_column_wrapper<T> const col2({1, 127, 1, 2, 8}, {0, 1, 1, 1, 1});

  auto const input1 = cudf::table_view({col1});
  auto const input2 = cudf::table_view({col2});

  auto const output1 = cudf::hash(input1, cudf::hash_id::HASH_SHA512);
  auto const output2 = cudf::hash(input2, cudf::hash_id::HASH_SHA512);

  EXPECT_EQ(input1.num_rows(), output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

template <typename T>
class SHA512HashTestFloatTyped : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(SHA512HashTestFloatTyped, cudf::test::FloatingPointTypes);

TYPED_TEST(SHA512HashTestFloatTyped, TestExtremes)
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

  auto const output1 = cudf::hash(input1, cudf::hash_id::HASH_SHA512);
  auto const output2 = cudf::hash(input2, cudf::hash_id::HASH_SHA512);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}
