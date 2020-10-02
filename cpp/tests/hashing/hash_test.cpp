/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

using cudf::test::fixed_width_column_wrapper;
using cudf::test::strings_column_wrapper;
using namespace cudf::test;

class HashTest : public cudf::test::BaseFixture {
};

TEST_F(HashTest, MultiValue)
{
  strings_column_wrapper const strings_col({"",
                                            "The quick brown fox",
                                            "jumps over the lazy dog.",
                                            "All work and no play makes Jack a dull boy",
                                            "!\"#$%&\'()*+,-./0123456789:;<=>?@[\\]^_`{|}~"});

  using limits = std::numeric_limits<int32_t>;
  fixed_width_column_wrapper<int32_t> const ints_col({0, 100, -100, limits::min(), limits::max()});

  // Different truth values should be equal
  fixed_width_column_wrapper<bool> const bools_col1({0, 1, 1, 1, 0});
  fixed_width_column_wrapper<bool> const bools_col2({0, 1, 2, 255, 0});

  using ts = cudf::timestamp_s;
  fixed_width_column_wrapper<ts, ts::duration> const secs_col({ts::duration::zero(),
                                                               static_cast<ts::duration>(100),
                                                               static_cast<ts::duration>(-100),
                                                               ts::duration::min(),
                                                               ts::duration::max()});

  auto const input1 = cudf::table_view({strings_col, ints_col, bools_col1, secs_col});
  auto const input2 = cudf::table_view({strings_col, ints_col, bools_col2, secs_col});

  auto const output1 = cudf::hash(input1);
  auto const output2 = cudf::hash(input2);

  EXPECT_EQ(input1.num_rows(), output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

TEST_F(HashTest, MultiValueNulls)
{
  // Nulls with different values should be equal
  strings_column_wrapper const strings_col1({"",
                                             "The quick brown fox",
                                             "jumps over the lazy dog.",
                                             "All work and no play makes Jack a dull boy",
                                             "!\"#$%&\'()*+,-./0123456789:;<=>?@[\\]^_`{|}~"},
                                            {0, 1, 1, 0, 1});
  strings_column_wrapper const strings_col2({"different but null",
                                             "The quick brown fox",
                                             "jumps over the lazy dog.",
                                             "I am Jack's complete lack of null value",
                                             "!\"#$%&\'()*+,-./0123456789:;<=>?@[\\]^_`{|}~"},
                                            {0, 1, 1, 0, 1});

  // Nulls with different values should be equal
  using limits = std::numeric_limits<int32_t>;
  fixed_width_column_wrapper<int32_t> const ints_col1({0, 100, -100, limits::min(), limits::max()},
                                                      {1, 0, 0, 1, 1});
  fixed_width_column_wrapper<int32_t> const ints_col2({0, -200, 200, limits::min(), limits::max()},
                                                      {1, 0, 0, 1, 1});

  // Nulls with different values should be equal
  // Different truth values should be equal
  fixed_width_column_wrapper<bool> const bools_col1({0, 1, 0, 1, 1}, {1, 1, 0, 0, 1});
  fixed_width_column_wrapper<bool> const bools_col2({0, 2, 1, 0, 255}, {1, 1, 0, 0, 1});

  // Nulls with different values should be equal
  using ts = cudf::timestamp_s;
  fixed_width_column_wrapper<ts, ts::duration> const secs_col1({ts::duration::zero(),
                                                                static_cast<ts::duration>(100),
                                                                static_cast<ts::duration>(-100),
                                                                ts::duration::min(),
                                                                ts::duration::max()},
                                                               {1, 0, 0, 1, 1});
  fixed_width_column_wrapper<ts, ts::duration> const secs_col2({ts::duration::zero(),
                                                                static_cast<ts::duration>(-200),
                                                                static_cast<ts::duration>(200),
                                                                ts::duration::min(),
                                                                ts::duration::max()},
                                                               {1, 0, 0, 1, 1});

  auto const input1 = cudf::table_view({strings_col1, ints_col1, bools_col1, secs_col1});
  auto const input2 = cudf::table_view({strings_col2, ints_col2, bools_col2, secs_col2});

  auto const output1 = cudf::hash(input1);
  auto const output2 = cudf::hash(input2);

  EXPECT_EQ(input1.num_rows(), output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

template <typename T>
class HashTestTyped : public cudf::test::BaseFixture {
};

TYPED_TEST_CASE(HashTestTyped, cudf::test::FixedWidthTypes);

TYPED_TEST(HashTestTyped, Equality)
{
  fixed_width_column_wrapper<TypeParam, int32_t> const col{0, 127, 1, 2, 8};
  auto const input = cudf::table_view({col});

  // Hash of same input should be equal
  auto const output1 = cudf::hash(input);
  auto const output2 = cudf::hash(input);

  EXPECT_EQ(input.num_rows(), output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

TYPED_TEST(HashTestTyped, EqualityNulls)
{
  using T = TypeParam;

  // Nulls with different values should be equal
  fixed_width_column_wrapper<T, int32_t> const col1({0, 127, 1, 2, 8}, {0, 1, 1, 1, 1});
  fixed_width_column_wrapper<T, int32_t> const col2({1, 127, 1, 2, 8}, {0, 1, 1, 1, 1});

  auto const input1 = cudf::table_view({col1});
  auto const input2 = cudf::table_view({col2});

  auto const output1 = cudf::hash(input1);
  auto const output2 = cudf::hash(input2);

  EXPECT_EQ(input1.num_rows(), output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

template <typename T>
class HashTestFloatTyped : public cudf::test::BaseFixture {
};

TYPED_TEST_CASE(HashTestFloatTyped, cudf::test::FloatingPointTypes);

TYPED_TEST(HashTestFloatTyped, TestExtremes)
{
  using T = TypeParam;
  T min   = std::numeric_limits<T>::min();
  T max   = std::numeric_limits<T>::max();
  T nan   = std::numeric_limits<T>::quiet_NaN();
  T inf   = std::numeric_limits<T>::infinity();

  fixed_width_column_wrapper<T> const col1({T(0.0), T(100.0), T(-100.0), min, max, nan, inf, -inf});
  fixed_width_column_wrapper<T> const col2(
    {T(-0.0), T(100.0), T(-100.0), min, max, -nan, inf, -inf});

  auto const input1 = cudf::table_view({col1});
  auto const input2 = cudf::table_view({col2});

  auto const output1 = cudf::hash(input1);
  auto const output2 = cudf::hash(input2);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view(), true);
}

class MD5HashTest : public cudf::test::BaseFixture {
};

TEST_F(MD5HashTest, MultiValue)
{
  strings_column_wrapper const strings_col(
    {"",
     "A 60 character string to test MD5's message padding algorithm",
     "A very long (greater than 128 bytes/char string) to test a multi hash-step data point in the "
     "MD5 hash function. This string needed to be longer.",
     "All work and no play makes Jack a dull boy",
     "!\"#$%&\'()*+,-./0123456789:;<=>?@[\\]^_`{|}~"});

  strings_column_wrapper const md5_string_results1({"d41d8cd98f00b204e9800998ecf8427e",
                                                    "682240021651ae166d08fe2a014d5c09",
                                                    "3669d5225fddbb34676312ca3b78bbd9",
                                                    "c61a4185135eda043f35e92c3505e180",
                                                    "52da74c75cb6575d25be29e66bd0adde"});

  strings_column_wrapper const md5_string_results2({"d41d8cd98f00b204e9800998ecf8427e",
                                                    "e5a5682e82278e78dbaad9a689df7a73",
                                                    "4121ab1bb6e84172fd94822645862ae9",
                                                    "28970886501efe20164213855afe5850",
                                                    "6bc1b872103cc6a02d882245b8516e2e"});

  using limits = std::numeric_limits<int32_t>;
  fixed_width_column_wrapper<int32_t> const ints_col({0, 100, -100, limits::min(), limits::max()});

  // Different truth values should be equal
  fixed_width_column_wrapper<bool> const bools_col1({0, 1, 1, 1, 0});
  fixed_width_column_wrapper<bool> const bools_col2({0, 1, 2, 255, 0});

  auto const string_input1      = cudf::table_view({strings_col});
  auto const string_input2      = cudf::table_view({strings_col, strings_col});
  auto const md5_string_output1 = cudf::hash(string_input1, cudf::hash_id::HASH_MD5);
  auto const md5_string_output2 = cudf::hash(string_input2, cudf::hash_id::HASH_MD5);
  EXPECT_EQ(string_input1.num_rows(), md5_string_output1->size());
  EXPECT_EQ(string_input2.num_rows(), md5_string_output2->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(md5_string_output1->view(), md5_string_results1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(md5_string_output2->view(), md5_string_results2);

  auto const input1      = cudf::table_view({strings_col, ints_col, bools_col1});
  auto const input2      = cudf::table_view({strings_col, ints_col, bools_col2});
  auto const md5_output1 = cudf::hash(input1, cudf::hash_id::HASH_MD5);
  auto const md5_output2 = cudf::hash(input2, cudf::hash_id::HASH_MD5);
  EXPECT_EQ(input1.num_rows(), md5_output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(md5_output1->view(), md5_output2->view());
}

TEST_F(MD5HashTest, MultiValueNulls)
{
  // Nulls with different values should be equal
  strings_column_wrapper const strings_col1(
    {"",
     "Different but null!",
     "A very long (greater than 128 bytes/char string) to test a multi hash-step data point in the "
     "MD5 hash function. This string needed to be longer.",
     "All work and no play makes Jack a dull boy",
     "!\"#$%&\'()*+,-./0123456789:;<=>?@[\\]^_`{|}~"},
    {1, 0, 0, 1, 0});
  strings_column_wrapper const strings_col2(
    {"",
     "A 60 character string to test MD5's message padding algorithm",
     "Very different... but null",
     "All work and no play makes Jack a dull boy",
     ""},
    {1, 0, 0, 1, 1});  // empty string is equivalent to null

  // Nulls with different values should be equal
  using limits = std::numeric_limits<int32_t>;
  fixed_width_column_wrapper<int32_t> const ints_col1({0, 100, -100, limits::min(), limits::max()},
                                                      {1, 0, 0, 1, 1});
  fixed_width_column_wrapper<int32_t> const ints_col2({0, -200, 200, limits::min(), limits::max()},
                                                      {1, 0, 0, 1, 1});

  // Nulls with different values should be equal
  // Different truth values should be equal
  fixed_width_column_wrapper<bool> const bools_col1({0, 1, 0, 1, 1}, {1, 1, 0, 0, 1});
  fixed_width_column_wrapper<bool> const bools_col2({0, 2, 1, 0, 255}, {1, 1, 0, 0, 1});

  auto const input1 = cudf::table_view({strings_col1, ints_col1, bools_col1});
  auto const input2 = cudf::table_view({strings_col2, ints_col2, bools_col2});

  auto const output1 = cudf::hash(input1, cudf::hash_id::HASH_MD5);
  auto const output2 = cudf::hash(input2, cudf::hash_id::HASH_MD5);

  EXPECT_EQ(input1.num_rows(), output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

TEST_F(MD5HashTest, StringListsNulls)
{
  auto validity = make_counting_transform_iterator(0, [](auto i) { return i != 0; });

  strings_column_wrapper const strings_col(
    {"",
     "A 60 character string to test MD5's message padding algorithm",
     "A very long (greater than 128 bytes/char string) to test a multi hash-step data point in the "
     "MD5 hash function. This string needed to be longer. It needed to be even longer.",
     "All work and no play makes Jack a dull boy",
     "!\"#$%&\'()*+,-./0123456789:;<=>?@[\\]^_`{|}~"});

  lists_column_wrapper<cudf::string_view> strings_list_col(
    {{""},
     {{"NULL", "A 60 character string to test MD5's message padding algorithm"}, validity},
     {"A very long (greater than 128 bytes/char string) to test a multi hash-step data point in "
      "the "
      "MD5 hash function. This string needed to be longer.",
      " It needed to be even longer."},
     {"All ", "work ", "and", " no", " play ", "makes Jack", " a dull boy"},
     {"!\"#$%&\'()*+,-./0123456789:;<=>?@[\\]^_`", "{|}~"}});

  auto const input1 = cudf::table_view({strings_col});
  auto const input2 = cudf::table_view({strings_list_col});

  auto const output1 = cudf::hash(input1, cudf::hash_id::HASH_MD5);
  auto const output2 = cudf::hash(input2, cudf::hash_id::HASH_MD5);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

template <typename T>
class MD5HashTestTyped : public cudf::test::BaseFixture {
};

TYPED_TEST_CASE(MD5HashTestTyped, cudf::test::NumericTypes);

TYPED_TEST(MD5HashTestTyped, Equality)
{
  fixed_width_column_wrapper<TypeParam> const col({0, 127, 1, 2, 8});
  auto const input = cudf::table_view({col});

  // Hash of same input should be equal
  auto const output1 = cudf::hash(input, cudf::hash_id::HASH_MD5);
  auto const output2 = cudf::hash(input, cudf::hash_id::HASH_MD5);

  EXPECT_EQ(input.num_rows(), output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

TYPED_TEST(MD5HashTestTyped, EqualityNulls)
{
  using T = TypeParam;

  // Nulls with different values should be equal
  fixed_width_column_wrapper<T> const col1({0, 127, 1, 2, 8}, {0, 1, 1, 1, 1});
  fixed_width_column_wrapper<T> const col2({1, 127, 1, 2, 8}, {0, 1, 1, 1, 1});

  auto const input1 = cudf::table_view({col1});
  auto const input2 = cudf::table_view({col2});

  auto const output1 = cudf::hash(input1, cudf::hash_id::HASH_MD5);
  auto const output2 = cudf::hash(input2, cudf::hash_id::HASH_MD5);

  EXPECT_EQ(input1.num_rows(), output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

TEST_F(MD5HashTest, TestBoolListsWithNulls)
{
  fixed_width_column_wrapper<bool> const col1({0, 255, 255, 16, 27, 18, 100, 1, 2},
                                              {1, 0, 0, 0, 1, 1, 1, 0, 0});
  fixed_width_column_wrapper<bool> const col2({0, 255, 255, 32, 81, 68, 3, 101, 4},
                                              {1, 0, 0, 1, 0, 1, 0, 1, 0});
  fixed_width_column_wrapper<bool> const col3({0, 255, 255, 64, 49, 42, 5, 6, 102},
                                              {1, 0, 0, 1, 1, 0, 0, 0, 1});

  auto validity = make_counting_transform_iterator(0, [](auto i) { return i != 1; });
  lists_column_wrapper<bool> const list_col(
    {{0, 0, 0}, {1}, {}, {{1, 1, 1}, validity}, {1, 1}, {1, 1}, {1}, {1}, {1}}, validity);

  auto const input1 = cudf::table_view({col1, col2, col3});
  auto const input2 = cudf::table_view({list_col});

  auto const output1 = cudf::hash(input1, cudf::hash_id::HASH_MD5);
  auto const output2 = cudf::hash(input2, cudf::hash_id::HASH_MD5);

  EXPECT_EQ(input1.num_rows(), output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

template <typename T>
class MD5HashListTestTyped : public cudf::test::BaseFixture {
};

using NumericTypesNoBools = Concat<IntegralTypesNotBool, FloatingPointTypes>;
TYPED_TEST_CASE(MD5HashListTestTyped, NumericTypesNoBools);

TYPED_TEST(MD5HashListTestTyped, TestListsWithNulls)
{
  using T = TypeParam;

  fixed_width_column_wrapper<T> const col1({0, 255, 255, 16, 27, 18, 100, 1, 2},
                                           {1, 0, 0, 0, 1, 1, 1, 0, 0});
  fixed_width_column_wrapper<T> const col2({0, 255, 255, 32, 81, 68, 3, 101, 4},
                                           {1, 0, 0, 1, 0, 1, 0, 1, 0});
  fixed_width_column_wrapper<T> const col3({0, 255, 255, 64, 49, 42, 5, 6, 102},
                                           {1, 0, 0, 1, 1, 0, 0, 0, 1});

  auto validity = make_counting_transform_iterator(0, [](auto i) { return i != 1; });
  lists_column_wrapper<T> const list_col(
    {{0, 0, 0}, {127}, {}, {{32, 127, 64}, validity}, {27, 49}, {18, 68}, {100}, {101}, {102}},
    validity);

  auto const input1 = cudf::table_view({col1, col2, col3});
  auto const input2 = cudf::table_view({list_col});

  auto const output1 = cudf::hash(input1, cudf::hash_id::HASH_MD5);
  auto const output2 = cudf::hash(input2, cudf::hash_id::HASH_MD5);

  EXPECT_EQ(input1.num_rows(), output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

template <typename T>
class MD5HashTestFloatTyped : public cudf::test::BaseFixture {
};

TYPED_TEST_CASE(MD5HashTestFloatTyped, cudf::test::FloatingPointTypes);

TYPED_TEST(MD5HashTestFloatTyped, TestExtremes)
{
  using T = TypeParam;
  T min   = std::numeric_limits<T>::min();
  T max   = std::numeric_limits<T>::max();
  T nan   = std::numeric_limits<T>::quiet_NaN();
  T inf   = std::numeric_limits<T>::infinity();

  fixed_width_column_wrapper<T> const col1({T(0.0), T(100.0), T(-100.0), min, max, nan, inf, -inf});
  fixed_width_column_wrapper<T> const col2(
    {T(-0.0), T(100.0), T(-100.0), min, max, -nan, inf, -inf});

  auto const input1 = cudf::table_view({col1});
  auto const input2 = cudf::table_view({col2});

  auto const output1 = cudf::hash(input1, cudf::hash_id::HASH_MD5);
  auto const output2 = cudf::hash(input2, cudf::hash_id::HASH_MD5);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view(), true);
}

TYPED_TEST(MD5HashTestFloatTyped, TestListExtremes)
{
  using T = TypeParam;
  T min   = std::numeric_limits<T>::min();
  T max   = std::numeric_limits<T>::max();
  T nan   = std::numeric_limits<T>::quiet_NaN();
  T inf   = std::numeric_limits<T>::infinity();

  lists_column_wrapper<T> const col1(
    {{T(0.0)}, {T(100.0), T(-100.0)}, {min, max, nan}, {inf, -inf}});
  lists_column_wrapper<T> const col2(
    {{T(-0.0)}, {T(100.0), T(-100.0)}, {min, max, -nan}, {inf, -inf}});

  auto const input1 = cudf::table_view({col1});
  auto const input2 = cudf::table_view({col2});

  auto const output1 = cudf::hash(input1, cudf::hash_id::HASH_MD5);
  auto const output2 = cudf::hash(input2, cudf::hash_id::HASH_MD5);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view(), true);
}

CUDF_TEST_PROGRAM_MAIN()
