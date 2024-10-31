/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/hashing.hpp>
#include <cudf/utilities/error.hpp>

class SHA256HashTest : public cudf::test::BaseFixture {};

TEST_F(SHA256HashTest, EmptyTable)
{
  auto const empty_table        = cudf::table_view{};
  auto const empty_column       = cudf::make_empty_column(cudf::data_type(cudf::type_id::STRING));
  auto const output_empty_table = cudf::hashing::sha256(empty_table);
  EXPECT_EQ(empty_column->size(), output_empty_table->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(empty_column->view(), output_empty_table->view());

  auto const table_one_empty_column  = cudf::table_view{{empty_column->view()}};
  auto const output_one_empty_column = cudf::hashing::sha256(empty_table);
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
     R"(!"#$%&'()*+,-./0123456789:;<=>?@[\]^_`{|}~)",
     "Multi-byte characters: é¼³⅝"});

  /*
  These outputs can be generated with shell:
  ```
  echo -n "input string" | sha256sum
  ```
  Or with Python:
  ```
  import hashlib
  print(hashlib.sha256("input string".encode()).hexdigest())
  ```
  */
  cudf::test::strings_column_wrapper const sha256_string_results1(
    {"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
     "5feceb66ffc86f38d952786c6d696c79c2dbc239dd4e91b46729d73a27fb57e9",
     "d16883c666112142c1d72c9080b41161be7563250539e3f6ab6e2fdf2210074b",
     "11174fa180460f5d683c2e63fcdd897dcbf10c28a9225d3ced9a8bbc3774415d",
     "10a7d211e692c6f71bb9f7524ba1437588c2797356f05fc585340f002fe7015e",
     "339d610dcb030bb4222bcf18c8ab82d911bfe7fb95b2cd9f6785fd4562b02401",
     "2ce9936a4a2234bf8a76c37d92e01d549d03949792242e7f8a1ad68575e4e4a8",
     "255fdd4d80a72f67921eb36f3e1157ea3e995068cee80e430c034e0d3692f614",
     "9f9a89d448937f853c0067a3e2cb732d703eca971e3fb0f88fc73a730b7a85f4"});

  cudf::test::strings_column_wrapper const sha256_string_results2(
    {"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
     "f1534392279bddbf9d43dde8701cb5be14b82f76ec6607bf8d6ad557f60f304e",
     "96c204fa5d44b2487abfec105a05f8ae634551604f6596202ca99e3724e3953a",
     "2e7be264f3ecbb2930e7c54bf6c5fc1f310a8c63c50916bb713f34699ed11719",
     "224e4dce71d5dbd5e79ba65aaced7ad9c4f45dda146278087b2b61d164f056f0",
     "91f3108d4e9c696fdb37ae49fdc6a2237f1d1f977b7216406cc8a6365355f43b",
     "490be480afe271685e9c1fdf46daac0b9bf7f25602e153ca92a0ddb0e4b662ef",
     "4ddc45855d7ce3ab09efacff1fbafb33502f7dd468dc5a62826689c1c658dbce",
     "bed32be19e1f432f5caec2b8bf914a968dfa5a5cba3868ea640ba9cbb0f9c9c8"});

  using limits = std::numeric_limits<int32_t>;
  cudf::test::fixed_width_column_wrapper<int32_t> const ints_col(
    {0, -1, 100, -100, limits::min(), limits::max(), 1, 2, 3});

  cudf::test::fixed_width_column_wrapper<bool> const bools_col({0, 1, 1, 1, 0, 1, 1, 1, 0});

  // Test string inputs against known outputs
  auto const string_input1         = cudf::table_view({strings_col});
  auto const string_input2         = cudf::table_view({strings_col, strings_col});
  auto const sha256_string_output1 = cudf::hashing::sha256(string_input1);
  auto const sha256_string_output2 = cudf::hashing::sha256(string_input2);
  EXPECT_EQ(string_input1.num_rows(), sha256_string_output1->size());
  EXPECT_EQ(string_input2.num_rows(), sha256_string_output2->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sha256_string_output1->view(), sha256_string_results1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sha256_string_output2->view(), sha256_string_results2);

  // Test non-string inputs for self-consistency
  auto const input1         = cudf::table_view({strings_col, ints_col, bools_col});
  auto const input2         = cudf::table_view({strings_col, ints_col, bools_col});
  auto const sha256_output1 = cudf::hashing::sha256(input1);
  auto const sha256_output2 = cudf::hashing::sha256(input2);
  EXPECT_EQ(input1.num_rows(), sha256_output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sha256_output1->view(), sha256_output2->view());
}
TEST_F(SHA256HashTest, EmptyNullEquivalence)
{
  // Test that empty strings hash the same as nulls
  cudf::test::strings_column_wrapper const strings_col1({"", ""}, {true, false});
  cudf::test::strings_column_wrapper const strings_col2({"", ""}, {false, true});

  auto const input1 = cudf::table_view({strings_col1});
  auto const input2 = cudf::table_view({strings_col2});

  auto const output1 = cudf::hashing::sha256(input1);
  auto const output2 = cudf::hashing::sha256(input2);

  EXPECT_EQ(input1.num_rows(), output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

TEST_F(SHA256HashTest, ListsUnsupported)
{
  cudf::test::lists_column_wrapper<cudf::string_view> strings_list_col(
    {{""},
     {"", "Some inputs"},
     {"All ", "work ", "and", " no", " play ", "makes Jack", " a dull boy"},
     {R"(!"#$%&'()*+,-./0123456789:;<=>?@[\]^_`)", "{|}~"}});

  auto const input = cudf::table_view({strings_list_col});

  EXPECT_THROW(cudf::hashing::sha256(input), cudf::data_type_error);
}

TEST_F(SHA256HashTest, StructsUnsupported)
{
  auto child_col   = cudf::test::fixed_width_column_wrapper<int32_t>{0, 1, 2, 3};
  auto struct_col  = cudf::test::structs_column_wrapper{{child_col}};
  auto const input = cudf::table_view({struct_col});

  EXPECT_THROW(cudf::hashing::sha256(input), cudf::data_type_error);
}

template <typename T>
class SHA256HashTestTyped : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(SHA256HashTestTyped, cudf::test::NumericTypes);

TYPED_TEST(SHA256HashTestTyped, NoNulls)
{
  cudf::test::fixed_width_column_wrapper<TypeParam> const col({0, 127, 1, 2, 8});
  auto const input = cudf::table_view({col});

  // Hash of same input should be equal
  auto const output1 = cudf::hashing::sha256(input);
  auto const output2 = cudf::hashing::sha256(input);

  EXPECT_EQ(input.num_rows(), output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

TYPED_TEST(SHA256HashTestTyped, WithNulls)
{
  cudf::test::fixed_width_column_wrapper<TypeParam> const col({0, 127, 1, 2, 8}, {0, 1, 1, 1, 1});
  auto const input = cudf::table_view({col});

  // Hash of same input should be equal
  auto const output1 = cudf::hashing::sha256(input);
  auto const output2 = cudf::hashing::sha256(input);

  EXPECT_EQ(input.num_rows(), output1->size());
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

  auto const output1 = cudf::hashing::sha256(input1);
  auto const output2 = cudf::hashing::sha256(input2);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}
