/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cudf/strings/contains.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

struct StringsLikeTests : public cudf::test::BaseFixture {
};

TEST_F(StringsLikeTests, Basic)
{
  cudf::test::strings_column_wrapper input({"abc", "a bc", "ABC", "abcd", " abc", "", ""},
                                           {1, 1, 1, 1, 1, 1, 0});
  auto sv      = cudf::strings_column_view(input);
  auto pattern = std::string("abc");
  auto results = cudf::strings::like(sv, pattern);
  cudf::test::fixed_width_column_wrapper<bool> expected(
    {true, false, false, false, false, false, false}, {1, 1, 1, 1, 1, 1, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
}

TEST_F(StringsLikeTests, Leading)
{
  cudf::test::strings_column_wrapper input({"a", "aa", "aaa", "b", "bb", "bba", ""});
  auto sv = cudf::strings_column_view(input);
  {
    auto results = cudf::strings::like(sv, std::string("a%"));
    cudf::test::fixed_width_column_wrapper<bool> expected(
      {true, true, true, false, false, false, false});
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
  }
  {
    auto results = cudf::strings::like(sv, std::string("__a%"));
    cudf::test::fixed_width_column_wrapper<bool> expected(
      {false, false, true, false, false, true, false});
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
  }
}

TEST_F(StringsLikeTests, Trailing)
{
  cudf::test::strings_column_wrapper input({"a", "aa", "aaa", "b", "bb", "bba", ""});
  auto sv = cudf::strings_column_view(input);
  {
    auto results = cudf::strings::like(sv, std::string("%a"));
    cudf::test::fixed_width_column_wrapper<bool> expected(
      {true, true, true, false, false, true, false});
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
    results = cudf::strings::like(sv, std::string("%a%"));
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
  }
  {
    auto results = cudf::strings::like(sv, std::string("%_a"));
    cudf::test::fixed_width_column_wrapper<bool> expected(
      {false, true, true, false, false, true, false});
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
  }
}

TEST_F(StringsLikeTests, Place)
{
  cudf::test::strings_column_wrapper input({"a", "aa", "aaa", "bab", "ab", "aba", ""});
  auto sv = cudf::strings_column_view(input);
  {
    auto results = cudf::strings::like(sv, std::string("a_"));
    cudf::test::fixed_width_column_wrapper<bool> expected(
      {false, true, false, false, true, false, false});
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
  }
  {
    auto results = cudf::strings::like(sv, std::string("_a_"));
    cudf::test::fixed_width_column_wrapper<bool> expected(
      {false, false, true, true, false, false, false});
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
  }
  {
    auto results = cudf::strings::like(sv, std::string("__a"));
    cudf::test::fixed_width_column_wrapper<bool> expected(
      {false, false, true, false, false, true, false});
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
  }
}

TEST_F(StringsLikeTests, Escape)
{
  cudf::test::strings_column_wrapper input(
    {"10%-20%", "10-20", "10%%-20%", "a_b", "b_a", "___", ""});
  auto sv = cudf::strings_column_view(input);
  {
    auto results = cudf::strings::like(sv, std::string("10\\%-20\\%"), std::string("\\"));
    cudf::test::fixed_width_column_wrapper<bool> expected(
      {true, false, false, false, false, false, false});
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
  }
  {
    auto results = cudf::strings::like(sv, std::string("\\__\\_"), std::string("\\"));
    cudf::test::fixed_width_column_wrapper<bool> expected(
      {false, false, false, false, false, true, false});
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
  }
}

TEST_F(StringsLikeTests, Empty)
{
  cudf::test::strings_column_wrapper input({"ooo", "20%", ""});
  auto sv       = cudf::strings_column_view(input);
  auto results  = cudf::strings::like(sv, std::string(""));
  auto expected = cudf::test::fixed_width_column_wrapper<bool>({false, false, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);

  auto empty          = cudf::make_empty_column(cudf::type_id::STRING);
  sv                  = cudf::strings_column_view(empty->view());
  results             = cudf::strings::like(sv, std::string("20%"));
  auto expected_empty = cudf::make_empty_column(cudf::type_id::BOOL8);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected_empty->view());
}

TEST_F(StringsLikeTests, Errors)
{
  cudf::test::strings_column_wrapper input({"3", "33"});
  auto sv = cudf::strings_column_view(input);

  EXPECT_THROW(cudf::strings::like(sv, cudf::string_scalar("", false)), cudf::logic_error);
  EXPECT_THROW(cudf::strings::like(sv, std::string("3"), cudf::string_scalar("", false)),
               cudf::logic_error);
}
