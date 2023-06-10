/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

struct StringsLikeTests : public cudf::test::BaseFixture {};

TEST_F(StringsLikeTests, Basic)
{
  cudf::test::strings_column_wrapper input({"abc", "a bc", "ABC", "abcd", " abc", "", "", "áéêú"},
                                           {1, 1, 1, 1, 1, 1, 0, 1});
  auto const sv      = cudf::strings_column_view(input);
  auto const pattern = std::string("abc");
  auto const results = cudf::strings::like(sv, pattern);
  cudf::test::fixed_width_column_wrapper<bool> expected(
    {true, false, false, false, false, false, false, false}, {1, 1, 1, 1, 1, 1, 0, 1});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
}

TEST_F(StringsLikeTests, Leading)
{
  cudf::test::strings_column_wrapper input({"a", "aa", "aaa", "b", "bb", "bba", "", "áéêú"});
  auto const sv = cudf::strings_column_view(input);
  {
    auto const results = cudf::strings::like(sv, std::string("a%"));
    cudf::test::fixed_width_column_wrapper<bool> expected(
      {true, true, true, false, false, false, false, false});
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
  }
  {
    auto const results = cudf::strings::like(sv, std::string("__a%"));
    cudf::test::fixed_width_column_wrapper<bool> expected(
      {false, false, true, false, false, true, false, false});
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
  }
  {
    auto const results = cudf::strings::like(sv, std::string("á%"));
    cudf::test::fixed_width_column_wrapper<bool> expected(
      {false, false, false, false, false, false, false, true});
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
  }
}

TEST_F(StringsLikeTests, Trailing)
{
  cudf::test::strings_column_wrapper input({"a", "aa", "aaa", "b", "bb", "bba", "", "áéêú"});
  auto const sv = cudf::strings_column_view(input);
  {
    auto results = cudf::strings::like(sv, std::string("%a"));
    cudf::test::fixed_width_column_wrapper<bool> expected(
      {true, true, true, false, false, true, false, false});
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
    results = cudf::strings::like(sv, std::string("%a%"));
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
  }
  {
    auto const results = cudf::strings::like(sv, std::string("%_a"));
    cudf::test::fixed_width_column_wrapper<bool> expected(
      {false, true, true, false, false, true, false, false});
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
  }
  {
    auto const results = cudf::strings::like(sv, std::string("%_êú"));
    cudf::test::fixed_width_column_wrapper<bool> expected(
      {false, false, false, false, false, false, false, true});
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
  }
}

TEST_F(StringsLikeTests, Place)
{
  cudf::test::strings_column_wrapper input({"a", "aa", "aaa", "bab", "ab", "aba", "", "éaé"});
  auto const sv = cudf::strings_column_view(input);
  {
    auto const results = cudf::strings::like(sv, std::string("a_"));
    cudf::test::fixed_width_column_wrapper<bool> expected(
      {false, true, false, false, true, false, false, false});
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
  }
  {
    auto const results = cudf::strings::like(sv, std::string("_a_"));
    cudf::test::fixed_width_column_wrapper<bool> expected(
      {false, false, true, true, false, false, false, true});
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
  }
  {
    auto const results = cudf::strings::like(sv, std::string("__a"));
    cudf::test::fixed_width_column_wrapper<bool> expected(
      {false, false, true, false, false, true, false, false});
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
  }
  {
    auto const results = cudf::strings::like(sv, std::string("é_é"));
    cudf::test::fixed_width_column_wrapper<bool> expected(
      {false, false, false, false, false, false, false, true});
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
  }
}

TEST_F(StringsLikeTests, Escape)
{
  cudf::test::strings_column_wrapper input(
    {"10%-20%", "10-20", "10%%-20%", "a_b", "b_a", "___", "", "aéb"});
  auto const sv = cudf::strings_column_view(input);
  {
    auto const pattern = std::string("10\\%-20\\%");
    auto const escape  = std::string("\\");
    auto const results = cudf::strings::like(sv, pattern, escape);
    cudf::test::fixed_width_column_wrapper<bool> expected(
      {true, false, false, false, false, false, false, false});
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
  }
  {
    auto const pattern = std::string("\\__\\_");
    auto const escape  = std::string("\\");
    auto const results = cudf::strings::like(sv, pattern, escape);
    cudf::test::fixed_width_column_wrapper<bool> expected(
      {false, false, false, false, false, true, false, false});
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
  }
  {
    auto const pattern = std::string("10%%%%-20%%");
    auto const escape  = std::string("%");
    auto const results = cudf::strings::like(sv, pattern, escape);
    cudf::test::fixed_width_column_wrapper<bool> expected(
      {false, false, true, false, false, false, false, false});
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
  }
  {
    auto const pattern = std::string("_%__");
    auto const escape  = std::string("%");
    auto const results = cudf::strings::like(sv, pattern, escape);
    cudf::test::fixed_width_column_wrapper<bool> expected(
      {false, false, false, true, true, true, false, false});
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
  }
  {
    auto const pattern = std::string("a__b");
    auto const escape  = std::string("_");
    auto const results = cudf::strings::like(sv, pattern, escape);
    cudf::test::fixed_width_column_wrapper<bool> expected(
      {false, false, false, true, false, false, false, false});
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
  }
}

TEST_F(StringsLikeTests, MultiplePatterns)
{
  cudf::test::strings_column_wrapper input({"abc", "a1a2b3b4c", "aaabbb", "bbbc", "", "áéêú"});
  cudf::test::strings_column_wrapper patterns({"a%b%c", "a%c", "a__b", "b__c", "", "áéêú"});

  auto const sv_input    = cudf::strings_column_view(input);
  auto const sv_patterns = cudf::strings_column_view(patterns);
  auto const results     = cudf::strings::like(sv_input, sv_patterns);
  cudf::test::fixed_width_column_wrapper<bool> expected({true, true, false, true, true, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
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

  results = cudf::strings::like(sv, sv);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected_empty->view());
}

TEST_F(StringsLikeTests, Errors)
{
  auto const input       = cudf::test::strings_column_wrapper({"3", "33"});
  auto const sv          = cudf::strings_column_view(input);
  auto const invalid_str = cudf::string_scalar("", false);

  EXPECT_THROW(cudf::strings::like(sv, invalid_str), cudf::logic_error);
  EXPECT_THROW(cudf::strings::like(sv, std::string("3"), invalid_str), cudf::logic_error);

  auto patterns          = cudf::test::strings_column_wrapper({"3", ""}, {1, 0});
  auto const sv_patterns = cudf::strings_column_view(patterns);
  EXPECT_THROW(cudf::strings::like(sv, sv_patterns), cudf::logic_error);
  EXPECT_THROW(cudf::strings::like(sv, sv, invalid_str), cudf::logic_error);
}
