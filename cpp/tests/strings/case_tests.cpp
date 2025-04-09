/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.
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

#include <cudf/column/column.hpp>
#include <cudf/strings/capitalize.hpp>
#include <cudf/strings/case.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <thrust/iterator/transform_iterator.h>

#include <vector>

struct StringsCaseTest : public cudf::test::BaseFixture {};

TEST_F(StringsCaseTest, ToLower)
{
  std::vector<char const*> h_strings{
    "Éxamples aBc", "123 456", nullptr, "ARE THE", "tést strings", ""};
  std::vector<char const*> h_expected{
    "éxamples abc", "123 456", nullptr, "are the", "tést strings", ""};

  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  auto strings_view = cudf::strings_column_view(strings);

  auto results = cudf::strings::to_lower(strings_view);

  cudf::test::strings_column_wrapper expected(
    h_expected.begin(),
    h_expected.end(),
    thrust::make_transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsCaseTest, ToUpper)
{
  std::vector<char const*> h_strings{
    "Éxamples aBc", "123 456", nullptr, "ARE THE", "tést strings", ""};
  std::vector<char const*> h_expected{
    "ÉXAMPLES ABC", "123 456", nullptr, "ARE THE", "TÉST STRINGS", ""};

  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  auto strings_view = cudf::strings_column_view(strings);

  auto results = cudf::strings::to_upper(strings_view);

  cudf::test::strings_column_wrapper expected(
    h_expected.begin(),
    h_expected.end(),
    thrust::make_transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsCaseTest, Swapcase)
{
  std::vector<char const*> h_strings{
    "Éxamples aBc", "123 456", nullptr, "ARE THE", "tést strings", ""};
  std::vector<char const*> h_expected{
    "éXAMPLES AbC", "123 456", nullptr, "are the", "TÉST STRINGS", ""};

  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  auto strings_view = cudf::strings_column_view(strings);

  auto results = cudf::strings::swapcase(strings_view);

  cudf::test::strings_column_wrapper expected(
    h_expected.begin(),
    h_expected.end(),
    thrust::make_transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsCaseTest, Capitalize)
{
  cudf::test::strings_column_wrapper strings(
    {"SȺȺnich xyZ", "Examples aBc", "thesé", "", "ARE\tTHE", "tést\tstrings", ""},
    {true, true, true, false, true, true, true});
  auto strings_view = cudf::strings_column_view(strings);

  {
    auto results = cudf::strings::capitalize(strings_view);
    cudf::test::strings_column_wrapper expected(
      {"Sⱥⱥnich xyz", "Examples abc", "Thesé", "", "Are\tthe", "Tést\tstrings", ""},
      {true, true, true, false, true, true, true});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto results = cudf::strings::capitalize(strings_view, std::string_view(" "));
    cudf::test::strings_column_wrapper expected(
      {"Sⱥⱥnich Xyz", "Examples Abc", "Thesé", "", "Are\tthe", "Tést\tstrings", ""},
      {true, true, true, false, true, true, true});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto results = cudf::strings::capitalize(strings_view, std::string_view(" \t"));
    cudf::test::strings_column_wrapper expected(
      {"Sⱥⱥnich Xyz", "Examples Abc", "Thesé", "", "Are\tThe", "Tést\tStrings", ""},
      {true, true, true, false, true, true, true});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(StringsCaseTest, Title)
{
  cudf::test::strings_column_wrapper input(
    {"SȺȺnich", "Examples aBc", "thesé", "", "ARE THE", "tést strings", "", "n2viDIA corp"},
    {true, true, true, false, true, true, true, true});
  auto strings_view = cudf::strings_column_view(input);

  auto results = cudf::strings::title(strings_view);

  cudf::test::strings_column_wrapper expected(
    {"Sⱥⱥnich", "Examples Abc", "Thesé", "", "Are The", "Tést Strings", "", "N2Vidia Corp"},
    {true, true, true, false, true, true, true, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  results = cudf::strings::title(strings_view, cudf::strings::string_character_types::ALPHANUM);

  cudf::test::strings_column_wrapper expected2(
    {"Sⱥⱥnich", "Examples Abc", "Thesé", "", "Are The", "Tést Strings", "", "N2vidia Corp"},
    {true, true, true, false, true, true, true, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected2);
}

TEST_F(StringsCaseTest, IsTitle)
{
  cudf::test::strings_column_wrapper input(
    {"Sⱥⱥnich",
     "Examples Abc",
     "Thesé Strings",
     "",
     "Are The",
     "Tést strings",
     "",
     "N2Vidia Corp",
     "SNAKE",
     "!Abc",
     " Eagle",
     "A Test",
     "12345",
     "Alpha Not Upper Or Lower: ƻC",
     "one More"},
    {true, true, true, false, true, true, true, true, true, true, true, true, true, true, true});

  auto results = cudf::strings::is_title(cudf::strings_column_view(input));

  cudf::test::fixed_width_column_wrapper<bool> expected(
    {1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0},
    {true, true, true, false, true, true, true, true, true, true, true, true, true, true, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsCaseTest, MultiCharUpper)
{
  cudf::test::strings_column_wrapper strings{"\u1f52 \u1f83", "\u1e98 \ufb05", "\u0149"};
  cudf::test::strings_column_wrapper expected{
    "\u03a5\u0313\u0300 \u1f0b\u0399", "\u0057\u030a \u0053\u0054", "\u02bc\u004e"};
  auto strings_view = cudf::strings_column_view(strings);

  auto results = cudf::strings::to_upper(strings_view);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  results = cudf::strings::capitalize(strings_view, std::string_view(" "));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  results = cudf::strings::title(strings_view);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsCaseTest, MultiCharLower)
{
  // there's only one of these
  cudf::test::strings_column_wrapper strings{"\u0130"};
  cudf::test::strings_column_wrapper expected{"\u0069\u0307"};
  auto strings_view = cudf::strings_column_view(strings);

  auto results = cudf::strings::to_lower(strings_view);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsCaseTest, Ascii)
{
  // triggering the ascii code path requires some long-ish strings
  cudf::test::strings_column_wrapper input{
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890!@#$%^&*()_+=- ",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890!@#$%^&*()_+=- ",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890!@#$%^&*()_+=- ",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890!@#$%^&*()_+=-"};
  auto view     = cudf::strings_column_view(input);
  auto expected = cudf::test::strings_column_wrapper{
    "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz1234567890!@#$%^&*()_+=- ",
    "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz1234567890!@#$%^&*()_+=- ",
    "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz1234567890!@#$%^&*()_+=- ",
    "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz1234567890!@#$%^&*()_+=-"};
  auto results = cudf::strings::to_lower(view);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  expected = cudf::test::strings_column_wrapper{
    "ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890!@#$%^&*()_+=- ",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890!@#$%^&*()_+=- ",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890!@#$%^&*()_+=- ",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890!@#$%^&*()_+=-"};
  results = cudf::strings::to_upper(view);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  results = cudf::strings::to_upper(cudf::strings_column_view(cudf::slice(input, {1, 3}).front()));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, cudf::slice(expected, {1, 3}).front());
}

TEST_F(StringsCaseTest, LongStrings)
{
  // average string length >= AVG_CHAR_BYTES_THRESHOLD as defined in case.cu
  cudf::test::strings_column_wrapper input{
    "abcdéfghijklmnopqrstuvwxyzABCDÉFGHIJKLMNOPQRSTUVWXYZ1234567890!@#$%^&*()_+=- ",
    "ABCDÉFGHIJKLMNOPQRSTUVWXYZabcdéfghijklmnopqrstuvwxyz1234567890!@#$%^&*()_+=- ",
    "ABCDÉFGHIJKLMNOPQRSTUVWXYZabcdéfghijklmnopqrstuvwxyz1234567890!@#$%^&*()_+=- ",
    "ABCDÉFGHIJKLMNOPQRSTUVWXYZabcdéfghijklmnopqrstuvwxyz1234567890!@#$%^&*()_+=-"};
  auto view     = cudf::strings_column_view(input);
  auto expected = cudf::test::strings_column_wrapper{
    "abcdéfghijklmnopqrstuvwxyzabcdéfghijklmnopqrstuvwxyz1234567890!@#$%^&*()_+=- ",
    "abcdéfghijklmnopqrstuvwxyzabcdéfghijklmnopqrstuvwxyz1234567890!@#$%^&*()_+=- ",
    "abcdéfghijklmnopqrstuvwxyzabcdéfghijklmnopqrstuvwxyz1234567890!@#$%^&*()_+=- ",
    "abcdéfghijklmnopqrstuvwxyzabcdéfghijklmnopqrstuvwxyz1234567890!@#$%^&*()_+=-"};
  auto results = cudf::strings::to_lower(view);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  expected = cudf::test::strings_column_wrapper{
    "ABCDÉFGHIJKLMNOPQRSTUVWXYZABCDÉFGHIJKLMNOPQRSTUVWXYZ1234567890!@#$%^&*()_+=- ",
    "ABCDÉFGHIJKLMNOPQRSTUVWXYZABCDÉFGHIJKLMNOPQRSTUVWXYZ1234567890!@#$%^&*()_+=- ",
    "ABCDÉFGHIJKLMNOPQRSTUVWXYZABCDÉFGHIJKLMNOPQRSTUVWXYZ1234567890!@#$%^&*()_+=- ",
    "ABCDÉFGHIJKLMNOPQRSTUVWXYZABCDÉFGHIJKLMNOPQRSTUVWXYZ1234567890!@#$%^&*()_+=-"};
  results = cudf::strings::to_upper(view);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  view    = cudf::strings_column_view(cudf::slice(input, {1, 3}).front());
  results = cudf::strings::to_upper(view);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, cudf::slice(expected, {1, 3}).front());
}

TEST_F(StringsCaseTest, EmptyStringsColumn)
{
  auto const zero_size_strings_column = cudf::make_empty_column(cudf::type_id::STRING)->view();

  auto strings_view = cudf::strings_column_view(zero_size_strings_column);

  auto results = cudf::strings::to_lower(strings_view);
  cudf::test::expect_column_empty(results->view());

  results = cudf::strings::to_upper(strings_view);
  cudf::test::expect_column_empty(results->view());

  results = cudf::strings::swapcase(strings_view);
  cudf::test::expect_column_empty(results->view());

  results = cudf::strings::capitalize(strings_view);
  cudf::test::expect_column_empty(results->view());

  results = cudf::strings::title(strings_view);
  cudf::test::expect_column_empty(results->view());
}

TEST_F(StringsCaseTest, ErrorTest)
{
  cudf::test::strings_column_wrapper input{"the column intentionally left blank"};
  auto view = cudf::strings_column_view(input);

  EXPECT_THROW(cudf::strings::capitalize(view, cudf::string_scalar("", false)), cudf::logic_error);
}
