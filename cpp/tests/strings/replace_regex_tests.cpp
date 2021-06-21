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

#include <tests/strings/utilities.h>
#include <cudf/strings/replace_re.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <vector>

struct StringsReplaceTests : public cudf::test::BaseFixture {
};

TEST_F(StringsReplaceTests, ReplaceRegexTest)
{
  std::vector<const char*> h_strings{"the quick brown fox jumps over the lazy dog",
                                     "the fat cat lays next to the other accénted cat",
                                     "a slow moving turtlé cannot catch the bird",
                                     "which can be composéd together to form a more complete",
                                     "thé result does not include the value in the sum in",
                                     "",
                                     nullptr};

  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  auto strings_view = cudf::strings_column_view(strings);

  std::vector<const char*> h_expected{"= quick brown fox jumps over = lazy dog",
                                      "= fat cat lays next to = other accénted cat",
                                      "a slow moving turtlé cannot catch = bird",
                                      "which can be composéd together to form a more complete",
                                      "thé result does not include = value in = sum in",
                                      "",
                                      nullptr};

  std::string pattern = "(\\bthe\\b)";
  auto results        = cudf::strings::replace_re(strings_view, pattern, cudf::string_scalar("="));
  cudf::test::strings_column_wrapper expected(
    h_expected.begin(),
    h_expected.end(),
    thrust::make_transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsReplaceTests, ReplaceMultiRegexTest)
{
  std::vector<const char*> h_strings{"the quick brown fox jumps over the lazy dog",
                                     "the fat cat lays next to the other accénted cat",
                                     "a slow moving turtlé cannot catch the bird",
                                     "which can be composéd together to form a more complete",
                                     "thé result does not include the value in the sum in",
                                     "",
                                     nullptr};

  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  auto strings_view = cudf::strings_column_view(strings);

  std::vector<const char*> h_expected{" quick brown fox jumps over  lazy dog",
                                      " fat cat lays next to  other accénted cat",
                                      "** slow moving turtlé cannot catch  bird",
                                      "which can be composéd together to form ** more complete",
                                      "thé result does not include  value N  sum N",
                                      "",
                                      nullptr};

  std::vector<std::string> patterns{"\\bthe\\b", "\\bin\\b", "\\ba\\b"};
  std::vector<std::string> h_repls{"", "N", "**"};
  cudf::test::strings_column_wrapper repls(h_repls.begin(), h_repls.end());
  auto repls_view = cudf::strings_column_view(repls);
  auto results    = cudf::strings::replace_re(strings_view, patterns, repls_view);
  cudf::test::strings_column_wrapper expected(
    h_expected.begin(),
    h_expected.end(),
    thrust::make_transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsReplaceTests, InvalidRegex)
{
  cudf::test::strings_column_wrapper strings(
    {"abc*def|ghi+jkl", ""});  // these do not really matter
  auto strings_view = cudf::strings_column_view(strings);

  // these are quantifiers that do not have a preceding character/class
  EXPECT_THROW(cudf::strings::replace_re(strings_view, "*", cudf::string_scalar("")),
               cudf::logic_error);
  EXPECT_THROW(cudf::strings::replace_re(strings_view, "|", cudf::string_scalar("")),
               cudf::logic_error);
  EXPECT_THROW(cudf::strings::replace_re(strings_view, "+", cudf::string_scalar("")),
               cudf::logic_error);
  EXPECT_THROW(cudf::strings::replace_re(strings_view, "ab(*)", cudf::string_scalar("")),
               cudf::logic_error);
  EXPECT_THROW(cudf::strings::replace_re(strings_view, "\\", cudf::string_scalar("")),
               cudf::logic_error);
  EXPECT_THROW(cudf::strings::replace_re(strings_view, "\\p", cudf::string_scalar("")),
               cudf::logic_error);
}

TEST_F(StringsReplaceTests, WithEmptyPattern)
{
  std::vector<const char*> h_strings{"asd", "xcv"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  auto strings_view = cudf::strings_column_view(strings);
  std::vector<std::string> patterns({""});
  cudf::test::strings_column_wrapper repls({"bbb"});
  auto repls_view = cudf::strings_column_view(repls);
  auto results    = cudf::strings::replace_re(strings_view, patterns, repls_view);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, strings);
  results = cudf::strings::replace_re(strings_view, "", cudf::string_scalar("bbb"));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, strings);
}

TEST_F(StringsReplaceTests, ReplaceBackrefsRegexTest)
{
  std::vector<const char*> h_strings{"the quick brown fox jumps over the lazy dog",
                                     "the fat cat lays next to the other accénted cat",
                                     "a slow moving turtlé cannot catch the bird",
                                     "which can be composéd together to form a more complete",
                                     "thé result does not include the value in the sum in",
                                     "",
                                     nullptr};

  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  auto strings_view = cudf::strings_column_view(strings);

  std::vector<const char*> h_expected{"the-quick-brown-fox-jumps-over-the-lazy-dog",
                                      "the-fat-cat-lays-next-to-the-other-accénted-cat",
                                      "a-slow-moving-turtlé-cannot-catch-the-bird",
                                      "which-can-be-composéd-together-to-form-a more-complete",
                                      "thé-result-does-not-include-the-value-in-the-sum-in",
                                      "",
                                      nullptr};

  std::string pattern       = "(\\w) (\\w)";
  std::string repl_template = "\\1-\\2";
  auto results = cudf::strings::replace_with_backrefs(strings_view, pattern, repl_template);
  cudf::test::strings_column_wrapper expected(
    h_expected.begin(),
    h_expected.end(),
    thrust::make_transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsReplaceTests, ReplaceBackrefsRegexReversedTest)
{
  cudf::test::strings_column_wrapper strings(
    {"A543", "Z756", "", "tést-string", "two-thréé four-fivé", "abcd-éfgh", "tést-string-again"});
  auto strings_view         = cudf::strings_column_view(strings);
  std::string pattern       = "([a-z])-([a-zé])";
  std::string repl_template = "X\\2+\\1Z";
  auto results = cudf::strings::replace_with_backrefs(strings_view, pattern, repl_template);

  cudf::test::strings_column_wrapper expected({"A543",
                                               "Z756",
                                               "",
                                               "tésXs+tZtring",
                                               "twXt+oZhréé fouXf+rZivé",
                                               "abcXé+dZfgh",
                                               "tésXs+tZtrinXa+gZgain"});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsReplaceTests, BackrefWithGreedyQuantifier)
{
  cudf::test::strings_column_wrapper input(
    {"<h1>title</h1><h2>ABC</h2>", "<h1>1234567</h1><h2>XYZ</h2>"});
  std::string replacement = "<h2>\\1</h2><p>\\2</p>";

  auto results = cudf::strings::replace_with_backrefs(
    cudf::strings_column_view(input), "<h1>(.*)</h1><h2>(.*)</h2>", replacement);
  cudf::test::strings_column_wrapper expected(
    {"<h2>title</h2><p>ABC</p>", "<h2>1234567</h2><p>XYZ</p>"});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  results = cudf::strings::replace_with_backrefs(
    cudf::strings_column_view(input), "<h1>([a-z\\d]+)</h1><h2>([A-Z]+)</h2>", replacement);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsReplaceTests, MediumReplaceRegex)
{
  // This results in 95 regex instructions and falls in the 'medium' range.
  std::string medium_regex =
    "hello @abc @def world The quick brown @fox jumps over the lazy @dog hello "
    "http://www.world.com";

  std::vector<const char*> h_strings{
    "hello @abc @def world The quick brown @fox jumps over the lazy @dog hello "
    "http://www.world.com thats all",
    "12345678901234567890",
    "abcdefghijklmnopqrstuvwxyz"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));

  auto strings_view = cudf::strings_column_view(strings);
  auto results      = cudf::strings::replace_re(strings_view, medium_regex);
  std::vector<const char*> h_expected{
    " thats all", "12345678901234567890", "abcdefghijklmnopqrstuvwxyz"};
  cudf::test::strings_column_wrapper expected(
    h_expected.begin(),
    h_expected.end(),
    thrust::make_transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsReplaceTests, LargeReplaceRegex)
{
  // This results in 117 regex instructions and falls in the 'large' range.
  std::string large_regex =
    "hello @abc @def world The (quick) brown @fox jumps over the lazy @dog hello "
    "http://www.world.com I'm here @home zzzz";

  std::vector<const char*> h_strings{
    "zzzz hello @abc @def world The quick brown @fox jumps over the lazy @dog hello "
    "http://www.world.com I'm here @home zzzz",
    "12345678901234567890",
    "abcdefghijklmnopqrstuvwxyz"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));

  auto strings_view = cudf::strings_column_view(strings);
  auto results      = cudf::strings::replace_re(strings_view, large_regex);
  std::vector<const char*> h_expected{
    "zzzz ", "12345678901234567890", "abcdefghijklmnopqrstuvwxyz"};
  cudf::test::strings_column_wrapper expected(
    h_expected.begin(),
    h_expected.end(),
    thrust::make_transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}
