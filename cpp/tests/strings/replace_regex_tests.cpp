/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <cudf/strings/replace_re.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <tests/strings/utilities.h>

#include <thrust/iterator/transform_iterator.h>

#include <vector>

struct StringsReplaceRegexTest : public cudf::test::BaseFixture {
};

TEST_F(StringsReplaceRegexTest, ReplaceRegexTest)
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

TEST_F(StringsReplaceRegexTest, ReplaceMultiRegexTest)
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

TEST_F(StringsReplaceRegexTest, InvalidRegex)
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

TEST_F(StringsReplaceRegexTest, WithEmptyPattern)
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

TEST_F(StringsReplaceRegexTest, MultiReplacement)
{
  cudf::test::strings_column_wrapper input({"aba bcd aba", "abababa abababa"});
  auto results =
    cudf::strings::replace_re(cudf::strings_column_view(input), "aba", cudf::string_scalar("_"), 2);
  cudf::test::strings_column_wrapper expected({"_ bcd _", "_b_ abababa"});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
  results =
    cudf::strings::replace_re(cudf::strings_column_view(input), "aba", cudf::string_scalar(""), 0);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, input);
}

TEST_F(StringsReplaceRegexTest, WordBoundary)
{
  cudf::test::strings_column_wrapper input({"aba bcd\naba", "zéz", "A1B2-é3", "e é", "_", "a_b"});
  auto results =
    cudf::strings::replace_re(cudf::strings_column_view(input), "\\b", cudf::string_scalar("X"));
  auto expected = cudf::test::strings_column_wrapper(
    {"XabaX XbcdX\nXabaX", "XzézX", "XA1B2X-Xé3X", "XeX XéX", "X_X", "Xa_bX"});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
  results =
    cudf::strings::replace_re(cudf::strings_column_view(input), "\\B", cudf::string_scalar("X"));
  expected = cudf::test::strings_column_wrapper(
    {"aXbXa bXcXd\naXbXa", "zXéXz", "AX1XBX2-éX3", "e é", "_", "aX_Xb"});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
}

TEST_F(StringsReplaceRegexTest, Alternation)
{
  cudf::test::strings_column_wrapper input(
    {"16  6  brr  232323  1  hello  90", "123 ABC 00 2022", "abé123  4567  89xyz"});
  auto results = cudf::strings::replace_re(
    cudf::strings_column_view(input), "(^|\\s)\\d+(\\s|$)", cudf::string_scalar("_"));
  auto expected =
    cudf::test::strings_column_wrapper({"__ brr __ hello _", "_ABC_2022", "abé123 _ 89xyz"});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
  results = cudf::strings::replace_re(
    cudf::strings_column_view(input), "(\\s|^)\\d+($|\\s)", cudf::string_scalar("_"));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
}

TEST_F(StringsReplaceRegexTest, ZeroLengthMatch)
{
  cudf::test::strings_column_wrapper input({"DD", "zéz", "DsDs", ""});
  auto repl     = cudf::string_scalar("_");
  auto results  = cudf::strings::replace_re(cudf::strings_column_view(input), "D*", repl);
  auto expected = cudf::test::strings_column_wrapper({"__", "_z_é_z_", "__s__s_", "_"});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
  results  = cudf::strings::replace_re(cudf::strings_column_view(input), "D?s?", repl);
  expected = cudf::test::strings_column_wrapper({"___", "_z_é_z_", "___", "_"});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
}

TEST_F(StringsReplaceRegexTest, Multiline)
{
  auto const multiline = cudf::strings::regex_flags::MULTILINE;

  cudf::test::strings_column_wrapper input({"bcd\naba\nefg", "aba\naba abab\naba", "aba"});
  auto sv = cudf::strings_column_view(input);

  // single-replace
  auto results =
    cudf::strings::replace_re(sv, "^aba$", cudf::string_scalar("_"), std::nullopt, multiline);
  cudf::test::strings_column_wrapper expected_ml({"bcd\n_\nefg", "_\naba abab\n_", "_"});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_ml);

  results = cudf::strings::replace_re(sv, "^aba$", cudf::string_scalar("_"));
  cudf::test::strings_column_wrapper expected({"bcd\naba\nefg", "aba\naba abab\naba", "_"});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);

  // multi-replace
  std::vector<std::string> patterns({"aba$", "^aba"});
  cudf::test::strings_column_wrapper repls({">", "<"});
  results = cudf::strings::replace_re(sv, patterns, cudf::strings_column_view(repls), multiline);
  cudf::test::strings_column_wrapper multi_expected_ml({"bcd\n>\nefg", ">\n< abab\n>", ">"});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, multi_expected_ml);

  results = cudf::strings::replace_re(sv, patterns, cudf::strings_column_view(repls));
  cudf::test::strings_column_wrapper multi_expected({"bcd\naba\nefg", "<\naba abab\n>", ">"});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, multi_expected);

  // backref-replace
  results = cudf::strings::replace_with_backrefs(sv, "(^aba)", "[\\1]", multiline);
  cudf::test::strings_column_wrapper br_expected_ml(
    {"bcd\n[aba]\nefg", "[aba]\n[aba] abab\n[aba]", "[aba]"});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, br_expected_ml);

  results = cudf::strings::replace_with_backrefs(sv, "(^aba)", "[\\1]");
  cudf::test::strings_column_wrapper br_expected(
    {"bcd\naba\nefg", "[aba]\naba abab\naba", "[aba]"});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, br_expected);
}

TEST_F(StringsReplaceRegexTest, ReplaceBackrefsRegexTest)
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

TEST_F(StringsReplaceRegexTest, ReplaceBackrefsRegexAltIndexPatternTest)
{
  cudf::test::strings_column_wrapper strings({"12-3 34-5 67-89", "0-99: 777-888:: 5673-0"});
  auto strings_view = cudf::strings_column_view(strings);

  std::string pattern       = "(\\d+)-(\\d+)";
  std::string repl_template = "${2} X ${1}0";
  auto results = cudf::strings::replace_with_backrefs(strings_view, pattern, repl_template);

  cudf::test::strings_column_wrapper expected(
    {"3 X 120 5 X 340 89 X 670", "99 X 00: 888 X 7770:: 0 X 56730"});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsReplaceRegexTest, ReplaceBackrefsRegexReversedTest)
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

TEST_F(StringsReplaceRegexTest, BackrefWithGreedyQuantifier)
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

TEST_F(StringsReplaceRegexTest, ReplaceBackrefsRegexZeroIndexTest)
{
  cudf::test::strings_column_wrapper strings(
    {"TEST123", "TEST1TEST2", "TEST2-TEST1122", "TEST1-TEST-T", "TES3"});
  auto strings_view         = cudf::strings_column_view(strings);
  std::string pattern       = "(TEST)(\\d+)";
  std::string repl_template = "${0}: ${1}, ${2}; ";
  auto results = cudf::strings::replace_with_backrefs(strings_view, pattern, repl_template);

  cudf::test::strings_column_wrapper expected({
    "TEST123: TEST, 123; ",
    "TEST1: TEST, 1; TEST2: TEST, 2; ",
    "TEST2: TEST, 2; -TEST1122: TEST, 1122; ",
    "TEST1: TEST, 1; -TEST-T",
    "TES3",
  });
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsReplaceRegexTest, ReplaceBackrefsRegexErrorTest)
{
  cudf::test::strings_column_wrapper strings({"this string left intentionally blank"});
  auto view = cudf::strings_column_view(strings);

  // group index(3) exceeds the group count(2)
  EXPECT_THROW(cudf::strings::replace_with_backrefs(view, "(\\w).(\\w)", "\\3"), cudf::logic_error);
  EXPECT_THROW(cudf::strings::replace_with_backrefs(view, "", "\\1"), cudf::logic_error);
  EXPECT_THROW(cudf::strings::replace_with_backrefs(view, "(\\w)", ""), cudf::logic_error);
}

TEST_F(StringsReplaceRegexTest, MediumReplaceRegex)
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

TEST_F(StringsReplaceRegexTest, LargeReplaceRegex)
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
