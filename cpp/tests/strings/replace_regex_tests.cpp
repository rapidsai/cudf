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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>

#include <cudf/strings/regex/regex_program.hpp>
#include <cudf/strings/replace_re.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <thrust/iterator/transform_iterator.h>

#include <vector>

struct StringsReplaceRegexTest : public cudf::test::BaseFixture {};

TEST_F(StringsReplaceRegexTest, ReplaceRegexTest)
{
  std::vector<char const*> h_strings{"the quick brown fox jumps over the lazy dog",
                                     "the fat cat lays next to the other accénted cat",
                                     "a slow moving turtlé cannot catch the bird",
                                     "which can be composéd together to form a more complete",
                                     "thé result does not include the value in the sum in",
                                     "",
                                     nullptr};

  cudf::test::strings_column_wrapper strings(
    h_strings.begin(), h_strings.end(), cudf::test::iterators::nulls_from_nullptrs(h_strings));
  auto strings_view = cudf::strings_column_view(strings);

  std::vector<char const*> h_expected{"= quick brown fox jumps over = lazy dog",
                                      "= fat cat lays next to = other accénted cat",
                                      "a slow moving turtlé cannot catch = bird",
                                      "which can be composéd together to form a more complete",
                                      "thé result does not include = value in = sum in",
                                      "",
                                      nullptr};

  auto pattern = std::string("(\\bthe\\b)");
  auto repl    = cudf::string_scalar("=");
  cudf::test::strings_column_wrapper expected(
    h_expected.begin(), h_expected.end(), cudf::test::iterators::nulls_from_nullptrs(h_expected));
  auto prog    = cudf::strings::regex_program::create(pattern);
  auto results = cudf::strings::replace_re(strings_view, *prog, repl);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
}

TEST_F(StringsReplaceRegexTest, ReplaceMultiRegexTest)
{
  std::vector<char const*> h_strings{"the quick brown fox jumps over the lazy dog",
                                     "the fat cat lays next to the other accénted cat",
                                     "a slow moving turtlé cannot catch the bird",
                                     "which can be composéd together to form a more complete",
                                     "thé result does not include the value in the sum in",
                                     "",
                                     nullptr};

  cudf::test::strings_column_wrapper strings(
    h_strings.begin(), h_strings.end(), cudf::test::iterators::nulls_from_nullptrs(h_strings));
  auto strings_view = cudf::strings_column_view(strings);

  std::vector<char const*> h_expected{" quick brown fox jumps over  lazy dog",
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
    h_expected.begin(), h_expected.end(), cudf::test::iterators::nulls_from_nullptrs(h_expected));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
}

TEST_F(StringsReplaceRegexTest, InvalidRegex)
{
  // these are quantifiers that do not have a preceding character/class
  EXPECT_THROW(cudf::strings::regex_program::create("*"), cudf::logic_error);
  EXPECT_THROW(cudf::strings::regex_program::create("|"), cudf::logic_error);
  EXPECT_THROW(cudf::strings::regex_program::create("+"), cudf::logic_error);
  EXPECT_THROW(cudf::strings::regex_program::create("ab(*)"), cudf::logic_error);
  EXPECT_THROW(cudf::strings::regex_program::create("\\"), cudf::logic_error);
  EXPECT_THROW(cudf::strings::regex_program::create("\\p"), cudf::logic_error);
}

TEST_F(StringsReplaceRegexTest, WithEmptyPattern)
{
  std::vector<char const*> h_strings{"asd", "xcv"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(), h_strings.end(), cudf::test::iterators::nulls_from_nullptrs(h_strings));
  auto strings_view = cudf::strings_column_view(strings);

  auto empty_pattern = std::string("");
  auto repl          = cudf::string_scalar("bbb");
  std::vector<std::string> patterns({empty_pattern});
  cudf::test::strings_column_wrapper repls({"bbb"});
  auto repls_view = cudf::strings_column_view(repls);
  auto results    = cudf::strings::replace_re(strings_view, patterns, repls_view);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, strings);
  auto prog = cudf::strings::regex_program::create(empty_pattern);
  results   = cudf::strings::replace_re(strings_view, *prog, repl);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, strings);
}

TEST_F(StringsReplaceRegexTest, MultiReplacement)
{
  cudf::test::strings_column_wrapper input({"aba bcd aba", "abababa abababa"});
  auto sv = cudf::strings_column_view(input);

  auto pattern = std::string("aba");
  auto repl    = cudf::string_scalar("_");
  cudf::test::strings_column_wrapper expected({"_ bcd _", "_b_ abababa"});
  auto prog    = cudf::strings::regex_program::create(pattern);
  auto results = cudf::strings::replace_re(sv, *prog, repl, 2);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);

  results = cudf::strings::replace_re(sv, *prog, repl, 0);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, input);
}

TEST_F(StringsReplaceRegexTest, WordBoundary)
{
  cudf::test::strings_column_wrapper input({"aba bcd\naba", "zéz", "A1B2-é3", "e é", "_", "a_b"});
  auto sv = cudf::strings_column_view(input);

  auto pattern  = std::string("\\b");
  auto repl     = cudf::string_scalar("X");
  auto expected = cudf::test::strings_column_wrapper(
    {"XabaX XbcdX\nXabaX", "XzézX", "XA1B2X-Xé3X", "XeX XéX", "X_X", "Xa_bX"});
  auto prog    = cudf::strings::regex_program::create(pattern);
  auto results = cudf::strings::replace_re(sv, *prog, repl);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);

  pattern  = std::string("\\B");
  expected = cudf::test::strings_column_wrapper(
    {"aXbXa bXcXd\naXbXa", "zXéXz", "AX1XBX2-éX3", "e é", "_", "aX_Xb"});
  prog    = cudf::strings::regex_program::create(pattern);
  results = cudf::strings::replace_re(sv, *prog, repl);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
}

TEST_F(StringsReplaceRegexTest, Alternation)
{
  cudf::test::strings_column_wrapper input(
    {"16  6  brr  232323  1  hello  90", "123 ABC 00 2022", "abé123  4567  89xyz"});
  auto sv = cudf::strings_column_view(input);

  auto pattern = std::string("(^|\\s)\\d+(\\s|$)");
  auto repl    = cudf::string_scalar("_");
  auto expected =
    cudf::test::strings_column_wrapper({"__ brr __ hello _", "_ABC_2022", "abé123 _ 89xyz"});
  auto prog    = cudf::strings::regex_program::create(pattern);
  auto results = cudf::strings::replace_re(sv, *prog, repl);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);

  pattern = std::string("(\\s|^)\\d+($|\\s)");
  prog    = cudf::strings::regex_program::create(pattern);
  results = cudf::strings::replace_re(sv, *prog, repl);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
}

TEST_F(StringsReplaceRegexTest, ZeroLengthMatch)
{
  cudf::test::strings_column_wrapper input({"DD", "zéz", "DsDs", ""});
  auto sv = cudf::strings_column_view(input);

  auto pattern  = std::string("D*");
  auto repl     = cudf::string_scalar("_");
  auto expected = cudf::test::strings_column_wrapper({"__", "_z_é_z_", "__s__s_", "_"});
  auto prog     = cudf::strings::regex_program::create(pattern);
  auto results  = cudf::strings::replace_re(sv, *prog, repl);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);

  pattern  = std::string("D?s?");
  expected = cudf::test::strings_column_wrapper({"___", "_z_é_z_", "___", "_"});
  prog     = cudf::strings::regex_program::create(pattern);
  results  = cudf::strings::replace_re(sv, *prog, repl);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
}

TEST_F(StringsReplaceRegexTest, Multiline)
{
  auto const multiline = cudf::strings::regex_flags::MULTILINE;

  cudf::test::strings_column_wrapper input({"bcd\naba\nefg", "aba\naba abab\naba", "aba"});
  auto sv = cudf::strings_column_view(input);

  // single-replace
  auto pattern = std::string("^aba$");
  auto repl    = cudf::string_scalar("_");
  cudf::test::strings_column_wrapper expected_ml({"bcd\n_\nefg", "_\naba abab\n_", "_"});
  auto prog    = cudf::strings::regex_program::create(pattern, multiline);
  auto results = cudf::strings::replace_re(sv, *prog, repl);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_ml);

  cudf::test::strings_column_wrapper expected({"bcd\naba\nefg", "aba\naba abab\naba", "_"});
  prog    = cudf::strings::regex_program::create(pattern);
  results = cudf::strings::replace_re(sv, *prog, repl);
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
  auto repl_template = std::string("[\\1]");
  pattern            = std::string("(^aba)");
  cudf::test::strings_column_wrapper br_expected_ml(
    {"bcd\n[aba]\nefg", "[aba]\n[aba] abab\n[aba]", "[aba]"});
  prog    = cudf::strings::regex_program::create(pattern, multiline);
  results = cudf::strings::replace_with_backrefs(sv, *prog, repl_template);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, br_expected_ml);

  cudf::test::strings_column_wrapper br_expected(
    {"bcd\naba\nefg", "[aba]\naba abab\naba", "[aba]"});
  prog    = cudf::strings::regex_program::create(pattern);
  results = cudf::strings::replace_with_backrefs(sv, *prog, repl_template);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, br_expected);
}

TEST_F(StringsReplaceRegexTest, ReplaceBackrefsRegexTest)
{
  std::vector<char const*> h_strings{"the quick brown fox jumps over the lazy dog",
                                     "the fat cat lays next to the other accénted cat",
                                     "a slow moving turtlé cannot catch the bird",
                                     "which can be composéd together to form a more complete",
                                     "thé result does not include the value in the sum in",
                                     "",
                                     nullptr};

  cudf::test::strings_column_wrapper strings(
    h_strings.begin(), h_strings.end(), cudf::test::iterators::nulls_from_nullptrs(h_strings));
  auto sv = cudf::strings_column_view(strings);

  std::vector<char const*> h_expected{"the-quick-brown-fox-jumps-over-the-lazy-dog",
                                      "the-fat-cat-lays-next-to-the-other-accénted-cat",
                                      "a-slow-moving-turtlé-cannot-catch-the-bird",
                                      "which-can-be-composéd-together-to-form-a more-complete",
                                      "thé-result-does-not-include-the-value-in-the-sum-in",
                                      "",
                                      nullptr};

  auto pattern       = std::string("(\\w) (\\w)");
  auto repl_template = std::string("\\1-\\2");
  cudf::test::strings_column_wrapper expected(
    h_expected.begin(), h_expected.end(), cudf::test::iterators::nulls_from_nullptrs(h_expected));
  auto prog    = cudf::strings::regex_program::create(pattern);
  auto results = cudf::strings::replace_with_backrefs(sv, *prog, repl_template);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
}

TEST_F(StringsReplaceRegexTest, ReplaceBackrefsRegexAltIndexPatternTest)
{
  cudf::test::strings_column_wrapper input({"12-3 34-5 67-89", "0-99: 777-888:: 5673-0"});
  auto sv = cudf::strings_column_view(input);

  auto pattern       = std::string("(\\d+)-(\\d+)");
  auto repl_template = std::string("${2} X ${1}0");

  cudf::test::strings_column_wrapper expected(
    {"3 X 120 5 X 340 89 X 670", "99 X 00: 888 X 7770:: 0 X 56730"});
  auto prog    = cudf::strings::regex_program::create(pattern);
  auto results = cudf::strings::replace_with_backrefs(sv, *prog, repl_template);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
}

TEST_F(StringsReplaceRegexTest, ReplaceBackrefsRegexReversedTest)
{
  cudf::test::strings_column_wrapper strings(
    {"A543", "Z756", "", "tést-string", "two-thréé four-fivé", "abcd-éfgh", "tést-string-again"});
  auto sv = cudf::strings_column_view(strings);

  auto pattern       = std::string("([a-z])-([a-zé])");
  auto repl_template = std::string("X\\2+\\1Z");

  cudf::test::strings_column_wrapper expected({"A543",
                                               "Z756",
                                               "",
                                               "tésXs+tZtring",
                                               "twXt+oZhréé fouXf+rZivé",
                                               "abcXé+dZfgh",
                                               "tésXs+tZtrinXa+gZgain"});
  auto prog    = cudf::strings::regex_program::create(pattern);
  auto results = cudf::strings::replace_with_backrefs(sv, *prog, repl_template);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
}

TEST_F(StringsReplaceRegexTest, BackrefWithGreedyQuantifier)
{
  cudf::test::strings_column_wrapper input(
    {"<h1>title</h1><h2>ABC</h2>", "<h1>1234567</h1><h2>XYZ</h2>"});
  auto sv = cudf::strings_column_view(input);

  auto pattern       = std::string("<h1>(.*)</h1><h2>(.*)</h2>");
  auto repl_template = std::string("<h2>\\1</h2><p>\\2</p>");

  cudf::test::strings_column_wrapper expected(
    {"<h2>title</h2><p>ABC</p>", "<h2>1234567</h2><p>XYZ</p>"});
  auto prog    = cudf::strings::regex_program::create(pattern);
  auto results = cudf::strings::replace_with_backrefs(sv, *prog, repl_template);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);

  pattern = std::string("<h1>([a-z\\d]+)</h1><h2>([A-Z]+)</h2>");
  prog    = cudf::strings::regex_program::create(pattern);
  results = cudf::strings::replace_with_backrefs(sv, *prog, repl_template);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
}

TEST_F(StringsReplaceRegexTest, ReplaceBackrefsRegexZeroIndexTest)
{
  cudf::test::strings_column_wrapper strings(
    {"TEST123", "TEST1TEST2", "TEST2-TEST1122", "TEST1-TEST-T", "TES3"});
  auto sv = cudf::strings_column_view(strings);

  auto pattern       = std::string("(TEST)(\\d+)");
  auto repl_template = std::string("${0}: ${1}, ${2}; ");

  cudf::test::strings_column_wrapper expected({
    "TEST123: TEST, 123; ",
    "TEST1: TEST, 1; TEST2: TEST, 2; ",
    "TEST2: TEST, 2; -TEST1122: TEST, 1122; ",
    "TEST1: TEST, 1; -TEST-T",
    "TES3",
  });
  auto prog    = cudf::strings::regex_program::create(pattern);
  auto results = cudf::strings::replace_with_backrefs(sv, *prog, repl_template);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
}

// https://github.com/rapidsai/cudf/issues/13404
TEST_F(StringsReplaceRegexTest, ReplaceBackrefsWithEmptyCapture)
{
  cudf::test::strings_column_wrapper input({"one\ntwo", "three\n\n", "four\r\n"});
  auto sv = cudf::strings_column_view(input);

  auto pattern       = std::string("(\r\n|\r)?$");
  auto repl_template = std::string("[\\1]");

  cudf::test::strings_column_wrapper expected({"one\ntwo[]", "three\n[]\n[]", "four[\r\n][]"});
  auto prog    = cudf::strings::regex_program::create(pattern);
  auto results = cudf::strings::replace_with_backrefs(sv, *prog, repl_template);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
}

TEST_F(StringsReplaceRegexTest, ReplaceBackrefsRegexErrorTest)
{
  cudf::test::strings_column_wrapper strings({"this string left intentionally blank"});
  auto view = cudf::strings_column_view(strings);

  // group index(3) exceeds the group count(2)
  auto prog = cudf::strings::regex_program::create("(\\w).(\\w)");
  EXPECT_THROW(cudf::strings::replace_with_backrefs(view, *prog, "\\3"), cudf::logic_error);
  prog = cudf::strings::regex_program::create("");
  EXPECT_THROW(cudf::strings::replace_with_backrefs(view, *prog, "\\1"), cudf::logic_error);
  prog = cudf::strings::regex_program::create("(\\w)");
  EXPECT_THROW(cudf::strings::replace_with_backrefs(view, *prog, ""), cudf::logic_error);
}

TEST_F(StringsReplaceRegexTest, MediumReplaceRegex)
{
  // This results in 95 regex instructions and falls in the 'medium' range.
  std::string medium_regex =
    "hello @abc @def world The quick brown @fox jumps over the lazy @dog hello "
    "http://www.world.com";
  auto prog = cudf::strings::regex_program::create(medium_regex);

  std::vector<char const*> h_strings{
    "hello @abc @def world The quick brown @fox jumps over the lazy @dog hello "
    "http://www.world.com thats all",
    "12345678901234567890",
    "abcdefghijklmnopqrstuvwxyz"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));

  auto strings_view = cudf::strings_column_view(strings);
  auto results      = cudf::strings::replace_re(strings_view, *prog);
  std::vector<char const*> h_expected{
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
  auto prog = cudf::strings::regex_program::create(large_regex);

  std::vector<char const*> h_strings{
    "zzzz hello @abc @def world The quick brown @fox jumps over the lazy @dog hello "
    "http://www.world.com I'm here @home zzzz",
    "12345678901234567890",
    "abcdefghijklmnopqrstuvwxyz"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));

  auto strings_view = cudf::strings_column_view(strings);
  auto results      = cudf::strings::replace_re(strings_view, *prog);
  std::vector<char const*> h_expected{
    "zzzz ", "12345678901234567890", "abcdefghijklmnopqrstuvwxyz"};
  cudf::test::strings_column_wrapper expected(
    h_expected.begin(),
    h_expected.end(),
    thrust::make_transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}
