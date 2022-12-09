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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>

#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/regex/regex_program.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <algorithm>
#include <vector>

struct StringsContainsTests : public cudf::test::BaseFixture {
};

TEST_F(StringsContainsTests, ContainsTest)
{
  std::vector<const char*> h_strings{"5",
                                     "hej",
                                     "\t \n",
                                     "12345",
                                     "\\",
                                     "d",
                                     "c:\\Tools",
                                     "+27",
                                     "1c2",
                                     "1C2",
                                     "0:00:0",
                                     "0:0:00",
                                     "00:0:0",
                                     "00:00:0",
                                     "00:0:00",
                                     "0:00:00",
                                     "00:00:00",
                                     "Hello world !",
                                     "Hello world!   ",
                                     "Hello worldcup  !",
                                     "0123456789",
                                     "1C2",
                                     "Xaa",
                                     "abcdefghxxx",
                                     "ABCDEFGH",
                                     "abcdefgh",
                                     "abc def",
                                     "abc\ndef",
                                     "aa\r\nbb\r\ncc\r\n\r\n",
                                     "abcabc",
                                     nullptr,
                                     ""};

  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  auto strings_view = cudf::strings_column_view(strings);

  std::vector<std::string> patterns{"\\d",
                                    "\\w+",
                                    "\\s",
                                    "\\S",
                                    "^.*\\\\.*$",
                                    "[1-5]+",
                                    "[a-h]+",
                                    "[A-H]+",
                                    "[a-h]*",
                                    "\n",
                                    "b.\\s*\n",
                                    ".*c",
                                    "\\d\\d:\\d\\d:\\d\\d",
                                    "\\d\\d?:\\d\\d?:\\d\\d?",
                                    "[Hh]ello [Ww]orld",
                                    "\\bworld\\b",
                                    ".*"};

  std::vector<bool> h_expecteds_std{
    // strings.size x patterns.size
    true,  false, false, true,  false, false, false, true,  true,  true,  true,  true,  true,
    true,  true,  true,  true,  false, false, false, true,  true,  false, false, false, false,
    false, false, false, false, false, false, true,  true,  false, true,  false, true,  true,
    true,  true,  true,  true,  true,  true,  true,  true,  true,  true,  true,  true,  true,
    true,  true,  true,  true,  true,  true,  true,  true,  true,  true,  false, false, false,
    false, true,  false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, true,  true,  true,  false, false, false, false, false, false, true,
    true,  true,  false, false, false, true,  true,  false, true,  true,  true,  true,  true,
    true,  true,  true,  true,  true,  true,  true,  true,  true,  true,  true,  true,  true,
    true,  true,  true,  true,  true,  true,  true,  true,  true,  false, false, false, false,
    false, false, true,  false, true,  false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, true,  false, false, true,  false, false, false, true,  true,
    true,  false, false, false, false, false, false, false, false, false, false, true,  true,
    false, false, false, false, false, false, false, false, false, false, false, true,  false,
    false, false, true,  true,  false, true,  false, false, false, false, false, false, false,
    false, true,  true,  true,  false, false, true,  true,  false, true,  true,  true,  true,
    true,  false, false, false, false, false, false, false, false, false, false, false, true,
    false, false, false, false, false, false, false, true,  true,  true,  false, true,  false,
    false, true,  false, false, false, false, false, false, false, true,  true,  true,  true,
    true,  true,  true,  true,  true,  true,  true,  true,  true,  true,  true,  true,  true,
    true,  true,  true,  true,  true,  true,  true,  true,  true,  true,  true,  true,  true,
    false, true,  false, false, true,  false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, true,  true,  false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, true,  true,  false, false,
    false, false, false, false, false, false, false, true,  false, true,  false, false, false,
    false, false, false, false, false, false, false, true,  false, false, false, true,  false,
    true,  true,  true,  true,  true,  false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, true,  false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, true,  true,  true,
    true,  true,  true,  true,  false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, true,  true,  true,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, true,  true,  false, false, false, false, false, false, false, false,
    false, false, false, false, false, true,  true,  true,  true,  true,  true,  true,  true,
    true,  true,  true,  true,  true,  true,  true,  true,  true,  true,  true,  true,  true,
    true,  true,  true,  true,  true,  true,  true,  true,  true,  false, true};

  thrust::host_vector<bool> h_expecteds(h_expecteds_std);

  for (int idx = 0; idx < static_cast<int>(patterns.size()); ++idx) {
    std::string ptn  = patterns[idx];
    auto results     = cudf::strings::contains_re(strings_view, ptn);
    bool* h_expected = h_expecteds.data() + (idx * h_strings.size());
    cudf::test::fixed_width_column_wrapper<bool> expected(
      h_expected,
      h_expected + h_strings.size(),
      thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
    auto prog = cudf::strings::regex_program::create(ptn);
    results   = cudf::strings::contains_re(strings_view, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(StringsContainsTests, MatchesTest)
{
  std::vector<const char*> h_strings{
    "The quick brown @fox jumps", "ovér the", "lazy @dog", "1234", "00:0:00", nullptr, ""};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));

  auto strings_view = cudf::strings_column_view(strings);
  {
    auto const pattern = std::string("lazy");
    auto results       = cudf::strings::matches_re(strings_view, pattern);
    bool h_expected[]  = {false, false, true, false, false, false, false};
    cudf::test::fixed_width_column_wrapper<bool> expected(
      h_expected,
      h_expected + h_strings.size(),
      thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
    auto prog = cudf::strings::regex_program::create(pattern);
    results   = cudf::strings::matches_re(strings_view, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto const pattern = std::string("\\d+");
    auto results       = cudf::strings::matches_re(strings_view, pattern);
    bool h_expected[]  = {false, false, false, true, true, false, false};
    cudf::test::fixed_width_column_wrapper<bool> expected(
      h_expected,
      h_expected + h_strings.size(),
      thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
    auto prog = cudf::strings::regex_program::create(pattern);
    results   = cudf::strings::matches_re(strings_view, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto const pattern = std::string("@\\w+");
    auto results       = cudf::strings::matches_re(strings_view, pattern);
    bool h_expected[]  = {false, false, false, false, false, false, false};
    cudf::test::fixed_width_column_wrapper<bool> expected(
      h_expected,
      h_expected + h_strings.size(),
      thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
    auto prog = cudf::strings::regex_program::create(pattern);
    results   = cudf::strings::matches_re(strings_view, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto const pattern = std::string(".*");
    auto results       = cudf::strings::matches_re(strings_view, pattern);
    bool h_expected[]  = {true, true, true, true, true, false, true};
    cudf::test::fixed_width_column_wrapper<bool> expected(
      h_expected,
      h_expected + h_strings.size(),
      thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
    auto prog = cudf::strings::regex_program::create(pattern);
    results   = cudf::strings::matches_re(strings_view, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(StringsContainsTests, MatchesIPV4Test)
{
  cudf::test::strings_column_wrapper strings({"5.79.97.178",
                                              "1.2.3.4",
                                              "5",
                                              "5.79",
                                              "5.79.97",
                                              "5.79.97.178.100",
                                              "224.0.0.0",
                                              "239.255.255.255",
                                              "5.79.97.178",
                                              "127.0.0.1"});
  auto strings_view = cudf::strings_column_view(strings);
  {  // is_ip: 58 instructions
    std::string pattern =
      "^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)"
      "$";
    auto results = cudf::strings::matches_re(strings_view, pattern);
    cudf::test::fixed_width_column_wrapper<bool> expected(
      {true, true, false, false, false, false, true, true, true, true});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
    auto prog = cudf::strings::regex_program::create(pattern);
    results   = cudf::strings::matches_re(strings_view, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
  }
  {  // is_loopback: 72 instructions
    std::string pattern =
      "^127\\.([0-9]|[1-9][0-9]|1([0-9][0-9])|2([0-4][0-9]|5[0-5]))"
      "\\.([0-9]|[1-9][0-9]|1([0-9][0-9])|2([0-4][0-9]|5[0-5]))"
      "\\.([0-9]|[1-9][0-9]|1([0-9][0-9])|2([0-4][0-9]|5[0-5]))$";
    auto results = cudf::strings::matches_re(strings_view, pattern);
    cudf::test::fixed_width_column_wrapper<bool> expected(
      {false, false, false, false, false, false, false, false, false, true});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
    auto prog = cudf::strings::regex_program::create(pattern);
    results   = cudf::strings::matches_re(strings_view, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
  }
  {  // is_multicast: 79 instructions
    std::string pattern =
      "^(2(2[4-9]|3[0-9]))\\.([0-9]|[1-9][0-9]|1([0-9][0-9])|2([0-4][0-9]|5[0-5]))"
      "\\.([0-9]|[1-9][0-9]|1([0-9][0-9])|2([0-4][0-9]|5[0-5]))"
      "\\.([0-9]|[1-9][0-9]|1([0-9][0-9])|2([0-4][0-9]|5[0-5]))$";
    auto results = cudf::strings::matches_re(strings_view, pattern);
    cudf::test::fixed_width_column_wrapper<bool> expected(
      {false, false, false, false, false, false, true, true, false, false});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
    auto prog = cudf::strings::regex_program::create(pattern);
    results   = cudf::strings::matches_re(strings_view, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
  }
}

TEST_F(StringsContainsTests, OctalTest)
{
  cudf::test::strings_column_wrapper strings({"A3", "B", "CDA3EY", "", "99", "\a\t\r"});
  auto strings_view = cudf::strings_column_view(strings);
  auto expected     = cudf::test::fixed_width_column_wrapper<bool>({1, 0, 1, 0, 0, 0});

  auto pattern = std::string("\\101");
  auto results = cudf::strings::contains_re(strings_view, pattern);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  auto prog = cudf::strings::regex_program::create(pattern);
  results   = cudf::strings::contains_re(strings_view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  pattern = std::string("\\1013");
  results = cudf::strings::contains_re(strings_view, pattern);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  prog    = cudf::strings::regex_program::create(pattern);
  results = cudf::strings::contains_re(strings_view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  pattern = std::string("D*\\101\\063");
  results = cudf::strings::contains_re(strings_view, pattern);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  prog    = cudf::strings::regex_program::create(pattern);
  results = cudf::strings::contains_re(strings_view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  pattern  = std::string("\\719");
  results  = cudf::strings::contains_re(strings_view, pattern);
  expected = cudf::test::fixed_width_column_wrapper<bool>({0, 0, 0, 0, 1, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  prog    = cudf::strings::regex_program::create(pattern);
  results = cudf::strings::contains_re(strings_view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  pattern  = std::string("[\\7][\\11][\\15]");
  results  = cudf::strings::contains_re(strings_view, pattern);
  expected = cudf::test::fixed_width_column_wrapper<bool>({0, 0, 0, 0, 0, 1});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  prog    = cudf::strings::regex_program::create(pattern);
  results = cudf::strings::contains_re(strings_view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsContainsTests, HexTest)
{
  std::vector<char> ascii_chars(  // all possible matchable chars
    {thrust::make_counting_iterator<char>(0), thrust::make_counting_iterator<char>(127)});
  auto const count = static_cast<cudf::size_type>(ascii_chars.size());
  std::vector<cudf::offset_type> offsets(
    {thrust::make_counting_iterator<cudf::offset_type>(0),
     thrust::make_counting_iterator<cudf::offset_type>(0) + count + 1});
  auto d_chars   = cudf::detail::make_device_uvector_sync(ascii_chars, cudf::get_default_stream());
  auto d_offsets = cudf::detail::make_device_uvector_sync(offsets, cudf::get_default_stream());
  auto input     = cudf::make_strings_column(d_chars, d_offsets);

  auto strings_view = cudf::strings_column_view(input->view());
  for (auto ch : ascii_chars) {
    std::stringstream str;
    str << "\\x" << std::setfill('0') << std::setw(2) << std::hex << static_cast<int32_t>(ch);
    std::string pattern = str.str();

    auto results = cudf::strings::contains_re(strings_view, pattern);
    // only one element in the input should match ch
    auto true_dat = cudf::detail::make_counting_transform_iterator(
      0, [ch](auto idx) { return ch == static_cast<char>(idx); });
    cudf::test::fixed_width_column_wrapper<bool> expected(true_dat, true_dat + count);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
    auto prog = cudf::strings::regex_program::create(pattern);
    results   = cudf::strings::contains_re(strings_view, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

    // also test hex character appearing in character class brackets
    pattern = "[" + pattern + "]";
    results = cudf::strings::contains_re(strings_view, pattern);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
    prog    = cudf::strings::regex_program::create(pattern);
    results = cudf::strings::contains_re(strings_view, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(StringsContainsTests, EmbeddedNullCharacter)
{
  std::vector<std::string> data(10);
  std::generate(data.begin(), data.end(), [n = 0]() mutable {
    char first      = static_cast<char>('A' + n++);
    char raw_data[] = {first, '\0', 'B'};
    return std::string{raw_data, 3};
  });
  cudf::test::strings_column_wrapper input(data.begin(), data.end());
  auto strings_view = cudf::strings_column_view(input);

  auto pattern  = std::string("A");
  auto results  = cudf::strings::contains_re(strings_view, pattern);
  auto expected = cudf::test::fixed_width_column_wrapper<bool>({1, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  auto prog = cudf::strings::regex_program::create(pattern);
  results   = cudf::strings::contains_re(strings_view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  pattern  = std::string("B");
  results  = cudf::strings::contains_re(strings_view, pattern);
  expected = cudf::test::fixed_width_column_wrapper<bool>({1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  prog    = cudf::strings::regex_program::create(pattern);
  results = cudf::strings::contains_re(strings_view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  pattern  = std::string("J\\0B");
  results  = cudf::strings::contains_re(strings_view, pattern);
  expected = cudf::test::fixed_width_column_wrapper<bool>({0, 0, 0, 0, 0, 0, 0, 0, 0, 1});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  prog    = cudf::strings::regex_program::create(pattern);
  results = cudf::strings::contains_re(strings_view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  pattern  = std::string("[G-J][\\0]B");
  results  = cudf::strings::contains_re(strings_view, pattern);
  expected = cudf::test::fixed_width_column_wrapper<bool>({0, 0, 0, 0, 0, 0, 1, 1, 1, 1});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  prog    = cudf::strings::regex_program::create(pattern);
  results = cudf::strings::contains_re(strings_view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  pattern  = std::string("[A-D][\\x00]B");
  results  = cudf::strings::contains_re(strings_view, pattern);
  expected = cudf::test::fixed_width_column_wrapper<bool>({1, 1, 1, 1, 0, 0, 0, 0, 0, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  prog    = cudf::strings::regex_program::create(pattern);
  results = cudf::strings::contains_re(strings_view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsContainsTests, Errors)
{
  EXPECT_THROW(cudf::strings::regex_program::create("(3?)+"), cudf::logic_error);
  EXPECT_THROW(cudf::strings::regex_program::create("(?:3?)+"), cudf::logic_error);
  EXPECT_THROW(cudf::strings::regex_program::create("3?+"), cudf::logic_error);
  EXPECT_THROW(cudf::strings::regex_program::create("{3}a"), cudf::logic_error);

  EXPECT_THROW(cudf::strings::regex_program::create("aaaa{1234,5678}"), cudf::logic_error);
  EXPECT_THROW(cudf::strings::regex_program::create("aaaa{123,5678}"), cudf::logic_error);
}

TEST_F(StringsContainsTests, CountTest)
{
  std::vector<const char*> h_strings{
    "The quick brown @fox jumps ovér the", "lazy @dog", "1:2:3:4", "00:0:00", nullptr, ""};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(), h_strings.end(), cudf::test::iterators::nulls_from_nullptrs(h_strings));

  auto strings_view = cudf::strings_column_view(strings);
  {
    auto pattern = std::string("[tT]he");
    auto results = cudf::strings::count_re(strings_view, pattern);
    cudf::test::fixed_width_column_wrapper<int32_t> expected(
      {2, 0, 0, 0, 0, 0}, cudf::test::iterators::nulls_from_nullptrs(h_strings));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
    auto prog = cudf::strings::regex_program::create(pattern);
    results   = cudf::strings::count_re(strings_view, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto pattern = std::string("@\\w+");
    auto results = cudf::strings::count_re(strings_view, pattern);
    cudf::test::fixed_width_column_wrapper<int32_t> expected(
      {1, 1, 0, 0, 0, 0}, cudf::test::iterators::nulls_from_nullptrs(h_strings));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
    auto prog = cudf::strings::regex_program::create(pattern);
    results   = cudf::strings::count_re(strings_view, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto pattern = std::string("\\d+:\\d+");
    auto results = cudf::strings::count_re(strings_view, pattern);
    cudf::test::fixed_width_column_wrapper<int32_t> expected(
      {0, 0, 2, 1, 0, 0}, cudf::test::iterators::nulls_from_nullptrs(h_strings));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
    auto prog = cudf::strings::regex_program::create(pattern);
    results   = cudf::strings::count_re(strings_view, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(StringsContainsTests, FixedQuantifier)
{
  auto input = cudf::test::strings_column_wrapper({"a", "aa", "aaa", "aaaa", "aaaaa", "aaaaaa"});
  auto sv    = cudf::strings_column_view(input);

  {
    // exact match
    auto pattern = std::string("a{3}");
    auto results = cudf::strings::count_re(sv, pattern);
    cudf::test::fixed_width_column_wrapper<int32_t> expected({0, 0, 1, 1, 1, 2});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
    auto prog = cudf::strings::regex_program::create(pattern);
    results   = cudf::strings::count_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    // range match (greedy quantifier)
    auto pattern = std::string("a{3,5}");
    auto results = cudf::strings::count_re(sv, pattern);
    cudf::test::fixed_width_column_wrapper<int32_t> expected({0, 0, 1, 1, 1, 1});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
    auto prog = cudf::strings::regex_program::create(pattern);
    results   = cudf::strings::count_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    // minimum match (greedy quantifier)
    auto pattern = std::string("a{2,}");
    auto results = cudf::strings::count_re(sv, pattern);
    cudf::test::fixed_width_column_wrapper<int32_t> expected({0, 1, 1, 1, 1, 1});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
    auto prog = cudf::strings::regex_program::create(pattern);
    results   = cudf::strings::count_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    // range match (lazy quantifier)
    auto pattern = std::string("a{2,4}?");
    auto results = cudf::strings::count_re(sv, pattern);
    cudf::test::fixed_width_column_wrapper<int32_t> expected({0, 1, 1, 2, 2, 3});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
    auto prog = cudf::strings::regex_program::create(pattern);
    results   = cudf::strings::count_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    // minimum match (lazy quantifier)
    auto pattern = std::string("a{1,}?");
    auto results = cudf::strings::count_re(sv, pattern);
    cudf::test::fixed_width_column_wrapper<int32_t> expected({1, 2, 3, 4, 5, 6});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
    auto prog = cudf::strings::regex_program::create(pattern);
    results   = cudf::strings::count_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    // zero match
    auto pattern = std::string("aaaa{0}");
    auto results = cudf::strings::count_re(sv, pattern);
    cudf::test::fixed_width_column_wrapper<int32_t> expected({0, 0, 1, 1, 1, 2});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
    auto prog = cudf::strings::regex_program::create(pattern);
    results   = cudf::strings::count_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    // poorly formed
    auto pattern = std::string("aaaa{n,m}");
    auto results = cudf::strings::count_re(sv, pattern);
    cudf::test::fixed_width_column_wrapper<int32_t> expected({0, 0, 0, 0, 0, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
    auto prog = cudf::strings::regex_program::create(pattern);
    results   = cudf::strings::count_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(StringsContainsTests, QuantifierErrors)
{
  EXPECT_THROW(cudf::strings::regex_program::create("^+"), cudf::logic_error);
  EXPECT_THROW(cudf::strings::regex_program::create("$+"), cudf::logic_error);
  EXPECT_THROW(cudf::strings::regex_program::create("(^)+"), cudf::logic_error);
  EXPECT_THROW(cudf::strings::regex_program::create("($)+"), cudf::logic_error);
  EXPECT_THROW(cudf::strings::regex_program::create("\\A+"), cudf::logic_error);
  EXPECT_THROW(cudf::strings::regex_program::create("\\Z+"), cudf::logic_error);
  EXPECT_THROW(cudf::strings::regex_program::create("(\\A)+"), cudf::logic_error);
  EXPECT_THROW(cudf::strings::regex_program::create("(\\Z)+"), cudf::logic_error);

  EXPECT_THROW(cudf::strings::regex_program::create("(^($))+"), cudf::logic_error);
  EXPECT_NO_THROW(cudf::strings::regex_program::create("(^a($))+"));
  EXPECT_NO_THROW(cudf::strings::regex_program::create("(^(a$))+"));
}

TEST_F(StringsContainsTests, OverlappedClasses)
{
  auto input = cudf::test::strings_column_wrapper({"abcdefg", "defghí", "", "éééééé", "ghijkl"});
  auto sv = cudf::strings_column_view(input);

  {
    auto pattern = std::string("[e-gb-da-c]");
    auto results = cudf::strings::count_re(sv, pattern);
    cudf::test::fixed_width_column_wrapper<int32_t> expected({7, 4, 0, 0, 1});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
    auto prog = cudf::strings::regex_program::create(pattern);
    results   = cudf::strings::count_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto pattern = std::string("[á-éê-ú]");
    auto results = cudf::strings::count_re(sv, pattern);
    cudf::test::fixed_width_column_wrapper<int32_t> expected({0, 1, 0, 6, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
    auto prog = cudf::strings::regex_program::create(pattern);
    results   = cudf::strings::count_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(StringsContainsTests, NegatedClasses)
{
  auto input = cudf::test::strings_column_wrapper({"abcdefg", "def\tghí", "", "éeé\néeé", "ABC"});
  auto sv    = cudf::strings_column_view(input);

  {
    auto pattern = std::string("[^a-f]");
    auto results = cudf::strings::count_re(sv, pattern);
    cudf::test::fixed_width_column_wrapper<int32_t> expected({1, 4, 0, 5, 3});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
    auto prog = cudf::strings::regex_program::create(pattern);
    results   = cudf::strings::count_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto pattern = std::string("[^a-eá-é]");
    auto results = cudf::strings::count_re(sv, pattern);
    cudf::test::fixed_width_column_wrapper<int32_t> expected({2, 5, 0, 1, 3});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
    auto prog = cudf::strings::regex_program::create(pattern);
    results   = cudf::strings::count_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(StringsContainsTests, IncompleteClassesRange)
{
  auto input = cudf::test::strings_column_wrapper({"abc-def", "---", "", "ghijkl", "-wxyz-"});
  auto sv    = cudf::strings_column_view(input);

  {
    cudf::test::fixed_width_column_wrapper<bool> expected({1, 0, 0, 1, 1});
    auto results = cudf::strings::contains_re(sv, "[a-z]");
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

    auto prog = cudf::strings::regex_program::create("[a-m-z]");  // same as [a-z]
    results   = cudf::strings::contains_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    cudf::test::fixed_width_column_wrapper<bool> expected({1, 1, 0, 1, 1});
    auto results = cudf::strings::contains_re(sv, "[g-]");
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

    auto prog = cudf::strings::regex_program::create("[-k]");
    results   = cudf::strings::contains_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    cudf::test::fixed_width_column_wrapper<bool> expected({1, 1, 0, 0, 1});
    auto results = cudf::strings::contains_re(sv, "[-]");
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
    results = cudf::strings::contains_re(sv, "[+--]");
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

    auto prog = cudf::strings::regex_program::create("[a-c-]");
    results   = cudf::strings::contains_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
    prog    = cudf::strings::regex_program::create("[-d-f]");
    results = cudf::strings::contains_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(StringsContainsTests, MultiLine)
{
  auto input = cudf::test::strings_column_wrapper(
    {"abé\nfff\nabé", "fff\nabé\nlll", "abé", "", "abé\n", "abe\nabé\n"});
  auto view = cudf::strings_column_view(input);

  auto pattern = std::string("^abé$");
  auto prog    = cudf::strings::regex_program::create(pattern);
  auto prog_ml =
    cudf::strings::regex_program::create(pattern, cudf::strings::regex_flags::MULTILINE);

  auto results = cudf::strings::contains_re(view, pattern, cudf::strings::regex_flags::MULTILINE);
  auto expected_contains = cudf::test::fixed_width_column_wrapper<bool>({1, 1, 1, 0, 1, 1});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_contains);
  results = cudf::strings::contains_re(view, *prog_ml);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_contains);
  results           = cudf::strings::contains_re(view, pattern);
  expected_contains = cudf::test::fixed_width_column_wrapper<bool>({0, 0, 1, 0, 1, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_contains);
  results = cudf::strings::contains_re(view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_contains);

  results = cudf::strings::matches_re(view, pattern, cudf::strings::regex_flags::MULTILINE);
  auto expected_matches = cudf::test::fixed_width_column_wrapper<bool>({1, 0, 1, 0, 1, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_matches);
  results = cudf::strings::matches_re(view, *prog_ml);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_matches);
  results          = cudf::strings::matches_re(view, pattern);
  expected_matches = cudf::test::fixed_width_column_wrapper<bool>({0, 0, 1, 0, 1, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_matches);
  results = cudf::strings::matches_re(view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_matches);

  results = cudf::strings::count_re(view, pattern, cudf::strings::regex_flags::MULTILINE);
  auto expected_count = cudf::test::fixed_width_column_wrapper<int32_t>({2, 1, 1, 0, 1, 1});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_count);
  results = cudf::strings::count_re(view, *prog_ml);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_count);
  results        = cudf::strings::count_re(view, pattern);
  expected_count = cudf::test::fixed_width_column_wrapper<int32_t>({0, 0, 1, 0, 1, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_count);
  results = cudf::strings::count_re(view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_count);
}

TEST_F(StringsContainsTests, EndOfString)
{
  auto input = cudf::test::strings_column_wrapper(
    {"abé\nfff\nabé", "fff\nabé\nlll", "abé", "", "abé\n", "abe\nabé\n"});
  auto view = cudf::strings_column_view(input);

  auto pattern = std::string("\\Aabé\\Z");
  auto prog    = cudf::strings::regex_program::create(pattern);
  auto prog_ml =
    cudf::strings::regex_program::create(pattern, cudf::strings::regex_flags::MULTILINE);

  auto results  = cudf::strings::contains_re(view, *prog);
  auto expected = cudf::test::fixed_width_column_wrapper<bool>({0, 0, 1, 0, 0, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
  results = cudf::strings::contains_re(view, *prog_ml);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);

  results = cudf::strings::matches_re(view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
  results = cudf::strings::matches_re(view, *prog_ml);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);

  results             = cudf::strings::count_re(view, *prog);
  auto expected_count = cudf::test::fixed_width_column_wrapper<int32_t>({0, 0, 1, 0, 0, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_count);
  results = cudf::strings::count_re(view, *prog_ml);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_count);
}

TEST_F(StringsContainsTests, DotAll)
{
  auto input = cudf::test::strings_column_wrapper({"abc\nfa\nef", "fff\nabbc\nfff", "abcdef", ""});
  auto view  = cudf::strings_column_view(input);

  auto pattern = std::string("a.*f");
  auto prog    = cudf::strings::regex_program::create(pattern);
  auto prog_dotall =
    cudf::strings::regex_program::create(pattern, cudf::strings::regex_flags::DOTALL);

  auto results = cudf::strings::contains_re(view, pattern, cudf::strings::regex_flags::DOTALL);
  auto expected_contains = cudf::test::fixed_width_column_wrapper<bool>({1, 1, 1, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_contains);
  results = cudf::strings::contains_re(view, *prog_dotall);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_contains);
  results           = cudf::strings::contains_re(view, pattern);
  expected_contains = cudf::test::fixed_width_column_wrapper<bool>({0, 0, 1, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_contains);
  results = cudf::strings::contains_re(view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_contains);

  results = cudf::strings::matches_re(view, pattern, cudf::strings::regex_flags::DOTALL);
  auto expected_matches = cudf::test::fixed_width_column_wrapper<bool>({1, 0, 1, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_matches);
  results = cudf::strings::matches_re(view, *prog_dotall);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_matches);
  results          = cudf::strings::matches_re(view, pattern);
  expected_matches = cudf::test::fixed_width_column_wrapper<bool>({0, 0, 1, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_matches);
  results = cudf::strings::matches_re(view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_matches);

  pattern     = std::string("a.*?f");
  prog        = cudf::strings::regex_program::create(pattern);
  prog_dotall = cudf::strings::regex_program::create(pattern, cudf::strings::regex_flags::DOTALL);

  results             = cudf::strings::count_re(view, pattern, cudf::strings::regex_flags::DOTALL);
  auto expected_count = cudf::test::fixed_width_column_wrapper<int32_t>({2, 1, 1, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_count);
  results = cudf::strings::count_re(view, *prog_dotall);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_count);
  results        = cudf::strings::count_re(view, pattern);
  expected_count = cudf::test::fixed_width_column_wrapper<int32_t>({0, 0, 1, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_count);
  results = cudf::strings::count_re(view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_count);

  auto both_flags = static_cast<cudf::strings::regex_flags>(cudf::strings::regex_flags::DOTALL |
                                                            cudf::strings::regex_flags::MULTILINE);
  results         = cudf::strings::count_re(view, pattern, both_flags);
  expected_count  = cudf::test::fixed_width_column_wrapper<int32_t>({2, 1, 1, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_count);
  auto prog_both = cudf::strings::regex_program::create(pattern, both_flags);
  results        = cudf::strings::count_re(view, *prog_both);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_count);
}

TEST_F(StringsContainsTests, ASCII)
{
  auto input = cudf::test::strings_column_wrapper({"abc \t\f\r 12", "áé 　❽❽", "aZ ❽4", "XYZ　8"});
  auto view = cudf::strings_column_view(input);

  std::string patterns[] = {"\\w+[\\s]+\\d+",
                            "[^\\W]+\\s+[^\\D]+",
                            "[\\w]+[^\\S]+[\\d]+",
                            "[\\w]+\\s+[\\d]+",
                            "\\w+\\s+\\d+"};

  for (auto ptn : patterns) {
    auto results = cudf::strings::contains_re(view, ptn, cudf::strings::regex_flags::ASCII);
    auto expected_contains = cudf::test::fixed_width_column_wrapper<bool>({1, 0, 0, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_contains);
    auto prog = cudf::strings::regex_program::create(ptn, cudf::strings::regex_flags::ASCII);
    results   = cudf::strings::contains_re(view, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_contains);

    results           = cudf::strings::contains_re(view, ptn);
    expected_contains = cudf::test::fixed_width_column_wrapper<bool>({1, 1, 1, 1});
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_contains);
    prog    = cudf::strings::regex_program::create(ptn);
    results = cudf::strings::contains_re(view, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_contains);
  }
}

TEST_F(StringsContainsTests, MediumRegex)
{
  // This results in 95 regex instructions and falls in the 'medium' range.
  std::string medium_regex =
    "hello @abc @def world The quick brown @fox jumps over the lazy @dog hello "
    "http://www.world.com";

  std::vector<const char*> h_strings{
    "hello @abc @def world The quick brown @fox jumps over the lazy @dog hello "
    "http://www.world.com thats all",
    "1234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234"
    "5678901234567890",
    "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnop"
    "qrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz"};
  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end());

  auto strings_view = cudf::strings_column_view(strings);
  {
    auto results      = cudf::strings::contains_re(strings_view, medium_regex);
    bool h_expected[] = {true, false, false};
    cudf::test::fixed_width_column_wrapper<bool> expected(h_expected,
                                                          h_expected + h_strings.size());
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
  }
  {
    auto results      = cudf::strings::matches_re(strings_view, medium_regex);
    bool h_expected[] = {true, false, false};
    cudf::test::fixed_width_column_wrapper<bool> expected(h_expected,
                                                          h_expected + h_strings.size());
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
  }
  {
    auto results         = cudf::strings::count_re(strings_view, medium_regex);
    int32_t h_expected[] = {1, 0, 0};
    cudf::test::fixed_width_column_wrapper<int32_t> expected(h_expected,
                                                             h_expected + h_strings.size());
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
  }
}

TEST_F(StringsContainsTests, LargeRegex)
{
  // This results in 115 regex instructions and falls in the 'large' range.
  std::string large_regex =
    "hello @abc @def world The quick brown @fox jumps over the lazy @dog hello "
    "http://www.world.com I'm here @home zzzz";

  std::vector<const char*> h_strings{
    "hello @abc @def world The quick brown @fox jumps over the lazy @dog hello "
    "http://www.world.com I'm here @home zzzz",
    "1234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234"
    "5678901234567890",
    "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnop"
    "qrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz"};
  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end());

  auto strings_view = cudf::strings_column_view(strings);
  {
    auto results      = cudf::strings::contains_re(strings_view, large_regex);
    bool h_expected[] = {true, false, false};
    cudf::test::fixed_width_column_wrapper<bool> expected(h_expected,
                                                          h_expected + h_strings.size());
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
  }
  {
    auto results      = cudf::strings::matches_re(strings_view, large_regex);
    bool h_expected[] = {true, false, false};
    cudf::test::fixed_width_column_wrapper<bool> expected(h_expected,
                                                          h_expected + h_strings.size());
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
  }
  {
    auto results         = cudf::strings::count_re(strings_view, large_regex);
    int32_t h_expected[] = {1, 0, 0};
    cudf::test::fixed_width_column_wrapper<int32_t> expected(h_expected,
                                                             h_expected + h_strings.size());
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
  }
}

TEST_F(StringsContainsTests, ExtraLargeRegex)
{
  // This results in 321 regex instructions which is above the 'large' range.
  std::string data(320, '0');
  cudf::test::strings_column_wrapper strings({data, data, data, data, data, "00"});
  std::string pattern = data;

  auto strings_view = cudf::strings_column_view(strings);
  {
    auto results = cudf::strings::contains_re(strings_view, pattern);
    cudf::test::fixed_width_column_wrapper<bool> expected({true, true, true, true, true, false});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto results = cudf::strings::matches_re(strings_view, pattern);
    cudf::test::fixed_width_column_wrapper<bool> expected({true, true, true, true, true, false});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto results = cudf::strings::count_re(strings_view, pattern);
    cudf::test::fixed_width_column_wrapper<int32_t> expected({1, 1, 1, 1, 1, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}
