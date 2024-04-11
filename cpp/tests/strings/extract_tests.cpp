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
#include <cudf_test/table_utilities.hpp>

#include <cudf/detail/iterator.cuh>
#include <cudf/strings/extract.hpp>
#include <cudf/strings/regex/regex_program.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_view.hpp>

#include <thrust/iterator/transform_iterator.h>

#include <vector>

struct StringsExtractTests : public cudf::test::BaseFixture {};

TEST_F(StringsExtractTests, ExtractTest)
{
  std::vector<char const*> h_strings{
    "First Last", "Joe Schmoe", "John Smith", "Jane Smith", "Beyonce", "Sting", nullptr, ""};

  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  auto strings_view = cudf::strings_column_view(strings);

  std::vector<char const*> h_expecteds{"First",
                                       "Joe",
                                       "John",
                                       "Jane",
                                       nullptr,
                                       nullptr,
                                       nullptr,
                                       nullptr,
                                       "Last",
                                       "Schmoe",
                                       "Smith",
                                       "Smith",
                                       nullptr,
                                       nullptr,
                                       nullptr,
                                       nullptr};

  std::string pattern = "(\\w+) (\\w+)";

  cudf::test::strings_column_wrapper expected1(
    h_expecteds.data(),
    h_expecteds.data() + h_strings.size(),
    thrust::make_transform_iterator(h_expecteds.begin(), [](auto str) { return str != nullptr; }));
  cudf::test::strings_column_wrapper expected2(
    h_expecteds.data() + h_strings.size(),
    h_expecteds.data() + h_expecteds.size(),
    thrust::make_transform_iterator(h_expecteds.data() + h_strings.size(),
                                    [](auto str) { return str != nullptr; }));
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(expected1.release());
  columns.push_back(expected2.release());
  cudf::table expected(std::move(columns));

  auto prog    = cudf::strings::regex_program::create(pattern);
  auto results = cudf::strings::extract(strings_view, *prog);
  CUDF_TEST_EXPECT_TABLES_EQUAL(*results, expected);
}

TEST_F(StringsExtractTests, ExtractDomainTest)
{
  cudf::test::strings_column_wrapper strings({"http://www.google.com",
                                              "gmail.com",
                                              "github.com",
                                              "https://pandas.pydata.org",
                                              "http://www.worldbank.org.kg/",
                                              "waiterrant.blogspot.com",
                                              "http://forums.news.cnn.com.ac/",
                                              "http://forums.news.cnn.ac/",
                                              "ftp://b.cnn.com/",
                                              "a.news.uk",
                                              "a.news.co.uk",
                                              "https://a.news.co.uk",
                                              "107-193-100-2.lightspeed.cicril.sbcglobal.net",
                                              "a23-44-13-2.deploy.static.akamaitechnologies.com"});
  auto strings_view = cudf::strings_column_view(strings);

  std::string pattern = "([\\w]+[\\.].*[^/]|[\\-\\w]+[\\.].*[^/])";

  cudf::test::strings_column_wrapper expected1({
    "www.google.com",
    "gmail.com",
    "github.com",
    "pandas.pydata.org",
    "www.worldbank.org.kg",
    "waiterrant.blogspot.com",
    "forums.news.cnn.com.ac",
    "forums.news.cnn.ac",
    "b.cnn.com",
    "a.news.uk",
    "a.news.co.uk",
    "a.news.co.uk",
    "107-193-100-2.lightspeed.cicril.sbcglobal.net",
    "a23-44-13-2.deploy.static.akamaitechnologies.com",
  });
  cudf::table_view expected{{expected1}};

  auto prog    = cudf::strings::regex_program::create(pattern);
  auto results = cudf::strings::extract(strings_view, *prog);
  CUDF_TEST_EXPECT_TABLES_EQUAL(*results, expected);
}

TEST_F(StringsExtractTests, ExtractEventTest)
{
  std::vector<std::string> patterns({"(^[0-9]+\\.?[0-9]*),",
                                     "search_name=\"([0-9A-Za-z\\s\\-\\(\\)]+)",
                                     "message.ip=\"([\\w\\.]+)",
                                     "message.hostname=\"([\\w\\.]+)",
                                     "message.user_name=\"([\\w\\.\\@]+)",
                                     "message\\.description=\"([\\w\\.\\s]+)"});

  cudf::test::strings_column_wrapper strings(
    {"15162388.26, search_name=\"Test Search Name\", orig_time=\"1516238826\", "
     "info_max_time=\"1566346500.000000000\", info_min_time=\"1566345300.000000000\", "
     "info_search_time=\"1566305689.361160000\", message.description=\"Test Message Description\", "
     "message.hostname=\"msg.test.hostname\", message.ip=\"100.100.100.123\", "
     "message.user_name=\"user@test.com\", severity=\"info\", urgency=\"medium\"'"});
  auto strings_view = cudf::strings_column_view(strings);

  std::vector<std::string> expecteds({"15162388.26",
                                      "Test Search Name",
                                      "100.100.100.123",
                                      "msg.test.hostname",
                                      "user@test.com",
                                      "Test Message Description"});

  for (std::size_t idx = 0; idx < patterns.size(); ++idx) {
    auto pattern = patterns[idx];
    cudf::test::strings_column_wrapper expected({expecteds[idx]});
    auto prog    = cudf::strings::regex_program::create(pattern);
    auto results = cudf::strings::extract(strings_view, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view().column(0), expected);
  }
}

TEST_F(StringsExtractTests, MultiLine)
{
  auto input = cudf::test::strings_column_wrapper(
    {"abc\nfff\nabc", "fff\nabc\nlll", "abc", "", "abc\n", "abé\nabc\n"});
  auto view = cudf::strings_column_view(input);

  auto pattern = std::string("(^[a-c]+$)");
  cudf::test::strings_column_wrapper expected_multiline({"abc", "abc", "abc", "", "abc", "abc"},
                                                        {1, 1, 1, 0, 1, 1});
  auto expected = cudf::table_view{{expected_multiline}};
  auto prog = cudf::strings::regex_program::create(pattern, cudf::strings::regex_flags::MULTILINE);
  auto results = cudf::strings::extract(view, *prog);
  CUDF_TEST_EXPECT_TABLES_EQUAL(*results, expected);

  pattern = std::string("^([a-c]+)$");
  cudf::test::strings_column_wrapper expected_default({"", "", "abc", "", "abc", ""},
                                                      {0, 0, 1, 0, 1, 0});
  expected = cudf::table_view{{expected_default}};
  prog     = cudf::strings::regex_program::create(pattern);
  results  = cudf::strings::extract(view, *prog);
  CUDF_TEST_EXPECT_TABLES_EQUAL(*results, expected);
}

TEST_F(StringsExtractTests, DotAll)
{
  auto input = cudf::test::strings_column_wrapper({"abc\nfa\nef", "fff\nabbc\nfff", "abcdef", ""});
  auto view  = cudf::strings_column_view(input);

  auto pattern = std::string("(a.*f)");
  cudf::test::strings_column_wrapper expected_dotall({"abc\nfa\nef", "abbc\nfff", "abcdef", ""},
                                                     {1, 1, 1, 0});
  auto expected = cudf::table_view{{expected_dotall}};
  auto prog     = cudf::strings::regex_program::create(pattern, cudf::strings::regex_flags::DOTALL);
  auto results  = cudf::strings::extract(view, *prog);
  CUDF_TEST_EXPECT_TABLES_EQUAL(*results, expected);

  cudf::test::strings_column_wrapper expected_default({"", "", "abcdef", ""}, {0, 0, 1, 0});
  expected = cudf::table_view{{expected_default}};
  prog     = cudf::strings::regex_program::create(pattern);
  results  = cudf::strings::extract(view, *prog);
  CUDF_TEST_EXPECT_TABLES_EQUAL(*results, expected);
}

TEST_F(StringsExtractTests, EmptyExtractTest)
{
  std::vector<char const*> h_strings{nullptr, "AAA", "AAA_A", "AAA_AAA_", "A__", ""};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  auto strings_view = cudf::strings_column_view(strings);

  auto pattern = std::string("([^_]*)\\Z");

  std::vector<char const*> h_expected{nullptr, "AAA", "A", "", "", ""};
  cudf::test::strings_column_wrapper expected(
    h_expected.data(),
    h_expected.data() + h_strings.size(),
    thrust::make_transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(expected.release());
  cudf::table table_expected(std::move(columns));
  auto prog    = cudf::strings::regex_program::create(pattern);
  auto results = cudf::strings::extract(strings_view, *prog);
  CUDF_TEST_EXPECT_TABLES_EQUAL(*results, table_expected);
}

TEST_F(StringsExtractTests, ExtractAllTest)
{
  std::vector<char const*> h_input(
    {"123 banana 7 eleven", "41 apple", "6 péar 0 pair", nullptr, "", "bees", "4 paré"});
  auto validity =
    thrust::make_transform_iterator(h_input.begin(), [](auto str) { return str != nullptr; });
  cudf::test::strings_column_wrapper input(h_input.begin(), h_input.end(), validity);
  auto sv = cudf::strings_column_view(input);

  auto pattern = std::string("(\\d+) (\\w+)");

  bool valids[] = {true, true, true, false, false, false, true};
  using LCW     = cudf::test::lists_column_wrapper<cudf::string_view>;
  LCW expected({LCW{"123", "banana", "7", "eleven"},
                LCW{"41", "apple"},
                LCW{"6", "péar", "0", "pair"},
                LCW{},
                LCW{},
                LCW{},
                LCW{"4", "paré"}},
               valids);
  auto prog    = cudf::strings::regex_program::create(pattern);
  auto results = cudf::strings::extract_all_record(sv, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
}

TEST_F(StringsExtractTests, Errors)
{
  cudf::test::strings_column_wrapper input({"this column intentionally left blank"});
  auto sv = cudf::strings_column_view(input);

  auto pattern = std::string("\\w+");
  auto prog    = cudf::strings::regex_program::create(pattern);

  EXPECT_THROW(cudf::strings::extract(sv, *prog), cudf::logic_error);
  EXPECT_THROW(cudf::strings::extract_all_record(sv, *prog), cudf::logic_error);
}

TEST_F(StringsExtractTests, MediumRegex)
{
  // This results in 95 regex instructions and falls in the 'medium' range.
  std::string medium_regex =
    "hello @abc @def (world) The quick brown @fox jumps over the lazy @dog hello "
    "http://www.world.com";
  auto prog = cudf::strings::regex_program::create(medium_regex);

  std::vector<char const*> h_strings{
    "hello @abc @def world The quick brown @fox jumps over the lazy @dog hello "
    "http://www.world.com thats all",
    "1234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234"
    "5678901234567890",
    "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnop"
    "qrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));

  auto strings_view = cudf::strings_column_view(strings);
  auto results      = cudf::strings::extract(strings_view, *prog);
  std::vector<char const*> h_expected{"world", nullptr, nullptr};
  cudf::test::strings_column_wrapper expected(
    h_expected.begin(),
    h_expected.end(),
    thrust::make_transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->get_column(0), expected);
}

TEST_F(StringsExtractTests, LargeRegex)
{
  // This results in 115 regex instructions and falls in the 'large' range.
  std::string large_regex =
    "hello @abc @def world The (quick) brown @fox jumps over the lazy @dog hello "
    "http://www.world.com I'm here @home zzzz";
  auto prog = cudf::strings::regex_program::create(large_regex);

  std::vector<char const*> h_strings{
    "hello @abc @def world The quick brown @fox jumps over the lazy @dog hello "
    "http://www.world.com I'm here @home zzzz",
    "1234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234"
    "5678901234567890",
    "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnop"
    "qrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));

  auto strings_view = cudf::strings_column_view(strings);
  auto results      = cudf::strings::extract(strings_view, *prog);
  std::vector<char const*> h_expected{"quick", nullptr, nullptr};
  cudf::test::strings_column_wrapper expected(
    h_expected.begin(),
    h_expected.end(),
    thrust::make_transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->get_column(0), expected);
}
