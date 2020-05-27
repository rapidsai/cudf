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

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <tests/strings/utilities.h>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/type_lists.hpp>

#include <vector>

struct StringsFindTest : public cudf::test::BaseFixture {
};

TEST_F(StringsFindTest, Find)
{
  std::vector<const char*> h_strings{"Héllo", "thesé", nullptr, "lease", "tést strings", ""};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  auto strings_view = cudf::strings_column_view(strings);

  {
    cudf::test::fixed_width_column_wrapper<int32_t> expected({1, 4, -1, -1, 1, -1},
                                                             {1, 1, 0, 1, 1, 1});
    auto results = cudf::strings::find(strings_view, cudf::string_scalar("é"));
    cudf::test::expect_columns_equal(*results, expected);
  }
  {
    cudf::test::fixed_width_column_wrapper<int32_t> expected({3, -1, -1, 0, -1, -1},
                                                             {1, 1, 0, 1, 1, 1});
    auto results = cudf::strings::rfind(strings_view, cudf::string_scalar("l"));
    cudf::test::expect_columns_equal(*results, expected);
  }
  {
    cudf::test::fixed_width_column_wrapper<bool> expected({0, 1, 0, 1, 0, 0}, {1, 1, 0, 1, 1, 1});
    auto results = cudf::strings::contains(strings_view, cudf::string_scalar("e"));
    cudf::test::expect_columns_equal(*results, expected);
  }
  {
    cudf::test::fixed_width_column_wrapper<bool> expected({0, 1, 0, 0, 1, 0}, {1, 1, 0, 1, 1, 1});
    auto results = cudf::strings::starts_with(strings_view, cudf::string_scalar("t"));
    cudf::test::expect_columns_equal(*results, expected);
  }
  {
    cudf::test::fixed_width_column_wrapper<bool> expected({0, 0, 0, 1, 0, 0}, {1, 1, 0, 1, 1, 1});
    auto results = cudf::strings::ends_with(strings_view, cudf::string_scalar("se"));
    cudf::test::expect_columns_equal(*results, expected);
  }
  {
    cudf::test::fixed_width_column_wrapper<bool> expected({0, 1, 0, 0, 0, 0}, {1, 1, 0, 1, 1, 1});
    auto results = cudf::strings::starts_with(strings_view, cudf::string_scalar("thesé"));
    cudf::test::expect_columns_equal(*results, expected);
  }
  {
    cudf::test::fixed_width_column_wrapper<bool> expected({0, 1, 0, 0, 0, 0}, {1, 1, 0, 1, 1, 1});
    auto results = cudf::strings::ends_with(strings_view, cudf::string_scalar("thesé"));
    cudf::test::expect_columns_equal(*results, expected);
  }
  {
    cudf::test::fixed_width_column_wrapper<bool> expected({1, 1, 1, 1, 1, 1}, {1, 1, 0, 1, 1, 1});
    auto results = cudf::strings::contains(strings_view, cudf::string_scalar(""));
    cudf::test::expect_columns_equal(*results, expected);
    results = cudf::strings::starts_with(strings_view, cudf::string_scalar(""));
    cudf::test::expect_columns_equal(*results, expected);
    results = cudf::strings::ends_with(strings_view, cudf::string_scalar(""));
    cudf::test::expect_columns_equal(*results, expected);
  }
  {
    cudf::test::fixed_width_column_wrapper<int32_t> expected({0, 0, 0, 0, 0, 0},
                                                             {1, 1, 0, 1, 1, 1});
    auto results = cudf::strings::find(strings_view, cudf::string_scalar(""));
    cudf::test::expect_columns_equal(*results, expected);
  }
  {
    cudf::test::fixed_width_column_wrapper<int32_t> expected({5, 5, 0, 5, 12, 0},
                                                             {1, 1, 0, 1, 1, 1});
    auto results = cudf::strings::rfind(strings_view, cudf::string_scalar(""));
    cudf::test::expect_columns_equal(*results, expected);
  }
}

TEST_F(StringsFindTest, ZeroSizeStringsColumn)
{
  cudf::column_view zero_size_strings_column(cudf::data_type{cudf::STRING}, 0, nullptr, nullptr, 0);
  auto strings_view = cudf::strings_column_view(zero_size_strings_column);
  auto results      = cudf::strings::find(strings_view, cudf::string_scalar("é"));
  EXPECT_EQ(results->size(), 0);
  results = cudf::strings::rfind(strings_view, cudf::string_scalar("é"));
  EXPECT_EQ(results->size(), 0);
  results = cudf::strings::contains(strings_view, cudf::string_scalar("é"));
  EXPECT_EQ(results->size(), 0);
  results = cudf::strings::starts_with(strings_view, cudf::string_scalar("é"));
  EXPECT_EQ(results->size(), 0);
  results = cudf::strings::ends_with(strings_view, cudf::string_scalar("é"));
  EXPECT_EQ(results->size(), 0);
}

TEST_F(StringsFindTest, AllEmpty)
{
  std::vector<std::string> h_strings{"", "", "", "", ""};
  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end());

  std::vector<int32_t> h_expected32(h_strings.size(), -1);
  cudf::test::fixed_width_column_wrapper<int32_t> expected32(h_expected32.begin(),
                                                             h_expected32.end());

  std::vector<bool> h_expected8(h_strings.size(), 0);
  cudf::test::fixed_width_column_wrapper<bool> expected8(h_expected8.begin(), h_expected8.end());

  auto strings_view = cudf::strings_column_view(strings);
  auto results      = cudf::strings::find(strings_view, cudf::string_scalar("e"));
  cudf::test::expect_columns_equal(*results, expected32);
  results = cudf::strings::rfind(strings_view, cudf::string_scalar("e"));
  cudf::test::expect_columns_equal(*results, expected32);
  results = cudf::strings::contains(strings_view, cudf::string_scalar("e"));
  cudf::test::expect_columns_equal(*results, expected8);
  results = cudf::strings::starts_with(strings_view, cudf::string_scalar("e"));
  cudf::test::expect_columns_equal(*results, expected8);
  results = cudf::strings::ends_with(strings_view, cudf::string_scalar("e"));
  cudf::test::expect_columns_equal(*results, expected8);
}

TEST_F(StringsFindTest, AllNull)
{
  std::vector<const char*> h_strings{nullptr, nullptr, nullptr, nullptr};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));

  std::vector<int32_t> h_expected32(h_strings.size(), -1);
  cudf::test::fixed_width_column_wrapper<int32_t> expected32(
    h_expected32.begin(),
    h_expected32.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));

  std::vector<bool> h_expected8(h_strings.size(), -1);
  cudf::test::fixed_width_column_wrapper<bool> expected8(
    h_expected8.begin(),
    h_expected8.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));

  auto strings_view = cudf::strings_column_view(strings);
  auto results      = cudf::strings::find(strings_view, cudf::string_scalar("e"));
  cudf::test::expect_columns_equal(*results, expected32);
  results = cudf::strings::rfind(strings_view, cudf::string_scalar("e"));
  cudf::test::expect_columns_equal(*results, expected32);
  results = cudf::strings::contains(strings_view, cudf::string_scalar("e"));
  cudf::test::expect_columns_equal(*results, expected8);
  results = cudf::strings::starts_with(strings_view, cudf::string_scalar("e"));
  cudf::test::expect_columns_equal(*results, expected8);
  results = cudf::strings::ends_with(strings_view, cudf::string_scalar("e"));
  cudf::test::expect_columns_equal(*results, expected8);
}

class FindParmsTest : public StringsFindTest, public testing::WithParamInterface<int32_t> {
};

TEST_P(FindParmsTest, Find)
{
  std::vector<std::string> h_strings{"hello", "", "these", "are stl", "safe"};
  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end());
  cudf::size_type position = GetParam();

  auto strings_view = cudf::strings_column_view(strings);
  {
    auto results = cudf::strings::find(strings_view, cudf::string_scalar("e"), position);
    std::vector<int32_t> h_expected;
    for (auto itr = h_strings.begin(); itr != h_strings.end(); ++itr)
      h_expected.push_back((int32_t)(*itr).find("e", position));
    cudf::test::fixed_width_column_wrapper<int32_t> expected(h_expected.begin(), h_expected.end());
    cudf::test::expect_columns_equal(*results, expected);
  }
  {
    auto results = cudf::strings::rfind(strings_view, cudf::string_scalar("e"), 0, position + 1);
    std::vector<int32_t> h_expected;
    for (auto itr = h_strings.begin(); itr != h_strings.end(); ++itr)
      h_expected.push_back((int32_t)(*itr).rfind("e", position));
    cudf::test::fixed_width_column_wrapper<int32_t> expected(h_expected.begin(), h_expected.end());
    cudf::test::expect_columns_equal(*results, expected);
  }
  {
    auto begin   = position;
    auto results = cudf::strings::find(strings_view, cudf::string_scalar(""), begin);
    cudf::test::fixed_width_column_wrapper<int32_t> expected(
      {begin, (begin > 0 ? -1 : 0), begin, begin, begin});
    cudf::test::expect_columns_equal(*results, expected);
    auto end = position + 1;
    results  = cudf::strings::rfind(strings_view, cudf::string_scalar(""), 0, end);
    cudf::test::fixed_width_column_wrapper<int32_t> rexpected({end, 0, end, end, end});
    cudf::test::expect_columns_equal(*results, rexpected);
  }
}

INSTANTIATE_TEST_CASE_P(StringsFindTest,
                        FindParmsTest,
                        testing::ValuesIn(std::array<int32_t, 4>{0, 1, 2, 3}));

struct StringsSubstringIndexWithScalarTest : public cudf::test::BaseFixture {
};

TEST_F(StringsSubstringIndexWithScalarTest, ZeroSizeStringsColumn)
{
  cudf::column_view col0(cudf::data_type{cudf::STRING}, 0, nullptr, nullptr, 0);
  auto strings_view = cudf::strings_column_view(col0);

  auto results = cudf::strings::substring_index(strings_view, cudf::string_scalar("foo"), 1);
  cudf::test::expect_strings_empty(results->view());
}

TEST_F(StringsSubstringIndexWithScalarTest, AllEmpty)
{
  auto strings_col  = cudf::test::strings_column_wrapper({"", "", "", "", ""});
  auto strings_view = cudf::strings_column_view(strings_col);

  auto exp_results = cudf::test::strings_column_wrapper({"", "", "", "", ""});

  auto results = cudf::strings::substring_index(strings_view, cudf::string_scalar("e"), -1);
  cudf::test::expect_columns_equal(*results, exp_results, true);
}

TEST_F(StringsSubstringIndexWithScalarTest, EmptyDelimiter)
{
  auto strings_col = cudf::test::strings_column_wrapper(
    {"Héllo", "thesé", "", "lease", "tést strings", ""}, {true, true, false, true, true, true});
  ;
  auto strings_view = cudf::strings_column_view(strings_col);

  auto exp_results = cudf::test::strings_column_wrapper({"", "", "", "", "", ""},
                                                        {true, true, false, true, true, true});
  auto results     = cudf::strings::substring_index(strings_view, cudf::string_scalar(""), 1);
  cudf::test::expect_columns_equal(*results, exp_results, true);
}

TEST_F(StringsSubstringIndexWithScalarTest, ZeroCount)
{
  auto strings_col = cudf::test::strings_column_wrapper(
    {"Héllo", "thesé", "", "lease", "tést strings", ""}, {true, true, false, true, true, true});
  ;
  auto strings_view = cudf::strings_column_view(strings_col);

  auto exp_results = cudf::test::strings_column_wrapper({"", "", "", "", "", ""},
                                                        {false, false, false, false, false, false});

  auto results = cudf::strings::substring_index(strings_view, cudf::string_scalar("é"), 0);
  cudf::test::expect_columns_equal(*results, exp_results, true);
}

TEST_F(StringsSubstringIndexWithScalarTest, SearchDelimiter)
{
  auto strings_col = cudf::test::strings_column_wrapper(
    {"Héllo", "thesé", "", "lease", "tést strings", ""}, {true, true, false, true, true, true});
  ;
  auto strings_view = cudf::strings_column_view(strings_col);

  {
    auto exp_results = cudf::test::strings_column_wrapper({"H", "thes", "", "lease", "t", ""},
                                                          {true, true, false, true, true, true});

    auto results = cudf::strings::substring_index(strings_view, cudf::string_scalar("é"), 1);
    cudf::test::expect_columns_equal(*results, exp_results, true);
  }

  {
    auto exp_results = cudf::test::strings_column_wrapper(
      {"llo", "", "", "lease", "st strings", ""}, {true, true, false, true, true, true});

    auto results = cudf::strings::substring_index(strings_view, cudf::string_scalar("é"), -1);
    cudf::test::expect_columns_equal(*results, exp_results, true);
  }

  {
    auto results = cudf::strings::substring_index(strings_view, cudf::string_scalar("é"), 2);
    cudf::test::expect_columns_equal(*results, strings_view.parent(), true);
  }

  {
    auto results = cudf::strings::substring_index(strings_view, cudf::string_scalar("é"), -2);
    cudf::test::expect_columns_equal(*results, strings_view.parent(), true);
  }

  {
    auto col0 = cudf::test::strings_column_wrapper(
      {"Hello LLollooogh", "oopppllo", "", "oppollo", "polo lop apploo po", ""},
      {true, true, false, true, true, true});

    auto exp_results = cudf::test::strings_column_wrapper({"Hello LL", "o", "", "opp", "pol", ""},
                                                          {true, true, false, true, true, true});

    auto results =
      cudf::strings::substring_index(cudf::strings_column_view{col0}, cudf::string_scalar("o"), 2);
    cudf::test::expect_columns_equal(*results, exp_results, true);
  }

  {
    auto col0 = cudf::test::strings_column_wrapper(
      {"Hello LLollooogh", "oopppllo", "", "oppollo", "polo lop apploo po", ""},
      {true, true, false, true, true, true});

    auto exp_results = cudf::test::strings_column_wrapper({"ogh", "pppllo", "", "llo", " po", ""},
                                                          {true, true, false, true, true, true});

    auto results =
      cudf::strings::substring_index(cudf::strings_column_view{col0}, cudf::string_scalar("o"), -2);
    cudf::test::expect_columns_equal(*results, exp_results, true);
  }

  {
    auto col0 = cudf::test::strings_column_wrapper(
      {"Héllo HélloHéllo", "Hélloééééé", "", "éééééé", "poloéé lopéé applooéé po", ""},
      {true, true, false, true, true, true});

    auto exp_results = cudf::test::strings_column_wrapper(
      {"Héllo HélloHéllo", "Hélloééééé", "", "éééé", "poloéé lopéé apploo", ""},
      {true, true, false, true, true, true});

    auto results =
      cudf::strings::substring_index(cudf::strings_column_view{col0}, cudf::string_scalar("éé"), 3);
    cudf::test::expect_columns_equal(*results, exp_results, true);
  }

  {
    auto col0 = cudf::test::strings_column_wrapper(
      {"Héllo HélloHéllo", "Hélloééééé", "", "éééééé", "poloéé lopéé applooéé po", ""},
      {true, true, false, true, true, true});

    auto exp_results = cudf::test::strings_column_wrapper(
      {"Héllo HélloHéllo", "Hélloééééé", "", "éééé", " lopéé applooéé po", ""},
      {true, true, false, true, true, true});

    auto results = cudf::strings::substring_index(
      cudf::strings_column_view{col0}, cudf::string_scalar("éé"), -3);
    cudf::test::expect_columns_equal(*results, exp_results, true);
  }

  {
    auto col0 = cudf::test::strings_column_wrapper({"www.yahoo.com",
                                                    "www.apache..org",
                                                    "tennis...com",
                                                    "nvidia....com",
                                                    "google...........com",
                                                    "microsoft...c.....co..m"});

    auto exp_results = cudf::test::strings_column_wrapper(
      {"www.yahoo.com", "www.apache.", "tennis..", "nvidia..", "google..", "microsoft.."});

    auto results =
      cudf::strings::substring_index(cudf::strings_column_view{col0}, cudf::string_scalar("."), 3);
    cudf::test::expect_columns_equal(*results, exp_results, true);
  }

  {
    auto col0 = cudf::test::strings_column_wrapper({"www.yahoo.com",
                                                    "www.apache..org",
                                                    "tennis..com",
                                                    "nvidia....com",
                                                    "google...........com",
                                                    ".",
                                                    "microsoft...c.....co..m"});

    auto exp_results = cudf::test::strings_column_wrapper(
      {"www.yahoo.com", "www.apache..org", "tennis..com", "..com", "..com", ".", "co..m"});

    auto results = cudf::strings::substring_index(
      cudf::strings_column_view{col0}, cudf::string_scalar(".."), -2);
    cudf::test::expect_columns_equal(*results, exp_results, true);
  }
}

struct StringsSubstringIndexWithColumnTest : public cudf::test::BaseFixture {
};

TEST_F(StringsSubstringIndexWithColumnTest, ZeroSizeStringsColumn)
{
  cudf::column_view col0(cudf::data_type{cudf::STRING}, 0, nullptr, nullptr, 0);
  auto strings_view = cudf::strings_column_view(col0);

  auto results = cudf::strings::substring_index(strings_view, strings_view, 1);
  // Check empty column
  cudf::test::expect_strings_empty(results->view());
}

TEST_F(StringsSubstringIndexWithColumnTest, GenerateExceptions)
{
  auto col0      = cudf::test::strings_column_wrapper({"", "", "", "", ""});
  auto delim_col = cudf::test::strings_column_wrapper({"", "foo", "bar", "."});

  EXPECT_THROW(cudf::strings::substring_index(
                 cudf::strings_column_view{col0}, cudf::strings_column_view{delim_col}, -1),
               cudf::logic_error);
}

TEST_F(StringsSubstringIndexWithColumnTest, ColumnAllEmpty)
{
  auto col0      = cudf::test::strings_column_wrapper({"", "", "", "", ""});
  auto delim_col = cudf::test::strings_column_wrapper({"", "foo", "bar", ".", "/"});

  auto exp_results = cudf::test::strings_column_wrapper({"", "", "", "", ""});

  auto results = cudf::strings::substring_index(
    cudf::strings_column_view{col0}, cudf::strings_column_view{delim_col}, -1);
  cudf::test::expect_columns_equal(*results, exp_results, true);
}

TEST_F(StringsSubstringIndexWithColumnTest, DelimiterAllEmptyAndInvalid)
{
  auto col0 = cudf::test::strings_column_wrapper(
    {"Héllo", "thesé", "", "lease", "tést strings", ""}, {true, true, false, true, true, true});
  auto delim_col = cudf::test::strings_column_wrapper({"", "", "", "", "", ""},
                                                      {true, false, true, false, true, false});

  auto exp_results = cudf::test::strings_column_wrapper({"", "", "", "", "", ""},
                                                        {true, false, false, false, true, false});

  auto results = cudf::strings::substring_index(
    cudf::strings_column_view{col0}, cudf::strings_column_view{delim_col}, 1);
  cudf::test::expect_columns_equal(*results, exp_results, true);
}

TEST_F(StringsSubstringIndexWithColumnTest, ZeroDelimiterCount)
{
  auto col0 = cudf::test::strings_column_wrapper(
    {"Héllo", "thesé", "", "lease", "tést strings", ""}, {true, true, false, true, true, true});
  auto delim_col = cudf::test::strings_column_wrapper({"", "", "", "", "", ""},
                                                      {true, false, true, false, true, false});

  auto exp_results = cudf::test::strings_column_wrapper({"", "", "", "", "", ""},
                                                        {false, false, false, false, false, false});

  auto results = cudf::strings::substring_index(
    cudf::strings_column_view{col0}, cudf::strings_column_view{delim_col}, 0);
  cudf::test::expect_columns_equal(*results, exp_results, true);
}

TEST_F(StringsSubstringIndexWithColumnTest, SearchDelimiter)
{
  {
    auto col0 = cudf::test::strings_column_wrapper(
      {"H™élloi ™◎oo™ff™", "thesé", "", "lease™", "tést strings", "™"},
      {true, true, false, true, true, true});
    auto delim_col = cudf::test::strings_column_wrapper({"™", "™", "", "e", "t", "™"});

    auto exp_results = cudf::test::strings_column_wrapper({"H", "thesé", "", "l", "", ""},
                                                          {true, true, false, true, true, true});

    auto results = cudf::strings::substring_index(
      cudf::strings_column_view{col0}, cudf::strings_column_view{delim_col}, 1);
    cudf::test::expect_columns_equal(*results, exp_results, true);
  }

  {
    auto col0      = cudf::test::strings_column_wrapper({"H™élloﬀ ﬀﬀi ™◎ooﬀ™ff™",
                                                    "tﬀﬀhﬀesé",
                                                    "",
                                                    "lﬀ fooﬀ ffﬀ eaﬀse™",
                                                    "tést ﬀstri.nﬀgs",
                                                    "ﬀﬀ ™ ﬀﬀ ﬀ"},
                                                   {true, true, false, true, true, true});
    auto delim_col = cudf::test::strings_column_wrapper({"ﬀ™", "ﬀ", "", "ﬀ ", "t", "ﬀ ™"});

    auto exp_results = cudf::test::strings_column_wrapper(
      {"ff™", "esé", "", "eaﬀse™", "ri.nﬀgs", " ﬀﬀ ﬀ"}, {true, true, false, true, true, true});

    auto results = cudf::strings::substring_index(
      cudf::strings_column_view{col0}, cudf::strings_column_view{delim_col}, -1);
    cudf::test::expect_columns_equal(*results, exp_results, true);
  }

  {
    auto col0 = cudf::test::strings_column_wrapper({"H™élloﬀ ﬀﬀi fooﬀ™ barﬀ™ gooﬀ™ ™◎ooﬀ™ff™",
                                                    "tﬀﬀhﬀesé",
                                                    "",
                                                    "lﬀ fooﬀ ffﬀ eaﬀse™",
                                                    "tést ﬀ™ffﬀ™ﬀ™ffﬀstri.ﬀ™ffﬀ™nﬀgs",
                                                    "ﬀﬀ ™ ﬀﬀ ﬀ™ ﬀ™ﬀ™ﬀ™ ﬀ™ﬀ™ ﬀ"},
                                                   {true, true, false, true, true, true});
    auto delim_col = cudf::test::strings_column_wrapper({"ﬀ™", "ﬀ", "", "e ", "ﬀ™ff", "ﬀ™ﬀ™"},
                                                        {true, true, false, true, true, true});

    auto exp_results = cudf::test::strings_column_wrapper({"H™élloﬀ ﬀﬀi fooﬀ™ barﬀ™ goo",
                                                           "tﬀﬀh",
                                                           "",
                                                           "lﬀ fooﬀ ffﬀ eaﬀse™",
                                                           "tést ﬀ™ffﬀ™ﬀ™ffﬀstri.",
                                                           "ﬀﬀ ™ ﬀﬀ ﬀ™ ﬀ™ﬀ™ﬀ™ ﬀ™ﬀ™ ﬀ"},
                                                          {true, true, false, true, true, true});

    auto results = cudf::strings::substring_index(
      cudf::strings_column_view{col0}, cudf::strings_column_view{delim_col}, 3);
    cudf::test::expect_columns_equal(*results, exp_results, true);
  }

  {
    auto col0 = cudf::test::strings_column_wrapper({"H™élloﬀ ﬀﬀi fooﬀ™ barﬀ™ gooﬀ™ ™◎ooﬀ™ff™",
                                                    "tﬀﬀhﬀesé",
                                                    "",
                                                    "lﬀ fooﬀ ffﬀ eaﬀse™",
                                                    "tést ﬀ™ffﬀ™ﬀ™ffﬀstri.ﬀ™ffﬀ™nﬀgs",
                                                    "ﬀﬀ ™ ﬀﬀ ﬀ™ ﬀ™ﬀ™ﬀ™ ﬀ™ﬀ™ ﬀ"});
    auto delim_col = cudf::test::strings_column_wrapper({"ﬀ™", "ﬀ", "", "e ", "ﬀ™ff", "ﬀ™ﬀ™"},
                                                        {true, true, false, true, true, true});

    auto exp_results = cudf::test::strings_column_wrapper({" gooﬀ™ ™◎ooﬀ™ff™",
                                                           "ﬀhﬀesé",
                                                           "",
                                                           "lﬀ fooﬀ ffﬀ eaﬀse™",
                                                           "ﬀ™ﬀ™ffﬀstri.ﬀ™ffﬀ™nﬀgs",
                                                           "ﬀﬀ ™ ﬀﬀ ﬀ™ ﬀ™ﬀ™ﬀ™ ﬀ™ﬀ™ ﬀ"},
                                                          {true, true, false, true, true, true});

    auto results = cudf::strings::substring_index(
      cudf::strings_column_view{col0}, cudf::strings_column_view{delim_col}, -3);
    cudf::test::expect_columns_equal(*results, exp_results, true);
  }
}
