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

#include <cudf/column/column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/substring.hpp>

#include <thrust/host_vector.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/sequence.h>

#include <string>
#include <vector>

struct StringsSubstringsTest : public cudf::test::BaseFixture {
};

TEST_F(StringsSubstringsTest, Substring)
{
  std::vector<const char*> h_strings{"Héllo", "thesé", nullptr, "ARE THE", "tést strings", ""};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  std::vector<const char*> h_expected({"llo", "esé", nullptr, "E T", "st ", ""});
  cudf::test::strings_column_wrapper expected(
    h_expected.begin(),
    h_expected.end(),
    thrust::make_transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));

  auto strings_column = static_cast<cudf::strings_column_view>(strings);
  auto results        = cudf::strings::slice_strings(strings_column, 2, 5);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

class Parameters : public StringsSubstringsTest,
                   public testing::WithParamInterface<cudf::size_type> {
};

TEST_P(Parameters, Substring)
{
  std::vector<std::string> h_strings{"basic strings", "that can", "be used", "with STL"};
  cudf::size_type start = GetParam();

  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end());
  auto strings_column = cudf::strings_column_view(strings);
  auto results        = cudf::strings::slice_strings(strings_column, start);

  std::vector<std::string> h_expected;
  for (auto itr = h_strings.begin(); itr != h_strings.end(); ++itr)
    h_expected.push_back((*itr).substr(start));

  cudf::test::strings_column_wrapper expected(h_expected.begin(), h_expected.end());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_P(Parameters, Substring_From)
{
  std::vector<std::string> h_strings{"basic strings", "that can", "be used", "with STL"};
  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end());
  auto strings_column = cudf::strings_column_view(strings);

  auto param_index = GetParam();
  thrust::host_vector<int32_t> starts(h_strings.size());
  thrust::sequence(starts.begin(), starts.end(), param_index);
  cudf::test::fixed_width_column_wrapper<int32_t> starts_column(starts.begin(), starts.end());
  thrust::host_vector<int32_t> stops(h_strings.size());
  thrust::sequence(stops.begin(), stops.end(), param_index + 2);
  cudf::test::fixed_width_column_wrapper<int32_t> stops_column(stops.begin(), stops.end());

  auto results = cudf::strings::slice_strings(strings_column, starts_column, stops_column);

  std::vector<std::string> h_expected;
  for (size_t idx = 0; idx < h_strings.size(); ++idx)
    h_expected.push_back(h_strings[idx].substr(starts[idx], stops[idx] - starts[idx]));

  cudf::test::strings_column_wrapper expected(h_expected.begin(), h_expected.end());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_P(Parameters, AllEmpty)
{
  std::vector<std::string> h_strings{"", "", "", ""};
  cudf::size_type start = GetParam();

  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end());
  auto strings_column = cudf::strings_column_view(strings);
  auto results        = cudf::strings::slice_strings(strings_column, start);

  std::vector<std::string> h_expected(h_strings);
  cudf::test::strings_column_wrapper expected(h_expected.begin(), h_expected.end());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  thrust::host_vector<int32_t> starts(h_strings.size(), 1);
  cudf::test::fixed_width_column_wrapper<int32_t> starts_column(starts.begin(), starts.end());
  thrust::host_vector<int32_t> stops(h_strings.size(), 2);
  cudf::test::fixed_width_column_wrapper<int32_t> stops_column(stops.begin(), stops.end());

  results = cudf::strings::slice_strings(strings_column, starts_column, stops_column);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_P(Parameters, AllNulls)
{
  std::vector<const char*> h_strings{nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  cudf::size_type start = GetParam();

  auto strings_column = cudf::strings_column_view(strings);
  auto results        = cudf::strings::slice_strings(strings_column, start);

  std::vector<const char*> h_expected(h_strings);
  cudf::test::strings_column_wrapper expected(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  thrust::host_vector<int32_t> starts(h_strings.size(), 1);
  cudf::test::fixed_width_column_wrapper<int32_t> starts_column(starts.begin(), starts.end());
  thrust::host_vector<int32_t> stops(h_strings.size(), 2);
  cudf::test::fixed_width_column_wrapper<int32_t> stops_column(stops.begin(), stops.end());

  results = cudf::strings::slice_strings(strings_column, starts_column, stops_column);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

INSTANTIATE_TEST_CASE_P(StringsSubstringsTest,
                        Parameters,
                        testing::ValuesIn(std::array<cudf::size_type, 3>{1, 2, 3}));

TEST_F(StringsSubstringsTest, NegativePositions)
{
  cudf::test::strings_column_wrapper strings{
    "a", "bc", "def", "ghij", "klmno", "pqrstu", "vwxyz", ""};
  auto strings_column = cudf::strings_column_view(strings);
  {
    auto results = cudf::strings::slice_strings(strings_column, -1);
    cudf::test::strings_column_wrapper expected{"a", "c", "f", "j", "o", "u", "z", ""};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto results = cudf::strings::slice_strings(strings_column, 0, -1);
    cudf::test::strings_column_wrapper expected{"", "b", "de", "ghi", "klmn", "pqrst", "vwxy", ""};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto results = cudf::strings::slice_strings(strings_column, 7, -2, -1);
    cudf::test::strings_column_wrapper expected{"a", "c", "f", "j", "o", "u", "z", ""};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto results = cudf::strings::slice_strings(strings_column, 7, -7, -1);
    cudf::test::strings_column_wrapper expected{
      "a", "cb", "fed", "jihg", "onmlk", "utsrqp", "zyxwv", ""};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto results = cudf::strings::slice_strings(strings_column, -3, -1);
    cudf::test::strings_column_wrapper expected{"", "b", "de", "hi", "mn", "st", "xy", ""};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(StringsSubstringsTest, NullPositions)
{
  cudf::test::strings_column_wrapper strings{"a", "bc", "def", "ghij", "klmno", "pqrstu", "vwxyz"};
  auto strings_column = cudf::strings_column_view(strings);
  {
    auto results = cudf::strings::slice_strings(strings_column,
                                                cudf::numeric_scalar<cudf::size_type>(0, false),
                                                cudf::numeric_scalar<cudf::size_type>(0, false),
                                                -1);
    cudf::test::strings_column_wrapper expected{
      "a", "cb", "fed", "jihg", "onmlk", "utsrqp", "zyxwv"};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto results = cudf::strings::slice_strings(strings_column,
                                                cudf::numeric_scalar<cudf::size_type>(0, false),
                                                cudf::numeric_scalar<cudf::size_type>(0, false),
                                                2);
    cudf::test::strings_column_wrapper expected{"a", "b", "df", "gi", "kmo", "prt", "vxz"};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto results = cudf::strings::slice_strings(
      strings_column, 0, cudf::numeric_scalar<cudf::size_type>(0, false), -1);
    cudf::test::strings_column_wrapper expected{"a", "b", "d", "g", "k", "p", "v"};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto results = cudf::strings::slice_strings(
      strings_column, cudf::numeric_scalar<cudf::size_type>(0, false), -2, -1);
    cudf::test::strings_column_wrapper expected{"a", "c", "f", "j", "o", "u", "z"};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto results = cudf::strings::slice_strings(
      strings_column, cudf::numeric_scalar<cudf::size_type>(0, false), -1, 2);
    cudf::test::strings_column_wrapper expected{"", "b", "d", "gi", "km", "prt", "vx"};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(StringsSubstringsTest, MaxPositions)
{
  cudf::test::strings_column_wrapper strings{"a", "bc", "def", "ghij", "klmno", "pqrstu", "vwxyz"};
  auto strings_column = cudf::strings_column_view(strings);
  cudf::test::strings_column_wrapper expected{"", "", "", "", "", "", ""};

  auto results = cudf::strings::slice_strings(strings_column, 10);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  results = cudf::strings::slice_strings(strings_column, 0, -10);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  results = cudf::strings::slice_strings(
    strings_column, cudf::numeric_scalar<cudf::size_type>(0, false), -10);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  results = cudf::strings::slice_strings(strings_column, 10, 19);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  results = cudf::strings::slice_strings(strings_column, 10, 19, 9);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  results = cudf::strings::slice_strings(strings_column, -10, -19);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  results = cudf::strings::slice_strings(strings_column, -10, -19, -1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsSubstringsTest, Error)
{
  cudf::test::strings_column_wrapper strings{"this string intentionally left blank"};
  auto strings_view = cudf::strings_column_view(strings);
  EXPECT_THROW(cudf::strings::slice_strings(strings_view, 0, 0, 0), cudf::logic_error);

  auto delim_col = cudf::test::strings_column_wrapper({"", ""});
  EXPECT_THROW(cudf::strings::slice_strings(strings_view, cudf::strings_column_view{delim_col}, -1),
               cudf::logic_error);

  auto indexes = cudf::test::fixed_width_column_wrapper<int32_t>({1, 2});
  EXPECT_THROW(cudf::strings::slice_strings(strings_view, indexes, indexes), cudf::logic_error);

  auto indexes_null = cudf::test::fixed_width_column_wrapper<int32_t>({1}, {0});
  EXPECT_THROW(cudf::strings::slice_strings(strings_view, indexes_null, indexes_null),
               cudf::logic_error);

  auto indexes_bad = cudf::test::fixed_width_column_wrapper<float>({1});
  EXPECT_THROW(cudf::strings::slice_strings(strings_view, indexes_bad, indexes_bad),
               cudf::logic_error);
}

TEST_F(StringsSubstringsTest, ZeroSizeStringsColumn)
{
  cudf::column_view zero_size_strings_column(
    cudf::data_type{cudf::type_id::STRING}, 0, nullptr, nullptr, 0);
  auto strings_view = cudf::strings_column_view(zero_size_strings_column);

  auto results = cudf::strings::slice_strings(strings_view, 1, 2);
  cudf::test::expect_column_empty(results->view());

  results = cudf::strings::slice_strings(strings_view, cudf::string_scalar("foo"), 1);
  cudf::test::expect_column_empty(results->view());

  cudf::column_view starts_column(cudf::data_type{cudf::type_id::INT32}, 0, nullptr, nullptr, 0);
  cudf::column_view stops_column(cudf::data_type{cudf::type_id::INT32}, 0, nullptr, nullptr, 0);
  results = cudf::strings::slice_strings(strings_view, starts_column, stops_column);
  cudf::test::expect_column_empty(results->view());

  results = cudf::strings::slice_strings(strings_view, strings_view, 1);
  cudf::test::expect_column_empty(results->view());
}

TEST_F(StringsSubstringsTest, AllEmpty)
{
  auto strings_col  = cudf::test::strings_column_wrapper({"", "", "", "", ""});
  auto strings_view = cudf::strings_column_view(strings_col);
  auto exp_results  = cudf::column_view(strings_col);

  auto results = cudf::strings::slice_strings(strings_view, cudf::string_scalar("e"), -1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results);
  results = cudf::strings::slice_strings(strings_view, strings_view, -1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results);
}

TEST_F(StringsSubstringsTest, EmptyDelimiter)
{
  auto strings_col = cudf::test::strings_column_wrapper(
    {"Héllo", "thesé", "", "lease", "tést strings", ""}, {true, true, false, true, true, true});
  ;
  auto strings_view = cudf::strings_column_view(strings_col);

  auto exp_results = cudf::test::strings_column_wrapper({"", "", "", "", "", ""},
                                                        {true, true, false, true, true, true});

  auto results = cudf::strings::slice_strings(strings_view, cudf::string_scalar(""), 1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results);

  auto delim_col = cudf::test::strings_column_wrapper({"", "", "", "", "", ""},
                                                      {true, false, true, false, true, false});

  results = cudf::strings::slice_strings(strings_view, cudf::strings_column_view{delim_col}, 1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results);
}

TEST_F(StringsSubstringsTest, ZeroCount)
{
  auto strings_col = cudf::test::strings_column_wrapper(
    {"Héllo", "thesé", "", "lease", "tést strings", ""}, {true, true, false, true, true, true});
  ;
  auto strings_view = cudf::strings_column_view(strings_col);

  auto exp_results = cudf::test::strings_column_wrapper({"", "", "", "", "", ""},
                                                        {true, true, false, true, true, true});

  auto results = cudf::strings::slice_strings(strings_view, cudf::string_scalar("é"), 0);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results);

  auto delim_col = cudf::test::strings_column_wrapper({"", "", "", "", "", ""},
                                                      {true, false, true, false, true, false});

  results = cudf::strings::slice_strings(strings_view, cudf::strings_column_view{delim_col}, 0);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results);
}

TEST_F(StringsSubstringsTest, SearchScalarDelimiter)
{
  auto strings_col = cudf::test::strings_column_wrapper(
    {"Héllo", "thesé", "", "lease", "tést strings", ""}, {true, true, false, true, true, true});
  ;
  auto strings_view = cudf::strings_column_view(strings_col);

  {
    auto exp_results = cudf::test::strings_column_wrapper({"H", "thes", "", "lease", "t", ""},
                                                          {true, true, false, true, true, true});

    auto results = cudf::strings::slice_strings(strings_view, cudf::string_scalar("é"), 1);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results);
  }

  {
    auto exp_results = cudf::test::strings_column_wrapper(
      {"llo", "", "", "lease", "st strings", ""}, {true, true, false, true, true, true});

    auto results = cudf::strings::slice_strings(strings_view, cudf::string_scalar("é"), -1);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results);
  }

  {
    auto results = cudf::strings::slice_strings(strings_view, cudf::string_scalar("é"), 2);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, strings_col);
  }

  {
    auto results = cudf::strings::slice_strings(strings_view, cudf::string_scalar("é"), -2);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, strings_col);
  }

  {
    auto col0 = cudf::test::strings_column_wrapper(
      {"Hello LLollooogh", "oopppllo", "", "oppollo", "polo lop apploo po", ""},
      {true, true, false, true, true, true});

    auto exp_results = cudf::test::strings_column_wrapper({"Hello LL", "o", "", "opp", "pol", ""},
                                                          {true, true, false, true, true, true});

    auto results =
      cudf::strings::slice_strings(cudf::strings_column_view{col0}, cudf::string_scalar("o"), 2);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results);
  }

  {
    auto col0 = cudf::test::strings_column_wrapper(
      {"Hello LLollooogh", "oopppllo", "", "oppollo", "polo lop apploo po", ""},
      {true, true, false, true, true, true});

    auto exp_results = cudf::test::strings_column_wrapper({"ogh", "pppllo", "", "llo", " po", ""},
                                                          {true, true, false, true, true, true});

    auto results =
      cudf::strings::slice_strings(cudf::strings_column_view{col0}, cudf::string_scalar("o"), -2);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results);
  }

  {
    auto col0 = cudf::test::strings_column_wrapper(
      {"Héllo HélloHéllo", "Hélloééééé", "", "éééééé", "poloéé lopéé applooéé po", ""},
      {true, true, false, true, true, true});

    auto exp_results = cudf::test::strings_column_wrapper(
      {"Héllo HélloHéllo", "Hélloééééé", "", "éééé", "poloéé lopéé apploo", ""},
      {true, true, false, true, true, true});

    auto results =
      cudf::strings::slice_strings(cudf::strings_column_view{col0}, cudf::string_scalar("éé"), 3);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results);
  }

  {
    auto col0 = cudf::test::strings_column_wrapper(
      {"Héllo HélloHéllo", "Hélloééééé", "", "éééééé", "poloéé lopéé applooéé po", ""},
      {true, true, false, true, true, true});

    auto exp_results = cudf::test::strings_column_wrapper(
      {"Héllo HélloHéllo", "Hélloééééé", "", "éééé", " lopéé applooéé po", ""},
      {true, true, false, true, true, true});

    auto results =
      cudf::strings::slice_strings(cudf::strings_column_view{col0}, cudf::string_scalar("éé"), -3);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results);
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
      cudf::strings::slice_strings(cudf::strings_column_view{col0}, cudf::string_scalar("."), 3);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results);
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

    auto results =
      cudf::strings::slice_strings(cudf::strings_column_view{col0}, cudf::string_scalar(".."), -2);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results);
  }
}

TEST_F(StringsSubstringsTest, SearchColumnDelimiter)
{
  {
    auto col0 = cudf::test::strings_column_wrapper(
      {"H™élloi ™◎oo™ff™", "thesé", "", "lease™", "tést strings", "™"},
      {true, true, false, true, true, true});
    auto delim_col = cudf::test::strings_column_wrapper({"™", "™", "", "e", "t", "™"});

    auto exp_results = cudf::test::strings_column_wrapper({"H", "thesé", "", "l", "", ""},
                                                          {true, true, false, true, true, true});

    auto results = cudf::strings::slice_strings(
      cudf::strings_column_view{col0}, cudf::strings_column_view{delim_col}, 1);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results);
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

    auto results = cudf::strings::slice_strings(
      cudf::strings_column_view{col0}, cudf::strings_column_view{delim_col}, -1);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results);
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

    auto results = cudf::strings::slice_strings(
      cudf::strings_column_view{col0}, cudf::strings_column_view{delim_col}, 3);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results);
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
                                                           "ﬀﬀ ™ ﬀﬀ ﬀ™ ﬀ™ﬀ™ﬀ™ ﬀ™ﬀ™ ﬀ"});

    auto results = cudf::strings::slice_strings(
      cudf::strings_column_view{col0}, cudf::strings_column_view{delim_col}, -3);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results);
  }
}
