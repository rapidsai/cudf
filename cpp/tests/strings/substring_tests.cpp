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
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/substring.hpp>

#include <tests/strings/utilities.h>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>

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
  cudf::test::expect_columns_equal(*results, expected);
}

class SubstringParmsTest : public StringsSubstringsTest,
                           public testing::WithParamInterface<int32_t> {
};

TEST_P(SubstringParmsTest, Substring)
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
  cudf::test::expect_columns_equal(*results, expected);
}

TEST_P(SubstringParmsTest, Substring_From)
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
  cudf::test::expect_columns_equal(*results, expected);
}

TEST_P(SubstringParmsTest, AllEmpty)
{
  std::vector<std::string> h_strings{"", "", "", ""};
  cudf::size_type start = GetParam();

  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end());
  auto strings_column = cudf::strings_column_view(strings);
  auto results        = cudf::strings::slice_strings(strings_column, start);

  std::vector<std::string> h_expected(h_strings);
  cudf::test::strings_column_wrapper expected(h_expected.begin(), h_expected.end());
  cudf::test::expect_columns_equal(*results, expected);

  thrust::host_vector<int32_t> starts(h_strings.size(), 1);
  cudf::test::fixed_width_column_wrapper<int32_t> starts_column(starts.begin(), starts.end());
  thrust::host_vector<int32_t> stops(h_strings.size(), 2);
  cudf::test::fixed_width_column_wrapper<int32_t> stops_column(stops.begin(), stops.end());

  results = cudf::strings::slice_strings(strings_column, starts_column, stops_column);
  cudf::test::expect_columns_equal(*results, expected);
}

TEST_P(SubstringParmsTest, AllNulls)
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
  cudf::test::expect_columns_equal(*results, expected);

  thrust::host_vector<int32_t> starts(h_strings.size(), 1);
  cudf::test::fixed_width_column_wrapper<int32_t> starts_column(starts.begin(), starts.end());
  thrust::host_vector<int32_t> stops(h_strings.size(), 2);
  cudf::test::fixed_width_column_wrapper<int32_t> stops_column(stops.begin(), stops.end());

  results = cudf::strings::slice_strings(strings_column, starts_column, stops_column);
  cudf::test::expect_columns_equal(*results, expected);
}

INSTANTIATE_TEST_CASE_P(StringsSubstringsTest,
                        SubstringParmsTest,
                        testing::ValuesIn(std::array<cudf::size_type, 3>{1, 2, 3}));

TEST_F(StringsSubstringsTest, ZeroSizeStringsColumn)
{
  cudf::column_view zero_size_strings_column(cudf::data_type{cudf::STRING}, 0, nullptr, nullptr, 0);
  auto strings_column = cudf::strings_column_view(zero_size_strings_column);
  auto results        = cudf::strings::slice_strings(strings_column, 1, 2);
  cudf::test::expect_strings_empty(results->view());

  cudf::column_view starts_column(cudf::data_type{cudf::INT32}, 0, nullptr, nullptr, 0);
  cudf::column_view stops_column(cudf::data_type{cudf::INT32}, 0, nullptr, nullptr, 0);
  results = cudf::strings::slice_strings(strings_column, starts_column, stops_column);
  cudf::test::expect_strings_empty(results->view());
}

TEST_F(StringsSubstringsTest, NegativePositions)
{
  cudf::test::strings_column_wrapper strings{
    "a", "bc", "def", "ghij", "klmno", "pqrstu", "vwxyz", ""};
  auto strings_column = cudf::strings_column_view(strings);
  {
    auto results = cudf::strings::slice_strings(strings_column, -1);
    cudf::test::strings_column_wrapper expected{"a", "c", "f", "j", "o", "u", "z", ""};
    cudf::test::expect_columns_equal(*results, expected);
  }
  {
    auto results = cudf::strings::slice_strings(strings_column, 0, -1);
    cudf::test::strings_column_wrapper expected{"", "b", "de", "ghi", "klmn", "pqrst", "vwxy", ""};
    cudf::test::expect_columns_equal(*results, expected);
  }
  {
    auto results = cudf::strings::slice_strings(strings_column, 7, -2, -1);
    cudf::test::strings_column_wrapper expected{"a", "c", "f", "j", "o", "u", "z", ""};
    cudf::test::expect_columns_equal(*results, expected);
  }
  {
    auto results = cudf::strings::slice_strings(strings_column, 7, -7, -1);
    cudf::test::strings_column_wrapper expected{
      "a", "cb", "fed", "jihg", "onmlk", "utsrqp", "zyxwv", ""};
    cudf::test::expect_columns_equal(*results, expected);
  }
  {
    auto results = cudf::strings::slice_strings(strings_column, -3, -1);
    cudf::test::strings_column_wrapper expected{"", "b", "de", "hi", "mn", "st", "xy", ""};
    cudf::test::expect_columns_equal(*results, expected);
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
    cudf::test::expect_columns_equal(*results, expected);
  }
  {
    auto results = cudf::strings::slice_strings(strings_column,
                                                cudf::numeric_scalar<cudf::size_type>(0, false),
                                                cudf::numeric_scalar<cudf::size_type>(0, false),
                                                2);
    cudf::test::strings_column_wrapper expected{"a", "b", "df", "gi", "kmo", "prt", "vxz"};
    cudf::test::expect_columns_equal(*results, expected);
  }
  {
    auto results = cudf::strings::slice_strings(
      strings_column, 0, cudf::numeric_scalar<cudf::size_type>(0, false), -1);
    cudf::test::strings_column_wrapper expected{"a", "b", "d", "g", "k", "p", "v"};
    cudf::test::expect_columns_equal(*results, expected);
  }
  {
    auto results = cudf::strings::slice_strings(
      strings_column, cudf::numeric_scalar<cudf::size_type>(0, false), -2, -1);
    cudf::test::strings_column_wrapper expected{"a", "c", "f", "j", "o", "u", "z"};
    cudf::test::expect_columns_equal(*results, expected);
  }
  {
    auto results = cudf::strings::slice_strings(
      strings_column, cudf::numeric_scalar<cudf::size_type>(0, false), -1, 2);
    cudf::test::strings_column_wrapper expected{"", "b", "d", "gi", "km", "prt", "vx"};
    cudf::test::expect_columns_equal(*results, expected);
  }
}

TEST_F(StringsSubstringsTest, MaxPositions)
{
  cudf::test::strings_column_wrapper strings{"a", "bc", "def", "ghij", "klmno", "pqrstu", "vwxyz"};
  auto strings_column = cudf::strings_column_view(strings);
  cudf::test::strings_column_wrapper expected{"", "", "", "", "", "", ""};

  auto results = cudf::strings::slice_strings(strings_column, 10);
  cudf::test::expect_columns_equal(*results, expected);

  results = cudf::strings::slice_strings(strings_column, 0, -10);
  cudf::test::expect_columns_equal(*results, expected);

  results = cudf::strings::slice_strings(
    strings_column, cudf::numeric_scalar<cudf::size_type>(0, false), -10);
  cudf::test::expect_columns_equal(*results, expected);

  results = cudf::strings::slice_strings(strings_column, 10, 19);
  cudf::test::expect_columns_equal(*results, expected);

  results = cudf::strings::slice_strings(strings_column, 10, 19, 9);
  cudf::test::expect_columns_equal(*results, expected);

  results = cudf::strings::slice_strings(strings_column, -10, -19);
  cudf::test::expect_columns_equal(*results, expected);

  results = cudf::strings::slice_strings(strings_column, -10, -19, -1);
  cudf::test::expect_columns_equal(*results, expected);
}

TEST_F(StringsSubstringsTest, Error)
{
  cudf::test::strings_column_wrapper strings{"this string intentionally left blank"};
  auto strings_column = cudf::strings_column_view(strings);
  EXPECT_THROW(cudf::strings::slice_strings(strings_column, 0, 0, 0), cudf::logic_error);
}
