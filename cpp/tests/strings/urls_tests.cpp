/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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
#include <cudf/strings/convert/convert_urls.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <random>
#include <vector>

struct StringsConvertTest : public cudf::test::BaseFixture {
};

TEST_F(StringsConvertTest, UrlEncode)
{
  std::vector<const char*> h_strings{"www.nvidia.com/rapids?p=é",
                                     "/_file-7.txt",
                                     "a b+c~d",
                                     "e\tfgh\\jklmnopqrstuvwxyz",
                                     "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                                     "0123456789",
                                     " \t\f\n",
                                     nullptr,
                                     ""};
  cudf::test::strings_column_wrapper strings(
    h_strings.cbegin(),
    h_strings.cend(),
    thrust::make_transform_iterator(h_strings.cbegin(),
                                    [](auto const str) { return str != nullptr; }));

  auto strings_view = cudf::strings_column_view(strings);
  auto results      = cudf::strings::url_encode(strings_view);

  std::vector<const char*> h_expected{"www.nvidia.com%2Frapids%3Fp%3D%C3%A9",
                                      "%2F_file-7.txt",
                                      "a%20b%2Bc~d",
                                      "e%09fgh%5Cjklmnopqrstuvwxyz",
                                      "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                                      "0123456789",
                                      "%20%09%0C%0A",
                                      nullptr,
                                      ""};
  cudf::test::strings_column_wrapper expected(
    h_expected.cbegin(),
    h_expected.cend(),
    thrust::make_transform_iterator(h_expected.cbegin(),
                                    [](auto const str) { return str != nullptr; }));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsConvertTest, UrlDecode)
{
  std::vector<const char*> h_strings{"www.nvidia.com/rapids/%3Fp%3D%C3%A9",
                                     "/_file-1234567890.txt",
                                     "a%20b%2Bc~defghijklmnopqrstuvwxyz",
                                     "%25-accent%c3%a9d",
                                     "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                                     "01234567890",
                                     nullptr,
                                     ""};
  cudf::test::strings_column_wrapper strings(
    h_strings.cbegin(),
    h_strings.cend(),
    thrust::make_transform_iterator(h_strings.cbegin(),
                                    [](auto const str) { return str != nullptr; }));

  auto strings_view = cudf::strings_column_view(strings);
  auto results      = cudf::strings::url_decode(strings_view);

  std::vector<const char*> h_expected{"www.nvidia.com/rapids/?p=é",
                                      "/_file-1234567890.txt",
                                      "a b+c~defghijklmnopqrstuvwxyz",
                                      "%-accentéd",
                                      "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                                      "01234567890",
                                      nullptr,
                                      ""};
  cudf::test::strings_column_wrapper expected(
    h_expected.cbegin(),
    h_expected.cend(),
    thrust::make_transform_iterator(h_expected.cbegin(),
                                    [](auto const str) { return str != nullptr; }));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsConvertTest, UrlDecodeNop)
{
  std::vector<const char*> h_strings{"www.nvidia.com/rapids/abc123",
                                     "/_file-1234567890.txt",
                                     "abcdefghijklmnopqrstuvwxyz",
                                     "ABCDEFGHIJKLMNOPQRSTUVWXYZ%",
                                     "0123456789%0",
                                     nullptr,
                                     ""};
  cudf::test::strings_column_wrapper strings(
    h_strings.cbegin(),
    h_strings.cend(),
    thrust::make_transform_iterator(h_strings.cbegin(),
                                    [](auto const str) { return str != nullptr; }));

  auto strings_view = cudf::strings_column_view(strings);
  auto results      = cudf::strings::url_decode(strings_view);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, strings);
}

TEST_F(StringsConvertTest, UrlDecodeSliced)
{
  std::vector<const char*> h_strings{"www.nvidia.com/rapids/%3Fp%3D%C3%A9%",
                                     "01/_file-1234567890.txt",
                                     "a%20b%2Bc~defghijklmnopqrstuvwxyz",
                                     "%25-accent%c3%a9d",
                                     "ABCDEFGHIJKLMNOPQRSTUVWXYZ%0",
                                     "01234567890",
                                     nullptr,
                                     ""};
  cudf::test::strings_column_wrapper strings(
    h_strings.cbegin(),
    h_strings.cend(),
    thrust::make_transform_iterator(h_strings.cbegin(),
                                    [](auto const str) { return str != nullptr; }));

  std::vector<const char*> h_expected{"www.nvidia.com/rapids/?p=é%",
                                      "01/_file-1234567890.txt",
                                      "a b+c~defghijklmnopqrstuvwxyz",
                                      "%-accentéd",
                                      "ABCDEFGHIJKLMNOPQRSTUVWXYZ%0",
                                      "01234567890",
                                      nullptr,
                                      ""};
  cudf::test::strings_column_wrapper expected(
    h_expected.cbegin(),
    h_expected.cend(),
    thrust::make_transform_iterator(h_expected.cbegin(),
                                    [](auto const str) { return str != nullptr; }));

  std::vector<cudf::size_type> slice_indices{0, 3, 3, 6, 6, 8};
  auto sliced_strings  = cudf::slice(strings, slice_indices);
  auto sliced_expected = cudf::slice(expected, slice_indices);
  for (size_t i = 0; i < sliced_strings.size(); ++i) {
    auto strings_view = cudf::strings_column_view(sliced_strings[i]);
    auto results      = cudf::strings::url_decode(strings_view);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, sliced_expected[i]);
  }
}

TEST_F(StringsConvertTest, UrlDecodeLargeStrings)
{
  constexpr int string_len = 35000;
  std::vector<char> string_encoded;
  string_encoded.reserve(string_len * 3);
  std::vector<char> string_plain;
  string_plain.reserve(string_len + 1);

  std::random_device rd;
  std::mt19937 random_number_generator(rd());
  std::uniform_int_distribution<int> distribution(0, 4);

  for (int character_idx = 0; character_idx < string_len; character_idx++) {
    switch (distribution(random_number_generator)) {
      case 0:
        string_encoded.push_back('a');
        string_plain.push_back('a');
        break;
      case 1:
        string_encoded.push_back('b');
        string_plain.push_back('b');
        break;
      case 2:
        string_encoded.push_back('c');
        string_plain.push_back('c');
        break;
      case 3:
        string_encoded.push_back('%');
        string_encoded.push_back('3');
        string_encoded.push_back('F');
        string_plain.push_back('?');
        break;
      case 4:
        string_encoded.push_back('%');
        string_encoded.push_back('3');
        string_encoded.push_back('D');
        string_plain.push_back('=');
        break;
    }
  }
  string_encoded.push_back('\0');
  string_plain.push_back('\0');

  std::vector<const char*> h_strings{string_encoded.data()};
  cudf::test::strings_column_wrapper strings(
    h_strings.cbegin(),
    h_strings.cend(),
    thrust::make_transform_iterator(h_strings.cbegin(),
                                    [](auto const str) { return str != nullptr; }));

  auto strings_view = cudf::strings_column_view(strings);
  auto results      = cudf::strings::url_decode(strings_view);

  std::vector<const char*> h_expected{string_plain.data()};
  cudf::test::strings_column_wrapper expected(
    h_expected.cbegin(),
    h_expected.cend(),
    thrust::make_transform_iterator(h_expected.cbegin(),
                                    [](auto const str) { return str != nullptr; }));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsConvertTest, ZeroSizeUrlStringsColumn)
{
  cudf::column_view zero_size_column(
    cudf::data_type{cudf::type_id::STRING}, 0, nullptr, nullptr, 0);
  auto results = cudf::strings::url_encode(zero_size_column);
  cudf::test::expect_strings_empty(results->view());
  results = cudf::strings::url_decode(zero_size_column);
  cudf::test::expect_strings_empty(results->view());
}
