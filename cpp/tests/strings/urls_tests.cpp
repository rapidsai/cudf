/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/convert/convert_urls.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/strings/utilities.h>

#include <vector>


struct StringsConvertTest : public cudf::test::BaseFixture {};


TEST_F(StringsConvertTest, UrlEncode)
{
    std::vector<const char*> h_strings{ "www.nvidia.com/rapids?p=é", "/_file-7.txt", "a b+c~d",
                                        "e\tfgh\\jklmnopqrstuvwxyz", "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                                        "0123456789", " \t\f\n",
                                         nullptr, "" };
    cudf::test::strings_column_wrapper strings( h_strings.cbegin(), h_strings.cend(),
        thrust::make_transform_iterator( h_strings.cbegin(), [] (auto const str) { return str!=nullptr; }));

    auto strings_view = cudf::strings_column_view(strings);
    auto results = cudf::strings::url_encode(strings_view);

    std::vector<const char*> h_expected{ "www.nvidia.com%2Frapids%3Fp%3D%C3%A9", "%2F_file-7.txt", "a%20b%2Bc~d",
                                         "e%09fgh%5Cjklmnopqrstuvwxyz", "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                                         "0123456789", "%20%09%0C%0A",
                                         nullptr, "" };
    cudf::test::strings_column_wrapper expected( h_expected.cbegin(), h_expected.cend(),
        thrust::make_transform_iterator( h_expected.cbegin(), [] (auto const str) { return str!=nullptr; }));
    cudf::test::expect_columns_equal(*results, expected);
}

TEST_F(StringsConvertTest, UrlDecode)
{
    std::vector<const char*> h_strings{ "www.nvidia.com/rapids/%3Fp%3D%C3%A9", "/_file-1234567890.txt", "a%20b%2Bc~defghijklmnopqrstuvwxyz",
                                        "%25-accent%c3%a9d", "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "01234567890",
                                        nullptr, "" };
    cudf::test::strings_column_wrapper strings( h_strings.cbegin(), h_strings.cend(),
        thrust::make_transform_iterator( h_strings.cbegin(), [] (auto const str) { return str!=nullptr; }));

    auto strings_view = cudf::strings_column_view(strings);
    auto results = cudf::strings::url_decode(strings_view);

    std::vector<const char*> h_expected{ "www.nvidia.com/rapids/?p=é", "/_file-1234567890.txt", "a b+c~defghijklmnopqrstuvwxyz",
                                         "%-accentéd", "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "01234567890",
                                         nullptr, "" };
    cudf::test::strings_column_wrapper expected( h_expected.cbegin(), h_expected.cend(),
        thrust::make_transform_iterator( h_expected.cbegin(), [] (auto const str) { return str!=nullptr; }));
    cudf::test::expect_columns_equal(*results, expected);
}

TEST_F(StringsConvertTest, ZeroSizeUrlStringsColumn)
{
    cudf::column_view zero_size_column( cudf::data_type{cudf::STRING}, 0, nullptr, nullptr, 0);
    auto results = cudf::strings::url_encode(zero_size_column);
    cudf::test::expect_strings_empty(results->view());
    results = cudf::strings::url_decode(zero_size_column);
    cudf::test::expect_strings_empty(results->view());
}

