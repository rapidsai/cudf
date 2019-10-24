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

#include <cudf/column/column.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/substring.hpp>

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/column_utilities.hpp>
#include "./utilities.h"

#include <string>
#include <vector>
#include <gmock/gmock.h>


struct StringsSubstringsTest : public cudf::test::BaseFixture {};

TEST_F(StringsSubstringsTest, Substring)
{
    std::vector<const char*> h_strings{ "Héllo", "thesé", nullptr, "ARE THE", "tést strings", "" };
    cudf::test::strings_column_wrapper strings( h_strings.begin(), h_strings.end(),
        thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));
    std::vector<const char*> h_expected({ "llo", "esé", nullptr, "E T", "st ", "" });
    cudf::test::strings_column_wrapper expected( h_expected.begin(), h_expected.end(),
        thrust::make_transform_iterator( h_expected.begin(), [] (auto str) { return str!=nullptr; }));

    auto strings_view = cudf::strings_column_view(strings);
    auto results = cudf::strings::substring(strings_view, 2, 5);
    cudf::test::expect_columns_equal(*results, expected);
}

class SubstringParmsTest : public StringsSubstringsTest,
                           public testing::WithParamInterface<int32_t> {};

TEST_P(SubstringParmsTest, Substring)
{
    std::vector<std::string> h_strings{ "basic strings", "that can", "be used", "with STL" };
    cudf::size_type start = GetParam();

    cudf::test::strings_column_wrapper strings( h_strings.begin(), h_strings.end() );
    auto strings_view = cudf::strings_column_view(strings);
    auto results = cudf::strings::substring(strings_view,start);

    std::vector<std::string> h_expected;
    for( auto itr = h_strings.begin(); itr != h_strings.end(); ++itr )
        h_expected.push_back( (*itr).substr(start) );

    cudf::test::strings_column_wrapper expected( h_expected.begin(), h_expected.end() );
    cudf::test::expect_columns_equal(*results,expected);
}

TEST_P(SubstringParmsTest, AllEmpty)
{
    std::vector<std::string> h_strings{ "", "", "", "" };
    cudf::size_type start = GetParam();

    cudf::test::strings_column_wrapper strings( h_strings.begin(), h_strings.end() );
    auto strings_view = cudf::strings_column_view(strings);
    auto results = cudf::strings::substring(strings_view,start);

    std::vector<std::string> h_expected(h_strings);
    cudf::test::strings_column_wrapper expected( h_expected.begin(), h_expected.end() );
    cudf::test::expect_columns_equal(*results,expected);
}

TEST_P(SubstringParmsTest, AllNulls)
{
    std::vector<const char*> h_strings{ nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr };
    cudf::test::strings_column_wrapper strings( h_strings.begin(), h_strings.end(),
        thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));
    cudf::size_type start = GetParam();

    auto strings_view = cudf::strings_column_view(strings);
    auto results = cudf::strings::substring(strings_view,start);

    std::vector<const char*> h_expected(h_strings);
    cudf::test::strings_column_wrapper expected( h_strings.begin(), h_strings.end(),
        thrust::make_transform_iterator( h_expected.begin(), [] (auto str) { return str!=nullptr; }));
    cudf::test::expect_columns_equal(*results,expected);
}

INSTANTIATE_TEST_CASE_P(StringsSubstringsTest, SubstringParmsTest,
                        testing::ValuesIn(std::array<cudf::size_type,3>{1,2,3}));

TEST_F(StringsSubstringsTest, ZeroSizeStringsColumn)
{
    cudf::column_view zero_size_strings_column( cudf::data_type{cudf::STRING}, 0, nullptr, nullptr, 0);
    auto strings_view = cudf::strings_column_view(zero_size_strings_column);
    auto results = cudf::strings::substring(strings_view,1,2);
    cudf::test::expect_strings_empty(results->view());
}
