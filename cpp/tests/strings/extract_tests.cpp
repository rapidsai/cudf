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
#include <cudf/strings/extract.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/table_utilities.hpp>
#include <tests/strings/utilities.h>

#include <gmock/gmock.h>
#include <vector>


struct StringsExtractTests : public cudf::test::BaseFixture {};


TEST_F(StringsExtractTests, ExtractTest)
{
    std::vector<const char*> h_strings{ "First Last", "Joe Schmoe", "John Smith", "Jane Smith", "Beyonce", "Sting", nullptr, "" };

    cudf::test::strings_column_wrapper strings( h_strings.begin(), h_strings.end(),
        thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));
    auto strings_view = cudf::strings_column_view(strings);

    std::vector<const char*> h_expecteds{ "First", "Joe", "John", "Jane", nullptr, nullptr, nullptr, nullptr,
                                          "Last", "Schmoe", "Smith", "Smith", nullptr, nullptr, nullptr, nullptr };

    std::string pattern = "(\\w+) (\\w+)";
    auto results = cudf::strings::extract(strings_view,pattern);

    cudf::test::strings_column_wrapper expected1( h_expecteds.data(), h_expecteds.data() + h_strings.size(),
        thrust::make_transform_iterator( h_expecteds.begin(), [] (auto str) { return str!=nullptr; }));
    cudf::test::strings_column_wrapper expected2( h_expecteds.data()+h_strings.size(), h_expecteds.data() + h_expecteds.size(),
        thrust::make_transform_iterator( h_expecteds.data()+h_strings.size(), [] (auto str) { return str!=nullptr; }));
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(expected1.release());
    columns.push_back(expected2.release());
    cudf::experimental::table expected(std::move(columns));
    cudf::test::expect_tables_equal(*results,expected);
}


TEST_F(StringsExtractTests, MediumRegex)
{
    // This results in 95 regex instructions and falls in the 'medium' range.
    std::string medium_regex = "hello @abc @def (world) The quick brown @fox jumps over the lazy @dog hello http://www.world.com";

    std::vector<const char*> h_strings{
        "hello @abc @def world The quick brown @fox jumps over the lazy @dog hello http://www.world.com thats all",
        "12345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890",
        "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz"
    };
    cudf::test::strings_column_wrapper strings( h_strings.begin(), h_strings.end(),
        thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));

    auto strings_view = cudf::strings_column_view(strings);
    auto results = cudf::strings::extract(strings_view, medium_regex);
    std::vector<const char*> h_expected{"world", nullptr, nullptr };
    cudf::test::strings_column_wrapper expected( h_expected.begin(), h_expected.end(),
        thrust::make_transform_iterator( h_expected.begin(), [] (auto str) { return str!=nullptr; }));
    cudf::test::expect_columns_equal(results->get_column(0),expected);
}

TEST_F(StringsExtractTests, LargeRegex)
{
    // This results in 115 regex instructions and falls in the 'large' range.
    std::string large_regex = "hello @abc @def world The (quick) brown @fox jumps over the lazy @dog hello http://www.world.com I'm here @home zzzz";

    std::vector<const char*> h_strings{
        "hello @abc @def world The quick brown @fox jumps over the lazy @dog hello http://www.world.com I'm here @home zzzz",
        "12345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890",
        "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz"
    };
    cudf::test::strings_column_wrapper strings( h_strings.begin(), h_strings.end(),
        thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));

    auto strings_view = cudf::strings_column_view(strings);
    auto results = cudf::strings::extract(strings_view, large_regex);
    std::vector<const char*> h_expected{"quick", nullptr, nullptr };
    cudf::test::strings_column_wrapper expected( h_expected.begin(), h_expected.end(),
        thrust::make_transform_iterator( h_expected.begin(), [] (auto str) { return str!=nullptr; }));
    cudf::test::expect_columns_equal(results->get_column(0),expected);
}
