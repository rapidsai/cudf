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
#include <cudf/strings/findall.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/table_utilities.hpp>
#include <tests/strings/utilities.h>

#include <gmock/gmock.h>
#include <vector>


struct StringsFindallTests : public cudf::test::BaseFixture {};


TEST_F(StringsFindallTests, FindallTest)
{
    std::vector<const char*> h_strings{ "First Last", "Joe Schmoe", "John Smith", "Jane Smith", "Beyonce", "Sting", nullptr, "" };

    cudf::test::strings_column_wrapper strings( h_strings.begin(), h_strings.end(),
        thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));
    auto strings_view = cudf::strings_column_view(strings);

    std::vector<const char*> h_expecteds{ "First", "Joe", "John", "Jane", "Beyonce", "Sting", nullptr, nullptr,
                                          "Last", "Schmoe", "Smith", "Smith", nullptr, nullptr, nullptr, nullptr };

    std::string pattern = "(\\w+)";
    auto results = cudf::strings::findall_re(strings_view,pattern);
    EXPECT_TRUE( results->num_columns()==2 );

    cudf::test::strings_column_wrapper expected1( h_expecteds.data(), h_expecteds.data() + h_strings.size(),
        thrust::make_transform_iterator( h_expecteds.begin(), [] (auto str) { return str!=nullptr; }));
    cudf::test::strings_column_wrapper expected2( h_expecteds.data()+h_strings.size(), h_expecteds.data() + h_expecteds.size(),
        thrust::make_transform_iterator( h_expecteds.data()+h_strings.size(), [] (auto str) { return str!=nullptr; }));
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back( expected1.release() );
    columns.push_back( expected2.release() );
    cudf::experimental::table expected(std::move(columns));
    cudf::test::expect_tables_equal(*results,expected);
}

TEST_F(StringsFindallTests, MediumRegex)
{
    // This results in 15 regex instructions and falls in the 'medium' range.
    std::string medium_regex = "(\\w+) (\\w+) (\\d+)";

    std::vector<const char*> h_strings{ "first words 1234 and just numbers 9876", "neither" };
    cudf::test::strings_column_wrapper strings( h_strings.begin(), h_strings.end(),
        thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));

    auto strings_view = cudf::strings_column_view(strings);
    auto results = cudf::strings::findall_re(strings_view, medium_regex);
    EXPECT_TRUE( results->num_columns()==2 );

    std::vector<const char*> h_expected1{"first words 1234", nullptr };
    cudf::test::strings_column_wrapper expected1( h_expected1.begin(), h_expected1.end(),
        thrust::make_transform_iterator( h_expected1.begin(), [] (auto str) { return str!=nullptr; }));
    cudf::test::expect_columns_equal(results->get_column(0),expected1);

    std::vector<const char*> h_expected2{"just numbers 9876", nullptr };
    cudf::test::strings_column_wrapper expected2( h_expected2.begin(), h_expected2.end(),
        thrust::make_transform_iterator( h_expected2.begin(), [] (auto str) { return str!=nullptr; }));
    cudf::test::expect_columns_equal(results->get_column(1),expected2);
}

TEST_F(StringsFindallTests, LargeRegex)
{
    // This results in 115 regex instructions and falls in the 'large' range.
    std::string large_regex = "hello @abc @def world The quick brown @fox jumps over the lazy @dog hello http://www.world.com I'm here @home zzzz";

    std::vector<const char*> h_strings{
        "hello @abc @def world The quick brown @fox jumps over the lazy @dog hello http://www.world.com I'm here @home zzzz",
        "12345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890",
        "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz"
    };
    cudf::test::strings_column_wrapper strings( h_strings.begin(), h_strings.end(),
        thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));

    auto strings_view = cudf::strings_column_view(strings);
    auto results = cudf::strings::findall_re(strings_view, large_regex);
    EXPECT_TRUE( results->num_columns()==1 );

    std::vector<const char*> h_expected{large_regex.c_str(), nullptr, nullptr };
    cudf::test::strings_column_wrapper expected( h_expected.begin(), h_expected.end(),
        thrust::make_transform_iterator( h_expected.begin(), [] (auto str) { return str!=nullptr; }));
    cudf::test::expect_columns_equal(results->get_column(0),expected);
}
