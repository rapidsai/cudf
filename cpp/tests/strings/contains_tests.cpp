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
#include <cudf/strings/contains.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/strings/utilities.h>

#include <gmock/gmock.h>
#include <vector>


struct StringsContainsTests : public cudf::test::BaseFixture {};


TEST_F(StringsContainsTests, ContainsTest)
{
    std::vector<const char*> h_strings{
        "5",
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
         nullptr, "" };

    cudf::test::strings_column_wrapper strings( h_strings.begin(), h_strings.end(),
        thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));
    auto strings_view = cudf::strings_column_view(strings);

    std::vector<std::string> patterns{
        "\\d",
        "\\w+",
        "\\s",
        "\\S",
        "^.*\\\\.*$",
        "[1-5]+",
        "[a-h]+",
        "[A-H]+",
        "\n",
        "b.\\s*\n",
        ".*c",
        "\\d\\d:\\d\\d:\\d\\d",
        "\\d\\d?:\\d\\d?:\\d\\d?",
        "[Hh]ello [Ww]orld",
        "\\bworld\\b" };

    std::vector<cudf::experimental::bool8> h_expecteds{ // strings.size x patterns.size
        true, false, false, true, false, false, false, true, true, true, true, true, true, true, true, true, true, false, false, false, true, true, false, false, false, false, false, false, false, false, false, false,
        true, true, false, true, false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false,
        false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, true, true, false, false, false, false, false, false, true, true, true, false, false, false,
        true, true, false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false,
        false, false, false, false, true, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
        true, false, false, true, false, false, false, true, true, true, false, false, false, false, false, false, false, false, false, false, true, true, false, false, false, false, false, false, false, false, false, false,
        false, true, false, false, false, true, true, false, true, false, false, false, false, false, false, false, false, true, true, true, false, false, true, true, false, true, true, true, true, true, false, false,
        false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, true, true, true, false, true, false, false, true, false, false, false, false, false, false, false,
        false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, true, false, false, false,
        false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, true, false, false, false,
        false, false, false, false, false, false, true, false, true, false, false, false, false, false, false, false, false, false, false, true, false, false, false, true, false, true, true, true, true, true, false, false,
        false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
        false, false, false, false, false, false, false, false, false, false, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
        false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false,
        false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false
    };

    for( int idx=0; idx < static_cast<int>(patterns.size()); ++idx )
    {
        std::string ptn = patterns[idx];
        auto results = cudf::strings::contains_re(strings_view,ptn);
        cudf::experimental::bool8* h_expected = h_expecteds.data() + (idx * h_strings.size());
        cudf::test::fixed_width_column_wrapper<cudf::experimental::bool8> expected( h_expected, h_expected+h_strings.size(),
            thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));
        cudf::test::expect_columns_equal(*results,expected);
    }
}


TEST_F(StringsContainsTests, MatchesTest)
{
    std::vector<const char*> h_strings{
        "The quick brown @fox jumps", "ovér the", "lazy @dog", "1234", "00:0:00", nullptr, "" };
    cudf::test::strings_column_wrapper strings( h_strings.begin(), h_strings.end(),
        thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));

    auto strings_view = cudf::strings_column_view(strings);
    {
        auto results = cudf::strings::matches_re(strings_view,"lazy");
        cudf::experimental::bool8 h_expected[] = {false,false,true,false,false,false,false};
        cudf::test::fixed_width_column_wrapper<cudf::experimental::bool8> expected( h_expected, h_expected+h_strings.size(),
            thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));
        cudf::test::expect_columns_equal(*results,expected);
    }
    {
        auto results = cudf::strings::matches_re(strings_view,"\\d+");
        cudf::experimental::bool8 h_expected[] = {false,false,false,true,true,false,false};
        cudf::test::fixed_width_column_wrapper<cudf::experimental::bool8> expected( h_expected, h_expected+h_strings.size(),
            thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));
        cudf::test::expect_columns_equal(*results,expected);
    }
    {
        auto results = cudf::strings::matches_re(strings_view,"@\\w+");
        cudf::experimental::bool8 h_expected[] = {false,false,false,false,false,false,false};
        cudf::test::fixed_width_column_wrapper<cudf::experimental::bool8> expected( h_expected, h_expected+h_strings.size(),
            thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));
        cudf::test::expect_columns_equal(*results,expected);
    }
}

TEST_F(StringsContainsTests, CountTest)
{
    std::vector<const char*> h_strings{
        "The quick brown @fox jumps ovér the", "lazy @dog", "1:2:3:4", "00:0:00", nullptr, "" };
    cudf::test::strings_column_wrapper strings( h_strings.begin(), h_strings.end(),
        thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));

    auto strings_view = cudf::strings_column_view(strings);
    {
        auto results = cudf::strings::count_re(strings_view,"[tT]he");
        int32_t h_expected[] = {2,0,0,0,0,0};
        cudf::test::fixed_width_column_wrapper<int32_t> expected( h_expected, h_expected+h_strings.size(),
            thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));
        cudf::test::expect_columns_equal(*results,expected);
    }
    {
        auto results = cudf::strings::count_re(strings_view,"@\\w+");
        int32_t h_expected[] = {1,1,0,0,0,0};
        cudf::test::fixed_width_column_wrapper<int32_t> expected( h_expected, h_expected+h_strings.size(),
            thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));
        cudf::test::expect_columns_equal(*results,expected);
    }
    {
        auto results = cudf::strings::count_re(strings_view,"\\d+:\\d+");
        int32_t h_expected[] = {0,0,2,1,0,0};
        cudf::test::fixed_width_column_wrapper<int32_t> expected( h_expected, h_expected+h_strings.size(),
            thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));
        cudf::test::expect_columns_equal(*results,expected);
    }
}

TEST_F(StringsContainsTests, MediumRegex)
{
    // This results in 95 regex instructions and falls in the 'medium' range.
    std::string medium_regex = "hello @abc @def world The quick brown @fox jumps over the lazy @dog hello http://www.world.com";

    std::vector<const char*> h_strings{
        "hello @abc @def world The quick brown @fox jumps over the lazy @dog hello http://www.world.com thats all",
        "12345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890",
        "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz"
    };
    cudf::test::strings_column_wrapper strings( h_strings.begin(), h_strings.end(),
        thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));

    auto strings_view = cudf::strings_column_view(strings);
    {
        auto results = cudf::strings::contains_re(strings_view, medium_regex);
        cudf::experimental::bool8 h_expected[] = {true, false, false};
        cudf::test::fixed_width_column_wrapper<cudf::experimental::bool8> expected( h_expected, h_expected+h_strings.size(),
            thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));
        cudf::test::expect_columns_equal(*results,expected);
    }
    {
        auto results = cudf::strings::matches_re(strings_view, medium_regex);
        cudf::experimental::bool8 h_expected[] = {true, false, false};
        cudf::test::fixed_width_column_wrapper<cudf::experimental::bool8> expected( h_expected, h_expected+h_strings.size(),
            thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));
        cudf::test::expect_columns_equal(*results,expected);
    }
    {
        auto results = cudf::strings::count_re(strings_view, medium_regex);
        int32_t h_expected[] = {1,0,0};
        cudf::test::fixed_width_column_wrapper<int32_t> expected( h_expected, h_expected+h_strings.size(),
            thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));
        cudf::test::expect_columns_equal(*results,expected);
    }
}

TEST_F(StringsContainsTests, LargeRegex)
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
    {
        auto results = cudf::strings::contains_re(strings_view, large_regex);
        cudf::experimental::bool8 h_expected[] = {true, false, false};
        cudf::test::fixed_width_column_wrapper<cudf::experimental::bool8> expected( h_expected, h_expected+h_strings.size(),
            thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));
        cudf::test::expect_columns_equal(*results,expected);
    }
    {
        auto results = cudf::strings::matches_re(strings_view, large_regex);
        cudf::experimental::bool8 h_expected[] = {true, false, false};
        cudf::test::fixed_width_column_wrapper<cudf::experimental::bool8> expected( h_expected, h_expected+h_strings.size(),
            thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));
        cudf::test::expect_columns_equal(*results,expected);
    }
    {
        auto results = cudf::strings::count_re(strings_view, large_regex);
        int32_t h_expected[] = {1,0,0};
        cudf::test::fixed_width_column_wrapper<int32_t> expected( h_expected, h_expected+h_strings.size(),
            thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));
        cudf::test::expect_columns_equal(*results,expected);
    }
}
