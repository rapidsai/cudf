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
#include "./utilities.h"

#include <gmock/gmock.h>
#include <vector>


struct StringsContainsTests : public cudf::test::BaseFixture {};


TEST_F(StringsContainsTests, ContainsTest)
{
    std::vector<const char*> h_strings{
        "The quick brown @fox jumps", "ovér the", "lazy @dog",
        "1234", "00:0:00", nullptr, "" };
    cudf::test::strings_column_wrapper strings( h_strings.begin(), h_strings.end(),
        thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));

    auto strings_view = cudf::strings_column_view(strings);
    {
        auto results = cudf::strings::contains_re(strings_view,"é");
        int8_t h_expected[] = {0,1,0,0,0,0,0};
        cudf::test::fixed_width_column_wrapper<cudf::experimental::bool8> expected( h_expected, h_expected+h_strings.size(),
            thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));
        cudf::test::expect_columns_equal(*results,expected);
    }
    {
        auto results = cudf::strings::contains_re(strings_view,"\\d+");
        int8_t h_expected[] = {0,0,0,1,1,0,0};
        cudf::test::fixed_width_column_wrapper<cudf::experimental::bool8> expected( h_expected, h_expected+h_strings.size(),
            thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));
        cudf::test::expect_columns_equal(*results,expected);
    }
    {
        auto results = cudf::strings::contains_re(strings_view,"@\\w+");
        int8_t h_expected[] = {1,0,1,0,0,0,0};
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
        int8_t h_expected[] = {0,0,1,0,0,0,0};
        cudf::test::fixed_width_column_wrapper<cudf::experimental::bool8> expected( h_expected, h_expected+h_strings.size(),
            thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));
        cudf::test::expect_columns_equal(*results,expected);
    }
    {
        auto results = cudf::strings::matches_re(strings_view,"\\d+");
        int8_t h_expected[] = {0,0,0,1,1,0,0};
        cudf::test::fixed_width_column_wrapper<cudf::experimental::bool8> expected( h_expected, h_expected+h_strings.size(),
            thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));
        cudf::test::expect_columns_equal(*results,expected);
    }
    {
        auto results = cudf::strings::matches_re(strings_view,"@\\w+");
        int8_t h_expected[] = {0,0,0,0,0,0,0};
        cudf::test::fixed_width_column_wrapper<cudf::experimental::bool8> expected( h_expected, h_expected+h_strings.size(),
            thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));
        cudf::test::expect_columns_equal(*results,expected);
    }
}
