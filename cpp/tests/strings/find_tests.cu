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
#include <cudf/column/column_factories.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/wrappers/bool.hpp>

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/type_lists.hpp>
#include "./utilities.h"

#include <vector>
#include <gmock/gmock.h>


struct StringsFindTest : public cudf::test::BaseFixture {};

TEST_F(StringsFindTest, Find)
{
    std::vector<const char*> h_strings{ "Héllo", "thesé", nullptr, "lease", "tést strings", "" };
    cudf::test::strings_column_wrapper strings( h_strings.begin(), h_strings.end(),
        thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));
    auto strings_view = cudf::strings_column_view(strings);

    {
        cudf::test::fixed_width_column_wrapper<int32_t> expected( {1,4,-1,-1,1,-1}, {1,1,0,1,1,1} );
        auto results = cudf::strings::find(strings_view, cudf::string_scalar("é"));
        cudf::test::expect_columns_equal(*results, expected);
    }
    {
        cudf::test::fixed_width_column_wrapper<int32_t> expected( {3,-1,-1,0,-1,-1}, {1,1,0,1,1,1} );
        auto results = cudf::strings::rfind(strings_view, cudf::string_scalar("l"));
        cudf::test::expect_columns_equal(*results, expected);
    }
    {
        cudf::test::fixed_width_column_wrapper<cudf::experimental::bool8> expected( {0,1,0,1,0,0}, {1,1,0,1,1,1} );
        auto results = cudf::strings::contains(strings_view, cudf::string_scalar("e"));
        cudf::test::expect_columns_equal(*results, expected);
    }
    {
        cudf::test::fixed_width_column_wrapper<cudf::experimental::bool8> expected( {0,1,0,0,1,0}, {1,1,0,1,1,1} );
        auto results = cudf::strings::starts_with(strings_view, cudf::string_scalar("t"));
        cudf::test::expect_columns_equal(*results, expected);
    }
    {
        cudf::test::fixed_width_column_wrapper<cudf::experimental::bool8> expected( {0,0,0,1,0,0}, {1,1,0,1,1,1} );
        auto results = cudf::strings::ends_with(strings_view, cudf::string_scalar("se"));
        cudf::test::expect_columns_equal(*results, expected);
    }
    {
        cudf::test::fixed_width_column_wrapper<cudf::experimental::bool8> expected( {0,1,0,0,0,0}, {1,1,0,1,1,1} );
        auto results = cudf::strings::starts_with(strings_view, cudf::string_scalar("thesé"));
        cudf::test::expect_columns_equal(*results, expected);
    }
    {
        cudf::test::fixed_width_column_wrapper<cudf::experimental::bool8> expected( {0,1,0,0,0,0}, {1,1,0,1,1,1} );
        auto results = cudf::strings::ends_with(strings_view, cudf::string_scalar("thesé"));
        cudf::test::expect_columns_equal(*results, expected);
    }
}

TEST_F(StringsFindTest, ZeroSizeStringsColumn)
{
    cudf::column_view zero_size_strings_column( cudf::data_type{cudf::STRING}, 0, nullptr, nullptr, 0);
    auto strings_view = cudf::strings_column_view(zero_size_strings_column);
    auto results = cudf::strings::find(strings_view, cudf::string_scalar("é"));
    EXPECT_EQ(results->size(),0);
    results = cudf::strings::rfind(strings_view, cudf::string_scalar("é"));
    EXPECT_EQ(results->size(),0);
    results = cudf::strings::contains(strings_view, cudf::string_scalar("é"));
    EXPECT_EQ(results->size(),0);
    results = cudf::strings::starts_with(strings_view, cudf::string_scalar("é"));
    EXPECT_EQ(results->size(),0);
    results = cudf::strings::ends_with(strings_view, cudf::string_scalar("é"));
    EXPECT_EQ(results->size(),0);
}

TEST_F(StringsFindTest, AllEmpty)
{
    std::vector<std::string> h_strings{ "", "", "", "", "" };
    cudf::test::strings_column_wrapper strings( h_strings.begin(), h_strings.end() );

    std::vector<int32_t> h_expected32(h_strings.size(),-1);
    cudf::test::fixed_width_column_wrapper<int32_t> expected32( h_expected32.begin(), h_expected32.end() );

    std::vector<cudf::experimental::bool8> h_expected8(h_strings.size(),0);
    cudf::test::fixed_width_column_wrapper<cudf::experimental::bool8> expected8( h_expected8.begin(), h_expected8.end() );

    auto strings_view = cudf::strings_column_view(strings);
    auto results = cudf::strings::find(strings_view,cudf::string_scalar("e"));
    cudf::test::expect_columns_equal(*results,expected32);
    results = cudf::strings::rfind(strings_view,cudf::string_scalar("e"));
    cudf::test::expect_columns_equal(*results,expected32);
    results = cudf::strings::contains(strings_view,cudf::string_scalar("e"));
    cudf::test::expect_columns_equal(*results,expected8);
    results = cudf::strings::starts_with(strings_view,cudf::string_scalar("e"));
    cudf::test::expect_columns_equal(*results,expected8);
    results = cudf::strings::ends_with(strings_view,cudf::string_scalar("e"));
    cudf::test::expect_columns_equal(*results,expected8);
}

TEST_F(StringsFindTest, AllNull)
{
    std::vector<const char*> h_strings{ nullptr, nullptr, nullptr, nullptr };
    cudf::test::strings_column_wrapper strings( h_strings.begin(), h_strings.end(),
        thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));

    std::vector<int32_t> h_expected32(h_strings.size(),-1);
    cudf::test::fixed_width_column_wrapper<int32_t> expected32( h_expected32.begin(), h_expected32.end(),
        thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));

    std::vector<cudf::experimental::bool8> h_expected8(h_strings.size(),-1);
    cudf::test::fixed_width_column_wrapper<cudf::experimental::bool8> expected8( h_expected8.begin(), h_expected8.end(),
        thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));

    auto strings_view = cudf::strings_column_view(strings);
    auto results = cudf::strings::find(strings_view,cudf::string_scalar("e"));
    cudf::test::expect_columns_equal(*results,expected32);
    results = cudf::strings::rfind(strings_view,cudf::string_scalar("e"));
    cudf::test::expect_columns_equal(*results,expected32);
    results = cudf::strings::contains(strings_view,cudf::string_scalar("e"));
    cudf::test::expect_columns_equal(*results,expected8);
    results = cudf::strings::starts_with(strings_view,cudf::string_scalar("e"));
    cudf::test::expect_columns_equal(*results,expected8);
    results = cudf::strings::ends_with(strings_view,cudf::string_scalar("e"));
    cudf::test::expect_columns_equal(*results,expected8);
}

class FindParmsTest : public StringsFindTest,
                      public testing::WithParamInterface<int32_t> {};

TEST_P(FindParmsTest, Find)
{
    std::vector<std::string> h_strings{ "hello", "", "these", "are stl", "safe" };
    cudf::test::strings_column_wrapper strings( h_strings.begin(), h_strings.end() );
    cudf::size_type position = GetParam();

    auto strings_view = cudf::strings_column_view(strings);
    {
        auto results = cudf::strings::find(strings_view,cudf::string_scalar("e"),position);
        std::vector<int32_t> h_expected;
        for( auto itr = h_strings.begin(); itr != h_strings.end(); ++itr )
            h_expected.push_back((int32_t)(*itr).find("e",position));
        cudf::test::fixed_width_column_wrapper<int32_t> expected( h_expected.begin(), h_expected.end() );
        cudf::test::expect_columns_equal(*results,expected);
    }
    {
        auto results = cudf::strings::rfind(strings_view,cudf::string_scalar("e"),0,position+1);
        std::vector<int32_t> h_expected;
        for( auto itr = h_strings.begin(); itr != h_strings.end(); ++itr )
            h_expected.push_back((int32_t)(*itr).rfind("e",position));
        cudf::test::fixed_width_column_wrapper<int32_t> expected( h_expected.begin(), h_expected.end() );
        cudf::test::expect_columns_equal(*results,expected);
    }
}

INSTANTIATE_TEST_CASE_P(StringFindTest, FindParmsTest,
                        testing::ValuesIn(std::array<int32_t,4>{0,1,2,3}));
