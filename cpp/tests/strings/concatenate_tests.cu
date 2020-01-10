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

#include <cudf/column/column_factories.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/detail/concatenate.hpp>

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/column_utilities.hpp>
#include "./utilities.h"

#include <vector>
#include <gmock/gmock.h>


struct StringsConcatenateTest : public cudf::test::BaseFixture {};

TEST_F(StringsConcatenateTest, Concatenate)
{
    std::vector<const char*> h_strings{ "aaa", "bb", "", "cccc", "d", "ééé", "ff", "gggg", "", "h", "iiii", "jjj", "k", "lllllll", "mmmmm", "n", "oo", "ppp" };
    cudf::test::strings_column_wrapper strings1( h_strings.data(), h_strings.data()+6 );
    cudf::test::strings_column_wrapper strings2( h_strings.data()+6, h_strings.data()+10 );
    cudf::test::strings_column_wrapper strings3( h_strings.data()+10, h_strings.data()+h_strings.size() );
    cudf::column_view zero_size_strings_column( cudf::data_type{cudf::STRING}, 0, nullptr, nullptr, 0);

    std::vector<cudf::strings_column_view> strings_columns;
    strings_columns.push_back(cudf::strings_column_view(strings1));
    strings_columns.push_back(cudf::strings_column_view(zero_size_strings_column));
    strings_columns.push_back(cudf::strings_column_view(strings2));
    strings_columns.push_back(cudf::strings_column_view(strings3));

    auto results = cudf::strings::detail::concatenate(strings_columns);

    cudf::test::strings_column_wrapper expected( h_strings.begin(), h_strings.end() );
    cudf::test::expect_columns_equal(*results,expected);
}

TEST_F(StringsConcatenateTest, ZeroSizeStringsColumns)
{
    cudf::column_view zero_size_strings_column( cudf::data_type{cudf::STRING}, 0, nullptr, nullptr, 0);
    std::vector<cudf::strings_column_view> strings_columns;
    strings_columns.push_back(cudf::strings_column_view(zero_size_strings_column));
    strings_columns.push_back(cudf::strings_column_view(zero_size_strings_column));
    strings_columns.push_back(cudf::strings_column_view(zero_size_strings_column));
    auto results = cudf::strings::detail::concatenate(strings_columns);
    cudf::test::expect_strings_empty(results->view());
}

TEST_F(StringsConcatenateTest, ZeroSizeStringsPlusNormal)
{
    cudf::column_view zero_size_strings_column( cudf::data_type{cudf::STRING}, 0, nullptr, nullptr, 0);
    std::vector<cudf::strings_column_view> strings_columns;
    strings_columns.push_back(cudf::strings_column_view(zero_size_strings_column));

    std::vector<const char*> h_strings{ "aaa", "bb", "", "cccc", "d", "ééé", "ff", "gggg", "", "h", "iiii", "jjj", "k", "lllllll", "mmmmm", "n", "oo", "ppp" };
    cudf::test::strings_column_wrapper strings1( h_strings.data(), h_strings.data()+h_strings.size() );
    strings_columns.push_back(cudf::strings_column_view(strings1));

    auto results = cudf::strings::detail::concatenate(strings_columns);
    cudf::test::expect_columns_equal(*results,strings1);
}
