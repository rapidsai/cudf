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
#include <cudf/strings/translate.hpp>
#include <cudf/utilities/error.hpp>

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/strings/utilities.h>

#include <vector>


struct StringsTranslateTest : public cudf::test::BaseFixture {};

std::pair<cudf::char_utf8,cudf::char_utf8> make_entry( const char* from, const char* to )
{
    cudf::char_utf8 in = 0;
    cudf::char_utf8 out = 0;
    cudf::strings::detail::to_char_utf8(from, in);
    if( to )
        cudf::strings::detail::to_char_utf8(to, out);
    return std::make_pair(in,out);
}

TEST_F(StringsTranslateTest, Translate)
{
    std::vector<const char*> h_strings{ "eee ddd", "bb cc", nullptr, "", "aa", "débd" };
    cudf::test::strings_column_wrapper strings( h_strings.begin(), h_strings.end(),
        thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));
    auto strings_view = cudf::strings_column_view(strings);

    std::vector< std::pair<cudf::char_utf8,cudf::char_utf8> > translate_table{make_entry("b",0),
                                                                              make_entry("a","A"),
                                                                              make_entry("é","E"),
                                                                              make_entry("e","_")};
    auto results = cudf::strings::translate(strings_view,translate_table);

    std::vector<const char*> h_expected{ "___ ddd", " cc", nullptr, "", "AA", "dEd" };
    cudf::test::strings_column_wrapper expected( h_expected.begin(), h_expected.end(),
        thrust::make_transform_iterator( h_expected.begin(), [] (auto str) { return str!=nullptr; }));
    cudf::test::expect_columns_equal(*results,expected);
}

TEST_F(StringsTranslateTest, ZeroSizeStringsColumn)
{
    cudf::column_view zero_size_strings_column( cudf::data_type{cudf::STRING}, 0, nullptr, nullptr, 0);
    auto strings_view = cudf::strings_column_view(zero_size_strings_column);
    std::vector< std::pair<cudf::char_utf8,cudf::char_utf8> > translate_table;
    auto results = cudf::strings::translate(strings_view,translate_table);
    cudf::test::expect_strings_empty(results->view());
}

