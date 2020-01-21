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
#include <cudf/strings/padding.hpp>
#include <cudf/utilities/error.hpp>

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/column_utilities.hpp>
#include "./utilities.h"

#include <vector>
#include <gmock/gmock.h>


struct StringsPadTest : public cudf::test::BaseFixture {};

TEST_F(StringsPadTest, Padding)
{
    std::vector<const char*> h_strings{ "eee ddd", "bb cc", nullptr, "", "aa", "bbb", "ééé", "o" };
    cudf::test::strings_column_wrapper strings( h_strings.begin(), h_strings.end(),
        thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));
    cudf::size_type width = 6;
    std::string phil = "+";
    auto strings_view = cudf::strings_column_view(strings);

    {
        auto results = cudf::strings::pad(strings_view, width, cudf::strings::pad_side::right, phil);

        std::vector<const char*> h_expected{ "eee ddd", "bb cc+", nullptr, "++++++", "aa++++", "bbb+++", "ééé+++", "o+++++" };
        cudf::test::strings_column_wrapper expected( h_expected.begin(), h_expected.end(),
             thrust::make_transform_iterator( h_expected.begin(), [] (auto str) { return str!=nullptr; }));
        cudf::test::expect_columns_equal(*results,expected);
    }
    {
        auto results = cudf::strings::pad(strings_view, width, cudf::strings::pad_side::left, phil);

        std::vector<const char*> h_expected{ "eee ddd", "+bb cc", nullptr, "++++++", "++++aa", "+++bbb", "+++ééé", "+++++o" };
        cudf::test::strings_column_wrapper expected( h_expected.begin(), h_expected.end(),
             thrust::make_transform_iterator( h_expected.begin(), [] (auto str) { return str!=nullptr; }));
        cudf::test::expect_columns_equal(*results,expected);
    }
    {
        auto results = cudf::strings::pad(strings_view, width, cudf::strings::pad_side::both, phil);
        
        std::vector<const char*> h_expected{ "eee ddd", "bb cc+", nullptr, "++++++", "++aa++", "+bbb++", "+ééé++", "++o+++" };
        cudf::test::strings_column_wrapper expected( h_expected.begin(), h_expected.end(),
             thrust::make_transform_iterator( h_expected.begin(), [] (auto str) { return str!=nullptr; }));
        cudf::test::expect_columns_equal(*results,expected);
    }
}

TEST_F(StringsPadTest, ZeroSizeStringsColumn)
{
    cudf::column_view zero_size_strings_column( cudf::data_type{cudf::STRING}, 0, nullptr, nullptr, 0);
    auto strings_view = cudf::strings_column_view(zero_size_strings_column);
    auto results = cudf::strings::pad(strings_view,5);
    cudf::test::expect_strings_empty(results->view());
}

class StringsPadParmsTest : public StringsPadTest,
                            public testing::WithParamInterface<cudf::size_type> {};

TEST_P(StringsPadParmsTest, Padding)
{
    std::vector<std::string> h_strings{ "eee ddd", "bb cc", "aa", "bbb", "fff", "", "o" };
    cudf::test::strings_column_wrapper strings( h_strings.begin(), h_strings.end() );
    cudf::size_type width = GetParam();
    auto strings_view = cudf::strings_column_view(strings);
    auto results = cudf::strings::pad(strings_view, width, cudf::strings::pad_side::right);

    std::vector<std::string> h_expected;
    for( auto itr=h_strings.begin(); itr != h_strings.end(); ++itr )
    {
        std::string str = *itr;
        cudf::size_type size = str.size();
        if( size < width )
            str.insert( size, width-size, ' ' );
        h_expected.push_back(str);
    }
    cudf::test::strings_column_wrapper expected( h_expected.begin(), h_expected.end() );

    cudf::test::expect_columns_equal(*results,expected);
}


INSTANTIATE_TEST_CASE_P(StringsPadParmWidthTest, StringsPadParmsTest,
                        testing::ValuesIn(std::array<cudf::size_type,3>{5,6,7}));


TEST_F(StringsPadTest, ZFill)
{
    std::vector<const char*> h_strings{ "654321", "-12345", nullptr, "", "-5", "0987", "4", "+8.5", "éé" };
    cudf::test::strings_column_wrapper strings( h_strings.begin(), h_strings.end(),
        thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));
    cudf::size_type width = 6;
    std::string phil = "+";
    auto strings_view = cudf::strings_column_view(strings);

    auto results = cudf::strings::zfill(strings_view, width);

    std::vector<const char*> h_expected{ "654321", "-12345", nullptr, "000000", "-00005", "000987", "000004", "+008.5", "0000éé" };
    cudf::test::strings_column_wrapper expected( h_expected.begin(), h_expected.end(),
        thrust::make_transform_iterator( h_expected.begin(), [] (auto str) { return str!=nullptr; }));
    cudf::test::expect_columns_equal(*results,expected);
}
