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
#include <cudf/strings/sorting.hpp>
#include <cudf/strings/copying.hpp>
#include <cudf/utilities/error.hpp>

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/column_utilities.hpp>
#include "./utilities.h"

#include <vector>
#include <gmock/gmock.h>


struct StringsColumnTest : public cudf::test::BaseFixture {};

TEST_F(StringsColumnTest, Sort)
{
    // cannot initialize std::string with a nullptr so use "<null>" as a place-holder
    cudf::test::strings_column_wrapper h_strings({ "eee", "bb", "<null>", "", "aa", "bbb", "ééé" },
                                                 {    1,    1,         0,  1,   1,     1,     1});
    cudf::test::strings_column_wrapper h_expected({ "<null>", "", "aa", "bb", "bbb", "eee", "ééé" },
                                                  {        0,  1,   1,    1,     1,     1,     1});

    auto strings_view = cudf::strings_column_view(h_strings);
    auto results = cudf::strings::detail::sort(strings_view, cudf::strings::detail::name);
    cudf::test::expect_columns_equal(*results, h_expected);
}

TEST_F(StringsColumnTest, SortZeroSizeStringsColumn)
{
    cudf::column_view zero_size_strings_column( cudf::data_type{cudf::STRING}, 0, nullptr, nullptr, 0);
    auto strings_view = cudf::strings_column_view(zero_size_strings_column);
    auto results = cudf::strings::detail::sort(strings_view, cudf::strings::detail::name);
    cudf::test::expect_strings_empty(results->view());
}

class SliceParmsTest : public StringsColumnTest,
                       public testing::WithParamInterface<cudf::size_type> {};

TEST_P(SliceParmsTest, Slice)
{
    std::vector<const char*> h_strings{ "eee", "bb", nullptr, "", "aa", "bbb", "ééé" };
    cudf::test::strings_column_wrapper strings( h_strings.begin(), h_strings.end(),
        thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));
    cudf::size_type start = 3;
    cudf::size_type end = GetParam();
    std::vector<const char*> h_expected;
    if( end > start )
    {
        for( cudf::size_type idx=start; (idx < end) && (idx < (cudf::size_type)h_strings.size()); ++idx )
            h_expected.push_back( h_strings[idx] );
    }
    auto strings_view = cudf::strings_column_view(strings);
    auto results = cudf::strings::detail::slice(strings_view,start,end);

    cudf::test::strings_column_wrapper expected( h_expected.begin(), h_expected.end() );
         //thrust::make_transform_iterator( h_expected.begin(), [] (auto str) { return str!=nullptr; }));
    cudf::test::expect_columns_equal(*results,expected);
}

TEST_P(SliceParmsTest, SliceAllNulls)
{
    std::vector<const char*> h_strings{ nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr };
    cudf::test::strings_column_wrapper strings( h_strings.begin(), h_strings.end(),
        thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));
    cudf::size_type start = 3;
    cudf::size_type end = GetParam();
    std::vector<const char*> h_expected;
    if( end > start )
    {
        for( cudf::size_type idx=start; (idx < end) && (idx < (cudf::size_type)h_strings.size()); ++idx )
            h_expected.push_back( h_strings[idx] );
    }
    auto strings_view = cudf::strings_column_view(strings);
    auto results = cudf::strings::detail::slice(strings_view,start,end);
    cudf::test::strings_column_wrapper expected( h_expected.begin(), h_expected.end(),
         thrust::make_transform_iterator( h_expected.begin(), [] (auto str) { return str!=nullptr; }));
    cudf::test::expect_columns_equal(*results,expected);
}

TEST_P(SliceParmsTest, SliceAllEmpty)
{
    std::vector<const char*> h_strings{ "", "", "", "", "", "", "" };
    cudf::test::strings_column_wrapper strings( h_strings.begin(), h_strings.end(),
        thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));
    cudf::size_type start = 3;
    cudf::size_type end = GetParam();
    std::vector<const char*> h_expected;
    if( end > start )
    {
        for( cudf::size_type idx=start; (idx < end) && (idx < (cudf::size_type)h_strings.size()); ++idx )
            h_expected.push_back( h_strings[idx] );
    }
    auto strings_view = cudf::strings_column_view(strings);
    auto results = cudf::strings::detail::slice(strings_view,start,end);
    cudf::test::strings_column_wrapper expected( h_expected.begin(), h_expected.end() );
         //thrust::make_transform_iterator( h_expected.begin(), [] (auto str) { return str!=nullptr; }));
    cudf::test::expect_columns_equal(*results,expected);
}

INSTANTIATE_TEST_CASE_P(SliceParms, SliceParmsTest,
                        testing::ValuesIn(std::array<cudf::size_type,3>{5,6,7}));

TEST_F(StringsColumnTest, SliceZeroSizeStringsColumn)
{
    cudf::column_view zero_size_strings_column( cudf::data_type{cudf::STRING}, 0, nullptr, nullptr, 0);
    auto strings_view = cudf::strings_column_view(zero_size_strings_column);
    auto results = cudf::strings::detail::slice(strings_view,1,2);
    cudf::test::expect_strings_empty(results->view());
}

TEST_F(StringsColumnTest, Gather)
{
    std::vector<const char*> h_strings{ "eee", "bb", nullptr, "", "aa", "bbb", "ééé" };
    cudf::test::strings_column_wrapper strings( h_strings.begin(), h_strings.end(),
        thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));
    auto strings_view = cudf::strings_column_view(strings);

    cudf::test::fixed_width_column_wrapper<int32_t> gather_map{{4,1}};
    auto results = cudf::strings::detail::gather(strings_view,gather_map);

    std::vector<const char*> h_expected{ "aa", "bb" };
    cudf::test::strings_column_wrapper expected( h_expected.begin(), h_expected.end() );
         //thrust::make_transform_iterator( h_expected.begin(), [] (auto str) { return str!=nullptr; }));
    cudf::test::expect_columns_equal(*results,expected);
}

TEST_F(StringsColumnTest, GatherZeroSizeStringsColumn)
{
    cudf::column_view zero_size_strings_column( cudf::data_type{cudf::STRING}, 0, nullptr, nullptr, 0);
    auto strings_view = cudf::strings_column_view(zero_size_strings_column);
    cudf::column_view map_view( cudf::data_type{cudf::INT32}, 0, nullptr, nullptr, 0);
    auto results = cudf::strings::detail::gather(strings_view,map_view);
    cudf::test::expect_strings_empty(results->view());
}

TEST_F(StringsColumnTest, Scatter)
{
    std::vector<const char*> h_strings1{ "eee", "bb", nullptr, "", "aa", "bbb", "ééé" };
    cudf::test::strings_column_wrapper strings1( h_strings1.begin(), h_strings1.end(),
        thrust::make_transform_iterator( h_strings1.begin(), [] (auto str) { return str!=nullptr; }));
    std::vector<const char*> h_strings2{ "1", "22" };
    cudf::test::strings_column_wrapper strings2( h_strings2.begin(), h_strings2.end(),
        thrust::make_transform_iterator( h_strings2.begin(), [] (auto str) { return str!=nullptr; }));

    auto view1 = cudf::strings_column_view(strings1);
    auto view2 = cudf::strings_column_view(strings2);

    cudf::test::fixed_width_column_wrapper<int32_t> scatter_map{{4,1}};
    auto results = cudf::strings::detail::scatter(view1,view2,scatter_map);

    std::vector<const char*> h_expected{ "eee", "22", nullptr, "", "1", "bbb", "ééé" };
    cudf::test::strings_column_wrapper expected( h_expected.begin(), h_expected.end(),
        thrust::make_transform_iterator( h_expected.begin(), [] (auto str) { return str!=nullptr; }));
    cudf::test::expect_columns_equal(*results,expected);
}

TEST_F(StringsColumnTest, ScatterZeroSizeStringsColumn)
{
    cudf::column_view zero_size_strings_column( cudf::data_type{cudf::STRING}, 0, nullptr, nullptr, 0);
    auto strings_view = cudf::strings_column_view(zero_size_strings_column);
    cudf::column_view map_view( cudf::data_type{cudf::INT32}, 0, nullptr, nullptr, 0);
    cudf::column_view values( cudf::data_type{cudf::STRING}, 0, nullptr, nullptr, 0);
    auto values_view = cudf::strings_column_view(values);
    auto results = cudf::strings::detail::scatter(strings_view,values_view,map_view);
    cudf::test::expect_strings_empty(results->view());
}

TEST_F(StringsColumnTest, ScatterScalar)
{
    std::vector<const char*> h_strings{ "eee", "bb", nullptr, "", "aa", "bbb", "ééé" };
    cudf::test::strings_column_wrapper strings( h_strings.begin(), h_strings.end(),
        thrust::make_transform_iterator( h_strings.begin(), [] (auto str) { return str!=nullptr; }));
    auto view = cudf::strings_column_view(strings);

    cudf::test::fixed_width_column_wrapper<int32_t> scatter_map{{4,1}};
    auto results = cudf::strings::detail::scatter(view,"---",scatter_map);

    std::vector<const char*> h_expected{ "eee", "---", nullptr, "", "---", "bbb", "ééé" };
    cudf::test::strings_column_wrapper expected( h_expected.begin(), h_expected.end(),
        thrust::make_transform_iterator( h_expected.begin(), [] (auto str) { return str!=nullptr; }));
    cudf::test::expect_columns_equal(*results,expected);
}

TEST_F(StringsColumnTest, ScatterScalarZeroSizeStringsColumn)
{
    cudf::column_view zero_size_strings_column( cudf::data_type{cudf::STRING}, 0, nullptr, nullptr, 0);
    auto strings_view = cudf::strings_column_view(zero_size_strings_column);
    cudf::column_view map_view( cudf::data_type{cudf::INT32}, 0, nullptr, nullptr, 0);
    auto results = cudf::strings::detail::scatter(strings_view,nullptr,map_view);
    cudf::test::expect_strings_empty(results->view());
}
