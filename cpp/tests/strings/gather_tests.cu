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
#include <cudf/detail/gather.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/utilities/error.hpp>

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/column_utilities.hpp>
#include "./utilities.h"

#include <vector>
#include <gmock/gmock.h>


struct StringsGatherTest : public cudf::test::BaseFixture {};

TEST_F(StringsGatherTest, Gather)
{
    std::vector<const char*> h_strings{ "eee", "bb", "", "aa", "bbb", "ééé" };
    cudf::test::strings_column_wrapper strings( h_strings.begin(), h_strings.end() );
    cudf::table_view source_table ({strings});

    std::vector<int32_t> h_map{ 4,1,5,2,7 };
    cudf::test::fixed_width_column_wrapper<int32_t> gather_map{h_map.begin(), h_map.end()};
    auto results = cudf::experimental::detail::gather(
           source_table,
           gather_map,
           false, true
           );

    std::vector<const char*> h_expected;
    std::vector<int32_t> expected_validity;
    for( auto itr = h_map.begin(); itr != h_map.end(); ++itr )
    {
        auto index = *itr;
        if( (0 <= index) && (index < static_cast<decltype(index)>(h_strings.size())) ) {
            h_expected.push_back( h_strings[index] );
            expected_validity.push_back(1);
        }
        else {
            h_expected.push_back( "" );
            expected_validity.push_back(0);
        }
    }
    cudf::test::strings_column_wrapper expected( h_expected.begin(), h_expected.end(),
        expected_validity.begin());
    cudf::test::expect_columns_equal(results->view().column(0),expected);
}

TEST_F(StringsGatherTest, GatherIgnoreOutOfBounds)
{
    std::vector<const char*> h_strings{ "eee", "bb", "", "aa", "bbb", "ééé" };
    cudf::test::strings_column_wrapper strings( h_strings.begin(), h_strings.end() );
    cudf::table_view source_table ({strings});

    std::vector<int32_t> h_map{ 3,4,0,0 };
    cudf::test::fixed_width_column_wrapper<int32_t> gather_map{h_map.begin(), h_map.end()};
    auto results = cudf::experimental::detail::gather(
           source_table,
           gather_map,
           false, true
           );

    std::vector<const char*> h_expected;
    std::vector<int32_t> expected_validity;
    for( auto itr = h_map.begin(); itr != h_map.end(); ++itr ) {
        h_expected.push_back( h_strings[*itr] );
        expected_validity.push_back(1);
    }
    cudf::test::strings_column_wrapper expected( h_expected.begin(), h_expected.end(),
        expected_validity.begin());
    cudf::test::expect_columns_equal(results->view().column(0),expected);
}


TEST_F(StringsGatherTest, GatherZeroSizeStringsColumn)
{
    cudf::column_view zero_size_strings_column( cudf::data_type{cudf::STRING}, 0, nullptr, nullptr, 0);
    auto strings_view = cudf::strings_column_view(zero_size_strings_column);
    thrust::device_vector<int32_t> gather_map;
    auto results = cudf::strings::detail::gather<true>(strings_view, gather_map.begin(), gather_map.end() );
    cudf::test::expect_strings_empty(results->view());
}
