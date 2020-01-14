/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <cudf/copying.hpp>

#include <vector>


struct DictionaryGatherTest : public cudf::test::BaseFixture {};

TEST_F(DictionaryGatherTest, Gather)
{
    std::vector<const char*> h_strings{ "eee", "aaa", "ddd", "bbb", "ccc", "ccc", "ccc", "eee", "aaa" };
    cudf::test::strings_column_wrapper strings( h_strings.begin(), h_strings.end() );

    auto dictionary = cudf::make_dictionary_column( strings );
    cudf::dictionary_column_view view(dictionary->view());
    auto keys = *(view.dictionary_keys());

    std::vector<const char*> h_keys{ "aaa", "bbb", "ccc", "ddd", "eee" };
    cudf::test::strings_column_wrapper keys_strings( h_keys.begin(), h_keys.end() );
    cudf::test::expect_columns_equal(keys, keys_strings);

    std::vector<int32_t> h_expected{4,0,3,1,2,2,2,4,0};
    cudf::test::fixed_width_column_wrapper<int32_t> expected( h_expected.begin(), h_expected.end() );
    cudf::test::expect_columns_equal(view.indices(), expected);

    cudf::test::fixed_width_column_wrapper<int16_t> gather_map{{0, 1, 3, 4}};
    cudf::table_view source_table{{dictionary->view()}};

//    auto got = cudf::experimental::gather(source_table, gather_map);
//    auto result = cudf::dictionary_column_view(got->view().column(0));
//    cudf::test::print( result.indices() );
//    cudf::test::print( *(result.dictionary_keys()) );

}

