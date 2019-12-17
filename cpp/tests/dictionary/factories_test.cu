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
#include <cudf/dictionary/dictionary_factories.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>

#include <vector>
#include <cstring>


struct DictionaryFactoriesTest : public cudf::test::BaseFixture {};

TEST_F(DictionaryFactoriesTest, CreateFromColumn)
{
    std::vector<const char*> h_strings{ "eee", "aaa", "ddd", "bbb", "ccc", "ccc", "ccc", "eee", "aaa" };
    cudf::test::strings_column_wrapper strings( h_strings.begin(), h_strings.end() );

    auto dictionary = cudf::make_dictionary_column( strings );
    printf("size = %d\n", (int)dictionary->size());
    printf("children = %d\n", (int)dictionary->num_children());
    printf("null_count = %d\n", (int)dictionary->null_count());
}


//TEST_F(DictionaryFactoriesTest, EmptyStringsColumn)
//{
//    rmm::device_vector<char> d_chars;
//    rmm::device_vector<cudf::size_type> d_offsets(1,0);
//    rmm::device_vector<cudf::bitmask_type> d_nulls;
//
//    auto results = cudf::make_strings_column( d_chars, d_offsets, d_nulls, 0 );
//    cudf::test::expect_strings_empty(results->view());
//
//    rmm::device_vector<thrust::pair<const char*,cudf::size_type>> d_strings;
//    results = cudf::make_strings_column( d_strings );
//    cudf::test::expect_strings_empty(results->view());
//}
