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

#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>


struct DictionaryFactoriesTest : public cudf::test::BaseFixture {};

TEST_F(DictionaryFactoriesTest, CreateFromColumns)
{
    std::vector<const char*> h_keys{ "aaa", "ccc", "ddd", "www" };
    cudf::test::strings_column_wrapper keys_strings( h_keys.begin(), h_keys.end() );
    std::vector<int32_t> h_values{2,0,3,1,2,2,2,3,0};
    cudf::test::fixed_width_column_wrapper<int32_t> values( h_values.begin(), h_values.end() );

    auto dictionary = cudf::make_dictionary_column( keys_strings, values );
    cudf::dictionary_column_view view(dictionary->view());

    cudf::test::expect_columns_equal(view.dictionary_keys(), keys_strings);
    cudf::test::expect_columns_equal(view.indices(), values);
}
