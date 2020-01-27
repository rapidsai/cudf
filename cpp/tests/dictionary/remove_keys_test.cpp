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
#include <cudf/dictionary/encode.hpp>
#include <cudf/dictionary/update_keys.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>

#include <vector>

struct DictionaryRemoveKeysTest : public cudf::test::BaseFixture {};

TEST_F(DictionaryRemoveKeysTest, StringsColumn)
{
    std::vector<const char*> h_strings{ "eee", "aaa", "ddd", "bbb", "ccc", "ccc", "ccc", "eee", "aaa" };
    cudf::test::strings_column_wrapper strings( h_strings.begin(), h_strings.end() );
    std::vector<const char*> h_remove_keys{ "ddd", "bbb", "fff" };
    cudf::test::strings_column_wrapper remove_keys( h_remove_keys.begin(), h_remove_keys.end() );

    auto dictionary = cudf::dictionary::encode( strings );
    auto result = cudf::dictionary::remove_keys( cudf::dictionary_column_view(dictionary->view()), remove_keys );
    cudf::dictionary_column_view view(result->view());

    std::vector<const char*> h_keys{ "aaa", "ccc", "eee" };
    cudf::test::strings_column_wrapper keys_expected( h_keys.begin(), h_keys.end() );
    cudf::test::expect_columns_equal(view.dictionary_keys(), keys_expected);

    std::vector<int32_t> h_expected{2,0,-1,-1,1,1,1,2,0};
    cudf::test::fixed_width_column_wrapper<int32_t> indices_expected( h_expected.begin(), h_expected.end() );
    cudf::test::expect_columns_equal(view.indices(), indices_expected);
}

TEST_F(DictionaryRemoveKeysTest, FloatColumn)
{
    cudf::test::fixed_width_column_wrapper<float> input{ 4.25, 7.125, 0.5, -11.75, 7.125, 0.5 };
    cudf::test::fixed_width_column_wrapper<float> remove_keys{ 4.25, -11.75, 5.0 };

    auto dictionary = cudf::dictionary::encode( input );
    auto result = cudf::dictionary::remove_keys( cudf::dictionary_column_view(dictionary->view()), remove_keys );
    cudf::dictionary_column_view view(result->view());

    cudf::test::fixed_width_column_wrapper<float> keys_expected{ 0.5, 7.125 };
    cudf::test::expect_columns_equal(view.dictionary_keys(), keys_expected);

    cudf::test::fixed_width_column_wrapper<int32_t> expected{-1,1,0,-1,1,0};
    cudf::test::expect_columns_equal(view.indices(), expected);
}

TEST_F(DictionaryRemoveKeysTest, WithNull)
{
    cudf::test::fixed_width_column_wrapper<int64_t> input{ { 444,0,333,111,222,222,222,444,0 }, {1,1,1,1,1,0,1,1,1}};
    cudf::test::fixed_width_column_wrapper<int64_t> remove_keys{ 0, 111, 777 };

    auto dictionary = cudf::dictionary::encode( input );
    auto result = cudf::dictionary::remove_keys( cudf::dictionary_column_view(dictionary->view()), remove_keys );
    cudf::dictionary_column_view view(result->view());

    cudf::test::fixed_width_column_wrapper<int64_t> keys_expected{ 222,333,444 };
    cudf::test::expect_columns_equal(view.dictionary_keys(), keys_expected);

    cudf::test::fixed_width_column_wrapper<int32_t> expected{2,-1,1,-1,0,0,0,2,-1};
    cudf::test::expect_columns_equal(view.indices(), expected);
}
