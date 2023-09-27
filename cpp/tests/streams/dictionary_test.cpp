/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
#include <cudf/dictionary/encode.hpp>
#include <cudf/dictionary/search.hpp>
#include <cudf/dictionary/update_keys.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>

class DictionaryTest : public cudf::test::BaseFixture {};

TEST_F(DictionaryTest, Encode)
{
  cudf::test::fixed_width_column_wrapper<int> col({1, 2, 3, 4, 5});
  cudf::data_type int32_type(cudf::type_id::UINT32);
  cudf::column_view col_view = col;
  cudf::dictionary::encode(col_view, int32_type, cudf::test::get_default_stream());
}

TEST_F(DictionaryTest, Decode)
{
  // keys = {0, 2, 6}, indices = {0, 1, 1, 2, 2}
  std::vector<int32_t> elements{0, 2, 2, 6, 6};
  cudf::test::dictionary_column_wrapper<int32_t> dict_col(elements.begin(), elements.end());
  cudf::dictionary_column_view dict_col_view = dict_col;
  cudf::dictionary::decode(dict_col_view, cudf::test::get_default_stream());
}

TEST_F(DictionaryTest, GetIndex)
{
  std::vector<int32_t> elements{0, 2, 2, 6, 6};
  cudf::test::dictionary_column_wrapper<int32_t> dict_col(elements.begin(), elements.end());
  cudf::dictionary_column_view dict_col_view = dict_col;
  cudf::numeric_scalar<int32_t> key_scalar(2, true, cudf::test::get_default_stream());
  cudf::dictionary::get_index(dict_col_view, key_scalar, cudf::test::get_default_stream());
}

TEST_F(DictionaryTest, AddKeys)
{
  std::vector<int32_t> elements{0, 2, 2, 6, 6};
  cudf::test::dictionary_column_wrapper<int32_t> dict_col(elements.begin(), elements.end());
  cudf::dictionary_column_view dict_col_view = dict_col;
  cudf::test::fixed_width_column_wrapper<int> new_keys_col({8, 9});
  cudf::dictionary::add_keys(dict_col_view, new_keys_col, cudf::test::get_default_stream());
}

TEST_F(DictionaryTest, RemoveKeys)
{
  std::vector<int32_t> elements{0, 2, 2, 6, 6};
  cudf::test::dictionary_column_wrapper<int32_t> dict_col(elements.begin(), elements.end());
  cudf::dictionary_column_view dict_col_view = dict_col;
  cudf::test::fixed_width_column_wrapper<int> keys_to_remove_col({2});
  cudf::dictionary::remove_keys(
    dict_col_view, keys_to_remove_col, cudf::test::get_default_stream());
}

TEST_F(DictionaryTest, RemoveUnsedKeys)
{
  std::vector<int32_t> elements{0, 2, 2, 6, 6};
  cudf::test::dictionary_column_wrapper<int32_t> dict_col(elements.begin(), elements.end());
  cudf::dictionary_column_view dict_col_view = dict_col;
  cudf::dictionary::remove_unused_keys(dict_col_view, cudf::test::get_default_stream());
}

TEST_F(DictionaryTest, SetKeys)
{
  std::vector<int32_t> elements{0, 2, 2, 6, 6};
  cudf::test::dictionary_column_wrapper<int32_t> dict_col(elements.begin(), elements.end());
  cudf::dictionary_column_view dict_col_view = dict_col;
  cudf::test::fixed_width_column_wrapper<int> keys_col({2, 6});
  cudf::dictionary::set_keys(dict_col_view, keys_col, cudf::test::get_default_stream());
}

TEST_F(DictionaryTest, MatchDictionaries)
{
  std::vector<int32_t> elements_a{0, 2, 2, 6, 6};
  cudf::test::dictionary_column_wrapper<int32_t> dict_col_a(elements_a.begin(), elements_a.end());
  cudf::dictionary_column_view dict_col_view_a = dict_col_a;

  std::vector<int32_t> elements_b{1, 3, 4, 5, 5};
  cudf::test::dictionary_column_wrapper<int32_t> dict_col_b(elements_b.begin(), elements_b.end());
  cudf::dictionary_column_view dict_col_view_b = dict_col_b;

  std::vector<cudf::dictionary_column_view> dicts = {dict_col_view_a, dict_col_view_b};

  cudf::test::fixed_width_column_wrapper<int> keys_col({2, 6});
  cudf::dictionary::match_dictionaries(dicts, cudf::test::get_default_stream());
}
