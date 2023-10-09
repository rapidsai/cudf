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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>

#include <cudf/lists/combine.hpp>
#include <cudf/lists/contains.hpp>
#include <cudf/lists/count_elements.hpp>

class ListTest : public cudf::test::BaseFixture {};

TEST_F(ListTest, ConcatenateRows)
{
  cudf::test::lists_column_wrapper<int> list_col_1{{0, 1}, {2, 3}, {4, 5}};
  cudf::test::lists_column_wrapper<int> list_col_2{{0, 1}, {2, 3}, {4, 5}};
  cudf::table_view lists_table({list_col_1, list_col_2});
  cudf::lists::concatenate_rows(
    lists_table, cudf::lists::concatenate_null_policy::IGNORE, cudf::test::get_default_stream());
}

TEST_F(ListTest, ConcatenateListElements)
{
  cudf::test::lists_column_wrapper<int> ll_column{{{0, 1}, {2, 3}}, {{4, 5}, {6, 7}}};
  cudf::lists::concatenate_list_elements(
    ll_column, cudf::lists::concatenate_null_policy::IGNORE, cudf::test::get_default_stream());
}

TEST_F(ListTest, ContainsNulls)
{
  cudf::test::lists_column_wrapper<int> list_col{{0, 1}, {2, 3}, {4, 5}};
  cudf::lists::contains_nulls(list_col, cudf::test::get_default_stream());
}

TEST_F(ListTest, ContainsSearchKey)
{
  cudf::test::lists_column_wrapper<int> list_col{{0, 1}, {2, 3}, {4, 5}};
  cudf::numeric_scalar<int32_t> search_key(2, true, cudf::test::get_default_stream());
  cudf::lists::contains(list_col, search_key, cudf::test::get_default_stream());
}

TEST_F(ListTest, ContainsSearchKeys)
{
  cudf::test::lists_column_wrapper<int> list_col{{0, 1}, {2, 3}, {4, 5}};
  cudf::test::fixed_width_column_wrapper<int> search_keys({1, 2, 3});
  cudf::lists::contains(list_col, search_keys, cudf::test::get_default_stream());
}

TEST_F(ListTest, IndexOfSearchKey)
{
  cudf::test::lists_column_wrapper<int> list_col{{0, 1}, {2, 3}, {4, 5}};
  cudf::numeric_scalar<int32_t> search_key(2, true, cudf::test::get_default_stream());
  cudf::lists::index_of(list_col,
                        search_key,
                        cudf::lists::duplicate_find_option::FIND_FIRST,
                        cudf::test::get_default_stream());
}

TEST_F(ListTest, IndexOfSearchKeys)
{
  cudf::test::lists_column_wrapper<int> list_col{{0, 1}, {2, 3}, {4, 5}};
  cudf::test::fixed_width_column_wrapper<int> search_keys({1, 2, 3});
  cudf::lists::index_of(list_col,
                        search_keys,
                        cudf::lists::duplicate_find_option::FIND_FIRST,
                        cudf::test::get_default_stream());
}

TEST_F(ListTest, CountElements)
{
  cudf::test::lists_column_wrapper<int> list_col{{0, 1}, {2, 3, 7}, {4, 5}};
  cudf::lists::count_elements(list_col, cudf::test::get_default_stream());
}
