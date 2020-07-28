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

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/table_utilities.hpp>
#include <tests/utilities/type_lists.hpp>

#include <cudf/column/column.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/table/table.hpp>

#include <thrust/sequence.h>

template <typename T>
using column_wrapper = cudf::test::fixed_width_column_wrapper<T>;

using s_col_wrapper = cudf::test::strings_column_wrapper;

using CVector     = std::vector<std::unique_ptr<cudf::column>>;
using column      = cudf::column;
using column_view = cudf::column_view;
using TView       = cudf::table_view;
using Table       = cudf::table;

template <typename T>
struct TypedColumnTest : public cudf::test::BaseFixture {
  static std::size_t data_size() { return 1000; }
  static std::size_t mask_size() { return 100; }
  cudf::data_type type() { return cudf::data_type{cudf::type_to_id<T>()}; }

  TypedColumnTest()
    : data{_num_elements * cudf::size_of(type())},
      mask{cudf::bitmask_allocation_size_bytes(_num_elements)}
  {
    auto typed_data = static_cast<char*>(data.data());
    auto typed_mask = static_cast<char*>(mask.data());
    thrust::sequence(thrust::device, typed_data, typed_data + data_size());
    thrust::sequence(thrust::device, typed_mask, typed_mask + mask_size());
  }

  cudf::size_type num_elements() { return _num_elements; }

  std::random_device r;
  std::default_random_engine generator{r()};
  std::uniform_int_distribution<cudf::size_type> distribution{200, 1000};
  cudf::size_type _num_elements{distribution(generator)};
  rmm::device_buffer data{};
  rmm::device_buffer mask{};
  rmm::device_buffer all_valid_mask{create_null_mask(num_elements(), cudf::mask_state::ALL_VALID)};
  rmm::device_buffer all_null_mask{create_null_mask(num_elements(), cudf::mask_state::ALL_NULL)};
};

TYPED_TEST_CASE(TypedColumnTest, cudf::test::Types<int32_t>);

TYPED_TEST(TypedColumnTest, ConcatenateEmptyColumns)
{
  cudf::test::fixed_width_column_wrapper<TypeParam> empty_first{};
  cudf::test::fixed_width_column_wrapper<TypeParam> empty_second{};
  cudf::test::fixed_width_column_wrapper<TypeParam> empty_third{};
  std::vector<column_view> columns_to_concat({empty_first, empty_second, empty_third});

  auto concat = cudf::concatenate(columns_to_concat);

  auto expected_type = cudf::column_view(empty_first).type();
  EXPECT_EQ(concat->size(), 0);
  EXPECT_EQ(concat->type(), expected_type);
}

TYPED_TEST(TypedColumnTest, ConcatenateNoColumns)
{
  std::vector<column_view> columns_to_concat{};
  EXPECT_THROW(cudf::concatenate(columns_to_concat), cudf::logic_error);
}

TYPED_TEST(TypedColumnTest, ConcatenateColumnView)
{
  cudf::column original{this->type(), this->num_elements(), this->data, this->mask};
  std::vector<cudf::size_type> indices{0,
                                       this->num_elements() / 3,
                                       this->num_elements() / 3,
                                       this->num_elements() / 2,
                                       this->num_elements() / 2,
                                       this->num_elements()};
  std::vector<cudf::column_view> views = cudf::slice(original, indices);

  auto concatenated_col = cudf::concatenate(views);

  cudf::test::expect_columns_equal(original, *concatenated_col);
}

struct StringColumnTest : public cudf::test::BaseFixture {
};

TEST_F(StringColumnTest, ConcatenateColumnView)
{
  std::vector<const char*> h_strings{"aaa",
                                     "bb",
                                     "",
                                     "cccc",
                                     "d",
                                     "ééé",
                                     "ff",
                                     "gggg",
                                     "",
                                     "h",
                                     "iiii",
                                     "jjj",
                                     "k",
                                     "lllllll",
                                     "mmmmm",
                                     "n",
                                     "oo",
                                     "ppp"};
  cudf::test::strings_column_wrapper strings1(h_strings.data(), h_strings.data() + 6);
  cudf::test::strings_column_wrapper strings2(h_strings.data() + 6, h_strings.data() + 10);
  cudf::test::strings_column_wrapper strings3(h_strings.data() + 10,
                                              h_strings.data() + h_strings.size());

  std::vector<cudf::column_view> strings_columns;
  strings_columns.push_back(strings1);
  strings_columns.push_back(strings2);
  strings_columns.push_back(strings3);

  auto results = cudf::concatenate(strings_columns);

  cudf::test::strings_column_wrapper expected(h_strings.begin(), h_strings.end());
  cudf::test::expect_columns_equal(*results, expected);
}

struct TableTest : public cudf::test::BaseFixture {
};

TEST_F(TableTest, ConcatenateTables)
{
  std::vector<const char*> h_strings{
    "Lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing", "elit"};

  CVector cols_gold;
  column_wrapper<int8_t> col1_gold{{1, 2, 3, 4, 5, 6, 7, 8}};
  column_wrapper<int16_t> col2_gold{{1, 2, 3, 4, 5, 6, 7, 8}};
  s_col_wrapper col3_gold(h_strings.data(), h_strings.data() + h_strings.size());
  cols_gold.push_back(col1_gold.release());
  cols_gold.push_back(col2_gold.release());
  cols_gold.push_back(col3_gold.release());
  Table gold_table(std::move(cols_gold));

  CVector cols_table1;
  column_wrapper<int8_t> col1_table1{{1, 2, 3, 4}};
  column_wrapper<int16_t> col2_table1{{1, 2, 3, 4}};
  s_col_wrapper col3_table1(h_strings.data(), h_strings.data() + 4);
  cols_table1.push_back(col1_table1.release());
  cols_table1.push_back(col2_table1.release());
  cols_table1.push_back(col3_table1.release());
  Table t1(std::move(cols_table1));

  CVector cols_table2;
  column_wrapper<int8_t> col1_table2{{5, 6, 7, 8}};
  column_wrapper<int16_t> col2_table2{{5, 6, 7, 8}};
  s_col_wrapper col3_table2(h_strings.data() + 4, h_strings.data() + h_strings.size());
  cols_table2.push_back(col1_table2.release());
  cols_table2.push_back(col2_table2.release());
  cols_table2.push_back(col3_table2.release());
  Table t2(std::move(cols_table2));

  auto concat_table = cudf::concatenate({t1.view(), t2.view()});

  cudf::test::expect_tables_equal(*concat_table, gold_table);
}

TEST_F(TableTest, ConcatenateTablesWithOffsets)
{
  column_wrapper<int32_t> col1_1{{5, 4, 3, 5, 8, 5, 6}};
  cudf::test::strings_column_wrapper col2_1(
    {"dada", "egg", "avocado", "dada", "kite", "dog", "ln"});
  cudf::table_view table_view_in1{{col1_1, col2_1}};

  column_wrapper<int32_t> col1_2{{5, 8, 5, 6, 15, 14, 13}};
  cudf::test::strings_column_wrapper col2_2(
    {"dada", "kite", "dog", "ln", "dado", "greg", "spinach"});
  cudf::table_view table_view_in2{{col1_2, col2_2}};

  std::vector<cudf::size_type> split_indexes1{3};
  std::vector<cudf::table_view> partitioned1 = cudf::split(table_view_in1, split_indexes1);

  std::vector<cudf::size_type> split_indexes2{3};
  std::vector<cudf::table_view> partitioned2 = cudf::split(table_view_in2, split_indexes2);

  {
    std::vector<cudf::table_view> table_views_to_concat;
    table_views_to_concat.push_back(partitioned1[1]);
    table_views_to_concat.push_back(partitioned2[1]);
    std::unique_ptr<cudf::table> concatenated_tables = cudf::concatenate(table_views_to_concat);

    column_wrapper<int32_t> exp1_1{{5, 8, 5, 6, 6, 15, 14, 13}};
    cudf::test::strings_column_wrapper exp2_1(
      {"dada", "kite", "dog", "ln", "ln", "dado", "greg", "spinach"});
    cudf::table_view table_view_exp1{{exp1_1, exp2_1}};
    cudf::test::expect_tables_equal(concatenated_tables->view(), table_view_exp1);
  }
  {
    std::vector<cudf::table_view> table_views_to_concat;
    table_views_to_concat.push_back(partitioned1[0]);
    table_views_to_concat.push_back(partitioned2[1]);
    std::unique_ptr<cudf::table> concatenated_tables = cudf::concatenate(table_views_to_concat);

    column_wrapper<int32_t> exp1_1{{5, 4, 3, 6, 15, 14, 13}};
    cudf::test::strings_column_wrapper exp2_1(
      {"dada", "egg", "avocado", "ln", "dado", "greg", "spinach"});
    cudf::table_view table_view_exp1{{exp1_1, exp2_1}};
    cudf::test::expect_tables_equal(concatenated_tables->view(), table_view_exp1);
  }
  {
    std::vector<cudf::table_view> table_views_to_concat;
    table_views_to_concat.push_back(partitioned1[1]);
    table_views_to_concat.push_back(partitioned2[0]);
    std::unique_ptr<cudf::table> concatenated_tables = cudf::concatenate(table_views_to_concat);

    column_wrapper<int32_t> exp1_1{{5, 8, 5, 6, 5, 8, 5}};
    cudf::test::strings_column_wrapper exp2_1({"dada", "kite", "dog", "ln", "dada", "kite", "dog"});
    cudf::table_view table_view_exp{{exp1_1, exp2_1}};
    cudf::test::expect_tables_equal(concatenated_tables->view(), table_view_exp);
  }
}

TEST_F(TableTest, ConcatenateTablesWithOffsetsAndNulls)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col1_1{{5, 4, 3, 5, 8, 5, 6},
                                                         {0, 1, 1, 1, 1, 1, 1}};
  cudf::test::strings_column_wrapper col2_1({"dada", "egg", "avocado", "dada", "kite", "dog", "ln"},
                                            {1, 1, 1, 0, 1, 1, 1});
  cudf::table_view table_view_in1{{col1_1, col2_1}};

  cudf::test::fixed_width_column_wrapper<int32_t> col1_2{{5, 8, 5, 6, 15, 14, 13},
                                                         {1, 1, 1, 1, 1, 1, 0}};
  cudf::test::strings_column_wrapper col2_2(
    {"dada", "kite", "dog", "ln", "dado", "greg", "spinach"}, {1, 0, 1, 1, 1, 1, 1});
  cudf::table_view table_view_in2{{col1_2, col2_2}};

  std::vector<cudf::size_type> split_indexes1{3};
  std::vector<cudf::table_view> partitioned1 = cudf::split(table_view_in1, split_indexes1);

  std::vector<cudf::size_type> split_indexes2{3};
  std::vector<cudf::table_view> partitioned2 = cudf::split(table_view_in2, split_indexes2);

  {
    std::vector<cudf::table_view> table_views_to_concat;
    table_views_to_concat.push_back(partitioned1[1]);
    table_views_to_concat.push_back(partitioned2[1]);
    std::unique_ptr<cudf::table> concatenated_tables = cudf::concatenate(table_views_to_concat);

    cudf::test::fixed_width_column_wrapper<int32_t> exp1_1{{5, 8, 5, 6, 6, 15, 14, 13},
                                                           {1, 1, 1, 1, 1, 1, 1, 0}};
    cudf::test::strings_column_wrapper exp2_1(
      {"dada", "kite", "dog", "ln", "ln", "dado", "greg", "spinach"}, {0, 1, 1, 1, 1, 1, 1, 1});
    cudf::table_view table_view_exp1{{exp1_1, exp2_1}};
    cudf::test::expect_tables_equal(concatenated_tables->view(), table_view_exp1);
  }
  {
    std::vector<cudf::table_view> table_views_to_concat;
    table_views_to_concat.push_back(partitioned1[1]);
    table_views_to_concat.push_back(partitioned2[0]);
    std::unique_ptr<cudf::table> concatenated_tables = cudf::concatenate(table_views_to_concat);

    cudf::test::fixed_width_column_wrapper<int32_t> exp1_1{5, 8, 5, 6, 5, 8, 5};
    cudf::test::strings_column_wrapper exp2_1({"dada", "kite", "dog", "ln", "dada", "kite", "dog"},
                                              {0, 1, 1, 1, 1, 0, 1});
    cudf::table_view table_view_exp1{{exp1_1, exp2_1}};
    cudf::test::expect_tables_equal(concatenated_tables->view(), table_view_exp1);
  }
}

struct ListsColumnTest : public cudf::test::BaseFixture {
};

TEST_F(ListsColumnTest, ConcatenateLists)
{
  {
    cudf::test::lists_column_wrapper<int> a{0, 1, 2, 3};
    cudf::test::lists_column_wrapper<int> b{4, 5, 6, 7, 8, 9, 10};
    cudf::test::lists_column_wrapper<int> expected{{0, 1, 2, 3}, {4, 5, 6, 7, 8, 9, 10}};

    auto result = cudf::concatenate({a, b});

    cudf::test::expect_columns_equal(*result, expected);
  }

  {
    cudf::test::lists_column_wrapper<int> a{{0, 1, 1}, {2, 3}, {4, 5}};
    cudf::test::lists_column_wrapper<int> b{{6}, {8, 9, 9, 9}, {10, 11}};
    cudf::test::lists_column_wrapper<int> expected{
      {0, 1, 1}, {2, 3}, {4, 5}, {6}, {8, 9, 9, 9}, {10, 11}};

    auto result = cudf::concatenate({a, b});

    cudf::test::expect_columns_equal(*result, expected);
  }

  {
    cudf::test::lists_column_wrapper<int> a{{0, 1}, {2, 3, 4, 5}, {6, 7, 8}};
    cudf::test::lists_column_wrapper<int> b{{9}, {10, 11}, {12, 13, 14, 15}};
    cudf::test::lists_column_wrapper<int> expected{
      {0, 1}, {2, 3, 4, 5}, {6, 7, 8}, {9}, {10, 11}, {12, 13, 14, 15}};

    auto result = cudf::concatenate({a, b});

    cudf::test::expect_columns_equal(*result, expected);
  }
}

TEST_F(ListsColumnTest, ConcatenateEmptyLists)
{
  // to disambiguate between {} == 0 and {} == List{0}
  // Also, see note about compiler issues when declaring nested
  // empty lists in lists_column_wrapper documentation
  using LCW = cudf::test::lists_column_wrapper<int>;
  {
    cudf::test::lists_column_wrapper<int> a{};
    cudf::test::lists_column_wrapper<int> b{4, 5, 6, 7};
    cudf::test::lists_column_wrapper<int> expected{4, 5, 6, 7};

    auto result = cudf::concatenate({a, b});

    cudf::test::expect_columns_equal(*result, expected);
  }

  {
    cudf::test::lists_column_wrapper<int> a{}, b{}, c{};
    cudf::test::lists_column_wrapper<int> d{4, 5, 6, 7};
    cudf::test::lists_column_wrapper<int> expected{4, 5, 6, 7};

    auto result = cudf::concatenate({a, b, c, d});

    cudf::test::expect_columns_equal(*result, expected);
  }

  {
    cudf::test::lists_column_wrapper<int> a{LCW{}};
    cudf::test::lists_column_wrapper<int> b{4, 5, 6, 7};
    cudf::test::lists_column_wrapper<int> expected{{}, {4, 5, 6, 7}};

    auto result = cudf::concatenate({a, b});

    cudf::test::expect_columns_equal(*result, expected);
  }

  {
    cudf::test::lists_column_wrapper<int> a{LCW{}}, b{LCW{}}, c{LCW{}};
    cudf::test::lists_column_wrapper<int> d{4, 5, 6, 7};
    cudf::test::lists_column_wrapper<int> expected{{}, {}, {}, {4, 5, 6, 7}};

    auto result = cudf::concatenate({a, b, c, d});

    cudf::test::expect_columns_equal(*result, expected);
  }

  {
    cudf::test::lists_column_wrapper<int> a{1, 2};
    cudf::test::lists_column_wrapper<int> b{LCW{}}, c{LCW{}};
    cudf::test::lists_column_wrapper<int> d{4, 5, 6, 7};
    cudf::test::lists_column_wrapper<int> expected{{1, 2}, {}, {}, {4, 5, 6, 7}};

    auto result = cudf::concatenate({a, b, c, d});

    cudf::test::expect_columns_equal(*result, expected);
  }
}

TEST_F(ListsColumnTest, ConcatenateListsWithNulls)
{
  auto valids = cudf::test::make_counting_transform_iterator(
    0, [](auto i) { return i % 2 == 0 ? true : false; });

  // nulls in the leaves
  {
    cudf::test::lists_column_wrapper<int> a{{{0, 1, 2, 3}, valids}};
    cudf::test::lists_column_wrapper<int> b{{{4, 6, 7}, valids}};
    cudf::test::lists_column_wrapper<int> expected{{{0, 1, 2, 3}, valids}, {{4, 6, 7}, valids}};

    auto result = cudf::concatenate({a, b});

    cudf::test::expect_columns_equal(*result, expected);
  }
}

TEST_F(ListsColumnTest, ConcatenateNestedLists)
{
  {
    cudf::test::lists_column_wrapper<int> a{{{0, 1}, {2}}, {{4, 5, 6, 7, 8, 9, 10}}};
    cudf::test::lists_column_wrapper<int> b{{{6, 7}}, {{8, 9, 10}, {11, 12}}};
    cudf::test::lists_column_wrapper<int> expected{
      {{0, 1}, {2}}, {{4, 5, 6, 7, 8, 9, 10}}, {{6, 7}}, {{8, 9, 10}, {11, 12}}};

    auto result = cudf::concatenate({a, b});

    cudf::test::expect_columns_equal(*result, expected);
  }

  {
    cudf::test::lists_column_wrapper<int> a{
      {{{0, 1, 2}, {3, 4}}, {{5}, {6, 7}}, {{8, 9}}},
      {{{10}, {11, 12}}, {{13, 14, 15, 16}, {15, 16}}, {{17, 18}, {19, 20}}},
      {{{50}, {51, 52}}, {{54}, {55, 16}}, {{57, 18}, {59, 60}}}};

    cudf::test::lists_column_wrapper<int> b{
      {{{21, 22}, {23, 24}}, {{25}, {26, 27}}, {{28, 29, 30}}},
      {{{31, 32}, {33, 34}}, {{35, 36}, {37, 38}}, {{39, 40}}},
      {{{71, 72}, {74}}, {{75, 76, 77, 78}, {77, 78}}, {{79, 80, 81}}}};

    cudf::test::lists_column_wrapper<int> expected{
      {{{0, 1, 2}, {3, 4}}, {{5}, {6, 7}}, {{8, 9}}},
      {{{10}, {11, 12}}, {{13, 14, 15, 16}, {15, 16}}, {{17, 18}, {19, 20}}},
      {{{50}, {51, 52}}, {{54}, {55, 16}}, {{57, 18}, {59, 60}}},
      {{{21, 22}, {23, 24}}, {{25}, {26, 27}}, {{28, 29, 30}}},
      {{{31, 32}, {33, 34}}, {{35, 36}, {37, 38}}, {{39, 40}}},
      {{{71, 72}, {74}}, {{75, 76, 77, 78}, {77, 78}}, {{79, 80, 81}}}};

    auto result = cudf::concatenate({a, b});

    cudf::test::expect_columns_equal(*result, expected);
  }
}

TEST_F(ListsColumnTest, ConcatenateNestedEmptyLists)
{
  using T = int;
  // to disambiguate between {} == 0 and {} == List{0}
  // Also, see note about compiler issues when declaring nested
  // empty lists in lists_column_wrapper documentation
  using LCW = cudf::test::lists_column_wrapper<T>;

  {
    cudf::test::lists_column_wrapper<T> a{{{LCW{}}}, {{0, 1}, {2, 3}}};
    cudf::test::lists_column_wrapper<int> b{{{6, 7}}, {LCW{}, {11, 12}}};
    cudf::test::lists_column_wrapper<int> expected{
      {{LCW{}}}, {{0, 1}, {2, 3}}, {{6, 7}}, {LCW{}, {11, 12}}};

    auto result = cudf::concatenate({a, b});

    cudf::test::expect_columns_equal(*result, expected);
  }

  {
    cudf::test::lists_column_wrapper<int> a{
      {{{0, 1, 2}, LCW{}}, {{5}, {6, 7}}, {{8, 9}}},
      {{{LCW{}}}, {{17, 18}, {19, 20}}},
      {{{LCW{}}}},
      {{{50}, {51, 52}}, {{53, 54}, {55, 16, 17}}, {{59, 60}}}};

    cudf::test::lists_column_wrapper<int> b{
      {{{21, 22}, {23, 24}}, {LCW{}, {26, 27}}, {{28, 29, 30}}},
      {{{31, 32}, {33, 34}}, {{35, 36}, {37, 38}, {1, 2}}, {{39, 40}}},
      {{{LCW{}}}}};

    cudf::test::lists_column_wrapper<int> expected{
      {{{0, 1, 2}, LCW{}}, {{5}, {6, 7}}, {{8, 9}}},
      {{{LCW{}}}, {{17, 18}, {19, 20}}},
      {{{LCW{}}}},
      {{{50}, {51, 52}}, {{53, 54}, {55, 16, 17}}, {{59, 60}}},
      {{{21, 22}, {23, 24}}, {LCW{}, {26, 27}}, {{28, 29, 30}}},
      {{{31, 32}, {33, 34}}, {{35, 36}, {37, 38}, {1, 2}}, {{39, 40}}},
      {{{LCW{}}}}};

    auto result = cudf::concatenate({a, b});

    cudf::test::expect_columns_equal(*result, expected);
  }
}

TEST_F(ListsColumnTest, ConcatenateNestedListsWithNulls)
{
  auto valids = cudf::test::make_counting_transform_iterator(
    0, [](auto i) { return i % 2 == 0 ? true : false; });

  // nulls in the lists
  {
    cudf::test::lists_column_wrapper<int> a{{{{0, 1}, {2, 3}}, valids}};
    cudf::test::lists_column_wrapper<int> b{{{{4}, {6, 7}}, valids}};

    cudf::test::lists_column_wrapper<int> expected{{{{0, 1}, {2, 3}}, valids},
                                                   {{{4}, {6, 7}}, valids}};

    auto result = cudf::concatenate({a, b});

    cudf::test::expect_columns_equal(*result, expected);
  }

  // nulls in the lists -and- the values
  {
    cudf::test::lists_column_wrapper<int> a{{{{{0}, valids}, {2, 3}}, valids}, {{4, 5}}};
    cudf::test::lists_column_wrapper<int> b{{{6, 7}}, {{{{8, 9, 10}, valids}, {11, 12}}, valids}};
    cudf::test::lists_column_wrapper<int> expected{{{{{0}, valids}, {2, 3}}, valids},
                                                   {{4, 5}},
                                                   {{6, 7}},
                                                   {{{{8, 9, 10}, valids}, {11, 12}}, valids}};

    auto result = cudf::concatenate({a, b});

    cudf::test::expect_columns_equal(*result, expected);
  }
}

TEST_F(ListsColumnTest, ConcatenateIncompleteHierarchies)
{
  // to disambiguate between {} == 0 and {} == List{0}
  // Also, see note about compiler issues when declaring nested
  // empty lists in lists_column_wrapper documentation
  using LCW = cudf::test::lists_column_wrapper<int>;

  {
    std::vector<bool> valids{false};
    cudf::test::lists_column_wrapper<int> a{{{{LCW{}}}}};
    cudf::test::lists_column_wrapper<int> b{{{LCW{}}}};
    cudf::test::lists_column_wrapper<int> c{{LCW{}}};
    auto result = cudf::concatenate({a, b, c});

    cudf::test::lists_column_wrapper<int> expected{{{{LCW{}}}}, {{LCW{}}}, {LCW{}}};

    cudf::test::expect_columns_equal(*result, expected);
  }

  {
    std::vector<bool> valids{false};
    cudf::test::lists_column_wrapper<int> a{{{{{LCW{}}}}, valids.begin()}};
    cudf::test::lists_column_wrapper<int> b{{{LCW{}}}};
    cudf::test::lists_column_wrapper<int> c{{LCW{}}};
    auto result = cudf::concatenate({a, b, c});

    cudf::test::lists_column_wrapper<int> expected{
      {{{{LCW{}}}}, valids.begin()}, {{LCW{}}}, {LCW{}}};

    cudf::test::expect_columns_equal(*result, expected);
  }
}

template <typename T>
struct FixedPointTestBothReps : public cudf::test::BaseFixture {
};

template <typename T>
using wrapper = cudf::test::fixed_width_column_wrapper<T>;
TYPED_TEST_CASE(FixedPointTestBothReps, cudf::test::FixedPointTypes);

TYPED_TEST(FixedPointTestBothReps, FixedPointConcatentate)
{
  using namespace numeric;
  using decimalXX = TypeParam;

  auto vec = std::vector<decimalXX>(1000);
  std::iota(std::begin(vec), std::end(vec), decimalXX{});

  auto const a = wrapper<decimalXX>(vec.begin(), /***/ vec.begin() + 300);
  auto const b = wrapper<decimalXX>(vec.begin() + 300, vec.begin() + 700);
  auto const c = wrapper<decimalXX>(vec.begin() + 700, vec.end());

  auto const fixed_point_columns = std::vector<cudf::column_view>{a, b, c};
  auto const results             = cudf::concatenate(fixed_point_columns);
  auto const expected            = wrapper<decimalXX>(vec.begin(), vec.end());

  cudf::test::expect_columns_equal(*results, expected);
}