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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/table/table.hpp>
#include <string>

#include <rmm/device_uvector.hpp>

#include <thrust/iterator/counting_iterator.h>
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

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(original, *concatenated_col);
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
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringColumnTest, ConcatenateTooManyColumns)
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

  std::vector<const char*> expected_strings;
  std::vector<cudf::test::strings_column_wrapper> wrappers;
  std::vector<cudf::column_view> strings_columns;
  std::string expected_string;
  for (int i = 0; i < 200; ++i) {
    wrappers.emplace_back(h_strings.data(), h_strings.data() + h_strings.size());
    strings_columns.push_back(wrappers[i]);
    expected_strings.insert(expected_strings.end(), h_strings.begin(), h_strings.end());
  }
  cudf::test::strings_column_wrapper expected(expected_strings.data(),
                                              expected_strings.data() + expected_strings.size());
  auto results = cudf::concatenate(strings_columns);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
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

  CUDF_TEST_EXPECT_TABLES_EQUAL(*concat_table, gold_table);
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
    CUDF_TEST_EXPECT_TABLES_EQUAL(concatenated_tables->view(), table_view_exp1);
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
    CUDF_TEST_EXPECT_TABLES_EQUAL(concatenated_tables->view(), table_view_exp1);
  }
  {
    std::vector<cudf::table_view> table_views_to_concat;
    table_views_to_concat.push_back(partitioned1[1]);
    table_views_to_concat.push_back(partitioned2[0]);
    std::unique_ptr<cudf::table> concatenated_tables = cudf::concatenate(table_views_to_concat);

    column_wrapper<int32_t> exp1_1{{5, 8, 5, 6, 5, 8, 5}};
    cudf::test::strings_column_wrapper exp2_1({"dada", "kite", "dog", "ln", "dada", "kite", "dog"});
    cudf::table_view table_view_exp{{exp1_1, exp2_1}};
    CUDF_TEST_EXPECT_TABLES_EQUAL(concatenated_tables->view(), table_view_exp);
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
    CUDF_TEST_EXPECT_TABLES_EQUAL(concatenated_tables->view(), table_view_exp1);
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
    CUDF_TEST_EXPECT_TABLES_EQUAL(concatenated_tables->view(), table_view_exp1);
  }
}

TEST_F(TableTest, SizeOverflowTest)
{
  // primitive column
  {
    constexpr cudf::size_type size =
      static_cast<cudf::size_type>(static_cast<uint32_t>(1024) * 1024 * 1024);

    // try and concatenate 6 char columns of size 1 billion each
    auto many_chars = cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::INT8}, size);

    cudf::table_view tbl({*many_chars});
    EXPECT_THROW(cudf::concatenate({tbl, tbl, tbl, tbl, tbl, tbl}), cudf::logic_error);
  }

  // string column, overflow on chars
  {
    constexpr cudf::size_type size =
      static_cast<cudf::size_type>(static_cast<uint32_t>(1024) * 1024 * 1024);

    // try and concatenate 6 string columns of with 1 billion chars in each
    auto offsets    = cudf::test::fixed_width_column_wrapper<int>{0, size};
    auto many_chars = cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::INT8}, size);
    auto col        = cudf::make_strings_column(
      1, offsets.release(), std::move(many_chars), 0, rmm::device_buffer{0});

    cudf::table_view tbl({*col});
    EXPECT_THROW(cudf::concatenate({tbl, tbl, tbl, tbl, tbl, tbl}), cudf::logic_error);
  }

  // string column, overflow on offsets (rows)
  {
    constexpr cudf::size_type size =
      static_cast<cudf::size_type>(static_cast<uint32_t>(1024) * 1024 * 1024);

    // try and concatenate 6 string columns 1 billion rows each
    auto many_offsets =
      cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::INT32}, size + 1);
    auto chars = cudf::test::fixed_width_column_wrapper<int8_t>{0, 1, 2};
    auto col   = cudf::make_strings_column(
      size, std::move(many_offsets), chars.release(), 0, rmm::device_buffer{0});

    cudf::table_view tbl({*col});
    EXPECT_THROW(cudf::concatenate({tbl, tbl, tbl, tbl, tbl, tbl}), cudf::logic_error);
  }

  // list<struct>, structs too long
  {
    constexpr cudf::size_type inner_size =
      static_cast<cudf::size_type>(static_cast<uint32_t>(512) * 1024 * 1024);

    // struct
    std::vector<std::unique_ptr<column>> children;
    children.push_back(
      cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::INT8}, inner_size));
    children.push_back(
      cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::INT8}, inner_size));
    auto struct_col =
      cudf::make_structs_column(inner_size, std::move(children), 0, rmm::device_buffer{0});

    // list
    auto offsets = cudf::test::fixed_width_column_wrapper<int>{0, inner_size};
    auto col     = cudf::make_lists_column(
      1, offsets.release(), std::move(struct_col), 0, rmm::device_buffer{0});

    cudf::table_view tbl({*col});
    EXPECT_THROW(cudf::concatenate({tbl, tbl, tbl, tbl, tbl, tbl, tbl, tbl, tbl, tbl, tbl, tbl}),
                 cudf::logic_error);
  }

  // struct<int, list>, list child too long
  {
    constexpr cudf::size_type inner_size =
      static_cast<cudf::size_type>(static_cast<uint32_t>(512) * 1024 * 1024);
    constexpr cudf::size_type size = 3;

    // list
    auto offsets = cudf::test::fixed_width_column_wrapper<int>{0, 0, 0, inner_size};
    auto many_chars =
      cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::INT8}, inner_size);
    auto list_col = cudf::make_lists_column(
      3, offsets.release(), std::move(many_chars), 0, rmm::device_buffer{0});

    // struct
    std::vector<std::unique_ptr<column>> children;
    children.push_back(cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::INT32}, size));
    children.push_back(std::move(list_col));
    auto col = cudf::make_structs_column(size, std::move(children), 0, rmm::device_buffer{0});

    cudf::table_view tbl({*col});
    EXPECT_THROW(cudf::concatenate({tbl, tbl, tbl, tbl, tbl, tbl, tbl, tbl, tbl, tbl, tbl, tbl}),
                 cudf::logic_error);
  }
}

struct StructsColumnTest : public cudf::test::BaseFixture {
};

TEST_F(StructsColumnTest, ConcatenateStructs)
{
  using namespace cudf::test;

  auto count_iter = thrust::make_counting_iterator(0);

  // 1. String "names" column.
  std::vector<std::vector<std::string>> names(
    {{"Vimes", "Carrot"}, {"Angua", "Cheery"}, {}, {"Detritus", "Slant"}});
  std::vector<std::vector<bool>> names_validity({{1, 1}, {1, 1}, {}, {1, 1}});
  std::vector<strings_column_wrapper> name_cols;
  std::transform(count_iter, count_iter + names.size(), std::back_inserter(name_cols), [&](int i) {
    return strings_column_wrapper(names[i].begin(), names[i].end(), names_validity[i].begin());
  });

  // 2. Numeric "ages" column.
  std::vector<std::vector<int>> ages({{5, 10}, {15, 20}, {}, {25, 30}});
  std::vector<std::vector<bool>> ages_validity({{1, 1}, {1, 1}, {}, {0, 1}});
  std::vector<fixed_width_column_wrapper<int>> age_cols;
  std::transform(count_iter, count_iter + ages.size(), std::back_inserter(age_cols), [&](int i) {
    return fixed_width_column_wrapper<int>(
      ages[i].begin(), ages[i].end(), ages_validity[i].begin());
  });

  // 3. Boolean "is_human" column.
  std::vector<std::vector<bool>> is_human({{true, true}, {false, false}, {}, {false, false}});
  std::vector<std::vector<bool>> is_human_validity({{1, 1}, {1, 0}, {}, {1, 1}});
  std::vector<fixed_width_column_wrapper<bool>> is_human_cols;
  std::transform(
    count_iter, count_iter + is_human.size(), std::back_inserter(is_human_cols), [&](int i) {
      return fixed_width_column_wrapper<bool>(
        is_human[i].begin(), is_human[i].end(), is_human_validity[i].begin());
    });

  // build expected output
  std::vector<std::unique_ptr<column>> expected_children;
  expected_children.push_back(
    cudf::concatenate({name_cols[0], name_cols[1], name_cols[2], name_cols[3]}));
  expected_children.push_back(
    cudf::concatenate({age_cols[0], age_cols[1], age_cols[2], age_cols[3]}));
  expected_children.push_back(
    cudf::concatenate({is_human_cols[0], is_human_cols[1], is_human_cols[2], is_human_cols[3]}));
  std::vector<bool> struct_validity({1, 0, 1, 1, 1, 0});
  auto expected = make_structs_column(
    6,
    std::move(expected_children),
    2,
    cudf::test::detail::make_null_mask(struct_validity.begin(), struct_validity.end()));

  // concatenate as structs
  std::vector<structs_column_wrapper> src;
  src.push_back(structs_column_wrapper({name_cols[0], age_cols[0], is_human_cols[0]}, {1, 0}));
  src.push_back(structs_column_wrapper({name_cols[1], age_cols[1], is_human_cols[1]}, {1, 1}));
  src.push_back(structs_column_wrapper({name_cols[2], age_cols[2], is_human_cols[2]}, {}));
  src.push_back(structs_column_wrapper({name_cols[3], age_cols[3], is_human_cols[3]}, {1, 0}));

  // concatenate
  auto result = cudf::concatenate({src[0], src[1], src[2], src[3]});
  cudf::test::expect_columns_equivalent(*result, *expected);
}

TEST_F(StructsColumnTest, ConcatenateSplitStructs)
{
  using namespace cudf::test;

  auto count_iter = thrust::make_counting_iterator(0);

  std::vector<int> splits({2});

  // 1. String "names" column.
  std::vector<std::vector<std::string>> names(
    {{"Vimes", "Carrot", "Angua", "Cheery", "Detritus", "Slant"},
     {"Bill", "Bob", "Sam", "Fred", "Tom"}});
  std::vector<std::vector<bool>> names_validity({{1, 1, 1, 1, 1, 1}, {0, 1, 0, 1, 0}});
  std::vector<strings_column_wrapper> name_cols;
  std::transform(count_iter, count_iter + names.size(), std::back_inserter(name_cols), [&](int i) {
    return strings_column_wrapper(names[i].begin(), names[i].end(), names_validity[i].begin());
  });

  // 2. Numeric "ages" column.
  std::vector<std::vector<int>> ages({{5, 10, 15, 20, 25, 30}, {11, 16, 17, 41, 42}});
  std::vector<std::vector<bool>> ages_validity({{1, 1, 1, 1, 0, 1}, {1, 1, 1, 0, 0}});
  std::vector<fixed_width_column_wrapper<int>> age_cols;
  std::transform(count_iter, count_iter + ages.size(), std::back_inserter(age_cols), [&](int i) {
    return fixed_width_column_wrapper<int>(
      ages[i].begin(), ages[i].end(), ages_validity[i].begin());
  });

  // 3. Boolean "is_human" column.
  std::vector<std::vector<bool>> is_human(
    {{true, true, false, false, false, false}, {true, true, true, false, true}});
  std::vector<std::vector<bool>> is_human_validity({{1, 1, 1, 0, 1, 1}, {0, 0, 0, 1, 1}});
  std::vector<fixed_width_column_wrapper<bool>> is_human_cols;
  std::transform(
    count_iter, count_iter + is_human.size(), std::back_inserter(is_human_cols), [&](int i) {
      return fixed_width_column_wrapper<bool>(
        is_human[i].begin(), is_human[i].end(), is_human_validity[i].begin());
    });

  // split the columns, keep the one on the end
  std::vector<column_view> split_names_cols(
    {cudf::split(name_cols[0], splits)[1], cudf::split(name_cols[1], splits)[1]});
  std::vector<column_view> split_ages_cols(
    {cudf::split(age_cols[0], splits)[1], cudf::split(age_cols[1], splits)[1]});
  std::vector<column_view> split_is_human_cols(
    {cudf::split(is_human_cols[0], splits)[1], cudf::split(is_human_cols[1], splits)[1]});

  // build expected output
  std::vector<std::unique_ptr<column>> expected_children;
  expected_children.push_back(cudf::concatenate({split_names_cols[0], split_names_cols[1]}));
  expected_children.push_back(cudf::concatenate({split_ages_cols[0], split_ages_cols[1]}));
  expected_children.push_back(cudf::concatenate({split_is_human_cols[0], split_is_human_cols[1]}));
  auto expected = make_structs_column(7, std::move(expected_children), 0, rmm::device_buffer{});

  // concatenate as structs
  std::vector<structs_column_wrapper> src;
  for (size_t idx = 0; idx < split_names_cols.size(); idx++) {
    std::vector<std::unique_ptr<column>> inputs;
    inputs.push_back(std::make_unique<column>(split_names_cols[idx]));
    inputs.push_back(std::make_unique<column>(split_ages_cols[idx]));
    inputs.push_back(std::make_unique<column>(split_is_human_cols[idx]));
    src.push_back(structs_column_wrapper(std::move(inputs)));
  }

  // concatenate
  auto result = cudf::concatenate({src[0], src[1]});
  cudf::test::expect_columns_equivalent(*result, *expected);
}

TEST_F(StructsColumnTest, ConcatenateStructsNested)
{
  // includes Struct<Struct> and Struct<List>
  using namespace cudf::test;

  auto count_iter = thrust::make_counting_iterator(0);

  // inner structs
  std::vector<structs_column_wrapper> inner_structs;
  {
    // 1. String "names" column.
    std::vector<std::vector<std::string>> names(
      {{"Vimes", "Carrot", "Angua", "Cheery", "Detritus", "Slant"},
       {"Bill", "Bob", "Sam", "Fred", "Tom"}});
    std::vector<std::vector<bool>> names_validity({{1, 1, 1, 1, 1, 1}, {0, 1, 0, 1, 0}});
    std::vector<strings_column_wrapper> name_cols;
    std::transform(
      count_iter, count_iter + names.size(), std::back_inserter(name_cols), [&](int i) {
        return strings_column_wrapper(names[i].begin(), names[i].end(), names_validity[i].begin());
      });

    // 2. Numeric "ages" column.
    std::vector<std::vector<int>> ages({{5, 10, 15, 20, 25, 30}, {11, 16, 17, 41, 42}});
    std::vector<std::vector<bool>> ages_validity({{1, 1, 1, 1, 0, 1}, {1, 1, 1, 0, 0}});
    std::vector<fixed_width_column_wrapper<int>> age_cols;
    std::transform(count_iter, count_iter + ages.size(), std::back_inserter(age_cols), [&](int i) {
      return fixed_width_column_wrapper<int>(
        ages[i].begin(), ages[i].end(), ages_validity[i].begin());
    });

    for (size_t idx = 0; idx < names.size(); idx++) {
      std::vector<std::unique_ptr<column>> children;
      children.push_back(name_cols[idx].release());
      children.push_back(age_cols[idx].release());
      inner_structs.push_back(structs_column_wrapper(std::move(children)));
    }
  }

  // inner lists
  using LCW = lists_column_wrapper<cudf::string_view>;
  std::vector<lists_column_wrapper<cudf::string_view>> inner_lists;
  {
    inner_lists.push_back(lists_column_wrapper<cudf::string_view>{
      {"abc", "d"}, {"ef", "ghi", "j"}, {"klm", "no"}, LCW{}, LCW{"whee"}, {"xyz", "ab", "g"}});

    inner_lists.push_back(lists_column_wrapper<cudf::string_view>{
      {"er", "hyj"}, {"", "", "uvw"}, LCW{}, LCW{"oipq", "te"}, LCW{"yay", "bonk"}});
  }

  // build expected output
  std::vector<std::unique_ptr<column>> expected_children;
  expected_children.push_back(cudf::concatenate({inner_structs[0], inner_structs[1]}));
  expected_children.push_back(cudf::concatenate({inner_lists[0], inner_lists[1]}));
  auto expected = make_structs_column(11, std::move(expected_children), 0, rmm::device_buffer{});

  // concatenate as structs
  std::vector<structs_column_wrapper> src;
  for (size_t idx = 0; idx < inner_structs.size(); idx++) {
    std::vector<std::unique_ptr<column>> inputs;
    inputs.push_back(std::make_unique<column>(inner_structs[idx]));
    inputs.push_back(std::make_unique<column>(inner_lists[idx]));
    src.push_back(structs_column_wrapper(std::move(inputs)));
  }

  // concatenate
  auto result = cudf::concatenate({src[0], src[1]});
  cudf::test::expect_columns_equivalent(*result, *expected);
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

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
  }

  {
    cudf::test::lists_column_wrapper<int> a{{0, 1, 1}, {2, 3}, {4, 5}};
    cudf::test::lists_column_wrapper<int> b{{6}, {8, 9, 9, 9}, {10, 11}};
    cudf::test::lists_column_wrapper<int> expected{
      {0, 1, 1}, {2, 3}, {4, 5}, {6}, {8, 9, 9, 9}, {10, 11}};

    auto result = cudf::concatenate({a, b});

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
  }

  {
    cudf::test::lists_column_wrapper<int> a{{0, 1}, {2, 3, 4, 5}, {6, 7, 8}};
    cudf::test::lists_column_wrapper<int> b{{9}, {10, 11}, {12, 13, 14, 15}};
    cudf::test::lists_column_wrapper<int> expected{
      {0, 1}, {2, 3, 4, 5}, {6, 7, 8}, {9}, {10, 11}, {12, 13, 14, 15}};

    auto result = cudf::concatenate({a, b});

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
  }
}

TEST_F(ListsColumnTest, ConcatenateEmptyLists)
{
  // to disambiguate between {} == 0 and {} == List{0}
  // Also, see note about compiler issues when declaring nested
  // empty lists in lists_column_wrapper documentation
  using LCW = cudf::test::lists_column_wrapper<int>;
  {
    cudf::test::lists_column_wrapper<int> a;
    cudf::test::lists_column_wrapper<int> b{4, 5, 6, 7};
    cudf::test::lists_column_wrapper<int> expected{4, 5, 6, 7};

    auto result = cudf::concatenate({a, b});

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
  }

  {
    cudf::test::lists_column_wrapper<int> a, b, c;
    cudf::test::lists_column_wrapper<int> d{4, 5, 6, 7};
    cudf::test::lists_column_wrapper<int> expected{4, 5, 6, 7};

    auto result = cudf::concatenate({a, b, c, d});

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
  }

  {
    cudf::test::lists_column_wrapper<int> a{LCW{}};
    cudf::test::lists_column_wrapper<int> b{4, 5, 6, 7};
    cudf::test::lists_column_wrapper<int> expected{LCW{}, {4, 5, 6, 7}};

    auto result = cudf::concatenate({a, b});

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
  }

  {
    cudf::test::lists_column_wrapper<int> a{LCW{}}, b{LCW{}}, c{LCW{}};
    cudf::test::lists_column_wrapper<int> d{4, 5, 6, 7};
    cudf::test::lists_column_wrapper<int> expected{LCW{}, LCW{}, LCW{}, {4, 5, 6, 7}};

    auto result = cudf::concatenate({a, b, c, d});

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
  }

  {
    cudf::test::lists_column_wrapper<int> a{1, 2};
    cudf::test::lists_column_wrapper<int> b{LCW{}}, c{LCW{}};
    cudf::test::lists_column_wrapper<int> d{4, 5, 6, 7};
    cudf::test::lists_column_wrapper<int> expected{{1, 2}, LCW{}, LCW{}, {4, 5, 6, 7}};

    auto result = cudf::concatenate({a, b, c, d});

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
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

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
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

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
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

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
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

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
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

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
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

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
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

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
  }
}

TEST_F(ListsColumnTest, ConcatenateMismatchedHierarchies)
{
  // to disambiguate between {} == 0 and {} == List{0}
  // Also, see note about compiler issues when declaring nested
  // empty lists in lists_column_wrapper documentation
  using LCW = cudf::test::lists_column_wrapper<int>;
  {
    cudf::test::lists_column_wrapper<int> a{{{{LCW{}}}}};
    cudf::test::lists_column_wrapper<int> b{{{LCW{}}}};
    cudf::test::lists_column_wrapper<int> c{{LCW{}}};
    EXPECT_THROW(cudf::concatenate({a, b, c}), cudf::logic_error);
  }

  {
    std::vector<bool> valids{false};
    cudf::test::lists_column_wrapper<int> a{{{{{LCW{}}}}, valids.begin()}};
    cudf::test::lists_column_wrapper<int> b{{{LCW{}}}};
    cudf::test::lists_column_wrapper<int> c{{LCW{}}};
    EXPECT_THROW(cudf::concatenate({a, b, c}), cudf::logic_error);
  }

  {
    cudf::test::lists_column_wrapper<int> a{{{{LCW{}}}}};
    cudf::test::lists_column_wrapper<int> b{1, 2, 3};
    cudf::test::lists_column_wrapper<int> c{{3, 4, 5}};
    EXPECT_THROW(cudf::concatenate({a, b, c}), cudf::logic_error);
  }

  {
    cudf::test::lists_column_wrapper<int> a{{{1, 2, 3}}};
    cudf::test::lists_column_wrapper<int> b{{4, 5}};
    EXPECT_THROW(cudf::concatenate({a, b}), cudf::logic_error);
  }
}

TEST_F(ListsColumnTest, SlicedColumns)
{
  using LCW = cudf::test::lists_column_wrapper<int>;

  {
    cudf::test::lists_column_wrapper<int> a{{{1, 1, 1}, {2, 2}, {3, 3}},
                                            {{4, 4, 4}, {5, 5}, {6, 6}},
                                            {{7, 7, 7}, {8, 8}, {9, 9}},
                                            {{10, 10, 10}, {11, 11}, {12, 12}}};
    auto split_a = cudf::split(a, {2});

    cudf::test::lists_column_wrapper<int> b{{{-1, -1, -1, -1}, {-2}},
                                            {{-3, -3, -3, -3}, {-4}},
                                            {{-5, -5, -5, -5}, {-6}},
                                            {{-7, -7, -7, -7}, {-8}}};
    auto split_b = cudf::split(b, {2});

    cudf::test::lists_column_wrapper<int> expected0{{{1, 1, 1}, {2, 2}, {3, 3}},
                                                    {{4, 4, 4}, {5, 5}, {6, 6}},
                                                    {{-1, -1, -1, -1}, {-2}},
                                                    {{-3, -3, -3, -3}, {-4}}};
    auto result0 = cudf::concatenate({split_a[0], split_b[0]});
    cudf::test::expect_columns_equivalent(*result0, expected0);

    cudf::test::lists_column_wrapper<int> expected1{{{1, 1, 1}, {2, 2}, {3, 3}},
                                                    {{4, 4, 4}, {5, 5}, {6, 6}},
                                                    {{-5, -5, -5, -5}, {-6}},
                                                    {{-7, -7, -7, -7}, {-8}}};
    auto result1 = cudf::concatenate({split_a[0], split_b[1]});
    cudf::test::expect_columns_equivalent(*result1, expected1);

    cudf::test::lists_column_wrapper<int> expected2{
      {{7, 7, 7}, {8, 8}, {9, 9}},
      {{10, 10, 10}, {11, 11}, {12, 12}},
      {{-1, -1, -1, -1}, {-2}},
      {{-3, -3, -3, -3}, {-4}},
    };
    auto result2 = cudf::concatenate({split_a[1], split_b[0]});
    cudf::test::expect_columns_equivalent(*result2, expected2);

    cudf::test::lists_column_wrapper<int> expected3{{{7, 7, 7}, {8, 8}, {9, 9}},
                                                    {{10, 10, 10}, {11, 11}, {12, 12}},
                                                    {{-5, -5, -5, -5}, {-6}},
                                                    {{-7, -7, -7, -7}, {-8}}};
    auto result3 = cudf::concatenate({split_a[1], split_b[1]});
    cudf::test::expect_columns_equivalent(*result3, expected3);
  }

  {
    cudf::test::lists_column_wrapper<int> a{
      {{{1, 1, 1}, {2, 2}}, {{3, 3}}, {{10, 9, 16}, {8, 7, 1}, {6, 8, 2}}},
      {LCW{}, {LCW{}}, {{6, 6}, {2}}},
      {LCW{}, LCW{}},
      {LCW{}, LCW{}, {{10, 10, 10}, {11, 11}, {12, 12}}, LCW{}}};
    auto split_a = cudf::split(a, {2});

    cudf::test::lists_column_wrapper<int> b{
      {{LCW{}}},
      {LCW{}, {LCW{}}},
      {{{1, 2, 9}, LCW{}}, {{5, 6, 7, 8, 9}, {0}, {15, 17}}},
      {{LCW{}}},
    };
    auto split_b = cudf::split(b, {2});

    cudf::test::lists_column_wrapper<int> expected0{
      {{{1, 1, 1}, {2, 2}}, {{3, 3}}, {{10, 9, 16}, {8, 7, 1}, {6, 8, 2}}},
      {LCW{}, {LCW{}}, {{6, 6}, {2}}},
      {{LCW{}}},
      {LCW{}, {LCW{}}}};
    auto result0 = cudf::concatenate({split_a[0], split_b[0]});
    cudf::test::expect_columns_equivalent(*result0, expected0);

    cudf::test::lists_column_wrapper<int> expected1{
      {{{1, 1, 1}, {2, 2}}, {{3, 3}}, {{10, 9, 16}, {8, 7, 1}, {6, 8, 2}}},
      {LCW{}, {LCW{}}, {{6, 6}, {2}}},
      {{{1, 2, 9}, LCW{}}, {{5, 6, 7, 8, 9}, {0}, {15, 17}}},
      {{LCW{}}},
    };
    auto result1 = cudf::concatenate({split_a[0], split_b[1]});
    cudf::test::expect_columns_equivalent(*result1, expected1);

    cudf::test::lists_column_wrapper<int> expected2{
      {LCW{}, LCW{}},
      {LCW{}, LCW{}, {{10, 10, 10}, {11, 11}, {12, 12}}, LCW{}},
      {{LCW{}}},
      {LCW{}, {LCW{}}}};
    auto result2 = cudf::concatenate({split_a[1], split_b[0]});
    cudf::test::expect_columns_equivalent(*result2, expected2);

    cudf::test::lists_column_wrapper<int> expected3{
      {LCW{}, LCW{}},
      {LCW{}, LCW{}, {{10, 10, 10}, {11, 11}, {12, 12}}, LCW{}},
      {{{1, 2, 9}, LCW{}}, {{5, 6, 7, 8, 9}, {0}, {15, 17}}},
      {{LCW{}}},
    };
    auto result3 = cudf::concatenate({split_a[1], split_b[1]});
    cudf::test::expect_columns_equivalent(*result3, expected3);
  }
}

TEST_F(ListsColumnTest, SlicedColumnsWithNulls)
{
  using LCW = cudf::test::lists_column_wrapper<int>;

  auto valids = cudf::test::make_counting_transform_iterator(0, [](auto i) { return i % 2 == 0; });

  {
    cudf::test::lists_column_wrapper<int> a{{{{1, 1, 1}, valids}, {2, 2}, {{3, 3}, valids}},
                                            {{{4, 4, 4}, {{5, 5}, valids}, {6, 6}}, valids},
                                            {{7, 7, 7}, {8, 8}, {9, 9}},
                                            {{{10, 10, 10}, {11, 11}, {{12, 12}, valids}}, valids}};
    auto split_a = cudf::split(a, {3});

    cudf::test::lists_column_wrapper<int> b{{{{{-1, -1, -1, -1}, valids}, {-2}}, valids},
                                            {{{{-3, -3, -3, -3}, valids}, {-4}}, valids},
                                            {{{{-5, -5, -5, -5}, valids}, {-6}}, valids},
                                            {{{{-7, -7, -7, -7}, valids}, {-8}}, valids}};
    auto split_b = cudf::split(b, {3});

    cudf::test::lists_column_wrapper<int> expected0{{{{1, 1, 1}, valids}, {2, 2}, {{3, 3}, valids}},
                                                    {{{4, 4, 4}, {{5, 5}, valids}, {6, 6}}, valids},
                                                    {{7, 7, 7}, {8, 8}, {9, 9}},
                                                    {{{{-1, -1, -1, -1}, valids}, {-2}}, valids},
                                                    {{{{-3, -3, -3, -3}, valids}, {-4}}, valids},
                                                    {{{{-5, -5, -5, -5}, valids}, {-6}}, valids}};
    auto result0 = cudf::concatenate({split_a[0], split_b[0]});
    cudf::test::expect_columns_equivalent(*result0, expected0);

    cudf::test::lists_column_wrapper<int> expected1{{{{1, 1, 1}, valids}, {2, 2}, {{3, 3}, valids}},
                                                    {{{4, 4, 4}, {{5, 5}, valids}, {6, 6}}, valids},
                                                    {{7, 7, 7}, {8, 8}, {9, 9}},
                                                    {{{{-7, -7, -7, -7}, valids}, {-8}}, valids}};
    auto result1 = cudf::concatenate({split_a[0], split_b[1]});
    cudf::test::expect_columns_equivalent(*result1, expected1);

    cudf::test::lists_column_wrapper<int> expected2{
      {{{10, 10, 10}, {11, 11}, {{12, 12}, valids}}, valids},
      {{{{-1, -1, -1, -1}, valids}, {-2}}, valids},
      {{{{-3, -3, -3, -3}, valids}, {-4}}, valids},
      {{{{-5, -5, -5, -5}, valids}, {-6}}, valids}};
    auto result2 = cudf::concatenate({split_a[1], split_b[0]});
    cudf::test::expect_columns_equivalent(*result2, expected2);

    cudf::test::lists_column_wrapper<int> expected3{
      {{{10, 10, 10}, {11, 11}, {{12, 12}, valids}}, valids},
      {{{{-7, -7, -7, -7}, valids}, {-8}}, valids}};
    auto result3 = cudf::concatenate({split_a[1], split_b[1]});
    cudf::test::expect_columns_equivalent(*result3, expected3);
  }

  {
    cudf::test::lists_column_wrapper<int> a{
      {{{{1, 1, 1}, valids}, {2, 2}},
       {{{3, 3}}, valids},
       {{{10, 9, 16}, valids}, {8, 7, 1}, {{6, 8, 2}, valids}}},
      {{LCW{}, {{LCW{}}, valids}, {{6, 6}, {2}}}, valids},
      {{{LCW{}, LCW{}}, valids}},
      {LCW{}, LCW{}, {{{10, 10, 10}, {{11, 11}, valids}, {12, 12}}, valids}, LCW{}}};
    auto split_a = cudf::split(a, {3});

    cudf::test::lists_column_wrapper<int> b{
      {{{LCW{}}, valids}},
      {{LCW{}, {{LCW{}}, valids}}, valids},
      {{{{1, 2, 9}, LCW{}}, {{5, 6, 7, 8, 9}, {0}, {15, 17}}}, valids},
      {{LCW{}}},
    };
    auto split_b = cudf::split(b, {3});

    cudf::test::lists_column_wrapper<int> expected0{
      {{{{1, 1, 1}, valids}, {2, 2}},
       {{{3, 3}}, valids},
       {{{10, 9, 16}, valids}, {8, 7, 1}, {{6, 8, 2}, valids}}},
      {{LCW{}, {{LCW{}}, valids}, {{6, 6}, {2}}}, valids},
      {{{LCW{}, LCW{}}, valids}},
      {{{LCW{}}, valids}},
      {{LCW{}, {{LCW{}}, valids}}, valids},
      {{{{1, 2, 9}, LCW{}}, {{5, 6, 7, 8, 9}, {0}, {15, 17}}}, valids},
    };
    auto result0 = cudf::concatenate({split_a[0], split_b[0]});
    cudf::test::expect_columns_equivalent(*result0, expected0);

    cudf::test::lists_column_wrapper<int> expected1{
      {{{{1, 1, 1}, valids}, {2, 2}},
       {{{3, 3}}, valids},
       {{{10, 9, 16}, valids}, {8, 7, 1}, {{6, 8, 2}, valids}}},
      {{LCW{}, {{LCW{}}, valids}, {{6, 6}, {2}}}, valids},
      {{{LCW{}, LCW{}}, valids}},
      {{LCW{}}},
    };
    auto result1 = cudf::concatenate({split_a[0], split_b[1]});
    cudf::test::expect_columns_equivalent(*result1, expected1);

    cudf::test::lists_column_wrapper<int> expected2{
      {LCW{}, LCW{}, {{{10, 10, 10}, {{11, 11}, valids}, {12, 12}}, valids}, LCW{}},
      {{{LCW{}}, valids}},
      {{LCW{}, {{LCW{}}, valids}}, valids},
      {{{{1, 2, 9}, LCW{}}, {{5, 6, 7, 8, 9}, {0}, {15, 17}}}, valids},
    };
    auto result2 = cudf::concatenate({split_a[1], split_b[0]});
    cudf::test::expect_columns_equivalent(*result2, expected2);

    cudf::test::lists_column_wrapper<int> expected3{
      {LCW{}, LCW{}, {{{10, 10, 10}, {{11, 11}, valids}, {12, 12}}, valids}, LCW{}},
      {{LCW{}}},
    };
    auto result3 = cudf::concatenate({split_a[1], split_b[1]});
    cudf::test::expect_columns_equivalent(*result3, expected3);
  }
}

TEST_F(ListsColumnTest, ListOfStructs)
{
  using namespace cudf::test;

  auto count_iter = thrust::make_counting_iterator(0);

  // inner structs
  std::vector<structs_column_wrapper> inner_structs;
  {
    // 1. String "names" column.
    std::vector<std::vector<std::string>> names(
      {{"Vimes", "Carrot", "Angua", "Cheery", "Detritus", "Slant"},
       {},
       {},
       {"Bill", "Bob", "Sam", "Fred", "Tom"}});
    std::vector<std::vector<bool>> names_validity({{1, 1, 1, 1, 1, 1}, {}, {}, {0, 1, 0, 1, 0}});
    std::vector<strings_column_wrapper> name_cols;
    std::transform(
      count_iter, count_iter + names.size(), std::back_inserter(name_cols), [&](int i) {
        return strings_column_wrapper(names[i].begin(), names[i].end(), names_validity[i].begin());
      });

    // 2. Numeric "ages" column.
    std::vector<std::vector<int>> ages({{5, 10, 15, 20, 25, 30}, {}, {}, {11, 16, 17, 41, 42}});
    std::vector<std::vector<bool>> ages_validity({{1, 1, 1, 1, 0, 1}, {}, {}, {1, 1, 1, 0, 0}});
    std::vector<fixed_width_column_wrapper<int>> age_cols;
    std::transform(count_iter, count_iter + ages.size(), std::back_inserter(age_cols), [&](int i) {
      return fixed_width_column_wrapper<int>(
        ages[i].begin(), ages[i].end(), ages_validity[i].begin());
    });

    for (size_t idx = 0; idx < names.size(); idx++) {
      std::vector<std::unique_ptr<column>> children;
      children.push_back(name_cols[idx].release());
      children.push_back(age_cols[idx].release());
      inner_structs.push_back(structs_column_wrapper(std::move(children)));
    }
  }

  // build expected output
  auto expected_child =
    cudf::concatenate({inner_structs[0], inner_structs[1], inner_structs[2], inner_structs[3]});
  fixed_width_column_wrapper<int> offsets_w{0, 1, 1, 1, 1, 4, 6, 6, 6, 10, 11};
  auto expected = make_lists_column(
    10, std::move(offsets_w.release()), std::move(expected_child), 0, rmm::device_buffer{});

  // lists
  std::vector<fixed_width_column_wrapper<int>> offsets;
  offsets.push_back({0, 1, 1, 1, 1, 4, 6, 6});
  offsets.push_back({0});
  offsets.push_back({0});
  offsets.push_back({0, 0, 4, 5});

  // concatenate as lists
  std::vector<std::unique_ptr<cudf::column>> src;
  for (size_t idx = 0; idx < inner_structs.size(); idx++) {
    int size = static_cast<column_view>(offsets[idx]).size() - 1;
    src.push_back(make_lists_column(
      size, offsets[idx].release(), inner_structs[idx].release(), 0, rmm::device_buffer{}));
  }

  // concatenate
  auto result = cudf::concatenate({*src[0], *src[1], *src[2], *src[3]});
  cudf::test::expect_columns_equivalent(*result, *expected);
}

template <typename T>
struct FixedPointTestBothReps : public cudf::test::BaseFixture {
};

struct FixedPointTest : public cudf::test::BaseFixture {
};

TYPED_TEST_CASE(FixedPointTestBothReps, cudf::test::FixedPointTypes);

TYPED_TEST(FixedPointTestBothReps, FixedPointConcatentate)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<decimalXX>;

  auto begin = cudf::test::make_counting_transform_iterator(0, [](auto i) { return decimalXX{i}; });
  auto const vec = std::vector<decimalXX>(begin, begin + 1000);

  auto const a = fw_wrapper(vec.begin(), /***/ vec.begin() + 300);
  auto const b = fw_wrapper(vec.begin() + 300, vec.begin() + 700);
  auto const c = fw_wrapper(vec.begin() + 700, vec.end());

  auto const columns  = std::vector<cudf::column_view>{a, b, c};
  auto const results  = cudf::concatenate(columns);
  auto const expected = fw_wrapper(vec.begin(), vec.end());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(FixedPointTest, FixedPointConcatentate)
{
  using namespace numeric;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<int32_t>;

  auto begin     = thrust::make_counting_iterator(0);
  auto const vec = std::vector<int32_t>(begin, begin + 1000);

  auto const a = fp_wrapper(vec.begin(), /***/ vec.begin() + 300, scale_type{-2});
  auto const b = fp_wrapper(vec.begin() + 300, vec.begin() + 700, scale_type{-2});
  auto const c = fp_wrapper(vec.begin() + 700, vec.end(), /*****/ scale_type{-2});

  auto const columns  = std::vector<cudf::column_view>{a, b, c};
  auto const results  = cudf::concatenate(columns);
  auto const expected = fp_wrapper(vec.begin(), vec.end(), scale_type{-2});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(FixedPointTest, FixedPointScaleMismatch)
{
  using namespace numeric;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<int32_t>;

  auto begin     = thrust::make_counting_iterator(0);
  auto const vec = std::vector<int32_t>(begin, begin + 1000);

  auto const a = fp_wrapper(vec.begin(), /***/ vec.begin() + 300, scale_type{-1});
  auto const b = fp_wrapper(vec.begin() + 300, vec.begin() + 700, scale_type{-2});
  auto const c = fp_wrapper(vec.begin() + 700, vec.end(), /*****/ scale_type{-3});

  auto const columns = std::vector<cudf::column_view>{a, b, c};
  EXPECT_THROW(cudf::concatenate(columns), cudf::logic_error);
}

struct DictionaryConcatTest : public cudf::test::BaseFixture {
};

TEST_F(DictionaryConcatTest, StringsKeys)
{
  cudf::test::strings_column_wrapper strings(
    {"eee", "aaa", "ddd", "bbb", "", "", "ccc", "ccc", "ccc", "eee", "aaa"},
    {1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1});
  auto dictionary = cudf::dictionary::encode(strings);

  std::vector<cudf::size_type> splits{0, 2, 2, 5, 5, 7, 7, 7, 7, 11};
  std::vector<cudf::column_view> views = cudf::slice(dictionary->view(), splits);
  // concatenate should recreate the original column
  auto result  = cudf::concatenate(views);
  auto decoded = cudf::dictionary::decode(result->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*decoded, strings);
}

template <typename T>
struct DictionaryConcatTestFW : public cudf::test::BaseFixture {
};

TYPED_TEST_CASE(DictionaryConcatTestFW, cudf::test::FixedWidthTypes);

TYPED_TEST(DictionaryConcatTestFW, FixedWidthKeys)
{
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> original(
    {20, 10, 0, 5, 15, 15, 10, 5, 20}, {1, 1, 0, 1, 1, 1, 1, 1, 1});
  auto dictionary = cudf::dictionary::encode(original);
  std::vector<cudf::size_type> splits{0, 3, 3, 5, 5, 9};
  std::vector<cudf::column_view> views = cudf::slice(dictionary->view(), splits);
  // concatenated result should equal the original column
  auto result  = cudf::concatenate(views);
  auto decoded = cudf::dictionary::decode(result->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*decoded, original);
}

TEST_F(DictionaryConcatTest, ErrorsTest)
{
  cudf::test::strings_column_wrapper strings({"aaa", "ddd", "bbb"});
  auto dictionary1 = cudf::dictionary::encode(strings);
  cudf::test::fixed_width_column_wrapper<int32_t> integers({10, 30, 20});
  auto dictionary2 = cudf::dictionary::encode(integers);
  std::vector<cudf::column_view> views({dictionary1->view(), dictionary2->view()});
  EXPECT_THROW(cudf::concatenate(views), cudf::logic_error);
  std::vector<cudf::column_view> empty;
  EXPECT_THROW(cudf::concatenate(empty), cudf::logic_error);
}
