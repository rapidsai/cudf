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
#include <cudf/copying.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>
#include <tests/copying/slice_tests.cuh>

#include <string>
#include <vector>

template <typename T>
struct SliceTest : public cudf::test::BaseFixture {
};

TYPED_TEST_CASE(SliceTest, cudf::test::NumericTypes);

TYPED_TEST(SliceTest, NumericColumnsWithNulls)
{
  using T = TypeParam;

  cudf::size_type start = 0;
  cudf::size_type size  = 10;
  auto valids           = cudf::test::make_counting_transform_iterator(
    start, [](auto i) { return i % 2 == 0 ? true : false; });

  cudf::test::fixed_width_column_wrapper<T> col = create_fixed_columns<T>(start, size, valids);

  std::vector<cudf::size_type> indices{1, 3, 2, 2, 5, 9};
  std::vector<cudf::test::fixed_width_column_wrapper<T>> expected =
    create_expected_columns<T>(indices, true);
  std::vector<cudf::column_view> result = cudf::slice(col, indices);

  EXPECT_EQ(expected.size(), result.size());

  for (unsigned long index = 0; index < result.size(); index++) {
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected[index], result[index]);
  }
}

TYPED_TEST(SliceTest, NumericColumnsWithNullsAsColumn)
{
  using T = TypeParam;

  cudf::size_type start = 0;
  cudf::size_type size  = 10;
  auto valids           = cudf::test::make_counting_transform_iterator(
    start, [](auto i) { return i % 2 == 0 ? true : false; });

  cudf::test::fixed_width_column_wrapper<T> col = create_fixed_columns<T>(start, size, valids);

  std::vector<cudf::size_type> indices{1, 3, 2, 2, 5, 9};
  std::vector<cudf::test::fixed_width_column_wrapper<T>> expected =
    create_expected_columns<T>(indices, true);
  std::vector<cudf::column_view> result = cudf::slice(col, indices);

  EXPECT_EQ(expected.size(), result.size());

  for (unsigned long index = 0; index < result.size(); index++) {
    auto col = cudf::column(result[index]);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected[index], col);
  }
}

struct SliceStringTest : public SliceTest<std::string> {
};

TEST_F(SliceStringTest, StringWithNulls)
{
  std::vector<std::string> strings{
    "", "this", "is", "a", "column", "of", "strings", "with", "in", "valid"};
  auto valids = cudf::test::make_counting_transform_iterator(
    0, [](auto i) { return i % 2 == 0 ? true : false; });
  cudf::test::strings_column_wrapper s(strings.begin(), strings.end(), valids);

  std::vector<cudf::size_type> indices{1, 3, 2, 4, 1, 9};

  std::vector<cudf::test::strings_column_wrapper> expected =
    create_expected_string_columns(strings, indices, true);
  std::vector<cudf::column_view> result = cudf::slice(s, indices);

  EXPECT_EQ(expected.size(), result.size());

  for (unsigned long index = 0; index < result.size(); index++) {
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected[index], result[index]);
  }
}

TEST_F(SliceStringTest, StringWithNullsAsColumn)
{
  std::vector<std::string> strings{
    "", "this", "is", "a", "column", "of", "strings", "with", "in", "valid"};
  auto valids = cudf::test::make_counting_transform_iterator(
    0, [](auto i) { return i % 2 == 0 ? true : false; });
  cudf::test::strings_column_wrapper s(strings.begin(), strings.end(), valids);

  std::vector<cudf::size_type> indices{1, 3, 2, 4, 1, 9};

  std::vector<cudf::test::strings_column_wrapper> expected =
    create_expected_string_columns(strings, indices, true);
  std::vector<cudf::column_view> result = cudf::slice(s, indices);

  EXPECT_EQ(expected.size(), result.size());

  for (unsigned long index = 0; index < result.size(); index++) {
    // this materializes a column to test slicing + materialization
    auto result_col = cudf::column(result[index]);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected[index], result_col);
  }
}

struct SliceListTest : public SliceTest<int> {
};

TEST_F(SliceListTest, Lists)
{
  using LCW = cudf::test::lists_column_wrapper<int>;

  {
    cudf::test::lists_column_wrapper<int> list{{1, 2, 3},
                                               {4, 5},
                                               {6},
                                               {7, 8},
                                               {9, 10, 11},
                                               LCW{},
                                               LCW{},
                                               {-1, -2, -3, -4, -5},
                                               {-10},
                                               {-100, -200}};

    std::vector<cudf::size_type> indices{1, 3, 2, 4, 1, 9};

    std::vector<cudf::test::lists_column_wrapper<int>> expected;
    expected.push_back(LCW{{4, 5}, {6}});
    expected.push_back(LCW{{6}, {7, 8}});
    expected.push_back(
      LCW{{4, 5}, {6}, {7, 8}, {9, 10, 11}, LCW{}, LCW{}, {-1, -2, -3, -4, -5}, {-10}});

    std::vector<cudf::column_view> result = cudf::slice(list, indices);
    EXPECT_EQ(expected.size(), result.size());

    for (unsigned long index = 0; index < result.size(); index++) {
      cudf::test::expect_columns_equivalent(expected[index], result[index]);
    }
  }

  {
    cudf::test::lists_column_wrapper<int> list{{{1, 2, 3}, {4, 5}},
                                               {LCW{}, LCW{}, {7, 8}, LCW{}},
                                               {{{6}}},
                                               {{7, 8}, {9, 10, 11}, LCW{}},
                                               {LCW{}, {-1, -2, -3, -4, -5}},
                                               {LCW{}},
                                               {{-10}, {-100, -200}}};

    std::vector<cudf::size_type> indices{1, 3, 3, 6};

    std::vector<cudf::test::lists_column_wrapper<int>> expected;
    expected.push_back(LCW{{LCW{}, LCW{}, {7, 8}, LCW{}}, {{{6}}}});
    expected.push_back(LCW{{{7, 8}, {9, 10, 11}, LCW{}}, {LCW{}, {-1, -2, -3, -4, -5}}, {LCW{}}});

    std::vector<cudf::column_view> result = cudf::slice(list, indices);
    EXPECT_EQ(expected.size(), result.size());

    for (unsigned long index = 0; index < result.size(); index++) {
      cudf::test::expect_columns_equivalent(expected[index], result[index]);
    }
  }
}

TEST_F(SliceListTest, ListsWithNulls)
{
  using LCW = cudf::test::lists_column_wrapper<int>;

  auto valids = cudf::test::make_counting_transform_iterator(0, [](auto i) { return i % 2 == 0; });

  {
    cudf::test::lists_column_wrapper<int> list{{1, 2, 3},
                                               {4, 5},
                                               {6},
                                               {{7, 8}, valids},
                                               {9, 10, 11},
                                               LCW{},
                                               LCW{},
                                               {{-1, -2, -3, -4, -5}, valids},
                                               {-10},
                                               {{-100, -200}, valids}};

    std::vector<cudf::size_type> indices{1, 3, 2, 4, 1, 9};

    std::vector<cudf::test::lists_column_wrapper<int>> expected;
    expected.push_back(LCW{{4, 5}, {6}});
    expected.push_back(LCW{{6}, {{7, 8}, valids}});
    expected.push_back(LCW{{4, 5},
                           {6},
                           {{7, 8}, valids},
                           {9, 10, 11},
                           LCW{},
                           LCW{},
                           {{-1, -2, -3, -4, -5}, valids},
                           {-10}});

    std::vector<cudf::column_view> result = cudf::slice(list, indices);
    EXPECT_EQ(expected.size(), result.size());

    for (unsigned long index = 0; index < result.size(); index++) {
      cudf::test::expect_columns_equivalent(expected[index], result[index]);
    }
  }

  {
    cudf::test::lists_column_wrapper<int> list{{{{1, 2, 3}, valids}, {4, 5}},
                                               {{LCW{}, LCW{}, {7, 8}, LCW{}}, valids},
                                               {{{6}}},
                                               {{{7, 8}, {{9, 10, 11}, valids}, LCW{}}, valids},
                                               {{LCW{}, {-1, -2, -3, -4, -5}}, valids},
                                               {LCW{}},
                                               {{-10}, {-100, -200}}};

    std::vector<cudf::size_type> indices{1, 3, 3, 6};

    std::vector<cudf::test::lists_column_wrapper<int>> expected;
    expected.push_back(LCW{{{LCW{}, LCW{}, {7, 8}, LCW{}}, valids}, {{{6}}}});
    expected.push_back(LCW{{{{7, 8}, {{9, 10, 11}, valids}, LCW{}}, valids},
                           {{LCW{}, {-1, -2, -3, -4, -5}}, valids},
                           {LCW{}}});

    std::vector<cudf::column_view> result = cudf::slice(list, indices);
    EXPECT_EQ(expected.size(), result.size());

    for (unsigned long index = 0; index < result.size(); index++) {
      cudf::test::expect_columns_equivalent(expected[index], result[index]);
    }
  }
}

struct SliceCornerCases : public SliceTest<int8_t> {
};

TEST_F(SliceCornerCases, EmptyColumn)
{
  cudf::column col{};
  std::vector<cudf::size_type> indices{0, 0, 0, 0, 0, 0};

  std::vector<cudf::column_view> result = cudf::slice(col.view(), indices);

  unsigned long expected = 3;

  EXPECT_EQ(expected, result.size());

  auto type_match_count = std::count_if(result.cbegin(), result.cend(), [](auto const& col) {
    return col.type().id() == cudf::type_id::EMPTY;
  });
  EXPECT_EQ(static_cast<std::size_t>(type_match_count), expected);
}

TEST_F(SliceCornerCases, EmptyIndices)
{
  cudf::size_type start = 0;
  cudf::size_type size  = 10;
  auto valids           = cudf::test::make_counting_transform_iterator(
    start, [](auto i) { return i % 2 == 0 ? true : false; });

  cudf::test::fixed_width_column_wrapper<int8_t> col =
    create_fixed_columns<int8_t>(start, size, valids);
  std::vector<cudf::size_type> indices{};

  std::vector<cudf::column_view> result = cudf::slice(col, indices);

  unsigned long expected = 0;

  EXPECT_EQ(expected, result.size());
}

TEST_F(SliceCornerCases, InvalidSetOfIndices)
{
  cudf::size_type start = 0;
  cudf::size_type size  = 10;
  auto valids           = cudf::test::make_counting_transform_iterator(
    start, [](auto i) { return i % 2 == 0 ? true : false; });
  cudf::test::fixed_width_column_wrapper<int8_t> col =
    create_fixed_columns<int8_t>(start, size, valids);
  std::vector<cudf::size_type> indices{11, 12};

  EXPECT_THROW(cudf::slice(col, indices), cudf::logic_error);
}

TEST_F(SliceCornerCases, ImproperRange)
{
  cudf::size_type start = 0;
  cudf::size_type size  = 10;
  auto valids           = cudf::test::make_counting_transform_iterator(
    start, [](auto i) { return i % 2 == 0 ? true : false; });

  cudf::test::fixed_width_column_wrapper<int8_t> col =
    create_fixed_columns<int8_t>(start, size, valids);
  std::vector<cudf::size_type> indices{5, 4};

  EXPECT_THROW(cudf::slice(col, indices), cudf::logic_error);
}

TEST_F(SliceCornerCases, NegativeOffset)
{
  cudf::size_type start = 0;
  cudf::size_type size  = 10;
  auto valids           = cudf::test::make_counting_transform_iterator(
    start, [](auto i) { return i % 2 == 0 ? true : false; });

  cudf::test::fixed_width_column_wrapper<int8_t> col =
    create_fixed_columns<int8_t>(start, size, valids);
  std::vector<cudf::size_type> indices{-1, 4};

  EXPECT_THROW(cudf::slice(col, indices), cudf::logic_error);
}

template <typename T>
struct SliceTableTest : public cudf::test::BaseFixture {
};

TYPED_TEST_CASE(SliceTableTest, cudf::test::NumericTypes);

TYPED_TEST(SliceTableTest, NumericColumnsWithNulls)
{
  using T = TypeParam;

  cudf::size_type start    = 0;
  cudf::size_type col_size = 10;
  auto valids              = cudf::test::make_counting_transform_iterator(
    start, [](auto i) { return i % 2 == 0 ? true : false; });

  cudf::size_type num_cols = 5;
  cudf::table src_table    = create_fixed_table<T>(num_cols, start, col_size, valids);

  std::vector<cudf::size_type> indices{1, 3, 2, 2, 5, 9};
  std::vector<cudf::table> expected = create_expected_tables<T>(num_cols, indices, true);

  std::vector<cudf::table_view> result = cudf::slice(src_table, indices);

  EXPECT_EQ(expected.size(), result.size());

  for (unsigned long index = 0; index < result.size(); index++) {
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected[index], result[index]);
  }
}

struct SliceStringTableTest : public SliceTableTest<std::string> {
};

TEST_F(SliceStringTableTest, StringWithNulls)
{
  auto valids = cudf::test::make_counting_transform_iterator(
    0, [](auto i) { return i % 2 == 0 ? true : false; });

  std::vector<std::string> strings[2] = {
    {"", "this", "is", "a", "column", "of", "strings", "with", "in", "valid"},
    {"", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"}};
  cudf::test::strings_column_wrapper sw[2] = {{strings[0].begin(), strings[0].end(), valids},
                                              {strings[1].begin(), strings[1].end(), valids}};

  std::vector<std::unique_ptr<cudf::column>> scols;
  scols.push_back(sw[0].release());
  scols.push_back(sw[1].release());
  cudf::table src_table(std::move(scols));

  std::vector<cudf::size_type> indices{1, 3, 2, 4, 1, 9};

  std::vector<cudf::table> expected = create_expected_string_tables(strings, indices, true);

  std::vector<cudf::table_view> result = cudf::slice(src_table, indices);

  EXPECT_EQ(expected.size(), result.size());

  for (unsigned long index = 0; index < result.size(); index++) {
    CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(expected[index], result[index]);
  }
}

struct SliceTableCornerCases : public SliceTableTest<int8_t> {
};

TEST_F(SliceTableCornerCases, EmptyTable)
{
  std::vector<cudf::size_type> indices{1, 3, 2, 4, 5, 9};

  cudf::table src_table{};
  std::vector<cudf::table_view> result = cudf::slice(src_table.view(), indices);

  unsigned long expected = 3;

  EXPECT_EQ(expected, result.size());
}

TEST_F(SliceTableCornerCases, EmptyIndices)
{
  cudf::size_type start    = 0;
  cudf::size_type col_size = 10;
  auto valids              = cudf::test::make_counting_transform_iterator(
    start, [](auto i) { return i % 2 == 0 ? true : false; });

  cudf::size_type num_cols = 5;
  cudf::table src_table    = create_fixed_table<int8_t>(num_cols, start, col_size, valids);
  std::vector<cudf::size_type> indices{};

  std::vector<cudf::table_view> result = cudf::slice(src_table, indices);

  unsigned long expected = 0;

  EXPECT_EQ(expected, result.size());
}

TEST_F(SliceTableCornerCases, InvalidSetOfIndices)
{
  cudf::size_type start    = 0;
  cudf::size_type col_size = 10;
  auto valids              = cudf::test::make_counting_transform_iterator(
    start, [](auto i) { return i % 2 == 0 ? true : false; });

  cudf::size_type num_cols = 5;
  cudf::table src_table    = create_fixed_table<int8_t>(num_cols, start, col_size, valids);

  std::vector<cudf::size_type> indices{11, 12};

  EXPECT_THROW(cudf::slice(src_table, indices), cudf::logic_error);
}

TEST_F(SliceTableCornerCases, ImproperRange)
{
  cudf::size_type start    = 0;
  cudf::size_type col_size = 10;
  auto valids              = cudf::test::make_counting_transform_iterator(
    start, [](auto i) { return i % 2 == 0 ? true : false; });

  cudf::size_type num_cols = 5;
  cudf::table src_table    = create_fixed_table<int8_t>(num_cols, start, col_size, valids);

  std::vector<cudf::size_type> indices{5, 4};

  EXPECT_THROW(cudf::slice(src_table, indices), cudf::logic_error);
}

TEST_F(SliceTableCornerCases, NegativeOffset)
{
  cudf::size_type start    = 0;
  cudf::size_type col_size = 10;
  auto valids              = cudf::test::make_counting_transform_iterator(
    start, [](auto i) { return i % 2 == 0 ? true : false; });

  cudf::size_type num_cols = 5;
  cudf::table src_table    = create_fixed_table<int8_t>(num_cols, start, col_size, valids);

  std::vector<cudf::size_type> indices{-1, 4};

  EXPECT_THROW(cudf::slice(src_table, indices), cudf::logic_error);
}

TEST_F(SliceTableCornerCases, MiscOffset)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col2{
    {3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
     3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0}};
  cudf::test::fixed_width_column_wrapper<int32_t> col3{
    {3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0}};
  std::vector<cudf::size_type> indices{19, 38};
  std::vector<cudf::column_view> result = cudf::slice(col2, indices);
  cudf::column result_column(result[0]);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(col3, result_column);
}
