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
#include <cudf/utilities/type_dispatcher.hpp>

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/table_utilities.hpp>
#include <tests/utilities/type_lists.hpp>

#include <string>
#include <vector>

#include <tests/copying/slice_tests.cuh>

std::vector<cudf::size_type> splits_to_indices(std::vector<cudf::size_type> splits,
                                               cudf::size_type size)
{
  std::vector<cudf::size_type> indices{0};

  std::for_each(splits.begin(), splits.end(), [&indices](auto split) {
    indices.push_back(split);  // This for end
    indices.push_back(split);  // This for the start
  });

  indices.push_back(size);  // This to include rest of the elements

  return indices;
}

template <typename T>
std::vector<cudf::test::fixed_width_column_wrapper<T>> create_expected_columns_for_splits(
  std::vector<cudf::size_type> const& splits, cudf::size_type size, bool nullable)
{
  // convert splits to slice indices
  std::vector<cudf::size_type> indices = splits_to_indices(splits, size);
  return create_expected_columns<T>(indices, nullable);
}

std::vector<cudf::test::strings_column_wrapper> create_expected_string_columns_for_splits(
  std::vector<std::string> strings, std::vector<cudf::size_type> const& splits, bool nullable)
{
  std::vector<cudf::size_type> indices = splits_to_indices(splits, strings.size());
  return create_expected_string_columns(strings, indices, nullable);
}

template <typename T>
std::vector<cudf::table> create_expected_tables_for_splits(
  cudf::size_type num_cols,
  std::vector<cudf::size_type> const& splits,
  cudf::size_type col_size,
  bool nullable)
{
  std::vector<cudf::size_type> indices = splits_to_indices(splits, col_size);
  return create_expected_tables<T>(num_cols, indices, nullable);
}

std::vector<cudf::table> create_expected_string_tables_for_splits(
  std::vector<std::string> const strings[2],
  std::vector<cudf::size_type> const& splits,
  bool nullable)
{
  std::vector<cudf::size_type> indices = splits_to_indices(splits, strings[0].size());
  return create_expected_string_tables(strings, indices, nullable);
}

template <typename T>
struct SplitTest : public cudf::test::BaseFixture {
};

TYPED_TEST_CASE(SplitTest, cudf::test::NumericTypes);

TYPED_TEST(SplitTest, SplitEndLessThanSize)
{
  using T = TypeParam;

  cudf::size_type start = 0;
  cudf::size_type size  = 10;
  auto valids           = cudf::test::make_counting_transform_iterator(
    start, [](auto i) { return i % 2 == 0 ? true : false; });

  cudf::test::fixed_width_column_wrapper<T> col = create_fixed_columns<T>(start, size, valids);

  std::vector<cudf::size_type> splits{2, 5, 7};
  std::vector<cudf::test::fixed_width_column_wrapper<T>> expected =
    create_expected_columns_for_splits<T>(splits, size, true);
  std::vector<cudf::column_view> result = cudf::split(col, splits);

  EXPECT_EQ(expected.size(), result.size());

  for (unsigned long index = 0; index < result.size(); index++) {
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected[index], result[index]);
  }
}

TYPED_TEST(SplitTest, SplitEndToSize)
{
  using T = TypeParam;

  cudf::size_type start = 0;
  cudf::size_type size  = 10;
  auto valids           = cudf::test::make_counting_transform_iterator(
    start, [](auto i) { return i % 2 == 0 ? true : false; });

  cudf::test::fixed_width_column_wrapper<T> col = create_fixed_columns<T>(start, size, valids);

  std::vector<cudf::size_type> splits{2, 5, 10, 10, 10, 10};
  std::vector<cudf::test::fixed_width_column_wrapper<T>> expected =
    create_expected_columns_for_splits<T>(splits, size, true);
  std::vector<cudf::column_view> result = cudf::split(col, splits);

  EXPECT_EQ(expected.size(), result.size());

  for (unsigned long index = 0; index < result.size(); index++) {
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected[index], result[index]);
  }
}

struct SplitStringTest : public SplitTest<std::string> {
};

TEST_F(SplitStringTest, StringWithInvalids)
{
  std::vector<std::string> strings{
    "", "this", "is", "a", "column", "of", "strings", "with", "in", "valid"};
  auto valids = cudf::test::make_counting_transform_iterator(
    0, [](auto i) { return i % 2 == 0 ? true : false; });
  cudf::test::strings_column_wrapper s(strings.begin(), strings.end(), valids);

  std::vector<cudf::size_type> splits{2, 5, 9};

  std::vector<cudf::test::strings_column_wrapper> expected =
    create_expected_string_columns_for_splits(strings, splits, true);
  std::vector<cudf::column_view> result = cudf::split(s, splits);

  EXPECT_EQ(expected.size(), result.size());

  for (unsigned long index = 0; index < result.size(); index++) {
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected[index], result[index]);
  }
}

struct SplitListTest : public SplitTest<int> {
};

TEST_F(SplitListTest, Lists)
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

    std::vector<cudf::size_type> splits{0, 1, 4, 5, 6, 9};

    std::vector<cudf::test::lists_column_wrapper<int>> expected;
    expected.push_back(LCW{});
    expected.push_back(LCW{{1, 2, 3}});
    expected.push_back(LCW{{4, 5}, {6}, {7, 8}});
    expected.push_back(LCW{{9, 10, 11}});
    expected.push_back(LCW{LCW{}});
    expected.push_back(LCW{LCW{}, {-1, -2, -3, -4, -5}, {-10}});
    expected.push_back(LCW{{-100, -200}});

    std::vector<cudf::column_view> result = cudf::split(list, splits);
    EXPECT_EQ(expected.size(), result.size());

    for (unsigned long index = 0; index < result.size(); index++) {
      cudf::test::expect_columns_equivalent(expected[index], result[index]);
    }
  }

  {
    cudf::test::lists_column_wrapper<int> list{{{1, 2, 3}, {4, 5}},
                                               {LCW{}, LCW{}, {7, 8}, LCW{}},
                                               {LCW{6}},
                                               {{7, 8}, {9, 10, 11}, LCW{}},
                                               {LCW{}, {-1, -2, -3, -4, -5}},
                                               {LCW{}},
                                               {{-10}, {-100, -200}}};

    std::vector<cudf::size_type> splits{1, 3, 4};

    std::vector<cudf::test::lists_column_wrapper<int>> expected;
    expected.push_back(LCW{{{1, 2, 3}, {4, 5}}});
    expected.push_back(LCW{{LCW{}, LCW{}, {7, 8}, LCW{}}, {LCW{6}}});
    expected.push_back(LCW{{{7, 8}, {9, 10, 11}, LCW{}}});
    expected.push_back(LCW{{LCW{}, {-1, -2, -3, -4, -5}}, {LCW{}}, {{-10}, {-100, -200}}});

    std::vector<cudf::column_view> result = cudf::split(list, splits);
    EXPECT_EQ(expected.size(), result.size());

    for (unsigned long index = 0; index < result.size(); index++) {
      cudf::test::expect_columns_equivalent(expected[index], result[index]);
    }
  }
}

TEST_F(SplitListTest, ListsWithNulls)
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

    std::vector<cudf::size_type> splits{0, 1, 4, 5, 6, 9};

    std::vector<cudf::test::lists_column_wrapper<int>> expected;
    expected.push_back(LCW{});
    expected.push_back(LCW{{1, 2, 3}});
    expected.push_back(LCW{{4, 5}, {6}, {{7, 8}, valids}});
    expected.push_back(LCW{{9, 10, 11}});
    expected.push_back(LCW{LCW{}});
    expected.push_back(LCW{LCW{}, {{-1, -2, -3, -4, -5}, valids}, {-10}});
    expected.push_back(LCW{{{-100, -200}, valids}});

    std::vector<cudf::column_view> result = cudf::split(list, splits);
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

    std::vector<cudf::size_type> splits{1, 3, 4};

    std::vector<cudf::test::lists_column_wrapper<int>> expected;
    expected.push_back(LCW{{{{1, 2, 3}, valids}, {4, 5}}});
    expected.push_back(LCW{{{LCW{}, LCW{}, {7, 8}, LCW{}}, valids}, {{{6}}}});
    expected.push_back(LCW{{{{7, 8}, {{9, 10, 11}, valids}, LCW{}}, valids}});
    expected.push_back(
      LCW{{{LCW{}, {-1, -2, -3, -4, -5}}, valids}, {LCW{}}, {{-10}, {-100, -200}}});

    std::vector<cudf::column_view> result = cudf::split(list, splits);
    EXPECT_EQ(expected.size(), result.size());

    for (unsigned long index = 0; index < result.size(); index++) {
      cudf::test::expect_columns_equivalent(expected[index], result[index]);
    }
  }
}

struct SplitCornerCases : public SplitTest<int8_t> {
};

TEST_F(SplitCornerCases, EmptyColumn)
{
  cudf::column col{};
  std::vector<cudf::size_type> splits{2, 5, 9};

  std::vector<cudf::column_view> result = cudf::split(col.view(), splits);

  unsigned long expected = 1;

  EXPECT_EQ(expected, result.size());
}

TEST_F(SplitCornerCases, EmptyIndices)
{
  cudf::size_type start = 0;
  cudf::size_type size  = 10;
  auto valids           = cudf::test::make_counting_transform_iterator(
    start, [](auto i) { return i % 2 == 0 ? true : false; });

  cudf::test::fixed_width_column_wrapper<int8_t> col =
    create_fixed_columns<int8_t>(start, size, valids);
  std::vector<cudf::size_type> splits{};

  std::vector<cudf::column_view> result = cudf::split(col, splits);

  unsigned long expected = 1;

  EXPECT_EQ(expected, result.size());
}

TEST_F(SplitCornerCases, InvalidSetOfIndices)
{
  cudf::size_type start = 0;
  cudf::size_type size  = 10;
  auto valids           = cudf::test::make_counting_transform_iterator(
    start, [](auto i) { return i % 2 == 0 ? true : false; });
  cudf::test::fixed_width_column_wrapper<int8_t> col =
    create_fixed_columns<int8_t>(start, size, valids);
  std::vector<cudf::size_type> splits{11, 12};

  EXPECT_THROW(cudf::split(col, splits), cudf::logic_error);
}

TEST_F(SplitCornerCases, ImproperRange)
{
  cudf::size_type start = 0;
  cudf::size_type size  = 10;
  auto valids           = cudf::test::make_counting_transform_iterator(
    start, [](auto i) { return i % 2 == 0 ? true : false; });

  cudf::test::fixed_width_column_wrapper<int8_t> col =
    create_fixed_columns<int8_t>(start, size, valids);
  std::vector<cudf::size_type> splits{5, 4};

  EXPECT_THROW(cudf::split(col, splits), cudf::logic_error);
}

TEST_F(SplitCornerCases, NegativeValue)
{
  cudf::size_type start = 0;
  cudf::size_type size  = 10;
  auto valids           = cudf::test::make_counting_transform_iterator(
    start, [](auto i) { return i % 2 == 0 ? true : false; });

  cudf::test::fixed_width_column_wrapper<int8_t> col =
    create_fixed_columns<int8_t>(start, size, valids);
  std::vector<cudf::size_type> splits{-1, 4};

  EXPECT_THROW(cudf::split(col, splits), cudf::logic_error);
}

// common functions for testing split/contiguous_split
template <typename T, typename SplitFunc, typename CompareFunc>
void split_end_less_than_size(SplitFunc Split, CompareFunc Compare)
{
  cudf::size_type start    = 0;
  cudf::size_type col_size = 10;
  auto valids              = cudf::test::make_counting_transform_iterator(
    start, [](auto i) { return i % 2 == 0 ? true : false; });

  cudf::size_type num_cols = 5;
  cudf::table src_table    = create_fixed_table<T>(num_cols, start, col_size, valids);

  std::vector<cudf::size_type> splits{2, 5, 7};
  std::vector<cudf::table> expected =
    create_expected_tables_for_splits<T>(num_cols, splits, col_size, true);

  auto result = Split(src_table, splits);

  EXPECT_EQ(expected.size(), result.size());

  for (unsigned long index = 0; index < result.size(); index++) {
    Compare(expected[index], result[index]);
  }
}

template <typename T, typename SplitFunc, typename CompareFunc>
void split_end_to_size(SplitFunc Split, CompareFunc Compare)
{
  cudf::size_type start    = 0;
  cudf::size_type col_size = 10;
  auto valids              = cudf::test::make_counting_transform_iterator(
    start, [](auto i) { return i % 2 == 0 ? true : false; });

  cudf::size_type num_cols = 5;
  cudf::table src_table    = create_fixed_table<T>(num_cols, start, col_size, valids);

  std::vector<cudf::size_type> splits{2, 5, 10};
  std::vector<cudf::table> expected =
    create_expected_tables_for_splits<T>(num_cols, splits, col_size, true);

  auto result = Split(src_table, splits);

  EXPECT_EQ(expected.size(), result.size());

  for (unsigned long index = 0; index < result.size(); index++) {
    Compare(expected[index], result[index]);
  }
}

template <typename SplitFunc>
void split_empty_table(SplitFunc Split)
{
  std::vector<cudf::size_type> splits{2, 5, 9};

  cudf::table src_table{};
  auto result = Split(src_table, splits);

  unsigned long expected = 0;

  EXPECT_EQ(expected, result.size());
}

template <typename SplitFunc>
void split_empty_indices(SplitFunc Split)
{
  cudf::size_type start    = 0;
  cudf::size_type col_size = 10;
  auto valids              = cudf::test::make_counting_transform_iterator(
    start, [](auto i) { return i % 2 == 0 ? true : false; });

  cudf::size_type num_cols = 5;
  cudf::table src_table    = create_fixed_table<int8_t>(num_cols, start, col_size, valids);
  std::vector<cudf::size_type> splits{};

  auto result = Split(src_table, splits);

  unsigned long expected = 1;

  EXPECT_EQ(expected, result.size());
}

template <typename SplitFunc>
void split_invalid_indices(SplitFunc Split)
{
  cudf::size_type start    = 0;
  cudf::size_type col_size = 10;
  auto valids              = cudf::test::make_counting_transform_iterator(
    start, [](auto i) { return i % 2 == 0 ? true : false; });

  cudf::size_type num_cols = 5;
  cudf::table src_table    = create_fixed_table<int8_t>(num_cols, start, col_size, valids);

  std::vector<cudf::size_type> splits{11, 12};

  EXPECT_THROW(Split(src_table, splits), cudf::logic_error);
}

template <typename SplitFunc>
void split_improper_range(SplitFunc Split)
{
  cudf::size_type start    = 0;
  cudf::size_type col_size = 10;
  auto valids              = cudf::test::make_counting_transform_iterator(
    start, [](auto i) { return i % 2 == 0 ? true : false; });

  cudf::size_type num_cols = 5;
  cudf::table src_table    = create_fixed_table<int8_t>(num_cols, start, col_size, valids);

  std::vector<cudf::size_type> splits{5, 4};

  EXPECT_THROW(Split(src_table, splits), cudf::logic_error);
}

template <typename SplitFunc>
void split_negative_value(SplitFunc Split)
{
  cudf::size_type start    = 0;
  cudf::size_type col_size = 10;
  auto valids              = cudf::test::make_counting_transform_iterator(
    start, [](auto i) { return i % 2 == 0 ? true : false; });

  cudf::size_type num_cols = 5;
  cudf::table src_table    = create_fixed_table<int8_t>(num_cols, start, col_size, valids);

  std::vector<cudf::size_type> splits{-1, 4};

  EXPECT_THROW(Split(src_table, splits), cudf::logic_error);
}

template <typename SplitFunc, typename CompareFunc>
void split_empty_output_column_value(SplitFunc Split, CompareFunc Compare)
{
  cudf::size_type start    = 0;
  cudf::size_type col_size = 10;
  auto valids              = cudf::test::make_counting_transform_iterator(
    start, [](auto i) { return i % 2 == 0 ? true : false; });

  cudf::size_type num_cols = 5;
  cudf::table src_table    = create_fixed_table<int8_t>(num_cols, start, col_size, valids);

  std::vector<cudf::size_type> splits{0, 2, 2};

  EXPECT_NO_THROW(Split(src_table, splits));

  auto result = Split(src_table, splits);
  EXPECT_NO_THROW(Compare(result[0], num_cols));
}

// regular splits
template <typename T>
struct SplitTableTest : public cudf::test::BaseFixture {
};
TYPED_TEST_CASE(SplitTableTest, cudf::test::NumericTypes);

TYPED_TEST(SplitTableTest, SplitEndLessThanSize)
{
  split_end_less_than_size<TypeParam>(
    [](cudf::table_view const& t, std::vector<cudf::size_type> const& splits) {
      return cudf::split(t, splits);
    },
    [](cudf::table_view const& expected, cudf::table_view const& result) {
      CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result);
    });
}

TYPED_TEST(SplitTableTest, SplitEndToSize)
{
  split_end_to_size<TypeParam>(
    [](cudf::table_view const& t, std::vector<cudf::size_type> const& splits) {
      return cudf::split(t, splits);
    },
    [](cudf::table_view const& expected, cudf::table_view const& result) {
      CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result);
    });
}

struct SplitTableCornerCases : public SplitTest<int8_t> {
};

TEST_F(SplitTableCornerCases, EmptyTable)
{
  split_empty_table([](cudf::table_view const& t, std::vector<cudf::size_type> const& splits) {
    return cudf::split(t, splits);
  });
}

TEST_F(SplitTableCornerCases, EmptyIndices)
{
  split_empty_indices([](cudf::table_view const& t, std::vector<cudf::size_type> const& splits) {
    return cudf::split(t, splits);
  });
}

TEST_F(SplitTableCornerCases, InvalidSetOfIndices)
{
  split_invalid_indices([](cudf::table_view const& t, std::vector<cudf::size_type> const& splits) {
    return cudf::split(t, splits);
  });
}

TEST_F(SplitTableCornerCases, ImproperRange)
{
  split_improper_range([](cudf::table_view const& t, std::vector<cudf::size_type> const& splits) {
    return cudf::split(t, splits);
  });
}

TEST_F(SplitTableCornerCases, NegativeValue)
{
  split_negative_value([](cudf::table_view const& t, std::vector<cudf::size_type> const& splits) {
    return cudf::split(t, splits);
  });
}

TEST_F(SplitTableCornerCases, EmptyOutputColumn)
{
  split_empty_output_column_value(
    [](cudf::table_view const& t, std::vector<cudf::size_type> const& splits) {
      return cudf::split(t, splits);
    },
    [](cudf::table_view const& t, int num_cols) { EXPECT_EQ(t.num_columns(), num_cols); });
}

template <typename SplitFunc, typename CompareFunc>
void split_string_with_invalids(SplitFunc Split, CompareFunc Compare)
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

  std::vector<cudf::size_type> splits{2, 5, 9};

  std::vector<cudf::table> expected =
    create_expected_string_tables_for_splits(strings, splits, true);

  auto result = Split(src_table, splits);

  EXPECT_EQ(expected.size(), result.size());

  for (unsigned long index = 0; index < result.size(); index++) {
    Compare(expected[index], result[index]);
  }
}

template <typename SplitFunc, typename CompareFunc>
void split_empty_output_strings_column_value(SplitFunc Split, CompareFunc Compare)
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

  cudf::size_type num_cols = 2;

  std::vector<cudf::size_type> splits{0, 2, 2};

  EXPECT_NO_THROW(Split(src_table, splits));

  auto result = Split(src_table, splits);
  EXPECT_NO_THROW(Compare(result[0], num_cols));
}

// split with strings
struct SplitStringTableTest : public SplitTest<std::string> {
};

TEST_F(SplitStringTableTest, StringWithInvalids)
{
  split_string_with_invalids(
    [](cudf::table_view const& t, std::vector<cudf::size_type> const& splits) {
      return cudf::split(t, splits);
    },
    [](cudf::table_view const& expected, cudf::table_view const& result) {
      CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result);
    });
}

TEST_F(SplitStringTableTest, EmptyOutputColumn)
{
  split_empty_output_strings_column_value(
    [](cudf::table_view const& t, std::vector<cudf::size_type> const& splits) {
      return cudf::split(t, splits);
    },
    [](cudf::table_view const& t, int num_cols) { EXPECT_EQ(t.num_columns(), num_cols); });
}

// contiguous split with strings
struct ContiguousSplitStringTableTest : public SplitTest<std::string> {
};

TEST_F(ContiguousSplitStringTableTest, StringWithInvalids)
{
  split_string_with_invalids(
    [](cudf::table_view const& t, std::vector<cudf::size_type> const& splits) {
      return cudf::contiguous_split(t, splits);
    },
    [](cudf::table_view const& expected, cudf::contiguous_split_result const& result) {
      CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.table);
    });
}

TEST_F(ContiguousSplitStringTableTest, EmptyOutputColumn)
{
  split_empty_output_strings_column_value(
    [](cudf::table_view const& t, std::vector<cudf::size_type> const& splits) {
      return cudf::contiguous_split(t, splits);
    },
    [](cudf::contiguous_split_result const& t, int num_cols) {
      EXPECT_EQ(t.table.num_columns(), num_cols);
    });
}

// contiguous splits
template <typename T>
struct ContiguousSplitTableTest : public cudf::test::BaseFixture {
};
TYPED_TEST_CASE(ContiguousSplitTableTest, cudf::test::NumericTypes);

TYPED_TEST(ContiguousSplitTableTest, SplitEndLessThanSize)
{
  split_end_less_than_size<TypeParam>(
    [](cudf::table_view const& t, std::vector<cudf::size_type> const& splits) {
      return cudf::contiguous_split(t, splits);
    },
    [](cudf::table_view const& expected, cudf::contiguous_split_result const& result) {
      CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.table);
    });
}

TYPED_TEST(ContiguousSplitTableTest, SplitEndToSize)
{
  split_end_to_size<TypeParam>(
    [](cudf::table_view const& t, std::vector<cudf::size_type> const& splits) {
      return cudf::contiguous_split(t, splits);
    },
    [](cudf::table_view const& expected, cudf::contiguous_split_result const& result) {
      CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.table);
    });
}

struct ContiguousSplitTableCornerCases : public SplitTest<int8_t> {
};

TEST_F(ContiguousSplitTableCornerCases, EmptyTable)
{
  split_empty_table([](cudf::table_view const& t, std::vector<cudf::size_type> const& splits) {
    return cudf::contiguous_split(t, splits);
  });
}

TEST_F(ContiguousSplitTableCornerCases, EmptyIndices)
{
  split_empty_indices([](cudf::table_view const& t, std::vector<cudf::size_type> const& splits) {
    return cudf::contiguous_split(t, splits);
  });
}

TEST_F(ContiguousSplitTableCornerCases, InvalidSetOfIndices)
{
  split_invalid_indices([](cudf::table_view const& t, std::vector<cudf::size_type> const& splits) {
    return cudf::contiguous_split(t, splits);
  });
}

TEST_F(ContiguousSplitTableCornerCases, ImproperRange)
{
  split_improper_range([](cudf::table_view const& t, std::vector<cudf::size_type> const& splits) {
    return cudf::contiguous_split(t, splits);
  });
}

TEST_F(ContiguousSplitTableCornerCases, NegativeValue)
{
  split_negative_value([](cudf::table_view const& t, std::vector<cudf::size_type> const& splits) {
    return cudf::contiguous_split(t, splits);
  });
}

TEST_F(ContiguousSplitTableCornerCases, EmptyOutputColumn)
{
  split_empty_output_column_value(
    [](cudf::table_view const& t, std::vector<cudf::size_type> const& splits) {
      return cudf::contiguous_split(t, splits);
    },
    [](cudf::contiguous_split_result const& t, int num_cols) {
      EXPECT_EQ(t.table.num_columns(), num_cols);
    });
}

TEST_F(ContiguousSplitTableCornerCases, MixedColumnTypes)
{
  cudf::size_type start = 0;
  auto valids = cudf::test::make_counting_transform_iterator(start, [](auto i) { return true; });

  std::vector<std::string> strings[2] = {
    {"", "this", "is", "a", "column", "of", "strings", "with", "in", "valid"},
    {"", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"}};

  std::vector<std::unique_ptr<cudf::column>> cols;

  auto iter0 = cudf::test::make_counting_transform_iterator(0, [](auto i) { return (i); });
  auto c0    = cudf::test::fixed_width_column_wrapper<int>(iter0, iter0 + 10, valids);
  cols.push_back(c0.release());

  auto iter1 = cudf::test::make_counting_transform_iterator(10, [](auto i) { return (i); });
  auto c1    = cudf::test::fixed_width_column_wrapper<int>(iter1, iter1 + 10, valids);
  cols.push_back(c1.release());

  auto c2 = cudf::test::strings_column_wrapper(strings[0].begin(), strings[0].end(), valids);
  cols.push_back(c2.release());

  auto c3 = cudf::test::strings_column_wrapper(strings[1].begin(), strings[1].end(), valids);
  cols.push_back(c3.release());

  auto iter4 = cudf::test::make_counting_transform_iterator(20, [](auto i) { return (i); });
  auto c4    = cudf::test::fixed_width_column_wrapper<int>(iter4, iter4 + 10, valids);
  cols.push_back(c4.release());

  auto tbl = cudf::table(std::move(cols));

  std::vector<cudf::size_type> splits{5};

  auto result   = cudf::contiguous_split(tbl, splits);
  auto expected = cudf::split(tbl, splits);

  for (unsigned long index = 0; index < expected.size(); index++) {
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected[index], result[index].table);
  }
}
