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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_list_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <rmm/device_buffer.hpp>

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

template <typename T>
std::vector<cudf::test::fixed_width_column_wrapper<T>> create_expected_columns_for_splits(
  std::vector<cudf::size_type> const& splits, std::vector<T> const& elements, bool nullable)
{
  // convert splits to slice indices
  std::vector<cudf::size_type> indices = splits_to_indices(splits, elements.size());
  return create_expected_columns<T>(indices, elements.begin(), nullable);
}

template <typename T>
std::vector<cudf::test::fixed_width_column_wrapper<T>> create_expected_columns_for_splits(
  std::vector<cudf::size_type> const& splits,
  cudf::size_type size,
  std::vector<bool> const& validity)
{
  // convert splits to slice indices
  std::vector<cudf::size_type> indices = splits_to_indices(splits, size);
  return create_expected_columns<T>(indices, validity);
}

template <typename T>
std::vector<cudf::test::fixed_width_column_wrapper<T>> create_expected_columns_for_splits(
  std::vector<cudf::size_type> const& splits,
  std::vector<T> const& elements,
  std::vector<bool> const& validity)
{
  // convert splits to slice indices
  std::vector<cudf::size_type> indices = splits_to_indices(splits, elements.size());
  return create_expected_columns<T>(indices, elements.begin(), validity);
}

std::vector<cudf::test::strings_column_wrapper> create_expected_string_columns_for_splits(
  std::vector<std::string> strings, std::vector<cudf::size_type> const& splits, bool nullable)
{
  std::vector<cudf::size_type> indices = splits_to_indices(splits, strings.size());
  return create_expected_string_columns(strings, indices, nullable);
}

std::vector<cudf::test::strings_column_wrapper> create_expected_string_columns_for_splits(
  std::vector<std::string> strings,
  std::vector<cudf::size_type> const& splits,
  std::vector<bool> const& validity)
{
  std::vector<cudf::size_type> indices = splits_to_indices(splits, strings.size());
  return create_expected_string_columns(strings, indices, validity);
}

std::vector<std::vector<bool>> create_expected_validity(std::vector<cudf::size_type> const& splits,
                                                        std::vector<bool> const& validity)
{
  std::vector<std::vector<bool>> result = {};
  std::vector<cudf::size_type> indices  = splits_to_indices(splits, validity.size());

  for (unsigned long index = 0; index < indices.size(); index += 2) {
    result.push_back(
      std::vector<bool>(validity.begin() + indices[index], validity.begin() + indices[index + 1]));
  }

  return std::move(result);
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

// common functions for testing split/contiguous_split
template <typename T, typename SplitFunc, typename CompareFunc>
void split_long_column(SplitFunc Split, CompareFunc Compare)
{
  // the intent here is to stress the various boundary conditions in contiguous_split -
  // especially the validity copying code.
  cudf::size_type start = 0;
  cudf::size_type size  = 10002;

  srand(824);
  auto rvalids = cudf::test::make_counting_transform_iterator(start, [](auto i) {
    return static_cast<float>(rand()) / static_cast<float>(RAND_MAX) < 0.5f ? 0 : 1;
  });
  std::vector<bool> valids{rvalids, rvalids + size};
  cudf::test::fixed_width_column_wrapper<T> col =
    create_fixed_columns<T>(start, size, valids.begin());
  std::vector<cudf::size_type> splits{
    2, 16, 31, 35, 64, 97, 158, 190, 638, 899, 900, 901, 996, 4200, 7131, 8111};
  std::vector<cudf::test::fixed_width_column_wrapper<T>> expected =
    create_expected_columns_for_splits<T>(splits, size, valids);

  auto result = Split(col, splits);

  EXPECT_EQ(expected.size(), result.size());

  for (unsigned long index = 0; index < result.size(); index++) {
    Compare(expected[index], result[index]);
  }
}

TYPED_TEST(SplitTest, LongColumn)
{
  split_long_column<TypeParam>(
    [](cudf::column_view const& t, std::vector<cudf::size_type> const& splits) {
      return cudf::split(t, splits);
    },
    [](cudf::column_view const& expected, cudf::column_view const& result) {
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result);
    });
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

struct SplitNestedTypesTest : public cudf::test::BaseFixture {
};

// common functions for testing split/contiguous_split
template <typename T, typename SplitFunc, typename CompareFunc>
void split_lists(SplitFunc Split, CompareFunc Compare)
{
  using LCW = cudf::test::lists_column_wrapper<T>;

  {
    cudf::test::lists_column_wrapper<T> list{{1, 2, 3},
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

    std::vector<cudf::test::lists_column_wrapper<T>> expected;
    expected.push_back(LCW{});
    expected.push_back(LCW{{1, 2, 3}});
    expected.push_back(LCW{{4, 5}, {6}, {7, 8}});
    expected.push_back(LCW{{9, 10, 11}});
    expected.push_back(LCW{LCW{}});
    expected.push_back(LCW{LCW{}, {-1, -2, -3, -4, -5}, {-10}});
    expected.push_back(LCW{{-100, -200}});

    auto result = Split(list, splits);
    EXPECT_EQ(expected.size(), result.size());

    for (unsigned long index = 0; index < result.size(); index++) {
      Compare(expected[index], result[index]);
    }
  }

  {
    cudf::test::lists_column_wrapper<T> list{{{1, 2, 3}, {4, 5}},
                                             {LCW{}, LCW{}, {7, 8}, LCW{}},
                                             {LCW{6}},
                                             {{7, 8}, {9, 10, 11}, LCW{}},
                                             {LCW{}, {-1, -2, -3, -4, -5}},
                                             {LCW{}},
                                             {{-10}, {-100, -200}}};

    std::vector<cudf::size_type> splits{1, 3, 4};

    std::vector<cudf::test::lists_column_wrapper<T>> expected;
    expected.push_back(LCW{{{1, 2, 3}, {4, 5}}});
    expected.push_back(LCW{{LCW{}, LCW{}, {7, 8}, LCW{}}, {LCW{6}}});
    expected.push_back(LCW{{{7, 8}, {9, 10, 11}, LCW{}}});
    expected.push_back(LCW{{LCW{}, {-1, -2, -3, -4, -5}}, {LCW{}}, {{-10}, {-100, -200}}});

    auto result = Split(list, splits);
    EXPECT_EQ(expected.size(), result.size());

    for (unsigned long index = 0; index < result.size(); index++) {
      Compare(expected[index], result[index]);
    }
  }
}

template <typename T, typename SplitFunc, typename CompareFunc>
void split_lists_with_nulls(SplitFunc Split, CompareFunc Compare)
{
  using LCW = cudf::test::lists_column_wrapper<int>;

  auto valids = cudf::test::make_counting_transform_iterator(0, [](auto i) { return i % 2 == 0; });

  {
    cudf::test::lists_column_wrapper<T> list{{1, 2, 3},
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

    std::vector<cudf::test::lists_column_wrapper<T>> expected;
    expected.push_back(LCW{});
    expected.push_back(LCW{{1, 2, 3}});
    expected.push_back(LCW{{4, 5}, {6}, {{7, 8}, valids}});
    expected.push_back(LCW{{9, 10, 11}});
    expected.push_back(LCW{LCW{}});
    expected.push_back(LCW{LCW{}, {{-1, -2, -3, -4, -5}, valids}, {-10}});
    expected.push_back(LCW{{{-100, -200}, valids}});

    auto result = Split(list, splits);
    EXPECT_EQ(expected.size(), result.size());

    for (unsigned long index = 0; index < result.size(); index++) {
      Compare(expected[index], result[index]);
    }
  }

  {
    cudf::test::lists_column_wrapper<T> list{{{{1, 2, 3}, valids}, {4, 5}},
                                             {{LCW{}, LCW{}, {7, 8}, LCW{}}, valids},
                                             {{{6}}},
                                             {{{7, 8}, {{9, 10, 11}, valids}, LCW{}}, valids},
                                             {{LCW{}, {-1, -2, -3, -4, -5}}, valids},
                                             {LCW{}},
                                             {{-10}, {-100, -200}}};

    std::vector<cudf::size_type> splits{1, 3, 4};

    std::vector<cudf::test::lists_column_wrapper<T>> expected;
    expected.push_back(LCW{{{{1, 2, 3}, valids}, {4, 5}}});
    expected.push_back(LCW{{{LCW{}, LCW{}, {7, 8}, LCW{}}, valids}, {{{6}}}});
    expected.push_back(LCW{{{{7, 8}, {{9, 10, 11}, valids}, LCW{}}, valids}});
    expected.push_back(
      LCW{{{LCW{}, {-1, -2, -3, -4, -5}}, valids}, {LCW{}}, {{-10}, {-100, -200}}});

    auto result = Split(list, splits);
    EXPECT_EQ(expected.size(), result.size());

    for (unsigned long index = 0; index < result.size(); index++) {
      Compare(expected[index], result[index]);
    }
  }
}

template <typename SplitFunc, typename CompareFunc>
void split_structs(bool include_validity, SplitFunc Split, CompareFunc Compare)
{
  using namespace cudf::test;

  // 1. String "names" column.
  std::vector<std::string> names{
    "Vimes", "Carrot", "Angua", "Cheery", "Detritus", "Slant", "Fred", "Todd", "Kevin"};
  std::vector<bool> names_validity{1, 1, 1, 1, 1, 1, 1, 1, 1};
  strings_column_wrapper names_column(names.begin(), names.end());

  // 2. Numeric "ages" column.
  std::vector<int> ages{5, 10, 15, 20, 25, 30, 100, 101, 102};
  std::vector<bool> ages_validity = {1, 1, 1, 1, 0, 1, 0, 0, 1};
  auto ages_column                = include_validity ? fixed_width_column_wrapper<int>(
                                          ages.begin(), ages.end(), ages_validity.begin())
                                      : fixed_width_column_wrapper<int>(ages.begin(), ages.end());

  // 3. Boolean "is_human" column.
  std::vector<bool> is_human{true, true, false, false, false, false, true, true, true};
  std::vector<bool> is_human_validity{1, 1, 1, 0, 1, 1, 1, 1, 0};
  auto is_human_col = include_validity
                        ? fixed_width_column_wrapper<bool>(
                            is_human.begin(), is_human.end(), is_human_validity.begin())
                        : fixed_width_column_wrapper<bool>(is_human.begin(), is_human.end());

  // Assemble struct column.
  auto const struct_validity = std::vector<bool>{1, 1, 1, 1, 1, 0, 0, 1, 0};
  auto struct_column =
    include_validity
      ? structs_column_wrapper({names_column, ages_column, is_human_col}, struct_validity.begin())
      : structs_column_wrapper({names_column, ages_column, is_human_col});

  // split
  std::vector<cudf::size_type> splits{0, 1, 3, 8};
  auto result = Split(struct_column, splits);

  // expected outputs
  auto expected_names = include_validity
                          ? create_expected_string_columns_for_splits(names, splits, names_validity)
                          : create_expected_string_columns_for_splits(names, splits, false);
  auto expected_ages = include_validity
                         ? create_expected_columns_for_splits<int>(splits, ages, ages_validity)
                         : create_expected_columns_for_splits<int>(splits, ages, false);
  auto expected_is_human =
    include_validity ? create_expected_columns_for_splits<bool>(splits, is_human, is_human_validity)
                     : create_expected_columns_for_splits<bool>(splits, is_human, false);

  auto expected_struct_validity = create_expected_validity(splits, struct_validity);

  EXPECT_EQ(expected_names.size(), result.size());

  for (unsigned long index = 0; index < result.size(); index++) {
    auto expected = include_validity
                      ? structs_column_wrapper(
                          {expected_names[index], expected_ages[index], expected_is_human[index]},
                          expected_struct_validity[index])
                      : structs_column_wrapper(
                          {expected_names[index], expected_ages[index], expected_is_human[index]});

    Compare(expected, result[index]);
  }
}

template <typename SplitFunc, typename CompareFunc>
void split_nested_struct_of_list(SplitFunc Split, CompareFunc Compare)
{
  // Struct<List<List>>
  using namespace cudf::test;
  using LCW = cudf::test::lists_column_wrapper<float>;

  // 1. String "names" column.
  std::vector<std::string> names{
    "Vimes", "Carrot", "Angua", "Cheery", "Detritus", "Slant", "Fred", "Todd", "Kevin"};
  std::vector<bool> names_validity{1, 1, 1, 1, 1, 1, 1, 1, 1};
  strings_column_wrapper names_column(names.begin(), names.end());

  // 2. Numeric "ages" column.
  std::vector<int> ages{5, 10, 15, 20, 25, 30, 100, 101, 102};
  std::vector<bool> ages_validity = {1, 1, 1, 1, 0, 1, 0, 0, 1};
  auto ages_column =
    fixed_width_column_wrapper<int>(ages.begin(), ages.end(), ages_validity.begin());

  // 3. List column
  std::vector<bool> list_validity{1, 1, 1, 1, 1, 0, 1, 0, 1};
  lists_column_wrapper<float> list({{{1, 2, 3}, {4}},
                                    {{-1, -2}, LCW{}},
                                    LCW{},
                                    {{10}, {20, 30, 40}, {100, -100}},
                                    {LCW{}, LCW{}, {8, 9}},
                                    LCW{},
                                    {{8}, {10, 9, 8, 7, 6, 5}},
                                    {{5, 6}, LCW{}, {8}},
                                    {LCW{-3, 4, -5}}},
                                   list_validity.begin());

  // Assemble struct column.
  auto const struct_validity = std::vector<bool>{1, 1, 1, 1, 1, 0, 0, 1, 0};
  auto struct_column =
    structs_column_wrapper({names_column, ages_column, list}, struct_validity.begin());

  // split
  std::vector<cudf::size_type> splits{1, 3, 8};
  auto result = Split(struct_column, splits);

  // expected results
  auto expected_names = create_expected_string_columns_for_splits(names, splits, names_validity);
  auto expected_ages  = create_expected_columns_for_splits<int>(splits, ages, ages_validity);
  std::vector<lists_column_wrapper<float>> expected_lists;
  expected_lists.push_back(LCW({{{1, 2, 3}, {4}}}));
  expected_lists.push_back(LCW({{{-1, -2}, LCW{}}, LCW{}}));
  std::vector<bool> ex_v{1, 1, 0, 1, 0};
  expected_lists.push_back(LCW({{{10}, {20, 30, 40}, {100, -100}},
                                {LCW{}, LCW{}, {8, 9}},
                                LCW{},
                                {{8}, {10, 9, 8, 7, 6, 5}},
                                {{5, 6}, LCW{}, {8}}},
                               ex_v.begin()));
  expected_lists.push_back(LCW({{LCW{-3, 4, -5}}}));

  auto expected_struct_validity = create_expected_validity(splits, struct_validity);
  EXPECT_EQ(expected_names.size(), result.size());

  for (int index = 0; index < result.size(); index++) {
    auto expected =
      structs_column_wrapper({expected_names[index], expected_ages[index], expected_lists[index]},
                             expected_struct_validity[index]);
    Compare(expected, result[index]);
  }
}

TEST_F(SplitNestedTypesTest, Lists)
{
  split_lists<int>(
    [](cudf::column_view const& t, std::vector<cudf::size_type> const& splits) {
      return cudf::split(t, splits);
    },
    [](cudf::column_view const& expected, cudf::column_view const& result) {
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result);
    });
}

TEST_F(SplitNestedTypesTest, ListsWithNulls)
{
  split_lists_with_nulls<int>(
    [](cudf::column_view const& t, std::vector<cudf::size_type> const& splits) {
      return cudf::split(t, splits);
    },
    [](cudf::column_view const& expected, cudf::column_view const& result) {
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result);
    });
}

TEST_F(SplitNestedTypesTest, Structs)
{
  split_structs(
    false,
    [](cudf::column_view const& t, std::vector<cudf::size_type> const& splits) {
      return cudf::split(t, splits);
    },
    [](cudf::column_view const& expected, cudf::column_view const& result) {
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result);
    });
}

TEST_F(SplitNestedTypesTest, StructsWithNulls)
{
  split_structs(
    true,
    [](cudf::column_view const& t, std::vector<cudf::size_type> const& splits) {
      return cudf::split(t, splits);
    },
    [](cudf::column_view const& expected, cudf::column_view const& result) {
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result);
    });
}

TEST_F(SplitNestedTypesTest, StructsOfList)
{
  split_nested_struct_of_list(
    [](cudf::column_view const& t, std::vector<cudf::size_type> const& splits) {
      return cudf::split(t, splits);
    },
    [](cudf::column_view const& expected, cudf::column_view const& result) {
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result);
    });
}

template <typename T>
struct ContiguousSplitTest : public cudf::test::BaseFixture {
};

// the various utility functions in slice_tests.cuh don't like the chrono types
using FixedWidthTypesWithoutChrono =
  cudf::test::Concat<cudf::test::NumericTypes, cudf::test::FixedPointTypes>;

TYPED_TEST_CASE(ContiguousSplitTest, FixedWidthTypesWithoutChrono);

TYPED_TEST(ContiguousSplitTest, LongColumn)
{
  split_long_column<TypeParam>(
    [](cudf::column_view const& c, std::vector<cudf::size_type> const& splits) {
      cudf::table_view t({c});
      return cudf::contiguous_split(t, splits);
    },
    [](cudf::column_view const& expected, cudf::contiguous_split_result const& result) {
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result.table.column(0));
    });
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

TEST_F(ContiguousSplitStringTableTest, EmptyInputColumn)
{
  // build a bunch of empty stuff
  cudf::test::strings_column_wrapper sw;
  cudf::test::lists_column_wrapper<int> lw;
  cudf::test::fixed_width_column_wrapper<float> fw;
  //
  cudf::test::strings_column_wrapper ssw;
  cudf::test::lists_column_wrapper<int> slw;
  cudf::test::fixed_width_column_wrapper<float> sfw;
  cudf::test::structs_column_wrapper st_w({sfw, ssw, slw});

  cudf::table_view src_table({sw, lw, fw, st_w});

  {
    std::vector<cudf::size_type> splits;
    auto result = cudf::contiguous_split(src_table, splits);
    CUDF_EXPECTS(result.size() == 1, "Incorrect returned contiguous_split_result size!");

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(src_table, result[0].table);
  }

  {
    std::vector<cudf::size_type> splits{0, 0, 0, 0};
    auto result = cudf::contiguous_split(src_table, splits);
    CUDF_EXPECTS(result.size() == 5, "Incorrect returned contiguous_split_result size!");

    for (size_t idx = 0; idx < result.size(); idx++) {
      CUDF_TEST_EXPECT_TABLES_EQUIVALENT(src_table, result[idx].table);
    }
  }
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
TYPED_TEST_CASE(ContiguousSplitTableTest, FixedWidthTypesWithoutChrono);

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

TEST_F(ContiguousSplitTableCornerCases, PreSplitTable)
{
  // test splitting a table that is already split (has an offset)
  cudf::size_type start        = 0;
  cudf::size_type presplit_pos = 47;
  cudf::size_type size         = 10002;

  srand(824);
  auto rvalids = cudf::test::make_counting_transform_iterator(start, [](auto i) {
    return static_cast<float>(rand()) / static_cast<float>(RAND_MAX) < 0.5f ? 0 : 1;
  });

  std::vector<bool> pre_split_valids{rvalids, rvalids + size};
  cudf::test::fixed_width_column_wrapper<int> pre_split =
    create_fixed_columns<int>(start, size, pre_split_valids.begin());

  // pre-split this column
  auto split_cols = cudf::split(pre_split, {47});

  std::vector<cudf::size_type> splits{
    2, 16, 31, 35, 64, 97, 158, 190, 638, 899, 900, 901, 996, 4200, 7131, 8111};

  auto const post_split_start = start + presplit_pos;
  auto const post_split_size  = size - presplit_pos;
  auto el_iter                = thrust::make_counting_iterator(post_split_start);
  std::vector<int> post_split_elements{el_iter, el_iter + post_split_size};
  std::vector<bool> post_split_valids{
    pre_split_valids.begin() + post_split_start,
    pre_split_valids.begin() + post_split_start + post_split_size};

  std::vector<cudf::test::fixed_width_column_wrapper<int>> expected =
    create_expected_columns_for_splits<int>(splits, post_split_elements, post_split_valids);

  cudf::table_view t({split_cols[1]});
  auto result = cudf::contiguous_split(t, splits);

  EXPECT_EQ(expected.size(), result.size());

  for (unsigned long index = 0; index < result.size(); index++) {
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected[index], result[index].table.column(0));
  }
}

struct ContiguousSplitNestedTypesTest : public cudf::test::BaseFixture {
};

TEST_F(ContiguousSplitNestedTypesTest, Lists)
{
  split_lists<int>(
    [](cudf::column_view const& c, std::vector<cudf::size_type> const& splits) {
      cudf::table_view t({c});
      return cudf::contiguous_split(t, splits);
    },
    [](cudf::column_view const& expected, cudf::contiguous_split_result const& result) {
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result.table.column(0));
    });
}

TEST_F(ContiguousSplitNestedTypesTest, ListsWithNulls)
{
  split_lists_with_nulls<int>(
    [](cudf::column_view const& c, std::vector<cudf::size_type> const& splits) {
      cudf::table_view t({c});
      return cudf::contiguous_split(t, splits);
    },
    [](cudf::column_view const& expected, cudf::contiguous_split_result const& result) {
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result.table.column(0));
    });
}

TEST_F(ContiguousSplitNestedTypesTest, Structs)
{
  split_structs(
    false,
    [](cudf::column_view const& c, std::vector<cudf::size_type> const& splits) {
      cudf::table_view t({c});
      return cudf::contiguous_split(t, splits);
    },
    [](cudf::column_view const& expected, cudf::contiguous_split_result const& result) {
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result.table.column(0));
    });
}

TEST_F(ContiguousSplitNestedTypesTest, StructsWithNulls)
{
  split_structs(
    true,
    [](cudf::column_view const& c, std::vector<cudf::size_type> const& splits) {
      cudf::table_view t({c});
      return cudf::contiguous_split(t, splits);
    },
    [](cudf::column_view const& expected, cudf::contiguous_split_result const& result) {
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result.table.column(0));
    });
}

TEST_F(ContiguousSplitNestedTypesTest, StructsOfList)
{
  split_nested_struct_of_list(
    [](cudf::column_view const& c, std::vector<cudf::size_type> const& splits) {
      cudf::table_view t({c});
      return cudf::contiguous_split(t, splits);
    },
    [](cudf::column_view const& expected, cudf::contiguous_split_result const& result) {
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result.table.column(0));
    });
}

TEST_F(ContiguousSplitNestedTypesTest, ListOfStruct)
{
  // List<Struct<List<>>
  using namespace cudf::test;
  using LCW = cudf::test::lists_column_wrapper<cudf::string_view>;

  // 1. String "names" column.
  std::vector<std::string> names{"Vimes",
                                 "Carrot",
                                 "Angua",
                                 "Cheery",
                                 "Detritus",
                                 "Slant",
                                 "Fred",
                                 "Todd",
                                 "Kevin",
                                 "Jason",
                                 "Clark",
                                 "Bob",
                                 "Mithun",
                                 "Sameer",
                                 "Tim",
                                 "Mark",
                                 "Herman",
                                 "Will"};
  std::vector<bool> names_validity{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  strings_column_wrapper names_column(names.begin(), names.end());

  // 2. Numeric "ages" column.
  std::vector<int> ages{5, 10, 15, 20, 25, 30, 100, 101, 102, 26, 64, 12, 17, 16, 120, 44, 23, 50};
  std::vector<bool> ages_validity = {1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0};
  auto ages_column =
    fixed_width_column_wrapper<int>(ages.begin(), ages.end(), ages_validity.begin());

  // 3. List column
  std::vector<bool> list_validity{1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1};
  lists_column_wrapper<cudf::string_view> list(
    {{"ab", "cd", "ef"},
     LCW{"gh"},
     {"ijk", "lmn"},
     LCW{},
     LCW{"o"},
     {"pqr", "stu", "vwx"},
     {"yz", "aaaa"},
     LCW{"bbbb"},
     {"cccc", "ddd", "eee", "fff", "ggg", "hh"},
     {"b", "cdr", "efh", "um"},
     LCW{"gh", "iu"},
     {"lmn"},
     LCW{"org"},
     LCW{},
     {"stu", "vwx"},
     {"yz", "aaaa", "kem"},
     LCW{"bbbb"},
     {"cccc", "eee", "faff", "jiea", "fff", "ggg", "hh"}},
    list_validity.begin());

  // Assembly struct column
  auto const struct_validity =
    std::vector<bool>{1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1};
  auto struct_column =
    structs_column_wrapper({names_column, ages_column, list}, struct_validity.begin());

  // wrap in a list
  std::vector<int> outer_offsets{0, 3, 4, 8, 13, 16, 17, 18};
  cudf::test::fixed_width_column_wrapper<int> outer_offsets_col(outer_offsets.begin(),
                                                                outer_offsets.end());
  std::vector<bool> outer_validity{1, 1, 1, 0, 1, 1, 0};
  auto outer_null_mask =
    cudf::test::detail::make_null_mask(outer_validity.begin(), outer_validity.end());
  auto outer_list = make_lists_column(static_cast<cudf::size_type>(outer_validity.size()),
                                      outer_offsets_col.release(),
                                      struct_column.release(),
                                      cudf::UNKNOWN_NULL_COUNT,
                                      std::move(outer_null_mask));

  // split
  std::vector<cudf::size_type> splits{1, 3, 7};
  cudf::table_view tbl({static_cast<cudf::column_view>(*outer_list)});

  // we are testing the results of contiguous_split against regular cudf::split, which may seem
  // weird. however, cudf::split() is a simple operation that just sets offsets at the topmost
  // output column, whereas contiguous_split is a deep copy of the data to contiguous output
  // buffers. so as long as we believe the comparison code (expect_columns_equivalent) can compare
  // these outputs correctly, this should be safe.
  auto result   = cudf::contiguous_split(tbl, splits);
  auto expected = cudf::split(static_cast<cudf::column_view>(*outer_list), splits);
  CUDF_EXPECTS(result.size() == expected.size(), "Split result size mismatch");

  for (int index = 0; index < result.size(); index++) {
    cudf::test::expect_columns_equivalent(expected[index], result[index].table.column(0));
  }
}