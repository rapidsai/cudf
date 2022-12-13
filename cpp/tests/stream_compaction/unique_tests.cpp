/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <cudf/copying.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <algorithm>
#include <cmath>

using cudf::nan_policy;
using cudf::null_equality;
using cudf::null_policy;
using cudf::test::iterators::null_at;
using cudf::test::iterators::nulls_at;

auto constexpr KEEP_ANY     = cudf::duplicate_keep_option::KEEP_ANY;
auto constexpr KEEP_FIRST   = cudf::duplicate_keep_option::KEEP_FIRST;
auto constexpr KEEP_LAST    = cudf::duplicate_keep_option::KEEP_LAST;
auto constexpr KEEP_NONE    = cudf::duplicate_keep_option::KEEP_NONE;
auto constexpr NULL_EQUAL   = cudf::null_equality::EQUAL;
auto constexpr NULL_UNEQUAL = cudf::null_equality::UNEQUAL;

using int32s_col  = cudf::test::fixed_width_column_wrapper<int32_t>;
using floats_col  = cudf::test::fixed_width_column_wrapper<float>;
using lists_col   = cudf::test::lists_column_wrapper<int32_t>;
using strings_col = cudf::test::strings_column_wrapper;
using structs_col = cudf::test::structs_column_wrapper;

struct Unique : public cudf::test::BaseFixture {
};

TEST_F(Unique, StringKeyColumn)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col{{5, 4, 4, 5, 5, 8, 1}, {1, 0, 0, 1, 1, 1, 1}};
  cudf::test::strings_column_wrapper key_col{{"all", "new", "new", "all", "new", "the", "strings"},
                                             {1, 1, 1, 1, 0, 1, 1}};
  cudf::table_view input{{col, key_col}};
  std::vector<cudf::size_type> keys{1};

  cudf::test::fixed_width_column_wrapper<int32_t> exp_col{{5, 4, 5, 5, 8, 1}, {1, 0, 1, 1, 1, 1}};
  cudf::test::strings_column_wrapper exp_key_col{{"all", "new", "all", "new", "the", "strings"},
                                                 {1, 1, 1, 0, 1, 1}};
  cudf::table_view expected{{exp_col, exp_key_col}};

  auto got = unique(input, keys, cudf::duplicate_keep_option::KEEP_LAST);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, got->view());
}

TEST_F(Unique, EmptyInputTable)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col(std::initializer_list<int32_t>{});
  cudf::table_view input{{col}};
  std::vector<cudf::size_type> keys{1, 2};

  auto got = unique(input, keys, cudf::duplicate_keep_option::KEEP_FIRST, null_equality::EQUAL);
  CUDF_TEST_EXPECT_TABLES_EQUAL(input, got->view());
}

TEST_F(Unique, NoColumnInputTable)
{
  cudf::table_view input{std::vector<cudf::column_view>()};
  std::vector<cudf::size_type> keys{1, 2};

  auto got = unique(input, keys, cudf::duplicate_keep_option::KEEP_FIRST, null_equality::EQUAL);
  CUDF_TEST_EXPECT_TABLES_EQUAL(input, got->view());
}

TEST_F(Unique, EmptyKeys)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col{{5, 4, 3, 5, 8, 1}, {1, 0, 1, 1, 1, 1}};
  cudf::test::fixed_width_column_wrapper<int32_t> empty_col{};
  cudf::table_view input{{col}};
  std::vector<cudf::size_type> keys{};

  auto got = unique(input, keys, cudf::duplicate_keep_option::KEEP_FIRST, null_equality::EQUAL);
  CUDF_TEST_EXPECT_TABLES_EQUAL(cudf::table_view{{empty_col}}, got->view());
}

TEST_F(Unique, NonNullTable)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col1{{5, 4, 3, 5, 8, 5}};
  cudf::test::fixed_width_column_wrapper<float> col2{{4, 5, 3, 4, 9, 4}};
  cudf::test::fixed_width_column_wrapper<int32_t> col1_key{{20, 20, 20, 19, 21, 9}};
  cudf::test::fixed_width_column_wrapper<int32_t> col2_key{{19, 19, 20, 20, 9, 21}};

  cudf::table_view input{{col1, col2, col1_key, col2_key}};
  std::vector<cudf::size_type> keys{2, 3};

  // Keep the first duplicate row
  // The expected table would be sorted in ascending order with respect to keys
  cudf::test::fixed_width_column_wrapper<int32_t> exp_col1_first{{5, 3, 5, 8, 5}};
  cudf::test::fixed_width_column_wrapper<float> exp_col2_first{{4, 3, 4, 9, 4}};
  cudf::test::fixed_width_column_wrapper<int32_t> exp_col1_key_first{{20, 20, 19, 21, 9}};
  cudf::test::fixed_width_column_wrapper<int32_t> exp_col2_key_first{{19, 20, 20, 9, 21}};
  cudf::table_view expected_first{
    {exp_col1_first, exp_col2_first, exp_col1_key_first, exp_col2_key_first}};

  auto got_first = unique(input, keys, cudf::duplicate_keep_option::KEEP_FIRST);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_first, got_first->view());

  // Keep the last duplicate row
  cudf::test::fixed_width_column_wrapper<int32_t> exp_col1_last{{4, 3, 5, 8, 5}};
  cudf::test::fixed_width_column_wrapper<float> exp_col2_last{{5, 3, 4, 9, 4}};
  cudf::test::fixed_width_column_wrapper<int32_t> exp_col1_key_last{{20, 20, 19, 21, 9}};
  cudf::test::fixed_width_column_wrapper<int32_t> exp_col2_key_last{{19, 20, 20, 9, 21}};
  cudf::table_view expected_last{
    {exp_col1_last, exp_col2_last, exp_col1_key_last, exp_col2_key_last}};

  auto got_last = unique(input, keys, cudf::duplicate_keep_option::KEEP_LAST);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_last, got_last->view());

  // Keep no duplicate rows
  cudf::test::fixed_width_column_wrapper<int32_t> exp_col1_unique{{3, 5, 8, 5}};
  cudf::test::fixed_width_column_wrapper<float> exp_col2_unique{{3, 4, 9, 4}};
  cudf::test::fixed_width_column_wrapper<int32_t> exp_col1_key_unique{{20, 19, 21, 9}};
  cudf::test::fixed_width_column_wrapper<int32_t> exp_col2_key_unique{{20, 20, 9, 21}};
  cudf::table_view expected_unique{
    {exp_col1_unique, exp_col2_unique, exp_col1_key_unique, exp_col2_key_unique}};

  auto got_unique = unique(input, keys, cudf::duplicate_keep_option::KEEP_NONE);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_unique, got_unique->view());
}

TEST_F(Unique, KeepFirstWithNull)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col{{5, 4, 3, 2, 5, 8, 1}, {1, 0, 1, 1, 1, 1, 1}};
  cudf::test::fixed_width_column_wrapper<int32_t> key{{20, 20, 20, 20, 19, 21, 19},
                                                      {1, 1, 0, 0, 1, 1, 1}};
  cudf::table_view input{{col, key}};
  std::vector<cudf::size_type> keys{1};

  // nulls are equal
  cudf::test::fixed_width_column_wrapper<int32_t> exp_col_first_equal{{5, 3, 5, 8, 1},
                                                                      {1, 1, 1, 1, 1}};
  cudf::test::fixed_width_column_wrapper<int32_t> exp_key_col_first_equal{{20, 20, 19, 21, 19},
                                                                          {1, 0, 1, 1, 1}};
  cudf::table_view expected_first_equal{{exp_col_first_equal, exp_key_col_first_equal}};
  auto got_first_equal =
    unique(input, keys, cudf::duplicate_keep_option::KEEP_FIRST, null_equality::EQUAL);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_first_equal, got_first_equal->view());

  // nulls are unequal
  cudf::test::fixed_width_column_wrapper<int32_t> exp_col_first_unequal{{5, 3, 2, 5, 8, 1},
                                                                        {1, 1, 1, 1, 1, 1}};
  cudf::test::fixed_width_column_wrapper<int32_t> exp_key_col_first_unequal{
    {20, 20, 20, 19, 21, 19}, {1, 0, 0, 1, 1, 1}};
  cudf::table_view expected_first_unequal{{exp_col_first_unequal, exp_key_col_first_unequal}};
  auto got_first_unequal =
    unique(input, keys, cudf::duplicate_keep_option::KEEP_FIRST, null_equality::UNEQUAL);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_first_unequal, got_first_unequal->view());
}

TEST_F(Unique, KeepLastWithNull)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col{{5, 4, 3, 2, 5, 8, 1}, {1, 0, 1, 1, 1, 1, 1}};
  cudf::test::fixed_width_column_wrapper<int32_t> key{{20, 20, 20, 20, 19, 21, 19},
                                                      {1, 1, 0, 0, 1, 1, 1}};
  cudf::table_view input{{col, key}};
  std::vector<cudf::size_type> keys{1};

  // nulls are equal
  cudf::test::fixed_width_column_wrapper<int32_t> exp_col_last_equal{{4, 2, 5, 8, 1},
                                                                     {0, 1, 1, 1, 1}};
  cudf::test::fixed_width_column_wrapper<int32_t> exp_key_col_last_equal{{20, 20, 19, 21, 19},
                                                                         {1, 0, 1, 1, 1}};
  cudf::table_view expected_last_equal{{exp_col_last_equal, exp_key_col_last_equal}};
  auto got_last_equal =
    unique(input, keys, cudf::duplicate_keep_option::KEEP_LAST, null_equality::EQUAL);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_last_equal, got_last_equal->view());

  // nulls are unequal
  cudf::test::fixed_width_column_wrapper<int32_t> exp_col_last_unequal{{4, 3, 2, 5, 8, 1},
                                                                       {0, 1, 1, 1, 1, 1}};
  cudf::test::fixed_width_column_wrapper<int32_t> exp_key_col_last_unequal{{20, 20, 20, 19, 21, 19},
                                                                           {1, 0, 0, 1, 1, 1}};
  cudf::table_view expected_last_unequal{{exp_col_last_unequal, exp_key_col_last_unequal}};
  auto got_last_unequal =
    unique(input, keys, cudf::duplicate_keep_option::KEEP_LAST, null_equality::UNEQUAL);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_last_unequal, got_last_unequal->view());
}

TEST_F(Unique, KeepNoneWithNull)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col{{5, 4, 3, 2, 5, 8, 1}, {1, 0, 1, 1, 1, 1, 1}};
  cudf::test::fixed_width_column_wrapper<int32_t> key{{20, 20, 20, 20, 19, 21, 19},
                                                      {1, 1, 0, 0, 1, 1, 1}};
  cudf::table_view input{{col, key}};
  std::vector<cudf::size_type> keys{1};

  // nulls are equal
  cudf::test::fixed_width_column_wrapper<int32_t> exp_col_unique_equal{{5, 8, 1}, {1, 1, 1}};
  cudf::test::fixed_width_column_wrapper<int32_t> exp_key_col_unique_equal{{19, 21, 19}, {1, 1, 1}};
  cudf::table_view expected_unique_equal{{exp_col_unique_equal, exp_key_col_unique_equal}};
  auto got_unique_equal =
    unique(input, keys, cudf::duplicate_keep_option::KEEP_NONE, null_equality::EQUAL);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_unique_equal, got_unique_equal->view());

  // nulls are unequal
  cudf::test::fixed_width_column_wrapper<int32_t> exp_col_unique_unequal{{3, 2, 5, 8, 1},
                                                                         {1, 1, 1, 1, 1}};
  cudf::test::fixed_width_column_wrapper<int32_t> exp_key_col_unique_unequal{{20, 20, 19, 21, 19},
                                                                             {0, 0, 1, 1, 1}};
  cudf::table_view expected_unique_unequal{{exp_col_unique_unequal, exp_key_col_unique_unequal}};
  auto got_unique_unequal =
    unique(input, keys, cudf::duplicate_keep_option::KEEP_NONE, null_equality::UNEQUAL);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_unique_unequal, got_unique_unequal->view());
}

TEST_F(Unique, ListsKeepAny)
{
  // Column(s) used to test KEEP_ANY needs to have same rows for same keys because KEEP_ANY is
  // nondeterministic.
  // clang-format off
  auto const idx = int32s_col{0,  0,      2,   1,   1,      3,   5,   5,      6,      4,      4,      4};
  auto const keys = lists_col{{}, {}, {1, 1}, {1}, {1}, {1, 2}, {2}, {2}, {2, 1}, {2, 2}, {2, 2}, {2, 2}};
  // clang-format on
  auto const input   = cudf::table_view{{idx, keys}};
  auto const key_idx = std::vector<cudf::size_type>{1};

  auto const exp_idx  = int32s_col{0, 2, 1, 3, 5, 6, 4};
  auto const exp_keys = lists_col{{}, {1, 1}, {1}, {1, 2}, {2}, {2, 1}, {2, 2}};
  auto const expected = cudf::table_view{{exp_idx, exp_keys}};

  auto const result = cudf::unique(input, key_idx, KEEP_ANY);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, *result);
}

TEST_F(Unique, ListsKeepFirstLastNone)
{
  // clang-format off
  auto const idx = int32s_col{0,  1,      2,   1,   2,      3,   5,   6,      6,      4,      5,      6};
  auto const keys = lists_col{{}, {}, {1, 1}, {1}, {1}, {1, 2}, {2}, {2}, {2, 1}, {2, 2}, {2, 2}, {2, 2}};
  // clang-format on
  auto const input   = cudf::table_view{{idx, keys}};
  auto const key_idx = std::vector<cudf::size_type>{1};

  // KEEP FIRST
  {
    auto const exp_idx  = int32s_col{0, 2, 1, 3, 5, 6, 4};
    auto const exp_keys = lists_col{{}, {1, 1}, {1}, {1, 2}, {2}, {2, 1}, {2, 2}};
    auto const expected = cudf::table_view{{exp_idx, exp_keys}};

    auto const result = cudf::unique(input, key_idx, KEEP_FIRST);
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, *result);
  }

  // KEEP LAST
  {
    auto const exp_idx  = int32s_col{1, 2, 2, 3, 6, 6, 6};
    auto const exp_keys = lists_col{{}, {1, 1}, {1}, {1, 2}, {2}, {2, 1}, {2, 2}};
    auto const expected = cudf::table_view{{exp_idx, exp_keys}};

    auto const result = cudf::unique(input, key_idx, KEEP_LAST);
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, *result);
  }

  // KEEP NONE
  {
    auto const exp_idx  = int32s_col{2, 3, 6};
    auto const exp_keys = lists_col{{1, 1}, {1, 2}, {2, 1}};
    auto const expected = cudf::table_view{{exp_idx, exp_keys}};

    auto const result = cudf::unique(input, key_idx, KEEP_NONE);
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, *result);
  }
}

TEST_F(Unique, NullableListsKeepAny)
{
  // Column(s) used to test KEEP_ANY needs to have same rows for same keys because KEEP_ANY is
  // nondeterministic.
  // clang-format off
  auto const idx = int32s_col{0,   0,      2,    1,   1,      3,               3,           5,   5,      6,      4,      4};
  auto const keys = lists_col{{{}, {}, {1, 1}, {1}, {1},     {} /*NULL*/,     {} /*NULL*/, {2}, {2}, {2, 1}, {2, 2}, {2, 2}}, 
                              nulls_at({5, 6})};
  // clang-format on
  auto const input   = cudf::table_view{{idx, keys}};
  auto const key_idx = std::vector<cudf::size_type>{1};

  // Nulls are equal.
  {
    auto const exp_idx = int32s_col{0, 2, 1, 3, 5, 6, 4};
    auto const exp_keys =
      lists_col{{{}, {1, 1}, {1}, {} /*NULL*/, {2}, {2, 1}, {2, 2}}, null_at(3)};
    auto const expected = cudf::table_view{{exp_idx, exp_keys}};

    auto const result = cudf::unique(input, key_idx, KEEP_ANY);

    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, *result);
  }

  // Nulls are unequal.
  {
    auto const exp_idx = int32s_col{0, 2, 1, 3, 3, 5, 6, 4};
    auto const exp_keys =
      lists_col{{{}, {1, 1}, {1}, {} /*NULL*/, {} /*NULL*/, {2}, {2, 1}, {2, 2}}, nulls_at({3, 4})};
    auto const expected = cudf::table_view{{exp_idx, exp_keys}};

    auto const result = cudf::unique(input, key_idx, KEEP_ANY, NULL_UNEQUAL);

    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, *result);
  }
}

TEST_F(Unique, NullableListsKeepFirstLastNone)
{
  // Column(s) used to test KEEP_ANY needs to have same rows for same keys because KEEP_ANY is
  // nondeterministic.
  // clang-format off
  auto const idx = int32s_col{0,   1,      2,    1,   2,      3,               4,           5,   6,      6,      4,      5};
  auto const keys = lists_col{{{}, {}, {1, 1}, {1}, {1},     {} /*NULL*/,     {} /*NULL*/, {2}, {2}, {2, 1}, {2, 2}, {2, 2}}, 
                              nulls_at({5, 6})};
  // clang-format on
  auto const input   = cudf::table_view{{idx, keys}};
  auto const key_idx = std::vector<cudf::size_type>{1};

  // KEEP FIRST
  {// Nulls are equal.
   {auto const exp_idx = int32s_col{0, 2, 1, 3, 5, 6, 4};
  auto const exp_keys = lists_col{{{}, {1, 1}, {1}, {} /*NULL*/, {2}, {2, 1}, {2, 2}}, null_at(3)};
  auto const expected = cudf::table_view{{exp_idx, exp_keys}};

  auto const result = cudf::unique(input, key_idx, KEEP_FIRST);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, *result);
}

// Nulls are unequal.
{
  auto const exp_idx = int32s_col{0, 2, 1, 3, 4, 5, 6, 4};
  auto const exp_keys =
    lists_col{{{}, {1, 1}, {1}, {} /*NULL*/, {} /*NULL*/, {2}, {2, 1}, {2, 2}}, nulls_at({3, 4})};
  auto const expected = cudf::table_view{{exp_idx, exp_keys}};

  auto const result = cudf::unique(input, key_idx, KEEP_FIRST, NULL_UNEQUAL);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, *result);
}
}

// KEEP LAST
{// Nulls are equal.
 {auto const exp_idx = int32s_col{1, 2, 2, 4, 6, 6, 5};
auto const exp_keys = lists_col{{{}, {1, 1}, {1}, {} /*NULL*/, {2}, {2, 1}, {2, 2}}, null_at(3)};
auto const expected = cudf::table_view{{exp_idx, exp_keys}};

auto const result = cudf::unique(input, key_idx, KEEP_LAST);

CUDF_TEST_EXPECT_TABLES_EQUAL(expected, *result);
}

// Nulls are unequal.
{
  auto const exp_idx = int32s_col{1, 2, 2, 3, 4, 6, 6, 5};
  auto const exp_keys =
    lists_col{{{}, {1, 1}, {1}, {} /*NULL*/, {} /*NULL*/, {2}, {2, 1}, {2, 2}}, nulls_at({3, 4})};
  auto const expected = cudf::table_view{{exp_idx, exp_keys}};

  auto const result = cudf::unique(input, key_idx, KEEP_LAST, NULL_UNEQUAL);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, *result);
}
}

// KEEP NONE
{
  // Nulls are equal.
  {
    auto const exp_idx  = int32s_col{2, 6};
    auto const exp_keys = lists_col{{{1, 1}, {2, 1}}, nulls_at({})};
    auto const expected = cudf::table_view{{exp_idx, exp_keys}};

    auto const result = cudf::unique(input, key_idx, KEEP_NONE);

    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, *result);
  }

  // Nulls are unequal.
  {
    auto const exp_idx  = int32s_col{2, 3, 4, 6};
    auto const exp_keys = lists_col{{{1, 1}, {} /*NULL*/, {} /*NULL*/, {2, 1}}, nulls_at({1, 2})};
    auto const expected = cudf::table_view{{exp_idx, exp_keys}};

    auto const result = cudf::unique(input, key_idx, KEEP_NONE, NULL_UNEQUAL);

    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, *result);
  }
}
}
