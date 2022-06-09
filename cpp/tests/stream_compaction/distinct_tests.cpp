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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/copying.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <cmath>

auto constexpr null{0};  // null at current level
auto constexpr XXX{0};   // null pushed down from parent level
auto constexpr KEEP_ANY   = cudf::duplicate_keep_option::KEEP_ANY;
auto constexpr KEEP_FIRST = cudf::duplicate_keep_option::KEEP_FIRST;
auto constexpr KEEP_LAST  = cudf::duplicate_keep_option::KEEP_LAST;
auto constexpr KEEP_NONE  = cudf::duplicate_keep_option::KEEP_NONE;

using int32s_col  = cudf::test::fixed_width_column_wrapper<int32_t>;
using floats_col  = cudf::test::fixed_width_column_wrapper<float>;
using lists_col   = cudf::test::lists_column_wrapper<int32_t>;
using strings_col = cudf::test::strings_column_wrapper;
using structs_col = cudf::test::structs_column_wrapper;
using cudf::nan_policy;
using cudf::null_equality;
using cudf::null_policy;
using cudf::test::iterators::no_nulls;
using cudf::test::iterators::null_at;
using cudf::test::iterators::nulls_at;

struct Distinct : public cudf::test::BaseFixture {
};

TEST_F(Distinct, StringKeyColumn_KEEP_ANY)
{
  // Column(s) used to test KEEP_ANY needs to have same rows for same keys.
  auto const col = int32s_col{{5, null, null, 5, 5, 8, 1}, nulls_at({1, 2})};
  auto const keys =
    strings_col{{"all", "new", "new", "all", "" /*NULL*/, "the", "strings"}, null_at(4)};
  auto const input   = cudf::table_view{{col, keys}};
  auto const key_idx = std::vector<cudf::size_type>{1};

  auto const exp_col_sort  = int32s_col{{5, 5, null, 1, 8}, null_at(2)};
  auto const exp_keys_sort = strings_col{{"" /*NULL*/, "all", "new", "strings", "the"}, null_at(0)};
  auto const expected_sort = cudf::table_view{{exp_col_sort, exp_keys_sort}};

  {
    auto const result      = cudf::distinct(input, key_idx);
    auto const result_sort = cudf::sort_by_key(*result, result->select(key_idx));
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected_sort, *result_sort);
  }

  {
    auto const result      = cudf::distinct(input, key_idx, KEEP_ANY);
    auto const result_sort = cudf::sort_by_key(*result, result->select(key_idx));
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected_sort, *result_sort);
  }
}

TEST_F(Distinct, StringKeyColumn_KEEP_EXCEPT_ANY)
{
  // Column(s) used to test needs to have different rows for the same keys.
  auto const col = int32s_col{{0, null, 2, 3, 4, 5, 6}, null_at(1)};
  auto const keys =
    strings_col{{"all", "new", "new", "all", "" /*NULL*/, "the", "strings"}, null_at(4)};
  auto const input   = cudf::table_view{{col, keys}};
  auto const key_idx = std::vector<cudf::size_type>{1};

  {
    auto const exp_col_sort = int32s_col{{4, 0, null, 6, 5}, null_at(2)};
    auto const exp_keys_sort =
      strings_col{{"" /*NULL*/, "all", "new", "strings", "the"}, null_at(0)};
    auto const expected_sort = cudf::table_view{{exp_col_sort, exp_keys_sort}};

    auto const result      = cudf::distinct(input, key_idx, KEEP_FIRST);
    auto const result_sort = cudf::sort_by_key(*result, result->select(key_idx));
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected_sort, *result_sort);
  }

  {
    auto const exp_col_sort = int32s_col{{4, 3, 2, 6, 5}, no_nulls()};
    auto const exp_keys_sort =
      strings_col{{"" /*NULL*/, "all", "new", "strings", "the"}, null_at(0)};
    auto const expected_sort = cudf::table_view{{exp_col_sort, exp_keys_sort}};

    auto const result      = cudf::distinct(input, key_idx, KEEP_LAST);
    auto const result_sort = cudf::sort_by_key(*result, result->select(key_idx));
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected_sort, *result_sort);
  }

  {
    auto const exp_col_sort  = int32s_col{{4, 6, 5}, no_nulls()};
    auto const exp_keys_sort = strings_col{{"" /*NULL*/, "strings", "the"}, null_at(0)};
    auto const expected_sort = cudf::table_view{{exp_col_sort, exp_keys_sort}};

    auto const result      = cudf::distinct(input, key_idx, KEEP_NONE);
    auto const result_sort = cudf::sort_by_key(*result, result->select(key_idx));
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected_sort, *result_sort);
  }
}

TEST_F(Distinct, EmptyInputTable)
{
  int32s_col col(std::initializer_list<int32_t>{});
  cudf::table_view input{{col}};
  std::vector<cudf::size_type> key_idx{1, 2};

  auto got = cudf::distinct(input, key_idx, null_equality::EQUAL);
  CUDF_TEST_EXPECT_TABLES_EQUAL(input, got->view());
}

TEST_F(Distinct, NoColumnInputTable)
{
  cudf::table_view input{std::vector<cudf::column_view>()};
  std::vector<cudf::size_type> key_idx{1, 2};

  auto got = cudf::distinct(input, key_idx, null_equality::EQUAL);
  CUDF_TEST_EXPECT_TABLES_EQUAL(input, got->view());
}

TEST_F(Distinct, EmptyKeys)
{
  int32s_col col{{5, 4, 3, 5, 8, 1}, {1, 0, 1, 1, 1, 1}};
  int32s_col empty_col{};
  cudf::table_view input{{col}};
  std::vector<cudf::size_type> key_idx{};

  auto got = cudf::distinct(input, key_idx, null_equality::EQUAL);
  CUDF_TEST_EXPECT_TABLES_EQUAL(cudf::table_view{{empty_col}}, got->view());
}

TEST_F(Distinct, NonNullTable_KEEP_ANY)
{
  // Column(s) used to test KEEP_ANY needs to have same rows for same keys.
  auto const col1  = int32s_col{6, 6, 6, 3, 5, 8, 5};
  auto const col2  = floats_col{6, 6, 6, 3, 4, 9, 4};
  auto const keys1 = int32s_col{20, 20, 20, 20, 19, 21, 9};
  auto const keys2 = int32s_col{19, 19, 19, 20, 20, 9, 21};

  auto const input   = cudf::table_view{{col1, col2, keys1, keys2}};
  auto const key_idx = std::vector<cudf::size_type>{2, 3};

  auto const exp_col1_sort  = int32s_col{5, 5, 6, 3, 8};
  auto const exp_col2_sort  = floats_col{4, 4, 6, 3, 9};
  auto const exp_keys1_sort = int32s_col{9, 19, 20, 20, 21};
  auto const exp_keys2_sort = int32s_col{21, 20, 19, 20, 9};
  auto const expected_sort =
    cudf::table_view{{exp_col1_sort, exp_col2_sort, exp_keys1_sort, exp_keys2_sort}};

  {
    auto const result      = cudf::distinct(input, key_idx);
    auto const result_sort = cudf::sort_by_key(*result, result->select(key_idx));
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected_sort, *result_sort);
  }

  {
    auto const result      = cudf::distinct(input, key_idx, KEEP_ANY);
    auto const result_sort = cudf::sort_by_key(*result, result->select(key_idx));
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected_sort, *result_sort);
  }
}

TEST_F(Distinct, NonNullTable_KEEP_EXCEPT_ANY)
{
  // Column(s) used to test needs to have different rows for the same keys.
  auto const col1  = int32s_col{0, 1, 2, 3, 4, 5, 6};
  auto const col2  = floats_col{10, 11, 12, 13, 14, 15, 16};
  auto const keys1 = int32s_col{20, 20, 20, 20, 19, 21, 9};
  auto const keys2 = int32s_col{19, 19, 19, 20, 20, 9, 21};

  auto const input   = cudf::table_view{{col1, col2, keys1, keys2}};
  auto const key_idx = std::vector<cudf::size_type>{2, 3};

  {
    auto const exp_col1_sort  = int32s_col{6, 4, 0, 3, 5};
    auto const exp_col2_sort  = floats_col{16, 14, 10, 13, 15};
    auto const exp_keys1_sort = int32s_col{9, 19, 20, 20, 21};
    auto const exp_keys2_sort = int32s_col{21, 20, 19, 20, 9};
    auto const expected_sort =
      cudf::table_view{{exp_col1_sort, exp_col2_sort, exp_keys1_sort, exp_keys2_sort}};

    auto const result      = cudf::distinct(input, key_idx, KEEP_FIRST);
    auto const result_sort = cudf::sort_by_key(*result, result->select(key_idx));
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected_sort, *result_sort);
  }

  {
    auto const exp_col1_sort  = int32s_col{6, 4, 2, 3, 5};
    auto const exp_col2_sort  = floats_col{16, 14, 12, 13, 15};
    auto const exp_keys1_sort = int32s_col{9, 19, 20, 20, 21};
    auto const exp_keys2_sort = int32s_col{21, 20, 19, 20, 9};
    auto const expected_sort =
      cudf::table_view{{exp_col1_sort, exp_col2_sort, exp_keys1_sort, exp_keys2_sort}};

    auto const result      = cudf::distinct(input, key_idx, KEEP_LAST);
    auto const result_sort = cudf::sort_by_key(*result, result->select(key_idx));
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected_sort, *result_sort);
  }

  {
    auto const exp_col1_sort  = int32s_col{6, 4, 3, 5};
    auto const exp_col2_sort  = floats_col{16, 14, 13, 15};
    auto const exp_keys1_sort = int32s_col{9, 19, 20, 21};
    auto const exp_keys2_sort = int32s_col{21, 20, 20, 9};
    auto const expected_sort =
      cudf::table_view{{exp_col1_sort, exp_col2_sort, exp_keys1_sort, exp_keys2_sort}};

    auto const result      = cudf::distinct(input, key_idx, KEEP_NONE);
    auto const result_sort = cudf::sort_by_key(*result, result->select(key_idx));
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected_sort, *result_sort);
  }
}

TEST_F(Distinct, SlicedNonNullTable_KEEP_ANY)
{
  auto constexpr dont_care = int32_t{0};

  // Column(s) used to test KEEP_ANY needs to have same rows for same keys.
  auto const col1  = int32s_col{dont_care, dont_care, 6, 6, 6, 3, 5, 8, 5, dont_care};
  auto const col2  = floats_col{dont_care, dont_care, 6, 6, 6, 3, 4, 9, 4, dont_care};
  auto const keys1 = int32s_col{dont_care, dont_care, 20, 20, 20, 20, 19, 21, 9, dont_care};
  auto const keys2 = int32s_col{dont_care, dont_care, 19, 19, 19, 20, 20, 9, 21, dont_care};

  auto const input_original = cudf::table_view{{col1, col2, keys1, keys2}};
  auto const input          = cudf::slice(input_original, {2, 9})[0];
  auto const key_idx        = std::vector<cudf::size_type>{2, 3};

  auto const exp_col1_sort  = int32s_col{5, 5, 6, 3, 8};
  auto const exp_col2_sort  = floats_col{4, 4, 6, 3, 9};
  auto const exp_keys1_sort = int32s_col{9, 19, 20, 20, 21};
  auto const exp_keys2_sort = int32s_col{21, 20, 19, 20, 9};
  auto const expected_sort =
    cudf::table_view{{exp_col1_sort, exp_col2_sort, exp_keys1_sort, exp_keys2_sort}};

  {
    auto const result      = cudf::distinct(input, key_idx);
    auto const result_sort = cudf::sort_by_key(*result, result->select(key_idx));
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected_sort, *result_sort);
  }

  {
    auto const result      = cudf::distinct(input, key_idx, KEEP_ANY);
    auto const result_sort = cudf::sort_by_key(*result, result->select(key_idx));
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected_sort, *result_sort);
  }
}

TEST_F(Distinct, SlicedNonNullTable_KEEP_EXCEPT_ANY)
{
  auto constexpr dont_care = int32_t{0};

  // Column(s) used to test needs to have different rows for the same keys.
  // clang-format off
  auto const col1  = int32s_col{0, 1, 2, // <- don't care
                                3, 4, 5, 6, 7, 8, 9, dont_care};
  auto const col2  = floats_col{10, 11, 12, // <- don't care
                                13, 14, 15, 16, 17, 18, 19, dont_care};
  auto const keys1 = int32s_col{20, 20, 20, // <- don't care
                                20, 20, 20, 20, 19, 21, 9, dont_care};
  auto const keys2 = int32s_col{19, 19, 19, // <- don't care
                                19, 19, 19, 20, 20, 9, 21, dont_care};
  // clang-format on
  auto const input_original = cudf::table_view{{col1, col2, keys1, keys2}};
  auto const input          = cudf::slice(input_original, {3, 10})[0];
  auto const key_idx        = std::vector<cudf::size_type>{2, 3};

  {
    auto const exp_col1_sort  = int32s_col{9, 7, 3, 6, 8};
    auto const exp_col2_sort  = floats_col{19, 17, 13, 16, 18};
    auto const exp_keys1_sort = int32s_col{9, 19, 20, 20, 21};
    auto const exp_keys2_sort = int32s_col{21, 20, 19, 20, 9};
    auto const expected_sort =
      cudf::table_view{{exp_col1_sort, exp_col2_sort, exp_keys1_sort, exp_keys2_sort}};

    auto const result      = cudf::distinct(input, key_idx, KEEP_FIRST);
    auto const result_sort = cudf::sort_by_key(*result, result->select(key_idx));
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected_sort, *result_sort);
  }

  {
    auto const exp_col1_sort  = int32s_col{9, 7, 5, 6, 8};
    auto const exp_col2_sort  = floats_col{19, 17, 15, 16, 18};
    auto const exp_keys1_sort = int32s_col{9, 19, 20, 20, 21};
    auto const exp_keys2_sort = int32s_col{21, 20, 19, 20, 9};
    auto const expected_sort =
      cudf::table_view{{exp_col1_sort, exp_col2_sort, exp_keys1_sort, exp_keys2_sort}};

    auto const result      = cudf::distinct(input, key_idx, KEEP_LAST);
    auto const result_sort = cudf::sort_by_key(*result, result->select(key_idx));
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected_sort, *result_sort);
  }

  {
    auto const exp_col1_sort  = int32s_col{9, 7, 6, 8};
    auto const exp_col2_sort  = floats_col{19, 17, 16, 18};
    auto const exp_keys1_sort = int32s_col{9, 19, 20, 21};
    auto const exp_keys2_sort = int32s_col{21, 20, 20, 9};
    auto const expected_sort =
      cudf::table_view{{exp_col1_sort, exp_col2_sort, exp_keys1_sort, exp_keys2_sort}};

    auto const result      = cudf::distinct(input, key_idx, KEEP_NONE);
    auto const result_sort = cudf::sort_by_key(*result, result->select(key_idx));
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected_sort, *result_sort);
  }
}

TEST_F(Distinct, InputWithNulls_KEEP_ANY)
{
  // Column(s) used to test KEEP_ANY needs to have same rows for same keys.
  auto const col     = int32s_col{5, 4, 4, 1, 8, 1};
  auto const keys    = int32s_col{{20, null, null, 19, 21, 19}, nulls_at({1, 2})};
  auto const input   = cudf::table_view{{col, keys}};
  auto const key_idx = std::vector<cudf::size_type>{1};

  {
    auto const exp_col_sort  = int32s_col{4, 1, 5, 8};
    auto const exp_keys_sort = int32s_col{{null, 19, 20, 21}, null_at(0)};
    auto const expected_sort = cudf::table_view{{exp_col_sort, exp_keys_sort}};

    {
      auto const result      = cudf::distinct(input, key_idx);
      auto const result_sort = cudf::sort_by_key(*result, result->select(key_idx));
      CUDF_TEST_EXPECT_TABLES_EQUAL(expected_sort, *result_sort);
    }

    {
      auto const result      = cudf::distinct(input, key_idx, KEEP_ANY);
      auto const result_sort = cudf::sort_by_key(*result, result->select(key_idx));
      CUDF_TEST_EXPECT_TABLES_EQUAL(expected_sort, *result_sort);
    }
  }

  {
    auto const exp_col_sort  = int32s_col{4, 4, 1, 5, 8};
    auto const exp_keys_sort = int32s_col{{null, null, 19, 20, 21}, nulls_at({0, 1})};
    auto const expected_sort = cudf::table_view{{exp_col_sort, exp_keys_sort}};

    {
      auto const result      = cudf::distinct(input, key_idx, null_equality::UNEQUAL);
      auto const result_sort = cudf::sort_by_key(*result, result->select(key_idx));
      CUDF_TEST_EXPECT_TABLES_EQUAL(expected_sort, *result_sort);
    }

    {
      auto const result      = cudf::distinct(input, key_idx, KEEP_ANY, null_equality::UNEQUAL);
      auto const result_sort = cudf::sort_by_key(*result, result->select(key_idx));
      CUDF_TEST_EXPECT_TABLES_EQUAL(expected_sort, *result_sort);
    }
  }
}

TEST_F(Distinct, InputWithNulls_KEEP_EXCEPT_ANY)
{
  // Column(s) used to test needs to have different rows for the same keys.
  auto const col     = int32s_col{0, 1, 2, 3, 4, 5, 6};
  auto const keys    = int32s_col{{20, null, null, 19, 21, 19, 22}, nulls_at({1, 2})};
  auto const input   = cudf::table_view{{col, keys}};
  auto const key_idx = std::vector<cudf::size_type>{1};

  {
    auto const exp_col_sort  = int32s_col{1, 3, 0, 4, 6};
    auto const exp_keys_sort = int32s_col{{null, 19, 20, 21, 22}, null_at(0)};
    auto const expected_sort = cudf::table_view{{exp_col_sort, exp_keys_sort}};

    auto const result      = cudf::distinct(input, key_idx, KEEP_FIRST);
    auto const result_sort = cudf::sort_by_key(*result, result->select(key_idx));
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected_sort, *result_sort);
  }

  {
    auto const exp_col_sort  = int32s_col{2, 5, 0, 4, 6};
    auto const exp_keys_sort = int32s_col{{null, 19, 20, 21, 22}, null_at(0)};
    auto const expected_sort = cudf::table_view{{exp_col_sort, exp_keys_sort}};

    auto const result      = cudf::distinct(input, key_idx, KEEP_LAST);
    auto const result_sort = cudf::sort_by_key(*result, result->select(key_idx));
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected_sort, *result_sort);
  }

  {
    auto const exp_col_sort  = int32s_col{0, 4, 6};
    auto const exp_keys_sort = int32s_col{{20, 21, 22}, no_nulls()};
    auto const expected_sort = cudf::table_view{{exp_col_sort, exp_keys_sort}};

    auto const result      = cudf::distinct(input, key_idx, KEEP_NONE);
    auto const result_sort = cudf::sort_by_key(*result, result->select(key_idx));
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected_sort, *result_sort);
  }
}

TEST_F(Distinct, BasicLists_KEEP_ANY)
{
  // Column(s) used to test KEEP_ANY needs to have same rows for same keys.
  // clang-format off
  auto const idx = int32s_col{ 0,  0,   1,      2,   1,      3,      4,  5,   5,      6,      4,       4};
  auto const keys = lists_col{{}, {}, {1}, {1, 1}, {1}, {1, 2}, {2, 2}, {2}, {2}, {2, 1}, {2, 2}, {2, 2}};
  // clang-format on
  auto const input   = cudf::table_view{{idx, keys}};
  auto const key_idx = std::vector<cudf::size_type>{1};

  auto const exp_idx_sort  = int32s_col{0, 1, 2, 3, 4, 5, 6};
  auto const exp_keys_sort = lists_col{{}, {1}, {1, 1}, {1, 2}, {2, 2}, {2}, {2, 1}};
  auto const expected_sort = cudf::table_view{{exp_idx_sort, exp_keys_sort}};

  {
    auto const result      = cudf::distinct(input, key_idx);
    auto const result_sort = cudf::sort_by_key(*result, result->select({0}));
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected_sort, *result_sort);
  }

  {
    auto const result      = cudf::distinct(input, key_idx, KEEP_ANY);
    auto const result_sort = cudf::sort_by_key(*result, result->select({0}));
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected_sort, *result_sort);
  }
}

TEST_F(Distinct, BasicLists_KEEP_EXCEPT_ANY)
{
  // Column(s) used to test needs to have different rows for the same keys.
  // clang-format off
  auto const idx = int32s_col{ 0,  1,  2,      3,   4,      5,      6,   7,   8,       9,     10,     11};
  auto const keys = lists_col{{}, {}, {1}, {1, 1}, {1}, {1, 2}, {2, 2}, {2}, {2}, {2, 1}, {2, 2}, {2, 2}};
  // clang-format on
  auto const input   = cudf::table_view{{idx, keys}};
  auto const key_idx = std::vector<cudf::size_type>{1};

  {
    auto const exp_idx_sort  = int32s_col{0, 2, 3, 5, 6, 7, 9};
    auto const exp_keys_sort = lists_col{{}, {1}, {1, 1}, {1, 2}, {2, 2}, {2}, {2, 1}};
    auto const expected_sort = cudf::table_view{{exp_idx_sort, exp_keys_sort}};

    auto const result      = cudf::distinct(input, key_idx, KEEP_FIRST);
    auto const result_sort = cudf::sort_by_key(*result, result->select({0}));
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected_sort, *result_sort);
  }

  {
    auto const exp_idx_sort  = int32s_col{1, 3, 4, 5, 8, 9, 11};
    auto const exp_keys_sort = lists_col{{}, {1, 1}, {1}, {1, 2}, {2}, {2, 1}, {2, 2}};
    auto const expected_sort = cudf::table_view{{exp_idx_sort, exp_keys_sort}};

    auto const result      = cudf::distinct(input, key_idx, KEEP_LAST);
    auto const result_sort = cudf::sort_by_key(*result, result->select({0}));
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected_sort, *result_sort);
  }

  {
    auto const exp_idx_sort  = int32s_col{3, 5, 9};
    auto const exp_keys_sort = lists_col{{1, 1}, {1, 2}, {2, 1}};
    auto const expected_sort = cudf::table_view{{exp_idx_sort, exp_keys_sort}};

    auto const result      = cudf::distinct(input, key_idx, KEEP_NONE);
    auto const result_sort = cudf::sort_by_key(*result, result->select({0}));
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected_sort, *result_sort);
  }
}

TEST_F(Distinct, SlicedBasicLists_KEEP_ANY)
{
  auto constexpr dont_care = int32_t{0};

  // Column(s) used to test KEEP_ANY needs to have same rows for same keys.
  auto const idx  = int32s_col{dont_care, dont_care, 1, 2, 1, 3, 4, 5, 5, 6, 4, 4, dont_care};
  auto const keys = lists_col{
    {0, 0}, {0, 0}, {1}, {1, 1}, {1}, {1, 2}, {2, 2}, {2}, {2}, {2, 1}, {2, 2}, {2, 2}, {5, 5}};
  auto const input_original = cudf::table_view{{idx, keys}};
  auto const input          = cudf::slice(input_original, {2, 12})[0];
  auto const key_idx        = std::vector<cudf::size_type>{1};

  auto const exp_idx_sort  = int32s_col{1, 2, 3, 4, 5, 6};
  auto const exp_val_sort  = lists_col{{1}, {1, 1}, {1, 2}, {2, 2}, {2}, {2, 1}};
  auto const expected_sort = cudf::table_view{{exp_idx_sort, exp_val_sort}};

  {
    auto const result      = cudf::distinct(input, key_idx);
    auto const result_sort = cudf::sort_by_key(*result, result->select({0}));
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected_sort, *result_sort);
  }

  {
    auto const result      = cudf::distinct(input, key_idx, KEEP_ANY);
    auto const result_sort = cudf::sort_by_key(*result, result->select({0}));
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected_sort, *result_sort);
  }
}

TEST_F(Distinct, NullableLists_KEEP_ANY)
{
  // Column(s) used to test KEEP_ANY needs to have same rows for same keys.
  auto const idx = int32s_col{0, 0, 1, 1, 4, 5, 5, 6, 4, 4, 6};
  auto const keys =
    lists_col{{{}, {}, {1}, {1}, {2, 2}, {2}, {2}, {} /*NULL*/, {2, 2}, {2, 2}, {} /*NULL*/},
              nulls_at({7, 10})};
  auto const input   = cudf::table_view{{idx, keys}};
  auto const key_idx = std::vector<cudf::size_type>{1};

  {
    auto const exp_idx_sort  = int32s_col{0, 1, 4, 5, 6};
    auto const exp_keys_sort = lists_col{{{}, {1}, {2, 2}, {2}, {} /*NULL*/}, null_at(4)};
    auto const expected_sort = cudf::table_view{{exp_idx_sort, exp_keys_sort}};

    {
      auto const result      = cudf::distinct(input, key_idx);
      auto const result_sort = cudf::sort_by_key(*result, result->select({0}));
      CUDF_TEST_EXPECT_TABLES_EQUAL(expected_sort, *result_sort);
    }

    {
      auto const result      = cudf::distinct(input, key_idx, KEEP_ANY);
      auto const result_sort = cudf::sort_by_key(*result, result->select({0}));
      CUDF_TEST_EXPECT_TABLES_EQUAL(expected_sort, *result_sort);
    }
  }

  {
    auto const exp_idx_sort = int32s_col{0, 1, 4, 5, 6, 6};
    auto const exp_keys_sort =
      lists_col{{{}, {1}, {2, 2}, {2}, {} /*NULL*/, {} /*NULL*/}, nulls_at({4, 5})};
    auto const expected_sort = cudf::table_view{{exp_idx_sort, exp_keys_sort}};

    {
      auto const result      = cudf::distinct(input, key_idx, null_equality::UNEQUAL);
      auto const result_sort = cudf::sort_by_key(*result, result->select({0}));
      CUDF_TEST_EXPECT_TABLES_EQUAL(expected_sort, *result_sort);
    }

    {
      auto const result      = cudf::distinct(input, key_idx, KEEP_ANY, null_equality::UNEQUAL);
      auto const result_sort = cudf::sort_by_key(*result, result->select({0}));
      CUDF_TEST_EXPECT_TABLES_EQUAL(expected_sort, *result_sort);
    }
  }
}

TEST_F(Distinct, NullableLists_KEEP_EXCEPT_ANY)
{
  // Column(s) used to test needs to have different rows for the same keys.
  // clang-format off
  auto const idx = int32s_col{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  auto const keys =
    lists_col{{{}, {}, {1}, {1}, {2, 2}, {2}, {2}, {} /*NULL*/, {2, 2}, {2, 2}, {} /*NULL*/},
              nulls_at({7, 10})};
  // clang-format on
  auto const input   = cudf::table_view{{idx, keys}};
  auto const key_idx = std::vector<cudf::size_type>{1};

  {
    auto const exp_idx_sort  = int32s_col{0, 2, 4, 5, 7};
    auto const exp_keys_sort = lists_col{{{}, {1}, {2, 2}, {2}, {} /*NULL*/}, null_at(4)};
    auto const expected_sort = cudf::table_view{{exp_idx_sort, exp_keys_sort}};

    auto const result      = cudf::distinct(input, key_idx, KEEP_FIRST);
    auto const result_sort = cudf::sort_by_key(*result, result->select({0}));
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected_sort, *result_sort);
  }

  {
    auto const exp_idx_sort  = int32s_col{1, 3, 6, 9, 10};
    auto const exp_keys_sort = lists_col{{{}, {1}, {2}, {2, 2}, {} /*NULL*/}, null_at(4)};
    auto const expected_sort = cudf::table_view{{exp_idx_sort, exp_keys_sort}};

    auto const result      = cudf::distinct(input, key_idx, KEEP_LAST);
    auto const result_sort = cudf::sort_by_key(*result, result->select({0}));
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected_sort, *result_sort);
  }

  {
    auto const exp_idx  = int32s_col{};
    auto const exp_keys = lists_col{};
    auto const expected = cudf::table_view{{exp_idx, exp_keys}};

    auto const result = cudf::distinct(input, key_idx, KEEP_NONE);
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, *result);
  }
}

TEST_F(Distinct, ListsOfStructs_KEEP_ANY)
{
  // Constructing a list of structs of two elements
  // 0.   []                  ==
  // 1.   []                  !=
  // 2.   Null                ==
  // 3.   Null                !=
  // 4.   [Null, Null]        !=
  // 5.   [Null]              ==
  // 6.   [Null]              ==
  // 7.   [Null]              !=
  // 8.   [{Null, Null}]      !=
  // 9.   [{1,'a'}, {2,'b'}]  !=
  // 10.  [{0,'a'}, {2,'b'}]  !=
  // 11.  [{0,'a'}, {2,'c'}]  ==
  // 12.  [{0,'a'}, {2,'c'}]  !=
  // 13.  [{0,Null}]          ==
  // 14.  [{0,Null}]          !=
  // 15.  [{Null, 'b'}]       ==
  // 16.  [{Null, 'b'}]

  auto const structs = [] {
    auto child1 =
      int32s_col{{XXX, XXX, XXX, XXX, XXX, null, 1, 2, 0, 2, 0, 2, 0, 2, 0, 0, null, null},
                 nulls_at({5, 16, 17})};
    auto child2 = strings_col{{"" /*XXX*/,
                               "" /*XXX*/,
                               "" /*XXX*/,
                               "" /*XXX*/,
                               "" /*XXX*/,
                               "" /*null*/,
                               "a",
                               "b",
                               "a",
                               "b",
                               "a",
                               "c",
                               "a",
                               "c",
                               "" /*null*/,
                               "" /*null*/,
                               "b",
                               "b"},
                              nulls_at({5, 14, 15})};

    return structs_col{{child1, child2}, nulls_at({0, 1, 2, 3, 4})};
  }();

  auto const offsets      = int32s_col{0, 0, 0, 0, 0, 2, 3, 4, 5, 6, 8, 10, 12, 14, 15, 16, 17, 18};
  auto const null_it      = nulls_at({2, 3});
  auto const nullmask_buf = cudf::test::detail::make_null_mask(null_it, null_it + 17);
  auto const keys         = cudf::column_view(cudf::data_type(cudf::type_id::LIST),
                                      17,
                                      nullptr,
                                      static_cast<cudf::bitmask_type const*>(nullmask_buf.data()),
                                      cudf::UNKNOWN_NULL_COUNT,
                                      0,
                                      {offsets, structs});

  auto const idx     = int32s_col{1, 1, 2, 2, 3, 4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 10, 10};
  auto const input   = cudf::table_view{{idx, keys}};
  auto const key_idx = std::vector<cudf::size_type>{1};

  {
    auto const expect_map   = int32s_col{0, 2, 4, 5, 8, 9, 10, 11, 13, 15};
    auto const expect_table = cudf::gather(input, expect_map);

    {
      auto const result      = cudf::distinct(input, key_idx);
      auto const result_sort = cudf::sort_by_key(*result, result->select({0}));
      CUDF_TEST_EXPECT_TABLES_EQUAL(*expect_table, *result_sort);
    }

    {
      auto const result      = cudf::distinct(input, key_idx, KEEP_ANY);
      auto const result_sort = cudf::sort_by_key(*result, result->select({0}));
      CUDF_TEST_EXPECT_TABLES_EQUAL(*expect_table, *result_sort);
    }
  }

  {
    auto const expect_map   = int32s_col{0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16};
    auto const expect_table = cudf::gather(input, expect_map);

    {
      auto const result      = cudf::distinct(input, key_idx, null_equality::UNEQUAL);
      auto const result_sort = cudf::sort_by_key(*result, result->select({0}));
      CUDF_TEST_EXPECT_TABLES_EQUAL(*expect_table, *result_sort);
    }

    {
      auto const result      = cudf::distinct(input, key_idx, KEEP_ANY, null_equality::UNEQUAL);
      auto const result_sort = cudf::sort_by_key(*result, result->select({0}));
      CUDF_TEST_EXPECT_TABLES_EQUAL(*expect_table, *result_sort);
    }
  }
}

TEST_F(Distinct, ListsOfStructs_KEEP_EXCEPT_ANY)
{
  // Constructing a list of structs of two elements
  // 0.   []                  ==
  // 1.   []                  !=
  // 2.   Null                ==
  // 3.   Null                !=
  // 4.   [Null, Null]        !=
  // 5.   [Null]              ==
  // 6.   [Null]              ==
  // 7.   [Null]              !=
  // 8.   [{Null, Null}]      !=
  // 9.   [{1,'a'}, {2,'b'}]  !=
  // 10.  [{0,'a'}, {2,'b'}]  !=
  // 11.  [{0,'a'}, {2,'c'}]  ==
  // 12.  [{0,'a'}, {2,'c'}]  !=
  // 13.  [{0,Null}]          ==
  // 14.  [{0,Null}]          !=
  // 15.  [{Null, 'b'}]       ==
  // 16.  [{Null, 'b'}]

  auto const structs = [] {
    auto child1 =
      int32s_col{{XXX, XXX, XXX, XXX, XXX, null, 1, 2, 0, 2, 0, 2, 0, 2, 0, 0, null, null},
                 nulls_at({5, 16, 17})};
    auto child2 = strings_col{{"" /*XXX*/,
                               "" /*XXX*/,
                               "" /*XXX*/,
                               "" /*XXX*/,
                               "" /*XXX*/,
                               "" /*null*/,
                               "a",
                               "b",
                               "a",
                               "b",
                               "a",
                               "c",
                               "a",
                               "c",
                               "" /*null*/,
                               "" /*null*/,
                               "b",
                               "b"},
                              nulls_at({5, 14, 15})};

    return structs_col{{child1, child2}, nulls_at({0, 1, 2, 3, 4})};
  }();

  auto const offsets      = int32s_col{0, 0, 0, 0, 0, 2, 3, 4, 5, 6, 8, 10, 12, 14, 15, 16, 17, 18};
  auto const null_it      = nulls_at({2, 3});
  auto const nullmask_buf = cudf::test::detail::make_null_mask(null_it, null_it + 17);
  auto const keys         = cudf::column_view(cudf::data_type(cudf::type_id::LIST),
                                      17,
                                      nullptr,
                                      static_cast<cudf::bitmask_type const*>(nullmask_buf.data()),
                                      cudf::UNKNOWN_NULL_COUNT,
                                      0,
                                      {offsets, structs});

  auto const idx     = int32s_col{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  auto const input   = cudf::table_view{{idx, keys}};
  auto const key_idx = std::vector<cudf::size_type>{1};

  {
    auto const expect_map   = int32s_col{0, 2, 4, 5, 8, 9, 10, 11, 13, 15};
    auto const expect_table = cudf::gather(input, expect_map);

    auto const result      = cudf::distinct(input, key_idx, KEEP_FIRST);
    auto const result_sort = cudf::sort_by_key(*result, result->select({0}));
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expect_table, *result_sort);
  }

  {
    auto const expect_map   = int32s_col{1, 3, 4, 7, 8, 9, 10, 12, 14, 16};
    auto const expect_table = cudf::gather(input, expect_map);

    auto const result      = cudf::distinct(input, key_idx, KEEP_LAST);
    auto const result_sort = cudf::sort_by_key(*result, result->select({0}));
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expect_table, *result_sort);
  }

  {
    auto const expect_map   = int32s_col{4, 8, 9, 10};
    auto const expect_table = cudf::gather(input, expect_map);

    auto const result      = cudf::distinct(input, key_idx, KEEP_NONE);
    auto const result_sort = cudf::sort_by_key(*result, result->select({0}));
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expect_table, *result_sort);
  }
}

TEST_F(Distinct, SlicedListsOfStructs)
{
  // Constructing a list of struct of two elements
  // 0.   []                  ==                <- Don't care
  // 1.   []                  !=                <- Don't care
  // 2.   Null                ==                <- Don't care
  // 3.   Null                !=                <- Don't care
  // 4.   [Null, Null]        !=                <- Don't care
  // 5.   [Null]              ==                <- Don't care
  // 6.   [Null]              ==                <- Don't care
  // 7.   [Null]              !=                <- Don't care
  // 8.   [{Null, Null}]      !=
  // 9.   [{1,'a'}, {2,'b'}]  !=
  // 10.  [{0,'a'}, {2,'b'}]  !=
  // 11.  [{0,'a'}, {2,'c'}]  ==
  // 12.  [{0,'a'}, {2,'c'}]  !=
  // 13.  [{0,Null}]          ==
  // 14.  [{0,Null}]          !=
  // 15.  [{Null, 'b'}]       ==                <- Don't care
  // 16.  [{Null, 'b'}]                         <- Don't care

  auto const structs = [] {
    auto child1 =
      int32s_col{{XXX, XXX, XXX, XXX, XXX, null, 1, 2, 0, 2, 0, 2, 0, 2, 0, 0, null, null},
                 nulls_at({5, 16, 17})};
    auto child2 = strings_col{{"" /*XXX*/,
                               "" /*XXX*/,
                               "" /*XXX*/,
                               "" /*XXX*/,
                               "" /*XXX*/,
                               "" /*null*/,
                               "a",
                               "b",
                               "a",
                               "b",
                               "a",
                               "c",
                               "a",
                               "c",
                               "" /*null*/,
                               "" /*null*/,
                               "b",
                               "b"},
                              nulls_at({5, 14, 15})};

    return structs_col{{child1, child2}, nulls_at({0, 1, 2, 3, 4})};
  }();

  auto const offsets      = int32s_col{0, 0, 0, 0, 0, 2, 3, 4, 5, 6, 8, 10, 12, 14, 15, 16, 17, 18};
  auto const null_it      = nulls_at({2, 3});
  auto const nullmask_buf = cudf::test::detail::make_null_mask(null_it, null_it + 17);
  auto const keys         = cudf::column_view(cudf::data_type(cudf::type_id::LIST),
                                      17,
                                      nullptr,
                                      static_cast<cudf::bitmask_type const*>(nullmask_buf.data()),
                                      cudf::UNKNOWN_NULL_COUNT,
                                      0,
                                      {offsets, structs});

  auto const idx            = int32s_col{1, 1, 2, 2, 3, 4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 10, 10};
  auto const input_original = cudf::table_view{{idx, keys}};
  auto const input          = cudf::slice(input_original, {8, 15})[0];
  auto const key_idx        = std::vector<cudf::size_type>{1};

  {
    auto const expect_map   = int32s_col{8, 9, 10, 11, 13};
    auto const expect_table = cudf::gather(input_original, expect_map);

    {
      auto const result      = cudf::distinct(input, key_idx);
      auto const result_sort = cudf::sort_by_key(*result, result->select({0}));
      CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*expect_table, *result_sort);
    }

    {
      auto const result      = cudf::distinct(input, key_idx, KEEP_ANY);
      auto const result_sort = cudf::sort_by_key(*result, result->select({0}));
      CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*expect_table, *result_sort);
    }
  }

  {
    auto const expect_map   = int32s_col{8, 9, 10, 11, 13, 14};
    auto const expect_table = cudf::gather(input_original, expect_map);

    {
      auto const result      = cudf::distinct(input, key_idx, null_equality::UNEQUAL);
      auto const result_sort = cudf::sort_by_key(*result, result->select({0}));
      CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*expect_table, *result_sort);
    }

    {
      auto const result      = cudf::distinct(input, key_idx, KEEP_ANY, null_equality::UNEQUAL);
      auto const result_sort = cudf::sort_by_key(*result, result->select({0}));
      CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*expect_table, *result_sort);
    }
  }
}

TEST_F(Distinct, ListsOfEmptyStructs)
{
  // 0.  []             ==
  // 1.  []             !=
  // 2.  Null           ==
  // 3.  Null           !=
  // 4.  [Null, Null]   ==
  // 5.  [Null, Null]   ==
  // 6.  [Null, Null]   !=
  // 7.  [Null]         ==
  // 8.  [Null]         !=
  // 9.  [{}]           ==
  // 10. [{}]           !=
  // 11. [{}, {}]       ==
  // 12. [{}, {}]

  auto const structs_null_it = nulls_at({0, 1, 2, 3, 4, 5, 6, 7});
  auto const structs_nullmask_buf =
    cudf::test::detail::make_null_mask(structs_null_it, structs_null_it + 14);
  auto const structs =
    cudf::column_view(cudf::data_type(cudf::type_id::STRUCT),
                      14,
                      nullptr,
                      static_cast<cudf::bitmask_type const*>(structs_nullmask_buf.data()),
                      cudf::UNKNOWN_NULL_COUNT,
                      0,
                      {});

  auto const offsets       = int32s_col{0, 0, 0, 0, 0, 2, 4, 6, 7, 8, 9, 10, 12, 14};
  auto const lists_null_it = nulls_at({2, 3});
  auto const lists_nullmask_buf =
    cudf::test::detail::make_null_mask(lists_null_it, lists_null_it + 13);
  auto const keys =
    cudf::column_view(cudf::data_type(cudf::type_id::LIST),
                      13,
                      nullptr,
                      static_cast<cudf::bitmask_type const*>(lists_nullmask_buf.data()),
                      cudf::UNKNOWN_NULL_COUNT,
                      0,
                      {offsets, structs});

  auto const idx     = int32s_col{1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6};
  auto const input   = cudf::table_view{{idx, keys}};
  auto const key_idx = std::vector<cudf::size_type>{1};

  {
    auto const expect_map   = int32s_col{0, 2, 4, 7, 9, 11};
    auto const expect_table = cudf::gather(input, expect_map);

    {
      auto const result      = cudf::distinct(input, key_idx);
      auto const result_sort = cudf::sort_by_key(*result, result->select({0}));
      CUDF_TEST_EXPECT_TABLES_EQUAL(*expect_table, *result_sort);
    }

    {
      auto const result      = cudf::distinct(input, key_idx, KEEP_ANY);
      auto const result_sort = cudf::sort_by_key(*result, result->select({0}));
      CUDF_TEST_EXPECT_TABLES_EQUAL(*expect_table, *result_sort);
    }
  }

  {
    auto const expect_map   = int32s_col{0, 2, 3, 4, 5, 6, 7, 8, 9, 11};
    auto const expect_table = cudf::gather(input, expect_map);

    {
      auto const result      = cudf::distinct(input, key_idx, null_equality::UNEQUAL);
      auto const result_sort = cudf::sort_by_key(*result, result->select({0}));
      CUDF_TEST_EXPECT_TABLES_EQUAL(*expect_table, *result_sort);
    }

    {
      auto const result      = cudf::distinct(input, key_idx, KEEP_ANY, null_equality::UNEQUAL);
      auto const result_sort = cudf::sort_by_key(*result, result->select({0}));
      CUDF_TEST_EXPECT_TABLES_EQUAL(*expect_table, *result_sort);
    }
  }
}

TEST_F(Distinct, StructsOfStructs)
{
  //  +-----------------+
  //  |  s1{s2{a,b}, c} |
  //  +-----------------+
  // 0 |  { {1, 1}, 5}  |
  // 1 |  { {1, 2}, 4}  |
  // 2 |  { Null,   6}  |
  // 3 |  { Null,   4}  |
  // 4 |  Null          |
  // 5 |  Null          |  // Same as 4
  // 6 |  { {1, 1}, 5}  |  // Same as 0
  // 7 |  { Null,   4}  |  // Same as 3
  // 8 |  { {2, 1}, 5}  |

  auto s1 = [&] {
    auto a  = int32s_col{1, 1, XXX, XXX, 2, 1, 1, XXX, 2};
    auto b  = int32s_col{1, 2, XXX, XXX, 2, 1, 1, XXX, 1};
    auto s2 = structs_col{{a, b}, nulls_at({2, 3, 7})};

    auto c = int32s_col{5, 4, 6, 4, 3, 3, 5, 4, 5};
    std::vector<std::unique_ptr<cudf::column>> s1_children;
    s1_children.emplace_back(s2.release());
    s1_children.emplace_back(c.release());
    auto const null_it = nulls_at({4, 5});
    return structs_col(std::move(s1_children), std::vector<bool>{null_it, null_it + 9});
  }();

  auto const idx     = int32s_col{0, 1, 2, 3, 4, 4, 0, 3, 8};
  auto const input   = cudf::table_view{{idx, s1}};
  auto const key_idx = std::vector<cudf::size_type>{1};

  {
    auto const expect_map   = int32s_col{0, 1, 2, 3, 4, 8};
    auto const expect_table = cudf::gather(input, expect_map);

    {
      auto const result      = cudf::distinct(input, key_idx);
      auto const result_sort = cudf::sort_by_key(*result, result->select({0}));
      CUDF_TEST_EXPECT_TABLES_EQUAL(*expect_table, *result_sort);
    }

    {
      auto const result      = cudf::distinct(input, key_idx, KEEP_ANY);
      auto const result_sort = cudf::sort_by_key(*result, result->select({0}));
      CUDF_TEST_EXPECT_TABLES_EQUAL(*expect_table, *result_sort);
    }
  }

  {
    auto const expect_map   = int32s_col{0, 1, 2, 3, 7, 4, 5, 8};
    auto const expect_table = cudf::gather(input, expect_map);

    {
      auto const result      = cudf::distinct(input, key_idx, null_equality::UNEQUAL);
      auto const result_sort = cudf::sort_by_key(*result, result->select({0}));
      CUDF_TEST_EXPECT_TABLES_EQUAL(*expect_table, *result_sort);
    }

    {
      auto const result      = cudf::distinct(input, key_idx, KEEP_ANY, null_equality::UNEQUAL);
      auto const result_sort = cudf::sort_by_key(*result, result->select({0}));
      CUDF_TEST_EXPECT_TABLES_EQUAL(*expect_table, *result_sort);
    }
  }
}

TEST_F(Distinct, SlicedStructsOfStructs)
{
  //  +-----------------+
  //  |  s1{s2{a,b}, c} |
  //  +-----------------+
  // 0 |  { {1, 1}, 5}  |
  // 1 |  { {1, 2}, 4}  |
  // 2 |  { Null,   6}  |
  // 3 |  { Null,   4}  |
  // 4 |  Null          |
  // 5 |  Null          |  // Same as 4
  // 6 |  { {1, 1}, 5}  |  // Same as 0
  // 7 |  { Null,   4}  |  // Same as 3
  // 8 |  { {2, 1}, 5}  |

  auto s1 = [&] {
    auto a  = int32s_col{1, 1, 2, 2, 2, 1, 1, 1, 2};
    auto b  = int32s_col{1, 2, 1, 2, 2, 1, 1, 1, 1};
    auto s2 = structs_col{{a, b}, nulls_at({2, 3, 7})};

    auto c = int32s_col{5, 4, 6, 4, 3, 3, 5, 4, 5};
    std::vector<std::unique_ptr<cudf::column>> s1_children;
    s1_children.emplace_back(s2.release());
    s1_children.emplace_back(c.release());
    auto const null_it = nulls_at({4, 5});
    return structs_col(std::move(s1_children), std::vector<bool>{null_it, null_it + 9});
  }();

  auto const idx            = int32s_col{0, 1, 2, 3, 4, 4, 0, 3, 8};
  auto const input_original = cudf::table_view{{idx, s1}};
  auto const input          = cudf::slice(input_original, {1, 7})[0];
  auto const key_idx        = std::vector<cudf::size_type>{1};

  {
    auto const expect_map   = int32s_col{6, 1, 2, 3, 4};
    auto const expect_table = cudf::gather(input_original, expect_map);

    {
      auto const result      = cudf::distinct(input, key_idx);
      auto const result_sort = cudf::sort_by_key(*result, result->select({0}));
      CUDF_TEST_EXPECT_TABLES_EQUAL(*expect_table, *result_sort);
    }

    {
      auto const result      = cudf::distinct(input, key_idx, KEEP_ANY);
      auto const result_sort = cudf::sort_by_key(*result, result->select({0}));
      CUDF_TEST_EXPECT_TABLES_EQUAL(*expect_table, *result_sort);
    }
  }

  {
    auto const expect_map   = int32s_col{6, 1, 2, 3, 4, 5};
    auto const expect_table = cudf::gather(input_original, expect_map);

    {
      auto const result      = cudf::distinct(input, key_idx, null_equality::UNEQUAL);
      auto const result_sort = cudf::sort_by_key(*result, result->select({0}));
      CUDF_TEST_EXPECT_TABLES_EQUAL(*expect_table, *result_sort);
    }

    {
      auto const result      = cudf::distinct(input, key_idx, KEEP_ANY, null_equality::UNEQUAL);
      auto const result_sort = cudf::sort_by_key(*result, result->select({0}));
      CUDF_TEST_EXPECT_TABLES_EQUAL(*expect_table, *result_sort);
    }
  }
}

TEST_F(Distinct, SlicedStructsOfLists)
{
  auto const idx  = int32s_col{0, 0, 1, 2, 1, 3, 4, 5, 5, 6, 4, 4, 70};
  auto const keys = [] {
    // All child columns are identical.
    auto child1 = lists_col{
      {0, 0}, {0, 0}, {1}, {1, 1}, {1}, {1, 2}, {2, 2}, {2}, {2}, {2, 1}, {2, 2}, {2, 2}, {5, 5}};
    auto child2 = lists_col{
      {0, 0}, {0, 0}, {1}, {1, 1}, {1}, {1, 2}, {2, 2}, {2}, {2}, {2, 1}, {2, 2}, {2, 2}, {5, 5}};
    auto child3 = lists_col{
      {0, 0}, {0, 0}, {1}, {1, 1}, {1}, {1, 2}, {2, 2}, {2}, {2}, {2, 1}, {2, 2}, {2, 2}, {5, 5}};
    return structs_col{{child1, child2, child3}};
  }();

  auto const input_original = cudf::table_view{{idx, keys}};
  auto const input          = cudf::slice(input_original, {2, 12})[0];
  auto const key_idx        = std::vector<cudf::size_type>{1};

  auto const exp_idx_sort  = int32s_col{1, 2, 3, 4, 5, 6};
  auto const exp_keys_sort = [] {
    auto child1 = lists_col{{1}, {1, 1}, {1, 2}, {2, 2}, {2}, {2, 1}};
    auto child2 = lists_col{{1}, {1, 1}, {1, 2}, {2, 2}, {2}, {2, 1}};
    auto child3 = lists_col{{1}, {1, 1}, {1, 2}, {2, 2}, {2}, {2, 1}};
    return structs_col{{child1, child2, child3}};
  }();
  auto const expected_sort = cudf::table_view{{exp_idx_sort, exp_keys_sort}};

  {
    auto const result      = cudf::distinct(input, key_idx);
    auto const result_sort = cudf::sort_by_key(*result, result->select({0}));
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected_sort, *result_sort);
  }

  {
    auto const result      = cudf::distinct(input, key_idx, KEEP_ANY);
    auto const result_sort = cudf::sort_by_key(*result, result->select({0}));
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected_sort, *result_sort);
  }
}

TEST_F(Distinct, EmptyDeepList)
{
  // List<List<int>>, where all lists are empty
  // []
  // []
  // Null
  // Null

  // Internal empty list
  auto list1 = cudf::test::lists_column_wrapper<int>{};

  auto offsets       = cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 0, 0, 0, 0};
  auto list_nullmask = std::vector<bool>{1, 1, 0, 0};
  auto list_validity_buffer =
    cudf::test::detail::make_null_mask(list_nullmask.begin(), list_nullmask.end());
  auto list_column = cudf::make_lists_column(4,
                                             offsets.release(),
                                             list1.release(),
                                             cudf::UNKNOWN_NULL_COUNT,
                                             std::move(list_validity_buffer));

  auto idx   = cudf::test::fixed_width_column_wrapper<cudf::size_type>{1, 1, 2, 2};
  auto input = cudf::table_view{{idx, *list_column}};

  auto expect_map = cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 2};
  auto expect     = cudf::gather(input, expect_map);

  auto result      = cudf::distinct(input, {1}, KEEP_ANY);
  auto result_sort = cudf::sort_by_key(*result, result->select({0}));
  CUDF_TEST_EXPECT_TABLES_EQUAL(*expect, *result_sort);
}
