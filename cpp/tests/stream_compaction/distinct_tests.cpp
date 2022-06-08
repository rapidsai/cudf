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

auto constexpr null{0};
auto constexpr KEEP_ANY = cudf::duplicate_keep_option::KEEP_ANY;

using int32s_col  = cudf::test::fixed_width_column_wrapper<int32_t>;
using floats_col  = cudf::test::fixed_width_column_wrapper<float>;
using strings_col = cudf::test::strings_column_wrapper;
using cudf::duplicate_keep_option;
using cudf::nan_policy;
using cudf::null_equality;
using cudf::null_policy;
using cudf::test::iterators::no_nulls;
using cudf::test::iterators::null_at;
using cudf::test::iterators::nulls_at;

template <typename Input, typename Keys>
auto sorted_distinct(Input const& input, Keys const& keys, duplicate_keep_option keep)
{
  auto const result = cudf::distinct(input, keys, keep);
  return cudf::sort_by_key(*result, result->select(keys));
}

struct Distinct : public cudf::test::BaseFixture {
};

TEST_F(Distinct, StringKeyColumn_KEEP_ANY)
{
  // Column used to test KEEP_ANY needs to have same rows for same keys.
  auto const col = int32s_col{{5, null, null, 5, 5, 8, 1}, nulls_at({1, 2})};
  auto const keys_col =
    strings_col{{"all", "new", "new", "all", "" /*NULL*/, "the", "strings"}, null_at(4)};
  auto const keys  = std::vector<cudf::size_type>{1};
  auto const input = cudf::table_view{{col, keys_col}};

  auto const exp_col_sort  = int32s_col{{5, 5, null, 1, 8}, null_at(2)};
  auto const exp_keys_sort = strings_col{{"" /*NULL*/, "all", "new", "strings", "the"}, null_at(0)};
  auto const expected_sort = cudf::table_view{{exp_col_sort, exp_keys_sort}};

  auto const result_sort = sorted_distinct(input, keys, duplicate_keep_option::KEEP_ANY);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_sort, *result_sort);
}

TEST_F(Distinct, StringKeyColumn_KEEP_EXCEPT_ANY)
{
  // Column used to test needs to have different rows for the same keys.
  auto const col = int32s_col{{0, null, 2, 3, 4, 5, 6}, null_at(1)};
  auto const keys_col =
    strings_col{{"all", "new", "new", "all", "" /*NULL*/, "the", "strings"}, null_at(4)};
  auto const keys  = std::vector<cudf::size_type>{1};
  auto const input = cudf::table_view{{col, keys_col}};

  {
    auto const exp_col_sort = int32s_col{{4, 0, null, 6, 5}, null_at(2)};
    auto const exp_keys_sort =
      strings_col{{"" /*NULL*/, "all", "new", "strings", "the"}, null_at(0)};
    auto const expected_sort = cudf::table_view{{exp_col_sort, exp_keys_sort}};

    auto const result_sort = sorted_distinct(input, keys, duplicate_keep_option::KEEP_FIRST);
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected_sort, *result_sort);
  }

  {
    auto const exp_col_sort = int32s_col{{4, 3, 2, 6, 5}, no_nulls()};
    auto const exp_keys_sort =
      strings_col{{"" /*NULL*/, "all", "new", "strings", "the"}, null_at(0)};
    auto const expected_sort = cudf::table_view{{exp_col_sort, exp_keys_sort}};

    auto const result_sort = sorted_distinct(input, keys, duplicate_keep_option::KEEP_LAST);
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected_sort, *result_sort);
  }

  {
    auto const exp_col_sort  = int32s_col{{4, 6, 5}, no_nulls()};
    auto const exp_keys_sort = strings_col{{"" /*NULL*/, "strings", "the"}, null_at(0)};
    auto const expected_sort = cudf::table_view{{exp_col_sort, exp_keys_sort}};

    auto const result_sort = sorted_distinct(input, keys, duplicate_keep_option::KEEP_NONE);
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected_sort, *result_sort);
  }
}

TEST_F(Distinct, EmptyInputTable)
{
  int32s_col col(std::initializer_list<int32_t>{});
  cudf::table_view input{{col}};
  std::vector<cudf::size_type> keys{1, 2};

  auto got = cudf::distinct(input, keys, null_equality::EQUAL);
  CUDF_TEST_EXPECT_TABLES_EQUAL(input, got->view());
}

TEST_F(Distinct, NoColumnInputTable)
{
  cudf::table_view input{std::vector<cudf::column_view>()};
  std::vector<cudf::size_type> keys{1, 2};

  auto got = cudf::distinct(input, keys, null_equality::EQUAL);
  CUDF_TEST_EXPECT_TABLES_EQUAL(input, got->view());
}

TEST_F(Distinct, EmptyKeys)
{
  int32s_col col{{5, 4, 3, 5, 8, 1}, {1, 0, 1, 1, 1, 1}};
  int32s_col empty_col{};
  cudf::table_view input{{col}};
  std::vector<cudf::size_type> keys{};

  auto got = cudf::distinct(input, keys, null_equality::EQUAL);
  CUDF_TEST_EXPECT_TABLES_EQUAL(cudf::table_view{{empty_col}}, got->view());
}

TEST_F(Distinct, NonNullTable)
{
  int32s_col col1{{6, 6, 3, 5, 8, 5}};
  floats_col col2{{6, 6, 3, 4, 9, 4}};
  int32s_col col1_key{{20, 20, 20, 19, 21, 9}};
  int32s_col col2_key{{19, 19, 20, 20, 9, 21}};

  cudf::table_view input{{col1, col2, col1_key, col2_key}};
  std::vector<cudf::size_type> keys{2, 3};

  // The expected table would be sorted in ascending order with respect to keys
  int32s_col exp_col1{{5, 5, 6, 3, 8}};
  floats_col exp_col2{{4, 4, 6, 3, 9}};
  int32s_col exp_col1_key{{9, 19, 20, 20, 21}};
  int32s_col exp_col2_key{{21, 20, 19, 20, 9}};
  cudf::table_view expected{{exp_col1, exp_col2, exp_col1_key, exp_col2_key}};

  auto result      = cudf::distinct(input, keys, KEEP_ANY);
  auto key_view    = result->select(keys.begin(), keys.end());
  auto result_sort = cudf::sort_by_key(result->view(), key_view);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result_sort->view());
}

TEST_F(Distinct, SlicedNonNullTable)
{
  using int32s_col         = int32s_col;
  using floats_col         = floats_col;
  auto constexpr dont_care = int32_t{0};

  auto const col1     = int32s_col{dont_care, dont_care, 6, 6, 3, 5, 8, 5, dont_care};
  auto const col2     = floats_col{dont_care, dont_care, 6, 6, 3, 4, 9, 4, dont_care};
  auto const col1_key = int32s_col{dont_care, dont_care, 20, 20, 20, 19, 21, 9, dont_care};
  auto const col2_key = int32s_col{dont_care, dont_care, 19, 19, 20, 20, 9, 21, dont_care};

  auto const input_original = cudf::table_view{{col1, col2, col1_key, col2_key}};
  auto const input          = cudf::slice(input_original, {2, 8})[0];
  auto const keys           = std::vector<cudf::size_type>{2, 3};

  // The expected table would be sorted in ascending order with respect to keys.
  auto const exp_col1     = int32s_col{{5, 5, 6, 3, 8}};
  auto const exp_col2     = floats_col{{4, 4, 6, 3, 9}};
  auto const exp_col1_key = int32s_col{{9, 19, 20, 20, 21}};
  auto const exp_col2_key = int32s_col{{21, 20, 19, 20, 9}};
  auto const expected     = cudf::table_view{{exp_col1, exp_col2, exp_col1_key, exp_col2_key}};

  auto const result      = cudf::distinct(input, keys, KEEP_ANY);
  auto const key_view    = result->select(keys.begin(), keys.end());
  auto const result_sort = cudf::sort_by_key(result->view(), key_view);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result_sort->view());
}

TEST_F(Distinct, WithNull)
{
  int32s_col col{{5, 4, 4, 1, 8, 1}, {1, 0, 1, 1, 1, 1}};
  int32s_col key{{20, 20, 20, 19, 21, 19}, {1, 0, 0, 1, 1, 1}};
  cudf::table_view input{{col, key}};
  std::vector<cudf::size_type> keys{1};

  // nulls are equal
  int32s_col exp_equal_col{{4, 1, 5, 8}, {0, 1, 1, 1}};
  int32s_col exp_equal_keys_col{{20, 19, 20, 21}, {0, 1, 1, 1}};
  cudf::table_view expected_equal{{exp_equal_col, exp_equal_keys_col}};
  auto res_equal    = cudf::distinct(input, keys, KEEP_ANY, null_equality::EQUAL);
  auto equal_keys   = res_equal->select(keys.begin(), keys.end());
  auto sorted_equal = cudf::sort_by_key(res_equal->view(), equal_keys);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_equal, sorted_equal->view());

  // nulls are unequal
  int32s_col exp_unequal_col{{4, 1, 4, 5, 8}, {0, 1, 1, 1, 1}};
  int32s_col exp_unequal_keys_col{{20, 19, 20, 20, 21}, {0, 1, 0, 1, 1}};
  cudf::table_view expected_unequal{{exp_unequal_col, exp_unequal_keys_col}};
  auto res_unequal    = cudf::distinct(input, keys, KEEP_ANY, null_equality::UNEQUAL);
  auto sorted_unequal = cudf::sort(res_unequal->view());

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_unequal, sorted_unequal->view());
}

TEST_F(Distinct, BasicList)
{
  using LCW = cudf::test::lists_column_wrapper<uint64_t>;
  using ICW = cudf::test::fixed_width_column_wrapper<cudf::size_type>;

  // clang-format off
auto const idx = ICW{ 0,  0,   1,      2,   1,      3,      4,   5,   5,      6,      4,     4 };
auto const col = LCW{{}, {}, {1}, {1, 1}, {1}, {1, 2}, {2, 2}, {2}, {2}, {2, 1}, {2, 2}, {2, 2}};
  // clang-format on
  auto const input = cudf::table_view({idx, col});

  auto const exp_idx = ICW{0, 1, 2, 3, 4, 5, 6};
  auto const exp_val = LCW{{}, {1}, {1, 1}, {1, 2}, {2, 2}, {2}, {2, 1}};
  auto const expect  = cudf::table_view({exp_idx, exp_val});

  auto result      = cudf::distinct(input, {1}, KEEP_ANY);
  auto result_sort = cudf::sort_by_key(*result, result->select({0}));

  CUDF_TEST_EXPECT_TABLES_EQUAL(expect, *result_sort);
}

TEST_F(Distinct, BasicSlicedLists)
{
  using int32s_col         = int32s_col;
  using lists_col          = cudf::test::lists_column_wrapper<int32_t>;
  auto constexpr dont_care = int32_t{0};

  auto const idx = int32s_col{dont_care, dont_care, 1, 2, 1, 3, 4, 5, 5, 6, 4, 4, dont_care};
  auto const col = lists_col{
    {0, 0}, {0, 0}, {1}, {1, 1}, {1}, {1, 2}, {2, 2}, {2}, {2}, {2, 1}, {2, 2}, {2, 2}, {5, 5}};
  auto const input_original = cudf::table_view({idx, col});
  auto const input          = cudf::slice(input_original, {2, 12})[0];

  auto const exp_idx  = int32s_col{1, 2, 3, 4, 5, 6};
  auto const exp_val  = lists_col{{1}, {1, 1}, {1, 2}, {2, 2}, {2}, {2, 1}};
  auto const expected = cudf::table_view({exp_idx, exp_val});

  auto const result      = cudf::distinct(input, {1}, KEEP_ANY);
  auto const result_sort = cudf::sort_by_key(*result, result->select({0}));

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, *result_sort);
}

TEST_F(Distinct, NullableList)
{
  using LCW  = cudf::test::lists_column_wrapper<uint64_t>;
  using ICW  = cudf::test::fixed_width_column_wrapper<cudf::size_type>;
  using mask = std::vector<bool>;

  // clang-format off
auto const idx    = ICW {  0,  0,   1,   1,      4,   5,   5,  6,       4,     4,  6};
auto const valids = mask{  1,  1,   1,   1,      1,   1,   1,  0,       1,     1,  0};
auto const col    = LCW {{{}, {}, {1}, {1}, {2, 2}, {2}, {2}, {}, {2, 2}, {2, 2}, {}},
valids.begin()};

auto const exp_idx    = ICW {  0,   1,      4,   5,  6};
auto const exp_valids = mask{  1,   1,      1,   1,  0};
auto const exp_val    = LCW {{{}, {1}, {2, 2}, {2}, {}}, exp_valids.begin()};

  // clang-format on
  auto const input  = cudf::table_view({idx, col});
  auto const expect = cudf::table_view({exp_idx, exp_val});

  auto result      = cudf::distinct(input, {1}, KEEP_ANY);
  auto result_sort = cudf::sort_by_key(*result, result->select({0}));

  CUDF_TEST_EXPECT_TABLES_EQUAL(expect, *result_sort);
}

TEST_F(Distinct, ListOfStruct)
{
  // Constructing a list of struct of two elements
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

  auto col1 = int32s_col{{-1, -1, 0, 2, 2, 2, 1, 2, 0, 2, 0, 2, 0, 2, 0, 0, 1, 2},
                         {1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0}};
  auto col2 = strings_col{
    {"x", "x", "a", "a", "b", "b", "a", "b", "a", "b", "a", "c", "a", "c", "a", "c", "b", "b"},
    {1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1}};
  auto struct_col = cudf::test::structs_column_wrapper{
    {col1, col2}, {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}};

  auto offsets = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    0, 0, 0, 0, 0, 2, 3, 4, 5, 6, 8, 10, 12, 14, 15, 16, 17, 18};

  auto list_nullmask = std::vector<bool>{1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  auto nullmask_buf =
    cudf::test::detail::make_null_mask(list_nullmask.begin(), list_nullmask.end());
  auto list_column = cudf::column_view(cudf::data_type(cudf::type_id::LIST),
                                       17,
                                       nullptr,
                                       static_cast<cudf::bitmask_type*>(nullmask_buf.data()),
                                       cudf::UNKNOWN_NULL_COUNT,
                                       0,
                                       {offsets, struct_col});

  auto idx = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    1, 1, 2, 2, 3, 4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 10, 10};

  auto input = cudf::table_view({idx, list_column});

  auto expect_map =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 2, 4, 5, 8, 9, 10, 11, 13, 15};

  auto expect_table = cudf::gather(input, expect_map);

  auto result      = cudf::distinct(input, {1}, KEEP_ANY);
  auto result_sort = cudf::sort_by_key(*result, result->select({0}));

  CUDF_TEST_EXPECT_TABLES_EQUAL(*expect_table, *result_sort);
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

  using int32s_col  = int32s_col;
  using strings_col = strings_col;
  using structs_col = cudf::test::structs_column_wrapper;
  using cudf::test::iterators::nulls_at;

  auto const structs = [] {
    auto child1 =
      int32s_col{{-1, -1, 0, 2, 2, 2, 1, 2, 0, 2, 0, 2, 0, 2, 0, 0, 1, 2}, nulls_at({5, 16, 17})};
    auto child2 = strings_col{
      {"x", "x", "a", "a", "b", "b", "a", "b", "a", "b", "a", "c", "a", "c", "a", "c", "b", "b"},
      nulls_at({5, 14, 15})};
    return structs_col{{child1, child2}, nulls_at({0, 1, 2, 3, 4})};
  }();

  auto const offsets = int32s_col{0, 0, 0, 0, 0, 2, 3, 4, 5, 6, 8, 10, 12, 14, 15, 16, 17, 18};
  auto const lists_nullmask = std::vector<bool>{1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  auto const nullmask_buf =
    cudf::test::detail::make_null_mask(lists_nullmask.begin(), lists_nullmask.end());
  auto const lists = cudf::column_view(cudf::data_type(cudf::type_id::LIST),
                                       17,
                                       nullptr,
                                       static_cast<cudf::bitmask_type const*>(nullmask_buf.data()),
                                       cudf::UNKNOWN_NULL_COUNT,
                                       0,
                                       {offsets, structs});

  auto const idx            = int32s_col{1, 1, 2, 2, 3, 4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 10, 10};
  auto const input_original = cudf::table_view({idx, lists});
  auto const input          = cudf::slice(input_original, {8, 15})[0];

  auto const result      = cudf::distinct(input, {1}, KEEP_ANY);
  auto const result_sort = cudf::sort_by_key(*result, result->select({0}));

  auto const exp_map = cudf::test::fixed_width_column_wrapper<cudf::size_type>{8, 9, 10, 11, 13};
  auto const expected_table = cudf::gather(input_original, exp_map);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*expected_table, *result_sort);
}

TEST_F(Distinct, StructOfStruct)
{
  using FWCW = cudf::test::fixed_width_column_wrapper<int>;
  using MASK = std::vector<bool>;

  /*
  `@` indicates null

  /+-------------+
  |s1{s2{a,b}, c}|
  +--------------+
  0 |  { {1, 1}, 5}|
  1 |  { {1, 2}, 4}|
  2 |  {@{2, 1}, 6}|
  3 |  {@{2, 2}, 4}|
  4 | @{ {2, 2}, 3}|
  5 | @{ {1, 1}, 3}|  // Same as 4
  6 |  { {1, 1}, 5}|  // Same as 0
  7 |  {@{1, 1}, 4}|  // Same as 3
  8 |  { {2, 1}, 5}|
  +--------------+
  */

  auto col_a   = FWCW{1, 1, 2, 2, 2, 1, 1, 1, 2};
  auto col_b   = FWCW{1, 2, 1, 2, 2, 1, 1, 1, 1};
  auto s2_mask = MASK{1, 1, 0, 0, 1, 1, 1, 0, 1};
  auto col_c   = FWCW{5, 4, 6, 4, 3, 3, 5, 4, 5};
  auto s1_mask = MASK{1, 1, 1, 1, 0, 0, 1, 1, 1};
  auto idx     = FWCW{0, 1, 2, 3, 4, 5, 6, 7, 8};

  std::vector<std::unique_ptr<cudf::column>> s2_children;
  s2_children.push_back(col_a.release());
  s2_children.push_back(col_b.release());
  auto s2 = cudf::test::structs_column_wrapper(std::move(s2_children), s2_mask);

  std::vector<std::unique_ptr<cudf::column>> s1_children;
  s1_children.push_back(s2.release());
  s1_children.push_back(col_c.release());
  auto s1 = cudf::test::structs_column_wrapper(std::move(s1_children), s1_mask);

  auto input = cudf::table_view({idx, s1});

  auto expect_map = cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 1, 2, 3, 4, 8};
  auto expect     = cudf::gather(input, expect_map);

  auto result      = cudf::distinct(input, {1}, KEEP_ANY);
  auto result_sort = cudf::sort_by_key(*result, result->select({0}));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect->get_column(1), result_sort->get_column(1));

  auto sliced_input      = cudf::slice(input, {1, 7});
  auto sliced_expect_map = cudf::test::fixed_width_column_wrapper<cudf::size_type>{1, 2, 3, 4, 6};
  auto sliced_expect     = cudf::gather(input, sliced_expect_map);

  auto sliced_result        = cudf::distinct(sliced_input, {1}, KEEP_ANY);
  auto sorted_sliced_result = cudf::sort_by_key(*sliced_result, sliced_result->select({0}));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sliced_expect->get_column(1), sorted_sliced_result->get_column(1));
}

TEST_F(Distinct, SlicedStructsOfLists)
{
  using lists_col   = cudf::test::lists_column_wrapper<int32_t>;
  using structs_col = cudf::test::structs_column_wrapper;

  auto const idx =
    cudf::test::fixed_width_column_wrapper<int>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  auto const structs = [] {
    auto child = lists_col{
      {0, 0}, {0, 0}, {1}, {1, 1}, {1}, {1, 2}, {2, 2}, {2}, {2}, {2, 1}, {2, 2}, {2, 2}, {5, 5}};
    return structs_col{{child}};
  }();

  auto const input_original = cudf::table_view({idx, structs});
  auto const input          = cudf::slice(input_original, {2, 12})[0];

  auto const expected_structs = [] {
    auto child = lists_col{{1}, {1, 1}, {1, 2}, {2, 2}, {2}, {2, 1}};
    return structs_col{{child}};
  }();
  auto const expected = cudf::table_view({expected_structs});

  auto const result      = cudf::distinct(input, {1}, KEEP_ANY);
  auto const result_sort = cudf::sort_by_key(*result, result->select({0}));
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, cudf::table_view{{result_sort->get_column(1)}});
}

TEST_F(Distinct, StructWithNullElement)
{
  using FWCW = cudf::test::fixed_width_column_wrapper<int>;
  using MASK = std::vector<bool>;

  /*
    `@` indicates null

  /+-------------+
    |s1{s2{a,b}, c}|
    +--------------+
                         0 |  { {1, 1}, 2}|
    1 |  {@{1, 1}, 2}|
    +--------------+
                           */

  auto col_a   = FWCW{1, 1};
  auto col_b   = FWCW{1, 1};
  auto s2_mask = MASK{1, 0};
  auto col_c   = FWCW{2, 2};
  auto s1_mask = MASK{1, 1};
  auto idx     = FWCW{0, 1};

  std::vector<std::unique_ptr<cudf::column>> s2_children;
  s2_children.push_back(col_a.release());
  s2_children.push_back(col_b.release());
  auto s2 = cudf::test::structs_column_wrapper(std::move(s2_children), s2_mask);

  std::vector<std::unique_ptr<cudf::column>> s1_children;
  s1_children.push_back(s2.release());
  s1_children.push_back(col_c.release());
  auto s1 = cudf::test::structs_column_wrapper(std::move(s1_children), s1_mask);

  auto input = cudf::table_view({idx, s1});

  auto result      = cudf::distinct(input, {1}, KEEP_ANY);
  auto result_sort = cudf::sort_by_key(*result, result->select({0}));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(input.column(1), result_sort->get_column(1));
}

TEST_F(Distinct, ListOfEmptyStruct)
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

  using mask = std::vector<bool>;

  auto struct_validity = mask{0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
  auto struct_validity_buffer =
    cudf::test::detail::make_null_mask(struct_validity.begin(), struct_validity.end());
  auto struct_col =
    cudf::make_structs_column(14, {}, cudf::UNKNOWN_NULL_COUNT, std::move(struct_validity_buffer));

  auto offsets = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    0, 0, 0, 0, 0, 2, 4, 6, 7, 8, 9, 10, 12, 14};
  auto list_nullmask = mask{1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  auto list_validity_buffer =
    cudf::test::detail::make_null_mask(list_nullmask.begin(), list_nullmask.end());
  auto list_column = cudf::make_lists_column(13,
                                             offsets.release(),
                                             std::move(struct_col),
                                             cudf::UNKNOWN_NULL_COUNT,
                                             std::move(list_validity_buffer));
  auto idx =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6};
  auto input = cudf::table_view({idx, *list_column});

  auto expect_map = cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 2, 4, 7, 9, 11};
  auto expect     = cudf::gather(input, expect_map);

  auto result      = cudf::distinct(input, {1}, KEEP_ANY);
  auto result_sort = cudf::sort_by_key(*result, result->select({0}));
  CUDF_TEST_EXPECT_TABLES_EQUAL(*expect, *result_sort);
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
  auto input = cudf::table_view({idx, *list_column});

  auto expect_map = cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 2};
  auto expect     = cudf::gather(input, expect_map);

  auto result      = cudf::distinct(input, {1}, KEEP_ANY);
  auto result_sort = cudf::sort_by_key(*result, result->select({0}));
  CUDF_TEST_EXPECT_TABLES_EQUAL(*expect, *result_sort);
}
