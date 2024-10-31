/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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
#include <cudf_test/testing_main.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/join.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <limits>

template <typename T>
using column_wrapper = cudf::test::fixed_width_column_wrapper<T>;
using strcol_wrapper = cudf::test::strings_column_wrapper;
using CVector        = std::vector<std::unique_ptr<cudf::column>>;
using Table          = cudf::table;
constexpr cudf::size_type NoneValue =
  std::numeric_limits<cudf::size_type>::min();  // TODO: how to test if this isn't public?

// This function is a wrapper around cudf's join APIs that takes the gather map
// from join APIs and materializes the table that would be created by gathering
// from the joined tables. Join APIs originally returned tables like this, but
// they were modified in https://github.com/rapidsai/cudf/pull/7454. This
// helper function allows us to avoid rewriting all our tests in terms of
// gather maps.
template <std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
                    std::unique_ptr<rmm::device_uvector<cudf::size_type>>> (*join_impl)(
            cudf::table_view const& left_keys,
            cudf::table_view const& right_keys,
            cudf::null_equality compare_nulls,
            rmm::cuda_stream_view stream,
            rmm::device_async_resource_ref mr),
          cudf::out_of_bounds_policy oob_policy = cudf::out_of_bounds_policy::DONT_CHECK>
std::unique_ptr<cudf::table> join_and_gather(
  cudf::table_view const& left_input,
  cudf::table_view const& right_input,
  std::vector<cudf::size_type> const& left_on,
  std::vector<cudf::size_type> const& right_on,
  cudf::null_equality compare_nulls,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref())
{
  auto left_selected  = left_input.select(left_on);
  auto right_selected = right_input.select(right_on);
  auto const [left_join_indices, right_join_indices] =
    join_impl(left_selected, right_selected, compare_nulls, stream, mr);

  auto left_indices_span  = cudf::device_span<cudf::size_type const>{*left_join_indices};
  auto right_indices_span = cudf::device_span<cudf::size_type const>{*right_join_indices};

  auto left_indices_col  = cudf::column_view{left_indices_span};
  auto right_indices_col = cudf::column_view{right_indices_span};

  auto left_result  = cudf::gather(left_input, left_indices_col, oob_policy);
  auto right_result = cudf::gather(right_input, right_indices_col, oob_policy);

  auto joined_cols = left_result->release();
  auto right_cols  = right_result->release();
  joined_cols.insert(joined_cols.end(),
                     std::make_move_iterator(right_cols.begin()),
                     std::make_move_iterator(right_cols.end()));
  return std::make_unique<cudf::table>(std::move(joined_cols));
}

std::unique_ptr<cudf::table> inner_join(
  cudf::table_view const& left_input,
  cudf::table_view const& right_input,
  std::vector<cudf::size_type> const& left_on,
  std::vector<cudf::size_type> const& right_on,
  cudf::null_equality compare_nulls = cudf::null_equality::EQUAL)
{
  return join_and_gather<cudf::inner_join>(
    left_input, right_input, left_on, right_on, compare_nulls);
}

std::unique_ptr<cudf::table> left_join(
  cudf::table_view const& left_input,
  cudf::table_view const& right_input,
  std::vector<cudf::size_type> const& left_on,
  std::vector<cudf::size_type> const& right_on,
  cudf::null_equality compare_nulls = cudf::null_equality::EQUAL)
{
  return join_and_gather<cudf::left_join, cudf::out_of_bounds_policy::NULLIFY>(
    left_input, right_input, left_on, right_on, compare_nulls);
}

std::unique_ptr<cudf::table> full_join(
  cudf::table_view const& full_input,
  cudf::table_view const& right_input,
  std::vector<cudf::size_type> const& full_on,
  std::vector<cudf::size_type> const& right_on,
  cudf::null_equality compare_nulls = cudf::null_equality::EQUAL)
{
  return join_and_gather<cudf::full_join, cudf::out_of_bounds_policy::NULLIFY>(
    full_input, right_input, full_on, right_on, compare_nulls);
}

struct JoinTest : public cudf::test::BaseFixture {
  std::pair<std::unique_ptr<cudf::table>, std::unique_ptr<cudf::table>> gather_maps_as_tables(
    cudf::column_view const& expected_left_map,
    cudf::column_view const& expected_right_map,
    std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
              std::unique_ptr<rmm::device_uvector<cudf::size_type>>> const& result)
  {
    auto result_table =
      cudf::table_view({cudf::column_view{cudf::data_type{cudf::type_id::INT32},
                                          static_cast<cudf::size_type>(result.first->size()),
                                          result.first->data(),
                                          nullptr,
                                          0},
                        cudf::column_view{cudf::data_type{cudf::type_id::INT32},
                                          static_cast<cudf::size_type>(result.second->size()),
                                          result.second->data(),
                                          nullptr,
                                          0}});
    auto result_sort_order = cudf::sorted_order(result_table);
    auto sorted_result     = cudf::gather(result_table, *result_sort_order);

    cudf::table_view gold({expected_left_map, expected_right_map});
    auto gold_sort_order = cudf::sorted_order(gold);
    auto sorted_gold     = cudf::gather(gold, *gold_sort_order);

    return std::pair(std::move(sorted_gold), std::move(sorted_result));
  }
};

TEST_F(JoinTest, EmptySentinelRepro)
{
  // This test reproduced an implementation specific behavior where the combination of these
  // particular values ended up hashing to the empty key sentinel value used by the hash join
  // This test may no longer be relevant if the implementation ever changes.
  auto const left_first_col  = cudf::test::fixed_width_column_wrapper<int32_t>{1197};
  auto const left_second_col = cudf::test::strings_column_wrapper{"201812"};
  auto const left_third_col  = cudf::test::fixed_width_column_wrapper<int64_t>{2550000371};

  auto const right_first_col  = cudf::test::fixed_width_column_wrapper<int32_t>{1197};
  auto const right_second_col = cudf::test::strings_column_wrapper{"201812"};
  auto const right_third_col  = cudf::test::fixed_width_column_wrapper<int64_t>{2550000371};

  cudf::table_view left({left_first_col, left_second_col, left_third_col});
  cudf::table_view right({right_first_col, right_second_col, right_third_col});

  auto result = inner_join(left, right, {0, 1, 2}, {0, 1, 2});

  EXPECT_EQ(result->num_rows(), 1);
}

TEST_F(JoinTest, LeftJoinNoNullsWithNoCommon)
{
  column_wrapper<int32_t> col0_0{{3, 1, 2, 0, 3}};
  strcol_wrapper col0_1({"s0", "s1", "s2", "s4", "s1"});
  column_wrapper<int32_t> col0_2{{0, 1, 2, 4, 1}};

  column_wrapper<int32_t> col1_0{{2, 2, 0, 4, 3}};
  strcol_wrapper col1_1{{"s1", "s0", "s1", "s2", "s1"}};
  column_wrapper<int32_t> col1_2{{1, 0, 1, 2, 1}};

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols0.push_back(col0_2.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());
  cols1.push_back(col1_2.release());

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  auto result            = left_join(t0, t1, {0}, {0});
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

  column_wrapper<int32_t> col_gold_0{{3, 1, 2, 2, 0, 3}, {true, true, true, true, true, true}};
  strcol_wrapper col_gold_1({"s0", "s1", "s2", "s2", "s4", "s1"},
                            {true, true, true, true, true, true});
  column_wrapper<int32_t> col_gold_2{{0, 1, 2, 2, 4, 1}, {true, true, true, true, true, true}};
  column_wrapper<int32_t> col_gold_3{{3, -1, 2, 2, 0, 3}, {true, false, true, true, true, true}};
  strcol_wrapper col_gold_4({"s1", "", "s1", "s0", "s1", "s1"},
                            {true, false, true, true, true, true});
  column_wrapper<int32_t> col_gold_5{{1, -1, 1, 0, 1, 1}, {true, false, true, true, true, true}};
  CVector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());
  cols_gold.push_back(col_gold_4.release());
  cols_gold.push_back(col_gold_5.release());
  Table gold(std::move(cols_gold));

  auto gold_sort_order = cudf::sorted_order(gold.view());
  auto sorted_gold     = cudf::gather(gold.view(), *gold_sort_order);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
}

TEST_F(JoinTest, FullJoinNoNulls)
{
  column_wrapper<int32_t> col0_0{{3, 1, 2, 0, 3}};
  strcol_wrapper col0_1({"s0", "s1", "s2", "s4", "s1"});
  column_wrapper<int32_t> col0_2{{0, 1, 2, 4, 1}};

  column_wrapper<int32_t> col1_0{{2, 2, 0, 4, 3}};
  strcol_wrapper col1_1{{"s1", "s0", "s1", "s2", "s1"}};
  column_wrapper<int32_t> col1_2{{1, 0, 1, 2, 1}};

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols0.push_back(col0_2.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());
  cols1.push_back(col1_2.release());

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  auto result            = full_join(t0, t1, {0, 1}, {0, 1});
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

  column_wrapper<int32_t> col_gold_0{{3, 1, 2, 0, 3, -1, -1, -1, -1},
                                     {true, true, true, true, true, false, false, false, false}};
  strcol_wrapper col_gold_1({"s0", "s1", "s2", "s4", "s1", "", "", "", ""},
                            {true, true, true, true, true, false, false, false, false});
  column_wrapper<int32_t> col_gold_2{{0, 1, 2, 4, 1, -1, -1, -1, -1},
                                     {true, true, true, true, true, false, false, false, false}};
  column_wrapper<int32_t> col_gold_3{{-1, -1, -1, -1, 3, 2, 2, 0, 4},
                                     {false, false, false, false, true, true, true, true, true}};
  strcol_wrapper col_gold_4({"", "", "", "", "s1", "s1", "s0", "s1", "s2"},
                            {false, false, false, false, true, true, true, true, true});
  column_wrapper<int32_t> col_gold_5{{-1, -1, -1, -1, 1, 1, 0, 1, 2},
                                     {false, false, false, false, true, true, true, true, true}};

  CVector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());
  cols_gold.push_back(col_gold_4.release());
  cols_gold.push_back(col_gold_5.release());

  Table gold(std::move(cols_gold));

  auto gold_sort_order = cudf::sorted_order(gold.view());
  auto sorted_gold     = cudf::gather(gold.view(), *gold_sort_order);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
}

TEST_F(JoinTest, FullJoinWithNulls)
{
  column_wrapper<int32_t> col0_0{{3, 1, 2, 0, 3}};
  strcol_wrapper col0_1({"s0", "s1", "s2", "s4", "s1"});
  column_wrapper<int32_t> col0_2{{0, 1, 2, 4, 1}};

  column_wrapper<int32_t> col1_0{{2, 2, 0, 4, 3}, {true, true, true, false, true}};
  strcol_wrapper col1_1{{"s1", "s0", "s1", "s2", "s1"}};
  column_wrapper<int32_t> col1_2{{1, 0, 1, 2, 1}};

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols0.push_back(col0_2.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());
  cols1.push_back(col1_2.release());

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  auto result            = full_join(t0, t1, {0, 1}, {0, 1});
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

  column_wrapper<int32_t> col_gold_0{{3, 1, 2, 0, 3, -1, -1, -1, -1},
                                     {true, true, true, true, true, false, false, false, false}};
  strcol_wrapper col_gold_1({"s0", "s1", "s2", "s4", "s1", "", "", "", ""},
                            {true, true, true, true, true, false, false, false, false});
  column_wrapper<int32_t> col_gold_2{{0, 1, 2, 4, 1, -1, -1, -1, -1},
                                     {true, true, true, true, true, false, false, false, false}};
  column_wrapper<int32_t> col_gold_3{{-1, -1, -1, -1, 3, 2, 2, 0, 4},
                                     {false, false, false, false, true, true, true, true, false}};
  strcol_wrapper col_gold_4({"", "", "", "", "s1", "s1", "s0", "s1", "s2"},
                            {false, false, false, false, true, true, true, true, true});
  column_wrapper<int32_t> col_gold_5{{-1, -1, -1, -1, 1, 1, 0, 1, 2},
                                     {false, false, false, false, true, true, true, true, true}};

  CVector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());
  cols_gold.push_back(col_gold_4.release());
  cols_gold.push_back(col_gold_5.release());

  Table gold(std::move(cols_gold));

  auto gold_sort_order = cudf::sorted_order(gold.view());
  auto sorted_gold     = cudf::gather(gold.view(), *gold_sort_order);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
}

TEST_F(JoinTest, FullJoinOnNulls)
{
  // clang-format off
  column_wrapper<int32_t> col0_0{{  3,    1 },
                                 {  true,    false  }};
  strcol_wrapper          col0_1({"s0", "s1" });
  column_wrapper<int32_t> col0_2{{  0,    1 }};

  column_wrapper<int32_t> col1_0{{  2,    5,    3,    7 },
                                 {  true,    true,    true,    false }};
  strcol_wrapper          col1_1({"s1", "s0", "s0", "s1" });
  column_wrapper<int32_t> col1_2{{  1,    4,    2,    8 }};

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols0.push_back(col0_2.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());
  cols1.push_back(col1_2.release());

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  auto result            = full_join(t0, t1, {0, 1}, {0, 1});
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

  column_wrapper<int32_t> col_gold_0{{   3,   -1,   -1,    -1},
                                     {   true,    false,    false,     false}};
  strcol_wrapper          col_gold_1{{ "s0", "s1",  "",    ""},
                                     {   true,    true,    false,     false}};
  column_wrapper<int32_t> col_gold_2{{   0,    1,   -1,    -1},
                                     {   true,    true,    false,     false}};
  column_wrapper<int32_t> col_gold_3{{   3,   -1,    2,     5},
                                     {   true,    false,    true,     true}};
  strcol_wrapper          col_gold_4{{ "s0", "s1", "s1",  "s0"}};
  column_wrapper<int32_t> col_gold_5{{   2,    8,    1,     4}};

  CVector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());
  cols_gold.push_back(col_gold_4.release());
  cols_gold.push_back(col_gold_5.release());

  Table gold(std::move(cols_gold));

  auto gold_sort_order = cudf::sorted_order(gold.view());
  auto sorted_gold     = cudf::gather(gold.view(), *gold_sort_order);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);

  // Repeat test with compare_nulls_equal=false,
  // as per SQL standard.

  result            = full_join(t0, t1, {0, 1}, {0, 1}, cudf::null_equality::UNEQUAL);
  result_sort_order = cudf::sorted_order(result->view());
  sorted_result     = cudf::gather(result->view(), *result_sort_order);

  col_gold_0 =               {{   3,   -1,   -1,    -1,   -1},
                              {   true,    false,    false,     false,    false}};
  col_gold_1 = strcol_wrapper{{ "s0", "s1",   "",    "",   ""},
                              {   true,    true,    false,     false,    false}};
  col_gold_2 =               {{   0,    1,   -1,    -1,   -1},
                              {   true,    true,    false,     false,    false}};
  col_gold_3 =               {{   3,   -1,    2,     5,   -1},
                              {   true,    false,    true,     true,    false}};
  col_gold_4 = strcol_wrapper{{ "s0",  "",  "s1",  "s0",  "s1"},
                              {   true,    false,    true,     true,    true}};
  col_gold_5 =               {{   2,   -1,    1,     4,    8},
                              {   true,    false,    true,     true,    true}};

  // clang-format on

  CVector cols_gold_nulls_unequal;
  cols_gold_nulls_unequal.push_back(col_gold_0.release());
  cols_gold_nulls_unequal.push_back(col_gold_1.release());
  cols_gold_nulls_unequal.push_back(col_gold_2.release());
  cols_gold_nulls_unequal.push_back(col_gold_3.release());
  cols_gold_nulls_unequal.push_back(col_gold_4.release());
  cols_gold_nulls_unequal.push_back(col_gold_5.release());

  Table gold_nulls_unequal{std::move(cols_gold_nulls_unequal)};

  gold_sort_order = cudf::sorted_order(gold_nulls_unequal.view());
  sorted_gold     = cudf::gather(gold_nulls_unequal.view(), *gold_sort_order);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
}

TEST_F(JoinTest, LeftJoinNoNulls)
{
  column_wrapper<int32_t> col0_0({3, 1, 2, 0, 3});
  strcol_wrapper col0_1({"s0", "s1", "s2", "s4", "s1"});
  column_wrapper<int32_t> col0_2({0, 1, 2, 4, 1});

  column_wrapper<int32_t> col1_0({2, 2, 0, 4, 3});
  strcol_wrapper col1_1({"s1", "s0", "s1", "s2", "s1"});
  column_wrapper<int32_t> col1_2({1, 0, 1, 2, 1});

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols0.push_back(col0_2.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());
  cols1.push_back(col1_2.release());

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  auto result            = left_join(t0, t1, {0, 1}, {0, 1});
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

  column_wrapper<int32_t> col_gold_0({3, 1, 2, 0, 3});
  strcol_wrapper col_gold_1({"s0", "s1", "s2", "s4", "s1"});
  column_wrapper<int32_t> col_gold_2({0, 1, 2, 4, 1});
  column_wrapper<int32_t> col_gold_3{{-1, -1, -1, -1, 3}, {false, false, false, false, true}};
  strcol_wrapper col_gold_4{{"", "", "", "", "s1"}, {false, false, false, false, true}};
  column_wrapper<int32_t> col_gold_5{{-1, -1, -1, -1, 1}, {false, false, false, false, true}};
  CVector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());
  cols_gold.push_back(col_gold_4.release());
  cols_gold.push_back(col_gold_5.release());
  Table gold(std::move(cols_gold));

  auto gold_sort_order = cudf::sorted_order(gold.view());
  auto sorted_gold     = cudf::gather(gold.view(), *gold_sort_order);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
}

TEST_F(JoinTest, LeftJoinWithNulls)
{
  column_wrapper<int32_t> col0_0{{3, 1, 2, 0, 2}};
  strcol_wrapper col0_1({"s1", "s1", "", "s4", "s0"}, {true, true, false, true, true});
  column_wrapper<int32_t> col0_2{{0, 1, 2, 4, 1}};

  column_wrapper<int32_t> col1_0{{2, 2, 0, 4, 3}};
  strcol_wrapper col1_1({"s1", "s0", "s1", "s2", "s1"});
  column_wrapper<int32_t> col1_2{{1, 0, 1, 2, 1}, {true, false, true, true, true}};

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols0.push_back(col0_2.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());
  cols1.push_back(col1_2.release());

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  auto result            = left_join(t0, t1, {0, 1}, {0, 1});
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

  column_wrapper<int32_t> col_gold_0{{3, 1, 2, 0, 2}, {true, true, true, true, true}};
  strcol_wrapper col_gold_1({"s1", "s1", "", "s4", "s0"}, {true, true, false, true, true});
  column_wrapper<int32_t> col_gold_2{{0, 1, 2, 4, 1}, {true, true, true, true, true}};
  column_wrapper<int32_t> col_gold_3{{3, -1, -1, -1, 2}, {true, false, false, false, true}};
  strcol_wrapper col_gold_4{{"s1", "", "", "", "s0"}, {true, false, false, false, true}};
  column_wrapper<int32_t> col_gold_5{{1, -1, -1, -1, -1}, {true, false, false, false, false}};

  CVector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());
  cols_gold.push_back(col_gold_4.release());
  cols_gold.push_back(col_gold_5.release());
  Table gold(std::move(cols_gold));

  auto gold_sort_order = cudf::sorted_order(gold.view());
  auto sorted_gold     = cudf::gather(gold.view(), *gold_sort_order);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
}

TEST_F(JoinTest, LeftJoinWithStructsAndNulls)
{
  column_wrapper<int32_t> col0_0{{3, 1, 2, 0, 2}};
  strcol_wrapper col0_1({"s1", "s1", "", "s4", "s0"}, {true, true, false, true, true});
  column_wrapper<int32_t> col0_2{{0, 1, 2, 4, 1}};
  auto col0_names_col = strcol_wrapper{
    "Samuel Vimes", "Carrot Ironfoundersson", "Detritus", "Samuel Vimes", "Angua von Überwald"};
  auto col0_ages_col = column_wrapper<int32_t>{{48, 27, 351, 31, 25}};

  auto col0_is_human_col =
    column_wrapper<bool>{{true, true, false, false, false}, {true, true, false, true, false}};

  auto col0_3 =
    cudf::test::structs_column_wrapper{{col0_names_col, col0_ages_col, col0_is_human_col}};

  column_wrapper<int32_t> col1_0{{2, 2, 0, 4, 3}};
  strcol_wrapper col1_1({"s1", "s0", "s1", "s2", "s1"});
  column_wrapper<int32_t> col1_2{{1, 0, 1, 2, 1}, {true, false, true, true, true}};
  auto col1_names_col = strcol_wrapper{
    "Samuel Vimes", "Detritus", "Detritus", "Carrot Ironfoundersson", "Angua von Überwald"};
  auto col1_ages_col = column_wrapper<int32_t>{{48, 35, 351, 22, 25}};

  auto col1_is_human_col =
    column_wrapper<bool>{{true, true, false, false, true}, {true, true, false, true, true}};

  auto col1_3 =
    cudf::test::structs_column_wrapper{{col1_names_col, col1_ages_col, col1_is_human_col}};

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols0.push_back(col0_2.release());
  cols0.push_back(col0_3.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());
  cols1.push_back(col1_2.release());
  cols1.push_back(col1_3.release());

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  auto result            = left_join(t0, t1, {3}, {3});
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

  column_wrapper<int32_t> col_gold_0{{3, 2, 1, 0, 2}, {true, true, true, true, true}};
  strcol_wrapper col_gold_1({"s1", "", "s1", "s4", "s0"}, {true, false, true, true, true});
  column_wrapper<int32_t> col_gold_2{{0, 2, 1, 4, 1}, {true, true, true, true, true}};
  auto col0_gold_names_col = strcol_wrapper{
    "Samuel Vimes", "Detritus", "Carrot Ironfoundersson", "Samuel Vimes", "Angua von Überwald"};
  auto col0_gold_ages_col = column_wrapper<int32_t>{{48, 351, 27, 31, 25}};

  auto col0_gold_is_human_col =
    column_wrapper<bool>{{true, false, true, false, false}, {true, false, true, true, false}};

  auto col_gold_3 = cudf::test::structs_column_wrapper{
    {col0_gold_names_col, col0_gold_ages_col, col0_gold_is_human_col}};

  column_wrapper<int32_t> col_gold_4{{2, 0, -1, -1, -1}, {true, true, false, false, false}};
  strcol_wrapper col_gold_5{{"s1", "s1", "", "", ""}, {true, true, false, false, false}};
  column_wrapper<int32_t> col_gold_6{{1, 1, -1, -1, -1}, {true, true, false, false, false}};
  auto col1_gold_names_col = strcol_wrapper{{
                                              "Samuel Vimes",
                                              "Detritus",
                                              "",
                                              "",
                                              "",
                                            },
                                            {true, true, false, false, false}};
  auto col1_gold_ages_col =
    column_wrapper<int32_t>{{48, 351, -1, -1, -1}, {true, true, false, false, false}};

  auto col1_gold_is_human_col =
    column_wrapper<bool>{{true, false, false, false, false}, {true, false, false, false, false}};

  auto col_gold_7 = cudf::test::structs_column_wrapper{
    {col1_gold_names_col, col1_gold_ages_col, col1_gold_is_human_col},
    {true, true, false, false, false}};

  CVector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());
  cols_gold.push_back(col_gold_4.release());
  cols_gold.push_back(col_gold_5.release());
  cols_gold.push_back(col_gold_6.release());
  cols_gold.push_back(col_gold_7.release());
  Table gold(std::move(cols_gold));

  auto gold_sort_order = cudf::sorted_order(gold.view());
  auto sorted_gold     = cudf::gather(gold.view(), *gold_sort_order);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
}

TEST_F(JoinTest, LeftJoinOnNulls)
{
  // clang-format off
  column_wrapper<int32_t> col0_0{{  3,    1,    2},
                                 {  true,    false,    true}};
  strcol_wrapper          col0_1({"s0", "s1", "s2" });
  column_wrapper<int32_t> col0_2{{  0,    1,    2 }};

  column_wrapper<int32_t> col1_0{{  2,    5,    3,    7 },
                                 {  true,    true,    true,    false }};
  strcol_wrapper          col1_1({"s1", "s0", "s0", "s1" });
  column_wrapper<int32_t> col1_2{{  1,    4,    2,    8 }};

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols0.push_back(col0_2.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());
  cols1.push_back(col1_2.release());

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  auto result            = left_join(t0, t1, {0, 1}, {0, 1});
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

  column_wrapper<int32_t> col_gold_0{{   3,    -1,    2},
                                     {   true,     false,    true}};
  strcol_wrapper          col_gold_1({ "s0",  "s1", "s2"},
                                     {   true,     true,    true});
  column_wrapper<int32_t> col_gold_2{{   0,     1,    2},
                                     {   true,     true,    true}};
  column_wrapper<int32_t> col_gold_3{{   3,    -1,   -1},
                                     {   true,     false,    false}};
  strcol_wrapper          col_gold_4({ "s0",  "s1",  ""},
                                     {   true,     true,    false});
  column_wrapper<int32_t> col_gold_5{{   2,     8,   -1},
                                     {   true,     true,    false}};

  CVector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());
  cols_gold.push_back(col_gold_4.release());
  cols_gold.push_back(col_gold_5.release());
  Table gold(std::move(cols_gold));

  auto gold_sort_order = cudf::sorted_order(gold.view());
  auto sorted_gold     = cudf::gather(gold.view(), *gold_sort_order);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);

  // Repeat test with compare_nulls_equal=false,
  // as per SQL standard.

  result            = left_join(t0, t1, {0, 1}, {0, 1}, cudf::null_equality::UNEQUAL);
  result_sort_order = cudf::sorted_order(result->view());
  sorted_result     = cudf::gather(result->view(), *result_sort_order);


  col_gold_0 = {{   3,    -1,    2},
                {   true,     false,    true}};
  col_gold_1 = {{ "s0",  "s1", "s2"},
                {   true,     true,    true}};
  col_gold_2 = {{   0,     1,    2},
                {   true,     true,    true}};
  col_gold_3 = {{   3,    -1,   -1},
                {   true,     false,    false}};
  col_gold_4 = {{ "s0",   "",   ""},
                {   true,     false,    false}};
  col_gold_5 = {{   2,    -1,   -1},
                {   true,     false,    false}};

  // clang-format on
  CVector cols_gold_nulls_unequal;
  cols_gold_nulls_unequal.push_back(col_gold_0.release());
  cols_gold_nulls_unequal.push_back(col_gold_1.release());
  cols_gold_nulls_unequal.push_back(col_gold_2.release());
  cols_gold_nulls_unequal.push_back(col_gold_3.release());
  Table gold_nulls_unequal{std::move(cols_gold_nulls_unequal)};

  gold_sort_order = cudf::sorted_order(gold_nulls_unequal.view());
  sorted_gold     = cudf::gather(gold_nulls_unequal.view(), *gold_sort_order);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
}

TEST_F(JoinTest, InnerJoinNoNulls)
{
  column_wrapper<int32_t> col0_0{{3, 1, 2, 0, 2}};
  strcol_wrapper col0_1({"s1", "s1", "s0", "s4", "s0"});
  column_wrapper<int32_t> col0_2{{0, 1, 2, 4, 1}};

  column_wrapper<int32_t> col1_0{{2, 2, 0, 4, 3}};
  strcol_wrapper col1_1({"s1", "s0", "s1", "s2", "s1"});
  column_wrapper<int32_t> col1_2{{1, 0, 1, 2, 1}};

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols0.push_back(col0_2.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());
  cols1.push_back(col1_2.release());

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  auto result            = inner_join(t0, t1, {0, 1}, {0, 1});
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

  column_wrapper<int32_t> col_gold_0{{3, 2, 2}};
  strcol_wrapper col_gold_1({"s1", "s0", "s0"});
  column_wrapper<int32_t> col_gold_2{{0, 2, 1}};
  column_wrapper<int32_t> col_gold_3{{3, 2, 2}};
  strcol_wrapper col_gold_4({"s1", "s0", "s0"});
  column_wrapper<int32_t> col_gold_5{{1, 0, 0}};
  CVector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());
  cols_gold.push_back(col_gold_4.release());
  cols_gold.push_back(col_gold_5.release());
  Table gold(std::move(cols_gold));

  auto gold_sort_order = cudf::sorted_order(gold.view());
  auto sorted_gold     = cudf::gather(gold.view(), *gold_sort_order);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
}

TEST_F(JoinTest, InnerJoinWithNulls)
{
  column_wrapper<int32_t> col0_0{{3, 1, 2, 0, 2}};
  strcol_wrapper col0_1({"s1", "s1", "s0", "s4", "s0"}, {true, true, false, true, true});
  column_wrapper<int32_t> col0_2{{0, 1, 2, 4, 1}};

  column_wrapper<int32_t> col1_0{{2, 2, 0, 4, 3}};
  strcol_wrapper col1_1({"s1", "s0", "s1", "s2", "s1"});
  column_wrapper<int32_t> col1_2{{1, 0, 1, 2, 1}, {true, false, true, true, true}};

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols0.push_back(col0_2.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());
  cols1.push_back(col1_2.release());

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  auto result            = inner_join(t0, t1, {0, 1}, {0, 1});
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

  column_wrapper<int32_t> col_gold_0{{3, 2}};
  strcol_wrapper col_gold_1({"s1", "s0"}, {true, true});
  column_wrapper<int32_t> col_gold_2{{0, 1}};
  column_wrapper<int32_t> col_gold_3{{3, 2}};
  strcol_wrapper col_gold_4({"s1", "s0"}, {true, true});
  column_wrapper<int32_t> col_gold_5{{1, -1}, {true, false}};
  CVector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());
  cols_gold.push_back(col_gold_4.release());
  cols_gold.push_back(col_gold_5.release());
  Table gold(std::move(cols_gold));

  auto gold_sort_order = cudf::sorted_order(gold.view());
  auto sorted_gold     = cudf::gather(gold.view(), *gold_sort_order);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
}

TEST_F(JoinTest, InnerJoinWithStructsAndNulls)
{
  column_wrapper<int32_t> col0_0{{3, 1, 2, 0, 2}};
  strcol_wrapper col0_1({"s1", "s1", "s0", "s4", "s0"}, {true, true, false, true, true});
  column_wrapper<int32_t> col0_2{{0, 1, 2, 4, 1}};
  std::initializer_list<std::string> col0_names = {
    "Samuel Vimes", "Carrot Ironfoundersson", "Detritus", "Samuel Vimes", "Angua von Überwald"};
  auto col0_names_col = strcol_wrapper{col0_names.begin(), col0_names.end()};
  auto col0_ages_col  = column_wrapper<int32_t>{{48, 27, 351, 31, 25}};

  auto col0_is_human_col =
    column_wrapper<bool>{{true, true, false, false, false}, {true, true, false, true, false}};

  auto col0_3 =
    cudf::test::structs_column_wrapper{{col0_names_col, col0_ages_col, col0_is_human_col}};

  column_wrapper<int32_t> col1_0{{2, 2, 0, 4, 3}};
  strcol_wrapper col1_1({"s1", "s0", "s1", "s2", "s1"});
  column_wrapper<int32_t> col1_2{{1, 0, 1, 2, 1}, {true, false, true, true, true}};
  std::initializer_list<std::string> col1_names = {"Carrot Ironfoundersson",
                                                   "Angua von Überwald",
                                                   "Detritus",
                                                   "Carrot Ironfoundersson",
                                                   "Samuel Vimes"};
  auto col1_names_col = strcol_wrapper{col1_names.begin(), col1_names.end()};
  auto col1_ages_col  = column_wrapper<int32_t>{{351, 25, 27, 31, 48}};

  auto col1_is_human_col =
    column_wrapper<bool>{{true, false, false, false, true}, {true, false, false, true, true}};

  auto col1_3 =
    cudf::test::structs_column_wrapper{{col1_names_col, col1_ages_col, col1_is_human_col}};

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols0.push_back(col0_2.release());
  cols0.push_back(col0_3.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());
  cols1.push_back(col1_2.release());
  cols1.push_back(col1_3.release());

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  auto result            = inner_join(t0, t1, {0, 1, 3}, {0, 1, 3});
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

  column_wrapper<int32_t> col_gold_0{{3, 2}};
  strcol_wrapper col_gold_1({"s1", "s0"}, {true, true});
  column_wrapper<int32_t> col_gold_2{{0, 1}};
  auto col_gold_3_names_col = strcol_wrapper{"Samuel Vimes", "Angua von Überwald"};
  auto col_gold_3_ages_col  = column_wrapper<int32_t>{{48, 25}};

  auto col_gold_3_is_human_col = column_wrapper<bool>{{true, false}, {true, false}};

  auto col_gold_3 = cudf::test::structs_column_wrapper{
    {col_gold_3_names_col, col_gold_3_ages_col, col_gold_3_is_human_col}};

  column_wrapper<int32_t> col_gold_4{{3, 2}};
  strcol_wrapper col_gold_5({"s1", "s0"}, {true, true});
  column_wrapper<int32_t> col_gold_6{{1, -1}, {true, false}};
  auto col_gold_7_names_col = strcol_wrapper{"Samuel Vimes", "Angua von Überwald"};
  auto col_gold_7_ages_col  = column_wrapper<int32_t>{{48, 25}};

  auto col_gold_7_is_human_col = column_wrapper<bool>{{true, false}, {true, false}};

  auto col_gold_7 = cudf::test::structs_column_wrapper{
    {col_gold_7_names_col, col_gold_7_ages_col, col_gold_7_is_human_col}};
  CVector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());
  cols_gold.push_back(col_gold_4.release());
  cols_gold.push_back(col_gold_5.release());
  cols_gold.push_back(col_gold_6.release());
  cols_gold.push_back(col_gold_7.release());
  Table gold(std::move(cols_gold));

  auto gold_sort_order = cudf::sorted_order(gold.view());
  auto sorted_gold     = cudf::gather(gold.view(), *gold_sort_order);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
}

// // Test to check join behavior when join keys are null.
TEST_F(JoinTest, InnerJoinOnNulls)
{
  // clang-format off
  column_wrapper<int32_t> col0_0{{  3,    1,    2,    0,    2}};
  strcol_wrapper          col0_1({"s1", "s1", "s8", "s4", "s0"},
                                 {  true,    true,    false,    true,    true});
  column_wrapper<int32_t> col0_2{{  0,    1,    2,    4,    1}};

  column_wrapper<int32_t> col1_0{{  2,    2,    0,    4,    3}};
  strcol_wrapper          col1_1({"s1", "s0", "s1", "s2", "s1"},
                                 {  true,    false,    true,    true,    true});
  column_wrapper<int32_t> col1_2{{  1,    0,    1,    2,    1}};

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols0.push_back(col0_2.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());
  cols1.push_back(col1_2.release());

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  auto result            = inner_join(t0, t1, {0, 1}, {0, 1});
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

  column_wrapper<int32_t> col_gold_0 {{  3,    2}};
  strcol_wrapper          col_gold_1 ({"s1", "s0"},
                                      {  true,    false});
  column_wrapper<int32_t> col_gold_2{{   0,    2}};
  column_wrapper<int32_t> col_gold_3 {{  3,    2}};
  strcol_wrapper          col_gold_4 ({"s1", "s0"},
                                      {  true,    false});
  column_wrapper<int32_t> col_gold_5{{   1,    0}};
  CVector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());
  cols_gold.push_back(col_gold_4.release());
  cols_gold.push_back(col_gold_5.release());

  Table gold(std::move(cols_gold));

  auto gold_sort_order = cudf::sorted_order(gold.view());
  auto sorted_gold     = cudf::gather(gold.view(), *gold_sort_order);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);

  // Repeat test with compare_nulls_equal=false,
  // as per SQL standard.

  result            = inner_join(t0, t1, {0, 1}, {0, 1},  cudf::null_equality::UNEQUAL);
  result_sort_order = cudf::sorted_order(result->view());
  sorted_result     = cudf::gather(result->view(), *result_sort_order);

  col_gold_0 =               {{  3}};
  col_gold_1 = strcol_wrapper({"s1"},
                              {  true});
  col_gold_2 =               {{  0}};
  col_gold_3 =               {{  3}};
  col_gold_4 = strcol_wrapper({"s1"},
                              {  true});
  col_gold_5 =               {{  1}};

  // clang-format on

  CVector cols_gold_sql;
  cols_gold_sql.push_back(col_gold_0.release());
  cols_gold_sql.push_back(col_gold_1.release());
  cols_gold_sql.push_back(col_gold_2.release());
  cols_gold_sql.push_back(col_gold_3.release());
  cols_gold_sql.push_back(col_gold_4.release());
  cols_gold_sql.push_back(col_gold_5.release());
  Table gold_sql(std::move(cols_gold_sql));

  gold_sort_order = cudf::sorted_order(gold_sql.view());
  sorted_gold     = cudf::gather(gold_sql.view(), *gold_sort_order);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
}

// Empty Left Table
TEST_F(JoinTest, EmptyLeftTableInnerJoin)
{
  column_wrapper<int32_t> col0_0;
  column_wrapper<int32_t> col0_1;

  column_wrapper<int32_t> col1_0{{2, 2, 0, 4, 3}};
  column_wrapper<int32_t> col1_1{{1, 0, 1, 2, 1}, {true, false, true, true, true}};

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());

  Table empty0(std::move(cols0));
  Table t1(std::move(cols1));

  auto result = inner_join(empty0, t1, {0, 1}, {0, 1});
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(empty0, *result);
}

TEST_F(JoinTest, EmptyLeftTableLeftJoin)
{
  column_wrapper<int32_t> col0_0;
  column_wrapper<int32_t> col0_1;

  column_wrapper<int32_t> col1_0{{2, 2, 0, 4, 3}};
  column_wrapper<int32_t> col1_1{{1, 0, 1, 2, 1}, {true, false, true, true, true}};

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());

  Table empty0(std::move(cols0));
  Table t1(std::move(cols1));

  auto result = left_join(empty0, t1, {0, 1}, {0, 1});
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(empty0, *result);
}

TEST_F(JoinTest, EmptyLeftTableFullJoin)
{
  column_wrapper<int32_t> col0_0;
  column_wrapper<int32_t> col0_1;

  column_wrapper<int32_t> col1_0{{2, 2, 0, 4, 3}};
  column_wrapper<int32_t> col1_1{{1, 0, 1, 2, 1}, {true, false, true, true, true}};

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());

  Table lhs(std::move(cols0));
  Table rhs(std::move(cols1));

  auto result            = full_join(lhs, rhs, {0, 1}, {0, 1});
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

  column_wrapper<int32_t> col_gold_0{{-1, -1, -1, -1, -1}, {false, false, false, false, false}};
  column_wrapper<int32_t> col_gold_1{{-1, -1, -1, -1, -1}, {false, false, false, false, false}};
  column_wrapper<int32_t> col_gold_2{{2, 2, 0, 4, 3}};
  column_wrapper<int32_t> col_gold_3{{1, 0, 1, 2, 1}, {true, false, true, true, true}};

  CVector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());
  Table gold(std::move(cols_gold));

  auto gold_sort_order = cudf::sorted_order(gold.view());
  auto sorted_gold     = cudf::gather(gold.view(), *gold_sort_order);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
}

// Empty Right Table
TEST_F(JoinTest, EmptyRightTableInnerJoin)
{
  column_wrapper<int32_t> col0_0{{2, 2, 0, 4, 3}};
  column_wrapper<int32_t> col0_1{{1, 0, 1, 2, 1}, {true, false, true, true, true}};

  column_wrapper<int32_t> col1_0;
  column_wrapper<int32_t> col1_1;

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());

  Table t0(std::move(cols0));
  Table empty1(std::move(cols1));

  {
    auto result = inner_join(t0, empty1, {0, 1}, {0, 1});
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(empty1, *result);
  }

  {
    cudf::hash_join hash_join(empty1, cudf::null_equality::EQUAL);

    auto output_size                         = hash_join.inner_join_size(t0);
    std::optional<std::size_t> optional_size = output_size;

    std::size_t const size_gold = 0;
    EXPECT_EQ(output_size, size_gold);

    auto result = hash_join.inner_join(t0, optional_size);
    column_wrapper<int32_t> col_gold_0{};
    column_wrapper<int32_t> col_gold_1{};
    auto const [sorted_gold, sorted_result] = gather_maps_as_tables(col_gold_0, col_gold_1, result);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
  }
}

TEST_F(JoinTest, EmptyRightTableLeftJoin)
{
  column_wrapper<int32_t> col0_0{{2, 2, 0, 4, 3}, {true, true, true, true, true}};
  column_wrapper<int32_t> col0_1{{1, 0, 1, 2, 1}, {true, false, true, true, true}};

  column_wrapper<int32_t> col1_0;
  column_wrapper<int32_t> col1_1;

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());

  Table t0(std::move(cols0));
  Table empty1(std::move(cols1));

  {
    auto result = left_join(t0, empty1, {0, 1}, {0, 1});
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(t0, *result);
  }

  {
    cudf::hash_join hash_join(empty1, cudf::null_equality::EQUAL);

    auto output_size                         = hash_join.left_join_size(t0);
    std::optional<std::size_t> optional_size = output_size;

    std::size_t const size_gold = 5;
    EXPECT_EQ(output_size, size_gold);

    auto result = hash_join.left_join(t0, optional_size);
    column_wrapper<int32_t> col_gold_0{{0, 1, 2, 3, 4}};
    column_wrapper<int32_t> col_gold_1{{NoneValue, NoneValue, NoneValue, NoneValue, NoneValue}};
    auto const [sorted_gold, sorted_result] = gather_maps_as_tables(col_gold_0, col_gold_1, result);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
  }
}

TEST_F(JoinTest, EmptyRightTableFullJoin)
{
  column_wrapper<int32_t> col0_0{{2, 2, 0, 4, 3}};
  column_wrapper<int32_t> col0_1{{1, 0, 1, 2, 1}, {true, false, true, true, true}};

  column_wrapper<int32_t> col1_0;
  column_wrapper<int32_t> col1_1;

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());

  Table t0(std::move(cols0));
  Table empty1(std::move(cols1));

  {
    auto result = full_join(t0, empty1, {0, 1}, {0, 1});
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(t0, *result);
  }

  {
    cudf::hash_join hash_join(empty1, cudf::null_equality::EQUAL);

    auto output_size                         = hash_join.full_join_size(t0);
    std::optional<std::size_t> optional_size = output_size;

    std::size_t const size_gold = 5;
    EXPECT_EQ(output_size, size_gold);

    auto result = hash_join.full_join(t0, optional_size);
    column_wrapper<int32_t> col_gold_0{{0, 1, 2, 3, 4}};
    column_wrapper<int32_t> col_gold_1{{NoneValue, NoneValue, NoneValue, NoneValue, NoneValue}};
    auto const [sorted_gold, sorted_result] = gather_maps_as_tables(col_gold_0, col_gold_1, result);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
  }
}

// Both tables empty
TEST_F(JoinTest, BothEmptyInnerJoin)
{
  column_wrapper<int32_t> col0_0;
  column_wrapper<int32_t> col0_1;

  column_wrapper<int32_t> col1_0;
  column_wrapper<int32_t> col1_1;

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());

  Table t0(std::move(cols0));
  Table empty1(std::move(cols1));

  auto result = inner_join(t0, empty1, {0, 1}, {0, 1});
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(empty1, *result);
}

TEST_F(JoinTest, BothEmptyLeftJoin)
{
  column_wrapper<int32_t> col0_0;
  column_wrapper<int32_t> col0_1;

  column_wrapper<int32_t> col1_0;
  column_wrapper<int32_t> col1_1;

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());

  Table t0(std::move(cols0));
  Table empty1(std::move(cols1));

  auto result = left_join(t0, empty1, {0, 1}, {0, 1});
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(empty1, *result);
}

TEST_F(JoinTest, BothEmptyFullJoin)
{
  column_wrapper<int32_t> col0_0;
  column_wrapper<int32_t> col0_1;

  column_wrapper<int32_t> col1_0;
  column_wrapper<int32_t> col1_1;

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());

  Table t0(std::move(cols0));
  Table empty1(std::move(cols1));

  auto result = full_join(t0, empty1, {0, 1}, {0, 1});
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(empty1, *result);
}

// // EqualValues X Inner,Left,Full

TEST_F(JoinTest, EqualValuesInnerJoin)
{
  column_wrapper<int32_t> col0_0{{0, 0}};
  strcol_wrapper col0_1({"s0", "s0"});

  column_wrapper<int32_t> col1_0{{0, 0}};
  strcol_wrapper col1_1({"s0", "s0"});

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  auto result = inner_join(t0, t1, {0, 1}, {0, 1});

  column_wrapper<int32_t> col_gold_0{{0, 0, 0, 0}};
  strcol_wrapper col_gold_1({"s0", "s0", "s0", "s0"});
  column_wrapper<int32_t> col_gold_2{{0, 0, 0, 0}};
  strcol_wrapper col_gold_3({"s0", "s0", "s0", "s0"});

  CVector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());

  Table gold(std::move(cols_gold));

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(gold, *result);
}

TEST_F(JoinTest, EqualValuesLeftJoin)
{
  column_wrapper<int32_t> col0_0{{0, 0}};
  strcol_wrapper col0_1({"s0", "s0"});

  column_wrapper<int32_t> col1_0{{0, 0}};
  strcol_wrapper col1_1({"s0", "s0"});

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  auto result = left_join(t0, t1, {0, 1}, {0, 1});

  column_wrapper<int32_t> col_gold_0{{0, 0, 0, 0}, {true, true, true, true}};
  strcol_wrapper col_gold_1({"s0", "s0", "s0", "s0"}, {true, true, true, true});
  column_wrapper<int32_t> col_gold_2{{0, 0, 0, 0}, {true, true, true, true}};
  strcol_wrapper col_gold_3({"s0", "s0", "s0", "s0"}, {true, true, true, true});

  CVector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());
  Table gold(std::move(cols_gold));

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(gold, *result);
}

TEST_F(JoinTest, EqualValuesFullJoin)
{
  column_wrapper<int32_t> col0_0{{0, 0}};
  strcol_wrapper col0_1({"s0", "s0"});

  column_wrapper<int32_t> col1_0{{0, 0}};
  strcol_wrapper col1_1({"s0", "s0"});

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  auto result = full_join(t0, t1, {0, 1}, {0, 1});

  column_wrapper<int32_t> col_gold_0{{0, 0, 0, 0}};
  strcol_wrapper col_gold_1({"s0", "s0", "s0", "s0"});
  column_wrapper<int32_t> col_gold_2{{0, 0, 0, 0}};
  strcol_wrapper col_gold_3({"s0", "s0", "s0", "s0"});

  CVector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());
  Table gold(std::move(cols_gold));

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(gold, *result);
}

TEST_F(JoinTest, InnerJoinCornerCase)
{
  column_wrapper<int64_t> col0_0{{4, 1, 3, 2, 2, 2, 2}};
  column_wrapper<int64_t> col1_0{{2}};

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols1.push_back(col1_0.release());

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  auto result            = inner_join(t0, t1, {0}, {0});
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

  column_wrapper<int64_t> col_gold_0{{2, 2, 2, 2}};
  column_wrapper<int64_t> col_gold_1{{2, 2, 2, 2}};
  CVector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  Table gold(std::move(cols_gold));

  auto gold_sort_order = cudf::sorted_order(gold.view());
  auto sorted_gold     = cudf::gather(gold.view(), *gold_sort_order);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
}

TEST_F(JoinTest, HashJoinSequentialProbes)
{
  CVector cols1;
  cols1.emplace_back(column_wrapper<int32_t>{{2, 2, 0, 4, 3}}.release());
  cols1.emplace_back(strcol_wrapper{{"s1", "s0", "s1", "s2", "s1"}}.release());

  Table t1(std::move(cols1));

  cudf::hash_join hash_join(t1, cudf::nullable_join::NO, cudf::null_equality::EQUAL);

  {
    CVector cols0;
    cols0.emplace_back(column_wrapper<int32_t>{{3, 1, 2, 0, 3}}.release());
    cols0.emplace_back(strcol_wrapper({"s0", "s1", "s2", "s4", "s1"}).release());

    Table t0(std::move(cols0));

    auto output_size                         = hash_join.full_join_size(t0);
    std::optional<std::size_t> optional_size = output_size;

    std::size_t const size_gold = 9;
    EXPECT_EQ(output_size, size_gold);

    auto result = hash_join.full_join(t0, optional_size);
    column_wrapper<int32_t> col_gold_0{{NoneValue, NoneValue, NoneValue, NoneValue, 4, 0, 1, 2, 3}};
    column_wrapper<int32_t> col_gold_1{{0, 1, 2, 3, 4, NoneValue, NoneValue, NoneValue, NoneValue}};
    auto const [sorted_gold, sorted_result] = gather_maps_as_tables(col_gold_0, col_gold_1, result);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
  }

  {
    CVector cols0;
    cols0.emplace_back(column_wrapper<int32_t>{{3, 1, 2, 0, 3}}.release());
    cols0.emplace_back(strcol_wrapper({"s0", "s1", "s2", "s4", "s1"}).release());

    Table t0(std::move(cols0));

    auto output_size                         = hash_join.left_join_size(t0);
    std::optional<std::size_t> optional_size = output_size;

    std::size_t const size_gold = 5;
    EXPECT_EQ(output_size, size_gold);

    auto result = hash_join.left_join(t0, optional_size);
    column_wrapper<int32_t> col_gold_0{{0, 1, 2, 3, 4}};
    column_wrapper<int32_t> col_gold_1{{NoneValue, NoneValue, NoneValue, NoneValue, 4}};
    auto const [sorted_gold, sorted_result] = gather_maps_as_tables(col_gold_0, col_gold_1, result);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
  }

  {
    CVector cols0;
    cols0.emplace_back(column_wrapper<int32_t>{{3, 1, 2, 0, 2}}.release());
    cols0.emplace_back(strcol_wrapper({"s1", "s1", "s0", "s4", "s0"}).release());

    Table t0(std::move(cols0));

    auto output_size                         = hash_join.inner_join_size(t0);
    std::optional<std::size_t> optional_size = output_size;

    std::size_t const size_gold = 3;
    EXPECT_EQ(output_size, size_gold);

    auto result = hash_join.inner_join(t0, optional_size);
    column_wrapper<int32_t> col_gold_0{{2, 4, 0}};
    column_wrapper<int32_t> col_gold_1{{1, 1, 4}};
    auto const [sorted_gold, sorted_result] = gather_maps_as_tables(col_gold_0, col_gold_1, result);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
  }
}

TEST_F(JoinTest, HashJoinWithStructsAndNulls)
{
  auto col0_names_col = strcol_wrapper{
    "Samuel Vimes", "Carrot Ironfoundersson", "Detritus", "Samuel Vimes", "Angua von Überwald"};
  auto col0_ages_col = column_wrapper<int32_t>{{48, 27, 351, 31, 25}};

  auto col0_is_human_col =
    column_wrapper<bool>{{true, true, false, false, false}, {true, true, false, true, false}};

  auto col0 =
    cudf::test::structs_column_wrapper{{col0_names_col, col0_ages_col, col0_is_human_col}};

  auto col1_names_col = strcol_wrapper{
    "Samuel Vimes", "Detritus", "Detritus", "Carrot Ironfoundersson", "Angua von Überwald"};
  auto col1_ages_col = column_wrapper<int32_t>{{48, 35, 351, 22, 25}};

  auto col1_is_human_col =
    column_wrapper<bool>{{true, true, false, false, true}, {true, true, false, true, true}};

  auto col1 =
    cudf::test::structs_column_wrapper{{col1_names_col, col1_ages_col, col1_is_human_col}};

  CVector cols0, cols1;
  cols0.push_back(col0.release());
  cols1.push_back(col1.release());

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));
  auto const has_nulls = cudf::has_nested_nulls(t0) || cudf::has_nested_nulls(t1)
                           ? cudf::nullable_join::YES
                           : cudf::nullable_join::NO;

  auto hash_join = cudf::hash_join(t1, has_nulls, cudf::null_equality::EQUAL);

  {
    auto output_size = hash_join.left_join_size(t0);
    EXPECT_EQ(5, output_size);
    auto result = hash_join.left_join(t0, output_size);
    column_wrapper<int32_t> col_gold_0{{0, 1, 2, 3, 4}};
    column_wrapper<int32_t> col_gold_1{{0, NoneValue, 2, NoneValue, NoneValue}};
    auto const [sorted_gold, sorted_result] = gather_maps_as_tables(col_gold_0, col_gold_1, result);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
  }

  {
    auto output_size = hash_join.inner_join_size(t0);
    EXPECT_EQ(2, output_size);
    auto result = hash_join.inner_join(t0, output_size);
    column_wrapper<int32_t> col_gold_0{{0, 2}};
    column_wrapper<int32_t> col_gold_1{{0, 2}};
    auto const [sorted_gold, sorted_result] = gather_maps_as_tables(col_gold_0, col_gold_1, result);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
  }

  {
    auto output_size = hash_join.full_join_size(t0);
    EXPECT_EQ(8, output_size);
    auto result = hash_join.full_join(t0, output_size);
    column_wrapper<int32_t> col_gold_0{{NoneValue, NoneValue, NoneValue, 0, 1, 2, 3, 4}};
    column_wrapper<int32_t> col_gold_1{{1, 3, 4, 0, NoneValue, 2, NoneValue, NoneValue}};
    auto const [sorted_gold, sorted_result] = gather_maps_as_tables(col_gold_0, col_gold_1, result);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
  }
}

TEST_F(JoinTest, HashJoinWithNullsOneSide)
{
  auto const t0 = [] {
    column_wrapper<int32_t> col0{2, 2, 0, 4, 3};
    column_wrapper<int32_t> col1{1, 10, 1, 2, 1};
    CVector cols;
    cols.emplace_back(col0.release());
    cols.emplace_back(col1.release());
    return Table{std::move(cols)};
  }();

  auto const t1 = [] {
    column_wrapper<int32_t> col0{1, 2, 3, 4, 5, 2, 2, 0, 4, 3, 1, 2, 3, 4, 5};
    column_wrapper<int32_t> col1{{1, 2, 3, 4, 5, 1, 0, 1, 2, 1, 1, 2, 3, 4, 5},
                                 cudf::test::iterators::null_at(6)};
    CVector cols;
    cols.emplace_back(col0.release());
    cols.emplace_back(col1.release());
    return Table{std::move(cols)};
  }();

  auto const hash_join   = cudf::hash_join(t0, cudf::null_equality::EQUAL);
  auto constexpr invalid = std::numeric_limits<int32_t>::min();  // invalid index sentinel

  auto const sort_result = [](auto const& result) {
    auto const left_cv  = cudf::column_view{cudf::data_type{cudf::type_id::INT32},
                                           static_cast<cudf::size_type>(result.first->size()),
                                           result.first->data(),
                                           nullptr,
                                           0};
    auto const right_cv = cudf::column_view{cudf::data_type{cudf::type_id::INT32},
                                            static_cast<cudf::size_type>(result.second->size()),
                                            result.second->data(),
                                            nullptr,
                                            0};
    auto sorted_left    = cudf::sort(cudf::table_view{{left_cv}});
    auto sorted_right   = cudf::sort(cudf::table_view{{right_cv}});
    return std::pair{std::move(sorted_left), std::move(sorted_right)};
  };

  {
    auto const output_size = hash_join.left_join_size(t1);
    auto const result      = hash_join.left_join(t1, std::optional<std::size_t>{output_size});
    auto const [sorted_left_indices, sorted_right_indices] = sort_result(result);

    auto const expected_left_indices =
      column_wrapper<int32_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
    auto const expected_right_indices = column_wrapper<int32_t>{invalid,
                                                                invalid,
                                                                invalid,
                                                                invalid,
                                                                invalid,
                                                                invalid,
                                                                invalid,
                                                                invalid,
                                                                invalid,
                                                                invalid,
                                                                invalid,
                                                                0,
                                                                2,
                                                                3,
                                                                4};

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_left_indices, sorted_left_indices->get_column(0));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_right_indices, sorted_right_indices->get_column(0));
  }

  {
    auto const output_size = hash_join.inner_join_size(t1);
    auto const result      = hash_join.inner_join(t1, std::optional<std::size_t>{output_size});
    auto const [sorted_left_indices, sorted_right_indices] = sort_result(result);

    auto const expected_left_indices  = column_wrapper<int32_t>{5, 7, 8, 9};
    auto const expected_right_indices = column_wrapper<int32_t>{0, 2, 3, 4};

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_left_indices, sorted_left_indices->get_column(0));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_right_indices, sorted_right_indices->get_column(0));
  }

  {
    auto const output_size = hash_join.full_join_size(t1);
    auto const result      = hash_join.full_join(t1, std::optional<std::size_t>{output_size});
    auto const [sorted_left_indices, sorted_right_indices] = sort_result(result);

    auto const expected_left_indices =
      column_wrapper<int32_t>{invalid, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
    auto const expected_right_indices = column_wrapper<int32_t>{invalid,
                                                                invalid,
                                                                invalid,
                                                                invalid,
                                                                invalid,
                                                                invalid,
                                                                invalid,
                                                                invalid,
                                                                invalid,
                                                                invalid,
                                                                invalid,
                                                                0,
                                                                1,
                                                                2,
                                                                3,
                                                                4};

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_left_indices, sorted_left_indices->get_column(0));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_right_indices, sorted_right_indices->get_column(0));
  }
}

TEST_F(JoinTest, HashJoinLargeOutputSize)
{
  // self-join a table of zeroes to generate an output row count that would overflow int32_t
  std::size_t col_size = 65567;
  rmm::device_buffer zeroes(col_size * sizeof(int32_t), cudf::get_default_stream());
  CUDF_CUDA_TRY(
    cudaMemsetAsync(zeroes.data(), 0, zeroes.size(), cudf::get_default_stream().value()));
  cudf::column_view col_zeros(
    cudf::data_type{cudf::type_id::INT32}, col_size, zeroes.data(), nullptr, 0);
  cudf::table_view tview{{col_zeros}};
  cudf::hash_join hash_join(tview, cudf::nullable_join::NO, cudf::null_equality::UNEQUAL);
  std::size_t output_size = hash_join.inner_join_size(tview);
  EXPECT_EQ(col_size * col_size, output_size);
}

struct JoinDictionaryTest : public cudf::test::BaseFixture {};

TEST_F(JoinDictionaryTest, LeftJoinNoNulls)
{
  column_wrapper<int32_t> col0_0{{3, 1, 2, 0, 3}};
  strcol_wrapper col0_1_w({"s0", "s1", "s2", "s4", "s1"});
  auto col0_1 = cudf::dictionary::encode(col0_1_w);
  column_wrapper<int32_t> col0_2{{0, 1, 2, 4, 1}};

  column_wrapper<int32_t> col1_0{{2, 2, 0, 4, 3}};
  strcol_wrapper col1_1_w{{"s1", "s0", "s1", "s2", "s1"}};
  auto col1_1 = cudf::dictionary::encode(col1_1_w);
  column_wrapper<int32_t> col1_2{{1, 0, 1, 2, 1}};

  auto t0 = cudf::table_view({col0_0, col0_1->view(), col0_2});
  auto t1 = cudf::table_view({col1_0, col1_1->view(), col1_2});
  auto g0 = cudf::table_view({col0_0, col0_1_w, col0_2});
  auto g1 = cudf::table_view({col1_0, col1_1_w, col1_2});

  auto result      = left_join(t0, t1, {0}, {0});
  auto result_view = result->view();
  auto decoded1    = cudf::dictionary::decode(result_view.column(1));
  auto decoded4    = cudf::dictionary::decode(result_view.column(4));
  std::vector<cudf::column_view> result_decoded({result_view.column(0),
                                                 decoded1->view(),
                                                 result_view.column(2),
                                                 result_view.column(3),
                                                 decoded4->view(),
                                                 result_view.column(5)});
  auto result_sort_order = cudf::sorted_order(cudf::table_view(result_decoded));
  auto sorted_result     = cudf::gather(cudf::table_view(result_decoded), *result_sort_order);

  auto gold            = left_join(g0, g1, {0}, {0});
  auto gold_sort_order = cudf::sorted_order(gold->view());
  auto sorted_gold     = cudf::gather(gold->view(), *gold_sort_order);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
}

TEST_F(JoinDictionaryTest, LeftJoinWithNulls)
{
  column_wrapper<int32_t> col0_0{{3, 1, 2, 0, 2}};
  strcol_wrapper col0_1({"s1", "s1", "s0", "s4", "s0"}, {true, true, false, true, true});
  column_wrapper<int32_t> col0_2_w{{0, 1, 2, 4, 1}};
  auto col0_2 = cudf::dictionary::encode(col0_2_w);

  column_wrapper<int32_t> col1_0{{2, 2, 0, 4, 3}};
  strcol_wrapper col1_1({"s1", "s0", "s1", "s2", "s1"});
  column_wrapper<int32_t> col1_2_w{{1, 0, 1, 2, 1}, {true, false, true, true, true}};
  auto col1_2 = cudf::dictionary::encode(col1_2_w);

  auto t0 = cudf::table_view({col0_0, col0_1, col0_2->view()});
  auto t1 = cudf::table_view({col1_0, col1_1, col1_2->view()});

  auto result      = left_join(t0, t1, {0, 1}, {0, 1});
  auto result_view = result->view();
  auto decoded2    = cudf::dictionary::decode(result_view.column(2));
  auto decoded5    = cudf::dictionary::decode(result_view.column(5));
  std::vector<cudf::column_view> result_decoded({result_view.column(0),
                                                 result_view.column(1),
                                                 decoded2->view(),
                                                 result_view.column(3),
                                                 result_view.column(4),
                                                 decoded5->view()});
  auto result_sort_order = cudf::sorted_order(cudf::table_view(result_decoded));
  auto sorted_result     = cudf::gather(cudf::table_view(result_decoded), *result_sort_order);

  auto g0              = cudf::table_view({col0_0, col0_1, col0_2_w});
  auto g1              = cudf::table_view({col1_0, col1_1, col1_2_w});
  auto gold            = left_join(g0, g1, {0, 1}, {0, 1});
  auto gold_sort_order = cudf::sorted_order(gold->view());
  auto sorted_gold     = cudf::gather(gold->view(), *gold_sort_order);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
}

TEST_F(JoinDictionaryTest, InnerJoinNoNulls)
{
  column_wrapper<int32_t> col0_0{{3, 1, 2, 0, 2}};
  strcol_wrapper col0_1_w({"s1", "s1", "s0", "s4", "s0"});
  auto col0_1 = cudf::dictionary::encode(col0_1_w);
  column_wrapper<int32_t> col0_2{{0, 1, 2, 4, 1}};

  column_wrapper<int32_t> col1_0{{2, 2, 0, 4, 3}};
  strcol_wrapper col1_1_w({"s1", "s0", "s1", "s2", "s1"});
  auto col1_1 = cudf::dictionary::encode(col1_1_w);
  column_wrapper<int32_t> col1_2{{1, 0, 1, 2, 1}};

  auto t0 = cudf::table_view({col0_0, col0_1->view(), col0_2});
  auto t1 = cudf::table_view({col1_0, col1_1->view(), col1_2});

  auto result      = inner_join(t0, t1, {0, 1}, {0, 1});
  auto result_view = result->view();
  auto decoded1    = cudf::dictionary::decode(result_view.column(1));
  auto decoded4    = cudf::dictionary::decode(result_view.column(4));
  std::vector<cudf::column_view> result_decoded({result_view.column(0),
                                                 decoded1->view(),
                                                 result_view.column(2),
                                                 result_view.column(3),
                                                 decoded4->view(),
                                                 result_view.column(5)});
  auto result_sort_order = cudf::sorted_order(cudf::table_view(result_decoded));
  auto sorted_result     = cudf::gather(cudf::table_view(result_decoded), *result_sort_order);

  auto g0              = cudf::table_view({col0_0, col0_1_w, col0_2});
  auto g1              = cudf::table_view({col1_0, col1_1_w, col1_2});
  auto gold            = inner_join(g0, g1, {0, 1}, {0, 1});
  auto gold_sort_order = cudf::sorted_order(gold->view());
  auto sorted_gold     = cudf::gather(gold->view(), *gold_sort_order);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
}

TEST_F(JoinDictionaryTest, InnerJoinWithNulls)
{
  column_wrapper<int32_t> col0_0{{3, 1, 2, 0, 2}};
  strcol_wrapper col0_1({"s1", "s1", "s0", "s4", "s0"}, {true, true, false, true, true});
  column_wrapper<int32_t> col0_2_w{{0, 1, 2, 4, 1}};
  auto col0_2 = cudf::dictionary::encode(col0_2_w);

  column_wrapper<int32_t> col1_0{{2, 2, 0, 4, 3}};
  strcol_wrapper col1_1({"s1", "s0", "s1", "s2", "s1"});
  column_wrapper<int32_t> col1_2_w{{1, 0, 1, 2, 1}, {true, false, true, true, true}};
  auto col1_2 = cudf::dictionary::encode(col1_2_w);

  auto t0 = cudf::table_view({col0_0, col0_1, col0_2->view()});
  auto t1 = cudf::table_view({col1_0, col1_1, col1_2->view()});

  auto result      = inner_join(t0, t1, {0, 1}, {0, 1});
  auto result_view = result->view();
  auto decoded2    = cudf::dictionary::decode(result_view.column(2));
  auto decoded5    = cudf::dictionary::decode(result_view.column(5));
  std::vector<cudf::column_view> result_decoded({result_view.column(0),
                                                 result_view.column(1),
                                                 decoded2->view(),
                                                 result_view.column(3),
                                                 result_view.column(4),
                                                 decoded5->view()});
  auto result_sort_order = cudf::sorted_order(cudf::table_view(result_decoded));
  auto sorted_result     = cudf::gather(cudf::table_view(result_decoded), *result_sort_order);

  auto g0              = cudf::table_view({col0_0, col0_1, col0_2_w});
  auto g1              = cudf::table_view({col1_0, col1_1, col1_2_w});
  auto gold            = inner_join(g0, g1, {0, 1}, {0, 1});
  auto gold_sort_order = cudf::sorted_order(gold->view());
  auto sorted_gold     = cudf::gather(gold->view(), *gold_sort_order);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
}

TEST_F(JoinDictionaryTest, FullJoinNoNulls)
{
  column_wrapper<int32_t> col0_0{{3, 1, 2, 0, 3}};
  strcol_wrapper col0_1_w({"s0", "s1", "s2", "s4", "s1"});
  auto col0_1 = cudf::dictionary::encode(col0_1_w);
  column_wrapper<int32_t> col0_2{{0, 1, 2, 4, 1}};

  column_wrapper<int32_t> col1_0{{2, 2, 0, 4, 3}};
  strcol_wrapper col1_1_w{{"s1", "s0", "s1", "s2", "s1"}};
  auto col1_1 = cudf::dictionary::encode(col1_1_w);
  column_wrapper<int32_t> col1_2{{1, 0, 1, 2, 1}};

  auto t0 = cudf::table_view({col0_0, col0_1->view(), col0_2});
  auto t1 = cudf::table_view({col1_0, col1_1->view(), col1_2});

  auto result      = full_join(t0, t1, {0, 1}, {0, 1});
  auto result_view = result->view();
  auto decoded1    = cudf::dictionary::decode(result_view.column(1));
  auto decoded4    = cudf::dictionary::decode(result_view.column(4));
  std::vector<cudf::column_view> result_decoded({result_view.column(0),
                                                 decoded1->view(),
                                                 result_view.column(2),
                                                 result_view.column(3),
                                                 decoded4->view(),
                                                 result_view.column(5)});
  auto result_sort_order = cudf::sorted_order(cudf::table_view(result_decoded));
  auto sorted_result     = cudf::gather(cudf::table_view(result_decoded), *result_sort_order);

  auto g0              = cudf::table_view({col0_0, col0_1_w, col0_2});
  auto g1              = cudf::table_view({col1_0, col1_1_w, col1_2});
  auto gold            = full_join(g0, g1, {0, 1}, {0, 1});
  auto gold_sort_order = cudf::sorted_order(gold->view());
  auto sorted_gold     = cudf::gather(gold->view(), *gold_sort_order);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
}

TEST_F(JoinDictionaryTest, FullJoinWithNulls)
{
  column_wrapper<int32_t> col0_0_w{{3, 1, 2, 0, 3}};
  auto col0_0 = cudf::dictionary::encode(col0_0_w);
  strcol_wrapper col0_1({"s0", "s1", "s2", "s4", "s1"});
  column_wrapper<int32_t> col0_2{{0, 1, 2, 4, 1}};

  column_wrapper<int32_t> col1_0_w{{2, 2, 0, 4, 3}, {true, true, true, false, true}};
  auto col1_0 = cudf::dictionary::encode(col1_0_w);
  strcol_wrapper col1_1{{"s1", "s0", "s1", "s2", "s1"}};
  column_wrapper<int32_t> col1_2{{1, 0, 1, 2, 1}};

  auto t0 = cudf::table_view({col0_0->view(), col0_1, col0_2});
  auto t1 = cudf::table_view({col1_0->view(), col1_1, col1_2});

  auto result      = full_join(t0, t1, {0, 1}, {0, 1});
  auto result_view = result->view();
  auto decoded0    = cudf::dictionary::decode(result_view.column(0));
  auto decoded3    = cudf::dictionary::decode(result_view.column(3));
  std::vector<cudf::column_view> result_decoded({decoded0->view(),
                                                 result_view.column(1),
                                                 result_view.column(2),
                                                 decoded3->view(),
                                                 result_view.column(4),
                                                 result_view.column(5)});
  auto result_sort_order = cudf::sorted_order(cudf::table_view(result_decoded));
  auto sorted_result     = cudf::gather(cudf::table_view(result_decoded), *result_sort_order);

  auto g0              = cudf::table_view({col0_0_w, col0_1, col0_2});
  auto g1              = cudf::table_view({col1_0_w, col1_1, col1_2});
  auto gold            = full_join(g0, g1, {0, 1}, {0, 1});
  auto gold_sort_order = cudf::sorted_order(gold->view());
  auto sorted_gold     = cudf::gather(gold->view(), *gold_sort_order);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
}

TEST_F(JoinTest, FullJoinWithStructsAndNulls)
{
  column_wrapper<int32_t> col0_0{{3, 1, 2, 0, 3}};
  strcol_wrapper col0_1({"s0", "s1", "s2", "s4", "s1"});
  column_wrapper<int32_t> col0_2{{0, 1, 2, 4, 1}};

  std::initializer_list<std::string> col0_names = {"Samuel Vimes",
                                                   "Carrot Ironfoundersson",
                                                   "Angua von Überwald",
                                                   "Detritus",
                                                   "Carrot Ironfoundersson"};
  auto col0_names_col = strcol_wrapper{col0_names.begin(), col0_names.end()};
  auto col0_ages_col  = column_wrapper<int32_t>{{48, 27, 25, 31, 351}};

  auto col0_is_human_col =
    column_wrapper<bool>{{true, true, false, false, false}, {true, true, false, true, true}};

  auto col0_3 = cudf::test::structs_column_wrapper{
    {col0_names_col, col0_ages_col, col0_is_human_col}, {true, true, true, true, true}};

  column_wrapper<int32_t> col1_0{{2, 2, 0, 4, 3}, {true, true, true, false, true}};
  strcol_wrapper col1_1{{"s1", "s0", "s1", "s2", "s1"}};
  column_wrapper<int32_t> col1_2{{1, 0, 1, 2, 1}};

  std::initializer_list<std::string> col1_names = {"Carrot Ironfoundersson",
                                                   "Samuel Vimes",
                                                   "Carrot Ironfoundersson",
                                                   "Angua von Überwald",
                                                   "Carrot Ironfoundersson"};
  auto col1_names_col = strcol_wrapper{col1_names.begin(), col1_names.end()};
  auto col1_ages_col  = column_wrapper<int32_t>{{27, 48, 27, 25, 27}};

  auto col1_is_human_col =
    column_wrapper<bool>{{true, true, true, false, true}, {true, true, true, false, true}};

  auto col1_3 =
    cudf::test::structs_column_wrapper{{col1_names_col, col1_ages_col, col1_is_human_col}};

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols0.push_back(col0_2.release());
  cols0.push_back(col0_3.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());
  cols1.push_back(col1_2.release());
  cols1.push_back(col1_3.release());

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  auto result            = full_join(t0, t1, {0, 1, 3}, {0, 1, 3});
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

  column_wrapper<int32_t> col_gold_0{
    {3, 1, 2, 0, 3, -1, -1, -1, -1, -1},
    {true, true, true, true, true, false, false, false, false, false}};
  strcol_wrapper col_gold_1({"s0", "s1", "s2", "s4", "s1", "", "", "", "", ""},
                            {true, true, true, true, true, false, false, false, false, false});
  column_wrapper<int32_t> col_gold_2{
    {0, 1, 2, 4, 1, -1, -1, -1, -1, -1},
    {true, true, true, true, true, false, false, false, false, false}};
  auto gold_names0_col =
    strcol_wrapper{{"Samuel Vimes",
                    "Carrot Ironfoundersson",
                    "Angua von Überwald",
                    "Detritus",
                    "Carrot Ironfoundersson",
                    "",
                    "",
                    "",
                    "",
                    ""},
                   {true, true, true, true, true, false, false, false, false, false}};
  auto gold_ages0_col =
    column_wrapper<int32_t>{{48, 27, 25, 31, 351, -1, -1, -1, -1, -1},
                            {true, true, true, true, true, false, false, false, false, false}};

  auto gold_is_human0_col =
    column_wrapper<bool>{{true, true, false, false, false, false, false, false, false, false},
                         {true, true, false, true, true, false, false, false, false, false}};

  auto col_gold_3 = cudf::test::structs_column_wrapper{
    {gold_names0_col, gold_ages0_col, gold_is_human0_col},
    {true, true, true, true, true, false, false, false, false, false}};

  column_wrapper<int32_t> col_gold_4{
    {-1, -1, -1, -1, -1, 3, 2, 2, 0, 4},
    {false, false, false, false, false, true, true, true, true, false}};
  strcol_wrapper col_gold_5({"", "", "", "", "", "s1", "s1", "s0", "s1", "s2"},
                            {false, false, false, false, false, true, true, true, true, true});
  column_wrapper<int32_t> col_gold_6{
    {-1, -1, -1, -1, -1, 1, 1, 0, 1, 2},
    {false, false, false, false, false, true, true, true, true, true}};
  auto gold_names1_col =
    strcol_wrapper{{"",
                    "",
                    "",
                    "",
                    "",
                    "Carrot Ironfoundersson",
                    "Carrot Ironfoundersson",
                    "Samuel Vimes",
                    "Carrot Ironfoundersson",
                    "Angua von Überwald"},
                   {false, false, false, false, false, true, true, true, true, true}};
  auto gold_ages1_col =
    column_wrapper<int32_t>{{-1, -1, -1, -1, -1, 27, 27, 48, 27, 25},
                            {false, false, false, false, false, true, true, true, true, true}};

  auto gold_is_human1_col =
    column_wrapper<bool>{{false, false, false, false, false, true, true, true, true, false},
                         {false, false, false, false, false, true, true, true, true, false}};

  auto col_gold_7 = cudf::test::structs_column_wrapper{
    {gold_names1_col, gold_ages1_col, gold_is_human1_col},
    {false, false, false, false, false, true, true, true, true, true}};

  CVector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());
  cols_gold.push_back(col_gold_4.release());
  cols_gold.push_back(col_gold_5.release());
  cols_gold.push_back(col_gold_6.release());
  cols_gold.push_back(col_gold_7.release());

  Table gold(std::move(cols_gold));

  auto gold_sort_order = cudf::sorted_order(gold.view());
  auto sorted_gold     = cudf::gather(gold.view(), *gold_sort_order);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
}

using lcw = cudf::test::lists_column_wrapper<int32_t>;
using cudf::test::iterators::null_at;

struct JoinTestLists : public cudf::test::BaseFixture {
  /*
    [
      NULL,      0
      [1],       1
      [2, NULL], 2
      [],        3
      [5, 6]     4
  */
  lcw build{{{0}, {1}, {{2, 0}, null_at(1)}, {}, {5, 6}}, null_at(0)};

  /*
    [
      [1],       0
      [3],       1
      NULL,      2
      [],        3
      [2, NULL], 4
      [5],       5
      [6]        6
    ]
  */
  lcw probe{{{1}, {3}, {0}, {}, {{2, 0}, null_at(1)}, {5}, {6}}, null_at(2)};

  auto column_view_from_device_uvector(rmm::device_uvector<cudf::size_type> const& vector)
  {
    auto const indices_span = cudf::device_span<cudf::size_type const>{vector};
    return cudf::column_view{indices_span};
  }

  auto sort_and_gather(
    cudf::table_view table,
    cudf::column_view gather_map,
    cudf::out_of_bounds_policy oob_policy = cudf::out_of_bounds_policy::DONT_CHECK)
  {
    auto const gather_table = cudf::gather(table, gather_map, oob_policy);
    auto const sort_order   = cudf::sorted_order(*gather_table);
    return cudf::gather(*gather_table, *sort_order);
  }

  template <typename JoinFunc>
  void join(cudf::column_view left_gold_map,
            cudf::column_view right_gold_map,
            cudf::null_equality nulls_equal,
            JoinFunc join_func,
            cudf::out_of_bounds_policy oob_policy)
  {
    auto const build_tv = cudf::table_view{{build}};
    auto const probe_tv = cudf::table_view{{probe}};

    auto const [left_result_map, right_result_map] =
      join_func(build_tv,
                probe_tv,
                nulls_equal,
                cudf::get_default_stream(),
                cudf::get_current_device_resource_ref());

    auto const left_result_table =
      sort_and_gather(build_tv, column_view_from_device_uvector(*left_result_map), oob_policy);
    auto const right_result_table =
      sort_and_gather(probe_tv, column_view_from_device_uvector(*right_result_map), oob_policy);

    auto const left_gold_table  = sort_and_gather(build_tv, left_gold_map, oob_policy);
    auto const right_gold_table = sort_and_gather(probe_tv, right_gold_map, oob_policy);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*left_result_table, *left_gold_table);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*right_result_table, *right_gold_table);
  }

  void inner_join(cudf::column_view left_gold_map,
                  cudf::column_view right_gold_map,
                  cudf::null_equality nulls_equal)
  {
    join(left_gold_map,
         right_gold_map,
         nulls_equal,
         cudf::inner_join,
         cudf::out_of_bounds_policy::DONT_CHECK);
  }

  void full_join(cudf::column_view left_gold_map,
                 cudf::column_view right_gold_map,
                 cudf::null_equality nulls_equal)
  {
    join(left_gold_map,
         right_gold_map,
         nulls_equal,
         cudf::full_join,
         cudf::out_of_bounds_policy::NULLIFY);
  }

  void left_join(cudf::column_view left_gold_map,
                 cudf::column_view right_gold_map,
                 cudf::null_equality nulls_equal)
  {
    join(left_gold_map,
         right_gold_map,
         nulls_equal,
         cudf::left_join,
         cudf::out_of_bounds_policy::NULLIFY);
  }
};

TEST_F(JoinTestLists, ListWithNullsEqualInnerJoin)
{
  auto const left_gold_map  = column_wrapper<int32_t>({0, 1, 2, 3});
  auto const right_gold_map = column_wrapper<int32_t>({0, 2, 3, 4});
  this->inner_join(left_gold_map, right_gold_map, cudf::null_equality::EQUAL);
}

TEST_F(JoinTestLists, ListWithNullsUnequalInnerJoin)
{
  auto const left_gold_map  = column_wrapper<int32_t>({1, 3});
  auto const right_gold_map = column_wrapper<int32_t>({0, 3});
  this->inner_join(left_gold_map, right_gold_map, cudf::null_equality::UNEQUAL);
}

TEST_F(JoinTestLists, ListWithNullsEqualFullJoin)
{
  auto const left_gold_map =
    column_wrapper<int32_t>({0, 1, 2, 3, 4, NoneValue, NoneValue, NoneValue});
  auto const right_gold_map = column_wrapper<int32_t>({2, 0, 4, 3, NoneValue, 1, 5, 6});
  this->full_join(left_gold_map, right_gold_map, cudf::null_equality::EQUAL);
}

TEST_F(JoinTestLists, ListWithNullsUnequalFullJoin)
{
  auto const left_gold_map =
    column_wrapper<int32_t>({0, 1, 2, 3, 4, NoneValue, NoneValue, NoneValue, NoneValue, NoneValue});
  auto const right_gold_map =
    column_wrapper<int32_t>({NoneValue, 0, NoneValue, 3, NoneValue, 1, 5, 6, 2, 4});
  this->full_join(left_gold_map, right_gold_map, cudf::null_equality::UNEQUAL);
}

TEST_F(JoinTestLists, ListWithNullsEqualLeftJoin)
{
  auto const left_gold_map  = column_wrapper<int32_t>({0, 1, 2, 3, 4});
  auto const right_gold_map = column_wrapper<int32_t>({2, 0, 4, 3, NoneValue});
  this->left_join(left_gold_map, right_gold_map, cudf::null_equality::EQUAL);
}

TEST_F(JoinTestLists, ListWithNullsUnequalLeftJoin)
{
  auto const left_gold_map  = column_wrapper<int32_t>({0, 1, 2, 3, 4});
  auto const right_gold_map = column_wrapper<int32_t>({NoneValue, 0, NoneValue, 3, NoneValue});
  this->left_join(left_gold_map, right_gold_map, cudf::null_equality::UNEQUAL);
}

CUDF_TEST_PROGRAM_MAIN()
