/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <cudf/ast/nodes.hpp>
#include <cudf/ast/operators.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/join.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <limits>

template <typename T>
using column_wrapper = cudf::test::fixed_width_column_wrapper<T>;
using strcol_wrapper = cudf::test::strings_column_wrapper;
using CVector        = std::vector<std::unique_ptr<cudf::column>>;
using Table          = cudf::table;
constexpr cudf::size_type NoneValue =
  std::numeric_limits<cudf::size_type>::min();  // TODO: how to test if this isn't public?

struct JoinTest : public cudf::test::BaseFixture {
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

  auto result = cudf::inner_join(left, right, {0, 1, 2}, {0, 1, 2});

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

  auto result            = cudf::left_join(t0, t1, {0}, {0});
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

  column_wrapper<int32_t> col_gold_0{{3, 1, 2, 2, 0, 3}, {1, 1, 1, 1, 1, 1}};
  strcol_wrapper col_gold_1({"s0", "s1", "s2", "s2", "s4", "s1"}, {1, 1, 1, 1, 1, 1});
  column_wrapper<int32_t> col_gold_2{{0, 1, 2, 2, 4, 1}, {1, 1, 1, 1, 1, 1}};
  column_wrapper<int32_t> col_gold_3{{3, -1, 2, 2, 0, 3}, {1, 0, 1, 1, 1, 1}};
  strcol_wrapper col_gold_4({"s1", "", "s1", "s0", "s1", "s1"}, {1, 0, 1, 1, 1, 1});
  column_wrapper<int32_t> col_gold_5{{1, -1, 1, 0, 1, 1}, {1, 0, 1, 1, 1, 1}};
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

  auto result            = cudf::full_join(t0, t1, {0, 1}, {0, 1});
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

  column_wrapper<int32_t> col_gold_0{{3, 1, 2, 0, 3, -1, -1, -1, -1}, {1, 1, 1, 1, 1, 0, 0, 0, 0}};
  strcol_wrapper col_gold_1({"s0", "s1", "s2", "s4", "s1", "", "", "", ""},
                            {1, 1, 1, 1, 1, 0, 0, 0, 0});
  column_wrapper<int32_t> col_gold_2{{0, 1, 2, 4, 1, -1, -1, -1, -1}, {1, 1, 1, 1, 1, 0, 0, 0, 0}};
  column_wrapper<int32_t> col_gold_3{{-1, -1, -1, -1, 3, 2, 2, 0, 4}, {0, 0, 0, 0, 1, 1, 1, 1, 1}};
  strcol_wrapper col_gold_4({"", "", "", "", "s1", "s1", "s0", "s1", "s2"},
                            {0, 0, 0, 0, 1, 1, 1, 1, 1});
  column_wrapper<int32_t> col_gold_5{{-1, -1, -1, -1, 1, 1, 0, 1, 2}, {0, 0, 0, 0, 1, 1, 1, 1, 1}};

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

  column_wrapper<int32_t> col1_0{{2, 2, 0, 4, 3}, {1, 1, 1, 0, 1}};
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

  auto result            = cudf::full_join(t0, t1, {0, 1}, {0, 1});
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

  column_wrapper<int32_t> col_gold_0{{3, 1, 2, 0, 3, -1, -1, -1, -1}, {1, 1, 1, 1, 1, 0, 0, 0, 0}};
  strcol_wrapper col_gold_1({"s0", "s1", "s2", "s4", "s1", "", "", "", ""},
                            {1, 1, 1, 1, 1, 0, 0, 0, 0});
  column_wrapper<int32_t> col_gold_2{{0, 1, 2, 4, 1, -1, -1, -1, -1}, {1, 1, 1, 1, 1, 0, 0, 0, 0}};
  column_wrapper<int32_t> col_gold_3{{-1, -1, -1, -1, 3, 2, 2, 0, 4}, {0, 0, 0, 0, 1, 1, 1, 1, 0}};
  strcol_wrapper col_gold_4({"", "", "", "", "s1", "s1", "s0", "s1", "s2"},
                            {0, 0, 0, 0, 1, 1, 1, 1, 1});
  column_wrapper<int32_t> col_gold_5{{-1, -1, -1, -1, 1, 1, 0, 1, 2}, {0, 0, 0, 0, 1, 1, 1, 1, 1}};

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
                                 {  1,    0  }};
  strcol_wrapper          col0_1({"s0", "s1" });
  column_wrapper<int32_t> col0_2{{  0,    1 }};

  column_wrapper<int32_t> col1_0{{  2,    5,    3,    7 },
                                 {  1,    1,    1,    0 }};
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

  auto result            = cudf::full_join(t0, t1, {0, 1}, {0, 1});
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

#if 0
  std::cout << "Actual Results:\n";
  cudf::test::print(sorted_result->get_column(0).view(), std::cout, ",\t\t");
  cudf::test::print(sorted_result->get_column(1).view(), std::cout, ",\t\t");
  cudf::test::print(sorted_result->get_column(2).view(), std::cout, ",\t\t");
  cudf::test::print(sorted_result->get_column(3).view(), std::cout, ",\t\t");
#endif

  column_wrapper<int32_t> col_gold_0{{   3,   -1,   -1,    -1},
                                     {   1,    0,    0,     0}};
  strcol_wrapper          col_gold_1{{ "s0", "s1",  "",    ""},
                                     {   1,    1,    0,     0}};
  column_wrapper<int32_t> col_gold_2{{   0,    1,   -1,    -1},
                                     {   1,    1,    0,     0}};
  column_wrapper<int32_t> col_gold_3{{   3,   -1,    2,     5},
                                     {   1,    0,    1,     1}};
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

#if 0
  std::cout << "Expected Results:\n";
  cudf::test::print(sorted_gold->get_column(0).view(), std::cout, ",\t\t");
  cudf::test::print(sorted_gold->get_column(1).view(), std::cout, ",\t\t");
  cudf::test::print(sorted_gold->get_column(2).view(), std::cout, ",\t\t");
  cudf::test::print(sorted_gold->get_column(3).view(), std::cout, ",\t\t");
#endif

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);

  // Repeat test with compare_nulls_equal=false,
  // as per SQL standard.

  result            = cudf::full_join(t0, t1, {0, 1}, {0, 1}, cudf::null_equality::UNEQUAL);
  result_sort_order = cudf::sorted_order(result->view());
  sorted_result     = cudf::gather(result->view(), *result_sort_order);

  col_gold_0 =               {{   3,   -1,   -1,    -1,   -1},
                              {   1,    0,    0,     0,    0}};
  col_gold_1 = strcol_wrapper{{ "s0", "s1",   "",    "",   ""},
                              {   1,    1,    0,     0,    0}};
  col_gold_2 =               {{   0,    1,   -1,    -1,   -1},
                              {   1,    1,    0,     0,    0}};
  col_gold_3 =               {{   3,   -1,    2,     5,   -1},
                              {   1,    0,    1,     1,    0}};
  col_gold_4 = strcol_wrapper{{ "s0",  "",  "s1",  "s0",  "s1"},
                              {   1,    0,    1,     1,    1}};
  col_gold_5 =               {{   2,   -1,    1,     4,    8},
                              {   1,    0,    1,     1,    1}};

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

  auto result            = cudf::left_join(t0, t1, {0, 1}, {0, 1});
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

  column_wrapper<int32_t> col_gold_0({3, 1, 2, 0, 3});
  strcol_wrapper col_gold_1({"s0", "s1", "s2", "s4", "s1"});
  column_wrapper<int32_t> col_gold_2({0, 1, 2, 4, 1});
  column_wrapper<int32_t> col_gold_3{{-1, -1, -1, -1, 3}, {0, 0, 0, 0, 1}};
  strcol_wrapper col_gold_4{{"", "", "", "", "s1"}, {0, 0, 0, 0, 1}};
  column_wrapper<int32_t> col_gold_5{{-1, -1, -1, -1, 1}, {0, 0, 0, 0, 1}};
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
  strcol_wrapper col0_1({"s1", "s1", "", "s4", "s0"}, {1, 1, 0, 1, 1});
  column_wrapper<int32_t> col0_2{{0, 1, 2, 4, 1}};

  column_wrapper<int32_t> col1_0{{2, 2, 0, 4, 3}};
  strcol_wrapper col1_1({"s1", "s0", "s1", "s2", "s1"});
  column_wrapper<int32_t> col1_2{{1, 0, 1, 2, 1}, {1, 0, 1, 1, 1}};

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols0.push_back(col0_2.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());
  cols1.push_back(col1_2.release());

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  auto result            = cudf::left_join(t0, t1, {0, 1}, {0, 1});
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

  column_wrapper<int32_t> col_gold_0{{3, 1, 2, 0, 2}, {1, 1, 1, 1, 1}};
  strcol_wrapper col_gold_1({"s1", "s1", "", "s4", "s0"}, {1, 1, 0, 1, 1});
  column_wrapper<int32_t> col_gold_2{{0, 1, 2, 4, 1}, {1, 1, 1, 1, 1}};
  column_wrapper<int32_t> col_gold_3{{3, -1, -1, -1, 2}, {1, 0, 0, 0, 1}};
  strcol_wrapper col_gold_4{{"s1", "", "", "", "s0"}, {1, 0, 0, 0, 1}};
  column_wrapper<int32_t> col_gold_5{{1, -1, -1, -1, -1}, {1, 0, 0, 0, 0}};

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
  strcol_wrapper col0_1({"s1", "s1", "", "s4", "s0"}, {1, 1, 0, 1, 1});
  column_wrapper<int32_t> col0_2{{0, 1, 2, 4, 1}};
  auto col0_names_col = strcol_wrapper{
    "Samuel Vimes", "Carrot Ironfoundersson", "Detritus", "Samuel Vimes", "Angua von Überwald"};
  auto col0_ages_col = column_wrapper<int32_t>{{48, 27, 351, 31, 25}};

  auto col0_is_human_col = column_wrapper<bool>{{true, true, false, false, false}, {1, 1, 0, 1, 0}};

  auto col0_3 =
    cudf::test::structs_column_wrapper{{col0_names_col, col0_ages_col, col0_is_human_col}};

  column_wrapper<int32_t> col1_0{{2, 2, 0, 4, 3}};
  strcol_wrapper col1_1({"s1", "s0", "s1", "s2", "s1"});
  column_wrapper<int32_t> col1_2{{1, 0, 1, 2, 1}, {1, 0, 1, 1, 1}};
  auto col1_names_col = strcol_wrapper{
    "Samuel Vimes", "Detritus", "Detritus", "Carrot Ironfoundersson", "Angua von Überwald"};
  auto col1_ages_col = column_wrapper<int32_t>{{48, 35, 351, 22, 25}};

  auto col1_is_human_col = column_wrapper<bool>{{true, true, false, false, true}, {1, 1, 0, 1, 1}};

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

  auto result            = cudf::left_join(t0, t1, {3}, {3});
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

  column_wrapper<int32_t> col_gold_0{{3, 2, 1, 0, 2}, {1, 1, 1, 1, 1}};
  strcol_wrapper col_gold_1({"s1", "", "s1", "s4", "s0"}, {1, 0, 1, 1, 1});
  column_wrapper<int32_t> col_gold_2{{0, 2, 1, 4, 1}, {1, 1, 1, 1, 1}};
  auto col0_gold_names_col = strcol_wrapper{
    "Samuel Vimes", "Detritus", "Carrot Ironfoundersson", "Samuel Vimes", "Angua von Überwald"};
  auto col0_gold_ages_col = column_wrapper<int32_t>{{48, 351, 27, 31, 25}};

  auto col0_gold_is_human_col =
    column_wrapper<bool>{{true, false, true, false, false}, {1, 0, 1, 1, 0}};

  auto col_gold_3 = cudf::test::structs_column_wrapper{
    {col0_gold_names_col, col0_gold_ages_col, col0_gold_is_human_col}};

  column_wrapper<int32_t> col_gold_4{{2, 0, -1, -1, -1}, {1, 1, 0, 0, 0}};
  strcol_wrapper col_gold_5{{"s1", "s1", "", "", ""}, {1, 1, 0, 0, 0}};
  column_wrapper<int32_t> col_gold_6{{1, 1, -1, -1, -1}, {1, 1, 0, 0, 0}};
  auto col1_gold_names_col = strcol_wrapper{{
                                              "Samuel Vimes",
                                              "Detritus",
                                              "",
                                              "",
                                              "",
                                            },
                                            {1, 1, 0, 0, 0}};
  auto col1_gold_ages_col  = column_wrapper<int32_t>{{48, 351, -1, -1, -1}, {1, 1, 0, 0, 0}};

  auto col1_gold_is_human_col =
    column_wrapper<bool>{{true, false, false, false, false}, {1, 0, 0, 0, 0}};

  auto col_gold_7 = cudf::test::structs_column_wrapper{
    {col1_gold_names_col, col1_gold_ages_col, col1_gold_is_human_col}, {1, 1, 0, 0, 0}};

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
                                 {  1,    0,    1}};
  strcol_wrapper          col0_1({"s0", "s1", "s2" });
  column_wrapper<int32_t> col0_2{{  0,    1,    2 }};

  column_wrapper<int32_t> col1_0{{  2,    5,    3,    7 },
                                 {  1,    1,    1,    0 }};
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

  auto result            = cudf::left_join(t0, t1, {0, 1}, {0, 1});
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

#if 0
  std::cout << "Actual Results:\n";
  cudf::test::print(sorted_result->get_column(0).view(), std::cout, ",\t\t");
  cudf::test::print(sorted_result->get_column(1).view(), std::cout, ",\t\t");
  cudf::test::print(sorted_result->get_column(2).view(), std::cout, ",\t\t");
  cudf::test::print(sorted_result->get_column(3).view(), std::cout, ",\t\t");
#endif

  column_wrapper<int32_t> col_gold_0{{   3,    -1,    2},
                                     {   1,     0,    1}};
  strcol_wrapper          col_gold_1({ "s0",  "s1", "s2"},
                                     {   1,     1,    1});
  column_wrapper<int32_t> col_gold_2{{   0,     1,    2},
                                     {   1,     1,    1}};
  column_wrapper<int32_t> col_gold_3{{   3,    -1,   -1},
                                     {   1,     0,    0}};
  strcol_wrapper          col_gold_4({ "s0",  "s1",  ""},
                                     {   1,     1,    0});
  column_wrapper<int32_t> col_gold_5{{   2,     8,   -1},
                                     {   1,     1,    0}};

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

#if 0
  std::cout << "Expected Results:\n";
  cudf::test::print(sorted_gold->get_column(0).view(), std::cout, ",\t\t");
  cudf::test::print(sorted_gold->get_column(1).view(), std::cout, ",\t\t");
  cudf::test::print(sorted_gold->get_column(2).view(), std::cout, ",\t\t");
  cudf::test::print(sorted_gold->get_column(3).view(), std::cout, ",\t\t");
#endif

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);

  // Repeat test with compare_nulls_equal=false,
  // as per SQL standard.

  result            = cudf::left_join(t0, t1, {0, 1}, {0, 1}, cudf::null_equality::UNEQUAL);
  result_sort_order = cudf::sorted_order(result->view());
  sorted_result     = cudf::gather(result->view(), *result_sort_order);


  col_gold_0 = {{   3,    -1,    2},
                {   1,     0,    1}};
  col_gold_1 = {{ "s0",  "s1", "s2"},
                {   1,     1,    1}};
  col_gold_2 = {{   0,     1,    2},
                {   1,     1,    1}};
  col_gold_3 = {{   3,    -1,   -1},
                {   1,     0,    0}};
  col_gold_4 = {{ "s0",   "",   ""},
                {   1,     0,    0}};
  col_gold_5 = {{   2,    -1,   -1},
                {   1,     0,    0}};

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

  auto result            = cudf::inner_join(t0, t1, {0, 1}, {0, 1});
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
  strcol_wrapper col0_1({"s1", "s1", "s0", "s4", "s0"}, {1, 1, 0, 1, 1});
  column_wrapper<int32_t> col0_2{{0, 1, 2, 4, 1}};

  column_wrapper<int32_t> col1_0{{2, 2, 0, 4, 3}};
  strcol_wrapper col1_1({"s1", "s0", "s1", "s2", "s1"});
  column_wrapper<int32_t> col1_2{{1, 0, 1, 2, 1}, {1, 0, 1, 1, 1}};

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols0.push_back(col0_2.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());
  cols1.push_back(col1_2.release());

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  auto result            = cudf::inner_join(t0, t1, {0, 1}, {0, 1});
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

  column_wrapper<int32_t> col_gold_0{{3, 2}};
  strcol_wrapper col_gold_1({"s1", "s0"}, {1, 1});
  column_wrapper<int32_t> col_gold_2{{0, 1}};
  column_wrapper<int32_t> col_gold_3{{3, 2}};
  strcol_wrapper col_gold_4({"s1", "s0"}, {1, 1});
  column_wrapper<int32_t> col_gold_5{{1, -1}, {1, 0}};
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
  strcol_wrapper col0_1({"s1", "s1", "s0", "s4", "s0"}, {1, 1, 0, 1, 1});
  column_wrapper<int32_t> col0_2{{0, 1, 2, 4, 1}};
  std::initializer_list<std::string> col0_names = {
    "Samuel Vimes", "Carrot Ironfoundersson", "Detritus", "Samuel Vimes", "Angua von Überwald"};
  auto col0_names_col = strcol_wrapper{col0_names.begin(), col0_names.end()};
  auto col0_ages_col  = column_wrapper<int32_t>{{48, 27, 351, 31, 25}};

  auto col0_is_human_col = column_wrapper<bool>{{true, true, false, false, false}, {1, 1, 0, 1, 0}};

  auto col0_3 =
    cudf::test::structs_column_wrapper{{col0_names_col, col0_ages_col, col0_is_human_col}};

  column_wrapper<int32_t> col1_0{{2, 2, 0, 4, 3}};
  strcol_wrapper col1_1({"s1", "s0", "s1", "s2", "s1"});
  column_wrapper<int32_t> col1_2{{1, 0, 1, 2, 1}, {1, 0, 1, 1, 1}};
  std::initializer_list<std::string> col1_names = {"Carrot Ironfoundersson",
                                                   "Angua von Überwald",
                                                   "Detritus",
                                                   "Carrot Ironfoundersson",
                                                   "Samuel Vimes"};
  auto col1_names_col = strcol_wrapper{col1_names.begin(), col1_names.end()};
  auto col1_ages_col  = column_wrapper<int32_t>{{351, 25, 27, 31, 48}};

  auto col1_is_human_col = column_wrapper<bool>{{true, false, false, false, true}, {1, 0, 0, 1, 1}};

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

  auto result            = cudf::inner_join(t0, t1, {0, 1, 3}, {0, 1, 3});
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

  column_wrapper<int32_t> col_gold_0{{3, 2}};
  strcol_wrapper col_gold_1({"s1", "s0"}, {1, 1});
  column_wrapper<int32_t> col_gold_2{{0, 1}};
  auto col_gold_3_names_col = strcol_wrapper{"Samuel Vimes", "Angua von Überwald"};
  auto col_gold_3_ages_col  = column_wrapper<int32_t>{{48, 25}};

  auto col_gold_3_is_human_col = column_wrapper<bool>{{true, false}, {1, 0}};

  auto col_gold_3 = cudf::test::structs_column_wrapper{
    {col_gold_3_names_col, col_gold_3_ages_col, col_gold_3_is_human_col}};

  column_wrapper<int32_t> col_gold_4{{3, 2}};
  strcol_wrapper col_gold_5({"s1", "s0"}, {1, 1});
  column_wrapper<int32_t> col_gold_6{{1, -1}, {1, 0}};
  auto col_gold_7_names_col = strcol_wrapper{"Samuel Vimes", "Angua von Überwald"};
  auto col_gold_7_ages_col  = column_wrapper<int32_t>{{48, 25}};

  auto col_gold_7_is_human_col = column_wrapper<bool>{{true, false}, {1, 0}};

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

// // Test to check join behaviour when join keys are null.
TEST_F(JoinTest, InnerJoinOnNulls)
{
  // clang-format off
  column_wrapper<int32_t> col0_0{{  3,    1,    2,    0,    2}};
  strcol_wrapper          col0_1({"s1", "s1", "s8", "s4", "s0"},
                                 {  1,    1,    0,    1,    1});
  column_wrapper<int32_t> col0_2{{  0,    1,    2,    4,    1}};

  column_wrapper<int32_t> col1_0{{  2,    2,    0,    4,    3}};
  strcol_wrapper          col1_1({"s1", "s0", "s1", "s2", "s1"},
                                 {  1,    0,    1,    1,    1});
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

  auto result            = cudf::inner_join(t0, t1, {0, 1}, {0, 1});
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

  column_wrapper<int32_t> col_gold_0 {{  3,    2}};
  strcol_wrapper          col_gold_1 ({"s1", "s0"},
                                      {  1,    0});
  column_wrapper<int32_t> col_gold_2{{   0,    2}};
  column_wrapper<int32_t> col_gold_3 {{  3,    2}};
  strcol_wrapper          col_gold_4 ({"s1", "s0"},
                                      {  1,    0});
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

  result            = cudf::inner_join(t0, t1, {0, 1}, {0, 1},  cudf::null_equality::UNEQUAL);
  result_sort_order = cudf::sorted_order(result->view());
  sorted_result     = cudf::gather(result->view(), *result_sort_order);

  col_gold_0 =               {{  3}};
  col_gold_1 = strcol_wrapper({"s1"},
                              {  1});
  col_gold_2 =               {{  0}};
  col_gold_3 =               {{  3}};
  col_gold_4 = strcol_wrapper({"s1"},
                              {  1});
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
  column_wrapper<int32_t> col1_1{{1, 0, 1, 2, 1}, {1, 0, 1, 1, 1}};

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());

  Table empty0(std::move(cols0));
  Table t1(std::move(cols1));

  auto result = cudf::inner_join(empty0, t1, {0, 1}, {0, 1});
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(empty0, *result);
}

TEST_F(JoinTest, EmptyLeftTableLeftJoin)
{
  column_wrapper<int32_t> col0_0;
  column_wrapper<int32_t> col0_1;

  column_wrapper<int32_t> col1_0{{2, 2, 0, 4, 3}};
  column_wrapper<int32_t> col1_1{{1, 0, 1, 2, 1}, {1, 0, 1, 1, 1}};

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());

  Table empty0(std::move(cols0));
  Table t1(std::move(cols1));

  auto result = cudf::left_join(empty0, t1, {0, 1}, {0, 1});
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(empty0, *result);
}

TEST_F(JoinTest, EmptyLeftTableFullJoin)
{
  column_wrapper<int32_t> col0_0;
  column_wrapper<int32_t> col0_1;

  column_wrapper<int32_t> col1_0{{2, 2, 0, 4, 3}};
  column_wrapper<int32_t> col1_1{{1, 0, 1, 2, 1}, {1, 0, 1, 1, 1}};

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());

  Table lhs(std::move(cols0));
  Table rhs(std::move(cols1));

  auto result            = cudf::full_join(lhs, rhs, {0, 1}, {0, 1});
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

  column_wrapper<int32_t> col_gold_0{{-1, -1, -1, -1, -1}, {0, 0, 0, 0, 0}};
  column_wrapper<int32_t> col_gold_1{{-1, -1, -1, -1, -1}, {0, 0, 0, 0, 0}};
  column_wrapper<int32_t> col_gold_2{{2, 2, 0, 4, 3}};
  column_wrapper<int32_t> col_gold_3{{1, 0, 1, 2, 1}, {1, 0, 1, 1, 1}};

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
  column_wrapper<int32_t> col0_1{{1, 0, 1, 2, 1}, {1, 0, 1, 1, 1}};

  column_wrapper<int32_t> col1_0;
  column_wrapper<int32_t> col1_1;

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());

  Table t0(std::move(cols0));
  Table empty1(std::move(cols1));

  auto result = cudf::inner_join(t0, empty1, {0, 1}, {0, 1});
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(empty1, *result);
}

TEST_F(JoinTest, EmptyRightTableLeftJoin)
{
  column_wrapper<int32_t> col0_0{{2, 2, 0, 4, 3}, {1, 1, 1, 1, 1}};
  column_wrapper<int32_t> col0_1{{1, 0, 1, 2, 1}, {1, 0, 1, 1, 1}};

  column_wrapper<int32_t> col1_0;
  column_wrapper<int32_t> col1_1;

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());

  Table t0(std::move(cols0));
  Table empty1(std::move(cols1));

  auto result = cudf::left_join(t0, empty1, {0, 1}, {0, 1});
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(t0, *result);
}

TEST_F(JoinTest, EmptyRightTableFullJoin)
{
  column_wrapper<int32_t> col0_0{{2, 2, 0, 4, 3}};
  column_wrapper<int32_t> col0_1{{1, 0, 1, 2, 1}, {1, 0, 1, 1, 1}};

  column_wrapper<int32_t> col1_0;
  column_wrapper<int32_t> col1_1;

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());

  Table t0(std::move(cols0));
  Table empty1(std::move(cols1));

  auto result = cudf::full_join(t0, empty1, {0, 1}, {0, 1});
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(t0, *result);
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

  auto result = cudf::inner_join(t0, empty1, {0, 1}, {0, 1});
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

  auto result = cudf::left_join(t0, empty1, {0, 1}, {0, 1});
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

  auto result = cudf::full_join(t0, empty1, {0, 1}, {0, 1});
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

  auto result = cudf::inner_join(t0, t1, {0, 1}, {0, 1});

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

  auto result = cudf::left_join(t0, t1, {0, 1}, {0, 1});

  column_wrapper<int32_t> col_gold_0{{0, 0, 0, 0}, {1, 1, 1, 1}};
  strcol_wrapper col_gold_1({"s0", "s0", "s0", "s0"}, {1, 1, 1, 1});
  column_wrapper<int32_t> col_gold_2{{0, 0, 0, 0}, {1, 1, 1, 1}};
  strcol_wrapper col_gold_3({"s0", "s0", "s0", "s0"}, {1, 1, 1, 1});

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

  auto result = cudf::full_join(t0, t1, {0, 1}, {0, 1});

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

  auto result            = cudf::inner_join(t0, t1, {0}, {0});
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

  cudf::hash_join hash_join(t1, cudf::null_equality::EQUAL);

  {
    CVector cols0;
    cols0.emplace_back(column_wrapper<int32_t>{{3, 1, 2, 0, 3}}.release());
    cols0.emplace_back(strcol_wrapper({"s0", "s1", "s2", "s4", "s1"}).release());

    Table t0(std::move(cols0));

    auto output_size                         = hash_join.full_join_size(t0);
    std::optional<std::size_t> optional_size = output_size;

    std::size_t const size_gold = 9;
    EXPECT_EQ(output_size, size_gold);

    auto result = hash_join.full_join(t0, cudf::null_equality::EQUAL, optional_size);
    auto result_table =
      cudf::table_view({cudf::column_view{cudf::data_type{cudf::type_id::INT32},
                                          static_cast<cudf::size_type>(result.first->size()),
                                          result.first->data()},
                        cudf::column_view{cudf::data_type{cudf::type_id::INT32},
                                          static_cast<cudf::size_type>(result.second->size()),
                                          result.second->data()}});
    auto result_sort_order = cudf::sorted_order(result_table);
    auto sorted_result     = cudf::gather(result_table, *result_sort_order);

    column_wrapper<int32_t> col_gold_0{{NoneValue, NoneValue, NoneValue, NoneValue, 4, 0, 1, 2, 3}};
    column_wrapper<int32_t> col_gold_1{{0, 1, 2, 3, 4, NoneValue, NoneValue, NoneValue, NoneValue}};

    CVector cols_gold;
    cols_gold.push_back(col_gold_0.release());
    cols_gold.push_back(col_gold_1.release());

    Table gold(std::move(cols_gold));
    auto gold_sort_order = cudf::sorted_order(gold.view());
    auto sorted_gold     = cudf::gather(gold.view(), *gold_sort_order);

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

    auto result = hash_join.left_join(t0, cudf::null_equality::EQUAL, optional_size);
    auto result_table =
      cudf::table_view({cudf::column_view{cudf::data_type{cudf::type_id::INT32},
                                          static_cast<cudf::size_type>(result.first->size()),
                                          result.first->data()},
                        cudf::column_view{cudf::data_type{cudf::type_id::INT32},
                                          static_cast<cudf::size_type>(result.second->size()),
                                          result.second->data()}});
    auto result_sort_order = cudf::sorted_order(result_table);
    auto sorted_result     = cudf::gather(result_table, *result_sort_order);

    column_wrapper<int32_t> col_gold_0{{0, 1, 2, 3, 4}};
    column_wrapper<int32_t> col_gold_1{{NoneValue, NoneValue, NoneValue, NoneValue, 4}};

    CVector cols_gold;
    cols_gold.push_back(col_gold_0.release());
    cols_gold.push_back(col_gold_1.release());

    Table gold(std::move(cols_gold));
    auto gold_sort_order = cudf::sorted_order(gold.view());
    auto sorted_gold     = cudf::gather(gold.view(), *gold_sort_order);

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

    auto result = hash_join.inner_join(t0, cudf::null_equality::EQUAL, optional_size);
    auto result_table =
      cudf::table_view({cudf::column_view{cudf::data_type{cudf::type_id::INT32},
                                          static_cast<cudf::size_type>(result.first->size()),
                                          result.first->data()},
                        cudf::column_view{cudf::data_type{cudf::type_id::INT32},
                                          static_cast<cudf::size_type>(result.second->size()),
                                          result.second->data()}});
    auto result_sort_order = cudf::sorted_order(result_table);
    auto sorted_result     = cudf::gather(result_table, *result_sort_order);

    column_wrapper<int32_t> col_gold_0{{2, 4, 0}};
    column_wrapper<int32_t> col_gold_1{{1, 1, 4}};

    CVector cols_gold;
    cols_gold.push_back(col_gold_0.release());
    cols_gold.push_back(col_gold_1.release());

    Table gold(std::move(cols_gold));
    auto gold_sort_order = cudf::sorted_order(gold.view());
    auto sorted_gold     = cudf::gather(gold.view(), *gold_sort_order);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
  }
}

struct JoinDictionaryTest : public cudf::test::BaseFixture {
};

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
  {
    auto result      = cudf::left_join(t0, t1, {0}, {0});
    auto result_view = result->view();
    auto decoded1    = cudf::dictionary::decode(result_view.column(1));
    auto decoded4    = cudf::dictionary::decode(result_view.column(4));
    std::vector<cudf::column_view> result_decoded({result_view.column(0),
                                                   decoded1->view(),
                                                   result_view.column(2),
                                                   result_view.column(3),
                                                   decoded4->view(),
                                                   result_view.column(5)});

    auto gold = cudf::left_join(g0, g1, {0}, {0});
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*gold, cudf::table_view(result_decoded));
  }
}

TEST_F(JoinDictionaryTest, LeftJoinWithNulls)
{
  column_wrapper<int32_t> col0_0{{3, 1, 2, 0, 2}};
  strcol_wrapper col0_1({"s1", "s1", "s0", "s4", "s0"}, {1, 1, 0, 1, 1});
  column_wrapper<int32_t> col0_2_w{{0, 1, 2, 4, 1}};
  auto col0_2 = cudf::dictionary::encode(col0_2_w);

  column_wrapper<int32_t> col1_0{{2, 2, 0, 4, 3}};
  strcol_wrapper col1_1({"s1", "s0", "s1", "s2", "s1"});
  column_wrapper<int32_t> col1_2_w{{1, 0, 1, 2, 1}, {1, 0, 1, 1, 1}};
  auto col1_2 = cudf::dictionary::encode(col1_2_w);

  auto t0 = cudf::table_view({col0_0, col0_1, col0_2->view()});
  auto t1 = cudf::table_view({col1_0, col1_1, col1_2->view()});

  auto result      = cudf::left_join(t0, t1, {0, 1}, {0, 1});
  auto result_view = result->view();
  auto decoded2    = cudf::dictionary::decode(result_view.column(2));
  auto decoded5    = cudf::dictionary::decode(result_view.column(5));
  std::vector<cudf::column_view> result_decoded({result_view.column(0),
                                                 result_view.column(1),
                                                 decoded2->view(),
                                                 result_view.column(3),
                                                 result_view.column(4),
                                                 decoded5->view()});

  auto g0   = cudf::table_view({col0_0, col0_1, col0_2_w});
  auto g1   = cudf::table_view({col1_0, col1_1, col1_2_w});
  auto gold = cudf::left_join(g0, g1, {0, 1}, {0, 1});
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*gold, cudf::table_view(result_decoded));
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

  auto result      = cudf::inner_join(t0, t1, {0, 1}, {0, 1});
  auto result_view = result->view();
  auto decoded1    = cudf::dictionary::decode(result_view.column(1));
  auto decoded4    = cudf::dictionary::decode(result_view.column(4));
  std::vector<cudf::column_view> result_decoded({result_view.column(0),
                                                 decoded1->view(),
                                                 result_view.column(2),
                                                 result_view.column(3),
                                                 decoded4->view(),
                                                 result_view.column(5)});

  auto g0   = cudf::table_view({col0_0, col0_1_w, col0_2});
  auto g1   = cudf::table_view({col1_0, col1_1_w, col1_2});
  auto gold = cudf::inner_join(g0, g1, {0, 1}, {0, 1});
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*gold, cudf::table_view(result_decoded));
}

TEST_F(JoinDictionaryTest, InnerJoinWithNulls)
{
  column_wrapper<int32_t> col0_0{{3, 1, 2, 0, 2}};
  strcol_wrapper col0_1({"s1", "s1", "s0", "s4", "s0"}, {1, 1, 0, 1, 1});
  column_wrapper<int32_t> col0_2_w{{0, 1, 2, 4, 1}};
  auto col0_2 = cudf::dictionary::encode(col0_2_w);

  column_wrapper<int32_t> col1_0{{2, 2, 0, 4, 3}};
  strcol_wrapper col1_1({"s1", "s0", "s1", "s2", "s1"});
  column_wrapper<int32_t> col1_2_w{{1, 0, 1, 2, 1}, {1, 0, 1, 1, 1}};
  auto col1_2 = cudf::dictionary::encode(col1_2_w);

  auto t0 = cudf::table_view({col0_0, col0_1, col0_2->view()});
  auto t1 = cudf::table_view({col1_0, col1_1, col1_2->view()});

  auto result      = cudf::inner_join(t0, t1, {0, 1}, {0, 1});
  auto result_view = result->view();
  auto decoded2    = cudf::dictionary::decode(result_view.column(2));
  auto decoded5    = cudf::dictionary::decode(result_view.column(5));
  std::vector<cudf::column_view> result_decoded({result_view.column(0),
                                                 result_view.column(1),
                                                 decoded2->view(),
                                                 result_view.column(3),
                                                 result_view.column(4),
                                                 decoded5->view()});

  auto g0   = cudf::table_view({col0_0, col0_1, col0_2_w});
  auto g1   = cudf::table_view({col1_0, col1_1, col1_2_w});
  auto gold = cudf::inner_join(g0, g1, {0, 1}, {0, 1});
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*gold, cudf::table_view(result_decoded));
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

  auto result      = cudf::full_join(t0, t1, {0, 1}, {0, 1});
  auto result_view = result->view();
  auto decoded1    = cudf::dictionary::decode(result_view.column(1));
  auto decoded4    = cudf::dictionary::decode(result_view.column(4));
  std::vector<cudf::column_view> result_decoded({result_view.column(0),
                                                 decoded1->view(),
                                                 result_view.column(2),
                                                 result_view.column(3),
                                                 decoded4->view(),
                                                 result_view.column(5)});

  auto g0   = cudf::table_view({col0_0, col0_1_w, col0_2});
  auto g1   = cudf::table_view({col1_0, col1_1_w, col1_2});
  auto gold = cudf::full_join(g0, g1, {0, 1}, {0, 1});
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*gold, cudf::table_view(result_decoded));
}

TEST_F(JoinDictionaryTest, FullJoinWithNulls)
{
  column_wrapper<int32_t> col0_0_w{{3, 1, 2, 0, 3}};
  auto col0_0 = cudf::dictionary::encode(col0_0_w);
  strcol_wrapper col0_1({"s0", "s1", "s2", "s4", "s1"});
  column_wrapper<int32_t> col0_2{{0, 1, 2, 4, 1}};

  column_wrapper<int32_t> col1_0_w{{2, 2, 0, 4, 3}, {1, 1, 1, 0, 1}};
  auto col1_0 = cudf::dictionary::encode(col1_0_w);
  strcol_wrapper col1_1{{"s1", "s0", "s1", "s2", "s1"}};
  column_wrapper<int32_t> col1_2{{1, 0, 1, 2, 1}};

  auto t0 = cudf::table_view({col0_0->view(), col0_1, col0_2});
  auto t1 = cudf::table_view({col1_0->view(), col1_1, col1_2});

  auto result      = cudf::full_join(t0, t1, {0, 1}, {0, 1});
  auto result_view = result->view();
  auto decoded0    = cudf::dictionary::decode(result_view.column(0));
  auto decoded3    = cudf::dictionary::decode(result_view.column(3));
  std::vector<cudf::column_view> result_decoded({decoded0->view(),
                                                 result_view.column(1),
                                                 result_view.column(2),
                                                 decoded3->view(),
                                                 result_view.column(4),
                                                 result_view.column(5)});

  auto g0   = cudf::table_view({col0_0_w, col0_1, col0_2});
  auto g1   = cudf::table_view({col1_0_w, col1_1, col1_2});
  auto gold = cudf::full_join(g0, g1, {0, 1}, {0, 1});
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*gold, cudf::table_view(result_decoded));
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

  auto col0_is_human_col = column_wrapper<bool>{{true, true, false, false, false}, {1, 1, 0, 1, 1}};

  auto col0_3 = cudf::test::structs_column_wrapper{
    {col0_names_col, col0_ages_col, col0_is_human_col}, {1, 1, 1, 1, 1}};

  column_wrapper<int32_t> col1_0{{2, 2, 0, 4, 3}, {1, 1, 1, 0, 1}};
  strcol_wrapper col1_1{{"s1", "s0", "s1", "s2", "s1"}};
  column_wrapper<int32_t> col1_2{{1, 0, 1, 2, 1}};

  std::initializer_list<std::string> col1_names = {"Carrot Ironfoundersson",
                                                   "Samuel Vimes",
                                                   "Carrot Ironfoundersson",
                                                   "Angua von Überwald",
                                                   "Carrot Ironfoundersson"};
  auto col1_names_col = strcol_wrapper{col1_names.begin(), col1_names.end()};
  auto col1_ages_col  = column_wrapper<int32_t>{{27, 48, 27, 25, 27}};

  auto col1_is_human_col = column_wrapper<bool>{{true, true, true, false, true}, {1, 1, 1, 0, 1}};

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

  auto result            = cudf::full_join(t0, t1, {0, 1, 3}, {0, 1, 3});
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

  column_wrapper<int32_t> col_gold_0{{3, 1, 2, 0, 3, -1, -1, -1, -1, -1},
                                     {1, 1, 1, 1, 1, 0, 0, 0, 0, 0}};
  strcol_wrapper col_gold_1({"s0", "s1", "s2", "s4", "s1", "", "", "", "", ""},
                            {1, 1, 1, 1, 1, 0, 0, 0, 0, 0});
  column_wrapper<int32_t> col_gold_2{{0, 1, 2, 4, 1, -1, -1, -1, -1, -1},
                                     {1, 1, 1, 1, 1, 0, 0, 0, 0, 0}};
  auto gold_names0_col = strcol_wrapper{{"Samuel Vimes",
                                         "Carrot Ironfoundersson",
                                         "Angua von Überwald",
                                         "Detritus",
                                         "Carrot Ironfoundersson",
                                         "",
                                         "",
                                         "",
                                         "",
                                         ""},
                                        {1, 1, 1, 1, 1, 0, 0, 0, 0, 0}};
  auto gold_ages0_col  = column_wrapper<int32_t>{{48, 27, 25, 31, 351, -1, -1, -1, -1, -1},
                                                {1, 1, 1, 1, 1, 0, 0, 0, 0, 0}};

  auto gold_is_human0_col =
    column_wrapper<bool>{{true, true, false, false, false, false, false, false, false, false},
                         {1, 1, 0, 1, 1, 0, 0, 0, 0, 0}};

  auto col_gold_3 = cudf::test::structs_column_wrapper{
    {gold_names0_col, gold_ages0_col, gold_is_human0_col}, {1, 1, 1, 1, 1, 0, 0, 0, 0, 0}};

  column_wrapper<int32_t> col_gold_4{{-1, -1, -1, -1, -1, 3, 2, 2, 0, 4},
                                     {0, 0, 0, 0, 0, 1, 1, 1, 1, 0}};
  strcol_wrapper col_gold_5({"", "", "", "", "", "s1", "s1", "s0", "s1", "s2"},
                            {0, 0, 0, 0, 0, 1, 1, 1, 1, 1});
  column_wrapper<int32_t> col_gold_6{{-1, -1, -1, -1, -1, 1, 1, 0, 1, 2},
                                     {0, 0, 0, 0, 0, 1, 1, 1, 1, 1}};
  auto gold_names1_col = strcol_wrapper{{"",
                                         "",
                                         "",
                                         "",
                                         "",
                                         "Carrot Ironfoundersson",
                                         "Carrot Ironfoundersson",
                                         "Samuel Vimes",
                                         "Carrot Ironfoundersson",
                                         "Angua von Überwald"},
                                        {0, 0, 0, 0, 0, 1, 1, 1, 1, 1}};
  auto gold_ages1_col  = column_wrapper<int32_t>{{-1, -1, -1, -1, -1, 27, 27, 48, 27, 25},
                                                {0, 0, 0, 0, 0, 1, 1, 1, 1, 1}};

  auto gold_is_human1_col =
    column_wrapper<bool>{{false, false, false, false, false, true, true, true, true, false},
                         {0, 0, 0, 0, 0, 1, 1, 1, 1, 0}};

  auto col_gold_7 = cudf::test::structs_column_wrapper{
    {gold_names1_col, gold_ages1_col, gold_is_human1_col}, {0, 0, 0, 0, 0, 1, 1, 1, 1, 1}};

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

CUDF_TEST_PROGRAM_MAIN()
