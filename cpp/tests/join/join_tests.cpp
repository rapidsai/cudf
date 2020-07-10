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

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/join.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/table_utilities.hpp>
#include <tests/utilities/type_lists.hpp>
#include "cudf/types.hpp"

template <typename T>
using column_wrapper = cudf::test::fixed_width_column_wrapper<T>;
using strcol_wrapper = cudf::test::strings_column_wrapper;
using CVector        = std::vector<std::unique_ptr<cudf::column>>;
using Table          = cudf::table;

struct JoinTest : public cudf::test::BaseFixture {
};

TEST_F(JoinTest, InvalidCommonColumnIndices)
{
  column_wrapper<int32_t> col0_0{{3, 1, 2, 0, 3}};
  column_wrapper<int32_t> col0_1{{0, 1, 2, 4, 1}};

  column_wrapper<int32_t> col1_0{{2, 2, 0, 4, 3}};
  column_wrapper<int32_t> col1_1{{1, 0, 1, 2, 1}};

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  EXPECT_THROW(cudf::inner_join(t0, t1, {0, 1}, {0, 1}, {{0, 1}, {1, 0}}), cudf::logic_error);
}

TEST_F(JoinTest, FullJoinNoCommon)
{
  column_wrapper<int32_t> col0_0{{0, 1}};
  column_wrapper<int32_t> col1_0{{0, 2}};
  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols1.push_back(col1_0.release());

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  column_wrapper<int32_t> exp_col0_0{{0, 1, -1}, {1, 1, 0}};
  column_wrapper<int32_t> exp_col0_1{{0, -1, 2}, {1, 0, 1}};
  CVector exp_cols;
  exp_cols.push_back(exp_col0_0.release());
  exp_cols.push_back(exp_col0_1.release());
  Table gold(std::move(exp_cols));

  auto result            = cudf::full_join(t0, t1, {0}, {0}, {});
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

  auto gold_sort_order = cudf::sorted_order(gold.view());
  auto sorted_gold     = cudf::gather(gold.view(), *gold_sort_order);
  cudf::test::expect_tables_equal(*sorted_gold, *sorted_result);
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

  auto result            = cudf::left_join(t0, t1, {0}, {0}, {});
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

  cudf::test::expect_tables_equal(*sorted_gold, *sorted_result);
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

  auto result            = cudf::full_join(t0, t1, {0, 1}, {0, 1}, {{0, 0}, {1, 1}});
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

  column_wrapper<int32_t> col_gold_0{{2, 2, 0, 4, 3, 3, 1, 2, 0}};
  strcol_wrapper col_gold_1({"s1", "s0", "s1", "s2", "s1", "s0", "s1", "s2", "s4"});
  column_wrapper<int32_t> col_gold_2{{-1, -1, -1, -1, 1, 0, 1, 2, 4}, {0, 0, 0, 0, 1, 1, 1, 1, 1}};
  column_wrapper<int32_t> col_gold_3{{1, 0, 1, 2, 1, -1, -1, -1, -1}, {1, 1, 1, 1, 1, 0, 0, 0, 0}};
  CVector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());
  Table gold(std::move(cols_gold));

  auto gold_sort_order = cudf::sorted_order(gold.view());
  auto sorted_gold     = cudf::gather(gold.view(), *gold_sort_order);
  cudf::test::expect_tables_equal(*sorted_gold, *sorted_result);
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

  auto result            = cudf::full_join(t0, t1, {0, 1}, {0, 1}, {{0, 0}, {1, 1}});
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

  column_wrapper<int32_t> col_gold_0{{2, 2, 0, -1, 3, 3, 1, 2, 0}, {1, 1, 1, 0, 1, 1, 1, 1, 1}};
  strcol_wrapper col_gold_1({"s1", "s0", "s1", "s2", "s1", "s0", "s1", "s2", "s4"});
  column_wrapper<int32_t> col_gold_2{{-1, -1, -1, -1, 1, 0, 1, 2, 4}, {0, 0, 0, 0, 1, 1, 1, 1, 1}};
  column_wrapper<int32_t> col_gold_3{{1, 0, 1, 2, 1, -1, -1, -1, -1}, {1, 1, 1, 1, 1, 0, 0, 0, 0}};
  CVector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());
  Table gold(std::move(cols_gold));

  auto gold_sort_order = cudf::sorted_order(gold.view());
  auto sorted_gold     = cudf::gather(gold.view(), *gold_sort_order);
  cudf::test::expect_tables_equal(*sorted_gold, *sorted_result);
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

  auto result            = cudf::full_join(t0, t1, {0, 1}, {0, 1}, {{0, 0}, {1, 1}});
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

#if 0
  std::cout << "Actual Results:\n";
  cudf::test::print(sorted_result->get_column(0).view(), std::cout, ",\t\t");
  cudf::test::print(sorted_result->get_column(1).view(), std::cout, ",\t\t");
  cudf::test::print(sorted_result->get_column(2).view(), std::cout, ",\t\t");
  cudf::test::print(sorted_result->get_column(3).view(), std::cout, ",\t\t");
#endif
 
  column_wrapper<int32_t> col_gold_0{{   2,    5,    3,    -1},
                                     {   1,    1,    1,     0}};
  strcol_wrapper          col_gold_1({ "s1", "s0", "s0",  "s1"});
  column_wrapper<int32_t> col_gold_2{{  -1,   -1,    0,     1}, 
                                     {   0,    0,    1,     1}};
  column_wrapper<int32_t> col_gold_3{{   1,    4,    2,     8}, 
                                     {   1,    1,    1,     1}};

  CVector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());
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

  cudf::test::expect_tables_equal(*sorted_gold, *sorted_result);

  // Repeat test with compare_nulls_equal=false,
  // as per SQL standard.

  result            = cudf::full_join(t0, t1, {0, 1}, {0, 1}, {{0, 0}, {1, 1}}, cudf::null_equality::UNEQUAL);
  result_sort_order = cudf::sorted_order(result->view());
  sorted_result     = cudf::gather(result->view(), *result_sort_order);

  col_gold_0 =               {{   2,    5,    3,    -1,   -1},
                              {   1,    1,    1,     0,    0}};
  col_gold_1 = strcol_wrapper({ "s1", "s0", "s0",  "s1", "s1"});
  col_gold_2 =               {{  -1,   -1,    0,    -1,    1}, 
                              {   0,    0,    1,     0,    1}};
  col_gold_3 =               {{   1,    4,    2,     8,   -1}, 
                              {   1,    1,    1,     1,    0}};

  // clang-format on

  CVector cols_gold_nulls_unequal;
  cols_gold_nulls_unequal.push_back(col_gold_0.release());
  cols_gold_nulls_unequal.push_back(col_gold_1.release());
  cols_gold_nulls_unequal.push_back(col_gold_2.release());
  cols_gold_nulls_unequal.push_back(col_gold_3.release());
  Table gold_nulls_unequal{std::move(cols_gold_nulls_unequal)};

  gold_sort_order = cudf::sorted_order(gold_nulls_unequal.view());
  sorted_gold     = cudf::gather(gold_nulls_unequal.view(), *gold_sort_order);

  cudf::test::expect_tables_equal(*sorted_gold, *sorted_result);
}

TEST_F(JoinTest, LeftJoinNoNulls)
{
  column_wrapper<int32_t> col0_0{{3, 1, 2, 0, 3}};
  strcol_wrapper col0_1({"s0", "s1", "s2", "s4", "s1"});
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

  auto result            = cudf::left_join(t0, t1, {0, 1}, {0, 1}, {{0, 0}, {1, 1}});
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

  column_wrapper<int32_t> col_gold_0{{3, 3, 1, 2, 0}, {1, 1, 1, 1, 1}};
  strcol_wrapper col_gold_1({"s1", "s0", "s1", "s2", "s4"}, {1, 1, 1, 1, 1, 1});
  column_wrapper<int32_t> col_gold_2{{1, 0, 1, 2, 4}, {1, 1, 1, 1, 1}};
  column_wrapper<int32_t> col_gold_3{{1, -1, -1, -1, -1}, {1, 0, 0, 0, 0}};
  CVector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());
  Table gold(std::move(cols_gold));

  auto gold_sort_order = cudf::sorted_order(gold.view());
  auto sorted_gold     = cudf::gather(gold.view(), *gold_sort_order);
  cudf::test::expect_tables_equal(*sorted_gold, *sorted_result);
}

TEST_F(JoinTest, LeftJoinWithNulls)
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

  auto result            = cudf::left_join(t0, t1, {0, 1}, {0, 1}, {{0, 0}, {1, 1}});
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

  column_wrapper<int32_t> col_gold_0{{3, 2, 1, 2, 0}, {1, 1, 1, 1, 1}};
  strcol_wrapper col_gold_1({"s1", "s0", "s1", "", "s4"}, {1, 1, 1, 0, 1});
  column_wrapper<int32_t> col_gold_2{{0, 1, 1, 2, 4}, {1, 1, 1, 1, 1}};
  column_wrapper<int32_t> col_gold_3{{1, -1, -1, -1, -1}, {1, 0, 0, 0, 0}};
  CVector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());
  Table gold(std::move(cols_gold));

  auto gold_sort_order = cudf::sorted_order(gold.view());
  auto sorted_gold     = cudf::gather(gold.view(), *gold_sort_order);
  cudf::test::expect_tables_equal(*sorted_gold, *sorted_result);
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

  auto result            = cudf::left_join(t0, t1, {0, 1}, {0, 1}, {{0, 0}, {1, 1}});
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
  column_wrapper<int32_t> col_gold_3{{   2,     8,   -1}, 
                                     {   1,     1,    0}};

  CVector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());
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

  cudf::test::expect_tables_equal(*sorted_gold, *sorted_result);

  // Repeat test with compare_nulls_equal=false,
  // as per SQL standard.

  result            = cudf::left_join(t0, t1, {0, 1}, {0, 1}, {{0, 0}, {1, 1}}, cudf::null_equality::UNEQUAL);
  result_sort_order = cudf::sorted_order(result->view());
  sorted_result     = cudf::gather(result->view(), *result_sort_order);

  col_gold_0 =               {{   3,    -1,    2},
                              {   1,     0,    1}};
  col_gold_1 = strcol_wrapper({ "s0",  "s1", "s2"},
                              {   1,     1,    1});
  col_gold_2 =               {{   0,     1,    2}, 
                              {   1,     1,    1}};
  col_gold_3 =               {{   2,    -1,   -1}, 
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

  cudf::test::expect_tables_equal(*sorted_gold, *sorted_result);
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

  auto result            = cudf::inner_join(t0, t1, {0, 1}, {0, 1}, {{0, 0}, {1, 1}});
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

  column_wrapper<int32_t> col_gold_0{{3, 2, 2}};
  strcol_wrapper col_gold_1({"s1", "s0", "s0"});
  column_wrapper<int32_t> col_gold_2{{0, 2, 1}};
  column_wrapper<int32_t> col_gold_3{{1, 0, 0}};
  CVector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());
  Table gold(std::move(cols_gold));

  auto gold_sort_order = cudf::sorted_order(gold.view());
  auto sorted_gold     = cudf::gather(gold.view(), *gold_sort_order);
  cudf::test::expect_tables_equal(*sorted_gold, *sorted_result);
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

  auto result            = cudf::inner_join(t0, t1, {0, 1}, {0, 1}, {{0, 0}, {1, 1}});
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

  column_wrapper<int32_t> col_gold_0{{3, 2}};
  strcol_wrapper col_gold_1({"s1", "s0"}, {1, 1});
  column_wrapper<int32_t> col_gold_2{{0, 1}};
  column_wrapper<int32_t> col_gold_3{{1, -1}, {1, 0}};
  CVector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());
  Table gold(std::move(cols_gold));

  auto gold_sort_order = cudf::sorted_order(gold.view());
  auto sorted_gold     = cudf::gather(gold.view(), *gold_sort_order);
  cudf::test::expect_tables_equal(*sorted_gold, *sorted_result);
}

// Test to check join behaviour when join keys are null.
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

  auto result            = cudf::inner_join(t0, t1, {0, 1}, {0, 1}, {{0, 0}, {1, 1}});
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

  column_wrapper<int32_t> col_gold_0 {{  3,    2}};
  strcol_wrapper          col_gold_1 ({"s1", "s0"}, 
                                      {  1,    0});
  column_wrapper<int32_t> col_gold_2{{   0,    2}};
  column_wrapper<int32_t> col_gold_3{{   1,    0}};
  CVector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());
  Table gold(std::move(cols_gold));

  auto gold_sort_order = cudf::sorted_order(gold.view());
  auto sorted_gold     = cudf::gather(gold.view(), *gold_sort_order);
  cudf::test::expect_tables_equal(*sorted_gold, *sorted_result);
  
  // Repeat test with compare_nulls_equal=false,
  // as per SQL standard.

  result            = cudf::inner_join(t0, t1, {0, 1}, {0, 1}, {{0, 0}, {1, 1}}, cudf::null_equality::UNEQUAL);
  result_sort_order = cudf::sorted_order(result->view());
  sorted_result     = cudf::gather(result->view(), *result_sort_order);

  col_gold_0 =               {{  3}};
  col_gold_1 = strcol_wrapper({"s1"}, 
                              {  1});
  col_gold_2 =               {{  0}};
  col_gold_3 =               {{  1}};

  // clang-format on

  CVector cols_gold_sql;
  cols_gold_sql.push_back(col_gold_0.release());
  cols_gold_sql.push_back(col_gold_1.release());
  cols_gold_sql.push_back(col_gold_2.release());
  cols_gold_sql.push_back(col_gold_3.release());
  Table gold_sql(std::move(cols_gold_sql));

  gold_sort_order = cudf::sorted_order(gold_sql.view());
  sorted_gold     = cudf::gather(gold_sql.view(), *gold_sort_order);
  cudf::test::expect_tables_equal(*sorted_gold, *sorted_result);
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

  auto result = cudf::inner_join(empty0, t1, {0, 1}, {0, 1}, {{0, 0}, {1, 1}});
  cudf::test::expect_tables_equal(empty0, *result);
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

  auto result = cudf::left_join(empty0, t1, {0, 1}, {0, 1}, {{0, 0}, {1, 1}});
  cudf::test::expect_tables_equal(empty0, *result);
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

  Table empty0(std::move(cols0));
  Table t1(std::move(cols1));

  auto result = cudf::full_join(empty0, t1, {0, 1}, {0, 1}, {{0, 0}, {1, 1}});
  cudf::test::expect_tables_equal(t1, *result);
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

  auto result = cudf::inner_join(t0, empty1, {0, 1}, {0, 1}, {{0, 0}, {1, 1}});
  cudf::test::expect_tables_equal(empty1, *result);
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

  auto result = cudf::left_join(t0, empty1, {0, 1}, {0, 1}, {{0, 0}, {1, 1}});
  cudf::test::expect_tables_equal(t0, *result);
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

  auto result = cudf::full_join(t0, empty1, {0, 1}, {0, 1}, {{0, 0}, {1, 1}});
  cudf::test::expect_tables_equal(t0, *result);
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

  auto result = cudf::inner_join(t0, empty1, {0, 1}, {0, 1}, {{0, 0}, {1, 1}});
  cudf::test::expect_tables_equal(empty1, *result);
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

  auto result = cudf::left_join(t0, empty1, {0, 1}, {0, 1}, {{0, 0}, {1, 1}});
  cudf::test::expect_tables_equal(empty1, *result);
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

  auto result = cudf::full_join(t0, empty1, {0, 1}, {0, 1}, {{0, 0}, {1, 1}});
  cudf::test::expect_tables_equal(empty1, *result);
}

// EqualValues X Inner,Left,Full

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

  auto result = cudf::inner_join(t0, t1, {0, 1}, {0, 1}, {{0, 0}, {1, 1}});

  column_wrapper<int32_t> col_gold_0{{0, 0, 0, 0}};
  strcol_wrapper col_gold_1({"s0", "s0", "s0", "s0"});
  CVector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  Table gold(std::move(cols_gold));

  cudf::test::expect_tables_equal(gold, *result);
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

  auto result = cudf::left_join(t0, t1, {0, 1}, {0, 1}, {{0, 0}, {1, 1}});

  column_wrapper<int32_t> col_gold_0{{0, 0, 0, 0}, {1, 1, 1, 1}};
  strcol_wrapper col_gold_1({"s0", "s0", "s0", "s0"}, {1, 1, 1, 1});
  CVector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  Table gold(std::move(cols_gold));

  cudf::test::expect_tables_equal(gold, *result);
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

  auto result = cudf::full_join(t0, t1, {0, 1}, {0, 1}, {{0, 0}, {1, 1}});

  column_wrapper<int32_t> col_gold_0{{0, 0, 0, 0}};
  strcol_wrapper col_gold_1({"s0", "s0", "s0", "s0"});
  CVector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  Table gold(std::move(cols_gold));

  cudf::test::expect_tables_equal(gold, *result);
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

  auto result            = cudf::inner_join(t0, t1, {0}, {0}, {{0, 0}});
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

  column_wrapper<int64_t> col_gold_0{{2, 2, 2, 2}};
  CVector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  Table gold(std::move(cols_gold));

  auto gold_sort_order = cudf::sorted_order(gold.view());
  auto sorted_gold     = cudf::gather(gold.view(), *gold_sort_order);
  cudf::test::expect_tables_equal(*sorted_gold, *sorted_result);
}

CUDF_TEST_PROGRAM_MAIN()
