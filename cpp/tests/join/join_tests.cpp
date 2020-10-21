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
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/join.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>
#include "cudf/utilities/error.hpp"

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
  CUDF_TEST_EXPECT_TABLES_EQUAL(*sorted_gold, *sorted_result);
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

  CUDF_TEST_EXPECT_TABLES_EQUAL(*sorted_gold, *sorted_result);
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
  CUDF_TEST_EXPECT_TABLES_EQUAL(*sorted_gold, *sorted_result);
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
  CUDF_TEST_EXPECT_TABLES_EQUAL(*sorted_gold, *sorted_result);
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

  CUDF_TEST_EXPECT_TABLES_EQUAL(*sorted_gold, *sorted_result);

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

  CUDF_TEST_EXPECT_TABLES_EQUAL(*sorted_gold, *sorted_result);
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
  CUDF_TEST_EXPECT_TABLES_EQUAL(*sorted_gold, *sorted_result);
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
  CUDF_TEST_EXPECT_TABLES_EQUAL(*sorted_gold, *sorted_result);
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

  CUDF_TEST_EXPECT_TABLES_EQUAL(*sorted_gold, *sorted_result);

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

  CUDF_TEST_EXPECT_TABLES_EQUAL(*sorted_gold, *sorted_result);
}

TEST_F(JoinTest, InnerJoinSizeOverflow)
{
  auto zero = cudf::make_numeric_scalar(cudf::data_type(cudf::type_id::INT32));
  zero->set_valid(true);
  static_cast<cudf::scalar_type_t<int32_t> *>(zero.get())->set_value(0);

  // Should cause size overflow, raise exception
  int32_t left  = 4;
  int32_t right = 1073741825;

  auto col0_0 = cudf::make_column_from_scalar(*zero, left);
  auto col1_0 = cudf::make_column_from_scalar(*zero, right);

  CVector cols0, cols1;
  cols0.push_back(std::move(col0_0));
  cols1.push_back(std::move(col1_0));

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  EXPECT_THROW(cudf::inner_join(t0, t1, {0}, {0}, {{0, 0}}), cudf::logic_error);
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
  CUDF_TEST_EXPECT_TABLES_EQUAL(*sorted_gold, *sorted_result);
}

TEST_F(JoinTest, InnerJoinNonAlignedCommon)
{
  CVector cols0, cols1;
  cols0.emplace_back(column_wrapper<int32_t>{{3, 1, 2, 0, 2}}.release());
  cols0.emplace_back(column_wrapper<int32_t>{{3, 1, 2, 0, 2}}.release());
  cols0.emplace_back(strcol_wrapper({"s1", "s1", "s0", "s4", "s0"}).release());
  cols0.emplace_back(column_wrapper<int32_t>{{0, 1, 2, 4, 1}}.release());
  cols1.emplace_back(column_wrapper<int32_t>{{2, 2, 0, 4, 3}}.release());
  cols1.emplace_back(strcol_wrapper({"s1", "s0", "s1", "s2", "s1"}).release());
  cols1.emplace_back(column_wrapper<int32_t>{{1, 0, 1, 2, 1}}.release());

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  auto result            = cudf::inner_join(t0, t1, {1, 2}, {0, 1}, {{1, 0}, {2, 1}});
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

  CVector cols_gold;
  cols_gold.emplace_back(column_wrapper<int32_t>{{3, 2, 2}}.release());
  cols_gold.emplace_back(column_wrapper<int32_t>{{3, 2, 2}}.release());
  cols_gold.emplace_back(strcol_wrapper({"s1", "s0", "s0"}).release());
  cols_gold.emplace_back(column_wrapper<int32_t>{{0, 2, 1}}.release());
  cols_gold.emplace_back(column_wrapper<int32_t>{{1, 0, 0}}.release());
  Table gold(std::move(cols_gold));

  auto gold_sort_order = cudf::sorted_order(gold.view());
  auto sorted_gold     = cudf::gather(gold.view(), *gold_sort_order);
  CUDF_TEST_EXPECT_TABLES_EQUAL(*sorted_gold, *sorted_result);
}

TEST_F(JoinTest, InnerJoinNonAlignedCommonSwap)
{
  CVector cols0, cols1;
  cols0.emplace_back(column_wrapper<int32_t>{{3, 1, 2, 0, 2}}.release());
  cols0.emplace_back(column_wrapper<int32_t>{{3, 1, 2, 0, 2}}.release());
  cols0.emplace_back(strcol_wrapper({"s1", "s1", "s0", "s4", "s0"}).release());
  cols0.emplace_back(column_wrapper<int32_t>{{0, 1, 2, 4, 1}}.release());
  cols1.emplace_back(column_wrapper<int32_t>{{2, 2, 0, 4, 3, 5}}.release());
  cols1.emplace_back(strcol_wrapper({"s1", "s0", "s1", "s2", "s1", "s0"}).release());
  cols1.emplace_back(column_wrapper<int32_t>{{1, 0, 1, 2, 1, 0}}.release());

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  auto result            = cudf::inner_join(t0, t1, {1, 2}, {0, 1}, {{1, 0}, {2, 1}});
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

  CVector cols_gold;
  cols_gold.emplace_back(column_wrapper<int32_t>{{3, 2, 2}}.release());
  cols_gold.emplace_back(column_wrapper<int32_t>{{3, 2, 2}}.release());
  cols_gold.emplace_back(strcol_wrapper({"s1", "s0", "s0"}).release());
  cols_gold.emplace_back(column_wrapper<int32_t>{{0, 2, 1}}.release());
  cols_gold.emplace_back(column_wrapper<int32_t>{{1, 0, 0}}.release());
  Table gold(std::move(cols_gold));

  auto gold_sort_order = cudf::sorted_order(gold.view());
  auto sorted_gold     = cudf::gather(gold.view(), *gold_sort_order);
  CUDF_TEST_EXPECT_TABLES_EQUAL(*sorted_gold, *sorted_result);
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
  CUDF_TEST_EXPECT_TABLES_EQUAL(*sorted_gold, *sorted_result);
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
  CUDF_TEST_EXPECT_TABLES_EQUAL(*sorted_gold, *sorted_result);
  
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
  CUDF_TEST_EXPECT_TABLES_EQUAL(*sorted_gold, *sorted_result);
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
  CUDF_TEST_EXPECT_TABLES_EQUAL(empty0, *result);
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
  CUDF_TEST_EXPECT_TABLES_EQUAL(empty0, *result);
}

TEST_F(JoinTest, EmptyLeftTableLeftJoinNonAlignedCommon)
{
  column_wrapper<int32_t> col0_0;

  column_wrapper<int32_t> col1_0{{2, 2, 0, 4, 3}};
  column_wrapper<int32_t> col1_1{{1, 0, 1, 2, 1}, {1, 0, 1, 1, 1}};

  CVector cols0, cols1;
  cols0.emplace_back(col0_0.release());
  cols1.emplace_back(col1_0.release());
  cols1.emplace_back(col1_1.release());

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  column_wrapper<int32_t> col_gold_0;
  column_wrapper<int32_t> col_gold_1;

  CVector cols_gold;
  cols_gold.emplace_back(col_gold_0.release());
  cols_gold.emplace_back(col_gold_1.release());

  Table gold(std::move(cols_gold));

  auto result = cudf::left_join(t0, t1, {0}, {1}, {{0, 1}});
  CUDF_TEST_EXPECT_TABLES_EQUAL(gold, *result);
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
  CUDF_TEST_EXPECT_TABLES_EQUAL(t1, *result);
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
  CUDF_TEST_EXPECT_TABLES_EQUAL(empty1, *result);
}

TEST_F(JoinTest, EmptyRightTableInnerJoinNonAlignedCommon)
{
  column_wrapper<int32_t> col0_0{{2, 2, 0, 4, 3}};
  column_wrapper<int32_t> col0_1{{1, 0, 1, 2, 1}, {1, 0, 1, 1, 1}};

  column_wrapper<int32_t> col1_0;

  CVector cols0, cols1;
  cols0.emplace_back(col0_0.release());
  cols0.emplace_back(col0_1.release());
  cols1.emplace_back(col1_0.release());

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  column_wrapper<int32_t> col_gold_0;
  column_wrapper<int32_t> col_gold_1;

  CVector cols_gold;
  cols_gold.emplace_back(col_gold_0.release());
  cols_gold.emplace_back(col_gold_1.release());

  Table gold(std::move(cols_gold));

  auto result = cudf::inner_join(t0, t1, {1}, {0}, {{1, 0}});
  CUDF_TEST_EXPECT_TABLES_EQUAL(gold, *result);
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
  CUDF_TEST_EXPECT_TABLES_EQUAL(t0, *result);
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
  CUDF_TEST_EXPECT_TABLES_EQUAL(t0, *result);
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
  CUDF_TEST_EXPECT_TABLES_EQUAL(empty1, *result);
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
  CUDF_TEST_EXPECT_TABLES_EQUAL(empty1, *result);
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
  CUDF_TEST_EXPECT_TABLES_EQUAL(empty1, *result);
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

  CUDF_TEST_EXPECT_TABLES_EQUAL(gold, *result);
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

  CUDF_TEST_EXPECT_TABLES_EQUAL(gold, *result);
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

  CUDF_TEST_EXPECT_TABLES_EQUAL(gold, *result);
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
  CUDF_TEST_EXPECT_TABLES_EQUAL(*sorted_gold, *sorted_result);
}

TEST_F(JoinTest, HashJoinSequentialProbes)
{
  CVector cols1;
  cols1.emplace_back(column_wrapper<int32_t>{{2, 2, 0, 4, 3}}.release());
  cols1.emplace_back(strcol_wrapper{{"s1", "s0", "s1", "s2", "s1"}}.release());
  cols1.emplace_back(column_wrapper<int32_t>{{1, 0, 1, 2, 1}}.release());

  Table t1(std::move(cols1));

  cudf::hash_join hash_join(t1, {0, 1});

  {
    CVector cols0;
    cols0.emplace_back(column_wrapper<int32_t>{{3, 1, 2, 0, 3}}.release());
    cols0.emplace_back(strcol_wrapper({"s0", "s1", "s2", "s4", "s1"}).release());
    cols0.emplace_back(column_wrapper<int32_t>{{0, 1, 2, 4, 1}}.release());

    Table t0(std::move(cols0));

    auto result            = hash_join.full_join(t0, {0, 1}, {{0, 0}, {1, 1}});
    auto result_sort_order = cudf::sorted_order(result->view());
    auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

    CVector cols_gold;
    cols_gold.emplace_back(column_wrapper<int32_t>{{2, 2, 0, 4, 3, 3, 1, 2, 0}}.release());
    cols_gold.emplace_back(
      strcol_wrapper({"s1", "s0", "s1", "s2", "s1", "s0", "s1", "s2", "s4"}).release());
    cols_gold.emplace_back(
      column_wrapper<int32_t>{{-1, -1, -1, -1, 1, 0, 1, 2, 4}, {0, 0, 0, 0, 1, 1, 1, 1, 1}}
        .release());
    cols_gold.emplace_back(
      column_wrapper<int32_t>{{1, 0, 1, 2, 1, -1, -1, -1, -1}, {1, 1, 1, 1, 1, 0, 0, 0, 0}}
        .release());
    Table gold(std::move(cols_gold));

    auto gold_sort_order = cudf::sorted_order(gold.view());
    auto sorted_gold     = cudf::gather(gold.view(), *gold_sort_order);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*sorted_gold, *sorted_result);
  }

  {
    CVector cols0;
    cols0.emplace_back(column_wrapper<int32_t>{{3, 1, 2, 0, 3}}.release());
    cols0.emplace_back(strcol_wrapper({"s0", "s1", "s2", "s4", "s1"}).release());
    cols0.emplace_back(column_wrapper<int32_t>{{0, 1, 2, 4, 1}}.release());

    Table t0(std::move(cols0));

    auto result            = hash_join.left_join(t0, {0, 1}, {{0, 0}, {1, 1}});
    auto result_sort_order = cudf::sorted_order(result->view());
    auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

    CVector cols_gold;
    cols_gold.emplace_back(column_wrapper<int32_t>{{3, 3, 1, 2, 0}, {1, 1, 1, 1, 1}}.release());
    cols_gold.emplace_back(
      strcol_wrapper({"s1", "s0", "s1", "s2", "s4"}, {1, 1, 1, 1, 1, 1}).release());
    cols_gold.emplace_back(column_wrapper<int32_t>{{1, 0, 1, 2, 4}, {1, 1, 1, 1, 1}}.release());
    cols_gold.emplace_back(column_wrapper<int32_t>{{1, -1, -1, -1, -1}, {1, 0, 0, 0, 0}}.release());
    Table gold(std::move(cols_gold));

    auto gold_sort_order = cudf::sorted_order(gold.view());
    auto sorted_gold     = cudf::gather(gold.view(), *gold_sort_order);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*sorted_gold, *sorted_result);
  }

  {
    CVector cols0;
    cols0.emplace_back(column_wrapper<int32_t>{{3, 1, 2, 0, 2}}.release());
    cols0.emplace_back(column_wrapper<int32_t>{{3, 1, 2, 0, 2}}.release());
    cols0.emplace_back(strcol_wrapper({"s1", "s1", "s0", "s4", "s0"}).release());
    cols0.emplace_back(column_wrapper<int32_t>{{0, 1, 2, 4, 1}}.release());

    Table t0(std::move(cols0));

    auto probe_build_pair = hash_join.inner_join(t0, {1, 2}, {{1, 0}, {2, 1}});
    auto joined_cols      = probe_build_pair.first->release();
    auto build_cols       = probe_build_pair.second->release();
    joined_cols.insert(joined_cols.end(),
                       std::make_move_iterator(build_cols.begin()),
                       std::make_move_iterator(build_cols.end()));
    auto result            = std::make_unique<cudf::table>(std::move(joined_cols));
    auto result_sort_order = cudf::sorted_order(result->view());
    auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

    CVector cols_gold;
    cols_gold.emplace_back(column_wrapper<int32_t>{{3, 2, 2}}.release());
    cols_gold.emplace_back(column_wrapper<int32_t>{{3, 2, 2}}.release());
    cols_gold.emplace_back(strcol_wrapper({"s1", "s0", "s0"}).release());
    cols_gold.emplace_back(column_wrapper<int32_t>{{0, 2, 1}}.release());
    cols_gold.emplace_back(column_wrapper<int32_t>{{1, 0, 0}}.release());
    Table gold(std::move(cols_gold));

    auto gold_sort_order = cudf::sorted_order(gold.view());
    auto sorted_gold     = cudf::gather(gold.view(), *gold_sort_order);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*sorted_gold, *sorted_result);
  }

  {
    CVector cols0;
    cols0.emplace_back(column_wrapper<int32_t>{{3, 1, 2, 0, 2}}.release());
    cols0.emplace_back(column_wrapper<int32_t>{{3, 1, 2, 0, 2}}.release());
    cols0.emplace_back(strcol_wrapper({"s1", "s1", "s0", "s4", "s0"}).release());
    cols0.emplace_back(column_wrapper<int32_t>{{0, 1, 2, 4, 1}}.release());

    Table t0(std::move(cols0));

    auto probe_build_pair = hash_join.inner_join(
      t0, {1, 2}, {{1, 0}, {2, 1}}, cudf::hash_join::common_columns_output_side::BUILD);
    auto joined_cols = probe_build_pair.second->release();
    auto probe_cols  = probe_build_pair.first->release();
    joined_cols.insert(joined_cols.end(),
                       std::make_move_iterator(probe_cols.begin()),
                       std::make_move_iterator(probe_cols.end()));
    auto result            = std::make_unique<cudf::table>(std::move(joined_cols));
    auto result_sort_order = cudf::sorted_order(result->view());
    auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

    CVector cols_gold;
    cols_gold.emplace_back(column_wrapper<int32_t>{{3, 2, 2}}.release());
    cols_gold.emplace_back(strcol_wrapper({"s1", "s0", "s0"}).release());
    cols_gold.emplace_back(column_wrapper<int32_t>{{1, 0, 0}}.release());
    cols_gold.emplace_back(column_wrapper<int32_t>{{3, 2, 2}}.release());
    cols_gold.emplace_back(column_wrapper<int32_t>{{0, 2, 1}}.release());
    Table gold(std::move(cols_gold));

    auto gold_sort_order = cudf::sorted_order(gold.view());
    auto sorted_gold     = cudf::gather(gold.view(), *gold_sort_order);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*sorted_gold, *sorted_result);
  }
}

CUDF_TEST_PROGRAM_MAIN()
