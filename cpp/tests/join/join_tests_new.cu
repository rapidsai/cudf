/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <execinfo.h>

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/join_new.hpp>
#include <cudf/copying.hpp>
#include <cudf/sorting.hpp>

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/table_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/type_lists.hpp>

using cudf::test::fixed_width_column_wrapper;
using cudf::test::strings_column_wrapper;
using cudf::experimental::table;
using cudf::join::join_comparison_operator;

struct JoinTest : public cudf::test::BaseFixture {};

#if 0
TEST_F(JoinTest, InvalidCommonColumnIndices)
{
  fixed_width_column_wrapper<int32_t> col0_0{3, 1, 2, 0, 3};
  fixed_width_column_wrapper<int32_t> col0_1{0, 1, 2, 4, 1};

  fixed_width_column_wrapper<int32_t> col1_0{2, 2, 0, 4, 3};
  fixed_width_column_wrapper<int32_t> col1_1{1, 0, 1, 2, 1};

  std::vector<std::unique_ptr<cudf::column>>  cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());

  table t0(std::move(cols0));
  table t1(std::move(cols1));

  //
  // NOTE:  Not sure this should be a logic error.  columns_in_common - as implemented in both old and current join
  //        is INDEPENDENT of the join specification.  While this could happen accidentally (and not checking it
  //        would leave the user susecptible to such a careless error) it is not clear to me that specifying a columns_in_common
  //        pair that is not part of the join operation is logically incorrect behavior.
  //
  //        Now... if the types don't match then the join will throw a logic_error.  Perhaps we should repurpose this test.
  //
  std::vector<cudf::join::join_operation> join_ops{ { join_comparison_operator::EQUAL, 0, 0}, { join_comparison_operator::EQUAL, 1, 1} };

  std::vector<std::pair<cudf::size_type, cudf::size_type>> columns_in_common{{0, 1}, {1, 0}};

  
  EXPECT_THROW (cudf::join::inner_join::nested_loop(t0, t1, join_ops, columns_in_common),
                cudf::logic_error);
}
#endif

TEST_F(JoinTest, FullJoinNoNulls)
{
  fixed_width_column_wrapper <int32_t> col0_0{{3, 1, 2, 0, 3}};
  strings_column_wrapper               col0_1({"s0", "s1", "s2", "s4", "s1"});
  fixed_width_column_wrapper <int32_t> col0_2{{0, 1, 2, 4, 1}};

  fixed_width_column_wrapper <int32_t> col1_0{{2, 2, 0, 4, 3}};
  strings_column_wrapper               col1_1{{"s1", "s0", "s1", "s2", "s1"}};
  fixed_width_column_wrapper <int32_t> col1_2{{1, 0, 1, 2, 1}};

  std::vector<std::unique_ptr<cudf::column>>  cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols0.push_back(col0_2.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());
  cols1.push_back(col1_2.release());

  table t0(std::move(cols0));
  table t1(std::move(cols1));

  std::vector<cudf::join::join_operation> join_ops{ { join_comparison_operator::EQUAL, 0, 0}, { join_comparison_operator::EQUAL, 1, 1} };
  std::vector<std::pair<cudf::size_type, cudf::size_type>> columns_in_common{{0, 0}, {1, 1}};

  auto result = cudf::join::full_join::nested_loop(t0, t1, join_ops, columns_in_common);
  auto result_sort_order = cudf::experimental::sorted_order(result->view());
  auto sorted_result = cudf::experimental::gather(result->view(), *result_sort_order);

  fixed_width_column_wrapper <int32_t> col_gold_0{{2, 2, 0, 4, 3, 3, 1, 2, 0}};
  strings_column_wrapper               col_gold_1({"s1", "s0", "s1", "s2", "s1", "s0", "s1", "s2", "s4"});
  fixed_width_column_wrapper <int32_t> col_gold_2{{-1, -1, -1, -1, 1, 0, 1, 2, 4}, {0, 0, 0, 0, 1, 1, 1, 1, 1}};
  fixed_width_column_wrapper <int32_t> col_gold_3{{ 1, 0, 1, 2, 1, -1, -1, -1, -1}, {1, 1, 1, 1, 1, 0, 0, 0, 0}};

  std::vector<std::unique_ptr<cudf::column>>  cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());

    table gold(std::move(cols_gold));

  auto gold_sort_order = cudf::experimental::sorted_order(gold.view());
  auto sorted_gold = cudf::experimental::gather(gold.view(), *gold_sort_order);
  cudf::test::expect_tables_equal(*sorted_gold, *sorted_result);
}

TEST_F(JoinTest, FullJoinWithNulls)
{
  fixed_width_column_wrapper <int32_t> col0_0{{3, 1, 2, 0, 3}};
  strings_column_wrapper               col0_1({"s0", "s1", "s2", "s4", "s1"});
  fixed_width_column_wrapper <int32_t> col0_2{{0, 1, 2, 4, 1}};

  fixed_width_column_wrapper <int32_t> col1_0{{2, 2, 0, 4, 3}, {1, 1, 1, 0, 1}};
  strings_column_wrapper               col1_1{{"s1", "s0", "s1", "s2", "s1"}};
  fixed_width_column_wrapper <int32_t> col1_2{{1, 0, 1, 2, 1}};

  std::vector<std::unique_ptr<cudf::column>>  cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols0.push_back(col0_2.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());
  cols1.push_back(col1_2.release());

  table t0(std::move(cols0));
  table t1(std::move(cols1));

  std::vector<cudf::join::join_operation> join_ops{ { join_comparison_operator::EQUAL, 0, 0}, { join_comparison_operator::EQUAL, 1, 1} };
  std::vector<std::pair<cudf::size_type, cudf::size_type>> columns_in_common{{0, 0}, {1, 1}};

  auto result = cudf::join::full_join::nested_loop(t0, t1, join_ops, columns_in_common);
  auto result_sort_order = cudf::experimental::sorted_order(result->view());
  auto sorted_result = cudf::experimental::gather(result->view(), *result_sort_order);

  fixed_width_column_wrapper <int32_t> col_gold_0{{2, 2, 0, -1, 3, 3, 1, 2, 0}, {1, 1, 1, 0, 1, 1, 1, 1, 1}};
  strings_column_wrapper               col_gold_1({"s1", "s0", "s1", "s2", "s1", "s0", "s1", "s2", "s4"});
  fixed_width_column_wrapper <int32_t> col_gold_2{{-1, -1, -1, -1, 1, 0, 1, 2, 4}, {0, 0, 0, 0, 1, 1, 1, 1, 1}};
  fixed_width_column_wrapper <int32_t> col_gold_3{{ 1, 0, 1, 2, 1, -1, -1, -1, -1}, {1, 1, 1, 1, 1, 0, 0, 0, 0}};

  std::vector<std::unique_ptr<cudf::column>>  cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());

  table gold(std::move(cols_gold));

  auto gold_sort_order = cudf::experimental::sorted_order(gold.view());
  auto sorted_gold = cudf::experimental::gather(gold.view(), *gold_sort_order);
  cudf::test::expect_tables_equal(*sorted_gold, *sorted_result);
}

TEST_F(JoinTest, LeftJoinNoNulls)
{
  fixed_width_column_wrapper <int32_t> col0_0{{3, 1, 2, 0, 3}};
  strings_column_wrapper               col0_1({"s0", "s1", "s2", "s4", "s1"});
  fixed_width_column_wrapper <int32_t> col0_2{{0, 1, 2, 4, 1}};

  fixed_width_column_wrapper <int32_t> col1_0{{2, 2, 0, 4, 3}};
  strings_column_wrapper               col1_1({"s1", "s0", "s1", "s2", "s1"});
  fixed_width_column_wrapper <int32_t> col1_2{{1, 0, 1, 2, 1}};

    std::vector<std::unique_ptr<cudf::column>>  cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols0.push_back(col0_2.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());
  cols1.push_back(col1_2.release());

  table t0(std::move(cols0));
  table t1(std::move(cols1));

  std::vector<cudf::join::join_operation> join_ops{ { join_comparison_operator::EQUAL, 0, 0}, { join_comparison_operator::EQUAL, 1, 1} };
  std::vector<std::pair<cudf::size_type, cudf::size_type>> columns_in_common{{0, 0}, {1, 1}};

  auto result = cudf::join::left_join::nested_loop(t0, t1, join_ops, columns_in_common);
  auto result_sort_order = cudf::experimental::sorted_order(result->view());
  auto sorted_result = cudf::experimental::gather(result->view(), *result_sort_order);

  fixed_width_column_wrapper <int32_t> col_gold_0{{3, 3, 1, 2, 0}, {1, 1, 1, 1, 1}};
  strings_column_wrapper               col_gold_1({"s1", "s0", "s1", "s2", "s4"}, {1, 1, 1, 1, 1, 1});
  fixed_width_column_wrapper <int32_t> col_gold_2{{1, 0, 1, 2, 4}, {1, 1, 1, 1, 1}};
  fixed_width_column_wrapper <int32_t> col_gold_3{{1, -1, -1, -1, -1}, {1, 0, 0, 0, 0}};

  std::vector<std::unique_ptr<cudf::column>>  cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());

  table gold(std::move(cols_gold));

  auto gold_sort_order = cudf::experimental::sorted_order(gold.view());
  auto sorted_gold = cudf::experimental::gather(gold.view(), *gold_sort_order);
  cudf::test::expect_tables_equal(*sorted_gold, *sorted_result);
}

TEST_F(JoinTest, LeftJoinWithNulls)
{
  fixed_width_column_wrapper <int32_t> col0_0{{3, 1, 2, 0, 2}};
  strings_column_wrapper               col0_1({"s1", "s1", "s0", "s4", "s0"}, {1, 1, 0, 1, 1});
  fixed_width_column_wrapper <int32_t> col0_2{{0, 1, 2, 4, 1}};

  fixed_width_column_wrapper <int32_t> col1_0{{2, 2, 0, 4, 3}};
  strings_column_wrapper               col1_1({"s1", "s0", "s1", "s2", "s1"});
  fixed_width_column_wrapper <int32_t> col1_2{{1, 0, 1, 2, 1}, {1, 0, 1, 1, 1}};

  std::vector<std::unique_ptr<cudf::column>>  cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols0.push_back(col0_2.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());
  cols1.push_back(col1_2.release());

  table t0(std::move(cols0));
  table t1(std::move(cols1));

  std::vector<cudf::join::join_operation> join_ops{ { join_comparison_operator::EQUAL, 0, 0}, { join_comparison_operator::EQUAL, 1, 1} };
  std::vector<std::pair<cudf::size_type, cudf::size_type>> columns_in_common{{0, 0}, {1, 1}};

  auto result = cudf::join::left_join::nested_loop(t0, t1, join_ops, columns_in_common);
  auto result_sort_order = cudf::experimental::sorted_order(result->view());
  auto sorted_result = cudf::experimental::gather(result->view(), *result_sort_order);

  fixed_width_column_wrapper <int32_t> col_gold_0{{3, 2, 1, 2, 0}, {1, 1, 1, 1, 1}};
  strings_column_wrapper               col_gold_1({"s1", "s0", "s1", "", "s4"}, {1, 1, 1, 0, 1});
  fixed_width_column_wrapper <int32_t> col_gold_2{{0, 1, 1, 2, 4}, {1, 1, 1, 1, 1}};
  fixed_width_column_wrapper <int32_t> col_gold_3{{1, -1, -1, -1, -1}, {1, 0, 0, 0, 0}};

  std::vector<std::unique_ptr<cudf::column>>  cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());

  table gold(std::move(cols_gold));

  auto gold_sort_order = cudf::experimental::sorted_order(gold.view());
  auto sorted_gold = cudf::experimental::gather(gold.view(), *gold_sort_order);
  cudf::test::expect_tables_equal(*sorted_gold, *sorted_result);
}

TEST_F(JoinTest, InnerJoinNoNulls)
{
  fixed_width_column_wrapper <int32_t> col0_0{{3, 1, 2, 0, 2}};
  strings_column_wrapper               col0_1({"s1", "s1", "s0", "s4", "s0"});
  fixed_width_column_wrapper <int32_t> col0_2{{0, 1, 2, 4, 1}};

  fixed_width_column_wrapper <int32_t> col1_0{{2, 2, 0, 4, 3}};
  strings_column_wrapper               col1_1({"s1", "s0", "s1", "s2", "s1"});
  fixed_width_column_wrapper <int32_t> col1_2{{1, 0, 1, 2, 1}};

  std::vector<std::unique_ptr<cudf::column>>  cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols0.push_back(col0_2.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());
  cols1.push_back(col1_2.release());

  table t0(std::move(cols0));
  table t1(std::move(cols1));

  std::vector<cudf::join::join_operation> join_ops{ { join_comparison_operator::EQUAL, 0, 0}, { join_comparison_operator::EQUAL, 1, 1} };
  std::vector<std::pair<cudf::size_type, cudf::size_type>> columns_in_common{{0, 0}, {1, 1}};

  auto result = cudf::join::inner_join::nested_loop(t0, t1, join_ops, columns_in_common);
  auto result_sort_order = cudf::experimental::sorted_order(result->view());
  auto sorted_result = cudf::experimental::gather(result->view(), *result_sort_order);

  fixed_width_column_wrapper <int32_t> col_gold_0{{3, 2, 2}};
  strings_column_wrapper               col_gold_1({"s1", "s0", "s0"});
  fixed_width_column_wrapper <int32_t> col_gold_2{{0, 2, 1}};
  fixed_width_column_wrapper <int32_t> col_gold_3{{1, 0, 0}};

  std::vector<std::unique_ptr<cudf::column>>  cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());

  table gold(std::move(cols_gold));

  auto gold_sort_order = cudf::experimental::sorted_order(gold.view());
  auto sorted_gold = cudf::experimental::gather(gold.view(), *gold_sort_order);
  cudf::test::expect_tables_equal(*sorted_gold, *sorted_result);
}

TEST_F(JoinTest, InnerJoinWithNulls)
{
  fixed_width_column_wrapper <int32_t> col0_0{{3, 1, 2, 0, 2}};
  strings_column_wrapper               col0_1({"s1", "s1", "s0", "s4", "s0"}, {1, 1, 0, 1, 1});
  fixed_width_column_wrapper <int32_t> col0_2{{0, 1, 2, 4, 1}};

  fixed_width_column_wrapper <int32_t> col1_0{{2, 2, 0, 4, 3}};
  strings_column_wrapper               col1_1({"s1", "s0", "s1", "s2", "s1"});
  fixed_width_column_wrapper <int32_t> col1_2{{1, 0, 1, 2, 1}, {1, 0, 1, 1, 1}};

  std::vector<std::unique_ptr<cudf::column>>  cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols0.push_back(col0_2.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());
  cols1.push_back(col1_2.release());

  table t0(std::move(cols0));
  table t1(std::move(cols1));

  std::vector<cudf::join::join_operation> join_ops{ { join_comparison_operator::EQUAL, 0, 0}, { join_comparison_operator::EQUAL, 1, 1} };
  std::vector<std::pair<cudf::size_type, cudf::size_type>> columns_in_common{{0, 0}, {1, 1}};

  auto result = cudf::join::inner_join::nested_loop(t0, t1, join_ops, columns_in_common);
  auto result_sort_order = cudf::experimental::sorted_order(result->view());
  auto sorted_result = cudf::experimental::gather(result->view(), *result_sort_order);

  fixed_width_column_wrapper <int32_t> col_gold_0{{3, 2}};
  strings_column_wrapper               col_gold_1({"s1", "s0"}, {1, 1});
  fixed_width_column_wrapper <int32_t> col_gold_2{{0, 1}};
  fixed_width_column_wrapper <int32_t> col_gold_3{{1, -1}, {1, 0}};

  std::vector<std::unique_ptr<cudf::column>>  cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());

  table gold(std::move(cols_gold));

  auto gold_sort_order = cudf::experimental::sorted_order(gold.view());
  auto sorted_gold = cudf::experimental::gather(gold.view(), *gold_sort_order);
  cudf::test::expect_tables_equal(*sorted_gold, *sorted_result);
}

//Empty Left Table
TEST_F(JoinTest, EmptyLeftTableInnerJoin)
{
  fixed_width_column_wrapper <int32_t> col0_0;
  fixed_width_column_wrapper <int32_t> col0_1;

  fixed_width_column_wrapper <int32_t> col1_0{{2, 2, 0, 4, 3}};
  fixed_width_column_wrapper <int32_t> col1_1{{1, 0, 1, 2, 1}, {1, 0, 1, 1, 1}};

  std::vector<std::unique_ptr<cudf::column>>  cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());

  table empty0(std::move(cols0));
  table t1(std::move(cols1));

  std::vector<cudf::join::join_operation> join_ops{ { join_comparison_operator::EQUAL, 0, 0}, { join_comparison_operator::EQUAL, 1, 1} };
  std::vector<std::pair<cudf::size_type, cudf::size_type>> columns_in_common{{0, 0}, {1, 1}};

  auto result = cudf::join::inner_join::nested_loop(empty0, t1, join_ops, columns_in_common);
  cudf::test::expect_tables_equal(empty0, *result);
}

TEST_F(JoinTest, EmptyLeftTableLeftJoin)
{
  fixed_width_column_wrapper <int32_t> col0_0;
  fixed_width_column_wrapper <int32_t> col0_1;

  fixed_width_column_wrapper <int32_t> col1_0{{2, 2, 0, 4, 3}};
  fixed_width_column_wrapper <int32_t> col1_1{{1, 0, 1, 2, 1}, {1, 0, 1, 1, 1}};

  std::vector<std::unique_ptr<cudf::column>>  cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());

  table empty0(std::move(cols0));
  table t1(std::move(cols1));

  std::vector<cudf::join::join_operation> join_ops{ { join_comparison_operator::EQUAL, 0, 0}, { join_comparison_operator::EQUAL, 1, 1} };
  std::vector<std::pair<cudf::size_type, cudf::size_type>> columns_in_common{{0, 0}, {1, 1}};

  auto result = cudf::join::left_join::nested_loop(empty0, t1, join_ops, columns_in_common);
  cudf::test::expect_tables_equal(empty0, *result);
}

TEST_F(JoinTest, EmptyLeftTableFullJoin)
{
  fixed_width_column_wrapper <int32_t> col0_0;
  fixed_width_column_wrapper <int32_t> col0_1;

  fixed_width_column_wrapper <int32_t> col1_0{{2, 2, 0, 4, 3}};
  fixed_width_column_wrapper <int32_t> col1_1{{1, 0, 1, 2, 1}, {1, 0, 1, 1, 1}};

  std::vector<std::unique_ptr<cudf::column>>  cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());

  table empty0(std::move(cols0));
  table t1(std::move(cols1));

  std::vector<cudf::join::join_operation> join_ops{ { join_comparison_operator::EQUAL, 0, 0}, { join_comparison_operator::EQUAL, 1, 1} };
  std::vector<std::pair<cudf::size_type, cudf::size_type>> columns_in_common{{0, 0}, {1, 1}};

  auto result = cudf::join::full_join::nested_loop(empty0, t1, join_ops, columns_in_common);
  cudf::test::expect_tables_equal(t1, *result);
}

//Empty Right Table
TEST_F(JoinTest, EmptyRightTableInnerJoin)
{
  fixed_width_column_wrapper <int32_t> col0_0{{2, 2, 0, 4, 3}};
  fixed_width_column_wrapper <int32_t> col0_1{{1, 0, 1, 2, 1}, {1, 0, 1, 1, 1}};

  fixed_width_column_wrapper <int32_t> col1_0;
  fixed_width_column_wrapper <int32_t> col1_1;

  std::vector<std::unique_ptr<cudf::column>>  cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());

  table t0(std::move(cols0));
  table empty1(std::move(cols1));

  std::vector<cudf::join::join_operation> join_ops{ { join_comparison_operator::EQUAL, 0, 0}, { join_comparison_operator::EQUAL, 1, 1} };
  std::vector<std::pair<cudf::size_type, cudf::size_type>> columns_in_common{{0, 0}, {1, 1}};

  auto result = cudf::join::inner_join::nested_loop(t0, empty1, join_ops, columns_in_common);
  cudf::test::expect_tables_equal(empty1, *result);
}

TEST_F(JoinTest, EmptyRightTableLeftJoin)
{
  fixed_width_column_wrapper <int32_t> col0_0{{2, 2, 0, 4, 3}, {1, 1, 1, 1, 1}};
  fixed_width_column_wrapper <int32_t> col0_1{{1, 0, 1, 2, 1}, {1, 0, 1, 1, 1}};

  fixed_width_column_wrapper <int32_t> col1_0;
  fixed_width_column_wrapper <int32_t> col1_1;

  std::vector<std::unique_ptr<cudf::column>>  cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());

  table t0(std::move(cols0));
  table empty1(std::move(cols1));

  std::vector<cudf::join::join_operation> join_ops{ { join_comparison_operator::EQUAL, 0, 0}, { join_comparison_operator::EQUAL, 1, 1} };
  std::vector<std::pair<cudf::size_type, cudf::size_type>> columns_in_common{{0, 0}, {1, 1}};

  auto result = cudf::join::left_join::nested_loop(t0, empty1, join_ops, columns_in_common);
  cudf::test::expect_tables_equal(t0, *result);
}

TEST_F(JoinTest, EmptyRightTableFullJoin)
{
  fixed_width_column_wrapper <int32_t> col0_0{{2, 2, 0, 4, 3}};
  fixed_width_column_wrapper <int32_t> col0_1{{1, 0, 1, 2, 1}, {1, 0, 1, 1, 1}};

  fixed_width_column_wrapper <int32_t> col1_0;
  fixed_width_column_wrapper <int32_t> col1_1;

  std::vector<std::unique_ptr<cudf::column>>  cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());

  table t0(std::move(cols0));
  table empty1(std::move(cols1));

  std::vector<cudf::join::join_operation> join_ops{ { join_comparison_operator::EQUAL, 0, 0}, { join_comparison_operator::EQUAL, 1, 1} };
  std::vector<std::pair<cudf::size_type, cudf::size_type>> columns_in_common{{0, 0}, {1, 1}};

  auto result = cudf::join::full_join::nested_loop(t0, empty1, join_ops, columns_in_common);
  cudf::test::expect_tables_equal(t0, *result);
}

//Both tables empty
TEST_F(JoinTest, BothEmptyInnerJoin)
{
  fixed_width_column_wrapper <int32_t> col0_0;
  fixed_width_column_wrapper <int32_t> col0_1;

  fixed_width_column_wrapper <int32_t> col1_0;
  fixed_width_column_wrapper <int32_t> col1_1;

  std::vector<std::unique_ptr<cudf::column>>  cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());

  table t0(std::move(cols0));
  table empty1(std::move(cols1));

  std::vector<cudf::join::join_operation> join_ops{ { join_comparison_operator::EQUAL, 0, 0}, { join_comparison_operator::EQUAL, 1, 1} };
  std::vector<std::pair<cudf::size_type, cudf::size_type>> columns_in_common{{0, 0}, {1, 1}};

  auto result = cudf::join::inner_join::nested_loop(t0, empty1, join_ops, columns_in_common);
  cudf::test::expect_tables_equal(empty1, *result);
}

TEST_F(JoinTest, BothEmptyLeftJoin)
{
  fixed_width_column_wrapper <int32_t> col0_0;
  fixed_width_column_wrapper <int32_t> col0_1;

  fixed_width_column_wrapper <int32_t> col1_0;
  fixed_width_column_wrapper <int32_t> col1_1;

  std::vector<std::unique_ptr<cudf::column>>  cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());

  table t0(std::move(cols0));
  table empty1(std::move(cols1));

  std::vector<cudf::join::join_operation> join_ops{ { join_comparison_operator::EQUAL, 0, 0}, { join_comparison_operator::EQUAL, 1, 1} };
  std::vector<std::pair<cudf::size_type, cudf::size_type>> columns_in_common{{0, 0}, {1, 1}};

  auto result = cudf::join::left_join::nested_loop(t0, empty1, join_ops, columns_in_common);
  cudf::test::expect_tables_equal(empty1, *result);
}

TEST_F(JoinTest, BothEmptyFullJoin)
{
  fixed_width_column_wrapper <int32_t> col0_0;
  fixed_width_column_wrapper <int32_t> col0_1;

  fixed_width_column_wrapper <int32_t> col1_0;
  fixed_width_column_wrapper <int32_t> col1_1;

  std::vector<std::unique_ptr<cudf::column>>  cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());

  table t0(std::move(cols0));
  table empty1(std::move(cols1));

  std::vector<cudf::join::join_operation> join_ops{ { join_comparison_operator::EQUAL, 0, 0}, { join_comparison_operator::EQUAL, 1, 1} };
  std::vector<std::pair<cudf::size_type, cudf::size_type>> columns_in_common{{0, 0}, {1, 1}};

  auto result = cudf::join::full_join::nested_loop(t0, empty1, join_ops, columns_in_common);
  cudf::test::expect_tables_equal(empty1, *result);
}

//EqualValues X Inner,Left,Full

TEST_F(JoinTest, EqualValuesInnerJoin)
{
  fixed_width_column_wrapper <int32_t> col0_0{{0, 0}};
  strings_column_wrapper               col0_1({"s0", "s0"});

  fixed_width_column_wrapper <int32_t> col1_0{{0, 0}};
  strings_column_wrapper               col1_1({"s0", "s0"});

  std::vector<std::unique_ptr<cudf::column>>  cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());

  table t0(std::move(cols0));
  table t1(std::move(cols1));

  std::vector<cudf::join::join_operation> join_ops{ { join_comparison_operator::EQUAL, 0, 0}, { join_comparison_operator::EQUAL, 1, 1} };
  std::vector<std::pair<cudf::size_type, cudf::size_type>> columns_in_common{{0, 0}, {1, 1}};

  auto result = cudf::join::inner_join::nested_loop(t0, t1, join_ops, columns_in_common);

  fixed_width_column_wrapper <int32_t> col_gold_0{{0, 0, 0, 0}};
  strings_column_wrapper               col_gold_1({"s0", "s0", "s0", "s0"});

  std::vector<std::unique_ptr<cudf::column>>  cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());

  table gold(std::move(cols_gold));

  cudf::test::expect_tables_equal(gold, *result);
}

TEST_F(JoinTest, EqualValuesLeftJoin)
{
  fixed_width_column_wrapper <int32_t> col0_0{{0, 0}};
  strings_column_wrapper               col0_1({"s0", "s0"});

  fixed_width_column_wrapper <int32_t> col1_0{{0, 0}};
  strings_column_wrapper               col1_1({"s0", "s0"});

  std::vector<std::unique_ptr<cudf::column>>  cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());

  table t0(std::move(cols0));
  table t1(std::move(cols1));

  std::vector<cudf::join::join_operation> join_ops{ { join_comparison_operator::EQUAL, 0, 0}, { join_comparison_operator::EQUAL, 1, 1} };
  std::vector<std::pair<cudf::size_type, cudf::size_type>> columns_in_common{{0, 0}, {1, 1}};

  auto result = cudf::join::left_join::nested_loop(t0, t1, join_ops, columns_in_common);

  fixed_width_column_wrapper <int32_t> col_gold_0{{0, 0, 0, 0}, {1, 1, 1, 1}};
  strings_column_wrapper               col_gold_1({"s0", "s0", "s0", "s0"}, {1, 1, 1, 1});

  std::vector<std::unique_ptr<cudf::column>>  cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());

  table gold(std::move(cols_gold));

  cudf::test::expect_tables_equal(gold, *result);
}

TEST_F(JoinTest, EqualValuesFullJoin)
{
  fixed_width_column_wrapper <int32_t> col0_0{{0, 0}};
  strings_column_wrapper               col0_1({"s0", "s0"});

  fixed_width_column_wrapper <int32_t> col1_0{{0, 0}};
  strings_column_wrapper               col1_1({"s0", "s0"});

  std::vector<std::unique_ptr<cudf::column>>  cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());

  table t0(std::move(cols0));
  table t1(std::move(cols1));

  std::vector<cudf::join::join_operation> join_ops{ { join_comparison_operator::EQUAL, 0, 0}, { join_comparison_operator::EQUAL, 1, 1} };
  std::vector<std::pair<cudf::size_type, cudf::size_type>> columns_in_common{{0, 0}, {1, 1}};

  auto result = cudf::join::full_join::nested_loop(t0, t1, join_ops, columns_in_common);

  fixed_width_column_wrapper <int32_t> col_gold_0{{0, 0, 0, 0}};
  strings_column_wrapper               col_gold_1({"s0", "s0", "s0", "s0"});

  std::vector<std::unique_ptr<cudf::column>>  cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());

  table gold(std::move(cols_gold));

  cudf::test::expect_tables_equal(gold, *result);
}

TEST_F(JoinTest, InnerJoinCornerCase)
{
  fixed_width_column_wrapper <int64_t> col0_0{{4, 1, 3, 2, 2, 2, 2}};
  fixed_width_column_wrapper <int64_t> col1_0{{2}};

  std::vector<std::unique_ptr<cudf::column>>  cols0, cols1;
  cols0.push_back(col0_0.release());
  cols1.push_back(col1_0.release());

  table t0(std::move(cols0));
  table t1(std::move(cols1));

  std::vector<cudf::join::join_operation> join_ops{ { join_comparison_operator::EQUAL, 0, 0} };
  std::vector<std::pair<cudf::size_type, cudf::size_type>> columns_in_common{{0, 0}};

  auto result = cudf::join::inner_join::nested_loop(t0, t1, join_ops, columns_in_common);

  auto result_sort_order = cudf::experimental::sorted_order(result->view());
  auto sorted_result = cudf::experimental::gather(result->view(), *result_sort_order);

  fixed_width_column_wrapper <int64_t> col_gold_0{{2, 2, 2, 2}};

  std::vector<std::unique_ptr<cudf::column>>  cols_gold;
  cols_gold.push_back(col_gold_0.release());

  table gold(std::move(cols_gold));

  auto gold_sort_order = cudf::experimental::sorted_order(gold.view());
  auto sorted_gold = cudf::experimental::gather(gold.view(), *gold_sort_order);
  cudf::test::expect_tables_equal(*sorted_gold, *sorted_result);
}
