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
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/join.hpp>

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/table_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>

template <typename T>
using column_wrapper = cudf::test::fixed_width_column_wrapper<T>;
using strcol_wrapper = cudf::test::strings_column_wrapper;
using CVector     = std::vector<std::unique_ptr<cudf::column>>;
using Table       = cudf::experimental::table;

struct JoinTest : public cudf::test::BaseFixture {};

//TEST_F(JoinTest, FullJoinNoNulls)
//TEST_F(JoinTest, LeftJoinNoNulls)
//TEST_F(JoinTest, InnerJoinNoNulls)

//TEST_F(JoinTest, FullJoinWithNulls)
//TEST_F(JoinTest, LeftJoinWithNulls)
//TEST_F(JoinTest, InnerJoinWithNulls)

/*
 *
 * Full Join :
 *
 * | c0 c1 ca |     | c0 c1 cb |
 * |  3 s0  0 |     |  2 s1  1 | 
 * |  1 s1  1 |     |  2 s0  0 | 
 * |  2 s2  2 |  ⟗  |  0 s1  1 | 
 * |  0 s4  4 |     |  4 s2  2 | 
 * |  3 s1  1 |     |  3 s1  1 | 
 * 
 *  =
 * 
 * | c0 c1 ca cb |
 * |  2 s1 na  1 |
 * |  2 s0 na  0 |
 * |  0 s1 na  1 |
 * |  4 s2 na  2 |
 * |  3 s1  1  1 |
 * |  3 s0  0 na |
 * |  1 s1  1 na |
 * |  2 s2  2 na |
 * |  0 s4  4 na |
 *
 */


TEST_F(JoinTest, FullJoin)
{
  column_wrapper <int32_t> col0_0{{3, 1, 2, 0, 3}};
  strcol_wrapper           col0_1({"s0", "s1", "s2", "s4", "s1"});
  column_wrapper <int32_t> col0_2{{0, 1, 2, 4, 1}};

  column_wrapper <int32_t> col1_0{{2, 2, 0, 4, 3}};
  strcol_wrapper           col1_1{{"s1", "s0", "s1", "s2", "s1"}};
  column_wrapper <int32_t> col1_2{{1, 0, 1, 2, 1}};

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols0.push_back(col0_2.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());
  cols1.push_back(col1_2.release());

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  auto join_table = cudf::full_join(t0, t1, {0, 1}, {0, 1}, {{0, 0}, {1,1}});
  for (auto&c : join_table->view()) {
    cudf::test::print(c); std::cout<<"\n";
  }
}

TEST_F(JoinTest, LeftJoin)
{
  column_wrapper <int32_t> col0_0{{3, 1, 2, 0, 3}};
  column_wrapper <int32_t> col0_1{{0, 1, 2, 4, 1}};

  column_wrapper <int32_t> col1_0{{2, 2, 0, 4, 3}};
  column_wrapper <int32_t> col1_1{{1, 0, 1, 2, 1}};

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  auto join_table = cudf::left_join(t0, t1, {0, 1}, {0, 1}, {{0, 0}, {1,1}});
  cudf::test::print(join_table->get_column(0)); std::cout<<"\n";
  cudf::test::print(join_table->get_column(1)); std::cout<<"\n";
}

TEST_F(JoinTest, InnerJoin)
{
  column_wrapper <int32_t> col0_0{{3, 1, 2, 0, 3}};
  column_wrapper <int32_t> col0_1{{0, 1, 2, 4, 1}};

  column_wrapper <int32_t> col1_0{{2, 2, 0, 4, 3}};
  column_wrapper <int32_t> col1_1{{1, 0, 1, 2, 1}};

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  auto join_table = cudf::inner_join(t0, t1, {0, 1}, {0, 1}, {{0, 0}, {1,1}});
  cudf::test::print(join_table->get_column(0)); std::cout<<"\n";
  cudf::test::print(join_table->get_column(1)); std::cout<<"\n";
}
