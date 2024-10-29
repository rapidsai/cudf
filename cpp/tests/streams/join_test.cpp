/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>

#include <cudf/column/column.hpp>
#include <cudf/join.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <cstdint>
#include <memory>
#include <optional>

class JoinTest : public cudf::test::BaseFixture {
  static inline cudf::table make_table()
  {
    cudf::test::fixed_width_column_wrapper<int32_t> col0{{3, 1, 2, 0, 3}};
    cudf::test::strings_column_wrapper col1{{"s0", "s1", "s2", "s4", "s1"}};
    cudf::test::fixed_width_column_wrapper<int32_t> col2{{0, 1, 2, 4, 1}};

    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(col0.release());
    columns.push_back(col1.release());
    columns.push_back(col2.release());

    return cudf::table{std::move(columns)};
  }

 public:
  cudf::table table0{make_table()};
  cudf::table table1{make_table()};
  cudf::table conditional0{make_table()};
  cudf::table conditional1{make_table()};
  cudf::ast::column_reference col_ref_left_0{0};
  cudf::ast::column_reference col_ref_right_0{0, cudf::ast::table_reference::RIGHT};
  cudf::ast::operation left_zero_eq_right_zero{
    cudf::ast::ast_operator::EQUAL, col_ref_left_0, col_ref_right_0};
};

TEST_F(JoinTest, InnerJoin)
{
  cudf::inner_join(table0, table1, cudf::null_equality::EQUAL, cudf::test::get_default_stream());
}

TEST_F(JoinTest, LeftJoin)
{
  cudf::left_join(table0, table1, cudf::null_equality::EQUAL, cudf::test::get_default_stream());
}

TEST_F(JoinTest, FullJoin)
{
  cudf::full_join(table0, table1, cudf::null_equality::EQUAL, cudf::test::get_default_stream());
}

TEST_F(JoinTest, LeftSemiJoin)
{
  cudf::left_semi_join(
    table0, table1, cudf::null_equality::EQUAL, cudf::test::get_default_stream());
}

TEST_F(JoinTest, LeftAntiJoin)
{
  cudf::left_anti_join(
    table0, table1, cudf::null_equality::EQUAL, cudf::test::get_default_stream());
}

TEST_F(JoinTest, CrossJoin) { cudf::cross_join(table0, table1, cudf::test::get_default_stream()); }

TEST_F(JoinTest, ConditionalInnerJoin)
{
  cudf::conditional_inner_join(
    table0, table1, left_zero_eq_right_zero, std::nullopt, cudf::test::get_default_stream());
}

TEST_F(JoinTest, ConditionalLeftJoin)
{
  cudf::conditional_left_join(
    table0, table1, left_zero_eq_right_zero, std::nullopt, cudf::test::get_default_stream());
}

TEST_F(JoinTest, ConditionalFullJoin)
{
  cudf::conditional_full_join(
    table0, table1, left_zero_eq_right_zero, cudf::test::get_default_stream());
}

TEST_F(JoinTest, ConditionalLeftSemiJoin)
{
  cudf::conditional_left_semi_join(
    table0, table1, left_zero_eq_right_zero, std::nullopt, cudf::test::get_default_stream());
}

TEST_F(JoinTest, ConditionalLeftAntiJoin)
{
  cudf::conditional_left_anti_join(
    table0, table1, left_zero_eq_right_zero, std::nullopt, cudf::test::get_default_stream());
}

TEST_F(JoinTest, MixedInnerJoin)
{
  cudf::mixed_inner_join(table0,
                         table1,
                         conditional0,
                         conditional1,
                         left_zero_eq_right_zero,
                         cudf::null_equality::EQUAL,
                         std::nullopt,
                         cudf::test::get_default_stream());
}

TEST_F(JoinTest, MixedLeftJoin)
{
  cudf::mixed_left_join(table0,
                        table1,
                        conditional0,
                        conditional1,
                        left_zero_eq_right_zero,
                        cudf::null_equality::EQUAL,
                        std::nullopt,
                        cudf::test::get_default_stream());
}

TEST_F(JoinTest, MixedFullJoin)
{
  cudf::mixed_full_join(table0,
                        table1,
                        conditional0,
                        conditional1,
                        left_zero_eq_right_zero,
                        cudf::null_equality::EQUAL,
                        std::nullopt,
                        cudf::test::get_default_stream());
}

TEST_F(JoinTest, MixedLeftSemiJoin)
{
  cudf::mixed_left_semi_join(table0,
                             table1,
                             conditional0,
                             conditional1,
                             left_zero_eq_right_zero,
                             cudf::null_equality::EQUAL,
                             cudf::test::get_default_stream());
}

TEST_F(JoinTest, MixedLeftAntiJoin)
{
  cudf::mixed_left_anti_join(table0,
                             table1,
                             conditional0,
                             conditional1,
                             left_zero_eq_right_zero,
                             cudf::null_equality::EQUAL,
                             cudf::test::get_default_stream());
}

TEST_F(JoinTest, MixedInnerJoinSize)
{
  cudf::mixed_inner_join_size(table0,
                              table1,
                              conditional0,
                              conditional1,
                              left_zero_eq_right_zero,
                              cudf::null_equality::EQUAL,
                              cudf::test::get_default_stream());
}

TEST_F(JoinTest, MixedLeftJoinSize)
{
  cudf::mixed_left_join_size(table0,
                             table1,
                             conditional0,
                             conditional1,
                             left_zero_eq_right_zero,
                             cudf::null_equality::EQUAL,
                             cudf::test::get_default_stream());
}

TEST_F(JoinTest, ConditionalInnerJoinSize)
{
  cudf::conditional_inner_join_size(
    table0, table1, left_zero_eq_right_zero, cudf::test::get_default_stream());
}

TEST_F(JoinTest, ConditionalLeftJoinSize)
{
  cudf::conditional_left_join_size(
    table0, table1, left_zero_eq_right_zero, cudf::test::get_default_stream());
}

TEST_F(JoinTest, ConditionalLeftSemiJoinSize)
{
  cudf::conditional_left_semi_join_size(
    table0, table1, left_zero_eq_right_zero, cudf::test::get_default_stream());
}

TEST_F(JoinTest, ConditionalLeftAntiJoinSize)
{
  cudf::conditional_left_anti_join_size(
    table0, table1, left_zero_eq_right_zero, cudf::test::get_default_stream());
}
