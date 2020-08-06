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

#include <cudf/ast/ast.cuh>
#include <cudf/ast/operators.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <simt/type_traits>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/table_utilities.hpp>
#include <tests/utilities/type_lists.hpp>
#include <type_traits>

template <typename T>
using column_wrapper = cudf::test::fixed_width_column_wrapper<T>;

struct ASTTest : public cudf::test::BaseFixture {
};

TEST_F(ASTTest, BasicAddition)
{
  auto c_0   = column_wrapper<int32_t>{3, 20, 1, 50};
  auto c_1   = column_wrapper<int32_t>{10, 7, 20, 0};
  auto table = cudf::table_view{{c_0, c_1}};

  auto col_ref_0 = cudf::ast::column_reference(0);
  auto col_ref_1 = cudf::ast::column_reference(1);
  auto expression =
    cudf::ast::binary_expression(cudf::ast::ast_operator::ADD, col_ref_0, col_ref_1);

  auto expected = column_wrapper<int32_t>{13, 27, 21, 50};
  auto result   = cudf::ast::compute_column(table, expression);

  cudf::test::expect_columns_equal(expected, result->view(), true);
}

TEST_F(ASTTest, LessComparator)
{
  auto c_0   = column_wrapper<int32_t>{3, 20, 1, 50};
  auto c_1   = column_wrapper<int32_t>{10, 7, 20, 0};
  auto table = cudf::table_view{{c_0, c_1}};

  auto col_ref_0 = cudf::ast::column_reference(0);
  auto col_ref_1 = cudf::ast::column_reference(1);
  auto expression =
    cudf::ast::binary_expression(cudf::ast::ast_operator::LESS, col_ref_0, col_ref_1);

  auto expected = column_wrapper<bool>{true, false, true, false};
  auto result   = cudf::ast::compute_column(table, expression);

  cudf::test::expect_columns_equal(expected, result->view(), true);
}

TEST_F(ASTTest, MultiLevelTreeArithmetic)
{
  auto c_0   = column_wrapper<int32_t>{3, 20, 1, 50};
  auto c_1   = column_wrapper<int32_t>{10, 7, 20, 0};
  auto c_2   = column_wrapper<int32_t>{-3, 66, 2, -99};
  auto table = cudf::table_view{{c_0, c_1, c_2}};

  auto col_ref_0 = cudf::ast::column_reference(0);
  auto col_ref_1 = cudf::ast::column_reference(1);
  auto col_ref_2 = cudf::ast::column_reference(2);

  auto expression_left_subtree =
    cudf::ast::binary_expression(cudf::ast::ast_operator::ADD, col_ref_0, col_ref_1);

  auto expression_right_subtree =
    cudf::ast::binary_expression(cudf::ast::ast_operator::SUB, col_ref_2, col_ref_0);

  auto expression_tree = cudf::ast::binary_expression(
    cudf::ast::ast_operator::ADD, expression_left_subtree, expression_right_subtree);

  auto result   = cudf::ast::compute_column(table, expression_tree);
  auto expected = column_wrapper<int32_t>{7, 73, 22, -99};

  cudf::test::expect_columns_equal(expected, result->view(), true);
}

TEST_F(ASTTest, ImbalancedTreeArithmetic)
{
  auto c_0   = column_wrapper<double>{0.15, 0.37, 4.2, 21.3};
  auto c_1   = column_wrapper<double>{0.0, -42.0, 1.0, 98.6};
  auto c_2   = column_wrapper<double>{0.6, std::numeric_limits<double>::infinity(), 0.999, 1.0};
  auto table = cudf::table_view{{c_0, c_1, c_2}};

  auto col_ref_0 = cudf::ast::column_reference(0);
  auto col_ref_1 = cudf::ast::column_reference(1);
  auto col_ref_2 = cudf::ast::column_reference(2);

  auto expression_right_subtree =
    cudf::ast::binary_expression(cudf::ast::ast_operator::MUL, col_ref_0, col_ref_1);

  auto expression_tree =
    cudf::ast::binary_expression(cudf::ast::ast_operator::SUB, col_ref_2, expression_right_subtree);

  auto result = cudf::ast::compute_column(table, expression_tree);
  auto expected =
    column_wrapper<double>{0.6, std::numeric_limits<double>::infinity(), -3.201, -2099.18};

  cudf::test::expect_columns_equal(expected, result->view(), true);
}

TEST_F(ASTTest, MultiLevelTreeComparator)
{
  auto c_0   = column_wrapper<int32_t>{3, 20, 1, 50};
  auto c_1   = column_wrapper<int32_t>{10, 7, 20, 0};
  auto c_2   = column_wrapper<int32_t>{-3, 66, 2, -99};
  auto table = cudf::table_view{{c_0, c_1, c_2}};

  auto col_ref_0 = cudf::ast::column_reference(0);
  auto col_ref_1 = cudf::ast::column_reference(1);
  auto col_ref_2 = cudf::ast::column_reference(2);

  auto expression_left_subtree =
    cudf::ast::binary_expression(cudf::ast::ast_operator::GREATER_EQUAL, col_ref_0, col_ref_1);

  auto expression_right_subtree =
    cudf::ast::binary_expression(cudf::ast::ast_operator::GREATER, col_ref_2, col_ref_0);

  auto expression_tree = cudf::ast::binary_expression(
    cudf::ast::ast_operator::LOGICAL_AND, expression_left_subtree, expression_right_subtree);

  auto result   = cudf::ast::compute_column(table, expression_tree);
  auto expected = column_wrapper<bool>{false, true, false, false};

  cudf::test::expect_columns_equal(expected, result->view(), true);
}

TEST_F(ASTTest, LiteralComparison)
{
  auto c_0   = column_wrapper<int32_t>{3, 20, 1, 50};
  auto table = cudf::table_view{{c_0}};

  auto col_ref_0     = cudf::ast::column_reference(0);
  auto literal_value = cudf::numeric_scalar<int32_t>(41);
  auto literal       = cudf::ast::literal(literal_value);

  auto expression =
    cudf::ast::binary_expression(cudf::ast::ast_operator::GREATER, col_ref_0, literal);

  auto result   = cudf::ast::compute_column(table, expression);
  auto expected = column_wrapper<bool>{false, false, false, true};

  cudf::test::expect_columns_equal(expected, result->view(), true);
}

struct custom_functor {
  template <typename OperatorFunctor,
            typename LHS,
            typename RHS,
            typename Out = simt::std::invoke_result_t<OperatorFunctor, LHS, RHS>,
            std::enable_if_t<cudf::ast::is_valid_binary_op<OperatorFunctor, LHS, RHS>>* = nullptr>
  CUDA_HOST_DEVICE_CALLABLE decltype(auto) operator()(int* result)
  {
    *result = 42;
  }

  template <typename OperatorFunctor,
            typename LHS,
            typename RHS,
            typename Out                                                                 = void,
            std::enable_if_t<!cudf::ast::is_valid_binary_op<OperatorFunctor, LHS, RHS>>* = nullptr>
  CUDA_HOST_DEVICE_CALLABLE decltype(auto) operator()(int* result)
  {
  }
};

TEST_F(ASTTest, CustomASTFunctor)
{
  int result = 0;
  cudf::ast::ast_operator_dispatcher(cudf::ast::ast_operator::ADD,
                                     cudf::data_type(cudf::type_id::INT32),
                                     cudf::data_type(cudf::type_id::INT32),
                                     custom_functor{},
                                     &result);
  EXPECT_EQ(result, 42);
}

CUDF_TEST_PROGRAM_MAIN()
