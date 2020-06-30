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
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/table_utilities.hpp>
#include <tests/utilities/type_lists.hpp>
#include "cudf/ast/operators.cuh"
#include "cudf/scalar/scalar.hpp"
#include "cudf/scalar/scalar_factories.hpp"

template <typename T>
using column_wrapper = cudf::test::fixed_width_column_wrapper<T>;

struct ASTTest : public cudf::test::BaseFixture {
};

TEST_F(ASTTest, BasicASTEvaluation)
{
  auto a_0 = column_wrapper<int32_t>{3, 20, 1, 50};
  auto a_1 = column_wrapper<int32_t>{10, 7, 20, 0};
  auto a_2 = column_wrapper<int32_t>{-3, 66, 2, -99};

  auto b_0 = column_wrapper<int32_t>{2, 1, 5};
  auto b_1 = column_wrapper<int32_t>{7, 0, 4};

  auto expect_add  = column_wrapper<int32_t>{13, 27, 21, 50};
  auto expect_less = column_wrapper<bool>{true, false, true, false};

  auto table_a = cudf::table_view{{a_0, a_1, a_2}};
  auto table_b = cudf::table_view{{b_0, b_1}};

  auto col_ref_a_0   = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);
  auto col_ref_a_1   = cudf::ast::column_reference(1, cudf::ast::table_reference::LEFT);
  auto col_ref_a_2   = cudf::ast::column_reference(2, cudf::ast::table_reference::LEFT);
  auto literal_value = cudf::numeric_scalar<int32_t>(42);
  auto literal       = cudf::ast::literal(literal_value);
  auto expression_add =
    cudf::ast::binary_expression(cudf::ast::ast_operator::ADD, col_ref_a_0, col_ref_a_1);
  auto expression_less =
    cudf::ast::binary_expression(cudf::ast::ast_operator::LESS, col_ref_a_0, col_ref_a_1);

  auto expression_tree_1_1 =
    cudf::ast::binary_expression(cudf::ast::ast_operator::ADD, col_ref_a_0, col_ref_a_1);

  auto expression_tree_1_2 =
    cudf::ast::binary_expression(cudf::ast::ast_operator::SUB, col_ref_a_2, literal);

  auto expression_tree_1 = cudf::ast::binary_expression(
    cudf::ast::ast_operator::MUL, expression_tree_1_1, expression_tree_1_2);

  auto result_add    = cudf::ast::compute_column(table_a, expression_add);
  auto result_less   = cudf::ast::compute_column(table_a, expression_less);
  auto result_tree_1 = cudf::ast::compute_column(table_a, expression_tree_1);

  // cudf::test::expect_columns_equal(expect_add, result_add->view(), true);
  // cudf::test::expect_columns_equal(expect_less, result_less->view(), true);
}

CUDF_TEST_PROGRAM_MAIN()
