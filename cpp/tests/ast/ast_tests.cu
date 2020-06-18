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

template <typename T>
using column_wrapper = cudf::test::fixed_width_column_wrapper<T>;

struct ASTTest : public cudf::test::BaseFixture {
};

TEST_F(ASTTest, BasicASTEvaluation)
{
  printf("Starting AST test.\n");
  auto a_0 = column_wrapper<int32_t>{10, 20, 20, 50};
  auto a_1 = column_wrapper<int32_t>{3, 7, 1, 0};

  auto b_0 = column_wrapper<int32_t>{2, 1, 5};
  auto b_1 = column_wrapper<int32_t>{7, 0, 4};

  auto expect_0 = column_wrapper<int32_t>{13, 27, 21, 50};

  auto table_a = cudf::table_view{{a_0, a_1}};
  auto table_b = cudf::table_view{{b_0, b_1}};
  printf("Tables created.\n");

  auto lhs = cudf::ast_expression_source{cudf::ast_data_source::COLUMN, 0};
  auto rhs = cudf::ast_expression_source{cudf::ast_data_source::COLUMN, 1};
  auto basic_expression =
    cudf::ast_binary_expression<int32_t>{cudf::ast_binary_operator::ADD, lhs, rhs};

  printf("Performing evaluation:\n");
  auto result_0 = cudf::compute_ast_column<int32_t>(table_a, basic_expression);

  cudf::test::expect_columns_equal(expect_0, result_0->view(), true);
}

CUDF_TEST_PROGRAM_MAIN()
