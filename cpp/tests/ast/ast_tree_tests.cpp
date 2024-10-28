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
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/ast/expressions.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>

TEST_F(AstTreeTest, ExpressionTree)
{
  // compute y = mx + c, across multiple columns and apply weights to them and then fold them
  auto a     = column_wrapper<int32_t>{3, 20, 1, 50};
  auto b     = column_wrapper<int32_t>{10, 7, 20, 0};
  auto c     = column_wrapper<int32_t>{10, 7, 20, 0};
  auto d     = column_wrapper<float>{10, 7, 20, 0};
  auto e     = column_wrapper<int32_t>{10, 7, 20, 0};
  auto f     = column_wrapper<float>{10, 7, 20, 0};
  auto table = cudf::table_view{{a, b, c, d, e, f}};
  cudf::ast::tree tree;

  auto const& a_ref   = tree.push(cudf::ast::column_reference(0));
  auto const& b_ref   = tree.push(cudf::ast::column_reference(1));
  auto const& c_ref   = tree.push(cudf::ast::column_reference(2));
  auto const& d_ref   = tree.push(cudf::ast::column_reference(3));
  auto const& e_ref   = tree.push(cudf::ast::column_reference(4));
  auto const& f_ref   = tree.push(cudf::ast::column_reference(5));
  auto const& literal = tree.push(cudf::ast::literal{255});

  /// compute: (a + b) - c
  auto const& op_0 = tree.push(cudf::ast::operation{
    cudf::ast::ast_operator::SUBTRACT,
    tree.push(cudf::ast::operation{cudf::ast::ast_operator::ADD, a_ref, b_ref}),
    c_ref});

  auto const& op_1 = tree.push(cudf::ast::operation{
    cudf::ast::ast_operator::MULTIPLY,
    tree.push(cudf::ast::operation{cudf::ast::ast_operator::SUBTRACR, d_ref, e_ref}),
    e_ref});

  auto result = cudf::compute_column(
    table, tree.push(cudf::ast::operation{cudf::ast::ast_operator::ADD, op_0, op_1}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}
