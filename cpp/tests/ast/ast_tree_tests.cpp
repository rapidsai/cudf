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

#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/ast/expressions.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>

template <typename T>
using column_wrapper = cudf::test::fixed_width_column_wrapper<T>;

TEST(AstTreeTest, ExpressionTree)
{
  namespace ast   = cudf::ast;
  using op        = ast::ast_operator;
  using operation = ast::operation;

  // computes (y = mx + c)... and linearly interpolates them using interpolator t
  auto m0_col = column_wrapper<float>{10, 20, 50, 100};
  auto x0_col = column_wrapper<float>{10, 5, 2, 1};
  auto c0_col = column_wrapper<float>{100, 100, 100, 100};

  auto m1_col = column_wrapper<float>{10, 20, 50, 100};
  auto x1_col = column_wrapper<float>{20, 10, 4, 2};
  auto c1_col = column_wrapper<float>{200, 200, 200, 200};

  auto one_scalar = cudf::numeric_scalar<float>{1};
  auto t_scalar   = cudf::numeric_scalar<float>{0.5F};

  auto table = cudf::table_view{{m0_col, x0_col, c0_col, m1_col, x1_col, c1_col}};

  ast::tree tree{};

  auto const& one = tree.push(ast::literal{one_scalar});
  auto const& t   = tree.push(ast::literal{t_scalar});
  auto const& m0  = tree.push(ast::column_reference(0));
  auto const& x0  = tree.push(ast::column_reference(1));
  auto const& c0  = tree.push(ast::column_reference(2));
  auto const& m1  = tree.push(ast::column_reference(3));
  auto const& x1  = tree.push(ast::column_reference(4));
  auto const& c1  = tree.push(ast::column_reference(5));

  // compute: y0 = m0 x0 + c0
  auto const& y0 = tree.push(operation{op::ADD, tree.push(operation{op::MUL, m0, x0}), c0});

  // compute: y1 = m1 x1 + c1
  auto const& y1 = tree.push(operation{op::ADD, tree.push(operation{op::MUL, m1, x1}), c1});

  // compute weighted: (1 - t) * y0
  auto const& y0_w = tree.push(operation{op::MUL, tree.push(operation{op::SUB, one, t}), y0});

  // compute weighted: y = t * y1
  auto const& y1_w = tree.push(operation{op::MUL, t, y1});

  // add weighted: result = lerp(y0, y1, t) = (1 - t) * y0 + t * y1
  auto result = cudf::compute_column(table, tree.push(operation{op::ADD, y0_w, y1_w}));

  auto expected = column_wrapper<float>{300, 300, 300, 300};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}
