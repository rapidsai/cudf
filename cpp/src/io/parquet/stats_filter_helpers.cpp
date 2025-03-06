/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include "stats_filter_helpers.hpp"

#include "io/parquet/parquet_common.hpp"

#include <cudf/ast/detail/operators.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

namespace cudf::io::parquet::detail {

stats_expression_converter::stats_expression_converter(ast::expression const& expr,
                                                       size_type num_columns)
  : _num_columns{num_columns}
{
  expr.accept(*this);
}

std::reference_wrapper<ast::expression const> stats_expression_converter::visit(
  ast::literal const& expr)
{
  return expr;
}

std::reference_wrapper<ast::expression const> stats_expression_converter::visit(
  ast::column_reference const& expr)
{
  CUDF_EXPECTS(expr.get_table_source() == ast::table_reference::LEFT,
               "Statistics AST supports only left table");
  CUDF_EXPECTS(expr.get_column_index() < _num_columns,
               "Column index cannot be more than number of columns in the table");
  return expr;
}

std::reference_wrapper<ast::expression const> stats_expression_converter::visit(
  ast::column_name_reference const& expr)
{
  CUDF_FAIL("Column name reference is not supported in statistics AST");
}

std::reference_wrapper<ast::expression const> stats_expression_converter::visit(
  ast::operation const& expr)
{
  using cudf::ast::ast_operator;
  auto const operands = expr.get_operands();
  auto const op       = expr.get_operator();

  if (auto* v = dynamic_cast<ast::column_reference const*>(&operands[0].get())) {
    // First operand should be column reference, second should be literal.
    CUDF_EXPECTS(cudf::ast::detail::ast_operator_arity(op) == 2,
                 "Only binary operations are supported on column reference");
    CUDF_EXPECTS(dynamic_cast<ast::literal const*>(&operands[1].get()) != nullptr,
                 "Second operand of binary operation with column reference must be a literal");
    v->accept(*this);
    // Push literal into the ast::tree
    auto const& literal  = _stats_expr.push(*dynamic_cast<ast::literal const*>(&operands[1].get()));
    auto const col_index = v->get_column_index();
    switch (op) {
      /* transform to stats conditions. op(col, literal)
      col1 == val --> vmin <= val && vmax >= val
      col1 != val --> !(vmin == val && vmax == val)
      col1 >  val --> vmax > val
      col1 <  val --> vmin < val
      col1 >= val --> vmax >= val
      col1 <= val --> vmin <= val
      */
      case ast_operator::EQUAL: {
        auto const& vmin = _stats_expr.push(ast::column_reference{col_index * 2});
        auto const& vmax = _stats_expr.push(ast::column_reference{col_index * 2 + 1});
        _stats_expr.push(ast::operation{
          ast::ast_operator::LOGICAL_AND,
          _stats_expr.push(ast::operation{ast_operator::GREATER_EQUAL, vmax, literal}),
          _stats_expr.push(ast::operation{ast_operator::LESS_EQUAL, vmin, literal})});
        break;
      }
      case ast_operator::NOT_EQUAL: {
        auto const& vmin = _stats_expr.push(ast::column_reference{col_index * 2});
        auto const& vmax = _stats_expr.push(ast::column_reference{col_index * 2 + 1});
        _stats_expr.push(
          ast::operation{ast_operator::LOGICAL_OR,
                         _stats_expr.push(ast::operation{ast_operator::NOT_EQUAL, vmin, vmax}),
                         _stats_expr.push(ast::operation{ast_operator::NOT_EQUAL, vmax, literal})});
        break;
      }
      case ast_operator::LESS: [[fallthrough]];
      case ast_operator::LESS_EQUAL: {
        auto const& vmin = _stats_expr.push(ast::column_reference{col_index * 2});
        _stats_expr.push(ast::operation{op, vmin, literal});
        break;
      }
      case ast_operator::GREATER: [[fallthrough]];
      case ast_operator::GREATER_EQUAL: {
        auto const& vmax = _stats_expr.push(ast::column_reference{col_index * 2 + 1});
        _stats_expr.push(ast::operation{op, vmax, literal});
        break;
      }
      default: CUDF_FAIL("Unsupported operation in Statistics AST");
    };
  } else {
    auto new_operands = visit_operands(operands);
    if (cudf::ast::detail::ast_operator_arity(op) == 2) {
      _stats_expr.push(ast::operation{op, new_operands.front(), new_operands.back()});
    } else if (cudf::ast::detail::ast_operator_arity(op) == 1) {
      _stats_expr.push(ast::operation{op, new_operands.front()});
    }
  }
  return _stats_expr.back();
}

std::reference_wrapper<ast::expression const> stats_expression_converter::get_stats_expr() const
{
  return _stats_expr.back();
}

std::vector<std::reference_wrapper<ast::expression const>>
stats_expression_converter::visit_operands(
  cudf::host_span<std::reference_wrapper<ast::expression const> const> operands)
{
  std::vector<std::reference_wrapper<ast::expression const>> transformed_operands;
  std::transform(operands.begin(),
                 operands.end(),
                 std::back_inserter(transformed_operands),
                 [t = this](auto& operand) { return operand.get().accept(*t); });

  return transformed_operands;
}

}  // namespace cudf::io::parquet::detail
