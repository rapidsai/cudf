/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "jit/row_ir.hpp"

#include <cudf/ast/detail/expression_parser.hpp>
#include <cudf/ast/detail/expression_transformer.hpp>
#include <cudf/ast/detail/operators.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <stdexcept>

namespace cudf {
namespace ast {

operation::operation(ast_operator op, expression const& input) : op{op}, operands{input}
{
  CUDF_EXPECTS(cudf::ast::detail::ast_operator_arity(op) == 1,
               "The provided operator is not a unary operator.",
               std::invalid_argument);
}

operation::operation(ast_operator op, expression const& left, expression const& right)
  : op{op}, operands{left, right}
{
  CUDF_EXPECTS(cudf::ast::detail::ast_operator_arity(op) == 2,
               "The provided operator is not a binary operator.",
               std::invalid_argument);
}

cudf::size_type literal::accept(detail::expression_parser& visitor) const
{
  return visitor.visit(*this);
}

cudf::size_type column_reference::accept(detail::expression_parser& visitor) const
{
  return visitor.visit(*this);
}

cudf::size_type operation::accept(detail::expression_parser& visitor) const
{
  return visitor.visit(*this);
}

cudf::size_type column_name_reference::accept(detail::expression_parser& visitor) const
{
  return visitor.visit(*this);
}

auto literal::accept(detail::expression_transformer& visitor) const
  -> decltype(visitor.visit(*this))
{
  return visitor.visit(*this);
}

auto column_reference::accept(detail::expression_transformer& visitor) const
  -> decltype(visitor.visit(*this))
{
  return visitor.visit(*this);
}

auto operation::accept(detail::expression_transformer& visitor) const
  -> decltype(visitor.visit(*this))
{
  return visitor.visit(*this);
}

bool operation::may_evaluate_null(table_view const& left,
                                  table_view const& right,
                                  rmm::cuda_stream_view stream) const
{
  return std::any_of(operands.cbegin(),
                     operands.cend(),
                     [&left, &right, &stream](std::reference_wrapper<expression const> subexpr) {
                       return subexpr.get().may_evaluate_null(left, right, stream);
                     });
};

auto column_name_reference::accept(detail::expression_transformer& visitor) const
  -> decltype(visitor.visit(*this))
{
  return visitor.visit(*this);
}

std::unique_ptr<cudf::detail::row_ir::node> literal::accept(
  cudf::detail::row_ir::ast_converter& converter) const
{
  return converter.add_ir_node(*this);
}

std::unique_ptr<cudf::detail::row_ir::node> column_reference::accept(
  cudf::detail::row_ir::ast_converter& converter) const
{
  return converter.add_ir_node(*this);
}

std::unique_ptr<cudf::detail::row_ir::node> operation::accept(
  cudf::detail::row_ir::ast_converter& converter) const
{
  return converter.add_ir_node(*this);
}

std::unique_ptr<cudf::detail::row_ir::node> column_name_reference::accept(
  cudf::detail::row_ir::ast_converter&) const
{
  CUDF_FAIL(
    "column_name_reference is not supported in row_ir. row_ir only supports resolved expressions",
    std::invalid_argument);
}

}  // namespace ast
}  // namespace cudf
