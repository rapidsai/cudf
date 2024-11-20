/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

auto column_name_reference::accept(detail::expression_transformer& visitor) const
  -> decltype(visitor.visit(*this))
{
  return visitor.visit(*this);
}
}  // namespace ast

}  // namespace cudf
