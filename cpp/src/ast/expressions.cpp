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
#include <cudf/ast/detail/operators.hpp>
#include <cudf/ast/expression_parser.hpp>
#include <cudf/ast/expression_transformer.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

namespace cudf::ast {

operation::operation(ast_operator op, expression const& input) : op(op), operands({input})
{
  if (cudf::ast::detail::ast_operator_arity(op) != 1) {
    CUDF_FAIL("The provided operator is not a unary operator.");
  }
}

operation::operation(ast_operator op, expression const& left, expression const& right)
  : op(op), operands({left, right})
{
  if (cudf::ast::detail::ast_operator_arity(op) != 2) {
    CUDF_FAIL("The provided operator is not a binary operator.");
  }
}

cudf::size_type literal::accept(expression_parser& visitor) const { return visitor.visit(*this); }
cudf::size_type column_reference::accept(expression_parser& visitor) const
{
  return visitor.visit(*this);
}
cudf::size_type operation::accept(expression_parser& visitor) const { return visitor.visit(*this); }
cudf::size_type column_name_reference::accept(expression_parser& visitor) const
{
  return visitor.visit(*this);
}

auto literal::accept(expression_transformer& visitor) const -> decltype(visitor.visit(*this))
{
  return visitor.visit(*this);
}
auto column_reference::accept(expression_transformer& visitor) const
  -> decltype(visitor.visit(*this))
{
  return visitor.visit(*this);
}
auto operation::accept(expression_transformer& visitor) const -> decltype(visitor.visit(*this))
{
  return visitor.visit(*this);
}
auto column_name_reference::accept(expression_transformer& visitor) const
  -> decltype(visitor.visit(*this))
{
  return visitor.visit(*this);
}
}  // namespace cudf::ast
