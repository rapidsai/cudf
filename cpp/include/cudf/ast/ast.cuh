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
#include <cudf/binaryop.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/error.hpp>
#include "cudf/table/table_device_view.cuh"
#include "cudf/types.hpp"

namespace cudf {
namespace detail {

}  // namespace detail

enum class ast_data_source {
  COLUMN,       // A value from a column
  LITERAL,      // A constant value
  INTERMEDIATE  // An internal node (not a leaf) in the AST
};

enum class ast_binary_operator {
  ADD,      // Addition
  SUBTRACT  // Subtraction
};

struct ast_expression_source {
  ast_data_source source;  // Source of data
  cudf::size_type
    data_index;  // The column index of a table, index of a literal, or index of an intermediate
};

template <typename Element>
struct ast_binary_expression {
  ast_binary_operator op;
  ast_expression_source lhs;
  ast_expression_source rhs;
};

template <typename Element>
Element ast_evaluate_operator(ast_binary_operator op, Element lhs, Element rhs)
{
  switch (op) {
    case ast_binary_operator::ADD: return lhs + rhs;
    case ast_binary_operator::SUBTRACT: return lhs - rhs;
    default:
      // TODO: Error
      return 0;
  }
}

template <typename Element>
Element ast_resolve_data_source(ast_expression_source expression_source, table_view const& table)
{
  const cudf::size_type row = 0;
  switch (expression_source.source) {
    case ast_data_source::COLUMN: {
      auto column = table.column(expression_source.data_index);
      auto elt    = column.data<Element>()[row];
      return elt;
    }
    case ast_data_source::LITERAL: {
      // TODO: Fetch and return literal.
      return static_cast<Element>(0);
    }
    case ast_data_source::INTERMEDIATE: {
      // TODO: Fetch and return intermediate.
      return static_cast<Element>(0);
    }
    default: {
      // TODO: Error
      return static_cast<Element>(0);
    }
  }
}

template <typename Element>
Element ast_evaluate_expression(ast_binary_expression<Element> binary_expression,
                                table_view const& table)
{
  const Element lhs = cudf::ast_resolve_data_source<Element>(binary_expression.lhs, table);
  const Element rhs = cudf::ast_resolve_data_source<Element>(binary_expression.rhs, table);
  return ast_evaluate_operator(binary_expression.op, lhs, rhs);
}

}  // namespace cudf
