/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/ast/expressions.hpp>
#include <cudf/scalar/scalar.hpp>

#include <memory>
#include <utility>
#include <vector>

namespace cudf {
namespace jni {
namespace ast {

/**
 * A class to capture all of the resources associated with a compiled AST expression.
 * AST nodes do not own their child nodes, so every node in the expression tree
 * must be explicitly tracked in order to free the underlying resources for each node.
 *
 * This should be cleaned up a bit after the libcudf AST refactoring in
 * https://github.com/rapidsai/cudf/pull/8815 when a virtual destructor is added to the
 * base AST node type. Then we do not have to track every AST node type separately.
 */
class compiled_expr {
  /** All expression nodes within the expression tree */
  std::vector<std::unique_ptr<cudf::ast::expression>> expressions;

  /** GPU scalar instances that correspond to literal nodes */
  std::vector<std::unique_ptr<cudf::scalar>> scalars;

 public:
  cudf::ast::literal& add_literal(std::unique_ptr<cudf::ast::literal> literal_ptr,
                                  std::unique_ptr<cudf::scalar> scalar_ptr)
  {
    expressions.push_back(std::move(literal_ptr));
    scalars.push_back(std::move(scalar_ptr));
    return static_cast<cudf::ast::literal&>(*expressions.back());
  }

  cudf::ast::column_reference& add_column_ref(std::unique_ptr<cudf::ast::column_reference> ref_ptr)
  {
    expressions.push_back(std::move(ref_ptr));
    return static_cast<cudf::ast::column_reference&>(*expressions.back());
  }

  cudf::ast::operation& add_operation(std::unique_ptr<cudf::ast::operation> expr_ptr)
  {
    expressions.push_back(std::move(expr_ptr));
    return static_cast<cudf::ast::operation&>(*expressions.back());
  }

  /** Return the expression node at the top of the tree */
  cudf::ast::expression& get_top_expression() const { return *expressions.back(); }
};

}  // namespace ast
}  // namespace jni
}  // namespace cudf
