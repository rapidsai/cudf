
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/ast/expressions.hpp>

namespace CUDF_EXPORT cudf {
namespace ast::detail {
/**
 * @brief Base "visitor" pattern class with the `expression` class for expression transformer.
 *
 * This class can be used to implement recursive traversal of AST tree, and used to validate or
 * translate an AST expression.
 */
class expression_transformer {
 public:
  /**
   * @brief Visit a literal expression.
   *
   * @param expr Literal expression
   * @return Reference wrapper of transformed expression
   */
  virtual std::reference_wrapper<expression const> visit(literal const& expr) = 0;

  /**
   * @brief Visit a column reference expression.
   *
   * @param expr Column reference expression
   * @return Reference wrapper of transformed expression
   */
  virtual std::reference_wrapper<expression const> visit(column_reference const& expr) = 0;

  /**
   * @brief Visit an expression expression
   *
   * @param expr Expression expression
   * @return Reference wrapper of transformed expression
   */
  virtual std::reference_wrapper<expression const> visit(operation const& expr) = 0;

  /**
   * @brief Visit a column name reference expression.
   *
   * @param expr Column name reference expression
   * @return Reference wrapper of transformed expression
   */
  virtual std::reference_wrapper<expression const> visit(column_name_reference const& expr) = 0;

  virtual ~expression_transformer() {}
};

}  // namespace ast::detail

}  // namespace CUDF_EXPORT cudf
