
/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
#pragma once

#include <cudf/ast/expressions.hpp>

namespace cudf::ast {
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
}  // namespace cudf::ast
