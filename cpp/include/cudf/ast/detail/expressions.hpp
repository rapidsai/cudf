/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <cudf/types.hpp>

namespace cudf {
namespace ast {
namespace detail {
// Forward declaration.
class expression_parser;

/**
 * @brief A generic expression node that can be evaluated to return a value.
 *
 * An expression is composed of operands acting on constants or variables.
 * Since an expression's components are themselves expressions, they must be
 * treated as defining an abstract syntax tree and parsed accordingly.
 * Expressions support parsing using the `expression_parser` class. The
 * recursive parsing of an expression tree is accomplished via the visitor
 * pattern, so all expressions must implement the `accept` method to enable the
 * necessary multiple-dispatch in conjunction with the `expression_parser`.
 */
struct node {
  /**
   * @brief Accepts a visitor class.
   *
   * @param visitor Visitor.
   * @return cudf::size_type The index of this expression in the array of device data references
   * stored by the `expression_parser`.
   */
  virtual cudf::size_type accept(detail::expression_parser& visitor) const = 0;
};

}  // namespace detail

}  // namespace ast

}  // namespace cudf
