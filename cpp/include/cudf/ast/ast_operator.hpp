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
#pragma once

#include <cudf/utilities/export.hpp>

#include <cuda/std/cstdint>

namespace CUDF_EXPORT cudf {

namespace ast {
/**
 * @addtogroup expressions
 * @{
 * @file
 */

/**
 * @brief Enum of supported operators.
 */
enum class ast_operator : int32_t {
  // Binary operators
  ADD,         ///< operator +
  SUB,         ///< operator -
  MUL,         ///< operator *
  DIV,         ///< operator / using common type of lhs and rhs
  TRUE_DIV,    ///< operator / after promoting type to floating point
  FLOOR_DIV,   ///< operator / after promoting to 64 bit floating point and then
               ///< flooring the result
  MOD,         ///< operator %
  PYMOD,       ///< operator % using Python's sign rules for negatives
  POW,         ///< lhs ^ rhs
  EQUAL,       ///< operator ==
  NULL_EQUAL,  ///< operator == with Spark rules: NULL_EQUAL(null, null) is true, NULL_EQUAL(null,
               ///< valid) is false, and
               ///< NULL_EQUAL(valid, valid) == EQUAL(valid, valid)
  NOT_EQUAL,   ///< operator !=
  LESS,        ///< operator <
  GREATER,     ///< operator >
  LESS_EQUAL,  ///< operator <=
  GREATER_EQUAL,     ///< operator >=
  BITWISE_AND,       ///< operator &
  BITWISE_OR,        ///< operator |
  BITWISE_XOR,       ///< operator ^
  LOGICAL_AND,       ///< operator &&
  NULL_LOGICAL_AND,  ///< operator && with Spark rules: NULL_LOGICAL_AND(null, null) is null,
                     ///< NULL_LOGICAL_AND(null, true) is
                     ///< null, NULL_LOGICAL_AND(null, false) is false, and NULL_LOGICAL_AND(valid,
                     ///< valid) == LOGICAL_AND(valid, valid)
  LOGICAL_OR,        ///< operator ||
  NULL_LOGICAL_OR,   ///< operator || with Spark rules: NULL_LOGICAL_OR(null, null) is null,
                     ///< NULL_LOGICAL_OR(null, true) is true,
                     ///< NULL_LOGICAL_OR(null, false) is null, and NULL_LOGICAL_OR(valid, valid) ==
                     ///< LOGICAL_OR(valid, valid)
  // Unary operators
  IDENTITY,        ///< Identity function
  IS_NULL,         ///< Check if operand is null
  SIN,             ///< Trigonometric sine
  COS,             ///< Trigonometric cosine
  TAN,             ///< Trigonometric tangent
  ARCSIN,          ///< Trigonometric sine inverse
  ARCCOS,          ///< Trigonometric cosine inverse
  ARCTAN,          ///< Trigonometric tangent inverse
  SINH,            ///< Hyperbolic sine
  COSH,            ///< Hyperbolic cosine
  TANH,            ///< Hyperbolic tangent
  ARCSINH,         ///< Hyperbolic sine inverse
  ARCCOSH,         ///< Hyperbolic cosine inverse
  ARCTANH,         ///< Hyperbolic tangent inverse
  EXP,             ///< Exponential (base e, Euler number)
  LOG,             ///< Natural Logarithm (base e)
  SQRT,            ///< Square-root (x^0.5)
  CBRT,            ///< Cube-root (x^(1.0/3))
  CEIL,            ///< Smallest integer value not less than arg
  FLOOR,           ///< largest integer value not greater than arg
  ABS,             ///< Absolute value
  RINT,            ///< Rounds the floating-point argument arg to an integer value
  BIT_INVERT,      ///< Bitwise Not (~)
  NOT,             ///< Logical Not (!)
  CAST_TO_INT64,   ///< Cast value to int64_t
  CAST_TO_UINT64,  ///< Cast value to uint64_t
  CAST_TO_FLOAT64  ///< Cast value to double
};

/** @} */  // end of group
}  // namespace ast

}  // namespace CUDF_EXPORT cudf
