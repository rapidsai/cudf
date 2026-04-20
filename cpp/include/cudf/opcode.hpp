

/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/utilities/export.hpp>

#include <cuda/std/cstdint>

namespace CUDF_EXPORT cudf {

/**
 * @brief Enum of supported opcodes.
 */
enum class opcode : int32_t {
  // Binary operators
  ADD       = 0,  ///< operator +
  SUB       = 1,  ///< operator -
  MUL       = 2,  ///< operator *
  DIV       = 3,  ///< operator / using common type of lhs and rhs
  TRUE_DIV  = 4,  ///< operator / after promoting type to floating point
  FLOOR_DIV = 5,  ///< operator / after promoting to the common type of lhs and rhs (integral or
                  ///< floating point), and then flooring the result
  MOD   = 6,      ///< operator %
  PYMOD = 7,      ///< operator % using Python's sign rules for negatives
  POW   = 8,      ///< lhs ^ rhs
  EQUAL = 9,      ///< operator ==
  NULL_EQUAL =
    10,  ///< operator == with Spark rules: NULL_EQUAL(null, null) is true, NULL_EQUAL(null,
         ///< valid) is false, and
         ///< NULL_EQUAL(valid, valid) == EQUAL(valid, valid)
  NOT_EQUAL        = 11,  ///< operator !=
  LESS             = 12,  ///< operator <
  GREATER          = 13,  ///< operator >
  LESS_EQUAL       = 14,  ///< operator <=
  GREATER_EQUAL    = 15,  ///< operator >=
  BITWISE_AND      = 16,  ///< operator &
  BITWISE_OR       = 17,  ///< operator |
  BITWISE_XOR      = 18,  ///< operator ^
  LOGICAL_AND      = 19,  ///< operator &&
  NULL_LOGICAL_AND = 20,  ///< operator && with Spark rules: NULL_LOGICAL_AND(null, null) is null,
                          ///< NULL_LOGICAL_AND(null, true) is
  ///< null, NULL_LOGICAL_AND(null, false) is false, and NULL_LOGICAL_AND(valid,
  ///< valid) == LOGICAL_AND(valid, valid)
  LOGICAL_OR      = 21,  ///< operator ||
  NULL_LOGICAL_OR = 22,  ///< operator || with Spark rules: NULL_LOGICAL_OR(null, null) is null,
                         ///< NULL_LOGICAL_OR(null, true) is true,
  ///< NULL_LOGICAL_OR(null, false) is null, and NULL_LOGICAL_OR(valid, valid) ==
  ///< LOGICAL_OR(valid, valid)
  // Unary operators
  IDENTITY        = 23,  ///< Identity function
  IS_NULL         = 24,  ///< Check if operand is null
  SIN             = 25,  ///< Trigonometric sine
  COS             = 26,  ///< Trigonometric cosine
  TAN             = 27,  ///< Trigonometric tangent
  ARCSIN          = 28,  ///< Trigonometric sine inverse
  ARCCOS          = 29,  ///< Trigonometric cosine inverse
  ARCTAN          = 30,  ///< Trigonometric tangent inverse
  SINH            = 31,  ///< Hyperbolic sine
  COSH            = 32,  ///< Hyperbolic cosine
  TANH            = 33,  ///< Hyperbolic tangent
  ARCSINH         = 34,  ///< Hyperbolic sine inverse
  ARCCOSH         = 35,  ///< Hyperbolic cosine inverse
  ARCTANH         = 36,  ///< Hyperbolic tangent inverse
  EXP             = 37,  ///< Exponential (base e, Euler number)
  LOG             = 38,  ///< Natural Logarithm (base e)
  SQRT            = 39,  ///< Square-root (x^0.5)
  CBRT            = 40,  ///< Cube-root (x^(1.0/3))
  CEIL            = 41,  ///< Smallest integer value not less than arg
  FLOOR           = 42,  ///< largest integer value not greater than arg
  ABS             = 43,  ///< Absolute value
  RINT            = 44,  ///< Rounds the floating-point argument arg to an integer value
  BIT_INVERT      = 45,  ///< Bitwise Not (~)
  NOT             = 46,  ///< Logical Not (!)
  CAST_TO_INT64   = 47,  ///< Cast value to int64_t
  CAST_TO_UINT64  = 48,  ///< Cast value to uint64_t
  CAST_TO_FLOAT64 = 49,  ///< Cast value to double

  ANSI_ADD = 50,  ///< operator +, with ANSI SQL semantics (e.g. overflow checking)
  ANSI_SUB = 51,  ///< operator -, with ANSI SQL semantics (e.g. overflow checking)
  ANSI_MUL = 52,  ///< operator *, with ANSI SQL semantics (e.g. overflow checking)
  ANSI_DIV = 53,  ///< operator / using common type of lhs and rhs, with ANSI SQL semantics (e.g.
                  ///< division by zero checking)
  ANSI_ABS = 54,  ///< Absolute value, with ANSI SQL semantics (e.g. overflow checking)
  ANSI_CAST_TO_INT64 =
    55,  ///< Cast value to int64_t, with ANSI SQL semantics (e.g. overflow checking)
  ANSI_CAST_TO_UINT64 =
    56,  ///< Cast value to uint64_t, with ANSI SQL semantics (e.g. overflow checking)

  TRY_ADD = 57,  ///< operator +, with TRY semantics (e.g. returns null on overflow)
  TRY_SUB = 58,  ///< operator -, with TRY semantics (e.g. returns null on overflow)
  TRY_MUL = 59,  ///< operator *, with TRY semantics (e.g. returns null on overflow)
  TRY_DIV = 60,  ///< operator / using common type of lhs and rhs, with TRY semantics (e.g. returns
                 ///< null on division by zero)
  TRY_ABS = 61,  ///< Absolute value, with TRY semantics (e.g. returns null on overflow)
  TRY_CAST_TO_INT64 =
    62,  ///< Cast value to int64_t, with TRY semantics (e.g. returns null on overflow)
  TRY_CAST_TO_UINT64 =
    63,  ///< Cast value to uint64_t, with TRY semantics (e.g. returns null on overflow)
};

}  // namespace CUDF_EXPORT cudf
