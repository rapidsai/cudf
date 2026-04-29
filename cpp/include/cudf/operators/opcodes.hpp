/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cudf/operators/types.cuh>
#include <cudf/utilities/error.hpp>

namespace cudf {
namespace detail {
namespace row_ir {

enum class [[nodiscard]] opcode : int32_t {
  GET_INPUT,
  SET_OUTPUT,

  // Identity operators
  IDENTITY,

  // Null handling operators
  IS_NULL,
  NULLIFY_IF,

  COALESCE,
  PREDICATE,

  /// Arithmetic operators
  ABS,
  ADD,
  DIV,
  TRUE_DIV,
  FLOOR_DIV,
  MOD,
  PYMOD,
  MUL,
  NEG,
  SUB,

  /// ANSI Arithmetic functions. raise errors on overflow, division by zero, etc.
  ANSI_ADD,
  ANSI_SUB,
  ANSI_MUL,
  ANSI_DIV,
  ANSI_MOD,
  ANSI_ABS,
  ANSI_NEG,
  ANSI_PRECISION_CHECK,

  /// ANSI TRY arithmetic functions. return NULL instead of raising errors
  ANSI_TRY_ADD,
  ANSI_TRY_SUB,
  ANSI_TRY_MUL,
  ANSI_TRY_DIV,
  ANSI_TRY_MOD,
  ANSI_TRY_ABS,
  ANSI_TRY_NEG,
  ANSI_TRY_PRECISION_CHECK,

  /// Bitwise operators
  BIT_AND,
  BIT_INVERT,
  BIT_OR,
  BIT_XOR,
  SHIFT_LEFT,
  SHIFT_RIGHT,

  /// Type conversion operators
  CAST_TO_B8,
  CAST_TO_I8,
  CAST_TO_I16,
  CAST_TO_I32,
  CAST_TO_I64,
  CAST_TO_U8,
  CAST_TO_U16,
  CAST_TO_U32,
  CAST_TO_U64,
  CAST_TO_F32,
  CAST_TO_F64,
  CAST_TO_DEC32,
  CAST_TO_DEC64,
  CAST_TO_DEC128,
  RESCALE,

  /// Comparison & Logic operators
  EQUAL,
  NOT_EQUAL,
  GREATER,
  GREATER_EQUAL,
  LESS,
  LESS_EQUAL,
  NULL_EQUAL,
  NULL_LOGICAL_AND,
  NULL_LOGICAL_OR,
  LOGICAL_AND,
  LOGICAL_OR,
  LOGICAL_NOT,
  IF_ELSE,

  /// Mathematical operators
  CBRT,
  CEIL,
  FLOOR,
  RINT,
  SQRT,
  POW,
  EXP,
  LOG,

  /// Trigonometric operators
  ARCCOS,
  ARCCOSH,
  ARCSIN,
  ARCSINH,
  ARCTAN,
  ARCTANH,
  COS,
  COSH,
  SIN,
  SINH,
  TAN,
  TANH,
};

}  // namespace row_ir
}  // namespace detail
}  // namespace cudf
