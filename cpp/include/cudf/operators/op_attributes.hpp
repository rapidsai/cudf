/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cudf/operators/opcodes.hpp>

namespace cudf::detail::row_ir {

enum typing : uint64_t {
  NONE                   = 0x0,
  BOOL8                  = 0x1,
  INT8                   = 0x2,
  INT16                  = 0x4,
  INT32                  = 0x8,
  INT64                  = 0x10,
  UINT8                  = 0x20,
  UINT16                 = 0x40,
  UINT32                 = 0x80,
  UINT64                 = 0x100,
  FLOAT32                = 0x200,
  FLOAT64                = 0x400,
  DECIMAL32              = 0x800,
  DECIMAL64              = 0x1000,
  DECIMAL128             = 0x2000,
  TIMESTAMP_DAYS         = 0x4000,
  TIMESTAMP_SECONDS      = 0x8000,
  TIMESTAMP_MILLISECONDS = 0x10000,
  TIMESTAMP_MICROSECONDS = 0x20000,
  TIMESTAMP_NANOSECONDS  = 0x40000,
  DURATION_DAYS          = 0x80000,
  DURATION_SECONDS       = 0x100000,
  DURATION_MILLISECONDS  = 0x200000,
  DURATION_MICROSECONDS  = 0x400000,
  DURATION_NANOSECONDS   = 0x800000,
  STRING                 = 0x1000000,
  INTEGERS               = INT8 | INT16 | INT32 | INT64 | UINT8 | UINT16 | UINT32 | UINT64,
  SIGNED_INTEGERS        = INT8 | INT16 | INT32 | INT64,
  UNSIGNED_INTEGERS      = UINT8 | UINT16 | UINT32 | UINT64,
  FLOATS                 = FLOAT32 | FLOAT64,
  DECIMALS               = DECIMAL32 | DECIMAL64 | DECIMAL128,
  ARITHMETIC             = SIGNED_INTEGERS | UNSIGNED_INTEGERS | FLOATS | DECIMALS,
  SIGNED_ARITHMETIC      = SIGNED_INTEGERS | FLOATS | DECIMALS,
  ALL                    = 0x0FFFFFFF,
  ARG0                   = 0x10000000,
  ARG1                   = 0x10000001,
  ARG2                   = 0x10000002,
  INPUT                  = 0x20000000,
};

struct op_typing {
  typing output = typing::NONE;
  typing arg0   = typing::NONE;
  typing arg1   = typing::NONE;
  typing arg2   = typing::NONE;
};

/**
 * @brief Indicates how an operator propagates null values
 */
enum class null_output : uint8_t {
  PROPAGATE       = 0,
  ALWAYS_VALID    = 1,
  ALWAYS_NULLABLE = 2,
};

inline std::string_view get_op_name(opcode op)
{
  switch (op) {
    case opcode::GET_INPUT: return "get_input";
    case opcode::SET_OUTPUT: return "set_output";
    case opcode::IS_NULL: return "is_null";
    case opcode::NULLIFY_IF: return "nullify_if";
    case opcode::COALESCE: return "coalesce";
    case opcode::ABS: return "abs";
    case opcode::ADD: return "add";
    case opcode::DIV: return "div";
    case opcode::MOD: return "mod";
    case opcode::MUL: return "mul";
    case opcode::NEG: return "neg";
    case opcode::SUB: return "sub";
    case opcode::ANSI_ADD: return "ansi_add";
    case opcode::ANSI_SUB: return "ansi_sub";
    case opcode::ANSI_MUL: return "ansi_mul";
    case opcode::ANSI_DIV: return "ansi_div";
    case opcode::ANSI_MOD: return "ansi_mod";
    case opcode::ANSI_ABS: return "ansi_abs";
    case opcode::ANSI_NEG: return "ansi_neg";
    case opcode::ANSI_PRECISION_CAST: return "ansi_precision_cast";
    case opcode::ANSI_TRY_ADD: return "ansi_try_add";
    case opcode::ANSI_TRY_SUB: return "ansi_try_sub";
    case opcode::ANSI_TRY_MUL: return "ansi_try_mul";
    case opcode::ANSI_TRY_DIV: return "ansi_try_div";
    case opcode::ANSI_TRY_MOD: return "ansi_try_mod";
    case opcode::ANSI_TRY_ABS: return "ansi_try_abs";
    case opcode::ANSI_TRY_NEG: return "ansi_try_neg";
    case opcode::ANSI_TRY_PRECISION_CAST: return "ansi_try_precision_cast";
    case opcode::BIT_AND: return "bit_and";
    case opcode::BIT_INVERT: return "bit_invert";
    case opcode::BIT_OR: return "bit_or";
    case opcode::BIT_XOR: return "bit_xor";
    case opcode::CAST_TO_I32: return "cast_to_i32";
    case opcode::CAST_TO_I64: return "cast_to_i64";
    case opcode::CAST_TO_U32: return "cast_to_u32";
    case opcode::CAST_TO_U64: return "cast_to_u64";
    case opcode::CAST_TO_F32: return "cast_to_f32";
    case opcode::CAST_TO_F64: return "cast_to_f64";
    case opcode::CAST_TO_DEC32: return "cast_to_dec32";
    case opcode::CAST_TO_DEC64: return "cast_to_dec64";
    case opcode::CAST_TO_DEC128: return "cast_to_dec128";
    case opcode::EQUAL: return "equal";
    case opcode::GREATER: return "greater";
    case opcode::GREATER_EQUAL: return "greater_equal";
    case opcode::LESS: return "less";
    case opcode::LESS_EQUAL: return "less_equal";
    case opcode::NULL_EQUAL: return "null_equal";
    case opcode::NULL_LOGICAL_AND: return "null_logical_and";
    case opcode::NULL_LOGICAL_OR: return "null_logical_or";
    case opcode::LOGICAL_NOT: return "logical_not";
    case opcode::IF_ELSE: return "if_else";
    case opcode::CBRT: return "cbrt";
    case opcode::CEIL: return "ceil";
    case opcode::FLOOR: return "floor";
    case opcode::SQRT: return "sqrt";
    case opcode::POW: return "pow";
    case opcode::EXP: return "exp";
    case opcode::LOG: return "log";
    case opcode::ARCCOS: return "arccos";
    case opcode::ARCCOSH: return "arccosh";
    case opcode::ARCSIN: return "arcsin";
    case opcode::ARCSINH: return "arcsinh";
    case opcode::ARCTAN: return "arctan";
    case opcode::ARCTANH: return "arctanh";
    case opcode::COS: return "cos";
    case opcode::COSH: return "cosh";
    case opcode::SIN: return "sin";
    case opcode::SINH: return "sinh";
    case opcode::TAN: return "tan";
    case opcode::TANH: return "tanh";
  }
}

inline null_output get_op_null_output(opcode op)
{
  switch (op) {
    case opcode::IS_NULL:
    case opcode::NULL_EQUAL: return null_output::ALWAYS_VALID;

    case opcode::GET_INPUT:
    case opcode::SET_OUTPUT:
    case opcode::LOGICAL_NOT:
    case opcode::ABS:
    case opcode::ADD:
    case opcode::DIV:
    case opcode::MOD:
    case opcode::MUL:
    case opcode::NEG:
    case opcode::SUB:
    case opcode::ANSI_ADD:
    case opcode::ANSI_SUB:
    case opcode::ANSI_MUL:
    case opcode::ANSI_DIV:
    case opcode::ANSI_MOD:
    case opcode::ANSI_ABS:
    case opcode::ANSI_NEG:
    case opcode::ANSI_PRECISION_CAST:
    case opcode::IF_ELSE:
    case opcode::CBRT:
    case opcode::CEIL:
    case opcode::FLOOR:
    case opcode::SQRT:
    case opcode::POW:
    case opcode::EXP:
    case opcode::LOG:
    case opcode::ARCCOS:
    case opcode::ARCCOSH:
    case opcode::ARCSIN:
    case opcode::ARCSINH:
    case opcode::ARCTAN:
    case opcode::ARCTANH:
    case opcode::COS:
    case opcode::COSH:
    case opcode::SIN:
    case opcode::SINH:
    case opcode::TAN:
    case opcode::BIT_AND:
    case opcode::BIT_INVERT:
    case opcode::BIT_OR:
    case opcode::BIT_XOR:
    case opcode::CAST_TO_I32:
    case opcode::CAST_TO_I64:
    case opcode::CAST_TO_U32:
    case opcode::CAST_TO_U64:
    case opcode::CAST_TO_F32:
    case opcode::CAST_TO_F64:
    case opcode::CAST_TO_DEC32:
    case opcode::CAST_TO_DEC64:
    case opcode::CAST_TO_DEC128:
    case opcode::EQUAL:
    case opcode::GREATER:
    case opcode::GREATER_EQUAL:
    case opcode::LESS:
    case opcode::LESS_EQUAL:
    case opcode::TANH: return null_output::PROPAGATE;

    case opcode::NULLIFY_IF:
    case opcode::COALESCE:
    case opcode::ANSI_TRY_ADD:
    case opcode::ANSI_TRY_SUB:
    case opcode::ANSI_TRY_MUL:
    case opcode::ANSI_TRY_DIV:
    case opcode::ANSI_TRY_MOD:
    case opcode::ANSI_TRY_ABS:
    case opcode::ANSI_TRY_NEG:
    case opcode::ANSI_TRY_PRECISION_CAST:
    case opcode::NULL_LOGICAL_AND:
    case opcode::NULL_LOGICAL_OR: return null_output::ALWAYS_NULLABLE;
  }
}

/**
 * @brief Indicates whether the output of the operator will be different when it is called with or
 * without the null-ness of a value.
 */
inline bool get_op_requires_nulls(opcode op)
{
  switch (op) {
    case opcode::GET_INPUT:
    case opcode::SET_OUTPUT:
    case opcode::NULLIFY_IF:
    case opcode::ABS:
    case opcode::ADD:
    case opcode::DIV:
    case opcode::MOD:
    case opcode::MUL:
    case opcode::NEG:
    case opcode::SUB:
    case opcode::ANSI_ADD:
    case opcode::ANSI_SUB:
    case opcode::ANSI_MUL:
    case opcode::ANSI_DIV:
    case opcode::ANSI_MOD:
    case opcode::ANSI_ABS:
    case opcode::ANSI_NEG:
    case opcode::ANSI_PRECISION_CAST:
    case opcode::ANSI_TRY_ADD:
    case opcode::ANSI_TRY_SUB:
    case opcode::ANSI_TRY_MUL:
    case opcode::ANSI_TRY_DIV:
    case opcode::ANSI_TRY_MOD:
    case opcode::ANSI_TRY_ABS:
    case opcode::ANSI_TRY_NEG:
    case opcode::ANSI_TRY_PRECISION_CAST:
    case opcode::BIT_AND:
    case opcode::BIT_INVERT:
    case opcode::BIT_OR:
    case opcode::BIT_XOR:
    case opcode::CAST_TO_I32:
    case opcode::CAST_TO_I64:
    case opcode::CAST_TO_U32:
    case opcode::CAST_TO_U64:
    case opcode::CAST_TO_F32:
    case opcode::CAST_TO_F64:
    case opcode::CAST_TO_DEC32:
    case opcode::CAST_TO_DEC64:
    case opcode::CAST_TO_DEC128:
    case opcode::EQUAL:
    case opcode::GREATER:
    case opcode::GREATER_EQUAL:
    case opcode::LESS:
    case opcode::LESS_EQUAL:
    case opcode::LOGICAL_NOT:
    case opcode::IF_ELSE:
    case opcode::CBRT:
    case opcode::CEIL:
    case opcode::FLOOR:
    case opcode::SQRT:
    case opcode::POW:
    case opcode::EXP:
    case opcode::LOG:
    case opcode::ARCCOS:
    case opcode::ARCCOSH:
    case opcode::ARCSIN:
    case opcode::ARCSINH:
    case opcode::ARCTAN:
    case opcode::ARCTANH:
    case opcode::COS:
    case opcode::COSH:
    case opcode::SIN:
    case opcode::SINH:
    case opcode::TAN:
    case opcode::TANH: return false;

    case opcode::COALESCE:
    case opcode::IS_NULL:
    case opcode::NULL_EQUAL:
    case opcode::NULL_LOGICAL_AND:
    case opcode::NULL_LOGICAL_OR: return true;
  }
}

inline bool get_op_is_fallible(opcode op)
{
  switch (op) {
    case opcode::ANSI_ADD:
    case opcode::ANSI_SUB:
    case opcode::ANSI_MUL:
    case opcode::ANSI_DIV:
    case opcode::ANSI_MOD:
    case opcode::ANSI_ABS:
    case opcode::ANSI_NEG:
    case opcode::ANSI_PRECISION_CAST: return true;

    case opcode::GET_INPUT:
    case opcode::SET_OUTPUT:
    case opcode::IS_NULL:
    case opcode::NULLIFY_IF:
    case opcode::COALESCE:
    case opcode::ABS:
    case opcode::ADD:
    case opcode::DIV:
    case opcode::MOD:
    case opcode::MUL:
    case opcode::NEG:
    case opcode::SUB:
    case opcode::ANSI_TRY_ADD:
    case opcode::ANSI_TRY_SUB:
    case opcode::ANSI_TRY_MUL:
    case opcode::ANSI_TRY_DIV:
    case opcode::ANSI_TRY_MOD:
    case opcode::ANSI_TRY_ABS:
    case opcode::ANSI_TRY_NEG:
    case opcode::ANSI_TRY_PRECISION_CAST:
    case opcode::BIT_AND:
    case opcode::BIT_INVERT:
    case opcode::BIT_OR:
    case opcode::BIT_XOR:
    case opcode::CAST_TO_I32:
    case opcode::CAST_TO_I64:
    case opcode::CAST_TO_U32:
    case opcode::CAST_TO_U64:
    case opcode::CAST_TO_F32:
    case opcode::CAST_TO_F64:
    case opcode::CAST_TO_DEC32:
    case opcode::CAST_TO_DEC64:
    case opcode::CAST_TO_DEC128:
    case opcode::EQUAL:
    case opcode::GREATER:
    case opcode::GREATER_EQUAL:
    case opcode::LESS:
    case opcode::LESS_EQUAL:
    case opcode::NULL_EQUAL:
    case opcode::NULL_LOGICAL_AND:
    case opcode::NULL_LOGICAL_OR:
    case opcode::LOGICAL_NOT:
    case opcode::IF_ELSE:
    case opcode::CBRT:
    case opcode::CEIL:
    case opcode::FLOOR:
    case opcode::SQRT:
    case opcode::POW:
    case opcode::EXP:
    case opcode::LOG:
    case opcode::ARCCOS:
    case opcode::ARCCOSH:
    case opcode::ARCSIN:
    case opcode::ARCSINH:
    case opcode::ARCTAN:
    case opcode::ARCTANH:
    case opcode::COS:
    case opcode::COSH:
    case opcode::SIN:
    case opcode::SINH:
    case opcode::TAN:
    case opcode::TANH: return false;
  }
}

inline constexpr int32_t get_op_arity(opcode op)
{
  switch (op) {
    case opcode::GET_INPUT: return 0;

    case opcode::SET_OUTPUT:
    case opcode::IS_NULL:
    case opcode::NULLIFY_IF:
    case opcode::ABS:
    case opcode::NEG:
    case opcode::ANSI_ABS:
    case opcode::ANSI_NEG:
    case opcode::ANSI_TRY_ABS:
    case opcode::ANSI_TRY_NEG:
    case opcode::BIT_INVERT:
    case opcode::CAST_TO_I32:
    case opcode::CAST_TO_I64:
    case opcode::CAST_TO_U32:
    case opcode::CAST_TO_U64:
    case opcode::CAST_TO_F32:
    case opcode::CAST_TO_F64:
    case opcode::CAST_TO_DEC32:
    case opcode::CAST_TO_DEC64:
    case opcode::CAST_TO_DEC128:
    case opcode::LOGICAL_NOT:
    case opcode::CBRT:
    case opcode::CEIL:
    case opcode::FLOOR:
    case opcode::SQRT:
    case opcode::EXP:
    case opcode::LOG:
    case opcode::ARCCOS:
    case opcode::ARCCOSH:
    case opcode::ARCSIN:
    case opcode::ARCSINH:
    case opcode::ARCTAN:
    case opcode::ARCTANH:
    case opcode::COS:
    case opcode::COSH:
    case opcode::SIN:
    case opcode::SINH:
    case opcode::TAN:
    case opcode::TANH: return 1;

    case opcode::COALESCE:
    case opcode::ADD:
    case opcode::DIV:
    case opcode::MOD:
    case opcode::MUL:
    case opcode::SUB:
    case opcode::ANSI_ADD:
    case opcode::ANSI_SUB:
    case opcode::ANSI_MUL:
    case opcode::ANSI_DIV:
    case opcode::ANSI_MOD:
    case opcode::ANSI_TRY_ADD:
    case opcode::ANSI_TRY_SUB:
    case opcode::ANSI_TRY_MUL:
    case opcode::ANSI_TRY_DIV:
    case opcode::ANSI_TRY_MOD:
    case opcode::ANSI_PRECISION_CAST:
    case opcode::ANSI_TRY_PRECISION_CAST:
    case opcode::BIT_AND:
    case opcode::BIT_OR:
    case opcode::BIT_XOR:
    case opcode::EQUAL:
    case opcode::GREATER:
    case opcode::GREATER_EQUAL:
    case opcode::LESS:
    case opcode::LESS_EQUAL:
    case opcode::NULL_EQUAL:
    case opcode::NULL_LOGICAL_OR:
    case opcode::NULL_LOGICAL_AND:
    case opcode::POW: return 2;

    case opcode::IF_ELSE: return 3;
  }
}

/**
 * @brief Get the typing information for a given operator
 * This function returns the expected input and output types for a given operator. The typing
 * information can be used for type checking and inference when constructing expression trees.
 * @param op The operator for which to get the typing information
 * @return An `op_typing` struct containing the expected output type and input types for the
 * operator
 */
inline op_typing get_op_typing(opcode op)
{
  switch (op) {
    case opcode::GET_INPUT: return {typing::INPUT, typing::NONE, typing::NONE, typing::NONE};

    case opcode::SET_OUTPUT: return {typing::NONE, typing::ALL, typing::NONE, typing::NONE};

    case opcode::IS_NULL: return {typing::ARG0, typing::ALL, typing::NONE, typing::NONE};

    case opcode::NULLIFY_IF: return {typing::ARG1, typing::BOOL8, typing::ALL, typing::NONE};
    case opcode::COALESCE: return {typing::ARG0, typing::ALL, typing::ARG0, typing::NONE};

    case opcode::ABS:
    case opcode::NEG:
    case opcode::ANSI_ABS:
    case opcode::ANSI_NEG:
    case opcode::ANSI_TRY_NEG:
    case opcode::ANSI_TRY_ABS:
      return {typing::ARG0, typing::ARITHMETIC, typing::NONE, typing::NONE};

    case opcode::ADD:
    case opcode::DIV:
    case opcode::MOD:
    case opcode::MUL:
    case opcode::SUB:
    case opcode::ANSI_ADD:
    case opcode::ANSI_SUB:
    case opcode::ANSI_MUL:
    case opcode::ANSI_DIV:
    case opcode::ANSI_MOD:
    case opcode::ANSI_TRY_ADD:
    case opcode::ANSI_TRY_SUB:
    case opcode::ANSI_TRY_MUL:
    case opcode::ANSI_TRY_DIV:
    case opcode::ANSI_TRY_MOD:
      return {typing::ARG0, typing::ARITHMETIC, typing::ARG0, typing::NONE};

    case opcode::ANSI_PRECISION_CAST:
    case opcode::ANSI_TRY_PRECISION_CAST:
      return {typing::ARG0, typing::DECIMALS, typing::INT32, typing::NONE};

    case opcode::BIT_AND:
    case opcode::BIT_INVERT:
    case opcode::BIT_OR:
    case opcode::BIT_XOR: return {typing::ARG0, typing::INTEGERS, typing::ARG0, typing::NONE};

    case opcode::CAST_TO_I32:
    case opcode::CAST_TO_I64:
    case opcode::CAST_TO_U32:
    case opcode::CAST_TO_U64:
    case opcode::CAST_TO_F32:
    case opcode::CAST_TO_F64:
      return {typing::ARG0, typing{typing::INTEGERS | typing::FLOATS}, typing::NONE, typing::NONE};

    case opcode::CAST_TO_DEC32:
    case opcode::CAST_TO_DEC64:
    case opcode::CAST_TO_DEC128:
      return {typing::ARG0, typing::DECIMALS, typing::NONE, typing::NONE};

    case opcode::EQUAL:
    case opcode::GREATER:
    case opcode::GREATER_EQUAL:
    case opcode::LESS:
    case opcode::LESS_EQUAL: return {typing::BOOL8, typing::ALL, typing::ARG0, typing::NONE};

    case opcode::NULL_EQUAL:
    case opcode::NULL_LOGICAL_AND:
    case opcode::NULL_LOGICAL_OR: return {typing::BOOL8, typing::BOOL8, typing::ARG0, typing::NONE};

    case opcode::LOGICAL_NOT: return {typing::ARG0, typing::BOOL8, typing::NONE, typing::NONE};

    case opcode::IF_ELSE: return {typing::ARG1, typing::BOOL8, typing::ALL, typing::ARG0};

    case opcode::CBRT:
    case opcode::CEIL:
    case opcode::FLOOR:
    case opcode::SQRT:
    case opcode::POW:
    case opcode::EXP:
    case opcode::LOG:
    case opcode::ARCCOS:
    case opcode::ARCCOSH:
    case opcode::ARCSIN:
    case opcode::ARCSINH:
    case opcode::ARCTAN:
    case opcode::ARCTANH:
    case opcode::COS:
    case opcode::COSH:
    case opcode::SIN:
    case opcode::SINH:
    case opcode::TAN:
    case opcode::TANH: return {typing::ARG0, typing::FLOATS, typing::NONE, typing::NONE};
  }
}

}  // namespace cudf::detail::row_ir
