/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cudf/operators/opcodes.hpp>

#include <vector>

namespace cudf::detail::row_ir {

enum [[nodiscard]] type : uint64_t {
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
  ARG_MASK               = 0x10000000,
  ARG0                   = 0x10000000,
  ARG1                   = 0x10000001,
  ARG2                   = 0x10000002,
  ARG3                   = 0x10000003,
  INPUT                  = 0x20000000,
};

struct [[nodiscard]] op_type {
  type output            = type::NONE;
  std::vector<type> args = {};
};

/**
 * @brief Indicates how an operator propagates null values
 */
enum class [[nodiscard]] null_output : uint8_t {
  PROPAGATE       = 0,
  ALWAYS_VALID    = 1,
  ALWAYS_NULLABLE = 2,
};

[[nodiscard]] inline std::string_view get_op_name(opcode op)
{
  switch (op) {
    case opcode::GET_INPUT: return "get_input";
    case opcode::SET_OUTPUT: return "set_output";
    case opcode::IDENTITY: return "identity";
    case opcode::IS_NULL: return "is_null";
    case opcode::NULLIFY_IF: return "nullify_if";
    case opcode::COALESCE: return "coalesce";
    case opcode::PREDICATE: return "predicate";
    case opcode::ABS: return "abs";
    case opcode::ADD: return "add";
    case opcode::DIV: return "div";
    case opcode::TRUE_DIV: return "true_div";
    case opcode::FLOOR_DIV: return "floor_div";
    case opcode::MOD: return "mod";
    case opcode::PYMOD: return "pymod";
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
    case opcode::ANSI_PRECISION_CHECK: return "ansi_precision_check";
    case opcode::ANSI_TRY_ADD: return "ansi_try_add";
    case opcode::ANSI_TRY_SUB: return "ansi_try_sub";
    case opcode::ANSI_TRY_MUL: return "ansi_try_mul";
    case opcode::ANSI_TRY_DIV: return "ansi_try_div";
    case opcode::ANSI_TRY_MOD: return "ansi_try_mod";
    case opcode::ANSI_TRY_ABS: return "ansi_try_abs";
    case opcode::ANSI_TRY_NEG: return "ansi_try_neg";
    case opcode::ANSI_TRY_PRECISION_CHECK: return "ansi_try_precision_check";
    case opcode::BIT_AND: return "bit_and";
    case opcode::BIT_INVERT: return "bit_invert";
    case opcode::BIT_OR: return "bit_or";
    case opcode::BIT_XOR: return "bit_xor";
    case opcode::SHIFT_LEFT: return "shift_left";
    case opcode::SHIFT_RIGHT: return "shift_right";
    case opcode::CAST_TO_B8: return "cast_to_b8";
    case opcode::CAST_TO_I8: return "cast_to_i8";
    case opcode::CAST_TO_I16: return "cast_to_i16";
    case opcode::CAST_TO_I32: return "cast_to_i32";
    case opcode::CAST_TO_I64: return "cast_to_i64";
    case opcode::CAST_TO_U8: return "cast_to_u8";
    case opcode::CAST_TO_U16: return "cast_to_u16";
    case opcode::CAST_TO_U32: return "cast_to_u32";
    case opcode::CAST_TO_U64: return "cast_to_u64";
    case opcode::CAST_TO_F32: return "cast_to_f32";
    case opcode::CAST_TO_F64: return "cast_to_f64";
    case opcode::CAST_TO_DEC32: return "cast_to_dec32";
    case opcode::CAST_TO_DEC64: return "cast_to_dec64";
    case opcode::CAST_TO_DEC128: return "cast_to_dec128";
    case opcode::RESCALE: return "rescale";
    case opcode::EQUAL: return "equal";
    case opcode::NOT_EQUAL: return "not_equal";
    case opcode::GREATER: return "greater";
    case opcode::GREATER_EQUAL: return "greater_equal";
    case opcode::LESS: return "less";
    case opcode::LESS_EQUAL: return "less_equal";
    case opcode::NULL_EQUAL: return "null_equal";
    case opcode::NULL_LOGICAL_AND: return "null_logical_and";
    case opcode::NULL_LOGICAL_OR: return "null_logical_or";
    case opcode::LOGICAL_AND: return "logical_and";
    case opcode::LOGICAL_OR: return "logical_or";
    case opcode::LOGICAL_NOT: return "logical_not";
    case opcode::IF_ELSE: return "if_else";
    case opcode::CBRT: return "cbrt";
    case opcode::CEIL: return "ceil";
    case opcode::FLOOR: return "floor";
    case opcode::RINT: return "rint";
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
    default: CUDF_UNREACHABLE("Invalid opcode");
  }
}

[[nodiscard]] inline null_output get_op_null_output(opcode op)
{
  switch (op) {
    case opcode::IS_NULL:
    case opcode::NULL_EQUAL:
    case opcode::PREDICATE: return null_output::ALWAYS_VALID;

    case opcode::NULLIFY_IF:
    case opcode::COALESCE:
    case opcode::ANSI_TRY_ADD:
    case opcode::ANSI_TRY_SUB:
    case opcode::ANSI_TRY_MUL:
    case opcode::ANSI_TRY_DIV:
    case opcode::ANSI_TRY_MOD:
    case opcode::ANSI_TRY_ABS:
    case opcode::ANSI_TRY_NEG:
    case opcode::ANSI_TRY_PRECISION_CHECK:
    case opcode::NULL_LOGICAL_AND:
    case opcode::NULL_LOGICAL_OR: return null_output::ALWAYS_NULLABLE;

    default: return null_output::PROPAGATE;
  }
}

/**
 * @brief Indicates whether the output of the operator will be different when it is called with or
 * without the null-ness of a value.
 */
[[nodiscard]] inline bool get_op_requires_nulls(opcode op)
{
  switch (op) {
    case opcode::COALESCE:
    case opcode::IS_NULL:
    case opcode::NULL_EQUAL:
    case opcode::NULL_LOGICAL_AND:
    case opcode::NULL_LOGICAL_OR:
    case opcode::PREDICATE: return true;

    default: return false;
  }
}

[[nodiscard]] inline bool get_op_is_fallible(opcode op)
{
  switch (op) {
    case opcode::ANSI_ADD:
    case opcode::ANSI_SUB:
    case opcode::ANSI_MUL:
    case opcode::ANSI_DIV:
    case opcode::ANSI_MOD:
    case opcode::ANSI_ABS:
    case opcode::ANSI_NEG:
    case opcode::ANSI_PRECISION_CHECK: return true;

    default: return false;
  }
}

[[nodiscard]] inline int32_t get_output_decimal_scale(opcode op,
                                                      std::span<int32_t const> arg_scales,
                                                      std::optional<int32_t> output_scale)
{
  // TODO: finish up
  switch (op) {
    case opcode::GET_INPUT:
    case opcode::SET_OUTPUT:
    case opcode::IDENTITY:
    case opcode::COALESCE:
    case opcode::PREDICATE:
    case opcode::IS_NULL:
    case opcode::ABS:
    case opcode::NEG:
    case opcode::ANSI_ABS:
    case opcode::ANSI_NEG:
    case opcode::ANSI_TRY_NEG:
    case opcode::ANSI_TRY_ABS:
    case opcode::CAST_TO_B8:
    case opcode::CAST_TO_I8:
    case opcode::CAST_TO_I16:
    case opcode::CAST_TO_I32:
    case opcode::CAST_TO_I64:
    case opcode::CAST_TO_U8:
    case opcode::CAST_TO_U16:
    case opcode::CAST_TO_U32:
    case opcode::CAST_TO_U64:
    case opcode::CAST_TO_F32:
    case opcode::CAST_TO_F64:
    case opcode::CAST_TO_DEC32:
    case opcode::CAST_TO_DEC64:
    case opcode::CAST_TO_DEC128:
    case opcode::NULLIFY_IF:
    case opcode::EQUAL:
    case opcode::GREATER:
    case opcode::GREATER_EQUAL:
    case opcode::LESS:
    case opcode::LESS_EQUAL:
    case opcode::NOT_EQUAL:
    case opcode::NULL_EQUAL:
    case opcode::NULL_LOGICAL_AND:
    case opcode::NULL_LOGICAL_OR:
    case opcode::LOGICAL_AND:
    case opcode::LOGICAL_OR:
    case opcode::LOGICAL_NOT:
    case opcode::IF_ELSE:
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
    case opcode::TANH:
    case opcode::ANSI_PRECISION_CHECK:
    case opcode::ANSI_TRY_PRECISION_CHECK:
    case opcode::BIT_AND:
    case opcode::BIT_INVERT:
    case opcode::BIT_OR:
    case opcode::BIT_XOR:
    case opcode::SHIFT_LEFT:
    case opcode::SHIFT_RIGHT:
    case opcode::CBRT:
    case opcode::CEIL:
    case opcode::FLOOR:
    case opcode::RINT:
    case opcode::SQRT:
    case opcode::POW:
    case opcode::EXP:
    case opcode::TRUE_DIV:
    case opcode::LOG: return arg_scales[0];
    case opcode::FLOOR_DIV:
    case opcode::ANSI_DIV:
    case opcode::DIV:
    case opcode::ANSI_TRY_DIV: return arg_scales[0] - arg_scales[1];
    case opcode::ADD:
    case opcode::SUB:
    case opcode::ANSI_ADD:
    case opcode::ANSI_SUB:
    case opcode::ANSI_TRY_ADD:
    case opcode::ANSI_TRY_SUB:
    case opcode::MOD:
    case opcode::ANSI_MOD:
    case opcode::ANSI_TRY_MOD:
    case opcode::PYMOD: return std::min(arg_scales[0], arg_scales[1]);
    case opcode::MUL:
    case opcode::ANSI_MUL:
    case opcode::ANSI_TRY_MUL: return arg_scales[0] + arg_scales[1];
    case opcode::RESCALE: return output_scale.value();
    default: CUDF_UNREACHABLE("Invalid opcode");
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
[[nodiscard]] inline op_type get_op_typing(opcode op)
{
  // TODO: finish up
  switch (op) {
    case opcode::GET_INPUT: return {type::INPUT, {}};
    case opcode::SET_OUTPUT: return {type::NONE, {type::ALL}};
    case opcode::IDENTITY: return {type::ARG0, {type::ALL}};
    case opcode::IS_NULL: return {type::BOOL8, {type::ALL}};
    case opcode::NULLIFY_IF: return {type::ARG0, {type::ALL, type::BOOL8}};
    case opcode::COALESCE: return {type::ARG0, {type::ALL, type::ARG0}};
    case opcode::PREDICATE: return {type::ARG0, {type::BOOL8}};
    case opcode::ABS:
    case opcode::NEG:
    case opcode::ANSI_ABS:
    case opcode::ANSI_NEG:
    case opcode::ANSI_TRY_NEG:
    case opcode::ANSI_TRY_ABS: return {type::ARG0, {type::ARITHMETIC}};
    case opcode::FLOOR_DIV:
    case opcode::TRUE_DIV: return {type::ARG0, {type{type::FLOATS | type::INTEGERS}, type::ARG0}};
    case opcode::ADD:
    case opcode::DIV:
    case opcode::MOD:
    case opcode::PYMOD:
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
    case opcode::ANSI_TRY_MOD: return {type::ARG0, {type::ARITHMETIC, type::ARG0}};
    case opcode::ANSI_PRECISION_CHECK:
    case opcode::ANSI_TRY_PRECISION_CHECK: return {type::ARG0, {type::DECIMALS, type::INT32}};
    case opcode::BIT_AND:
    case opcode::BIT_INVERT:
    case opcode::BIT_OR:
    case opcode::BIT_XOR:
    case opcode::SHIFT_LEFT:
    case opcode::SHIFT_RIGHT: return {type::ARG0, {type::INTEGERS, type::ARG0}};
    case opcode::CAST_TO_B8: return {type::BOOL8, {type{type::INTEGERS | type::FLOATS}}};
    case opcode::CAST_TO_I8: return {type::INT8, {type{type::INTEGERS | type::FLOATS}}};
    case opcode::CAST_TO_I16: return {type::INT16, {type{type::INTEGERS | type::FLOATS}}};
    case opcode::CAST_TO_I32: return {type::INT32, {type{type::INTEGERS | type::FLOATS}}};
    case opcode::CAST_TO_I64: return {type::INT64, {type{type::INTEGERS | type::FLOATS}}};
    case opcode::CAST_TO_U8: return {type::UINT8, {type{type::INTEGERS | type::FLOATS}}};
    case opcode::CAST_TO_U16: return {type::UINT16, {type{type::INTEGERS | type::FLOATS}}};
    case opcode::CAST_TO_U32: return {type::UINT32, {type{type::INTEGERS | type::FLOATS}}};
    case opcode::CAST_TO_U64: return {type::UINT64, {type{type::INTEGERS | type::FLOATS}}};
    case opcode::CAST_TO_F32: return {type::FLOAT32, {type{type::INTEGERS | type::FLOATS}}};
    case opcode::CAST_TO_F64: return {type::FLOAT64, {type{type::INTEGERS | type::FLOATS}}};
    case opcode::CAST_TO_DEC32: return {type::DECIMAL32, {type::DECIMALS}};
    case opcode::CAST_TO_DEC64: return {type::DECIMAL64, {type::DECIMALS}};
    case opcode::CAST_TO_DEC128: return {type::DECIMAL128, {type::DECIMALS}};
    case opcode::RESCALE: return {type::ARG0, {type::DECIMALS, type::INT32}};
    case opcode::EQUAL:
    case opcode::GREATER:
    case opcode::GREATER_EQUAL:
    case opcode::LESS:
    case opcode::LESS_EQUAL: return {type::BOOL8, {type::ALL, type::ARG0}};
    case opcode::NOT_EQUAL:
    case opcode::NULL_EQUAL: return {type::BOOL8, {type::ALL, type::ARG0}};
    case opcode::NULL_LOGICAL_AND:
    case opcode::NULL_LOGICAL_OR:
    case opcode::LOGICAL_AND:
    case opcode::LOGICAL_OR: return {type::BOOL8, {type::BOOL8, type::ARG0}};
    case opcode::LOGICAL_NOT: return {type::ARG0, {type::BOOL8}};
    case opcode::IF_ELSE: return {type::ARG0, {type::ALL, type::ARG0, type::BOOL8}};
    case opcode::CBRT:
    case opcode::CEIL:
    case opcode::FLOOR:
    case opcode::RINT:
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
    case opcode::TANH: return {type::ARG0, {type::FLOATS}};
    default: CUDF_UNREACHABLE("Invalid opcode");
  }
}

[[nodiscard]] inline int32_t get_op_arity(opcode op)
{
  return static_cast<int32_t>(get_op_typing(op).args.size());
}

}  // namespace cudf::detail::row_ir
