/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "jit/row_ir.hpp"

#include "runtime/context.hpp"

#include <cudf/column/column_factories.hpp>

#include <cuda/std/inplace_vector>

#include <algorithm>
#include <format>
#include <iostream>
#include <numeric>
#include <span>
#include <stdexcept>
#include <utility>

namespace cudf::detail::row_ir {

/**
 * @brief Get the opcode name for a given opcode
 * This function returns the name of the opcode corresponding to a given opcode.
 * The opcode name matches the operators in `cudf::detail::operators`.
 */
[[nodiscard]] std::string_view get_opcode_name(opcode op)
{
  switch (op) {
    case opcode::GET_INPUT: return "GET_INPUT";
    case opcode::SET_OUTPUT: return "SET_OUTPUT";
    case opcode::IDENTITY: return "IDENTITY";
    case opcode::IS_NULL: return "IS_NULL";
    case opcode::COALESCE: return "COALESCE";
    case opcode::PREDICATE: return "PREDICATE";
    case opcode::ABS: return "ABS";
    case opcode::ADD: return "ADD";
    case opcode::DIV: return "DIV";
    case opcode::TRUE_DIV: return "TRUE_DIV";
    case opcode::FLOOR_DIV: return "FLOOR_DIV";
    case opcode::MOD: return "MOD";
    case opcode::PYMOD: return "PYMOD";
    case opcode::MUL: return "MUL";
    case opcode::NEG: return "NEG";
    case opcode::SUB: return "SUB";
    case opcode::ADD_OVERFLOW: return "ADD_OVERFLOW";
    case opcode::SUB_OVERFLOW: return "SUB_OVERFLOW";
    case opcode::MUL_OVERFLOW: return "MUL_OVERFLOW";
    case opcode::DIV_OVERFLOW: return "DIV_OVERFLOW";
    case opcode::MOD_OVERFLOW: return "MOD_OVERFLOW";
    case opcode::ABS_OVERFLOW: return "ABS_OVERFLOW";
    case opcode::NEG_OVERFLOW: return "NEG_OVERFLOW";
    case opcode::CHECK_PRECISION: return "CHECK_PRECISION";
    case opcode::BITWISE_AND: return "BITWISE_AND";
    case opcode::BITWISE_INVERT: return "BITWISE_INVERT";
    case opcode::BITWISE_OR: return "BITWISE_OR";
    case opcode::BITWISE_XOR: return "BITWISE_XOR";
    case opcode::BITWISE_SHIFT_LEFT: return "BITWISE_SHIFT_LEFT";
    case opcode::BITWISE_SHIFT_RIGHT: return "BITWISE_SHIFT_RIGHT";
    case opcode::CAST_TO_BOOL8: return "CAST_TO_BOOL8";
    case opcode::CAST_TO_INT8: return "CAST_TO_INT8";
    case opcode::CAST_TO_INT16: return "CAST_TO_INT16";
    case opcode::CAST_TO_INT32: return "CAST_TO_INT32";
    case opcode::CAST_TO_INT64: return "CAST_TO_INT64";
    case opcode::CAST_TO_UINT8: return "CAST_TO_UINT8";
    case opcode::CAST_TO_UINT16: return "CAST_TO_UINT16";
    case opcode::CAST_TO_UINT32: return "CAST_TO_UINT32";
    case opcode::CAST_TO_UINT64: return "CAST_TO_UINT64";
    case opcode::CAST_TO_FLOAT32: return "CAST_TO_FLOAT32";
    case opcode::CAST_TO_FLOAT64: return "CAST_TO_FLOAT64";
    case opcode::CAST_TO_DECIMAL32: return "CAST_TO_DECIMAL32";
    case opcode::CAST_TO_DECIMAL64: return "CAST_TO_DECIMAL64";
    case opcode::CAST_TO_DECIMAL128: return "CAST_TO_DECIMAL128";
    case opcode::RESCALE: return "RESCALE";
    case opcode::EQUAL: return "EQUAL";
    case opcode::NOT_EQUAL: return "NOT_EQUAL";
    case opcode::GREATER: return "GREATER";
    case opcode::GREATER_EQUAL: return "GREATER_EQUAL";
    case opcode::LESS: return "LESS";
    case opcode::LESS_EQUAL: return "LESS_EQUAL";
    case opcode::NULL_EQUAL: return "NULL_EQUAL";
    case opcode::NULL_LOGICAL_AND: return "NULL_LOGICAL_AND";
    case opcode::NULL_LOGICAL_OR: return "NULL_LOGICAL_OR";
    case opcode::LOGICAL_AND: return "LOGICAL_AND";
    case opcode::LOGICAL_OR: return "LOGICAL_OR";
    case opcode::LOGICAL_NOT: return "LOGICAL_NOT";
    case opcode::IF_ELSE: return "IF_ELSE";
    case opcode::CBRT: return "CBRT";
    case opcode::CEIL: return "CEIL";
    case opcode::FLOOR: return "FLOOR";
    case opcode::RINT: return "RINT";
    case opcode::SQRT: return "SQRT";
    case opcode::POW: return "POW";
    case opcode::EXP: return "EXP";
    case opcode::LOG: return "LOG";
    case opcode::ARCCOS: return "ARCCOS";
    case opcode::ARCCOSH: return "ARCCOSH";
    case opcode::ARCSIN: return "ARCSIN";
    case opcode::ARCSINH: return "ARCSINH";
    case opcode::ARCTAN: return "ARCTAN";
    case opcode::ARCTANH: return "ARCTANH";
    case opcode::COS: return "COS";
    case opcode::COSH: return "COSH";
    case opcode::SIN: return "SIN";
    case opcode::SINH: return "SINH";
    case opcode::TAN: return "TAN";
    case opcode::TANH: return "TANH";
    default: CUDF_FAIL(std::format("Invalid opcode: {}", static_cast<int>(op)), std::runtime_error);
  }
}

/**
 * @brief Indicates how an operator propagates null values
 */
enum class [[nodiscard]] null_output : uint8_t {
  PROPAGATE       = 0,
  ALWAYS_VALID    = 1,
  ALWAYS_NULLABLE = 2,
};

[[nodiscard]] null_output get_op_null_output(opcode op, bool nullify_on_error)
{
  switch (op) {
    case opcode::IS_NULL:
    case opcode::NULL_EQUAL:
    case opcode::PREDICATE: return null_output::ALWAYS_VALID;

    case opcode::ADD_OVERFLOW:
    case opcode::SUB_OVERFLOW:
    case opcode::MUL_OVERFLOW:
    case opcode::DIV_OVERFLOW:
    case opcode::MOD_OVERFLOW:
    case opcode::ABS_OVERFLOW:
    case opcode::NEG_OVERFLOW:
    case opcode::CHECK_PRECISION:
      return nullify_on_error ? null_output::ALWAYS_NULLABLE : null_output::PROPAGATE;

    case opcode::NULL_LOGICAL_AND:
    case opcode::NULL_LOGICAL_OR: return null_output::ALWAYS_NULLABLE;

    default: return null_output::PROPAGATE;
  }
}

/**
 * @brief Indicates whether the output of the operator will be different when it is called with or
 * without the null-ness of a value.
 */
[[nodiscard]] bool get_op_is_null_dependent(opcode op)
{
  switch (op) {
    case opcode::IS_NULL:
    case opcode::NULL_EQUAL:
    case opcode::NULL_LOGICAL_AND:
    case opcode::NULL_LOGICAL_OR:
    case opcode::PREDICATE:
    case opcode::COALESCE: return true;

    default: return false;
  }
}

[[nodiscard]] bool get_op_is_fallible(opcode op, bool nullify_on_error)
{
  switch (op) {
    case opcode::ADD_OVERFLOW:
    case opcode::SUB_OVERFLOW:
    case opcode::MUL_OVERFLOW:
    case opcode::DIV_OVERFLOW:
    case opcode::MOD_OVERFLOW:
    case opcode::ABS_OVERFLOW:
    case opcode::NEG_OVERFLOW:
    case opcode::CHECK_PRECISION: return !nullify_on_error;
    default: return false;
  }
}

/**
 * @brief Get the output scale for a given operator and input scales
 * This function returns the output scale for a given operator based on the input scales and the
 * target scale (if applicable). The output scale is used for decimal operators to determine the
 * scale of the result based on the scales of the inputs and the operator being performed.
 */
[[nodiscard]] int32_t get_op_output_scale(opcode op,
                                          std::span<int32_t const> arg_scales,
                                          std::optional<int32_t> target_scale)
{
  switch (op) {
    // pseudo-opcode with no argument
    case opcode::GET_INPUT: return 0;
    // divisions
    case opcode::FLOOR_DIV:
    case opcode::DIV_OVERFLOW:
    case opcode::DIV: return arg_scales[0] - arg_scales[1];
    // add/sub/mod
    case opcode::ADD:
    case opcode::SUB:
    case opcode::ADD_OVERFLOW:
    case opcode::SUB_OVERFLOW:
    case opcode::MOD:
    case opcode::MOD_OVERFLOW:
    case opcode::PYMOD: return std::min(arg_scales[0], arg_scales[1]);
    case opcode::MUL:
    case opcode::MUL_OVERFLOW: return arg_scales[0] + arg_scales[1];
    case opcode::RESCALE: return target_scale.value();
    default: return arg_scales[0];
  }
}

enum class [[nodiscard]] type : uint64_t {
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
  SIGNED_INTEGERS        = INT8 | INT16 | INT32 | INT64,
  UNSIGNED_INTEGERS      = UINT8 | UINT16 | UINT32 | UINT64,
  INTEGERS               = SIGNED_INTEGERS | UNSIGNED_INTEGERS,
  FLOATS                 = FLOAT32 | FLOAT64,
  DECIMALS               = DECIMAL32 | DECIMAL64 | DECIMAL128,
  ARITHMETIC             = INTEGERS | FLOATS | DECIMALS,
  SIGNED_ARITHMETIC      = SIGNED_INTEGERS | FLOATS | DECIMALS,
  ALL                    = 0x0FFFFFFF,
  ARG_MASK               = 0x10000000,
  ARG0                   = 0x10000000,
  ARG1                   = 0x10000001,
  ARG2                   = 0x10000002,
  ARG3                   = 0x10000003,
  INPUT                  = 0x20000000,
};

constexpr type operator|(type lhs, type rhs)
{
  return static_cast<type>(static_cast<uint64_t>(lhs) | static_cast<uint64_t>(rhs));
}

constexpr type operator&(type lhs, type rhs)
{
  return static_cast<type>(static_cast<uint64_t>(lhs) & static_cast<uint64_t>(rhs));
}

constexpr type operator~(type t) { return static_cast<type>(~static_cast<uint64_t>(t)); }

struct [[nodiscard]] op_type {
  type output                             = type::NONE;
  cuda::std::inplace_vector<type, 4> args = {};
};

/**
 * @brief Get the typing information for a given operator
 * This function returns the expected input and output types for a given operator. The typing
 * information can be used for type checking and inference when constructing expression trees.
 * @param op The operator for which to get the typing information
 * @return An `op_typing` struct containing the expected output type and input types for the
 * operator
 */
[[nodiscard]] op_type get_op_type(opcode op)
{
  switch (op) {
    case opcode::GET_INPUT: return {type::INPUT, {}};
    case opcode::SET_OUTPUT: return {type::NONE, {type::ALL}};
    case opcode::IDENTITY: return {type::ARG0, {type::ALL}};
    case opcode::IS_NULL: return {type::BOOL8, {type::ALL}};
    case opcode::COALESCE: return {type::ARG0, {type::ALL, type::ARG0}};
    case opcode::PREDICATE: return {type::ARG0, {type::BOOL8}};
    case opcode::ABS:
    case opcode::NEG:
    case opcode::ABS_OVERFLOW:
    case opcode::NEG_OVERFLOW: return {type::ARG0, {type::ARITHMETIC}};
    case opcode::FLOOR_DIV: return {type::ARG0, {type{type::FLOATS | type::INTEGERS}, type::ARG0}};
    case opcode::TRUE_DIV:
      return {type::FLOAT64, {type{type::FLOATS | type::INTEGERS}, type::ARG0}};
    case opcode::ADD:
    case opcode::DIV:
    case opcode::MOD:
    case opcode::PYMOD:
    case opcode::MUL:
    case opcode::SUB:
    case opcode::ADD_OVERFLOW:
    case opcode::SUB_OVERFLOW:
    case opcode::MUL_OVERFLOW:
    case opcode::DIV_OVERFLOW:
    case opcode::MOD_OVERFLOW: return {type::ARG0, {type::ARITHMETIC, type::ARG0}};
    case opcode::CHECK_PRECISION: return {type::ARG0, {type::DECIMALS, type::INT32}};
    case opcode::BITWISE_AND:
    case opcode::BITWISE_INVERT:
    case opcode::BITWISE_OR:
    case opcode::BITWISE_XOR:
    case opcode::BITWISE_SHIFT_LEFT:
    case opcode::BITWISE_SHIFT_RIGHT: return {type::ARG0, {type::INTEGERS, type::ARG0}};
    case opcode::CAST_TO_BOOL8: return {type::BOOL8, {type{type::ARITHMETIC | type::BOOL8}}};
    case opcode::CAST_TO_INT8: return {type::INT8, {type{type::ARITHMETIC | type::BOOL8}}};
    case opcode::CAST_TO_INT16: return {type::INT16, {type{type::ARITHMETIC | type::BOOL8}}};
    case opcode::CAST_TO_INT32: return {type::INT32, {type{type::ARITHMETIC | type::BOOL8}}};
    case opcode::CAST_TO_INT64: return {type::INT64, {type{type::ARITHMETIC | type::BOOL8}}};
    case opcode::CAST_TO_UINT8: return {type::UINT8, {type{type::ARITHMETIC | type::BOOL8}}};
    case opcode::CAST_TO_UINT16: return {type::UINT16, {type{type::ARITHMETIC | type::BOOL8}}};
    case opcode::CAST_TO_UINT32: return {type::UINT32, {type{type::ARITHMETIC | type::BOOL8}}};
    case opcode::CAST_TO_UINT64: return {type::UINT64, {type{type::ARITHMETIC | type::BOOL8}}};
    case opcode::CAST_TO_FLOAT32: return {type::FLOAT32, {type{type::ARITHMETIC | type::BOOL8}}};
    case opcode::CAST_TO_FLOAT64: return {type::FLOAT64, {type{type::ARITHMETIC | type::BOOL8}}};
    case opcode::CAST_TO_DECIMAL32: return {type::DECIMAL32, {type::DECIMALS}};
    case opcode::CAST_TO_DECIMAL64: return {type::DECIMAL64, {type::DECIMALS}};
    case opcode::CAST_TO_DECIMAL128: return {type::DECIMAL128, {type::DECIMALS}};
    case opcode::RESCALE: return {type::ARG0, {type::DECIMALS, type::INT32}};
    case opcode::EQUAL:
    case opcode::GREATER:
    case opcode::GREATER_EQUAL:
    case opcode::LESS:
    case opcode::LESS_EQUAL:
    case opcode::NOT_EQUAL:
    case opcode::NULL_EQUAL: return {type::BOOL8, {type::ALL, type::ARG0}};
    case opcode::NULL_LOGICAL_AND:
    case opcode::NULL_LOGICAL_OR:
    case opcode::LOGICAL_AND:
    case opcode::LOGICAL_OR:
      return {type::BOOL8, {type{type::ARITHMETIC | type::BOOL8}, type::ARG0}};
    case opcode::LOGICAL_NOT: return {type::BOOL8, {type{type::ARITHMETIC | type::BOOL8}}};
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
    default: CUDF_FAIL(std::format("Invalid opcode: {}", static_cast<int>(op)), std::runtime_error);
  }
}

[[nodiscard]] size_t get_op_arity(opcode op) { return get_op_type(op).args.size(); }

opcode as_opcode(ast::ast_operator op)
{
  switch (op) {
    case ast::ast_operator::ADD: return opcode::ADD;
    case ast::ast_operator::SUB: return opcode::SUB;
    case ast::ast_operator::MUL: return opcode::MUL;
    case ast::ast_operator::DIV: return opcode::DIV;
    case ast::ast_operator::TRUE_DIV: return opcode::TRUE_DIV;
    case ast::ast_operator::FLOOR_DIV: return opcode::FLOOR_DIV;
    case ast::ast_operator::MOD: return opcode::MOD;
    case ast::ast_operator::PYMOD: return opcode::PYMOD;
    case ast::ast_operator::POW: return opcode::POW;
    case ast::ast_operator::EQUAL: return opcode::EQUAL;
    case ast::ast_operator::NULL_EQUAL: return opcode::NULL_EQUAL;
    case ast::ast_operator::NOT_EQUAL: return opcode::NOT_EQUAL;
    case ast::ast_operator::LESS: return opcode::LESS;
    case ast::ast_operator::GREATER: return opcode::GREATER;
    case ast::ast_operator::LESS_EQUAL: return opcode::LESS_EQUAL;
    case ast::ast_operator::GREATER_EQUAL: return opcode::GREATER_EQUAL;
    case ast::ast_operator::BITWISE_AND: return opcode::BITWISE_AND;
    case ast::ast_operator::BITWISE_OR: return opcode::BITWISE_OR;
    case ast::ast_operator::BITWISE_XOR: return opcode::BITWISE_XOR;
    case ast::ast_operator::LOGICAL_AND: return opcode::LOGICAL_AND;
    case ast::ast_operator::NULL_LOGICAL_AND: return opcode::NULL_LOGICAL_AND;
    case ast::ast_operator::LOGICAL_OR: return opcode::LOGICAL_OR;
    case ast::ast_operator::NULL_LOGICAL_OR: return opcode::NULL_LOGICAL_OR;
    case ast::ast_operator::IDENTITY: return opcode::IDENTITY;
    case ast::ast_operator::IS_NULL: return opcode::IS_NULL;
    case ast::ast_operator::SIN: return opcode::SIN;
    case ast::ast_operator::COS: return opcode::COS;
    case ast::ast_operator::TAN: return opcode::TAN;
    case ast::ast_operator::ARCSIN: return opcode::ARCSIN;
    case ast::ast_operator::ARCCOS: return opcode::ARCCOS;
    case ast::ast_operator::ARCTAN: return opcode::ARCTAN;
    case ast::ast_operator::SINH: return opcode::SINH;
    case ast::ast_operator::COSH: return opcode::COSH;
    case ast::ast_operator::TANH: return opcode::TANH;
    case ast::ast_operator::ARCSINH: return opcode::ARCSINH;
    case ast::ast_operator::ARCCOSH: return opcode::ARCCOSH;
    case ast::ast_operator::ARCTANH: return opcode::ARCTANH;
    case ast::ast_operator::EXP: return opcode::EXP;
    case ast::ast_operator::LOG: return opcode::LOG;
    case ast::ast_operator::SQRT: return opcode::SQRT;
    case ast::ast_operator::CBRT: return opcode::CBRT;
    case ast::ast_operator::CEIL: return opcode::CEIL;
    case ast::ast_operator::FLOOR: return opcode::FLOOR;
    case ast::ast_operator::ABS: return opcode::ABS;
    case ast::ast_operator::RINT: return opcode::RINT;
    case ast::ast_operator::BIT_INVERT: return opcode::BITWISE_INVERT;
    case ast::ast_operator::NOT: return opcode::LOGICAL_NOT;
    case ast::ast_operator::CAST_TO_INT64: return opcode::CAST_TO_INT64;
    case ast::ast_operator::CAST_TO_UINT64: return opcode::CAST_TO_UINT64;
    case ast::ast_operator::CAST_TO_FLOAT64: return opcode::CAST_TO_FLOAT64;
    default:
      CUDF_FAIL(std::format("Unrecognized operator type: {}.", static_cast<int>(op)),
                std::runtime_error);
  }
}

row_ir::type as_type(data_type type)
{
  switch (type.id()) {
    case type_id::BOOL8: return type::BOOL8;
    case type_id::INT8: return type::INT8;
    case type_id::INT16: return type::INT16;
    case type_id::INT32: return type::INT32;
    case type_id::INT64: return type::INT64;
    case type_id::UINT8: return type::UINT8;
    case type_id::UINT16: return type::UINT16;
    case type_id::UINT32: return type::UINT32;
    case type_id::UINT64: return type::UINT64;
    case type_id::FLOAT32: return type::FLOAT32;
    case type_id::FLOAT64: return type::FLOAT64;
    case type_id::DECIMAL32: return type::DECIMAL32;
    case type_id::DECIMAL64: return type::DECIMAL64;
    case type_id::DECIMAL128: return type::DECIMAL128;
    case type_id::TIMESTAMP_DAYS: return type::TIMESTAMP_DAYS;
    case type_id::TIMESTAMP_SECONDS: return type::TIMESTAMP_SECONDS;
    case type_id::TIMESTAMP_MILLISECONDS: return type::TIMESTAMP_MILLISECONDS;
    case type_id::TIMESTAMP_MICROSECONDS: return type::TIMESTAMP_MICROSECONDS;
    case type_id::TIMESTAMP_NANOSECONDS: return type::TIMESTAMP_NANOSECONDS;
    case type_id::DURATION_DAYS: return type::DURATION_DAYS;
    case type_id::DURATION_SECONDS: return type::DURATION_SECONDS;
    case type_id::DURATION_MILLISECONDS: return type::DURATION_MILLISECONDS;
    case type_id::DURATION_MICROSECONDS: return type::DURATION_MICROSECONDS;
    case type_id::DURATION_NANOSECONDS: return type::DURATION_NANOSECONDS;
    case type_id::STRING: return type::STRING;
    default:
      CUDF_FAIL(std::format("Unsupported data type for Row IR: {}", type_to_name(type)),
                std::invalid_argument);
  }
}

type_id as_type_id(type type)
{
  switch (type) {
    case type::BOOL8: return type_id::BOOL8;
    case type::INT8: return type_id::INT8;
    case type::INT16: return type_id::INT16;
    case type::INT32: return type_id::INT32;
    case type::INT64: return type_id::INT64;
    case type::UINT8: return type_id::UINT8;
    case type::UINT16: return type_id::UINT16;
    case type::UINT32: return type_id::UINT32;
    case type::UINT64: return type_id::UINT64;
    case type::FLOAT32: return type_id::FLOAT32;
    case type::FLOAT64: return type_id::FLOAT64;
    case type::DECIMAL32: return type_id::DECIMAL32;
    case type::DECIMAL64: return type_id::DECIMAL64;
    case type::DECIMAL128: return type_id::DECIMAL128;
    case type::TIMESTAMP_DAYS: return type_id::TIMESTAMP_DAYS;
    case type::TIMESTAMP_SECONDS: return type_id::TIMESTAMP_SECONDS;
    case type::TIMESTAMP_MILLISECONDS: return type_id::TIMESTAMP_MILLISECONDS;
    case type::TIMESTAMP_MICROSECONDS: return type_id::TIMESTAMP_MICROSECONDS;
    case type::TIMESTAMP_NANOSECONDS: return type_id::TIMESTAMP_NANOSECONDS;
    case type::DURATION_DAYS: return type_id::DURATION_DAYS;
    case type::DURATION_SECONDS: return type_id::DURATION_SECONDS;
    case type::DURATION_MILLISECONDS: return type_id::DURATION_MILLISECONDS;
    case type::DURATION_MICROSECONDS: return type_id::DURATION_MICROSECONDS;
    case type::DURATION_NANOSECONDS: return type_id::DURATION_NANOSECONDS;
    case type::STRING: return type_id::STRING;
    default:
      CUDF_FAIL(std::format("Invalid typing for {}: {}", __FUNCTION__, static_cast<int>(type)),
                std::invalid_argument);
  }
}

data_type get_return_type(opcode op,
                          std::span<data_type const> args,
                          std::optional<int32_t> target_scale)
{
  std::vector<row_ir::type> arg_types;
  std::vector<int32_t> arg_scales;

  for (auto& type : args) {
    arg_types.emplace_back(as_type(type));
    arg_scales.emplace_back(type.scale());
  }

  auto op_type_match = get_op_type(op);
  auto rescaled      = get_op_output_scale(op, arg_scales, target_scale);

  for (size_t i = 0; i < args.size(); ++i) {
    auto required_type = op_type_match.args[i];
    auto arg_type      = arg_types[i];

    if ((required_type & type::ARG_MASK) != type::NONE) {
      auto src_index = static_cast<size_t>(required_type & ~type::ARG_MASK);
      CUDF_EXPECTS(
        src_index < i,
        std::format(
          "Invalid type match rule for operator `{}` at argument #{}", get_opcode_name(op), i),
        std::runtime_error);
      CUDF_EXPECTS(args[i].id() == args[src_index].id(),
                   std::format("Argument #{} of operator `{}` does not match type of argument "
                               "#{}. Got `{}`, expected `{}`",
                               i,
                               get_opcode_name(op),
                               src_index,
                               type_to_name(args[i]),
                               type_to_name(args[src_index])));
    } else {
      CUDF_EXPECTS(
        (arg_type & required_type) != type::NONE,
        std::format("Argument #{} of operator `{}` does not match expected types. Got {}",
                    i,
                    get_opcode_name(op),
                    type_to_name(args[i])));
    }
  }

  if ((op_type_match.output & type::ARG_MASK) != type::NONE) {
    auto arg_index = static_cast<size_t>(op_type_match.output & ~type::ARG_MASK);
    auto type      = args[arg_index].id();
    auto scale     = numeric::scale_type{is_fixed_point(data_type{type}) ? rescaled : 0};
    return data_type{type, scale};
  } else {
    CUDF_EXPECTS(
      op_type_match.output != type::NONE,
      std::format("Invalid type match rule for operator `{}` return type", get_opcode_name(op)),
      std::runtime_error);
    auto type  = as_type_id(op_type_match.output);
    auto scale = numeric::scale_type{is_fixed_point(data_type{type}) ? rescaled : 0};
    return data_type{type, scale};
  }
}

int32_t instance_context::add_output()
{
  auto id     = static_cast<int32_t>(output_vars_.size());
  auto id_str = std::format("out_{}", id);
  output_vars_.emplace_back(std::move(id_str));
  return id;
}

int32_t instance_context::add_input(input in)
{
  auto id     = static_cast<int32_t>(inputs_.size());
  auto id_str = std::format("in_{}", id);

  data_type const type = [&in] {
    if (auto* col = std::get_if<column_input>(&in)) {
      return col->column.type();
    } else {
      auto& scalar = std::get<scalar_input>(in);
      return scalar.scalar_column->type();
    }
  }();
  inputs_.emplace_back(std::move(in));
  input_vars_.emplace_back(std::move(id_str), type);
  return id;
}

std::string instance_context::make_tmp_id()
{
  return std::format("{}{}", tmp_prefix_, num_tmp_vars_++);
}

bool instance_context::has_nulls() const { return has_nulls_; }

void instance_context::set_has_nulls(bool has_nulls) { has_nulls_ = has_nulls; }

std::span<input const> instance_context::get_inputs() const { return inputs_; }

std::span<var_info const> instance_context::get_input_vars() const { return input_vars_; }

std::span<untyped_var_info const> instance_context::get_output_vars() const { return output_vars_; }

node::node(opcode op,
           std::optional<int32_t> target_scale,
           bool nullify_on_error,
           std::vector<std::unique_ptr<node>> args)
  : op_{op},
    target_scale_{target_scale},
    nullify_on_error_{nullify_on_error},
    args_{std::move(args)}
{
  CUDF_EXPECTS(op_ != opcode::GET_INPUT && op_ != opcode::SET_OUTPUT,
               std::format("Invalid opcode `{}` for operation node.", get_opcode_name(op_)),
               std::runtime_error);

  if (op != opcode::RESCALE) {
    auto expected_arity = get_op_arity(op_);
    auto actual_arity   = args_.size();
    CUDF_EXPECTS(actual_arity == expected_arity,
                 std::format("Invalid number of arguments for operator `{}`. Expected {}, Got {}.",
                             get_opcode_name(op_),
                             expected_arity,
                             actual_arity),
                 std::runtime_error);
  } else {
    CUDF_EXPECTS(args_.size() == 1,
                 std::format("RESCALE operator expects exactly 1 argument. Got {}.", args_.size()));
    CUDF_EXPECTS(
      target_scale_.has_value(),
      std::format("Target scale must be provided for RESCALE operator and must be nullopt "
                  "for other operators."));
  }
}

node::node(input_reference input)
  : reference_{input}, op_{opcode::GET_INPUT}  // NOLINT(modernize-use-default-member-init)
{
}

node::node(output_reference reference, std::unique_ptr<node> arg)
  : reference_{reference}, op_{opcode::SET_OUTPUT}
{
  args_.emplace_back(std::move(arg));
}

node::node(output_reference reference, node arg)
  : node{reference, std::make_unique<node>(std::move(arg))}
{
}

std::string_view node::get_id() const { return id_; }

data_type node::get_type() const { return type_; }

std::optional<int32_t> node::get_target_scale() const { return target_scale_; }

opcode node::get_opcode() const { return op_; }

std::span<std::unique_ptr<node> const> node::get_args() const { return args_; }

bool node::is_null_aware() const
{
  if (op_ == opcode::GET_INPUT) { return false; }

  // to emit nulls for always-nullable operators, we  need to mark them as null-aware
  if (get_op_null_output(op_, nullify_on_error_) == null_output::ALWAYS_NULLABLE) { return true; }

  if (get_op_is_null_dependent(op_)) { return true; }

  CUDF_EXPECTS(!args_.empty(),
               "Unexpectedly found an operator node with no arguments. All operator nodes should "
               "have at least one argument.",
               std::runtime_error);

  return std::any_of(args_.begin(), args_.end(), [](auto& a) { return a->is_null_aware(); });
}

bool node::is_always_valid() const
{
  if (op_ == opcode::GET_INPUT) { return false; }

  if (get_op_null_output(op_, nullify_on_error_) == null_output::ALWAYS_VALID) { return true; }

  CUDF_EXPECTS(!args_.empty(),
               "Unexpectedly found an operator node with no arguments. All operator nodes should "
               "have at least one argument.",
               std::runtime_error);

  return std::all_of(args_.begin(), args_.end(), [](auto& a) { return a->is_always_valid(); });
}

std::string to_cuda_type(cudf::data_type type, bool nullable)
{
  auto name = type_to_name(type);
  return nullable ? std::format("cuda::std::optional<{}>", name) : name;
}

void node::instantiate(instance_context& ctx)
{
  for (auto& arg : args_) {
    arg->instantiate(ctx);
  }

  id_ = ctx.make_tmp_id();

  switch (op_) {
    case opcode::GET_INPUT: {
      type_ = ctx.get_input_vars()[std::get<input_reference>(reference_).index].type;
    } break;
    case opcode::SET_OUTPUT: {
      type_ = args_[0]->get_type();
    } break;
    default: {
      std::vector<data_type> arg_types;
      for (auto& arg : args_) {
        arg_types.emplace_back(arg->get_type());
      }

      if (op_ == opcode::RESCALE) {
        scale_reference_ =
          input_reference{ctx.add_input(cudf::numeric_scalar<int32_t>{target_scale_.value_or(0)})};
        arg_types.emplace_back(cudf::type_id::INT32);
      }

      type_ = get_return_type(op_, arg_types, target_scale_);
    } break;
  }
}

void node::emit_code(instance_context& instance, target_info const& info, code_sink& sink) const
{
  for (auto& arg : args_) {
    arg->emit_code(instance, info, sink);
  }

  switch (info.id) {
    case target::CUDA: {
      auto type = to_cuda_type(type_, instance.has_nulls());

      switch (op_) {
        case opcode::GET_INPUT: {
          sink.emit(
            std::format(R"***({} {} = {};
)***",
                        type,
                        id_,
                        instance.get_input_vars()[std::get<input_reference>(reference_).index].id));
        } break;

        case opcode::SET_OUTPUT: {
          sink.emit(std::format(
            R"***({} {} = {};
*{} = {};
)***",
            type,
            id_,
            args_[0]->get_id(),
            instance.get_output_vars()[std::get<output_reference>(reference_).index].id,
            id_));
        } break;

        default: {
          auto first_arg = std::format("{}", args_[0]->get_id());
          auto args_str  = (args_.size() == 1)
                             ? std::string{first_arg}
                             : std::accumulate(args_.begin() + 1,
                                              args_.end(),
                                              std::string{first_arg},
                                              [](auto const& a, auto& node) {
                                                return std::format("{}, {}", a, node->get_id());
                                              });

          if (op_ == opcode::RESCALE) {
            args_str =
              std::format("{}, {}", args_str, instance.get_input_vars()[scale_reference_.index].id);
          }

          sink.emit(std::format(
            R"***({} {} = cudf::detail::row_ir::evaluate<cudf::detail::row_ir::opcode::{}, {}>(&error_flag, {});
)***",
            type,
            id_,
            get_opcode_name(op_),
            nullify_on_error_,
            args_str));

          if (get_op_is_fallible(op_, nullify_on_error_)) {
            sink.emit(R"***(CUDF_CHECK_OPCODE_ERROR_FLAG(error_flag);
)***");
          }
        }
      }
      break;
    }

    default:
      CUDF_FAIL(std::format("Unsupported target: {}", static_cast<int>(info.id)),
                std::invalid_argument);
  }
}

std::unique_ptr<row_ir::node> ast_converter::add_ir_node(ast::literal const& expr)
{
  auto id = instance_.add_input(expr.get_scalar());
  return std::make_unique<row_ir::node>(input_reference{id});
}

std::unique_ptr<row_ir::node> ast_converter::add_ir_node(ast::column_reference const& expr)
{
  // resolve the table for a column input spec, preferring left_table/right_table for join cases,
  // falling back to args.table for the single-table case.
  auto resolve = [&](ast::table_reference ref) {
    CUDF_EXPECTS(ref == ast::table_reference::LEFT || ref == ast::table_reference::RIGHT,
                 "Invalid table reference in column expression",
                 std::invalid_argument);
    return ref == ast::table_reference::LEFT ? left_table_ : right_table_;
  };

  auto table = resolve(expr.get_table_source());
  auto id    = instance_.add_input(
    column_input{.column       = table.column(expr.get_column_index()),
                    .table_source = (expr.get_table_source() == ast::table_reference::LEFT ? 0 : 1),
                    .column_index = static_cast<int32_t>(expr.get_column_index())});
  return std::make_unique<row_ir::node>(input_reference{id});
}

std::unique_ptr<row_ir::node> ast_converter::add_ir_node(ast::operation const& expr)
{
  std::vector<std::unique_ptr<row_ir::node>> args;
  for (auto& operand : expr.get_operands()) {
    args.emplace_back(operand.get().accept(*this));
  }
  return std::make_unique<row_ir::node>(
    as_opcode(expr.get_operator()), std::nullopt, false, std::move(args));
}

std::unique_ptr<row_ir::node> ast_converter::add_ir_node(ast::detail::predicate const& expr)
{
  return std::make_unique<row_ir::node>(
    row_ir::opcode::PREDICATE, std::nullopt, false, expr.get_operand().accept(*this));
}

std::unique_ptr<row_ir::node> ast_converter::add_ir_node(ast::jit::detail::operation const& expr)
{
  std::vector<std::unique_ptr<row_ir::node>> args;
  for (auto& arg : expr.get_arguments()) {
    args.emplace_back(arg.get().accept(*this));
  }
  return std::make_unique<row_ir::node>(
    expr.get_opcode(), expr.get_target_scale(), expr.nullify_on_error(), std::move(args));
}

bool is_nullable(scalar_input const& in) { return in.scalar_column->view().nullable(); }

bool is_nullable(column_input const& in) { return in.column.nullable(); }

std::tuple<std::string, null_aware, output_nullability> ast_converter::generate_code(
  target target_id, ast::expression const& expr, std::string_view function_name)
{
  // add 1 auto-deduced output variable
  [[maybe_unused]] auto output_id = instance_.add_output();

  output_irs_.emplace_back(std::make_unique<row_ir::node>(output_reference{0}, expr.accept(*this)));

  bool has_nullable_inputs =
    std::any_of(instance_.inputs_.begin(), instance_.inputs_.end(), [&](auto& in) {
      return std::visit([](auto& c) { return is_nullable(c); }, in);
    });

  bool is_null_aware = std::any_of(
    output_irs_.cbegin(), output_irs_.cend(), [](auto& ir) { return ir->is_null_aware(); });

  bool output_is_always_valid = std::all_of(
    output_irs_.cbegin(), output_irs_.cend(), [](auto& ir) { return ir->is_always_valid(); });

  bool may_evaluate_null = output_is_always_valid ? false : (has_nullable_inputs || is_null_aware);

  auto null_policy =
    may_evaluate_null ? output_nullability::PRESERVE : output_nullability::ALL_VALID;

  instance_.set_has_nulls(is_null_aware);

  // instantiate the IR nodes
  for (auto& ir : output_irs_) {
    ir->instantiate(instance_);
  }

  target_info target{target_id};

  CUDF_EXPECTS(
    target.id == target::CUDA, "Unsupported target for code generation", std::invalid_argument);

  auto output_decl = [&](auto i) {
    auto& var = instance_.output_vars_[i];
    auto& ir  = output_irs_[i];
    return std::format("{}* {}", to_cuda_type(ir->get_type(), instance_.has_nulls()), var.id);
  };

  auto input_decl = [&](auto i) {
    auto& var = instance_.input_vars_[i];
    return std::format("{} {}", to_cuda_type(var.type, instance_.has_nulls()), var.id);
  };

  std::vector<std::string> arg_decls;

  for (size_t i = 0; i < instance_.output_vars_.size(); ++i) {
    arg_decls.emplace_back(output_decl(i));
  }

  for (size_t i = 0; i < instance_.input_vars_.size(); ++i) {
    arg_decls.emplace_back(input_decl(i));
  }

  auto args_decl = [&] {
    if (arg_decls.empty()) {
      return std::string{};
    } else if (arg_decls.size() == 1) {
      return arg_decls[0];
    } else {
      return std::accumulate(
        arg_decls.begin() + 1, arg_decls.end(), arg_decls[0], [](auto const& a, auto const& b) {
          return std::format("{}, {}", a, b);
        });
    }
  }();

  code_sink sink;
  sink.emit(std::format("__device__ cudf::errc {}(", function_name));
  sink.emit(args_decl);
  sink.emit(")\n{\n");
  sink.emit("[[maybe_unused]] cudf::errc error_flag = cudf::errc::SUCCESS;\n");
  for (auto& ir : output_irs_) {
    ir->emit_code(instance_, target, sink);
  }
  sink.emit("return cudf::errc::SUCCESS;\n}");
  return {sink.get_code(), is_null_aware ? null_aware::YES : null_aware::NO, null_policy};
}

std::variant<column_view, scalar_column_view> get_column_view(scalar_input const& in)
{
  return scalar_column_view{in.scalar_column->view()};
}

std::variant<column_view, scalar_column_view> get_column_view(column_input const& in)
{
  return column_view{in.column};
}

// Due to the AST expression tree structure, we can't generate the IR without the target
// tables
transform_args ast_converter::compute_column(target target_id,
                                             ast::expression const& expr,
                                             table_view const& left_table,
                                             table_view const& right_table,
                                             std::string_view function_name,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr)
{
  ast_converter converter{stream, mr, left_table, right_table};

  // TODO(lamarrr): consider deduplicating ast expression's input column references. See
  // TransformTest/1.DeeplyNestedArithmeticLogicalExpression for reference

  auto [code, is_null_aware, output_nullability] =
    converter.generate_code(target_id, expr, function_name);
  std::vector<std::variant<column_view, scalar_column_view>> inputs;
  std::vector<std::unique_ptr<column>> scalar_columns;
  std::vector<std::optional<int32_t>> table_sources;
  std::vector<std::optional<int32_t>> column_indices;

  for (auto& input : converter.instance_.inputs_) {
    if (std::holds_alternative<column_input>(input)) {
      auto& col = std::get<column_input>(input);
      table_sources.emplace_back(col.table_source);
      column_indices.emplace_back(col.column_index);
    } else {
      table_sources.emplace_back(std::nullopt);
      column_indices.emplace_back(std::nullopt);
    }

    auto view = std::visit([](auto& in) { return get_column_view(in); }, input);
    inputs.emplace_back(view);

    if (std::holds_alternative<scalar_input>(input)) {
      auto& scalar = std::get<scalar_input>(input);
      scalar_columns.emplace_back(std::move(scalar.scalar_column));
    }
  }

  auto& out               = converter.output_irs_[0];
  auto output_column_type = out->get_type();
  auto output   = transform_output{.type = output_column_type, .nullability = output_nullability};
  auto row_size = std::max({left_table.num_rows(), right_table.num_rows()});
  auto result   = transform_args{.scalar_columns       = std::move(scalar_columns),
                                 .input_table_sources  = std::move(table_sources),
                                 .input_column_indices = std::move(column_indices),
                                 .udf                  = std::move(code),
                                 .source_type          = cudf::udf_source_type::CUDA,
                                 .is_null_aware        = is_null_aware,
                                 .user_data            = std::nullopt,
                                 .inputs               = inputs,
                                 .outputs{output},
                                 .string_offsets{},
                                 .row_size = row_size};
  if (get_context().dump_codegen()) {
    std::cout << "Generated code for transform: \n" << result.udf << std::endl;
  }

  return result;
}

transform_args ast_converter::filter(target target_id,
                                     ast::expression const& expr,
                                     table_view const& left_table,
                                     table_view const& right_table,
                                     std::string_view function_name,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
{
  auto filter = ast::detail::predicate{expr};
  return compute_column(target_id, filter, left_table, right_table, function_name, stream, mr);
}

}  // namespace cudf::detail::row_ir
