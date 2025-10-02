/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/ast/detail/operator_functor.cuh>
#include <cudf/jit/lto/operators.cuh>
#include <cudf/jit/lto/thunk.cuh>
#include <cudf/jit/lto/types.cuh>

namespace CUDF_EXPORT cudf {
namespace lto {

template <ast::ast_operator op>
using func = ast::detail::operator_functor<op, false>;

template <ast::ast_operator op>
using null_func = ast::detail::operator_functor<op, true>;

using opcode = ast::ast_operator;

#define CUDF_LTO_DEF(op, OP, type)                                                        \
  __device__ void operators::op(type* out, type const* a, type const* b)                  \
  {                                                                                       \
    auto ret = func<opcode::OP>{}(*lto::lift(a), *lto::lift(b));                          \
    *out     = *lto::lower(&ret);                                                         \
  }                                                                                       \
                                                                                          \
  __device__ void operators::op(                                                          \
    lto::optional<type>* out, lto::optional<type> const* a, lto::optional<type> const* b) \
  {                                                                                       \
    auto ret = null_func<opcode::OP>{}(*lto::lift(a), *lto::lift(b));                     \
    *out     = *lto::lower(&ret);                                                         \
  }

CUDF_LTO_DEF(add, ADD, int32_t);
CUDF_LTO_DEF(add, ADD, int64_t);
CUDF_LTO_DEF(add, ADD, uint32_t);
CUDF_LTO_DEF(add, ADD, uint64_t);
CUDF_LTO_DEF(add, ADD, float32_t);
CUDF_LTO_DEF(add, ADD, float64_t);
CUDF_LTO_DEF(add, ADD, decimal32);
CUDF_LTO_DEF(add, ADD, decimal64);
CUDF_LTO_DEF(add, ADD, decimal128);
CUDF_LTO_DEF(add, ADD, duration_D);
CUDF_LTO_DEF(add, ADD, duration_s);
CUDF_LTO_DEF(add, ADD, duration_ms);
CUDF_LTO_DEF(add, ADD, duration_ns);

CUDF_LTO_DEF(sub, SUB, int32_t);
CUDF_LTO_DEF(sub, SUB, int64_t);
CUDF_LTO_DEF(sub, SUB, uint32_t);
CUDF_LTO_DEF(sub, SUB, uint64_t);
CUDF_LTO_DEF(sub, SUB, float32_t);
CUDF_LTO_DEF(sub, SUB, float64_t);
CUDF_LTO_DEF(sub, SUB, decimal32);
CUDF_LTO_DEF(sub, SUB, decimal64);
CUDF_LTO_DEF(sub, SUB, decimal128);
CUDF_LTO_DEF(sub, SUB, duration_D);
CUDF_LTO_DEF(sub, SUB, duration_s);
CUDF_LTO_DEF(sub, SUB, duration_ms);
CUDF_LTO_DEF(sub, SUB, duration_ns);

CUDF_LTO_DEF(mul, MUL, int32_t);
CUDF_LTO_DEF(mul, MUL, int64_t);
CUDF_LTO_DEF(mul, MUL, uint32_t);
CUDF_LTO_DEF(mul, MUL, uint64_t);
CUDF_LTO_DEF(mul, MUL, float32_t);
CUDF_LTO_DEF(mul, MUL, float64_t);
CUDF_LTO_DEF(mul, MUL, decimal32);
CUDF_LTO_DEF(mul, MUL, decimal64);
CUDF_LTO_DEF(mul, MUL, decimal128);

CUDF_LTO_DEF(div, DIV, int32_t);
CUDF_LTO_DEF(div, DIV, int64_t);
CUDF_LTO_DEF(div, DIV, uint32_t);
CUDF_LTO_DEF(div, DIV, uint64_t);
CUDF_LTO_DEF(div, DIV, float32_t);
CUDF_LTO_DEF(div, DIV, float64_t);
CUDF_LTO_DEF(div, DIV, decimal32);
CUDF_LTO_DEF(div, DIV, decimal64);
CUDF_LTO_DEF(div, DIV, decimal128);

CUDF_LTO_DEF(mod, MOD, float32_t);
CUDF_LTO_DEF(mod, MOD, float64_t);

CUDF_LTO_DEF(pymod, PYMOD, float32_t);
CUDF_LTO_DEF(pymod, PYMOD, float64_t);

CUDF_LTO_DEF(pow, POW, float32_t);
CUDF_LTO_DEF(pow, POW, float64_t);

#undef CUDF_LTO_DEF

#define CUDF_LTO_DEF(op, OP, type)                                                           \
  __device__ void operators::op(bool* out, type const* a, type const* b)                     \
  {                                                                                          \
    auto ret = func<opcode::OP>{}(*lto::lift(a), *lto::lift(b));                             \
    *out     = *lto::lower(&ret);                                                            \
  }                                                                                          \
                                                                                             \
  __device__ void operators::op(bool* out, optional<type> const* a, optional<type> const* b) \
  {                                                                                          \
    auto ret = null_func<opcode::OP>{}(*lto::lift(a), *lto::lift(b));                        \
    *out     = **lto::lower(&ret);                                                           \
  }                                                                                          \
                                                                                             \
  __device__ void operators::op(                                                             \
    optional<bool>* out, optional<type> const* a, optional<type> const* b)                   \
  {                                                                                          \
    auto ret = null_func<opcode::OP>{}(*lto::lift(a), *lto::lift(b));                        \
    *out     = *lto::lower(&ret);                                                            \
  }

CUDF_LTO_DEF(equal, EQUAL, bool);
CUDF_LTO_DEF(equal, EQUAL, int8_t);
CUDF_LTO_DEF(equal, EQUAL, int16_t);
CUDF_LTO_DEF(equal, EQUAL, int32_t);
CUDF_LTO_DEF(equal, EQUAL, int64_t);
CUDF_LTO_DEF(equal, EQUAL, uint8_t);
CUDF_LTO_DEF(equal, EQUAL, uint16_t);
CUDF_LTO_DEF(equal, EQUAL, uint32_t);
CUDF_LTO_DEF(equal, EQUAL, uint64_t);
CUDF_LTO_DEF(equal, EQUAL, float32_t);
CUDF_LTO_DEF(equal, EQUAL, float64_t);
CUDF_LTO_DEF(equal, EQUAL, decimal32);
CUDF_LTO_DEF(equal, EQUAL, decimal64);
CUDF_LTO_DEF(equal, EQUAL, decimal128);
CUDF_LTO_DEF(equal, EQUAL, timestamp_D);
CUDF_LTO_DEF(equal, EQUAL, timestamp_s);
CUDF_LTO_DEF(equal, EQUAL, timestamp_ms);
CUDF_LTO_DEF(equal, EQUAL, timestamp_us);
CUDF_LTO_DEF(equal, EQUAL, timestamp_ns);
CUDF_LTO_DEF(equal, EQUAL, duration_D);
CUDF_LTO_DEF(equal, EQUAL, duration_s);
CUDF_LTO_DEF(equal, EQUAL, duration_ms);
CUDF_LTO_DEF(equal, EQUAL, duration_ns);
CUDF_LTO_DEF(equal, EQUAL, string_view);

CUDF_LTO_DEF(null_equal, NULL_EQUAL, bool);
CUDF_LTO_DEF(null_equal, NULL_EQUAL, int8_t);
CUDF_LTO_DEF(null_equal, NULL_EQUAL, int16_t);
CUDF_LTO_DEF(null_equal, NULL_EQUAL, int32_t);
CUDF_LTO_DEF(null_equal, NULL_EQUAL, int64_t);
CUDF_LTO_DEF(null_equal, NULL_EQUAL, uint8_t);
CUDF_LTO_DEF(null_equal, NULL_EQUAL, uint16_t);
CUDF_LTO_DEF(null_equal, NULL_EQUAL, uint32_t);
CUDF_LTO_DEF(null_equal, NULL_EQUAL, uint64_t);
CUDF_LTO_DEF(null_equal, NULL_EQUAL, float32_t);
CUDF_LTO_DEF(null_equal, NULL_EQUAL, float64_t);
CUDF_LTO_DEF(null_equal, NULL_EQUAL, decimal32);
CUDF_LTO_DEF(null_equal, NULL_EQUAL, decimal64);
CUDF_LTO_DEF(null_equal, NULL_EQUAL, decimal128);
CUDF_LTO_DEF(null_equal, NULL_EQUAL, timestamp_D);
CUDF_LTO_DEF(null_equal, NULL_EQUAL, timestamp_s);
CUDF_LTO_DEF(null_equal, NULL_EQUAL, timestamp_ms);
CUDF_LTO_DEF(null_equal, NULL_EQUAL, timestamp_us);
CUDF_LTO_DEF(null_equal, NULL_EQUAL, timestamp_ns);
CUDF_LTO_DEF(null_equal, NULL_EQUAL, duration_D);
CUDF_LTO_DEF(null_equal, NULL_EQUAL, duration_s);
CUDF_LTO_DEF(null_equal, NULL_EQUAL, duration_ms);
CUDF_LTO_DEF(null_equal, NULL_EQUAL, duration_ns);
CUDF_LTO_DEF(null_equal, NULL_EQUAL, string_view);

#undef CUDF_LTO_DEF

#define CUDF_LTO_DEF(op, OP, type)                                         \
  __device__ void operators::op(bool* out, type const* a, type const* b)   \
  {                                                                        \
    auto ret = func<opcode::OP>{}(*lto::lift(a), *lto::lift(b));           \
    *out     = *lto::lower(&ret);                                          \
  }                                                                        \
                                                                           \
  __device__ void operators::op(                                           \
    optional<bool>* out, optional<type> const* a, optional<type> const* b) \
  {                                                                        \
    auto ret = null_func<opcode::OP>{}(*lto::lift(a), *lto::lift(b));      \
    *out     = *lto::lower(&ret);                                          \
  }

CUDF_LTO_DEF(less, LESS, bool);
CUDF_LTO_DEF(less, LESS, int8_t);
CUDF_LTO_DEF(less, LESS, int16_t);
CUDF_LTO_DEF(less, LESS, int32_t);
CUDF_LTO_DEF(less, LESS, int64_t);
CUDF_LTO_DEF(less, LESS, uint8_t);
CUDF_LTO_DEF(less, LESS, uint16_t);
CUDF_LTO_DEF(less, LESS, uint32_t);
CUDF_LTO_DEF(less, LESS, uint64_t);
CUDF_LTO_DEF(less, LESS, float32_t);
CUDF_LTO_DEF(less, LESS, float64_t);
CUDF_LTO_DEF(less, LESS, decimal32);
CUDF_LTO_DEF(less, LESS, decimal64);
CUDF_LTO_DEF(less, LESS, decimal128);
CUDF_LTO_DEF(less, LESS, timestamp_D);
CUDF_LTO_DEF(less, LESS, timestamp_s);
CUDF_LTO_DEF(less, LESS, timestamp_ms);
CUDF_LTO_DEF(less, LESS, timestamp_us);
CUDF_LTO_DEF(less, LESS, timestamp_ns);
CUDF_LTO_DEF(less, LESS, duration_D);
CUDF_LTO_DEF(less, LESS, duration_s);
CUDF_LTO_DEF(less, LESS, duration_ms);
CUDF_LTO_DEF(less, LESS, duration_ns);
CUDF_LTO_DEF(less, LESS, string_view);

CUDF_LTO_DEF(greater, GREATER, bool);
CUDF_LTO_DEF(greater, GREATER, int8_t);
CUDF_LTO_DEF(greater, GREATER, int16_t);
CUDF_LTO_DEF(greater, GREATER, int32_t);
CUDF_LTO_DEF(greater, GREATER, int64_t);
CUDF_LTO_DEF(greater, GREATER, uint8_t);
CUDF_LTO_DEF(greater, GREATER, uint16_t);
CUDF_LTO_DEF(greater, GREATER, uint32_t);
CUDF_LTO_DEF(greater, GREATER, uint64_t);
CUDF_LTO_DEF(greater, GREATER, float32_t);
CUDF_LTO_DEF(greater, GREATER, float64_t);
CUDF_LTO_DEF(greater, GREATER, decimal32);
CUDF_LTO_DEF(greater, GREATER, decimal64);
CUDF_LTO_DEF(greater, GREATER, decimal128);
CUDF_LTO_DEF(greater, GREATER, timestamp_D);
CUDF_LTO_DEF(greater, GREATER, timestamp_s);
CUDF_LTO_DEF(greater, GREATER, timestamp_ms);
CUDF_LTO_DEF(greater, GREATER, timestamp_us);
CUDF_LTO_DEF(greater, GREATER, timestamp_ns);
CUDF_LTO_DEF(greater, GREATER, duration_D);
CUDF_LTO_DEF(greater, GREATER, duration_s);
CUDF_LTO_DEF(greater, GREATER, duration_ms);
CUDF_LTO_DEF(greater, GREATER, duration_ns);
CUDF_LTO_DEF(greater, GREATER, string_view);

CUDF_LTO_DEF(less_equal, LESS_EQUAL, bool);
CUDF_LTO_DEF(less_equal, LESS_EQUAL, int8_t);
CUDF_LTO_DEF(less_equal, LESS_EQUAL, int16_t);
CUDF_LTO_DEF(less_equal, LESS_EQUAL, int32_t);
CUDF_LTO_DEF(less_equal, LESS_EQUAL, int64_t);
CUDF_LTO_DEF(less_equal, LESS_EQUAL, uint8_t);
CUDF_LTO_DEF(less_equal, LESS_EQUAL, uint16_t);
CUDF_LTO_DEF(less_equal, LESS_EQUAL, uint32_t);
CUDF_LTO_DEF(less_equal, LESS_EQUAL, uint64_t);
CUDF_LTO_DEF(less_equal, LESS_EQUAL, float32_t);
CUDF_LTO_DEF(less_equal, LESS_EQUAL, float64_t);
CUDF_LTO_DEF(less_equal, LESS_EQUAL, decimal32);
CUDF_LTO_DEF(less_equal, LESS_EQUAL, decimal64);
CUDF_LTO_DEF(less_equal, LESS_EQUAL, decimal128);
CUDF_LTO_DEF(less_equal, LESS_EQUAL, timestamp_D);
CUDF_LTO_DEF(less_equal, LESS_EQUAL, timestamp_s);
CUDF_LTO_DEF(less_equal, LESS_EQUAL, timestamp_ms);
CUDF_LTO_DEF(less_equal, LESS_EQUAL, timestamp_us);
CUDF_LTO_DEF(less_equal, LESS_EQUAL, timestamp_ns);
CUDF_LTO_DEF(less_equal, LESS_EQUAL, duration_D);
CUDF_LTO_DEF(less_equal, LESS_EQUAL, duration_s);
CUDF_LTO_DEF(less_equal, LESS_EQUAL, duration_ms);
CUDF_LTO_DEF(less_equal, LESS_EQUAL, duration_ns);
CUDF_LTO_DEF(less_equal, LESS_EQUAL, string_view);

CUDF_LTO_DEF(greater_equal, GREATER_EQUAL, bool);
CUDF_LTO_DEF(greater_equal, GREATER_EQUAL, int8_t);
CUDF_LTO_DEF(greater_equal, GREATER_EQUAL, int16_t);
CUDF_LTO_DEF(greater_equal, GREATER_EQUAL, int32_t);
CUDF_LTO_DEF(greater_equal, GREATER_EQUAL, int64_t);
CUDF_LTO_DEF(greater_equal, GREATER_EQUAL, uint8_t);
CUDF_LTO_DEF(greater_equal, GREATER_EQUAL, uint16_t);
CUDF_LTO_DEF(greater_equal, GREATER_EQUAL, uint32_t);
CUDF_LTO_DEF(greater_equal, GREATER_EQUAL, uint64_t);
CUDF_LTO_DEF(greater_equal, GREATER_EQUAL, float32_t);
CUDF_LTO_DEF(greater_equal, GREATER_EQUAL, float64_t);
CUDF_LTO_DEF(greater_equal, GREATER_EQUAL, decimal32);
CUDF_LTO_DEF(greater_equal, GREATER_EQUAL, decimal64);
CUDF_LTO_DEF(greater_equal, GREATER_EQUAL, decimal128);
CUDF_LTO_DEF(greater_equal, GREATER_EQUAL, timestamp_D);
CUDF_LTO_DEF(greater_equal, GREATER_EQUAL, timestamp_s);
CUDF_LTO_DEF(greater_equal, GREATER_EQUAL, timestamp_ms);
CUDF_LTO_DEF(greater_equal, GREATER_EQUAL, timestamp_us);
CUDF_LTO_DEF(greater_equal, GREATER_EQUAL, timestamp_ns);
CUDF_LTO_DEF(greater_equal, GREATER_EQUAL, duration_D);
CUDF_LTO_DEF(greater_equal, GREATER_EQUAL, duration_s);
CUDF_LTO_DEF(greater_equal, GREATER_EQUAL, duration_ms);
CUDF_LTO_DEF(greater_equal, GREATER_EQUAL, duration_ns);
CUDF_LTO_DEF(greater_equal, GREATER_EQUAL, string_view);

#undef CUDF_LTO_DEF

#define CUDF_LTO_DEF(op, OP, type)                                         \
  __device__ void operators::op(type* out, type const* a, type const* b)   \
  {                                                                        \
    auto ret = func<opcode::OP>{}(*lto::lift(a), *lto::lift(b));           \
    *out     = *lto::lower(&ret);                                          \
  }                                                                        \
                                                                           \
  __device__ void operators::op(                                           \
    optional<type>* out, optional<type> const* a, optional<type> const* b) \
  {                                                                        \
    auto ret = null_func<opcode::OP>{}(*lto::lift(a), *lto::lift(b));      \
    *out     = *lto::lower(&ret);                                          \
  }

CUDF_LTO_DEF(bitwise_and, BITWISE_AND, int32_t);
CUDF_LTO_DEF(bitwise_and, BITWISE_AND, int64_t);
CUDF_LTO_DEF(bitwise_and, BITWISE_AND, uint32_t);
CUDF_LTO_DEF(bitwise_and, BITWISE_AND, uint64_t);

CUDF_LTO_DEF(bitwise_or, BITWISE_OR, int32_t);
CUDF_LTO_DEF(bitwise_or, BITWISE_OR, int64_t);
CUDF_LTO_DEF(bitwise_or, BITWISE_OR, uint32_t);
CUDF_LTO_DEF(bitwise_or, BITWISE_OR, uint64_t);

CUDF_LTO_DEF(bitwise_xor, BITWISE_XOR, int32_t);
CUDF_LTO_DEF(bitwise_xor, BITWISE_XOR, int64_t);
CUDF_LTO_DEF(bitwise_xor, BITWISE_XOR, uint32_t);
CUDF_LTO_DEF(bitwise_xor, BITWISE_XOR, uint64_t);

#undef CUDF_LTO_DEF

#define CUDF_LTO_DEF(op, OP, type)                                         \
  __device__ void operators::op(type* out, type const* a, type const* b)   \
  {                                                                        \
    auto ret = func<opcode::OP>{}(*lto::lift(a), *lto::lift(b));           \
    *out     = *lto::lower(&ret);                                          \
  }                                                                        \
                                                                           \
  __device__ void operators::op(                                           \
    optional<type>* out, optional<type> const* a, optional<type> const* b) \
  {                                                                        \
    auto ret = null_func<opcode::OP>{}(*lto::lift(a), *lto::lift(b));      \
    *out     = *lto::lower(&ret);                                          \
  }

CUDF_LTO_DEF(logical_and, LOGICAL_AND, bool);

CUDF_LTO_DEF(null_logical_and, NULL_LOGICAL_AND, bool);

CUDF_LTO_DEF(logical_or, LOGICAL_OR, bool);

CUDF_LTO_DEF(null_logical_or, NULL_LOGICAL_OR, bool);

#undef CUDF_LTO_DEF

#define CUDF_LTO_DEF(op, OP, type)                                            \
  __device__ void operators::op(type* out, type const* a)                     \
  {                                                                           \
    auto ret = func<opcode::OP>{}(*lto::lift(a));                             \
    *out     = *lto::lower(&ret);                                             \
  }                                                                           \
                                                                              \
  __device__ void operators::op(optional<type>* out, optional<type> const* a) \
  {                                                                           \
    auto ret = null_func<opcode::OP>{}(*lto::lift(a));                        \
    *out     = *lto::lower(&ret);                                             \
  }

CUDF_LTO_DEF(identity, IDENTITY, bool);
CUDF_LTO_DEF(identity, IDENTITY, int8_t);
CUDF_LTO_DEF(identity, IDENTITY, int16_t);
CUDF_LTO_DEF(identity, IDENTITY, int32_t);
CUDF_LTO_DEF(identity, IDENTITY, int64_t);
CUDF_LTO_DEF(identity, IDENTITY, uint8_t);
CUDF_LTO_DEF(identity, IDENTITY, uint16_t);
CUDF_LTO_DEF(identity, IDENTITY, uint32_t);
CUDF_LTO_DEF(identity, IDENTITY, uint64_t);
CUDF_LTO_DEF(identity, IDENTITY, float32_t);
CUDF_LTO_DEF(identity, IDENTITY, float64_t);
CUDF_LTO_DEF(identity, IDENTITY, decimal32);
CUDF_LTO_DEF(identity, IDENTITY, decimal64);
CUDF_LTO_DEF(identity, IDENTITY, decimal128);
CUDF_LTO_DEF(identity, IDENTITY, timestamp_D);
CUDF_LTO_DEF(identity, IDENTITY, timestamp_s);
CUDF_LTO_DEF(identity, IDENTITY, timestamp_ms);
CUDF_LTO_DEF(identity, IDENTITY, timestamp_us);
CUDF_LTO_DEF(identity, IDENTITY, timestamp_ns);
CUDF_LTO_DEF(identity, IDENTITY, duration_D);
CUDF_LTO_DEF(identity, IDENTITY, duration_s);
CUDF_LTO_DEF(identity, IDENTITY, duration_ms);
CUDF_LTO_DEF(identity, IDENTITY, duration_ns);
CUDF_LTO_DEF(identity, IDENTITY, string_view);

CUDF_LTO_DEF(sin, SIN, float32_t);
CUDF_LTO_DEF(sin, SIN, float64_t);

CUDF_LTO_DEF(cos, COS, float32_t);
CUDF_LTO_DEF(cos, COS, float64_t);

CUDF_LTO_DEF(tan, TAN, float32_t);
CUDF_LTO_DEF(tan, TAN, float64_t);

CUDF_LTO_DEF(arcsin, ARCSIN, float32_t);
CUDF_LTO_DEF(arcsin, ARCSIN, float64_t);

CUDF_LTO_DEF(arccos, ARCCOS, float32_t);
CUDF_LTO_DEF(arccos, ARCCOS, float64_t);

CUDF_LTO_DEF(arctan, ARCTAN, float32_t);
CUDF_LTO_DEF(arctan, ARCTAN, float64_t);

CUDF_LTO_DEF(sinh, SINH, float32_t);
CUDF_LTO_DEF(sinh, SINH, float64_t);

CUDF_LTO_DEF(cosh, COSH, float32_t);
CUDF_LTO_DEF(cosh, COSH, float64_t);

CUDF_LTO_DEF(tanh, TANH, float32_t);
CUDF_LTO_DEF(tanh, TANH, float64_t);

CUDF_LTO_DEF(arcsinh, ARCSINH, float32_t);
CUDF_LTO_DEF(arcsinh, ARCSINH, float64_t);

CUDF_LTO_DEF(arccosh, ARCCOSH, float32_t);
CUDF_LTO_DEF(arccosh, ARCCOSH, float64_t);

CUDF_LTO_DEF(arctanh, ARCTANH, float32_t);
CUDF_LTO_DEF(arctanh, ARCTANH, float64_t);

CUDF_LTO_DEF(exp, EXP, float32_t);
CUDF_LTO_DEF(exp, EXP, float64_t);

CUDF_LTO_DEF(log, LOG, float32_t);
CUDF_LTO_DEF(log, LOG, float64_t);

CUDF_LTO_DEF(cbrt, CBRT, float32_t);
CUDF_LTO_DEF(cbrt, CBRT, float64_t);

CUDF_LTO_DEF(ceil, CEIL, float32_t);
CUDF_LTO_DEF(ceil, CEIL, float64_t);

CUDF_LTO_DEF(floor, FLOOR, float32_t);
CUDF_LTO_DEF(floor, FLOOR, float64_t);

CUDF_LTO_DEF(abs, ABS, int32_t);
CUDF_LTO_DEF(abs, ABS, int64_t);
CUDF_LTO_DEF(abs, ABS, float32_t);
CUDF_LTO_DEF(abs, ABS, float64_t);

CUDF_LTO_DEF(rint, RINT, float32_t);
CUDF_LTO_DEF(rint, RINT, float64_t);

CUDF_LTO_DEF(bit_invert, BIT_INVERT, uint32_t);
CUDF_LTO_DEF(bit_invert, BIT_INVERT, uint64_t);
CUDF_LTO_DEF(bit_invert, BIT_INVERT, int32_t);
CUDF_LTO_DEF(bit_invert, BIT_INVERT, int64_t);

#undef CUDF_LTO_DEF

#define CUDF_LTO_DEF(op, OP, ret_type, type)                                             \
  extern __device__ void operators::op(ret_type* out, type const* a)                     \
  {                                                                                      \
    auto ret = func<opcode::OP>{}(*lto::lift(a));                                        \
    *out     = *lto::lower(&ret);                                                        \
  }                                                                                      \
                                                                                         \
  extern __device__ void operators::op(optional<ret_type>* out, optional<type> const* a) \
  {                                                                                      \
    auto ret = null_func<opcode::OP>{}(*lto::lift(a));                                   \
    *out     = *lto::lower(&ret);                                                        \
  }

CUDF_LTO_DEF(cast_to_int64, CAST_TO_INT64, int64_t, bool);
CUDF_LTO_DEF(cast_to_int64, CAST_TO_INT64, int64_t, int8_t);
CUDF_LTO_DEF(cast_to_int64, CAST_TO_INT64, int64_t, int16_t);
CUDF_LTO_DEF(cast_to_int64, CAST_TO_INT64, int64_t, int32_t);
CUDF_LTO_DEF(cast_to_int64, CAST_TO_INT64, int64_t, int64_t);
CUDF_LTO_DEF(cast_to_int64, CAST_TO_INT64, int64_t, uint8_t);
CUDF_LTO_DEF(cast_to_int64, CAST_TO_INT64, int64_t, uint16_t);
CUDF_LTO_DEF(cast_to_int64, CAST_TO_INT64, int64_t, uint32_t);
CUDF_LTO_DEF(cast_to_int64, CAST_TO_INT64, int64_t, uint64_t);
CUDF_LTO_DEF(cast_to_int64, CAST_TO_INT64, int64_t, float32_t);
CUDF_LTO_DEF(cast_to_int64, CAST_TO_INT64, int64_t, float64_t);

CUDF_LTO_DEF(cast_to_uint64, CAST_TO_UINT64, uint64_t, bool);
CUDF_LTO_DEF(cast_to_uint64, CAST_TO_UINT64, uint64_t, int8_t);
CUDF_LTO_DEF(cast_to_uint64, CAST_TO_UINT64, uint64_t, int16_t);
CUDF_LTO_DEF(cast_to_uint64, CAST_TO_UINT64, uint64_t, int32_t);
CUDF_LTO_DEF(cast_to_uint64, CAST_TO_UINT64, uint64_t, int64_t);
CUDF_LTO_DEF(cast_to_uint64, CAST_TO_UINT64, uint64_t, uint8_t);
CUDF_LTO_DEF(cast_to_uint64, CAST_TO_UINT64, uint64_t, uint16_t);
CUDF_LTO_DEF(cast_to_uint64, CAST_TO_UINT64, uint64_t, uint32_t);
CUDF_LTO_DEF(cast_to_uint64, CAST_TO_UINT64, uint64_t, uint64_t);
CUDF_LTO_DEF(cast_to_uint64, CAST_TO_UINT64, uint64_t, float32_t);
CUDF_LTO_DEF(cast_to_uint64, CAST_TO_UINT64, uint64_t, float64_t);

CUDF_LTO_DEF(cast_to_float64, CAST_TO_FLOAT64, float64_t, bool);
CUDF_LTO_DEF(cast_to_float64, CAST_TO_FLOAT64, float64_t, int8_t);
CUDF_LTO_DEF(cast_to_float64, CAST_TO_FLOAT64, float64_t, int16_t);
CUDF_LTO_DEF(cast_to_float64, CAST_TO_FLOAT64, float64_t, int32_t);
CUDF_LTO_DEF(cast_to_float64, CAST_TO_FLOAT64, float64_t, int64_t);
CUDF_LTO_DEF(cast_to_float64, CAST_TO_FLOAT64, float64_t, uint8_t);
CUDF_LTO_DEF(cast_to_float64, CAST_TO_FLOAT64, float64_t, uint16_t);
CUDF_LTO_DEF(cast_to_float64, CAST_TO_FLOAT64, float64_t, uint32_t);
CUDF_LTO_DEF(cast_to_float64, CAST_TO_FLOAT64, float64_t, uint64_t);
CUDF_LTO_DEF(cast_to_float64, CAST_TO_FLOAT64, float64_t, float32_t);
CUDF_LTO_DEF(cast_to_float64, CAST_TO_FLOAT64, float64_t, float64_t);

#undef CUDF_LTO_DEF

#define CUDF_LTO_DEF(op, OP, type)                                            \
  __device__ void operators::op(bool* out, type const* a)                     \
  {                                                                           \
    auto ret = func<opcode::OP>{}(*lto::lift(a));                             \
    *out     = *lto::lower(&ret);                                             \
  }                                                                           \
                                                                              \
  __device__ void operators::op(bool* out, optional<type> const* a)           \
  {                                                                           \
    auto ret = null_func<opcode::OP>{}(*lto::lift(a));                        \
    *out     = *lto::lower(&ret);                                             \
  }                                                                           \
                                                                              \
  __device__ void operators::op(optional<bool>* out, optional<type> const* a) \
  {                                                                           \
    auto ret = null_func<opcode::OP>{}(*lto::lift(a));                        \
    *out     = *lto::lower(&ret);                                             \
  }

CUDF_LTO_DEF(is_null, IS_NULL, bool);
CUDF_LTO_DEF(is_null, IS_NULL, int8_t);
CUDF_LTO_DEF(is_null, IS_NULL, int16_t);
CUDF_LTO_DEF(is_null, IS_NULL, int32_t);
CUDF_LTO_DEF(is_null, IS_NULL, int64_t);
CUDF_LTO_DEF(is_null, IS_NULL, uint8_t);
CUDF_LTO_DEF(is_null, IS_NULL, uint16_t);
CUDF_LTO_DEF(is_null, IS_NULL, uint32_t);
CUDF_LTO_DEF(is_null, IS_NULL, uint64_t);
CUDF_LTO_DEF(is_null, IS_NULL, float32_t);
CUDF_LTO_DEF(is_null, IS_NULL, float64_t);
CUDF_LTO_DEF(is_null, IS_NULL, decimal32);
CUDF_LTO_DEF(is_null, IS_NULL, decimal64);
CUDF_LTO_DEF(is_null, IS_NULL, decimal128);
CUDF_LTO_DEF(is_null, IS_NULL, timestamp_D);
CUDF_LTO_DEF(is_null, IS_NULL, timestamp_s);
CUDF_LTO_DEF(is_null, IS_NULL, timestamp_ms);
CUDF_LTO_DEF(is_null, IS_NULL, timestamp_us);
CUDF_LTO_DEF(is_null, IS_NULL, timestamp_ns);
CUDF_LTO_DEF(is_null, IS_NULL, duration_D);
CUDF_LTO_DEF(is_null, IS_NULL, duration_s);
CUDF_LTO_DEF(is_null, IS_NULL, duration_ms);
CUDF_LTO_DEF(is_null, IS_NULL, duration_ns);
CUDF_LTO_DEF(is_null, IS_NULL, string_view);

#undef CUDF_LTO_DEF

#define CUDF_LTO_DEF(op, OP, type)                                            \
  __device__ void operators::op(bool* out, type const* a)                     \
  {                                                                           \
    auto ret = func<opcode::OP>{}(*lto::lift(a));                             \
    *out     = *lto::lower(&ret);                                             \
  }                                                                           \
                                                                              \
  __device__ void operators::op(optional<bool>* out, optional<type> const* a) \
  {                                                                           \
    auto ret = null_func<opcode::OP>{}(*lto::lift(a));                        \
    *out     = *lto::lower(&ret);                                             \
  }

CUDF_LTO_DEF(logical_not, NOT, bool);
CUDF_LTO_DEF(logical_not, NOT, int8_t);
CUDF_LTO_DEF(logical_not, NOT, int16_t);
CUDF_LTO_DEF(logical_not, NOT, int32_t);
CUDF_LTO_DEF(logical_not, NOT, int64_t);
CUDF_LTO_DEF(logical_not, NOT, uint8_t);
CUDF_LTO_DEF(logical_not, NOT, uint16_t);
CUDF_LTO_DEF(logical_not, NOT, uint32_t);
CUDF_LTO_DEF(logical_not, NOT, uint64_t);

#undef CUDF_LTO_DEF

}  // namespace lto
}  // namespace CUDF_EXPORT cudf
