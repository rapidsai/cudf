/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <jcudf/types.cuh>

namespace JCUDF_EXPORT jcudf {
namespace functions {

#define UNOP_T(op, R, T, expr)                                               \
  __device__ __forceinline__ void op(R* out, T const& a) { *out = expr; }    \
                                                                             \
  __device__ __forceinline__ void op(optional<R>* out, optional<T> const& a) \
  {                                                                          \
    if (a.has_value()) {                                                     \
      R r;                                                                   \
      op(&r, *a);                                                            \
      *out = r;                                                              \
    } else {                                                                 \
      *out = nullopt;                                                        \
    }                                                                        \
  }

#define BINOP_T(op, R, T, expr)                                                                    \
  __device__ __forceinline__ void op(R* out, T const& a, T const& b) { *out = expr; }              \
                                                                                                   \
  __device__ __forceinline__ void op(optional<R>* out, optional<T> const& a, optional<T> const& b) \
  {                                                                                                \
    if (a.has_value() && b.has_value()) {                                                          \
      R r;                                                                                         \
      op(&r, *a, *b);                                                                              \
      *out = r;                                                                                    \
    } else {                                                                                       \
      *out = nullopt;                                                                              \
    }                                                                                              \
  }

#define UNOP(op, T, expr)  UNOP_T(op, T, T, expr)
#define BINOP(op, T, expr) BINOP_T(op, T, T, expr)

#define ADD_OP(T)                  BINOP_T(add, T, T, (a + b))
#define SUB_OP(T)                  BINOP_T(sub, T, T, (a - b))
#define MUL_OP(T)                  BINOP_T(mul, T, T, (a * b))
#define DIV_OP(T)                  BINOP_T(div, T, T, (a / b))
#define EQ_OP(T)                   BINOP_T(equal, bool, T, (a == b))
#define LT_OP(T)                   BINOP_T(less, bool, T, (a < b))
#define GT_OP(T)                   BINOP_T(greater, bool, T, (a > b))
#define LE_OP(T)                   BINOP_T(less_equal, bool, T, (a <= b))
#define GE_OP(T)                   BINOP_T(greater_equal, bool, T, (a >= b))
#define BIT_AND_OP(T)              BINOP_T(bitwise_and, T, T, (a & b))
#define BIT_OR_OP(T)               BINOP_T(bitwise_or, T, T, (a | b))
#define BIT_XOR_OP(T)              BINOP_T(bitwise_xor, T, T, (a ^ b))
#define LOGICAL_AND_OP(T)          BINOP_T(logical_and, bool, T, (a && b))
#define LOGICAL_OR_OP(T)           BINOP_T(logical_or, bool, T, (a || b))
#define IDENTITY_OP(T)             UNOP_T(identity, T, T, a)
#define BIT_INVERT_OP(T)           UNOP_T(bit_invert, T, T, ~a)
#define CAST_OP(out_type, in_type) UNOP_T(cast_to_##out_type, out_type, in_type, (out_type)(a))
#define ABS_OP(T)                  UNOP_T(abs, T, T, ((a < 0) ? -a : a))
#define MOD_OP(T)                  BINOP_T(mod, T, T, (a % b))
#define PYMOD_OP(T)                BINOP_T(pymod, T, T, ((a % b + b) % b))

#define NULL_EQ_OP(T)                                                                              \
  __device__ __forceinline__ void null_equal(bool* out, T const& a, T const& b) { *out = a == b; } \
                                                                                                   \
  __device__ __forceinline__ void null_equal(                                                      \
    optional<bool>* out, optional<T> const& a, optional<T> const& b)                               \
  {                                                                                                \
    if (a.has_value() && b.has_value()) {                                                          \
      *out = (*a == *b);                                                                           \
    } else if (a.has_null() && b.has_null()) {                                                     \
      *out = true;                                                                                 \
    } else {                                                                                       \
      *out = false;                                                                                \
    }                                                                                              \
  }

#define IS_NULL_OP(T)                                                                \
  __device__ __forceinline__ void is_null(bool* out, T const& a) { *out = false; }   \
  __device__ __forceinline__ void is_null(optional<bool>* out, optional<T> const& a) \
  {                                                                                  \
    *out = a.has_null();                                                             \
  }

ADD_OP(i32)
ADD_OP(i64)
ADD_OP(u32)
ADD_OP(u64)
ADD_OP(f32)
ADD_OP(f64)
ADD_OP(dec32)
ADD_OP(dec64)
ADD_OP(dec128)
ADD_OP(duration_D)
ADD_OP(duration_s)
ADD_OP(duration_ms)
ADD_OP(duration_ns)

SUB_OP(i32)
SUB_OP(i64)
SUB_OP(u32)
SUB_OP(u64)
SUB_OP(f32)
SUB_OP(f64)
SUB_OP(dec32)
SUB_OP(dec64)
SUB_OP(dec128)
SUB_OP(duration_D)
SUB_OP(duration_s)
SUB_OP(duration_ms)
SUB_OP(duration_ns)

MUL_OP(i32)
MUL_OP(i64)
MUL_OP(u32)
MUL_OP(u64)
MUL_OP(f32)
MUL_OP(f64)
MUL_OP(dec32)
MUL_OP(dec64)
MUL_OP(dec128)

DIV_OP(i32)
DIV_OP(i64)
DIV_OP(u32)
DIV_OP(u64)
DIV_OP(f32)
DIV_OP(f64)
DIV_OP(dec32)
DIV_OP(dec64)
DIV_OP(dec128)

EQ_OP(bool)
EQ_OP(i8)
EQ_OP(i16)
EQ_OP(i32)
EQ_OP(i64)
EQ_OP(u8)
EQ_OP(u16)
EQ_OP(u32)
EQ_OP(u64)
EQ_OP(f32)
EQ_OP(f64)
EQ_OP(dec32)
EQ_OP(dec64)
EQ_OP(dec128)
EQ_OP(timestamp_D)
EQ_OP(timestamp_s)
EQ_OP(timestamp_ms)
EQ_OP(timestamp_us)
EQ_OP(timestamp_ns)
EQ_OP(duration_D)
EQ_OP(duration_s)
EQ_OP(duration_ms)
EQ_OP(duration_ns)
EQ_OP(string_view)

LT_OP(bool)
LT_OP(i8)
LT_OP(i16)
LT_OP(i32)
LT_OP(i64)
LT_OP(u8)
LT_OP(u16)
LT_OP(u32)
LT_OP(u64)
LT_OP(f32)
LT_OP(f64)
LT_OP(dec32)
LT_OP(dec64)
LT_OP(dec128)
LT_OP(timestamp_D)
LT_OP(timestamp_s)
LT_OP(timestamp_ms)
LT_OP(timestamp_us)
LT_OP(timestamp_ns)
LT_OP(duration_D)
LT_OP(duration_s)
LT_OP(duration_ms)
LT_OP(duration_ns)
LT_OP(string_view)

GT_OP(bool)
GT_OP(i8)
GT_OP(i16)
GT_OP(i32)
GT_OP(i64)
GT_OP(u8)
GT_OP(u16)
GT_OP(u32)
GT_OP(u64)
GT_OP(f32)
GT_OP(f64)
GT_OP(dec32)
GT_OP(dec64)
GT_OP(dec128)
GT_OP(timestamp_D)
GT_OP(timestamp_s)
GT_OP(timestamp_ms)
GT_OP(timestamp_us)
GT_OP(timestamp_ns)
GT_OP(duration_D)
GT_OP(duration_s)
GT_OP(duration_ms)
GT_OP(duration_ns)
GT_OP(string_view)

LE_OP(bool)
LE_OP(i8)
LE_OP(i16)
LE_OP(i32)
LE_OP(i64)
LE_OP(u8)
LE_OP(u16)
LE_OP(u32)
LE_OP(u64)
LE_OP(f32)
LE_OP(f64)
LE_OP(dec32)
LE_OP(dec64)
LE_OP(dec128)
LE_OP(timestamp_D)
LE_OP(timestamp_s)
LE_OP(timestamp_ms)
LE_OP(timestamp_us)
LE_OP(timestamp_ns)
LE_OP(duration_D)
LE_OP(duration_s)
LE_OP(duration_ms)
LE_OP(duration_ns)
LE_OP(string_view)

GE_OP(bool)
GE_OP(i8)
GE_OP(i16)
GE_OP(i32)
GE_OP(i64)
GE_OP(u8)
GE_OP(u16)
GE_OP(u32)
GE_OP(u64)
GE_OP(f32)
GE_OP(f64)
GE_OP(dec32)
GE_OP(dec64)
GE_OP(dec128)
GE_OP(timestamp_D)
GE_OP(timestamp_s)
GE_OP(timestamp_ms)
GE_OP(timestamp_us)
GE_OP(timestamp_ns)
GE_OP(duration_D)
GE_OP(duration_s)
GE_OP(duration_ms)
GE_OP(duration_ns)
GE_OP(string_view)

BIT_AND_OP(i32)
BIT_AND_OP(i64)
BIT_AND_OP(u32)
BIT_AND_OP(u64)

BIT_OR_OP(i32)
BIT_OR_OP(i64)
BIT_OR_OP(u32)
BIT_OR_OP(u64)

BIT_XOR_OP(i32)
BIT_XOR_OP(i64)
BIT_XOR_OP(u32)
BIT_XOR_OP(u64)

LOGICAL_AND_OP(bool)

LOGICAL_OR_OP(bool)

IDENTITY_OP(bool)
IDENTITY_OP(i8)
IDENTITY_OP(i16)
IDENTITY_OP(i32)
IDENTITY_OP(i64)
IDENTITY_OP(u8)
IDENTITY_OP(u16)
IDENTITY_OP(u32)
IDENTITY_OP(u64)
IDENTITY_OP(f32)
IDENTITY_OP(f64)
IDENTITY_OP(dec32)
IDENTITY_OP(dec64)
IDENTITY_OP(dec128)
IDENTITY_OP(timestamp_D)
IDENTITY_OP(timestamp_s)
IDENTITY_OP(timestamp_ms)
IDENTITY_OP(timestamp_us)
IDENTITY_OP(timestamp_ns)
IDENTITY_OP(duration_D)
IDENTITY_OP(duration_s)
IDENTITY_OP(duration_ms)
IDENTITY_OP(duration_ns)
IDENTITY_OP(string_view)

BIT_INVERT_OP(u32)
BIT_INVERT_OP(u64)
BIT_INVERT_OP(i32)
BIT_INVERT_OP(i64)

CAST_OP(i64, bool)
CAST_OP(i64, i8)
CAST_OP(i64, i16)
CAST_OP(i64, i32)
CAST_OP(i64, i64)
CAST_OP(i64, u8)
CAST_OP(i64, u16)
CAST_OP(i64, u32)
CAST_OP(i64, u64)
CAST_OP(i64, f32)
CAST_OP(i64, f64)

CAST_OP(u64, bool)
CAST_OP(u64, i8)
CAST_OP(u64, i16)
CAST_OP(u64, i32)
CAST_OP(u64, i64)
CAST_OP(u64, u8)
CAST_OP(u64, u16)
CAST_OP(u64, u32)
CAST_OP(u64, u64)
CAST_OP(u64, f32)
CAST_OP(u64, f64)

CAST_OP(f64, bool)
CAST_OP(f64, i8)
CAST_OP(f64, i16)
CAST_OP(f64, i32)
CAST_OP(f64, i64)
CAST_OP(f64, u8)
CAST_OP(f64, u16)
CAST_OP(f64, u32)
CAST_OP(f64, u64)
CAST_OP(f64, f32)
CAST_OP(f64, f64)

IS_NULL_OP(bool)
IS_NULL_OP(i8)
IS_NULL_OP(i16)
IS_NULL_OP(i32)
IS_NULL_OP(i64)
IS_NULL_OP(u8)
IS_NULL_OP(u16)
IS_NULL_OP(u32)
IS_NULL_OP(u64)
IS_NULL_OP(f32)
IS_NULL_OP(f64)
IS_NULL_OP(dec32)
IS_NULL_OP(dec64)
IS_NULL_OP(dec128)
IS_NULL_OP(timestamp_D)
IS_NULL_OP(timestamp_s)
IS_NULL_OP(timestamp_ms)
IS_NULL_OP(timestamp_us)
IS_NULL_OP(timestamp_ns)
IS_NULL_OP(duration_D)
IS_NULL_OP(duration_s)
IS_NULL_OP(duration_ms)
IS_NULL_OP(duration_ns)
IS_NULL_OP(string_view)

ABS_OP(i8)
ABS_OP(i16)
ABS_OP(i32)
ABS_OP(i64)
ABS_OP(f32)
ABS_OP(f64)

UNOP(sin, f32, (__builtin_sinf(a)))
UNOP(sin, f64, (__builtin_sin(a)))

UNOP(cos, f32, (__builtin_cosf(a)))
UNOP(cos, f64, (__builtin_cos(a)))

UNOP(tan, f32, (__builtin_tanf(a)))
UNOP(tan, f64, (__builtin_tan(a)))

UNOP(arcsin, f32, (__builtin_asinf(a)))
UNOP(arcsin, f64, (__builtin_asin(a)))

UNOP(arccos, f32, (__builtin_acosf(a)))
UNOP(arccos, f64, (__builtin_acos(a)))

UNOP(arctan, f32, (__builtin_atanf(a)))
UNOP(arctan, f64, (__builtin_atan(a)))

UNOP(sinh, f32, (__builtin_sinhf(a)))
UNOP(sinh, f64, (__builtin_sinh(a)))

UNOP(cosh, f32, (__builtin_coshf(a)))
UNOP(cosh, f64, (__builtin_cosh(a)))

UNOP(tanh, f32, (__builtin_tanhf(a)))
UNOP(tanh, f64, (__builtin_tanh(a)))

UNOP(arcsinh, f32, (__builtin_asinhf(a)))
UNOP(arcsinh, f64, (__builtin_asinh(a)))

UNOP(arccosh, f32, (__builtin_acoshf(a)))
UNOP(arccosh, f64, (__builtin_acosh(a)))

UNOP(arctanh, f32, (__builtin_atanhf(a)))
UNOP(arctanh, f64, (__builtin_atanh(a)))

UNOP(exp, f32, (__builtin_expf(a)))
UNOP(exp, f64, (__builtin_exp(a)))

UNOP(log, f32, (__builtin_logf(a)))
UNOP(log, f64, (__builtin_log(a)))

UNOP(cbrt, f32, (__builtin_cbrtf(a)))
UNOP(cbrt, f64, (__builtin_cbrt(a)))

UNOP(ceil, f32, (__builtin_ceilf(a)))
UNOP(ceil, f64, (__builtin_ceil(a)))

UNOP(floor, f32, (__builtin_floorf(a)))
UNOP(floor, f64, (__builtin_floor(a)))

UNOP(rint, f32, (__builtin_rintf(a)))
UNOP(rint, f64, (__builtin_rint(a)))

MOD_OP(i32)
MOD_OP(i64)
MOD_OP(u32)
MOD_OP(u64)
BINOP(mod, f32, (__builtin_fmodf(a, b)))
BINOP(mod, f64, (__builtin_fmod(a, b)))

PYMOD_OP(i32)
PYMOD_OP(i64)
PYMOD_OP(u32)
PYMOD_OP(u64)
BINOP(pymod, f32, (__builtin_fmodf(__builtin_fmodf(a, b) + b, b)))
BINOP(pymod, f64, (__builtin_fmod(__builtin_fmod(a, b) + b, b)))

BINOP(pow, f32, (__builtin_powf(a, b)))
BINOP(pow, f64, (__builtin_pow(a, b)))

NULL_EQ_OP(bool)
NULL_EQ_OP(i8)
NULL_EQ_OP(i16)
NULL_EQ_OP(i32)
NULL_EQ_OP(i64)
NULL_EQ_OP(u8)
NULL_EQ_OP(u16)
NULL_EQ_OP(u32)
NULL_EQ_OP(u64)
NULL_EQ_OP(f32)
NULL_EQ_OP(f64)
NULL_EQ_OP(dec32)
NULL_EQ_OP(dec64)
NULL_EQ_OP(dec128)
NULL_EQ_OP(timestamp_D)
NULL_EQ_OP(timestamp_s)
NULL_EQ_OP(timestamp_ms)
NULL_EQ_OP(timestamp_us)
NULL_EQ_OP(timestamp_ns)
NULL_EQ_OP(duration_D)
NULL_EQ_OP(duration_s)
NULL_EQ_OP(duration_ms)
NULL_EQ_OP(duration_ns)
NULL_EQ_OP(string_view)

__device__ __forceinline__ void null_logical_and(bool* out, bool const& a, bool const& b)
{
  *out = a && b;
}

__device__ __forceinline__ void null_logical_and(optional<bool>* out,
                                                 optional<bool> const& a,
                                                 optional<bool> const& b)
{
  if (a.has_value() && b.has_value()) {
    *out = (*a && *b);
  } else if (a.has_null() && b.has_null()) {
    *out = nullopt;
  } else {
    bool valid = a.has_value() ? *a : *b;
    if (valid) {
      *out = nullopt;
    } else {
      *out = false;
    }
  }
}

__device__ __forceinline__ void null_logical_or(bool* out, bool const& a, bool const& b)
{
  *out = a || b;
}

__device__ __forceinline__ void null_logical_or(optional<bool>* out,
                                                optional<bool> const& a,
                                                optional<bool> const& b)
{
  if (a.has_value() && b.has_value()) {
    *out = (*a || *b);
  } else if (a.has_null() && b.has_null()) {
    *out = nullopt;
  } else {
    bool valid = a.has_value() ? *a : *b;
    if (valid) {
      *out = true;
    } else {
      *out = nullopt;
    }
  }
}

}  // namespace functions
}  // namespace JCUDF_EXPORT jcudf
