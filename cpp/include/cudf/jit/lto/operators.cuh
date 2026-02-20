/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cudf/jit/lto/optional.cuh>
#include <cudf/jit/lto/string_view.cuh>
#include <cudf/jit/lto/types.cuh>

namespace CUDF_LTO_EXPORT cudf {

namespace lto {

namespace operators {

#define CUDF_UNOP(op, type, operand)                                                   \
  __device__ __forceinline__ void op(type* out, type const* a) { *out = operand(*a); } \
  __device__ __forceinline__ void op(optional<type>* out, optional<type> const* a)     \
  {                                                                                    \
    if (a->has_value()) {                                                              \
      op(&(*out), &(*a));                                                              \
    } else {                                                                           \
      *out = {};                                                                       \
    }                                                                                  \
  }

#define CUDF_UNOP_T(op, ret_type, type, operand)                                           \
  __device__ __forceinline__ void op(ret_type* out, type const* a) { *out = operand(*a); } \
  __device__ __forceinline__ void op(optional<ret_type>* out, optional<type> const* a)     \
  {                                                                                        \
    if (a->has_value()) {                                                                  \
      op(&(*out), &(*a));                                                                  \
    } else {                                                                               \
      *out = {};                                                                           \
    }                                                                                      \
  }

#define CUDF_BINOP(op, type, operand)                                         \
  __device__ __forceinline__ void op(type* out, type const* a, type const* b) \
  {                                                                           \
    *out = *a operand * b;                                                    \
  }                                                                           \
                                                                              \
  __device__ __forceinline__ void op(                                         \
    optional<type>* out, optional<type> const* a, optional<type> const* b)    \
  {                                                                           \
    if (a->has_value() && b->has_value()) {                                   \
      op(&(*out), &(*a), &(*b));                                              \
    } else {                                                                  \
      *out = {};                                                              \
    }                                                                         \
  }

#define CUDF_PRED_BINOP(op, type, operand)                                                        \
  __device__ __forceinline__ void op(bool* out, type const* a, type const* b)                     \
  {                                                                                               \
    *out = *a operand * b;                                                                        \
  }                                                                                               \
                                                                                                  \
  __device__ __forceinline__ void op(bool* out, optional<type> const* a, optional<type> const* b) \
  {                                                                                               \
    if (a->has_value() && b->has_value()) {                                                       \
      op(out, &(*a), &(*b));                                                                      \
    } else if (!a->has_value() && !b->has_value()) {                                              \
      *out = true;                                                                                \
    } else {                                                                                      \
      *out = false;                                                                               \
    }                                                                                             \
  }                                                                                               \
                                                                                                  \
  __device__ __forceinline__ void op(                                                             \
    optional<bool>* out, optional<type> const* a, optional<type> const* b)                        \
  {                                                                                               \
    bool r;                                                                                       \
    op(&r, a, b);                                                                                 \
    *out = r;                                                                                     \
  }

#define CUDF_ID_OP(op, type)                                                  \
  __device__ __forceinline__ void op(type* out, type const* a) { *out = *a; } \
  __device__ __forceinline__ void op(optional<type>* out, optional<type> const* a) { *out = *a; }

#define CUDF_IS_NULL_OP(op, type)                                                  \
  __device__ __forceinline__ void op(bool* out, type const* a) { *out = false; }   \
  __device__ __forceinline__ void op(bool* out, optional<type> const* a)           \
  {                                                                                \
    *out = !a->has_value();                                                        \
  }                                                                                \
  __device__ __forceinline__ void op(optional<bool>* out, optional<type> const* a) \
  {                                                                                \
    *out = !a->has_value();                                                        \
  }

#define CUDF_ABS_OP(op, type)                                                                      \
  __device__ __forceinline__ void op(type* out, type const* a) { *out = (*a < 0) ? -(*a) : (*a); } \
  __device__ __forceinline__ void op(optional<type>* out, optional<type> const* a)                 \
  {                                                                                                \
    if (a->has_value()) {                                                                          \
      op(&(*out), &(*a));                                                                          \
    } else {                                                                                       \
      *out = {};                                                                                   \
    }                                                                                              \
  }

#define CUDF_EXTERN_UNOP(op, type)                                        \
  __device__ extern void op(type* out, type const* a);                    \
  __device__ extern void op(optional<type>* out, optional<type> const* a) \
  {                                                                       \
    if (a->has_value()) {                                                 \
      op(&(*out), &(*a));                                                 \
    } else {                                                              \
      *out = {};                                                          \
    }                                                                     \
  }

CUDF_BINOP(add, int32_t, +);
CUDF_BINOP(add, int64_t, +);
CUDF_BINOP(add, uint32_t, +);
CUDF_BINOP(add, uint64_t, +);
CUDF_BINOP(add, float32_t, +);
CUDF_BINOP(add, float64_t, +);
CUDF_BINOP(add, decimal32, +);
CUDF_BINOP(add, decimal64, +);
CUDF_BINOP(add, decimal128, +);
CUDF_BINOP(add, duration_D, +);
CUDF_BINOP(add, duration_s, +);
CUDF_BINOP(add, duration_ms, +);
CUDF_BINOP(add, duration_ns, +);

CUDF_BINOP(sub, int32_t, -);
CUDF_BINOP(sub, int64_t, -);
CUDF_BINOP(sub, uint32_t, -);
CUDF_BINOP(sub, uint64_t, -);
CUDF_BINOP(sub, float32_t, -);
CUDF_BINOP(sub, float64_t, -);
CUDF_BINOP(sub, decimal32, -);
CUDF_BINOP(sub, decimal64, -);
CUDF_BINOP(sub, decimal128, -);
CUDF_BINOP(sub, duration_D, -);
CUDF_BINOP(sub, duration_s, -);
CUDF_BINOP(sub, duration_ms, -);
CUDF_BINOP(sub, duration_ns, -);

CUDF_BINOP(mul, int32_t, *);
CUDF_BINOP(mul, int64_t, *);
CUDF_BINOP(mul, uint32_t, *);
CUDF_BINOP(mul, uint64_t, *);
CUDF_BINOP(mul, float32_t, *);
CUDF_BINOP(mul, float64_t, *);
CUDF_BINOP(mul, decimal32, *);
CUDF_BINOP(mul, decimal64, *);
CUDF_BINOP(mul, decimal128, *);

CUDF_BINOP(div, int32_t, /);
CUDF_BINOP(div, int64_t, /);
CUDF_BINOP(div, uint32_t, /);
CUDF_BINOP(div, uint64_t, /);
CUDF_BINOP(div, float32_t, /);
CUDF_BINOP(div, float64_t, /);
CUDF_BINOP(div, decimal32, /);
CUDF_BINOP(div, decimal64, /);
CUDF_BINOP(div, decimal128, /);

// CUDF_OP(mod, float32_t);
// CUDF_OP(mod, float64_t);

// CUDF_OP(pymod, float32_t);
// CUDF_OP(pymod, float64_t);

// CUDF_OP(pow, float32_t);
// CUDF_OP(pow, float64_t);

CUDF_PRED_BINOP(equal, bool, ==);
CUDF_PRED_BINOP(equal, int8_t, ==);
CUDF_PRED_BINOP(equal, int16_t, ==);
CUDF_PRED_BINOP(equal, int32_t, ==);
CUDF_PRED_BINOP(equal, int64_t, ==);
CUDF_PRED_BINOP(equal, uint8_t, ==);
CUDF_PRED_BINOP(equal, uint16_t, ==);
CUDF_PRED_BINOP(equal, uint32_t, ==);
CUDF_PRED_BINOP(equal, uint64_t, ==);
CUDF_PRED_BINOP(equal, float32_t, ==);
CUDF_PRED_BINOP(equal, float64_t, ==);
CUDF_PRED_BINOP(equal, decimal32, ==);
CUDF_PRED_BINOP(equal, decimal64, ==);
CUDF_PRED_BINOP(equal, decimal128, ==);
CUDF_PRED_BINOP(equal, timestamp_D, ==);
CUDF_PRED_BINOP(equal, timestamp_s, ==);
CUDF_PRED_BINOP(equal, timestamp_ms, ==);
CUDF_PRED_BINOP(equal, timestamp_us, ==);
CUDF_PRED_BINOP(equal, timestamp_ns, ==);
CUDF_PRED_BINOP(equal, duration_D, ==);
CUDF_PRED_BINOP(equal, duration_s, ==);
CUDF_PRED_BINOP(equal, duration_ms, ==);
CUDF_PRED_BINOP(equal, duration_ns, ==);
CUDF_PRED_BINOP(equal, string_view, ==);

/*
CUDF_OP(null_equal, bool);
CUDF_OP(null_equal, int8_t);
CUDF_OP(null_equal, int16_t);
CUDF_OP(null_equal, int32_t);
CUDF_OP(null_equal, int64_t);
CUDF_OP(null_equal, uint8_t);
CUDF_OP(null_equal, uint16_t);
CUDF_OP(null_equal, uint32_t);
CUDF_OP(null_equal, uint64_t);
CUDF_OP(null_equal, float32_t);
CUDF_OP(null_equal, float64_t);
CUDF_OP(null_equal, decimal32);
CUDF_OP(null_equal, decimal64);
CUDF_OP(null_equal, decimal128);
CUDF_OP(null_equal, timestamp_D);
CUDF_OP(null_equal, timestamp_s);
CUDF_OP(null_equal, timestamp_ms);
CUDF_OP(null_equal, timestamp_us);
CUDF_OP(null_equal, timestamp_ns);
CUDF_OP(null_equal, duration_D);
CUDF_OP(null_equal, duration_s);
CUDF_OP(null_equal, duration_ms);
CUDF_OP(null_equal, duration_ns);
CUDF_OP(null_equal, string_view);
*/

CUDF_PRED_BINOP(less, bool, <);
CUDF_PRED_BINOP(less, int8_t, <);
CUDF_PRED_BINOP(less, int16_t, <);
CUDF_PRED_BINOP(less, int32_t, <);
CUDF_PRED_BINOP(less, int64_t, <);
CUDF_PRED_BINOP(less, uint8_t, <);
CUDF_PRED_BINOP(less, uint16_t, <);
CUDF_PRED_BINOP(less, uint32_t, <);
CUDF_PRED_BINOP(less, uint64_t, <);
CUDF_PRED_BINOP(less, float32_t, <);
CUDF_PRED_BINOP(less, float64_t, <);
CUDF_PRED_BINOP(less, decimal32, <);
CUDF_PRED_BINOP(less, decimal64, <);
CUDF_PRED_BINOP(less, decimal128, <);
CUDF_PRED_BINOP(less, timestamp_D, <);
CUDF_PRED_BINOP(less, timestamp_s, <);
CUDF_PRED_BINOP(less, timestamp_ms, <);
CUDF_PRED_BINOP(less, timestamp_us, <);
CUDF_PRED_BINOP(less, timestamp_ns, <);
CUDF_PRED_BINOP(less, duration_D, <);
CUDF_PRED_BINOP(less, duration_s, <);
CUDF_PRED_BINOP(less, duration_ms, <);
CUDF_PRED_BINOP(less, duration_ns, <);
CUDF_PRED_BINOP(less, string_view, <);

CUDF_PRED_BINOP(greater, bool, >);
CUDF_PRED_BINOP(greater, int8_t, >);
CUDF_PRED_BINOP(greater, int16_t, >);
CUDF_PRED_BINOP(greater, int32_t, >);
CUDF_PRED_BINOP(greater, int64_t, >);
CUDF_PRED_BINOP(greater, uint8_t, >);
CUDF_PRED_BINOP(greater, uint16_t, >);
CUDF_PRED_BINOP(greater, uint32_t, >);
CUDF_PRED_BINOP(greater, uint64_t, >);
CUDF_PRED_BINOP(greater, float32_t, >);
CUDF_PRED_BINOP(greater, float64_t, >);
CUDF_PRED_BINOP(greater, decimal32, >);
CUDF_PRED_BINOP(greater, decimal64, >);
CUDF_PRED_BINOP(greater, decimal128, >);
CUDF_PRED_BINOP(greater, timestamp_D, >);
CUDF_PRED_BINOP(greater, timestamp_s, >);
CUDF_PRED_BINOP(greater, timestamp_ms, >);
CUDF_PRED_BINOP(greater, timestamp_us, >);
CUDF_PRED_BINOP(greater, timestamp_ns, >);
CUDF_PRED_BINOP(greater, duration_D, >);
CUDF_PRED_BINOP(greater, duration_s, >);
CUDF_PRED_BINOP(greater, duration_ms, >);
CUDF_PRED_BINOP(greater, duration_ns, >);
CUDF_PRED_BINOP(greater, string_view, >);

CUDF_PRED_BINOP(less_equal, bool, <=);
CUDF_PRED_BINOP(less_equal, int8_t, <=);
CUDF_PRED_BINOP(less_equal, int16_t, <=);
CUDF_PRED_BINOP(less_equal, int32_t, <=);
CUDF_PRED_BINOP(less_equal, int64_t, <=);
CUDF_PRED_BINOP(less_equal, uint8_t, <=);
CUDF_PRED_BINOP(less_equal, uint16_t, <=);
CUDF_PRED_BINOP(less_equal, uint32_t, <=);
CUDF_PRED_BINOP(less_equal, uint64_t, <=);
CUDF_PRED_BINOP(less_equal, float32_t, <=);
CUDF_PRED_BINOP(less_equal, float64_t, <=);
CUDF_PRED_BINOP(less_equal, decimal32, <=);
CUDF_PRED_BINOP(less_equal, decimal64, <=);
CUDF_PRED_BINOP(less_equal, decimal128, <=);
CUDF_PRED_BINOP(less_equal, timestamp_D, <=);
CUDF_PRED_BINOP(less_equal, timestamp_s, <=);
CUDF_PRED_BINOP(less_equal, timestamp_ms, <=);
CUDF_PRED_BINOP(less_equal, timestamp_us, <=);
CUDF_PRED_BINOP(less_equal, timestamp_ns, <=);
CUDF_PRED_BINOP(less_equal, duration_D, <=);
CUDF_PRED_BINOP(less_equal, duration_s, <=);
CUDF_PRED_BINOP(less_equal, duration_ms, <=);
CUDF_PRED_BINOP(less_equal, duration_ns, <=);
CUDF_PRED_BINOP(less_equal, string_view, <=);

CUDF_PRED_BINOP(greater_equal, bool, >=);
CUDF_PRED_BINOP(greater_equal, int8_t, >=);
CUDF_PRED_BINOP(greater_equal, int16_t, >=);
CUDF_PRED_BINOP(greater_equal, int32_t, >=);
CUDF_PRED_BINOP(greater_equal, int64_t, >=);
CUDF_PRED_BINOP(greater_equal, uint8_t, >=);
CUDF_PRED_BINOP(greater_equal, uint16_t, >=);
CUDF_PRED_BINOP(greater_equal, uint32_t, >=);
CUDF_PRED_BINOP(greater_equal, uint64_t, >=);
CUDF_PRED_BINOP(greater_equal, float32_t, >=);
CUDF_PRED_BINOP(greater_equal, float64_t, >=);
CUDF_PRED_BINOP(greater_equal, decimal32, >=);
CUDF_PRED_BINOP(greater_equal, decimal64, >=);
CUDF_PRED_BINOP(greater_equal, decimal128, >=);
CUDF_PRED_BINOP(greater_equal, timestamp_D, >=);
CUDF_PRED_BINOP(greater_equal, timestamp_s, >=);
CUDF_PRED_BINOP(greater_equal, timestamp_ms, >=);
CUDF_PRED_BINOP(greater_equal, timestamp_us, >=);
CUDF_PRED_BINOP(greater_equal, timestamp_ns, >=);
CUDF_PRED_BINOP(greater_equal, duration_D, >=);
CUDF_PRED_BINOP(greater_equal, duration_s, >=);
CUDF_PRED_BINOP(greater_equal, duration_ms, >=);
CUDF_PRED_BINOP(greater_equal, duration_ns, >=);
CUDF_PRED_BINOP(greater_equal, string_view, >=);

CUDF_BINOP(bitwise_and, int32_t, &);
CUDF_BINOP(bitwise_and, int64_t, &);
CUDF_BINOP(bitwise_and, uint32_t, &);
CUDF_BINOP(bitwise_and, uint64_t, &);

CUDF_BINOP(bitwise_or, int32_t, |);
CUDF_BINOP(bitwise_or, int64_t, |);
CUDF_BINOP(bitwise_or, uint32_t, |);
CUDF_BINOP(bitwise_or, uint64_t, |);

CUDF_BINOP(bitwise_xor, int32_t, ^);
CUDF_BINOP(bitwise_xor, int64_t, ^);
CUDF_BINOP(bitwise_xor, uint32_t, ^);
CUDF_BINOP(bitwise_xor, uint64_t, ^);

CUDF_BINOP(logical_and, bool, &&);
CUDF_BINOP(null_logical_and, bool, &&);
CUDF_BINOP(logical_or, bool, ||);
CUDF_BINOP(null_logical_or, bool, ||);

CUDF_ID_OP(identity, bool);
CUDF_ID_OP(identity, int8_t);
CUDF_ID_OP(identity, int16_t);
CUDF_ID_OP(identity, int32_t);
CUDF_ID_OP(identity, int64_t);
CUDF_ID_OP(identity, uint8_t);
CUDF_ID_OP(identity, uint16_t);
CUDF_ID_OP(identity, uint32_t);
CUDF_ID_OP(identity, uint64_t);
CUDF_ID_OP(identity, float32_t);
CUDF_ID_OP(identity, float64_t);
CUDF_ID_OP(identity, decimal32);
CUDF_ID_OP(identity, decimal64);
CUDF_ID_OP(identity, decimal128);
CUDF_ID_OP(identity, timestamp_D);
CUDF_ID_OP(identity, timestamp_s);
CUDF_ID_OP(identity, timestamp_ms);
CUDF_ID_OP(identity, timestamp_us);
CUDF_ID_OP(identity, timestamp_ns);
CUDF_ID_OP(identity, duration_D);
CUDF_ID_OP(identity, duration_s);
CUDF_ID_OP(identity, duration_ms);
CUDF_ID_OP(identity, duration_ns);
CUDF_ID_OP(identity, string_view);

CUDF_UNOP(bit_invert, uint32_t, ~);
CUDF_UNOP(bit_invert, uint64_t, ~);
CUDF_UNOP(bit_invert, int32_t, ~);
CUDF_UNOP(bit_invert, int64_t, ~);

CUDF_UNOP_T(cast_to_int64, int64_t, bool, (int64_t));
CUDF_UNOP_T(cast_to_int64, int64_t, int8_t, (int64_t));
CUDF_UNOP_T(cast_to_int64, int64_t, int16_t, (int64_t));
CUDF_UNOP_T(cast_to_int64, int64_t, int32_t, (int64_t));
CUDF_UNOP_T(cast_to_int64, int64_t, int64_t, (int64_t));
CUDF_UNOP_T(cast_to_int64, int64_t, uint8_t, (int64_t));
CUDF_UNOP_T(cast_to_int64, int64_t, uint16_t, (int64_t));
CUDF_UNOP_T(cast_to_int64, int64_t, uint32_t, (int64_t));
CUDF_UNOP_T(cast_to_int64, int64_t, uint64_t, (int64_t));
CUDF_UNOP_T(cast_to_int64, int64_t, float32_t, (int64_t));
CUDF_UNOP_T(cast_to_int64, int64_t, float64_t, (int64_t));

CUDF_UNOP_T(cast_to_uint64, uint64_t, bool, (uint64_t));
CUDF_UNOP_T(cast_to_uint64, uint64_t, int8_t, (uint64_t));
CUDF_UNOP_T(cast_to_uint64, uint64_t, int16_t, (uint64_t));
CUDF_UNOP_T(cast_to_uint64, uint64_t, int32_t, (uint64_t));
CUDF_UNOP_T(cast_to_uint64, uint64_t, int64_t, (uint64_t));
CUDF_UNOP_T(cast_to_uint64, uint64_t, uint8_t, (uint64_t));
CUDF_UNOP_T(cast_to_uint64, uint64_t, uint16_t, (uint64_t));
CUDF_UNOP_T(cast_to_uint64, uint64_t, uint32_t, (uint64_t));
CUDF_UNOP_T(cast_to_uint64, uint64_t, uint64_t, (uint64_t));
CUDF_UNOP_T(cast_to_uint64, uint64_t, float32_t, (uint64_t));
CUDF_UNOP_T(cast_to_uint64, uint64_t, float64_t, (uint64_t));

CUDF_UNOP_T(cast_to_float64, float64_t, bool, (float64_t));
CUDF_UNOP_T(cast_to_float64, float64_t, int8_t, (float64_t));
CUDF_UNOP_T(cast_to_float64, float64_t, int16_t, (float64_t));
CUDF_UNOP_T(cast_to_float64, float64_t, int32_t, (float64_t));
CUDF_UNOP_T(cast_to_float64, float64_t, int64_t, (float64_t));
CUDF_UNOP_T(cast_to_float64, float64_t, uint8_t, (float64_t));
CUDF_UNOP_T(cast_to_float64, float64_t, uint16_t, (float64_t));
CUDF_UNOP_T(cast_to_float64, float64_t, uint32_t, (float64_t));
CUDF_UNOP_T(cast_to_float64, float64_t, uint64_t, (float64_t));
CUDF_UNOP_T(cast_to_float64, float64_t, float32_t, (float64_t));
CUDF_UNOP_T(cast_to_float64, float64_t, float64_t, (float64_t));

CUDF_IS_NULL_OP(is_null, bool);
CUDF_IS_NULL_OP(is_null, int8_t);
CUDF_IS_NULL_OP(is_null, int16_t);
CUDF_IS_NULL_OP(is_null, int32_t);
CUDF_IS_NULL_OP(is_null, int64_t);
CUDF_IS_NULL_OP(is_null, uint8_t);
CUDF_IS_NULL_OP(is_null, uint16_t);
CUDF_IS_NULL_OP(is_null, uint32_t);
CUDF_IS_NULL_OP(is_null, uint64_t);
CUDF_IS_NULL_OP(is_null, float32_t);
CUDF_IS_NULL_OP(is_null, float64_t);
CUDF_IS_NULL_OP(is_null, decimal32);
CUDF_IS_NULL_OP(is_null, decimal64);
CUDF_IS_NULL_OP(is_null, decimal128);
CUDF_IS_NULL_OP(is_null, timestamp_D);
CUDF_IS_NULL_OP(is_null, timestamp_s);
CUDF_IS_NULL_OP(is_null, timestamp_ms);
CUDF_IS_NULL_OP(is_null, timestamp_us);
CUDF_IS_NULL_OP(is_null, timestamp_ns);
CUDF_IS_NULL_OP(is_null, duration_D);
CUDF_IS_NULL_OP(is_null, duration_s);
CUDF_IS_NULL_OP(is_null, duration_ms);
CUDF_IS_NULL_OP(is_null, duration_ns);
CUDF_IS_NULL_OP(is_null, string_view);

CUDF_ABS_OP(abs, int8_t);
CUDF_ABS_OP(abs, int16_t);
CUDF_ABS_OP(abs, int32_t);
CUDF_ABS_OP(abs, int64_t);
CUDF_ABS_OP(abs, float32_t);
CUDF_ABS_OP(abs, float64_t);

CUDF_EXTERN_UNOP(sin, float32_t);
CUDF_EXTERN_UNOP(sin, float64_t);

CUDF_EXTERN_UNOP(cos, float32_t);
CUDF_EXTERN_UNOP(cos, float64_t);

CUDF_EXTERN_UNOP(tan, float32_t);
CUDF_EXTERN_UNOP(tan, float64_t);

CUDF_EXTERN_UNOP(arcsin, float32_t);
CUDF_EXTERN_UNOP(arcsin, float64_t);

CUDF_EXTERN_UNOP(arccos, float32_t);
CUDF_EXTERN_UNOP(arccos, float64_t);

CUDF_EXTERN_UNOP(arctan, float32_t);
CUDF_EXTERN_UNOP(arctan, float64_t);

CUDF_EXTERN_UNOP(sinh, float32_t);
CUDF_EXTERN_UNOP(sinh, float64_t);

CUDF_EXTERN_UNOP(cosh, float32_t);
CUDF_EXTERN_UNOP(cosh, float64_t);

CUDF_EXTERN_UNOP(tanh, float32_t);
CUDF_EXTERN_UNOP(tanh, float64_t);

CUDF_EXTERN_UNOP(arcsinh, float32_t);
CUDF_EXTERN_UNOP(arcsinh, float64_t);

CUDF_EXTERN_UNOP(arccosh, float32_t);
CUDF_EXTERN_UNOP(arccosh, float64_t);

CUDF_EXTERN_UNOP(arctanh, float32_t);
CUDF_EXTERN_UNOP(arctanh, float64_t);

CUDF_EXTERN_UNOP(exp, float32_t);
CUDF_EXTERN_UNOP(exp, float64_t);

CUDF_EXTERN_UNOP(log, float32_t);
CUDF_EXTERN_UNOP(log, float64_t);

CUDF_EXTERN_UNOP(cbrt, float32_t);
CUDF_EXTERN_UNOP(cbrt, float64_t);

CUDF_EXTERN_UNOP(ceil, float32_t);
CUDF_EXTERN_UNOP(ceil, float64_t);

CUDF_EXTERN_UNOP(floor, float32_t);
CUDF_EXTERN_UNOP(floor, float64_t);

CUDF_EXTERN_UNOP(rint, float32_t);
CUDF_EXTERN_UNOP(rint, float64_t);

}  // namespace operators

}  // namespace lto
}  // namespace CUDF_LTO_EXPORT cudf
