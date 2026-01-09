/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cudf/jit/lto/types.cuh>

namespace CUDF_LTO_EXPORT cudf {

namespace lto {

namespace operators {

#define CUDF_LTO_DECL(op, type)                                              \
  __device__ extern inline void op(type* out, type const* a, type const* b); \
                                                                             \
  __device__ extern inline void op(                                          \
    optional<type>* out, optional<type> const* a, optional<type> const* b)

CUDF_LTO_DECL(add, int32_t);
CUDF_LTO_DECL(add, int64_t);
CUDF_LTO_DECL(add, uint32_t);
CUDF_LTO_DECL(add, uint64_t);
CUDF_LTO_DECL(add, float32_t);
CUDF_LTO_DECL(add, float64_t);
CUDF_LTO_DECL(add, decimal32);
CUDF_LTO_DECL(add, decimal64);
CUDF_LTO_DECL(add, decimal128);
CUDF_LTO_DECL(add, duration_D);
CUDF_LTO_DECL(add, duration_s);
CUDF_LTO_DECL(add, duration_ms);
CUDF_LTO_DECL(add, duration_ns);

CUDF_LTO_DECL(sub, int32_t);
CUDF_LTO_DECL(sub, int64_t);
CUDF_LTO_DECL(sub, uint32_t);
CUDF_LTO_DECL(sub, uint64_t);
CUDF_LTO_DECL(sub, float32_t);
CUDF_LTO_DECL(sub, float64_t);
CUDF_LTO_DECL(sub, decimal32);
CUDF_LTO_DECL(sub, decimal64);
CUDF_LTO_DECL(sub, decimal128);
CUDF_LTO_DECL(sub, duration_D);
CUDF_LTO_DECL(sub, duration_s);
CUDF_LTO_DECL(sub, duration_ms);
CUDF_LTO_DECL(sub, duration_ns);

CUDF_LTO_DECL(mul, int32_t);
CUDF_LTO_DECL(mul, int64_t);
CUDF_LTO_DECL(mul, uint32_t);
CUDF_LTO_DECL(mul, uint64_t);
CUDF_LTO_DECL(mul, float32_t);
CUDF_LTO_DECL(mul, float64_t);
CUDF_LTO_DECL(mul, decimal32);
CUDF_LTO_DECL(mul, decimal64);
CUDF_LTO_DECL(mul, decimal128);

CUDF_LTO_DECL(div, int32_t);
CUDF_LTO_DECL(div, int64_t);
CUDF_LTO_DECL(div, uint32_t);
CUDF_LTO_DECL(div, uint64_t);
CUDF_LTO_DECL(div, float32_t);
CUDF_LTO_DECL(div, float64_t);
CUDF_LTO_DECL(div, decimal32);
CUDF_LTO_DECL(div, decimal64);
CUDF_LTO_DECL(div, decimal128);

CUDF_LTO_DECL(mod, float32_t);
CUDF_LTO_DECL(mod, float64_t);

CUDF_LTO_DECL(pymod, float32_t);
CUDF_LTO_DECL(pymod, float64_t);

CUDF_LTO_DECL(pow, float32_t);
CUDF_LTO_DECL(pow, float64_t);

#undef CUDF_LTO_DECL

#define CUDF_LTO_DECL(op, type)                                                                  \
  __device__ extern inline void op(bool* out, type const* a, type const* b);                     \
                                                                                                 \
  __device__ extern inline void op(bool* out, optional<type> const* a, optional<type> const* b); \
                                                                                                 \
  __device__ extern inline void op(                                                              \
    optional<bool>* out, optional<type> const* a, optional<type> const* b)

CUDF_LTO_DECL(equal, bool);
CUDF_LTO_DECL(equal, int8_t);
CUDF_LTO_DECL(equal, int16_t);
CUDF_LTO_DECL(equal, int32_t);
CUDF_LTO_DECL(equal, int64_t);
CUDF_LTO_DECL(equal, uint8_t);
CUDF_LTO_DECL(equal, uint16_t);
CUDF_LTO_DECL(equal, uint32_t);
CUDF_LTO_DECL(equal, uint64_t);
CUDF_LTO_DECL(equal, float32_t);
CUDF_LTO_DECL(equal, float64_t);
CUDF_LTO_DECL(equal, decimal32);
CUDF_LTO_DECL(equal, decimal64);
CUDF_LTO_DECL(equal, decimal128);
CUDF_LTO_DECL(equal, timestamp_D);
CUDF_LTO_DECL(equal, timestamp_s);
CUDF_LTO_DECL(equal, timestamp_ms);
CUDF_LTO_DECL(equal, timestamp_us);
CUDF_LTO_DECL(equal, timestamp_ns);
CUDF_LTO_DECL(equal, duration_D);
CUDF_LTO_DECL(equal, duration_s);
CUDF_LTO_DECL(equal, duration_ms);
CUDF_LTO_DECL(equal, duration_ns);
CUDF_LTO_DECL(equal, string_view);

CUDF_LTO_DECL(null_equal, bool);
CUDF_LTO_DECL(null_equal, int8_t);
CUDF_LTO_DECL(null_equal, int16_t);
CUDF_LTO_DECL(null_equal, int32_t);
CUDF_LTO_DECL(null_equal, int64_t);
CUDF_LTO_DECL(null_equal, uint8_t);
CUDF_LTO_DECL(null_equal, uint16_t);
CUDF_LTO_DECL(null_equal, uint32_t);
CUDF_LTO_DECL(null_equal, uint64_t);
CUDF_LTO_DECL(null_equal, float32_t);
CUDF_LTO_DECL(null_equal, float64_t);
CUDF_LTO_DECL(null_equal, decimal32);
CUDF_LTO_DECL(null_equal, decimal64);
CUDF_LTO_DECL(null_equal, decimal128);
CUDF_LTO_DECL(null_equal, timestamp_D);
CUDF_LTO_DECL(null_equal, timestamp_s);
CUDF_LTO_DECL(null_equal, timestamp_ms);
CUDF_LTO_DECL(null_equal, timestamp_us);
CUDF_LTO_DECL(null_equal, timestamp_ns);
CUDF_LTO_DECL(null_equal, duration_D);
CUDF_LTO_DECL(null_equal, duration_s);
CUDF_LTO_DECL(null_equal, duration_ms);
CUDF_LTO_DECL(null_equal, duration_ns);
CUDF_LTO_DECL(null_equal, string_view);

#undef CUDF_LTO_DECL

#define CUDF_LTO_DECL(op, type)                                              \
  __device__ extern inline void op(bool* out, type const* a, type const* b); \
                                                                             \
  __device__ extern inline void op(                                          \
    optional<bool>* out, optional<type> const* a, optional<type> const* b)

CUDF_LTO_DECL(less, bool);
CUDF_LTO_DECL(less, int8_t);
CUDF_LTO_DECL(less, int16_t);
CUDF_LTO_DECL(less, int32_t);
CUDF_LTO_DECL(less, int64_t);
CUDF_LTO_DECL(less, uint8_t);
CUDF_LTO_DECL(less, uint16_t);
CUDF_LTO_DECL(less, uint32_t);
CUDF_LTO_DECL(less, uint64_t);
CUDF_LTO_DECL(less, float32_t);
CUDF_LTO_DECL(less, float64_t);
CUDF_LTO_DECL(less, decimal32);
CUDF_LTO_DECL(less, decimal64);
CUDF_LTO_DECL(less, decimal128);
CUDF_LTO_DECL(less, timestamp_D);
CUDF_LTO_DECL(less, timestamp_s);
CUDF_LTO_DECL(less, timestamp_ms);
CUDF_LTO_DECL(less, timestamp_us);
CUDF_LTO_DECL(less, timestamp_ns);
CUDF_LTO_DECL(less, duration_D);
CUDF_LTO_DECL(less, duration_s);
CUDF_LTO_DECL(less, duration_ms);
CUDF_LTO_DECL(less, duration_ns);
CUDF_LTO_DECL(less, string_view);

CUDF_LTO_DECL(greater, bool);
CUDF_LTO_DECL(greater, int8_t);
CUDF_LTO_DECL(greater, int16_t);
CUDF_LTO_DECL(greater, int32_t);
CUDF_LTO_DECL(greater, int64_t);
CUDF_LTO_DECL(greater, uint8_t);
CUDF_LTO_DECL(greater, uint16_t);
CUDF_LTO_DECL(greater, uint32_t);
CUDF_LTO_DECL(greater, uint64_t);
CUDF_LTO_DECL(greater, float32_t);
CUDF_LTO_DECL(greater, float64_t);
CUDF_LTO_DECL(greater, decimal32);
CUDF_LTO_DECL(greater, decimal64);
CUDF_LTO_DECL(greater, decimal128);
CUDF_LTO_DECL(greater, timestamp_D);
CUDF_LTO_DECL(greater, timestamp_s);
CUDF_LTO_DECL(greater, timestamp_ms);
CUDF_LTO_DECL(greater, timestamp_us);
CUDF_LTO_DECL(greater, timestamp_ns);
CUDF_LTO_DECL(greater, duration_D);
CUDF_LTO_DECL(greater, duration_s);
CUDF_LTO_DECL(greater, duration_ms);
CUDF_LTO_DECL(greater, duration_ns);
CUDF_LTO_DECL(greater, string_view);

CUDF_LTO_DECL(less_equal, bool);
CUDF_LTO_DECL(less_equal, int8_t);
CUDF_LTO_DECL(less_equal, int16_t);
CUDF_LTO_DECL(less_equal, int32_t);
CUDF_LTO_DECL(less_equal, int64_t);
CUDF_LTO_DECL(less_equal, uint8_t);
CUDF_LTO_DECL(less_equal, uint16_t);
CUDF_LTO_DECL(less_equal, uint32_t);
CUDF_LTO_DECL(less_equal, uint64_t);
CUDF_LTO_DECL(less_equal, float32_t);
CUDF_LTO_DECL(less_equal, float64_t);
CUDF_LTO_DECL(less_equal, decimal32);
CUDF_LTO_DECL(less_equal, decimal64);
CUDF_LTO_DECL(less_equal, decimal128);
CUDF_LTO_DECL(less_equal, timestamp_D);
CUDF_LTO_DECL(less_equal, timestamp_s);
CUDF_LTO_DECL(less_equal, timestamp_ms);
CUDF_LTO_DECL(less_equal, timestamp_us);
CUDF_LTO_DECL(less_equal, timestamp_ns);
CUDF_LTO_DECL(less_equal, duration_D);
CUDF_LTO_DECL(less_equal, duration_s);
CUDF_LTO_DECL(less_equal, duration_ms);
CUDF_LTO_DECL(less_equal, duration_ns);
CUDF_LTO_DECL(less_equal, string_view);

CUDF_LTO_DECL(greater_equal, bool);
CUDF_LTO_DECL(greater_equal, int8_t);
CUDF_LTO_DECL(greater_equal, int16_t);
CUDF_LTO_DECL(greater_equal, int32_t);
CUDF_LTO_DECL(greater_equal, int64_t);
CUDF_LTO_DECL(greater_equal, uint8_t);
CUDF_LTO_DECL(greater_equal, uint16_t);
CUDF_LTO_DECL(greater_equal, uint32_t);
CUDF_LTO_DECL(greater_equal, uint64_t);
CUDF_LTO_DECL(greater_equal, float32_t);
CUDF_LTO_DECL(greater_equal, float64_t);
CUDF_LTO_DECL(greater_equal, decimal32);
CUDF_LTO_DECL(greater_equal, decimal64);
CUDF_LTO_DECL(greater_equal, decimal128);
CUDF_LTO_DECL(greater_equal, timestamp_D);
CUDF_LTO_DECL(greater_equal, timestamp_s);
CUDF_LTO_DECL(greater_equal, timestamp_ms);
CUDF_LTO_DECL(greater_equal, timestamp_us);
CUDF_LTO_DECL(greater_equal, timestamp_ns);
CUDF_LTO_DECL(greater_equal, duration_D);
CUDF_LTO_DECL(greater_equal, duration_s);
CUDF_LTO_DECL(greater_equal, duration_ms);
CUDF_LTO_DECL(greater_equal, duration_ns);
CUDF_LTO_DECL(greater_equal, string_view);

#undef CUDF_LTO_DECL

#define CUDF_LTO_DECL(op, type)                                              \
  __device__ extern inline void op(type* out, type const* a, type const* b); \
                                                                             \
  __device__ extern inline void op(                                          \
    optional<type>* out, optional<type> const* a, optional<type> const* b)

CUDF_LTO_DECL(bitwise_and, int32_t);
CUDF_LTO_DECL(bitwise_and, int64_t);
CUDF_LTO_DECL(bitwise_and, uint32_t);
CUDF_LTO_DECL(bitwise_and, uint64_t);

CUDF_LTO_DECL(bitwise_or, int32_t);
CUDF_LTO_DECL(bitwise_or, int64_t);
CUDF_LTO_DECL(bitwise_or, uint32_t);
CUDF_LTO_DECL(bitwise_or, uint64_t);

CUDF_LTO_DECL(bitwise_xor, int32_t);
CUDF_LTO_DECL(bitwise_xor, int64_t);
CUDF_LTO_DECL(bitwise_xor, uint32_t);
CUDF_LTO_DECL(bitwise_xor, uint64_t);

#undef CUDF_LTO_DECL

#define CUDF_LTO_DECL(op, type)                                              \
  __device__ extern inline void op(type* out, type const* a, type const* b); \
                                                                             \
  __device__ extern inline void op(                                          \
    optional<type>* out, optional<type> const* a, optional<type> const* b);

CUDF_LTO_DECL(logical_and, bool);

CUDF_LTO_DECL(null_logical_and, bool);

CUDF_LTO_DECL(logical_or, bool);

CUDF_LTO_DECL(null_logical_or, bool);

#undef CUDF_LTO_DECL

#define CUDF_LTO_DECL(op, type)                               \
  __device__ extern inline void op(type* out, type const* a); \
                                                              \
  __device__ extern inline void op(optional<type>* out, optional<type> const* a)

CUDF_LTO_DECL(identity, bool);
CUDF_LTO_DECL(identity, int8_t);
CUDF_LTO_DECL(identity, int16_t);
CUDF_LTO_DECL(identity, int32_t);
CUDF_LTO_DECL(identity, int64_t);
CUDF_LTO_DECL(identity, uint8_t);
CUDF_LTO_DECL(identity, uint16_t);
CUDF_LTO_DECL(identity, uint32_t);
CUDF_LTO_DECL(identity, uint64_t);
CUDF_LTO_DECL(identity, float32_t);
CUDF_LTO_DECL(identity, float64_t);
CUDF_LTO_DECL(identity, decimal32);
CUDF_LTO_DECL(identity, decimal64);
CUDF_LTO_DECL(identity, decimal128);
CUDF_LTO_DECL(identity, timestamp_D);
CUDF_LTO_DECL(identity, timestamp_s);
CUDF_LTO_DECL(identity, timestamp_ms);
CUDF_LTO_DECL(identity, timestamp_us);
CUDF_LTO_DECL(identity, timestamp_ns);
CUDF_LTO_DECL(identity, duration_D);
CUDF_LTO_DECL(identity, duration_s);
CUDF_LTO_DECL(identity, duration_ms);
CUDF_LTO_DECL(identity, duration_ns);
CUDF_LTO_DECL(identity, string_view);

CUDF_LTO_DECL(sin, float32_t);
CUDF_LTO_DECL(sin, float64_t);

CUDF_LTO_DECL(cos, float32_t);
CUDF_LTO_DECL(cos, float64_t);

CUDF_LTO_DECL(tan, float32_t);
CUDF_LTO_DECL(tan, float64_t);

CUDF_LTO_DECL(arcsin, float32_t);
CUDF_LTO_DECL(arcsin, float64_t);

CUDF_LTO_DECL(arccos, float32_t);
CUDF_LTO_DECL(arccos, float64_t);

CUDF_LTO_DECL(arctan, float32_t);
CUDF_LTO_DECL(arctan, float64_t);

CUDF_LTO_DECL(sinh, float32_t);
CUDF_LTO_DECL(sinh, float64_t);

CUDF_LTO_DECL(cosh, float32_t);
CUDF_LTO_DECL(cosh, float64_t);

CUDF_LTO_DECL(tanh, float32_t);
CUDF_LTO_DECL(tanh, float64_t);

CUDF_LTO_DECL(arcsinh, float32_t);
CUDF_LTO_DECL(arcsinh, float64_t);

CUDF_LTO_DECL(arccosh, float32_t);
CUDF_LTO_DECL(arccosh, float64_t);

CUDF_LTO_DECL(arctanh, float32_t);
CUDF_LTO_DECL(arctanh, float64_t);

CUDF_LTO_DECL(exp, float32_t);
CUDF_LTO_DECL(exp, float64_t);

CUDF_LTO_DECL(log, float32_t);
CUDF_LTO_DECL(log, float64_t);

CUDF_LTO_DECL(cbrt, float32_t);
CUDF_LTO_DECL(cbrt, float64_t);

CUDF_LTO_DECL(ceil, float32_t);
CUDF_LTO_DECL(ceil, float64_t);

CUDF_LTO_DECL(floor, float32_t);
CUDF_LTO_DECL(floor, float64_t);

CUDF_LTO_DECL(abs, int32_t);
CUDF_LTO_DECL(abs, int64_t);
CUDF_LTO_DECL(abs, float32_t);
CUDF_LTO_DECL(abs, float64_t);

CUDF_LTO_DECL(rint, float32_t);
CUDF_LTO_DECL(rint, float64_t);

CUDF_LTO_DECL(bit_invert, uint32_t);
CUDF_LTO_DECL(bit_invert, uint64_t);
CUDF_LTO_DECL(bit_invert, int32_t);
CUDF_LTO_DECL(bit_invert, int64_t);

#undef CUDF_LTO_DECL

#define CUDF_LTO_DECL(op, ret_type, type)                         \
  __device__ extern inline void op(ret_type* out, type const* a); \
                                                                  \
  __device__ extern inline void op(optional<ret_type>* out, optional<type> const* a)

CUDF_LTO_DECL(cast_to_int64, int64_t, bool);
CUDF_LTO_DECL(cast_to_int64, int64_t, int8_t);
CUDF_LTO_DECL(cast_to_int64, int64_t, int16_t);
CUDF_LTO_DECL(cast_to_int64, int64_t, int32_t);
CUDF_LTO_DECL(cast_to_int64, int64_t, int64_t);
CUDF_LTO_DECL(cast_to_int64, int64_t, uint8_t);
CUDF_LTO_DECL(cast_to_int64, int64_t, uint16_t);
CUDF_LTO_DECL(cast_to_int64, int64_t, uint32_t);
CUDF_LTO_DECL(cast_to_int64, int64_t, uint64_t);
CUDF_LTO_DECL(cast_to_int64, int64_t, float32_t);
CUDF_LTO_DECL(cast_to_int64, int64_t, float64_t);

CUDF_LTO_DECL(cast_to_uint64, uint64_t, bool);
CUDF_LTO_DECL(cast_to_uint64, uint64_t, int8_t);
CUDF_LTO_DECL(cast_to_uint64, uint64_t, int16_t);
CUDF_LTO_DECL(cast_to_uint64, uint64_t, int32_t);
CUDF_LTO_DECL(cast_to_uint64, uint64_t, int64_t);
CUDF_LTO_DECL(cast_to_uint64, uint64_t, uint8_t);
CUDF_LTO_DECL(cast_to_uint64, uint64_t, uint16_t);
CUDF_LTO_DECL(cast_to_uint64, uint64_t, uint32_t);
CUDF_LTO_DECL(cast_to_uint64, uint64_t, uint64_t);
CUDF_LTO_DECL(cast_to_uint64, uint64_t, float32_t);
CUDF_LTO_DECL(cast_to_uint64, uint64_t, float64_t);

CUDF_LTO_DECL(cast_to_float64, float64_t, bool);
CUDF_LTO_DECL(cast_to_float64, float64_t, int8_t);
CUDF_LTO_DECL(cast_to_float64, float64_t, int16_t);
CUDF_LTO_DECL(cast_to_float64, float64_t, int32_t);
CUDF_LTO_DECL(cast_to_float64, float64_t, int64_t);
CUDF_LTO_DECL(cast_to_float64, float64_t, uint8_t);
CUDF_LTO_DECL(cast_to_float64, float64_t, uint16_t);
CUDF_LTO_DECL(cast_to_float64, float64_t, uint32_t);
CUDF_LTO_DECL(cast_to_float64, float64_t, uint64_t);
CUDF_LTO_DECL(cast_to_float64, float64_t, float32_t);
CUDF_LTO_DECL(cast_to_float64, float64_t, float64_t);

#undef CUDF_LTO_DECL

#define CUDF_LTO_DECL(op, type)                                         \
  __device__ extern inline void op(bool* out, type const* a);           \
                                                                        \
  __device__ extern inline void op(bool* out, optional<type> const* a); \
                                                                        \
  __device__ extern inline void op(optional<bool>* out, optional<type> const* a)

CUDF_LTO_DECL(is_null, bool);
CUDF_LTO_DECL(is_null, int8_t);
CUDF_LTO_DECL(is_null, int16_t);
CUDF_LTO_DECL(is_null, int32_t);
CUDF_LTO_DECL(is_null, int64_t);
CUDF_LTO_DECL(is_null, uint8_t);
CUDF_LTO_DECL(is_null, uint16_t);
CUDF_LTO_DECL(is_null, uint32_t);
CUDF_LTO_DECL(is_null, uint64_t);
CUDF_LTO_DECL(is_null, float32_t);
CUDF_LTO_DECL(is_null, float64_t);
CUDF_LTO_DECL(is_null, decimal32);
CUDF_LTO_DECL(is_null, decimal64);
CUDF_LTO_DECL(is_null, decimal128);
CUDF_LTO_DECL(is_null, timestamp_D);
CUDF_LTO_DECL(is_null, timestamp_s);
CUDF_LTO_DECL(is_null, timestamp_ms);
CUDF_LTO_DECL(is_null, timestamp_us);
CUDF_LTO_DECL(is_null, timestamp_ns);
CUDF_LTO_DECL(is_null, duration_D);
CUDF_LTO_DECL(is_null, duration_s);
CUDF_LTO_DECL(is_null, duration_ms);
CUDF_LTO_DECL(is_null, duration_ns);
CUDF_LTO_DECL(is_null, string_view);

CUDF_LTO_DECL(logical_not, bool);
CUDF_LTO_DECL(logical_not, int8_t);
CUDF_LTO_DECL(logical_not, int16_t);
CUDF_LTO_DECL(logical_not, int32_t);
CUDF_LTO_DECL(logical_not, int64_t);
CUDF_LTO_DECL(logical_not, uint8_t);
CUDF_LTO_DECL(logical_not, uint16_t);
CUDF_LTO_DECL(logical_not, uint32_t);
CUDF_LTO_DECL(logical_not, uint64_t);

#undef CUDF_LTO_DECL

}  // namespace operators

}  // namespace lto
}  // namespace CUDF_LTO_EXPORT cudf
