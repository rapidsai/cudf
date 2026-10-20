/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "codecs.cuh"

#include <cudf/detail/row_ir/opcode.hpp>

namespace cudf::row_abi::detail {
template <typename T, typename P>
__device__ auto encode_result(P v)
{
  if constexpr (cudf::detail::row_ir::is_fallible_result<P>) {
    if (!v.has_value()) { return expected<T, status>{{}, static_cast<status>(v.error())}; }
    return expected<T, status>{codec<T>::encode(v.value()), success};
  } else if constexpr (cudf::detail::ops::nullable<P>) {
    return v.has_value() ? optional<T>{codec<T>::encode(v.value()), 1} : optional<T>{};
  } else {
    return codec<T>::encode(v);
  }
}

template <cudf::detail::row_ir::opcode Op, typename T, typename... In>
__device__ auto invoke(In... in)
{
  return encode_result<T>(
    cudf::detail::row_ir::evaluate<Op, cudf::error_policy::PROPAGATE>(decode(in)...));
}
}  // namespace cudf::row_abi::detail

#define CUDF_ROW_UNARY(OP, RESULT, INPUT, OUT, IN)                                        \
  extern "C" __device__ __inline_hint__ RESULT(OUT) cudf_row_op_##OP##_##IN(INPUT(IN) a0) \
  {                                                                                       \
    return detail::invoke<cudf::detail::row_ir::opcode::OP, OUT>(a0);                     \
  }
#define CUDF_ROW_BINARY(OP, RESULT, INPUT, OUT, LHS, RHS)                 \
  extern "C" __device__ __inline_hint__ RESULT(OUT)                       \
    cudf_row_op_##OP##_##LHS##_##RHS(INPUT(LHS) a0, INPUT(RHS) a1)        \
  {                                                                       \
    return detail::invoke<cudf::detail::row_ir::opcode::OP, OUT>(a0, a1); \
  }
#define CUDF_ROW_TERNARY(OP, RESULT, INPUT, OUT, LHS, RHS, PRED)                            \
  extern "C" __device__ __inline_hint__ RESULT(OUT)                                         \
    cudf_row_op_##OP##_##LHS##_##RHS##_##PRED(INPUT(LHS) a0, INPUT(RHS) a1, INPUT(PRED) a2) \
  {                                                                                         \
    return detail::invoke<cudf::detail::row_ir::opcode::OP, OUT>(a0, a1, a2);               \
  }
#define CUDF_ROW_TYPES_ALL(M, ...)       \
  M(__VA_ARGS__, bool8)                  \
  M(__VA_ARGS__, decimal128)             \
  M(__VA_ARGS__, decimal32)              \
  M(__VA_ARGS__, decimal64)              \
  M(__VA_ARGS__, duration_days)          \
  M(__VA_ARGS__, duration_microseconds)  \
  M(__VA_ARGS__, duration_milliseconds)  \
  M(__VA_ARGS__, duration_nanoseconds)   \
  M(__VA_ARGS__, duration_seconds)       \
  M(__VA_ARGS__, f32)                    \
  M(__VA_ARGS__, f64)                    \
  M(__VA_ARGS__, i16)                    \
  M(__VA_ARGS__, i32)                    \
  M(__VA_ARGS__, i64)                    \
  M(__VA_ARGS__, i8)                     \
  M(__VA_ARGS__, string_view)            \
  M(__VA_ARGS__, timestamp_days)         \
  M(__VA_ARGS__, timestamp_microseconds) \
  M(__VA_ARGS__, timestamp_milliseconds) \
  M(__VA_ARGS__, timestamp_nanoseconds)  \
  M(__VA_ARGS__, timestamp_seconds)      \
  M(__VA_ARGS__, u16)                    \
  M(__VA_ARGS__, u32)                    \
  M(__VA_ARGS__, u64)                    \
  M(__VA_ARGS__, u8)
#define CUDF_ROW_TYPES_ARITHMETIC(M, ...) \
  M(__VA_ARGS__, decimal128)              \
  M(__VA_ARGS__, decimal32)               \
  M(__VA_ARGS__, decimal64)               \
  M(__VA_ARGS__, f32)                     \
  M(__VA_ARGS__, f64)                     \
  M(__VA_ARGS__, i16)                     \
  M(__VA_ARGS__, i32)                     \
  M(__VA_ARGS__, i64)                     \
  M(__VA_ARGS__, i8)                      \
  M(__VA_ARGS__, u16)                     \
  M(__VA_ARGS__, u32)                     \
  M(__VA_ARGS__, u64)                     \
  M(__VA_ARGS__, u8)
#define CUDF_ROW_TYPES_INTEGERS(M, ...) \
  M(__VA_ARGS__, i16)                   \
  M(__VA_ARGS__, i32)                   \
  M(__VA_ARGS__, i64)                   \
  M(__VA_ARGS__, i8)                    \
  M(__VA_ARGS__, u16)                   \
  M(__VA_ARGS__, u32)                   \
  M(__VA_ARGS__, u64)                   \
  M(__VA_ARGS__, u8)
#define CUDF_ROW_TYPES_INTEGRALS(M, ...) \
  M(__VA_ARGS__, bool8)                  \
  M(__VA_ARGS__, i16)                    \
  M(__VA_ARGS__, i32)                    \
  M(__VA_ARGS__, i64)                    \
  M(__VA_ARGS__, i8)                     \
  M(__VA_ARGS__, u16)                    \
  M(__VA_ARGS__, u32)                    \
  M(__VA_ARGS__, u64)                    \
  M(__VA_ARGS__, u8)
#define CUDF_ROW_TYPES_FLOATS(M, ...) \
  M(__VA_ARGS__, f32)                 \
  M(__VA_ARGS__, f64)
#define CUDF_ROW_TYPES_DECIMALS(M, ...) \
  M(__VA_ARGS__, decimal128)            \
  M(__VA_ARGS__, decimal32)             \
  M(__VA_ARGS__, decimal64)
#define CUDF_ROW_TYPES_NUMERIC(M, ...) \
  M(__VA_ARGS__, f32)                  \
  M(__VA_ARGS__, f64)                  \
  M(__VA_ARGS__, i16)                  \
  M(__VA_ARGS__, i32)                  \
  M(__VA_ARGS__, i64)                  \
  M(__VA_ARGS__, i8)                   \
  M(__VA_ARGS__, u16)                  \
  M(__VA_ARGS__, u32)                  \
  M(__VA_ARGS__, u64)                  \
  M(__VA_ARGS__, u8)
#define CUDF_ROW_TYPES_SCALARS(M, ...) \
  M(__VA_ARGS__, bool8)                \
  M(__VA_ARGS__, decimal128)           \
  M(__VA_ARGS__, decimal32)            \
  M(__VA_ARGS__, decimal64)            \
  M(__VA_ARGS__, f32)                  \
  M(__VA_ARGS__, f64)                  \
  M(__VA_ARGS__, i16)                  \
  M(__VA_ARGS__, i32)                  \
  M(__VA_ARGS__, i64)                  \
  M(__VA_ARGS__, i8)                   \
  M(__VA_ARGS__, u16)                  \
  M(__VA_ARGS__, u32)                  \
  M(__VA_ARGS__, u64)                  \
  M(__VA_ARGS__, u8)

#define CUDF_ROW_VALUE(T)                              T
#define CUDF_ROW_OPTIONAL(T)                           optional<T>
#define CUDF_ROW_EXPECTED(T)                           expected<T, status>
#define CUDF_ROW_UNARY_SAME(OP, RESULT, INPUT, T)      CUDF_ROW_UNARY(OP, RESULT, INPUT, T, T)
#define CUDF_ROW_UNARY_TO(OP, RESULT, INPUT, OUT, T)   CUDF_ROW_UNARY(OP, RESULT, INPUT, OUT, T)
#define CUDF_ROW_BINARY_SAME(OP, RESULT, INPUT, T)     CUDF_ROW_BINARY(OP, RESULT, INPUT, T, T, T)
#define CUDF_ROW_BINARY_TO(OP, RESULT, INPUT, OUT, T)  CUDF_ROW_BINARY(OP, RESULT, INPUT, OUT, T, T)
#define CUDF_ROW_BINARY_RHS(OP, RESULT, INPUT, RHS, T) CUDF_ROW_BINARY(OP, RESULT, INPUT, T, T, RHS)
#define CUDF_ROW_IF_ELSE_PREDICATE(OP, RESULT, INPUT, T, PRED) \
  CUDF_ROW_TERNARY(OP, RESULT, INPUT, T, T, T, PRED)
#define CUDF_ROW_IF_ELSE(OP, RESULT, INPUT, T) \
  CUDF_ROW_TYPES_INTEGRALS(CUDF_ROW_IF_ELSE_PREDICATE, OP, RESULT, INPUT, T)
