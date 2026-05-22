/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/utilities/export.hpp>

#include <cuda/std/optional>

namespace CUDF_EXPORT cudf {
namespace ops {

/**
 * @brief Computes bitwise AND of two values.
 *
 * @tparam T Operand and result type.
 * @param out Destination for the computed value.
 * @param a Left input operand.
 * @param b Right input operand.
 */
template <typename T>
__device__ void bit_and(T* out, T const* a, T const* b)
{
  *out = (*a & *b);
}

/**
 * @brief Computes bitwise AND for optional operands.
 *
 * @tparam T Operand and result type.
 * @param out Destination optional result.
 * @param a Left optional operand.
 * @param b Right optional operand.
 */
template <typename T>
__device__ void bit_and(cuda::std::optional<T>* out,
                        cuda::std::optional<T> const* a,
                        cuda::std::optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    T r;
    bit_and(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = cuda::std::nullopt;
  }
}

/**
 * @brief Computes bitwise NOT of one value.
 *
 * @tparam T Operand and result type.
 * @param out Destination for the computed value.
 * @param a Input operand.
 */
template <typename T>
__device__ void bit_invert(T* out, T const* a)
{
  *out = ~(*a);
}

/**
 * @brief Computes bitwise NOT for an optional operand.
 *
 * @tparam T Operand and result type.
 * @param out Destination optional result.
 * @param a Optional input operand.
 */
template <typename T>
__device__ void bit_invert(cuda::std::optional<T>* out, cuda::std::optional<T> const* a)
{
  if (a->has_value()) {
    T r;
    bit_invert(&r, &a->value());
    *out = r;
  } else {
    *out = cuda::std::nullopt;
  }
}

/**
 * @brief Computes bitwise OR of two values.
 *
 * @tparam T Operand and result type.
 * @param out Destination for the computed value.
 * @param a Left input operand.
 * @param b Right input operand.
 */
template <typename T>
__device__ void bit_or(T* out, T const* a, T const* b)
{
  *out = (*a | *b);
}

/**
 * @brief Computes bitwise OR for optional operands.
 *
 * @tparam T Operand and result type.
 * @param out Destination optional result.
 * @param a Left optional operand.
 * @param b Right optional operand.
 */
template <typename T>
__device__ void bit_or(cuda::std::optional<T>* out,
                       cuda::std::optional<T> const* a,
                       cuda::std::optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    T r;
    bit_or(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = cuda::std::nullopt;
  }
}

/**
 * @brief Computes bitwise XOR of two values.
 *
 * @tparam T Operand and result type.
 * @param out Destination for the computed value.
 * @param a Left input operand.
 * @param b Right input operand.
 */
template <typename T>
__device__ void bit_xor(T* out, T const* a, T const* b)
{
  *out = (*a ^ *b);
}

/**
 * @brief Computes bitwise XOR for optional operands.
 *
 * @tparam T Operand and result type.
 * @param out Destination optional result.
 * @param a Left optional operand.
 * @param b Right optional operand.
 */
template <typename T>
__device__ void bit_xor(cuda::std::optional<T>* out,
                        cuda::std::optional<T> const* a,
                        cuda::std::optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    T r;
    bit_xor(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = cuda::std::nullopt;
  }
}

/**
 * @brief Shifts a value left by a bit count.
 *
 * @tparam T Operand and result type.
 * @param out Destination for the computed value.
 * @param a Input value.
 * @param b Shift count.
 */
template <typename T>
__device__ void bit_shift_left(T* out, T const* a, T const* b)
{
  *out = (*a << *b);
}

/**
 * @brief Shifts an optional value left by an optional bit count.
 *
 * @tparam T Operand and result type.
 * @param out Destination optional result.
 * @param a Optional input value.
 * @param b Optional shift count.
 */
template <typename T>
__device__ void bit_shift_left(cuda::std::optional<T>* out,
                               cuda::std::optional<T> const* a,
                               cuda::std::optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    T r;
    bit_shift_left(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = cuda::std::nullopt;
  }
}

/**
 * @brief Shifts a value right by a bit count.
 *
 * @tparam T Operand and result type.
 * @param out Destination for the computed value.
 * @param a Input value.
 * @param b Shift count.
 */
template <typename T>
__device__ void bit_shift_right(T* out, T const* a, T const* b)
{
  *out = (*a >> *b);
}

/**
 * @brief Shifts an optional value right by an optional bit count.
 *
 * @tparam T Operand and result type.
 * @param out Destination optional result.
 * @param a Optional input value.
 * @param b Optional shift count.
 */
template <typename T>
__device__ void bit_shift_right(cuda::std::optional<T>* out,
                                cuda::std::optional<T> const* a,
                                cuda::std::optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    T r;
    bit_shift_right(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = cuda::std::nullopt;
  }
}

}  // namespace ops
}  // namespace CUDF_EXPORT cudf
