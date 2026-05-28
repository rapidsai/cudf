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
 * @tparam T Value type.
 * @param out Result destination.
 * @param a Left operand.
 * @param b Right operand.
 */
template <typename T>
__device__ void bit_and(T* out, T const* a, T const* b)
{
  *out = (*a & *b);
}

/**
 * @brief Computes bitwise AND of two values.
 *
 * @tparam T Value type.
 * @param out Result destination.
 * @param a Left operand.
 * @param b Right operand.
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
 * @tparam T Value type.
 * @param out Result destination.
 * @param a Input value.
 */
template <typename T>
__device__ void bit_invert(T* out, T const* a)
{
  *out = ~(*a);
}

/**
 * @brief Computes bitwise NOT of one value.
 *
 * @tparam T Value type.
 * @param out Result destination.
 * @param a Input value.
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
 * @tparam T Value type.
 * @param out Result destination.
 * @param a Left operand.
 * @param b Right operand.
 */
template <typename T>
__device__ void bit_or(T* out, T const* a, T const* b)
{
  *out = (*a | *b);
}

/**
 * @brief Computes bitwise OR of two values.
 *
 * @tparam T Value type.
 * @param out Result destination.
 * @param a Left operand.
 * @param b Right operand.
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
 * @tparam T Value type.
 * @param out Result destination.
 * @param a Left operand.
 * @param b Right operand.
 */
template <typename T>
__device__ void bit_xor(T* out, T const* a, T const* b)
{
  *out = (*a ^ *b);
}

/**
 * @brief Computes bitwise XOR of two values.
 *
 * @tparam T Value type.
 * @param out Result destination.
 * @param a Left operand.
 * @param b Right operand.
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
 * @tparam T Value type.
 * @param out Result destination.
 * @param a Input value.
 * @param b Shift count.
 */
template <typename T>
__device__ void bit_shift_left(T* out, T const* a, T const* b)
{
  *out = (*a << *b);
}

/**
 * @brief Shifts a value left by a bit count.
 *
 * @tparam T Value type.
 * @param out Result destination.
 * @param a Input value.
 * @param b Shift count.
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
 * @tparam T Value type.
 * @param out Result destination.
 * @param a Input value.
 * @param b Shift count.
 */
template <typename T>
__device__ void bit_shift_right(T* out, T const* a, T const* b)
{
  *out = (*a >> *b);
}

/**
 * @brief Shifts a value right by a bit count.
 *
 * @tparam T Value type.
 * @param out Result destination.
 * @param a Input value.
 * @param b Shift count.
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
