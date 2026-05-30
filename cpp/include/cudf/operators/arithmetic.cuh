/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/detail/integral_math.cuh>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/operators/concepts.cuh>
#include <cudf/operators/types.cuh>
#include <cudf/utilities/export.hpp>

#include <cuda/std/cmath>
#include <cuda/std/concepts>
#include <cuda/std/type_traits>

namespace CUDF_EXPORT cudf {
namespace ops {

/**
 * @brief Computes absolute value.
 *
 * @tparam T Value type.
 * @param a Input value.
 */
template <typename T>
__device__ T abs(T a)
  requires(cuda::std::is_signed_v<T>)
{
  return cuda::std::abs(a);
}

/**
 * @brief Computes absolute value.
 *
 * @tparam T Value type.
 * @param a Input value.
 */
template <unsigned_integer T>
__device__ T abs(T a)
{
  return a;
}

/**
 * @brief Computes absolute value.
 *
 * @tparam R Decimal representation type.
 */
template <typename R>
__device__ numeric::decimal<R> abs(numeric::decimal<R> a)
{
  auto rep = a.value() < 0 ? -a.value() : a.value();
  return numeric::decimal<R>{numeric::scaled_integer<R>{rep, a.scale()}};
}

/**
 * @brief Computes sum of two values.
 *
 * @tparam T Value type.
 * @param a Left operand.
 * @param b Right operand.
 */
template <arithmetic T>
__device__ T add(T a, T b)
{
  return (a + b);
}

/**
 * @brief Computes quotient of two values.
 *
 * @tparam T Value type.
 * @param a Dividend.
 * @param b Divisor.
 */
template <arithmetic T>
__device__ T div(T a, T b)
{
  return (a / b);
}

/**
 * @brief Computes floor division of two values.
 *
 * @tparam T Value type.
 * @param a Dividend.
 * @param b Divisor.
 */
template <integer T>
__device__ T floor_div(T a, T b)
{
  return cudf::detail::integral_floor_div(a, b);
}

/**
 * @brief Computes floor division of two values.
 *
 * @tparam T Value type.
 * @param a Dividend.
 * @param b Divisor.
 */
template <floating_point T>
__device__ T floor_div(T a, T b)
{
  return cuda::std::floor(a / b);
}

/**
 * @brief Computes remainder of two values.
 *
 * @tparam T Value type.
 * @param a Dividend.
 * @param b Divisor.
 */
template <typename T>
__device__ T mod(T a, T b)
  requires(integer<T> || fixed_point<T>)
{
  return (a % b);
}

/**
 * @brief Computes remainder of two values.
 *
 * @tparam T Value type.
 * @param a Dividend.
 * @param b Divisor.
 */
template <typename T>
__device__ T mod(T a, T b)
  requires(floating_point<T>)
{
  return cuda::std::fmod(a, b);
}

/**
 * @brief Computes Python-style modulus.
 *
 * @tparam T Value type.
 * @param a Dividend.
 * @param b Divisor.
 */
template <typename T>
__device__ T pymod(T a, T b)
  requires(integer<T> || fixed_point<T>)
{
  return (a % b + b) % b;
}

/**
 * @brief Computes Python-style modulus.
 *
 * @tparam T Value type.
 * @param a Dividend.
 * @param b Divisor.
 */
template <floating_point T>
__device__ T pymod(T a, T b)
{
  return cuda::std::fmod(cuda::std::fmod(a, b) + b, b);
}

/**
 * @brief Computes product of two values.
 *
 * @tparam T Value type.
 * @param a Left operand.
 * @param b Right operand.
 */
template <arithmetic T>
__device__ T mul(T a, T b)
{
  return (a * b);
}

/**
 * @brief Computes unary negation.
 *
 * @tparam T Value type.
 * @param a Input value.
 */
template <typename T>
__device__ T neg(T a)
  requires(signed_integer<T> || floating_point<T>)
{
  return -a;
}

/**
 * @brief Computes unary negation.
 *
 * @tparam R Decimal representation type.
 * @param a Input value.
 */
template <typename R>
__device__ numeric::decimal<R> neg(numeric::decimal<R> a)
{
  auto rep = -a.value();
  return numeric::decimal<R>{numeric::scaled_integer<R>{rep, a.scale()}};
}

/**
 * @brief Computes subtraction of two values.
 *
 * @tparam T Value type.
 * @param a Minuend.
 * @param b Subtrahend.
 */
template <arithmetic T>
__device__ T sub(T a, T b)
{
  return a - b;
}

/**
 * @brief Computes true division and returns a double.
 *
 * @tparam T Value type.
 * @param a Dividend.
 * @param b Divisor.
 */
template <typename T>
__device__ double true_div(T a, T b)
  requires(floating_point<T> || integer<T>)
{
  return static_cast<double>(a) / static_cast<double>(b);
}

}  // namespace ops
}  // namespace CUDF_EXPORT cudf
