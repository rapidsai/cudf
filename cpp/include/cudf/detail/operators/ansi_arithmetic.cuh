/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/detail/operators/concepts.cuh>
#include <cudf/detail/operators/error.hpp>
#include <cudf/detail/operators/math.cuh>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/utilities/export.hpp>

#include <cuda/numeric>
#include <cuda/std/concepts>
#include <cuda/std/expected>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

namespace CUDF_EXPORT cudf {
namespace detail {
namespace ops {

/**
 * @brief Adds operands with overflow detection.
 *
 * @tparam T Value type.
 * @param a Left operand.
 * @param b Right operand.
 * @return `errc::OVERFLOW` on overflow, else the result.
 */
template <integer T>
__device__ cuda::std::expected<T, errc> ansi_add(T a, T b)
{
  T r;
  if (cuda::add_overflow(r, a, b).overflow) { return cuda::std::unexpected{errc::OVERFLOW}; }
  return r;
}

template <floating_point T>
__device__ cuda::std::expected<T, errc> ansi_add(T a, T b)
{
  return a + b;
}

/**
 * @brief Adds operands with overflow detection.
 *
 * @tparam R Decimal representation type.
 * @return `errc::OVERFLOW` on overflow, else the result.
 */
template <typename R>
__device__ cuda::std::expected<numeric::decimal<R>, errc> ansi_add(numeric::decimal<R> a,
                                                                   numeric::decimal<R> b)
{
  auto scale = cuda::std::min(a.scale(), b.scale());

  if (numeric::addition_overflow<R>(a.rescaled(scale).value(), b.rescaled(scale).value())) {
    return cuda::std::unexpected{errc::OVERFLOW};
  }

  return numeric::decimal<R>{numeric::scaled_integer<R>{
    a.rescaled(scale).value() + b.rescaled(scale).value(), numeric::scale_type{scale}}};
}

template <integer T>
__device__ cuda::std::expected<T, errc> ansi_sub(T a, T b)
{
  T r;
  if (cuda::sub_overflow(r, a, b).overflow) { return cuda::std::unexpected{errc::OVERFLOW}; }
  return r;
}

/**
 * @brief Subtracts operands with overflow detection.
 *
 * @tparam T Value type.
 * @return `errc::OVERFLOW` on overflow, else the result.
 */
template <floating_point T>
__device__ cuda::std::expected<T, errc> ansi_sub(T a, T b)
{
  return a - b;
}

template <typename R>
__device__ cuda::std::expected<numeric::decimal<R>, errc> ansi_sub(numeric::decimal<R> a,
                                                                   numeric::decimal<R> b)
{
  auto scale = cuda::std::min(a.scale(), b.scale());

  if (numeric::subtraction_overflow<R>(a.rescaled(scale).value(), b.rescaled(scale).value())) {
    return cuda::std::unexpected{errc::OVERFLOW};
  }

  return numeric::decimal<R>{numeric::scaled_integer<R>{
    a.rescaled(scale).value() - b.rescaled(scale).value(), numeric::scale_type{scale}}};
}

/**
 * @brief Multiplies operands with overflow detection.
 *
 * @tparam T Value type.
 * @param a Left operand.
 * @param b Right operand.
 * @return `errc::OVERFLOW` on overflow, else the result.
 */
template <integer T>
__device__ cuda::std::expected<T, errc> ansi_mul(T a, T b)
{
  T r;
  if (cuda::mul_overflow(r, a, b).overflow) { return cuda::std::unexpected{errc::OVERFLOW}; }
  return r;
}

template <floating_point T>
__device__ cuda::std::expected<T, errc> ansi_mul(T a, T b)
{
  return a * b;
}

template <typename R>
__device__ cuda::std::expected<numeric::decimal<R>, errc> ansi_mul(numeric::decimal<R> a,
                                                                   numeric::decimal<R> b)
{
  if (numeric::multiplication_overflow<R>(a.value(), b.value())) {
    return cuda::std::unexpected{errc::OVERFLOW};
  }

  return numeric::decimal<R>{
    numeric::scaled_integer<R>{a.value() * b.value(), numeric::scale_type{a.scale() + b.scale()}}};
}

/**
 * @brief Divides operands with ANSI checks.
 *
 * @tparam T Value type.
 * @param a Dividend.
 * @param b Divisor.
 * @return `errc::DIVISION_BY_ZERO` on zero divisor, `errc::OVERFLOW` on overflow, else
 * `errc::SUCCESS`.
 */
template <integer T>
__device__ cuda::std::expected<T, errc> ansi_div(T a, T b)
{
  if (b == 0) { return cuda::std::unexpected{errc::DIVISION_BY_ZERO}; }
  T r;
  if (cuda::div_overflow(r, a, b).overflow) { return cuda::std::unexpected{errc::OVERFLOW}; }
  return r;
}

template <floating_point T>
__device__ cuda::std::expected<T, errc> ansi_div(T a, T b)
{
  return a / b;
}

template <typename R>
__device__ cuda::std::expected<numeric::decimal<R>, errc> ansi_div(numeric::decimal<R> a,
                                                                   numeric::decimal<R> b)
{
  if (b.value() == 0) { return cuda::std::unexpected{errc::DIVISION_BY_ZERO}; }

  if (numeric::division_overflow<R>(a.value(), b.value())) {
    return cuda::std::unexpected{errc::OVERFLOW};
  }

  return numeric::decimal<R>{
    numeric::scaled_integer<R>{a.value() / b.value(), numeric::scale_type{a.scale() - b.scale()}}};
}

/**
 * @brief Computes modulus with ANSI checks.
 *
 * @tparam T Value type.
 * @param a Dividend.
 * @param b Divisor.
 * @return `errc::DIVISION_BY_ZERO` on zero divisor, else the result.
 */
template <signed_integer T>
__device__ cuda::std::expected<T, errc> ansi_mod(T a, T b)
{
  if (b == 0) { return cuda::std::unexpected{errc::DIVISION_BY_ZERO}; }

  // avoid signed overflow UB / trap for minimum value divided by -1.
  if (a == cuda::std::numeric_limits<T>::min() && b == T{-1}) { return T{0}; }

  return a % b;
}

template <unsigned_integer T>
__device__ cuda::std::expected<T, errc> ansi_mod(T a, T b)
{
  if (b == 0) { return cuda::std::unexpected{errc::DIVISION_BY_ZERO}; }
  return a % b;
}

template <floating_point T>
__device__ cuda::std::expected<T, errc> ansi_mod(T a, T b)
{
  if (b == 0) { return cuda::std::unexpected{errc::DIVISION_BY_ZERO}; }
  return a - b * cuda::std::floor(a / b);
}

template <typename R>
__device__ cuda::std::expected<numeric::decimal<R>, errc> ansi_mod(numeric::decimal<R> a,
                                                                   numeric::decimal<R> b)
{
  auto r = ansi_div(a, b);
  if (r.has_error()) { return r.error(); }
  return a - b * floor(r.value());
}

/**
 * @brief Computes absolute value with ANSI overflow checks.
 *
 * @tparam T Value type.
 * @param a Input value.
 * @return `errc::OVERFLOW` on overflow, else the result.
 */
template <signed_integer T>
__device__ cuda::std::expected<T, errc> ansi_abs(T a)
{
  if (a == cuda::std::numeric_limits<T>::min()) { return cuda::std::unexpected{errc::OVERFLOW}; }
  return (a < 0) ? -a : a;
}

template <unsigned_integer T>
__device__ cuda::std::expected<T, errc> ansi_abs(T a)
{
  return a;
}

template <floating_point T>
__device__ cuda::std::expected<T, errc> ansi_abs(T a)
{
  return cuda::std::fabs(a);
}

template <typename R>
__device__ cuda::std::expected<numeric::decimal<R>, errc> ansi_abs(numeric::decimal<R> a)
{
  if (a.value() == cuda::std::numeric_limits<R>::min()) {
    return cuda::std::unexpected{errc::OVERFLOW};
  }
  auto rep = a.value() < 0 ? -a.value() : a.value();
  return numeric::decimal<R>{numeric::scaled_integer<R>{rep, numeric::scale_type{a.scale()}}};
}

/**
 * @brief Computes unary negation with ANSI overflow checks.
 *
 * @tparam T Value type.
 * @param a Input value.
 * @return `errc::OVERFLOW` on overflow, else the result.
 */
template <signed_integer T>
__device__ cuda::std::expected<T, errc> ansi_neg(T a)
{
  if (a == cuda::std::numeric_limits<T>::min()) { return cuda::std::unexpected{errc::OVERFLOW}; }
  return -a;
}

template <floating_point T>
__device__ cuda::std::expected<T, errc> ansi_neg(T a)
{
  return -a;
}

template <typename R>
__device__ cuda::std::expected<numeric::decimal<R>, errc> ansi_neg(numeric::decimal<R> a)
{
  if (a.value() == cuda::std::numeric_limits<R>::min()) {
    return cuda::std::unexpected{errc::OVERFLOW};
  }
  auto rep = -a.value();
  return numeric::decimal<R>{numeric::scaled_integer<R>{rep, numeric::scale_type{a.scale()}}};
}

/**
 * @brief Validates decimal precision against a target precision value.
 *
 * @tparam R Decimal representation type.
 * @param a Input decimal value.
 * @param precision Maximum allowed precision.
 * @return `errc::OVERFLOW` when precision is invalid or exceeded, else the result.
 */
template <typename R>
__device__ cuda::std::expected<numeric::decimal<R>, errc> ansi_precision_check(
  numeric::decimal<R> a, int32_t precision)
{
  if (precision <= 0) { return cuda::std::unexpected{errc::OVERFLOW}; }

  auto value = a.value();
  if (value == cuda::std::numeric_limits<R>::min()) {
    return cuda::std::unexpected{errc::OVERFLOW};
  }

  auto abs_value = value < 0 ? -value : value;

  if (abs_value >= numeric::detail::ipow<R, numeric::Radix::BASE_10>(precision)) {
    return cuda::std::unexpected{errc::OVERFLOW};
  }

  return a;
}

}  // namespace ops
}  // namespace detail
}  // namespace CUDF_EXPORT cudf
