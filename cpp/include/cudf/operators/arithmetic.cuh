/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/detail/integral_math.cuh>
#include <cudf/operators/types.cuh>

namespace CUDF_EXPORT cudf {
namespace ops {

/**
 * @brief Computes absolute value for signed and floating-point inputs.
 *
 * @tparam T Input and output type.
 * @param out Destination value.
 * @param a Input value.
 * @return errc::OK.
 */
template <typename T>
  requires(cuda::std::is_signed_v<T> || cuda::std::is_floating_point_v<T>)
__device__ inline errc abs(T* out, T const* a)
{
  *out = (*a < 0) ? -*a : *a;
  return errc::OK;
}

/**
 * @brief Returns unsigned input unchanged for absolute value.
 *
 * @tparam T Unsigned input and output type.
 * @param out Destination value.
 * @param a Input value.
 * @return errc::OK.
 */
template <typename T>
  requires(cuda::std::is_unsigned_v<T>)
__device__ inline errc abs(T* out, T const* a)
{
  *out = *a;
  return errc::OK;
}

/**
 * @brief Computes absolute value for fixed-point decimal values.
 *
 * @tparam R Decimal representation type.
 * @param out Destination decimal value.
 * @param a Input decimal value.
 * @return errc::OK.
 */
template <typename R>
__device__ inline errc abs(decimal<R>* out, decimal<R> const* a)
{
  auto rep = a->value() < 0 ? -a->value() : a->value();
  *out     = decimal<R>{numeric::scaled_integer<R>{rep, a->scale()}};
  return errc::OK;
}

/**
 * @brief Computes absolute value for optional input.
 *
 * @tparam T Input and output type.
 * @param out Destination optional value.
 * @param a Optional input value.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc abs(optional<T>* out, optional<T> const* a)
{
  if (a->has_value()) {
    T r;
    abs(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

/**
 * @brief Computes sum of two values.
 *
 * @tparam T Operand and result type.
 * @param out Destination value.
 * @param a Left operand.
 * @param b Right operand.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc add(T* out, T const* a, T const* b)
{
  *out = (*a + *b);
  return errc::OK;
}

/**
 * @brief Computes sum for optional operands.
 *
 * @tparam T Operand and result type.
 * @param out Destination optional value.
 * @param a Left optional operand.
 * @param b Right optional operand.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc add(optional<T>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    T r;
    add(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

/**
 * @brief Computes quotient of two values.
 *
 * @tparam T Operand and result type.
 * @param out Destination value.
 * @param a Dividend.
 * @param b Divisor.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc div(T* out, T const* a, T const* b)
{
  *out = (*a / *b);
  return errc::OK;
}

/**
 * @brief Computes quotient for optional operands.
 *
 * @tparam T Operand and result type.
 * @param out Destination optional value.
 * @param a Optional dividend.
 * @param b Optional divisor.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc div(optional<T>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    T r;
    div(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

/**
 * @brief Computes floor division for integral operands.
 *
 * @tparam T Integral operand and result type.
 * @param out Destination value.
 * @param a Dividend.
 * @param b Divisor.
 * @return errc::OK.
 */
template <typename T>
  requires(cuda::std::is_integral_v<T>)
__device__ inline errc floor_div(T* out, T const* a, T const* b)
{
  *out = cudf::detail::integral_floor_div(*a, *b);
  return errc::OK;
}

/**
 * @brief Computes floor division for float operands.
 *
 * @param out Destination value.
 * @param a Dividend.
 * @param b Divisor.
 * @return errc::OK.
 */
__device__ inline errc floor_div(float* out, float const* a, float const* b)
{
  *out = ::floorf(*a / *b);
  return errc::OK;
}

/**
 * @brief Computes floor division for double operands.
 *
 * @param out Destination value.
 * @param a Dividend.
 * @param b Divisor.
 * @return errc::OK.
 */
__device__ inline errc floor_div(double* out, double const* a, double const* b)
{
  *out = ::floor(*a / *b);
  return errc::OK;
}

/**
 * @brief Computes floor division for optional operands.
 *
 * @tparam T Operand and result type.
 * @param out Destination optional value.
 * @param a Optional dividend.
 * @param b Optional divisor.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc floor_div(optional<T>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    T r;
    floor_div(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

/**
 * @brief Computes remainder of integer division.
 *
 * @tparam T Operand and result type.
 * @param out Destination value.
 * @param a Dividend.
 * @param b Divisor.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc mod(T* out, T const* a, T const* b)
{
  *out = (*a % *b);
  return errc::OK;
}

/**
 * @brief Computes floating-point remainder for float operands.
 *
 * @param out Destination value.
 * @param a Dividend.
 * @param b Divisor.
 * @return errc::OK.
 */
__device__ inline errc mod(float* out, float const* a, float const* b)
{
  *out = ::fmodf(*a, *b);
  return errc::OK;
}

/**
 * @brief Computes floating-point remainder for double operands.
 *
 * @param out Destination value.
 * @param a Dividend.
 * @param b Divisor.
 * @return errc::OK.
 */
__device__ inline errc mod(double* out, double const* a, double const* b)
{
  *out = ::fmod(*a, *b);
  return errc::OK;
}

/**
 * @brief Computes remainder for optional operands.
 *
 * @tparam T Operand and result type.
 * @param out Destination optional value.
 * @param a Optional dividend.
 * @param b Optional divisor.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc mod(optional<T>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    T r;
    mod(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

/**
 * @brief Computes Python-style modulus.
 *
 * @tparam T Integral operand and result type.
 * @param out Destination value.
 * @param a Dividend.
 * @param b Divisor.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc pymod(T* out, T const* a, T const* b)
{
  *out = (*a % *b + *b) % *b;
  return errc::OK;
}

/**
 * @brief Computes Python-style modulus for float operands.
 *
 * @param out Destination value.
 * @param a Dividend.
 * @param b Divisor.
 * @return errc::OK.
 */
__device__ inline errc pymod(float* out, float const* a, float const* b)
{
  *out = ::fmodf(::fmodf(*a, *b) + *b, *b);
  return errc::OK;
}

/**
 * @brief Computes Python-style modulus for double operands.
 *
 * @param out Destination value.
 * @param a Dividend.
 * @param b Divisor.
 * @return errc::OK.
 */
__device__ inline errc pymod(double* out, double const* a, double const* b)
{
  *out = ::fmod(::fmod(*a, *b) + *b, *b);
  return errc::OK;
}

/**
 * @brief Computes Python-style modulus for optional operands.
 *
 * @tparam T Operand and result type.
 * @param out Destination optional value.
 * @param a Optional dividend.
 * @param b Optional divisor.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc pymod(optional<T>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    T r;
    pymod(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

/**
 * @brief Computes product of two values.
 *
 * @tparam T Operand and result type.
 * @param out Destination value.
 * @param a Left operand.
 * @param b Right operand.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc mul(T* out, T const* a, T const* b)
{
  *out = (*a * *b);
  return errc::OK;
}

/**
 * @brief Computes product for optional operands.
 *
 * @tparam T Operand and result type.
 * @param out Destination optional value.
 * @param a Left optional operand.
 * @param b Right optional operand.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc mul(optional<T>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    T r;
    mul(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

/**
 * @brief Computes unary negation for signed inputs.
 *
 * @tparam T Signed input and output type.
 * @param out Destination value.
 * @param a Input value.
 * @return errc::OK.
 */
template <typename T>
  requires(cuda::std::is_signed_v<T>)
__device__ inline errc neg(T* out, T const* a)
{
  *out = -(*a);
  return errc::OK;
}

/**
 * @brief Computes unary negation for decimal inputs.
 *
 * @tparam Rep Decimal representation type.
 * @param out Destination decimal value.
 * @param a Input decimal value.
 * @return errc::OK.
 */
template <typename Rep>
__device__ inline errc neg(decimal<Rep>* out, decimal<Rep> const* a)
{
  auto rep = -a->value();
  *out     = decimal<Rep>{numeric::scaled_integer<Rep>{rep, a->scale()}};
  return errc::OK;
}

/**
 * @brief Computes unary negation for optional input.
 *
 * @tparam T Signed input and output type.
 * @param out Destination optional value.
 * @param a Optional input value.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc neg(optional<T>* out, optional<T> const* a)
{
  if (a->has_value()) {
    T r;
    neg(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

/**
 * @brief Computes subtraction of two values.
 *
 * @tparam T Operand and result type.
 * @param out Destination value.
 * @param a Minuend.
 * @param b Subtrahend.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc sub(T* out, T const* a, T const* b)
{
  *out = *a - *b;
  return errc::OK;
}

/**
 * @brief Computes subtraction for optional operands.
 *
 * @tparam T Operand and result type.
 * @param out Destination optional value.
 * @param a Optional minuend.
 * @param b Optional subtrahend.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc sub(optional<T>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    T r;
    sub(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

/**
 * @brief Computes true division and returns a double.
 *
 * @tparam T Input operand type.
 * @param out Destination double value.
 * @param a Dividend.
 * @param b Divisor.
 * @return errc::OK.
 */
template <typename T>
  requires(cuda::std::is_floating_point_v<T> || cuda::std::is_integral_v<T>)
__device__ inline errc true_div(double* out, T const* a, T const* b)
{
  *out = static_cast<double>(*a) / static_cast<double>(*b);
  return errc::OK;
}

/**
 * @brief Computes true division for optional operands and returns optional double.
 *
 * @tparam T Input operand type.
 * @param out Destination optional double value.
 * @param a Optional dividend.
 * @param b Optional divisor.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc true_div(optional<double>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    double r;
    true_div(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

}  // namespace ops
}  // namespace CUDF_EXPORT cudf
