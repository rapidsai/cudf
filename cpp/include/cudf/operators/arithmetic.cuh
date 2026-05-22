/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/detail/integral_math.cuh>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/utilities/export.hpp>

#include <cuda/std/optional>
#include <cuda/std/type_traits>

namespace CUDF_EXPORT cudf {
namespace ops {

/**
 * @brief Computes absolute value for signed and floating-point inputs.
 *
 * @tparam T Input and output type.
 * @param out Destination value.
 * @param a Input value.
 */
template <typename T>
__device__ void abs(T* out, T const* a)
  requires(cuda::std::is_signed_v<T> || cuda::std::is_floating_point_v<T>)
{
  *out = (*a < 0) ? -*a : *a;
}

/**
 * @brief Returns unsigned input unchanged for absolute value.
 *
 * @tparam T Unsigned input and output type.
 */
template <typename T>
__device__ void abs(T* out, T const* a)
  requires(cuda::std::is_unsigned_v<T>)
{
  *out = *a;
}

/**
 * @brief Computes absolute value for fixed-point decimal values.
 *
 * @tparam R Decimal representation type.
 */
template <typename R>
__device__ void abs(numeric::decimal<R>* out, numeric::decimal<R> const* a)
{
  auto rep = a->value() < 0 ? -a->value() : a->value();
  *out     = numeric::decimal<R>{numeric::scaled_integer<R>{rep, a->scale()}};
}

/**
 * @brief Computes absolute value for optional input.
 *
 * @tparam T Input and output type.
 * @param out Destination optional value.
 * @param a Optional input value.
 */
template <typename T>
__device__ void abs(cuda::std::optional<T>* out, cuda::std::optional<T> const* a)
{
  if (a->has_value()) {
    T r;
    abs(&r, &a->value());
    *out = r;
  } else {
    *out = cuda::std::nullopt;
  }
}

/**
 * @brief Computes sum of two values.
 *
 * @tparam T Operand and result type.
 * @param out Destination value.
 * @param a Left operand.
 * @param b Right operand.
 */
template <typename T>
__device__ void add(T* out, T const* a, T const* b)
{
  *out = (*a + *b);
}

/**
 * @brief Computes sum for optional operands.
 *
 * @tparam T Operand and result type.
 * @param out Destination optional value.
 * @param a Left optional operand.
 * @param b Right optional operand.
 */
template <typename T>
__device__ void add(cuda::std::optional<T>* out,
                    cuda::std::optional<T> const* a,
                    cuda::std::optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    T r;
    add(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = cuda::std::nullopt;
  }
}

/**
 * @brief Computes quotient of two values.
 *
 * @tparam T Operand and result type.
 * @param out Destination value.
 * @param a Dividend.
 * @param b Divisor.
 */
template <typename T>
__device__ void div(T* out, T const* a, T const* b)
{
  *out = (*a / *b);
}

/**
 * @brief Computes quotient for optional operands.
 *
 * @tparam T Operand and result type.
 * @param out Destination optional value.
 * @param a Optional dividend.
 * @param b Optional divisor.
 */
template <typename T>
__device__ void div(cuda::std::optional<T>* out,
                    cuda::std::optional<T> const* a,
                    cuda::std::optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    T r;
    div(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = cuda::std::nullopt;
  }
}

/**
 * @brief Computes floor division for integral operands.
 *
 * @tparam T Integral operand and result type.
 * @param out Destination value.
 * @param a Dividend.
 * @param b Divisor.
 */
template <typename T>
__device__ void floor_div(T* out, T const* a, T const* b)
  requires(cuda::std::is_integral_v<T>)
{
  *out = cudf::detail::integral_floor_div(*a, *b);
}

/**
 * @brief Computes floor division for float operands.
 *
 * @param out Destination value.
 * @param a Dividend.
 * @param b Divisor.
 */
__device__ inline void floor_div(float* out, float const* a, float const* b)
{
  *out = ::floorf(*a / *b);
}

/**
 * @brief Computes floor division for double operands.
 *
 * @param out Destination value.
 * @param a Dividend.
 * @param b Divisor.
 */
__device__ inline void floor_div(double* out, double const* a, double const* b)
{
  *out = ::floor(*a / *b);
}

/**
 * @brief Computes floor division for optional operands.
 *
 * @tparam T Operand and result type.
 * @param out Destination optional value.
 * @param a Optional dividend.
 * @param b Optional divisor.
 */
template <typename T>
__device__ void floor_div(cuda::std::optional<T>* out,
                          cuda::std::optional<T> const* a,
                          cuda::std::optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    T r;
    floor_div(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = cuda::std::nullopt;
  }
}

/**
 * @brief Computes remainder of integer division.
 *
 * @tparam T Operand and result type.
 * @param out Destination value.
 * @param a Dividend.
 * @param b Divisor.
 */
template <typename T>
__device__ void mod(T* out, T const* a, T const* b)
{
  *out = (*a % *b);
}

/**
 * @brief Computes floating-point remainder for float operands.
 *
 * @param out Destination value.
 * @param a Dividend.
 * @param b Divisor.
 */
__device__ inline void mod(float* out, float const* a, float const* b) { *out = ::fmodf(*a, *b); }

/**
 * @brief Computes floating-point remainder for double operands.
 *
 * @param out Destination value.
 * @param a Dividend.
 * @param b Divisor.
 */
__device__ inline void mod(double* out, double const* a, double const* b) { *out = ::fmod(*a, *b); }

/**
 * @brief Computes remainder for optional operands.
 *
 * @tparam T Operand and result type.
 * @param out Destination optional value.
 * @param a Optional dividend.
 * @param b Optional divisor.
 */
template <typename T>
__device__ void mod(cuda::std::optional<T>* out,
                    cuda::std::optional<T> const* a,
                    cuda::std::optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    T r;
    mod(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = cuda::std::nullopt;
  }
}

/**
 * @brief Computes Python-style modulus.
 *
 * @tparam T Integral operand and result type.
 * @param out Destination value.
 * @param a Dividend.
 * @param b Divisor.
 */
template <typename T>
__device__ void pymod(T* out, T const* a, T const* b)
{
  *out = (*a % *b + *b) % *b;
}

/**
 * @brief Computes Python-style modulus for float operands.
 *
 * @param out Destination value.
 * @param a Dividend.
 * @param b Divisor.
 */
__device__ inline void pymod(float* out, float const* a, float const* b)
{
  *out = ::fmodf(::fmodf(*a, *b) + *b, *b);
}

/**
 * @brief Computes Python-style modulus for double operands.
 *
 * @param out Destination value.
 * @param a Dividend.
 * @param b Divisor.
 */
__device__ inline void pymod(double* out, double const* a, double const* b)
{
  *out = ::fmod(::fmod(*a, *b) + *b, *b);
}

/**
 * @brief Computes Python-style modulus for optional operands.
 *
 * @tparam T Operand and result type.
 * @param out Destination optional value.
 * @param a Optional dividend.
 * @param b Optional divisor.
 */
template <typename T>
__device__ void pymod(cuda::std::optional<T>* out,
                      cuda::std::optional<T> const* a,
                      cuda::std::optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    T r;
    pymod(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = cuda::std::nullopt;
  }
}

/**
 * @brief Computes product of two values.
 *
 * @tparam T Operand and result type.
 * @param out Destination value.
 * @param a Left operand.
 * @param b Right operand.
 */
template <typename T>
__device__ void mul(T* out, T const* a, T const* b)
{
  *out = (*a * *b);
}

/**
 * @brief Computes product for optional operands.
 *
 * @tparam T Operand and result type.
 * @param out Destination optional value.
 * @param a Left optional operand.
 * @param b Right optional operand.
 */
template <typename T>
__device__ void mul(cuda::std::optional<T>* out,
                    cuda::std::optional<T> const* a,
                    cuda::std::optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    T r;
    mul(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = cuda::std::nullopt;
  }
}

/**
 * @brief Computes unary negation for signed inputs.
 *
 * @tparam T Signed input and output type.
 * @param out Destination value.
 * @param a Input value.
 */
template <typename T>
__device__ void neg(T* out, T const* a)
  requires(cuda::std::is_signed_v<T>)
{
  *out = -(*a);
}

/**
 * @brief Computes unary negation for decimal inputs.
 *
 * @tparam Rep Decimal representation type.
 * @param out Destination decimal value.
 * @param a Input decimal value.
 */
template <typename Rep>
__device__ void neg(numeric::decimal<Rep>* out, numeric::decimal<Rep> const* a)
{
  auto rep = -a->value();
  *out     = numeric::decimal<Rep>{numeric::scaled_integer<Rep>{rep, a->scale()}};
}

/**
 * @brief Computes unary negation for optional input.
 *
 * @tparam T Signed input and output type.
 * @param out Destination optional value.
 * @param a Optional input value.
 */
template <typename T>
__device__ void neg(cuda::std::optional<T>* out, cuda::std::optional<T> const* a)
{
  if (a->has_value()) {
    T r;
    neg(&r, &a->value());
    *out = r;
  } else {
    *out = cuda::std::nullopt;
  }
}

/**
 * @brief Computes subtraction of two values.
 *
 * @tparam T Operand and result type.
 * @param out Destination value.
 * @param a Minuend.
 * @param b Subtrahend.
 */
template <typename T>
__device__ void sub(T* out, T const* a, T const* b)
{
  *out = *a - *b;
}

/**
 * @brief Computes subtraction for optional operands.
 *
 * @tparam T Operand and result type.
 * @param out Destination optional value.
 * @param a Optional minuend.
 * @param b Optional subtrahend.
 */
template <typename T>
__device__ void sub(cuda::std::optional<T>* out,
                    cuda::std::optional<T> const* a,
                    cuda::std::optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    T r;
    sub(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = cuda::std::nullopt;
  }
}

/**
 * @brief Computes true division and returns a double.
 *
 * @tparam T Input operand type.
 * @param out Destination double value.
 * @param a Dividend.
 * @param b Divisor.
 */
template <typename T>
__device__ void true_div(double* out, T const* a, T const* b)
  requires(cuda::std::is_floating_point_v<T> || cuda::std::is_integral_v<T>)
{
  *out = static_cast<double>(*a) / static_cast<double>(*b);
}

/**
 * @brief Computes true division for optional operands and returns optional double.
 *
 * @tparam T Input operand type.
 * @param out Destination optional double value.
 * @param a Optional dividend.
 * @param b Optional divisor.
 */
template <typename T>
__device__ void true_div(cuda::std::optional<double>* out,
                         cuda::std::optional<T> const* a,
                         cuda::std::optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    double r;
    true_div(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = cuda::std::nullopt;
  }
}

}  // namespace ops
}  // namespace CUDF_EXPORT cudf
