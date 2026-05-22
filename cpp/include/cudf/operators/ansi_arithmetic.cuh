/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/operators/error.hpp>
#include <cudf/operators/math.cuh>
#include <cudf/utilities/export.hpp>

#include <cuda/numeric>
#include <cuda/std/limits>
#include <cuda/std/optional>
#include <cuda/std/type_traits>

namespace CUDF_EXPORT cudf {
namespace ops {

/**
 * @brief Adds integral operands with overflow detection.
 *
 * @tparam T Integral type.
 * @param out Destination value.
 * @param a Left operand.
 * @param b Right operand.
 * @return errc::OVERFLOW on overflow, else errc::SUCCESS.
 */
template <typename T>
__device__ errc ansi_add(T* out, T const* a, T const* b)
  requires(cuda::std::is_integral_v<T>)
{
  T r;
  if (cuda::add_overflow(r, *a, *b)) { return errc::OVERFLOW; }
  *out = r;
  return errc::SUCCESS;
}

/**
 * @brief Adds floating-point operands.
 *
 * @tparam T Floating-point type.
 * @return errc::SUCCESS.
 */
template <typename T>
__device__ errc ansi_add(T* out, T const* a, T const* b)
  requires(cuda::std::is_floating_point_v<T>)
{
  *out = *a + *b;
  return errc::SUCCESS;
}

/**
 * @brief Adds fixed-point decimal operands with overflow detection.
 *
 * @tparam R Decimal representation type.
 * @return errc::OVERFLOW on overflow, else errc::SUCCESS.
 */
template <typename R>
__device__ errc ansi_add(numeric::decimal<R>* out,
                         numeric::decimal<R> const* a,
                         numeric::decimal<R> const* b)
{
  auto scale = cuda::std::min(a->scale(), b->scale());

  if (numeric::addition_overflow<R>(a->rescaled(scale).value(), b->rescaled(scale).value())) {
    return errc::OVERFLOW;
  }

  *out = numeric::decimal<R>{numeric::scaled_integer<R>{
    a->rescaled(scale).value() + b->rescaled(scale).value(), numeric::scale_type{scale}}};
  return errc::SUCCESS;
}

/**
 * @brief Adds optional operands with ANSI overflow behavior.
 *
 * @tparam T Operand and result type.
 * @return Operation status from underlying add, or errc::SUCCESS.
 */
template <typename T>
__device__ errc ansi_add(cuda::std::optional<T>* out,
                         cuda::std::optional<T> const* a,
                         cuda::std::optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    T r;
    if (errc e = ansi_add(&r, &a->value(), &b->value()); e != errc::SUCCESS) {
      *out = cuda::std::nullopt;
      return e;
    }
    *out = r;
  } else {
    *out = cuda::std::nullopt;
  }

  return errc::SUCCESS;
}

/**
 * @brief Subtracts integral operands with overflow detection.
 *
 * @tparam T Integral type.
 * @param out Destination value.
 * @param a Minuend.
 * @param b Subtrahend.
 * @return errc::OVERFLOW on overflow, else errc::SUCCESS.
 */
template <typename T>
__device__ errc ansi_sub(T* out, T const* a, T const* b)
  requires(cuda::std::is_integral_v<T>)
{
  T r;
  if (cuda::sub_overflow(r, *a, *b)) { return errc::OVERFLOW; }
  *out = r;
  return errc::SUCCESS;
}

/**
 * @brief Subtracts floating-point operands.
 *
 * @tparam T Floating-point type.
 * @return errc::SUCCESS.
 */
template <typename T>
__device__ errc ansi_sub(T* out, T const* a, T const* b)
  requires(cuda::std::is_floating_point_v<T>)
{
  *out = *a - *b;
  return errc::SUCCESS;
}

/**
 * @brief Subtracts fixed-point decimal operands with overflow detection.
 *
 * @tparam R Decimal representation type.
 * @return errc::OVERFLOW on overflow, else errc::SUCCESS.
 */
template <typename R>
__device__ errc ansi_sub(numeric::decimal<R>* out,
                         numeric::decimal<R> const* a,
                         numeric::decimal<R> const* b)
{
  auto scale = cuda::std::min(a->scale(), b->scale());

  if (numeric::subtraction_overflow<R>(a->rescaled(scale).value(), b->rescaled(scale).value())) {
    return errc::OVERFLOW;
  }

  *out = numeric::decimal<R>{numeric::scaled_integer<R>{
    a->rescaled(scale).value() - b->rescaled(scale).value(), numeric::scale_type{scale}}};
  return errc::SUCCESS;
}

/**
 * @brief Subtracts optional operands with ANSI overflow behavior.
 *
 * @tparam T Operand and result type.
 * @return Operation status from underlying subtract, or errc::SUCCESS.
 */
template <typename T>
__device__ errc ansi_sub(cuda::std::optional<T>* out,
                         cuda::std::optional<T> const* a,
                         cuda::std::optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    T r;
    if (errc e = ansi_sub(&r, &a->value(), &b->value()); e != errc::SUCCESS) {
      *out = cuda::std::nullopt;
      return e;
    }
    *out = r;
  } else {
    *out = cuda::std::nullopt;
  }
  return errc::SUCCESS;
}

/**
 * @brief Multiplies integral operands with overflow detection.
 *
 * @tparam T Integral type.
 * @param out Destination value.
 * @param a Left operand.
 * @param b Right operand.
 * @return errc::OVERFLOW on overflow, else errc::SUCCESS.
 */
template <typename T>
__device__ errc ansi_mul(T* out, T const* a, T const* b)
  requires(cuda::std::is_integral_v<T>)
{
  T r;
  if (cuda::mul_overflow(r, *a, *b)) { return errc::OVERFLOW; }
  *out = r;
  return errc::SUCCESS;
}

/**
 * @brief Multiplies floating-point operands.
 *
 * @tparam T Floating-point type.
 * @return errc::SUCCESS.
 */
template <typename T>
__device__ errc ansi_mul(T* out, T const* a, T const* b)
  requires(cuda::std::is_floating_point_v<T>)
{
  *out = *a * *b;
  return errc::SUCCESS;
}

/**
 * @brief Multiplies fixed-point decimal operands with overflow detection.
 *
 * @tparam R Decimal representation type.
 * @return errc::OVERFLOW on overflow, else errc::SUCCESS.
 */
template <typename R>
__device__ errc ansi_mul(numeric::decimal<R>* out,
                         numeric::decimal<R> const* a,
                         numeric::decimal<R> const* b)
{
  if (numeric::multiplication_overflow<R>(a->value(), b->value())) { return errc::OVERFLOW; }

  *out = numeric::decimal<R>{numeric::scaled_integer<R>{
    a->value() * b->value(), numeric::scale_type{a->scale() + b->scale()}}};
  return errc::SUCCESS;
}

/**
 * @brief Multiplies optional operands with ANSI overflow behavior.
 *
 * @tparam T Operand and result type.
 * @return Operation status from underlying multiply, or errc::SUCCESS.
 */
template <typename T>
__device__ errc ansi_mul(cuda::std::optional<T>* out,
                         cuda::std::optional<T> const* a,
                         cuda::std::optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    T r;
    if (errc e = ansi_mul(&r, &a->value(), &b->value()); e != errc::SUCCESS) {
      *out = cuda::std::nullopt;
      return e;
    }
    *out = r;
  } else {
    *out = cuda::std::nullopt;
  }
  return errc::SUCCESS;
}

/**
 * @brief Divides integral operands with divide-by-zero checks.
 *
 * @tparam T Integral type.
 * @param out Destination value.
 * @param a Dividend.
 * @param b Divisor.
 * @return errc::DIVISION_BY_ZERO on zero divisor, errc::OVERFLOW on overflow, else errc::SUCCESS.
 */
template <typename T>
__device__ errc ansi_div(T* out, T const* a, T const* b)
  requires(cuda::std::is_integral_v<T>)
{
  if (*b == 0) { return errc::DIVISION_BY_ZERO; }
  T r;
  if (cuda::div_overflow(r, *a, *b)) { return errc::OVERFLOW; }
  *out = r;
  return errc::SUCCESS;
}

/**
 * @brief Divides floating-point operands.
 *
 * @tparam T Floating-point type.
 * @return errc::SUCCESS.
 */
template <typename T>
__device__ errc ansi_div(T* out, T const* a, T const* b)
  requires(cuda::std::is_floating_point_v<T>)
{
  *out = *a / *b;
  return errc::SUCCESS;
}

/**
 * @brief Divides fixed-point decimal operands with ANSI checks.
 *
 * @tparam R Decimal representation type.
 * @return errc::OVERFLOW on overflow or zero divisor, else errc::SUCCESS.
 */
template <typename R>
__device__ errc ansi_div(numeric::decimal<R>* out,
                         numeric::decimal<R> const* a,
                         numeric::decimal<R> const* b)
{
  if (b->value() == 0) { return errc::DIVISION_BY_ZERO; }

  if (numeric::division_overflow<R>(a->value(), b->value())) { return errc::OVERFLOW; }

  *out = numeric::decimal<R>{numeric::scaled_integer<R>{
    a->value() / b->value(), numeric::scale_type{a->scale() - b->scale()}}};
  return errc::SUCCESS;
}

/**
 * @brief Divides optional operands with ANSI overflow behavior.
 *
 * @tparam T Operand and result type.
 * @return Operation status from underlying divide, or errc::SUCCESS.
 */
template <typename T>
__device__ errc ansi_div(cuda::std::optional<T>* out,
                         cuda::std::optional<T> const* a,
                         cuda::std::optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    T r;
    if (errc e = ansi_div(&r, &a->value(), &b->value()); e != errc::SUCCESS) {
      *out = cuda::std::nullopt;
      return e;
    }
    *out = r;
  } else {
    *out = cuda::std::nullopt;
  }
  return errc::SUCCESS;
}

/**
 * @brief Computes signed integral modulus with ANSI checks.
 *
 * @tparam T Signed integral type.
 * @param out Destination value.
 * @param a Dividend.
 * @param b Divisor.
 * @return errc::DIVISION_BY_ZERO on zero divisor, else errc::SUCCESS.
 */
template <typename T>
__device__ errc ansi_mod(T* out, T const* a, T const* b)
  requires(cuda::std::is_integral_v<T> && cuda::std::is_signed_v<T>)
{
  if (*b == 0) { return errc::DIVISION_BY_ZERO; }

  // avoid signed overflow UB / trap for minimum value divided by -1.
  if (*a == cuda::std::numeric_limits<T>::min() && *b == T{-1}) {
    *out = T{0};
    return errc::SUCCESS;
  }

  *out = *a % *b;
  return errc::SUCCESS;
}

/**
 * @brief Computes unsigned integral modulus with ANSI checks.
 *
 * @tparam T Unsigned integral type.
 * @return errc::DIVISION_BY_ZERO on zero divisor, else errc::SUCCESS.
 */
template <typename T>
__device__ errc ansi_mod(T* out, T const* a, T const* b)
  requires(cuda::std::is_integral_v<T> && cuda::std::is_unsigned_v<T>)
{
  if (*b == 0) { return errc::DIVISION_BY_ZERO; }
  *out = *a % *b;
  return errc::SUCCESS;
}

/**
 * @brief Computes floating-point modulus for float operands with ANSI checks.
 *
 * @return errc::DIVISION_BY_ZERO on zero divisor, else errc::SUCCESS.
 */
__device__ errc ansi_mod(float* out, float const* a, float const* b)
{
  if (*b == 0) { return errc::DIVISION_BY_ZERO; }
  *out = (*a) - (*b) * ::floorf((*a) / (*b));
  return errc::SUCCESS;
}

/**
 * @brief Computes floating-point modulus for double operands with ANSI checks.
 *
 * @return errc::DIVISION_BY_ZERO on zero divisor, else errc::SUCCESS.
 */
__device__ errc ansi_mod(double* out, double const* a, double const* b)
{
  if (*b == 0) { return errc::DIVISION_BY_ZERO; }
  *out = (*a) - (*b) * ::floor((*a) / (*b));
  return errc::SUCCESS;
}

/**
 * @brief Computes fixed-point decimal modulus with ANSI checks.
 *
 * @tparam R Decimal representation type.
 * @return errc::DIVISION_BY_ZERO or propagated status, else errc::SUCCESS.
 */
template <typename R>
__device__ errc ansi_mod(numeric::decimal<R>* out,
                         numeric::decimal<R> const* a,
                         numeric::decimal<R> const* b)
{
  numeric::decimal<R> div;

  if (errc e = ansi_div(&div, a, b); e != errc::SUCCESS) { return e; }

  numeric::decimal<R> quotient;
  floor(&quotient, &div);
  *out = *a - *b * quotient;
  return errc::SUCCESS;
}

/**
 * @brief Computes modulus for optional operands with ANSI behavior.
 *
 * @tparam T Operand and result type.
 * @return Operation status from underlying modulus, or errc::SUCCESS.
 */
template <typename T>
__device__ errc ansi_mod(cuda::std::optional<T>* out,
                         cuda::std::optional<T> const* a,
                         cuda::std::optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    T r;
    if (errc e = ansi_mod(&r, &a->value(), &b->value()); e != errc::SUCCESS) {
      *out = cuda::std::nullopt;
      return e;
    }
    *out = r;
  } else {
    *out = cuda::std::nullopt;
  }
  return errc::SUCCESS;
}

/**
 * @brief Computes absolute value for signed integral inputs with ANSI overflow checks.
 *
 * @tparam T Signed integral type.
 * @param out Destination value.
 * @param a Input value.
 * @return errc::OVERFLOW on minimum representable input, else errc::SUCCESS.
 */
template <typename T>
__device__ errc ansi_abs(T* out, T const* a)
  requires(cuda::std::is_integral_v<T> && cuda::std::is_signed_v<T>)
{
  if (*a == cuda::std::numeric_limits<T>::min()) { return errc::OVERFLOW; }
  *out = (*a < 0) ? -(*a) : *a;
  return errc::SUCCESS;
}

/**
 * @brief Returns unsigned input unchanged for ANSI absolute value.
 *
 * @tparam T Unsigned integral type.
 * @return errc::SUCCESS.
 */
template <typename T>
__device__ errc ansi_abs(T* out, T const* a)
  requires(cuda::std::is_integral_v<T> && cuda::std::is_unsigned_v<T>)
{
  *out = *a;
  return errc::SUCCESS;
}

/**
 * @brief Computes absolute value for floating-point inputs.
 *
 * @tparam T Floating-point type.
 * @return errc::SUCCESS.
 */
template <typename T>
__device__ errc ansi_abs(T* out, T const* a)
  requires(cuda::std::is_floating_point_v<T>)
{
  *out = (*a < 0) ? -(*a) : *a;
  return errc::SUCCESS;
}

/**
 * @brief Computes absolute value for decimal inputs with ANSI overflow checks.
 *
 * @tparam R Decimal representation type.
 * @return errc::OVERFLOW on minimum representable input, else errc::SUCCESS.
 */
template <typename R>
__device__ errc ansi_abs(numeric::decimal<R>* out, numeric::decimal<R> const* a)
{
  if (a->value() == cuda::std::numeric_limits<R>::min()) { return errc::OVERFLOW; }
  auto rep = a->value() < 0 ? -a->value() : a->value();
  *out     = numeric::decimal<R>{numeric::scaled_integer<R>{rep, numeric::scale_type{a->scale()}}};
  return errc::SUCCESS;
}

/**
 * @brief Computes absolute value for optional inputs with ANSI overflow checks.
 *
 * @tparam T Value type.
 * @return Operation status from underlying abs, or errc::SUCCESS.
 */
template <typename T>
__device__ errc ansi_abs(cuda::std::optional<T>* out, cuda::std::optional<T> const* a)
{
  if (a->has_value()) {
    T r;
    if (errc e = ansi_abs(&r, &a->value()); e != errc::SUCCESS) {
      *out = cuda::std::nullopt;
      return e;
    }
    *out = r;
  } else {
    *out = cuda::std::nullopt;
  }
  return errc::SUCCESS;
}

/**
 * @brief Computes unary negation for signed inputs with ANSI overflow checks.
 *
 * @tparam T Signed type.
 * @param out Destination value.
 * @param a Input value.
 * @return errc::OVERFLOW on minimum representable input, else errc::SUCCESS.
 */
template <typename T>
__device__ errc ansi_neg(T* out, T const* a)
  requires(cuda::std::is_signed_v<T>)
{
  if (*a == cuda::std::numeric_limits<T>::min()) { return errc::OVERFLOW; }
  *out = -(*a);
  return errc::SUCCESS;
}

/**
 * @brief Computes unary negation for decimal inputs with ANSI overflow checks.
 *
 * @tparam R Decimal representation type.
 * @return errc::OVERFLOW on minimum representable input, else errc::SUCCESS.
 */
template <typename R>
__device__ errc ansi_neg(numeric::decimal<R>* out, numeric::decimal<R> const* a)
{
  if (a->value() == cuda::std::numeric_limits<R>::min()) { return errc::OVERFLOW; }
  auto rep = -a->value();
  *out     = numeric::decimal<R>{numeric::scaled_integer<R>{rep, numeric::scale_type{a->scale()}}};
  return errc::SUCCESS;
}

/**
 * @brief Computes unary negation for optional inputs with ANSI checks.
 *
 * @tparam T Signed type.
 * @return Operation status from underlying negate, or errc::SUCCESS.
 */
template <typename T>
__device__ errc ansi_neg(cuda::std::optional<T>* out, cuda::std::optional<T> const* a)
{
  if (a->has_value()) {
    T r;
    if (errc e = ansi_neg(&r, &a->value()); e != errc::SUCCESS) {
      *out = cuda::std::nullopt;
      return e;
    }
    *out = r;
  } else {
    *out = cuda::std::nullopt;
  }
  return errc::SUCCESS;
}

/**
 * @brief Validates decimal precision against a target precision value.
 *
 * @tparam R Decimal representation type.
 * @param out Destination decimal value.
 * @param a Input decimal value.
 * @param precision Maximum allowed precision.
 * @return errc::OVERFLOW when precision is invalid or exceeded, else errc::SUCCESS.
 */
template <typename R>
__device__ errc ansi_precision_check(numeric::decimal<R>* out,
                                     numeric::decimal<R> const* a,
                                     int32_t const* precision)
{
  if (*precision <= 0) { return errc::OVERFLOW; }

  auto value = a->value();
  if (value == cuda::std::numeric_limits<R>::min()) { return errc::OVERFLOW; }

  auto abs_value = value < 0 ? -value : value;

  if (abs_value >= numeric::detail::ipow<R, numeric::Radix::BASE_10>(static_cast<R>(*precision))) {
    return errc::OVERFLOW;
  }

  *out = *a;
  return errc::SUCCESS;
}

/**
 * @brief Validates optional decimal precision against a precision value.
 *
 * @tparam T Decimal value type.
 * @param out Destination optional value.
 * @param a Optional decimal input.
 * @param precision Precision.
 * @return Operation status from underlying precision check, or errc::SUCCESS.
 */
template <typename T>
__device__ errc ansi_precision_check(cuda::std::optional<T>* out,
                                     cuda::std::optional<T> const* a,
                                     cuda::std::optional<int32_t> const* precision)
{
  if (a->has_value() && precision->has_value()) {
    T r;
    if (errc e = ansi_precision_check(&r, &a->value(), &precision->value()); e != errc::SUCCESS) {
      *out = cuda::std::nullopt;
      return e;
    } else {
      *out = r;
      return errc::SUCCESS;
    }
  } else {
    *out = cuda::std::nullopt;
    return errc::SUCCESS;
  }
}

/**
 * @brief ANSI add that returns null instead of propagating arithmetic errors.
 *
 * @tparam T Operand and result type.
 * @param out Destination optional value.
 * @param a Left optional operand.
 * @param b Right optional operand.
 */
template <typename T>
__device__ void ansi_try_add(cuda::std::optional<T>* out,
                             cuda::std::optional<T> const* a,
                             cuda::std::optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    T r;
    if (errc e = ansi_add(&r, &a->value(), &b->value()); e != errc::SUCCESS) {
      *out = cuda::std::nullopt;
    } else {
      *out = r;
    }
  } else {
    *out = cuda::std::nullopt;
  }
}

/**
 * @brief ANSI subtract that returns null instead of propagating arithmetic errors.
 *
 * @tparam T Operand and result type.
 * @param out Destination optional value.
 * @param a Optional minuend.
 * @param b Optional subtrahend.
 */
template <typename T>
__device__ void ansi_try_sub(cuda::std::optional<T>* out,
                             cuda::std::optional<T> const* a,
                             cuda::std::optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    T r;
    if (errc e = ansi_sub(&r, &a->value(), &b->value()); e != errc::SUCCESS) {
      *out = cuda::std::nullopt;
    } else {
      *out = r;
    }
  } else {
    *out = cuda::std::nullopt;
  }
}

/**
 * @brief ANSI multiply that returns null instead of propagating arithmetic errors.
 *
 * @tparam T Operand and result type.
 * @param out Destination optional value.
 * @param a Left optional operand.
 * @param b Right optional operand.
 */
template <typename T>
__device__ void ansi_try_mul(cuda::std::optional<T>* out,
                             cuda::std::optional<T> const* a,
                             cuda::std::optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    T r;
    if (errc e = ansi_mul(&r, &a->value(), &b->value()); e != errc::SUCCESS) {
      *out = cuda::std::nullopt;
    } else {
      *out = r;
    }
  } else {
    *out = cuda::std::nullopt;
  }
}

/**
 * @brief ANSI divide that returns null instead of propagating arithmetic errors.
 *
 * @tparam T Operand and result type.
 * @param out Destination optional value.
 * @param a Optional dividend.
 * @param b Optional divisor.
 */
template <typename T>
__device__ void ansi_try_div(cuda::std::optional<T>* out,
                             cuda::std::optional<T> const* a,
                             cuda::std::optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    T r;
    if (errc e = ansi_div(&r, &a->value(), &b->value()); e != errc::SUCCESS) {
      *out = cuda::std::nullopt;
    } else {
      *out = r;
    }
  } else {
    *out = cuda::std::nullopt;
  }
}

/**
 * @brief ANSI modulus that returns null instead of propagating arithmetic errors.
 *
 * @tparam T Operand and result type.
 * @param out Destination optional value.
 * @param a Optional dividend.
 * @param b Optional divisor.
 */
template <typename T>
__device__ void ansi_try_mod(cuda::std::optional<T>* out,
                             cuda::std::optional<T> const* a,
                             cuda::std::optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    T r;
    if (errc e = ansi_mod(&r, &a->value(), &b->value()); e != errc::SUCCESS) {
      *out = cuda::std::nullopt;
    } else {
      *out = r;
    }
  } else {
    *out = cuda::std::nullopt;
  }
}

/**
 * @brief ANSI absolute value that returns null instead of propagating overflow errors.
 *
 * @tparam T Operand and result type.
 * @param out Destination optional value.
 * @param a Optional input value.
 */
template <typename T>
__device__ void ansi_try_abs(cuda::std::optional<T>* out, cuda::std::optional<T> const* a)
{
  if (a->has_value()) {
    T r;
    if (errc e = ansi_abs(&r, &a->value()); e != errc::SUCCESS) {
      *out = cuda::std::nullopt;
    } else {
      *out = r;
    }
  } else {
    *out = cuda::std::nullopt;
  }
}

/**
 * @brief ANSI unary negation that returns null instead of propagating overflow errors.
 *
 * @tparam T Operand and result type.
 * @param out Destination optional value.
 * @param a Optional input value.
 */
template <typename T>
__device__ void ansi_try_neg(cuda::std::optional<T>* out, cuda::std::optional<T> const* a)
{
  if (a->has_value()) {
    T r;
    if (errc e = ansi_neg(&r, &a->value()); e != errc::SUCCESS) {
      *out = cuda::std::nullopt;
    } else {
      *out = r;
    }
  } else {
    *out = cuda::std::nullopt;
  }
}

/**
 * @brief ANSI precision check that returns null instead of propagating precision errors.
 *
 * @tparam R Decimal representation type.
 * @param out Destination optional decimal value.
 * @param a Optional decimal input.
 * @param precision Optional precision.
 */
template <typename R>
__device__ void ansi_try_precision_check(cuda::std::optional<numeric::decimal<R>>* out,
                                         cuda::std::optional<numeric::decimal<R>> const* a,
                                         cuda::std::optional<int32_t> const* precision)
{
  if (a->has_value() && precision->has_value()) {
    numeric::decimal<R> r;
    if (errc e = ansi_precision_check(&r, &a->value(), &precision->value()); e != errc::SUCCESS) {
      *out = cuda::std::nullopt;
    } else {
      *out = r;
    }
  } else {
    *out = cuda::std::nullopt;
  }
}

}  // namespace ops
}  // namespace CUDF_EXPORT cudf
