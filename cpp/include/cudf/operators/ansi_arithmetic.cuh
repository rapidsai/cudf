/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/operators/math.cuh>
#include <cudf/operators/types.cuh>

#include <cuda/numeric>

namespace CUDF_EXPORT cudf {
namespace ops {

/**
 * @brief Adds integral operands with overflow detection.
 *
 * @tparam T Integral type.
 * @param out Destination value.
 * @param a Left operand.
 * @param b Right operand.
 * @return errc::OVERFLOW on overflow, else errc::OK.
 */
template <typename T>
__device__ inline errc ansi_add(T* out, T const* a, T const* b)
  requires(cuda::std::is_integral_v<T>)
{
  T r;
  if (cuda::add_overflow(r, *a, *b)) { return errc::OVERFLOW; }
  *out = r;
  return errc::OK;
}

/**
 * @brief Adds floating-point operands.
 *
 * @tparam T Floating-point type.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc ansi_add(T* out, T const* a, T const* b)
  requires(cuda::std::is_floating_point_v<T>)
{
  *out = *a + *b;
  return errc::OK;
}

/**
 * @brief Adds fixed-point decimal operands with overflow detection.
 *
 * @tparam R Decimal representation type.
 * @return errc::OVERFLOW on overflow, else errc::OK.
 */
template <typename R>
__device__ inline errc ansi_add(decimal<R>* out, decimal<R> const* a, decimal<R> const* b)
{
  auto scale = cuda::std::min(a->scale(), b->scale());

  if (numeric::addition_overflow<R>(a->rescaled(scale).value(), b->rescaled(scale).value())) {
    return errc::OVERFLOW;
  }

  *out = decimal<R>{numeric::scaled_integer<R>{
    a->rescaled(scale).value() + b->rescaled(scale).value(), numeric::scale_type{scale}}};
  return errc::OK;
}

/**
 * @brief Adds optional operands with ANSI overflow behavior.
 *
 * @tparam T Operand and result type.
 * @return Operation status from underlying add, or errc::OK.
 */
template <typename T>
__device__ inline errc ansi_add(optional<T>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    T r;
    if (errc e = ansi_add(&r, &a->value(), &b->value()); e != errc::OK) {
      *out = nullopt;
      return e;
    }
    *out = r;
  } else {
    *out = nullopt;
  }

  return errc::OK;
}

/**
 * @brief Subtracts integral operands with overflow detection.
 *
 * @tparam T Integral type.
 * @param out Destination value.
 * @param a Minuend.
 * @param b Subtrahend.
 * @return errc::OVERFLOW on overflow, else errc::OK.
 */
template <typename T>
__device__ inline errc ansi_sub(T* out, T const* a, T const* b)
  requires(cuda::std::is_integral_v<T>)
{
  T r;
  if (cuda::sub_overflow(r, *a, *b)) { return errc::OVERFLOW; }
  *out = r;
  return errc::OK;
}

/**
 * @brief Subtracts floating-point operands.
 *
 * @tparam T Floating-point type.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc ansi_sub(T* out, T const* a, T const* b)
  requires(cuda::std::is_floating_point_v<T>)
{
  *out = *a - *b;
  return errc::OK;
}

/**
 * @brief Subtracts fixed-point decimal operands with overflow detection.
 *
 * @tparam R Decimal representation type.
 * @return errc::OVERFLOW on overflow, else errc::OK.
 */
template <typename R>
__device__ inline errc ansi_sub(decimal<R>* out, decimal<R> const* a, decimal<R> const* b)
{
  auto scale = cuda::std::min(a->scale(), b->scale());

  if (numeric::subtraction_overflow<R>(a->rescaled(scale).value(), b->rescaled(scale).value())) {
    return errc::OVERFLOW;
  }

  *out = decimal<R>{numeric::scaled_integer<R>{
    a->rescaled(scale).value() - b->rescaled(scale).value(), numeric::scale_type{scale}}};
  return errc::OK;
}

/**
 * @brief Subtracts optional operands with ANSI overflow behavior.
 *
 * @tparam T Operand and result type.
 * @return Operation status from underlying subtract, or errc::OK.
 */
template <typename T>
__device__ inline errc ansi_sub(optional<T>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    T r;
    if (errc e = ansi_sub(&r, &a->value(), &b->value()); e != errc::OK) {
      *out = nullopt;
      return e;
    }
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

/**
 * @brief Multiplies integral operands with overflow detection.
 *
 * @tparam T Integral type.
 * @param out Destination value.
 * @param a Left operand.
 * @param b Right operand.
 * @return errc::OVERFLOW on overflow, else errc::OK.
 */
template <typename T>
__device__ inline errc ansi_mul(T* out, T const* a, T const* b)
  requires(cuda::std::is_integral_v<T>)
{
  T r;
  if (cuda::mul_overflow(r, *a, *b)) { return errc::OVERFLOW; }
  *out = r;
  return errc::OK;
}

/**
 * @brief Multiplies floating-point operands.
 *
 * @tparam T Floating-point type.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc ansi_mul(T* out, T const* a, T const* b)
  requires(cuda::std::is_floating_point_v<T>)
{
  *out = *a * *b;
  return errc::OK;
}

/**
 * @brief Multiplies fixed-point decimal operands with overflow detection.
 *
 * @tparam R Decimal representation type.
 * @return errc::OVERFLOW on overflow, else errc::OK.
 */
template <typename R>
__device__ inline errc ansi_mul(decimal<R>* out, decimal<R> const* a, decimal<R> const* b)
{
  if (numeric::multiplication_overflow<R>(a->value(), b->value())) { return errc::OVERFLOW; }

  *out = decimal<R>{numeric::scaled_integer<R>{a->value() * b->value(),
                                               numeric::scale_type{a->scale() + b->scale()}}};
  return errc::OK;
}

/**
 * @brief Multiplies optional operands with ANSI overflow behavior.
 *
 * @tparam T Operand and result type.
 * @return Operation status from underlying multiply, or errc::OK.
 */
template <typename T>
__device__ inline errc ansi_mul(optional<T>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    T r;
    if (errc e = ansi_mul(&r, &a->value(), &b->value()); e != errc::OK) {
      *out = nullopt;
      return e;
    }
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

/**
 * @brief Divides integral operands with divide-by-zero checks.
 *
 * @tparam T Integral type.
 * @param out Destination value.
 * @param a Dividend.
 * @param b Divisor.
 * @return errc::DIVISION_BY_ZERO on zero divisor, errc::OVERFLOW on overflow, else errc::OK.
 */
template <typename T>
__device__ inline errc ansi_div(T* out, T const* a, T const* b)
  requires(cuda::std::is_integral_v<T>)
{
  if (*b == 0) { return errc::DIVISION_BY_ZERO; }
  T r;
  if (cuda::div_overflow(r, *a, *b)) { return errc::OVERFLOW; }
  *out = r;
  return errc::OK;
}

/**
 * @brief Divides floating-point operands.
 *
 * @tparam T Floating-point type.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc ansi_div(T* out, T const* a, T const* b)
  requires(cuda::std::is_floating_point_v<T>)
{
  *out = *a / *b;
  return errc::OK;
}

/**
 * @brief Divides fixed-point decimal operands with ANSI checks.
 *
 * @tparam R Decimal representation type.
 * @return errc::OVERFLOW on overflow or zero divisor, else errc::OK.
 */
template <typename R>
__device__ inline errc ansi_div(decimal<R>* out, decimal<R> const* a, decimal<R> const* b)
{
  if (b->value() == 0) { return errc::DIVISION_BY_ZERO; }

  if (numeric::division_overflow<R>(a->value(), b->value())) { return errc::OVERFLOW; }

  *out = decimal<R>{numeric::scaled_integer<R>{a->value() / b->value(),
                                               numeric::scale_type{a->scale() - b->scale()}}};
  return errc::OK;
}

/**
 * @brief Divides optional operands with ANSI overflow behavior.
 *
 * @tparam T Operand and result type.
 * @return Operation status from underlying divide, or errc::OK.
 */
template <typename T>
__device__ inline errc ansi_div(optional<T>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    T r;
    if (errc e = ansi_div(&r, &a->value(), &b->value()); e != errc::OK) {
      *out = nullopt;
      return e;
    }
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

/**
 * @brief Computes signed integral modulus with ANSI checks.
 *
 * @tparam T Signed integral type.
 * @param out Destination value.
 * @param a Dividend.
 * @param b Divisor.
 * @return errc::DIVISION_BY_ZERO on zero divisor, else errc::OK.
 */
template <typename T>
__device__ inline errc ansi_mod(T* out, T const* a, T const* b)
  requires(cuda::std::is_integral_v<T> && cuda::std::is_signed_v<T>)
{
  if (*b == 0) { return errc::DIVISION_BY_ZERO; }

  // avoid signed overflow UB / trap for minimum value divided by -1.
  if (*a == cuda::std::numeric_limits<T>::min() && *b == T{-1}) {
    *out = T{0};
    return errc::OK;
  }

  *out = *a % *b;
  return errc::OK;
}

/**
 * @brief Computes unsigned integral modulus with ANSI checks.
 *
 * @tparam T Unsigned integral type.
 * @return errc::DIVISION_BY_ZERO on zero divisor, else errc::OK.
 */
template <typename T>
__device__ inline errc ansi_mod(T* out, T const* a, T const* b)
  requires(cuda::std::is_integral_v<T> && cuda::std::is_unsigned_v<T>)
{
  if (*b == 0) { return errc::DIVISION_BY_ZERO; }
  *out = *a % *b;
  return errc::OK;
}

/**
 * @brief Computes floating-point modulus for float operands with ANSI checks.
 *
 * @return errc::DIVISION_BY_ZERO on zero divisor, else errc::OK.
 */
__device__ inline errc ansi_mod(float* out, float const* a, float const* b)
{
  if (*b == 0) { return errc::DIVISION_BY_ZERO; }
  *out = (*a) - (*b) * ::floorf((*a) / (*b));
  return errc::OK;
}

/**
 * @brief Computes floating-point modulus for double operands with ANSI checks.
 *
 * @return errc::DIVISION_BY_ZERO on zero divisor, else errc::OK.
 */
__device__ inline errc ansi_mod(double* out, double const* a, double const* b)
{
  if (*b == 0) { return errc::DIVISION_BY_ZERO; }
  *out = (*a) - (*b) * ::floor((*a) / (*b));
  return errc::OK;
}

/**
 * @brief Computes fixed-point decimal modulus with ANSI checks.
 *
 * @tparam R Decimal representation type.
 * @return errc::DIVISION_BY_ZERO or propagated status, else errc::OK.
 */
template <typename R>
__device__ inline errc ansi_mod(decimal<R>* out, decimal<R> const* a, decimal<R> const* b)
{
  decimal<R> div;

  if (errc e = ansi_div(&div, a, b); e != errc::OK) { return e; }

  decimal<R> quotient;
  floor(&quotient, &div);
  *out = *a - *b * quotient;
  return errc::OK;
}

/**
 * @brief Computes modulus for optional operands with ANSI behavior.
 *
 * @tparam T Operand and result type.
 * @return Operation status from underlying modulus, or errc::OK.
 */
template <typename T>
__device__ inline errc ansi_mod(optional<T>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    T r;
    if (errc e = ansi_mod(&r, &a->value(), &b->value()); e != errc::OK) {
      *out = nullopt;
      return e;
    }
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

/**
 * @brief Computes absolute value for signed integral inputs with ANSI overflow checks.
 *
 * @tparam T Signed integral type.
 * @param out Destination value.
 * @param a Input value.
 * @return errc::OVERFLOW on minimum representable input, else errc::OK.
 */
template <typename T>
__device__ inline errc ansi_abs(T* out, T const* a)
  requires(cuda::std::is_integral_v<T> && cuda::std::is_signed_v<T>)
{
  if (*a == cuda::std::numeric_limits<T>::min()) { return errc::OVERFLOW; }
  *out = (*a < 0) ? -(*a) : *a;
  return errc::OK;
}

/**
 * @brief Returns unsigned input unchanged for ANSI absolute value.
 *
 * @tparam T Unsigned integral type.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc ansi_abs(T* out, T const* a)
  requires(cuda::std::is_integral_v<T> && cuda::std::is_unsigned_v<T>)
{
  *out = *a;
  return errc::OK;
}

/**
 * @brief Computes absolute value for floating-point inputs.
 *
 * @tparam T Floating-point type.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc ansi_abs(T* out, T const* a)
  requires(cuda::std::is_floating_point_v<T>)
{
  *out = (*a < 0) ? -(*a) : *a;
  return errc::OK;
}

/**
 * @brief Computes absolute value for decimal inputs with ANSI overflow checks.
 *
 * @tparam R Decimal representation type.
 * @return errc::OVERFLOW on minimum representable input, else errc::OK.
 */
template <typename R>
__device__ inline errc ansi_abs(decimal<R>* out, decimal<R> const* a)
{
  if (a->value() == cuda::std::numeric_limits<R>::min()) { return errc::OVERFLOW; }
  auto rep = a->value() < 0 ? -a->value() : a->value();
  *out     = decimal<R>{numeric::scaled_integer<R>{rep, numeric::scale_type{a->scale()}}};
  return errc::OK;
}

/**
 * @brief Computes absolute value for optional inputs with ANSI overflow checks.
 *
 * @tparam T Value type.
 * @return Operation status from underlying abs, or errc::OK.
 */
template <typename T>
__device__ inline errc ansi_abs(optional<T>* out, optional<T> const* a)
{
  if (a->has_value()) {
    T r;
    if (errc e = ansi_abs(&r, &a->value()); e != errc::OK) {
      *out = nullopt;
      return e;
    }
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

/**
 * @brief Computes unary negation for signed inputs with ANSI overflow checks.
 *
 * @tparam T Signed type.
 * @param out Destination value.
 * @param a Input value.
 * @return errc::OVERFLOW on minimum representable input, else errc::OK.
 */
template <typename T>
__device__ inline errc ansi_neg(T* out, T const* a)
  requires(cuda::std::is_signed_v<T>)
{
  if (*a == cuda::std::numeric_limits<T>::min()) { return errc::OVERFLOW; }
  *out = -(*a);
  return errc::OK;
}

/**
 * @brief Computes unary negation for decimal inputs with ANSI overflow checks.
 *
 * @tparam R Decimal representation type.
 * @return errc::OVERFLOW on minimum representable input, else errc::OK.
 */
template <typename R>
__device__ inline errc ansi_neg(decimal<R>* out, decimal<R> const* a)
{
  if (a->value() == cuda::std::numeric_limits<R>::min()) { return errc::OVERFLOW; }
  auto rep = -a->value();
  *out     = decimal<R>{numeric::scaled_integer<R>{rep, numeric::scale_type{a->scale()}}};
  return errc::OK;
}

/**
 * @brief Computes unary negation for optional inputs with ANSI checks.
 *
 * @tparam T Signed type.
 * @return Operation status from underlying negate, or errc::OK.
 */
template <typename T>
__device__ inline errc ansi_neg(optional<T>* out, optional<T> const* a)
{
  if (a->has_value()) {
    T r;
    if (errc e = ansi_neg(&r, &a->value()); e != errc::OK) {
      *out = nullopt;
      return e;
    }
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

/**
 * @brief Validates decimal precision against a target precision value.
 *
 * @tparam R Decimal representation type.
 * @param out Destination decimal value.
 * @param a Input decimal value.
 * @param precision Maximum allowed precision.
 * @return errc::OVERFLOW when precision is invalid or exceeded, else errc::OK.
 */
template <typename R>
__device__ inline errc ansi_precision_check(decimal<R>* out,
                                            decimal<R> const* a,
                                            int32_t const* precision)
{
  if (*precision <= 0) { return errc::OVERFLOW; }

  auto value = a->value();
  if (value == cuda::std::numeric_limits<R>::min()) { return errc::OVERFLOW; }

  auto abs_value = value < 0 ? -value : value;

  if (abs_value >= detail::ipow10(static_cast<R>(*precision))) { return errc::OVERFLOW; }

  *out = *a;
  return errc::OK;
}

/**
 * @brief Validates optional decimal precision against a precision value.
 *
 * @tparam T Decimal value type.
 * @param out Destination optional value.
 * @param a Optional decimal input.
 * @param precision Precision.
 * @return Operation status from underlying precision check, or errc::OK.
 */
template <typename T>
__device__ inline errc ansi_precision_check(optional<T>* out,
                                            optional<T> const* a,
                                            optional<int32_t> const* precision)
{
  if (a->has_value() && precision->has_value()) {
    T r;
    if (errc e = ansi_precision_check(&r, &a->value(), &precision->value()); e != errc::OK) {
      *out = nullopt;
      return e;
    } else {
      *out = r;
      return errc::OK;
    }
  } else {
    *out = nullopt;
    return errc::OK;
  }
}

/**
 * @brief ANSI add that returns null instead of propagating arithmetic errors.
 *
 * @tparam T Operand and result type.
 * @param out Destination optional value.
 * @param a Left optional operand.
 * @param b Right optional operand.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc ansi_try_add(optional<T>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    T r;
    if (errc e = ansi_add(&r, &a->value(), &b->value()); e != errc::OK) {
      *out = nullopt;
    } else {
      *out = r;
    }
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

/**
 * @brief ANSI subtract that returns null instead of propagating arithmetic errors.
 *
 * @tparam T Operand and result type.
 * @param out Destination optional value.
 * @param a Optional minuend.
 * @param b Optional subtrahend.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc ansi_try_sub(optional<T>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    T r;
    if (errc e = ansi_sub(&r, &a->value(), &b->value()); e != errc::OK) {
      *out = nullopt;
    } else {
      *out = r;
    }
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

/**
 * @brief ANSI multiply that returns null instead of propagating arithmetic errors.
 *
 * @tparam T Operand and result type.
 * @param out Destination optional value.
 * @param a Left optional operand.
 * @param b Right optional operand.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc ansi_try_mul(optional<T>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    T r;
    if (errc e = ansi_mul(&r, &a->value(), &b->value()); e != errc::OK) {
      *out = nullopt;
    } else {
      *out = r;
    }
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

/**
 * @brief ANSI divide that returns null instead of propagating arithmetic errors.
 *
 * @tparam T Operand and result type.
 * @param out Destination optional value.
 * @param a Optional dividend.
 * @param b Optional divisor.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc ansi_try_div(optional<T>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    T r;
    if (errc e = ansi_div(&r, &a->value(), &b->value()); e != errc::OK) {
      *out = nullopt;
    } else {
      *out = r;
    }
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

/**
 * @brief ANSI modulus that returns null instead of propagating arithmetic errors.
 *
 * @tparam T Operand and result type.
 * @param out Destination optional value.
 * @param a Optional dividend.
 * @param b Optional divisor.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc ansi_try_mod(optional<T>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    T r;
    if (errc e = ansi_mod(&r, &a->value(), &b->value()); e != errc::OK) {
      *out = nullopt;
    } else {
      *out = r;
    }
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

/**
 * @brief ANSI absolute value that returns null instead of propagating overflow errors.
 *
 * @tparam T Operand and result type.
 * @param out Destination optional value.
 * @param a Optional input value.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc ansi_try_abs(optional<T>* out, optional<T> const* a)
{
  if (a->has_value()) {
    T r;
    if (errc e = ansi_abs(&r, &a->value()); e != errc::OK) {
      *out = nullopt;
    } else {
      *out = r;
    }
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

/**
 * @brief ANSI unary negation that returns null instead of propagating overflow errors.
 *
 * @tparam T Operand and result type.
 * @param out Destination optional value.
 * @param a Optional input value.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc ansi_try_neg(optional<T>* out, optional<T> const* a)
{
  if (a->has_value()) {
    T r;
    if (errc e = ansi_neg(&r, &a->value()); e != errc::OK) {
      *out = nullopt;
    } else {
      *out = r;
    }
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

/**
 * @brief ANSI precision check that returns null instead of propagating precision errors.
 *
 * @tparam R Decimal representation type.
 * @param out Destination optional decimal value.
 * @param a Optional decimal input.
 * @param precision Optional precision.
 * @return errc::OK.
 */
template <typename R>
__device__ inline errc ansi_try_precision_check(optional<decimal<R>>* out,
                                                optional<decimal<R>> const* a,
                                                optional<int32_t> const* precision)
{
  if (a->has_value() && precision->has_value()) {
    decimal<R> r;
    if (errc e = ansi_precision_check(&r, &a->value(), &precision->value()); e != errc::OK) {
      *out = nullopt;
    } else {
      *out = r;
    }
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

}  // namespace ops
}  // namespace CUDF_EXPORT cudf
