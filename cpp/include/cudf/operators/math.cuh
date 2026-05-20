/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/operators/types.cuh>

namespace CUDF_EXPORT cudf {
namespace ops {

/**
 * @brief Computes cube root.
 *
 * Scalar overloads support float and double inputs, and an optional overload propagates nulls.
 * @param out Destination for the computed value.
 * @param a Input value.
 * @return errc::OK.
 */
__device__ inline errc cbrt(float* out, float const* a)
{
  *out = ::cbrtf(*a);
  return errc::OK;
}

/**
 * @brief Computes cube root for double input.
 *
 * @param out Destination for the computed value.
 * @param a Input value.
 * @return errc::OK.
 */
__device__ inline errc cbrt(double* out, double const* a)
{
  *out = ::cbrt(*a);
  return errc::OK;
}

/**
 * @brief Computes cube root for optional input.
 *
 * @tparam T Input and output type.
 * @param out Destination optional value.
 * @param a Optional input value.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc cbrt(optional<T>* out, optional<T> const* a)
{
  if (a->has_value()) {
    T r;
    cbrt(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

/**
 * @brief Computes ceiling.
 *
 * Scalar overloads support float and double, a decimal overload preserves scale, and an optional
 * overload propagates nulls.
 * @param out Destination for the computed value.
 * @param a Input value.
 * @return errc::OK.
 */
__device__ inline errc ceil(float* out, float const* a)
{
  *out = ::ceilf(*a);
  return errc::OK;
}

/**
 * @brief Computes ceiling for double input.
 *
 * @param out Destination for the computed value.
 * @param a Input value.
 * @return errc::OK.
 */
__device__ inline errc ceil(double* out, double const* a)
{
  *out = ::ceil(*a);
  return errc::OK;
}

/**
 * @brief Computes ceiling for decimal input.
 *
 * @tparam R Decimal representation type.
 * @param out Destination decimal value.
 * @param a Input decimal value.
 * @return errc::OK.
 */
template <typename R>
__device__ inline errc ceil(decimal<R>* out, decimal<R> const* a)
{
  if (a->scale() >= 0) {
    *out = *a;
    return errc::OK;
  }
  auto factor = detail::ipow10(-static_cast<R>(a->scale()));
  auto div    = a->value() / factor;
  auto rem    = a->value() % factor;
  if (rem == 0) {
    *out = *a;
  } else {
    auto val = a->value() > 0 ? (div + 1) : div;
    *out     = decimal<R>{numeric::scaled_integer<R>{val * factor, a->scale()}};
  }
  return errc::OK;
}

/**
 * @brief Computes ceiling for optional input.
 *
 * @tparam T Input and output type.
 * @param out Destination optional value.
 * @param a Optional input value.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc ceil(optional<T>* out, optional<T> const* a)
{
  if (a->has_value()) {
    T r;
    ceil(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

/**
 * @brief Computes natural exponential.
 *
 * Scalar overloads support float and double inputs, and an optional overload propagates nulls.
 * @param out Destination for the computed value.
 * @param a Input value.
 * @return errc::OK.
 */
__device__ inline errc exp(float* out, float const* a)
{
  *out = ::expf(*a);
  return errc::OK;
}

/**
 * @brief Computes natural exponential for double input.
 *
 * @param out Destination for the computed value.
 * @param a Input value.
 * @return errc::OK.
 */
__device__ inline errc exp(double* out, double const* a)
{
  *out = ::exp(*a);
  return errc::OK;
}

/**
 * @brief Computes natural exponential for optional input.
 *
 * @tparam T Input and output type.
 * @param out Destination optional value.
 * @param a Optional input value.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc exp(optional<T>* out, optional<T> const* a)
{
  if (a->has_value()) {
    T r;
    exp(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

/**
 * @brief Computes floor.
 *
 * Scalar overloads support float and double, a decimal overload preserves scale, and an optional
 * overload propagates nulls.
 * @param out Destination for the computed value.
 * @param a Input value.
 * @return errc::OK.
 */
__device__ inline errc floor(float* out, float const* a)
{
  *out = ::floorf(*a);
  return errc::OK;
}

/**
 * @brief Computes floor for double input.
 *
 * @param out Destination for the computed value.
 * @param a Input value.
 * @return errc::OK.
 */
__device__ inline errc floor(double* out, double const* a)
{
  *out = ::floor(*a);
  return errc::OK;
}

/**
 * @brief Computes floor for decimal input.
 *
 * @tparam R Decimal representation type.
 * @param out Destination decimal value.
 * @param a Input decimal value.
 * @return errc::OK.
 */
template <typename R>
__device__ inline errc floor(decimal<R>* out, decimal<R> const* a)
{
  if (a->scale() >= 0) {
    *out = *a;
    return errc::OK;
  }
  auto factor = detail::ipow10(-static_cast<R>(a->scale()));
  auto div    = a->value() / factor;
  auto rem    = a->value() % factor;
  if (rem == 0) {
    *out = *a;
  } else {
    auto val = a->value() > 0 ? div : (div - 1);
    *out     = decimal<R>{numeric::scaled_integer<R>{val * factor, a->scale()}};
  }
  return errc::OK;
}

/**
 * @brief Computes floor for optional input.
 *
 * @tparam T Input and output type.
 * @param out Destination optional value.
 * @param a Optional input value.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc floor(optional<T>* out, optional<T> const* a)
{
  if (a->has_value()) {
    T r;
    floor(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

/**
 * @brief Computes natural logarithm.
 *
 * Scalar overloads support float and double inputs, and an optional overload propagates nulls.
 * @param out Destination for the computed value.
 * @param a Input value.
 * @return errc::OK.
 */
__device__ inline errc log(float* out, float const* a)
{
  *out = ::logf(*a);
  return errc::OK;
}

/**
 * @brief Computes natural logarithm for double input.
 *
 * @param out Destination for the computed value.
 * @param a Input value.
 * @return errc::OK.
 */
__device__ inline errc log(double* out, double const* a)
{
  *out = ::log(*a);
  return errc::OK;
}

/**
 * @brief Computes natural logarithm for optional input.
 *
 * @tparam T Input and output type.
 * @param out Destination optional value.
 * @param a Optional input value.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc log(optional<T>* out, optional<T> const* a)
{
  if (a->has_value()) {
    T r;
    log(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

/**
 * @brief Computes exponentiation.
 *
 * Scalar overloads support float and double inputs, and an optional overload propagates nulls.
 * @param out Destination for the computed value.
 * @param a Base value.
 * @param b Exponent value.
 * @return errc::OK.
 */
__device__ inline errc pow(float* out, float const* a, float const* b)
{
  *out = ::powf(*a, *b);
  return errc::OK;
}

/**
 * @brief Computes exponentiation for double input.
 *
 * @param out Destination for the computed value.
 * @param a Base value.
 * @param b Exponent value.
 * @return errc::OK.
 */
__device__ inline errc pow(double* out, double const* a, double const* b)
{
  *out = ::pow(*a, *b);
  return errc::OK;
}

/**
 * @brief Computes exponentiation for optional input.
 *
 * @tparam T Input and output type.
 * @param out Destination optional value.
 * @param a Optional base value.
 * @param b Optional exponent value.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc pow(optional<T>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    T r;
    pow(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

/**
 * @brief Rounds to integral value using current rounding mode.
 *
 * Scalar overloads support float and double inputs, and an optional overload propagates nulls.
 * @param out Destination for the computed value.
 * @param a Input value.
 * @return errc::OK.
 */
__device__ inline errc rint(float* out, float const* a)
{
  *out = ::rintf(*a);
  return errc::OK;
}

/**
 * @brief Rounds to integral value for double input.
 *
 * @param out Destination for the computed value.
 * @param a Input value.
 * @return errc::OK.
 */
__device__ inline errc rint(double* out, double const* a)
{
  *out = ::rint(*a);
  return errc::OK;
}

/**
 * @brief Rounds to integral value for optional input.
 *
 * @tparam T Input and output type.
 * @param out Destination optional value.
 * @param a Optional input value.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc rint(optional<T>* out, optional<T> const* a)
{
  if (a->has_value()) {
    T r;
    rint(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

/**
 * @brief Computes square root.
 *
 * Scalar overloads support float and double inputs, and an optional overload propagates nulls.
 * @param out Destination for the computed value.
 * @param a Input value.
 * @return errc::OK.
 */
__device__ inline errc sqrt(float* out, float const* a)
{
  *out = ::sqrtf(*a);
  return errc::OK;
}

/**
 * @brief Computes square root for double input.
 *
 * @param out Destination for the computed value.
 * @param a Input value.
 * @return errc::OK.
 */
__device__ inline errc sqrt(double* out, double const* a)
{
  *out = ::sqrt(*a);
  return errc::OK;
}

/**
 * @brief Computes square root for optional input.
 *
 * @tparam T Input and output type.
 * @param out Destination optional value.
 * @param a Optional input value.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc sqrt(optional<T>* out, optional<T> const* a)
{
  if (a->has_value()) {
    T r;
    sqrt(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

}  // namespace ops
}  // namespace CUDF_EXPORT cudf
