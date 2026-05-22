/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/utilities/export.hpp>

#include <cuda/std/optional>

namespace CUDF_EXPORT cudf {
namespace ops {

/**
 * @brief Computes cube root.
 *
 * Scalar overloads support float and double inputs, and an optional overload propagates nulls.
 * @param out Destination for the computed value.
 * @param a Input value.
 */
__device__ inline void cbrt(float* out, float const* a) { *out = ::cbrtf(*a); }

/**
 * @brief Computes cube root for double input.
 *
 * @param out Destination for the computed value.
 * @param a Input value.
 */
__device__ inline void cbrt(double* out, double const* a) { *out = ::cbrt(*a); }

/**
 * @brief Computes cube root for optional input.
 *
 * @tparam T Input and output type.
 * @param out Destination optional value.
 * @param a Optional input value.
 */
template <typename T>
__device__ void cbrt(cuda::std::optional<T>* out, cuda::std::optional<T> const* a)
{
  if (a->has_value()) {
    T r;
    cbrt(&r, &a->value());
    *out = r;
  } else {
    *out = cuda::std::nullopt;
  }
}

/**
 * @brief Computes ceiling.
 *
 * Scalar overloads support float and double, a decimal overload preserves scale, and an optional
 * overload propagates nulls.
 * @param out Destination for the computed value.
 * @param a Input value.
 */
__device__ inline void ceil(float* out, float const* a) { *out = ::ceilf(*a); }

/**
 * @brief Computes ceiling for double input.
 *
 * @param out Destination for the computed value.
 * @param a Input value.
 */
__device__ inline void ceil(double* out, double const* a) { *out = ::ceil(*a); }

/**
 * @brief Computes ceiling for decimal input.
 *
 * @tparam R Decimal representation type.
 * @param out Destination decimal value.
 * @param a Input decimal value.
 */
template <typename R>
__device__ void ceil(numeric::decimal<R>* out, numeric::decimal<R> const* a)
{
  if (a->scale() >= 0) {
    *out = *a;
  } else {
    auto factor = detail::ipow10(-static_cast<R>(a->scale()));
    auto div    = a->value() / factor;
    auto rem    = a->value() % factor;
    if (rem == 0) {
      *out = *a;
    } else {
      auto val = a->value() > 0 ? (div + 1) : div;
      *out     = numeric::decimal<R>{numeric::scaled_integer<R>{val * factor, a->scale()}};
    }
  }
}

/**
 * @brief Computes ceiling for optional input.
 *
 * @tparam T Input and output type.
 * @param out Destination optional value.
 * @param a Optional input value.
 */
template <typename T>
__device__ void ceil(cuda::std::optional<T>* out, cuda::std::optional<T> const* a)
{
  if (a->has_value()) {
    T r;
    ceil(&r, &a->value());
    *out = r;
  } else {
    *out = cuda::std::nullopt;
  }
}

/**
 * @brief Computes natural exponential.
 *
 * Scalar overloads support float and double inputs, and an optional overload propagates nulls.
 * @param out Destination for the computed value.
 * @param a Input value.
 */
__device__ inline void exp(float* out, float const* a) { *out = ::expf(*a); }

/**
 * @brief Computes natural exponential for double input.
 *
 * @param out Destination for the computed value.
 * @param a Input value.
 */
__device__ inline void exp(double* out, double const* a) { *out = ::exp(*a); }

/**
 * @brief Computes natural exponential for optional input.
 *
 * @tparam T Input and output type.
 * @param out Destination optional value.
 * @param a Optional input value.
 */
template <typename T>
__device__ void exp(cuda::std::optional<T>* out, cuda::std::optional<T> const* a)
{
  if (a->has_value()) {
    T r;
    exp(&r, &a->value());
    *out = r;
  } else {
    *out = cuda::std::nullopt;
  }
}

/**
 * @brief Computes floor.
 *
 * Scalar overloads support float and double, a decimal overload preserves scale, and an optional
 * overload propagates nulls.
 * @param out Destination for the computed value.
 * @param a Input value.
 */
__device__ inline void floor(float* out, float const* a) { *out = ::floorf(*a); }

/**
 * @brief Computes floor for double input.
 *
 * @param out Destination for the computed value.
 * @param a Input value.
 */
__device__ inline void floor(double* out, double const* a) { *out = ::floor(*a); }

/**
 * @brief Computes floor for decimal input.
 *
 * @tparam R Decimal representation type.
 * @param out Destination decimal value.
 * @param a Input decimal value.
 */
template <typename R>
__device__ void floor(numeric::decimal<R>* out, numeric::decimal<R> const* a)
{
  if (a->scale() >= 0) {
    *out = *a;
  } else {
    auto factor = numeric::detail::ipow<R, numeric::Radix::BASE_10>(-static_cast<R>(a->scale()));
    auto div    = a->value() / factor;
    auto rem    = a->value() % factor;
    if (rem == 0) {
      *out = *a;
    } else {
      auto val = a->value() > 0 ? div : (div - 1);
      *out     = numeric::decimal<R>{numeric::scaled_integer<R>{val * factor, a->scale()}};
    }
  }
}

/**
 * @brief Computes floor for optional input.
 *
 * @tparam T Input and output type.
 * @param out Destination optional value.
 * @param a Optional input value.
 */
template <typename T>
__device__ void floor(cuda::std::optional<T>* out, cuda::std::optional<T> const* a)
{
  if (a->has_value()) {
    T r;
    floor(&r, &a->value());
    *out = r;
  } else {
    *out = cuda::std::nullopt;
  }
}

/**
 * @brief Computes natural logarithm.
 *
 * Scalar overloads support float and double inputs, and an optional overload propagates nulls.
 * @param out Destination for the computed value.
 * @param a Input value.
 */
__device__ inline void log(float* out, float const* a) { *out = ::logf(*a); }

/**
 * @brief Computes natural logarithm for double input.
 *
 * @param out Destination for the computed value.
 * @param a Input value.
 */
__device__ inline void log(double* out, double const* a) { *out = ::log(*a); }

/**
 * @brief Computes natural logarithm for optional input.
 *
 * @tparam T Input and output type.
 * @param out Destination optional value.
 * @param a Optional input value.
 */
template <typename T>
__device__ void log(cuda::std::optional<T>* out, cuda::std::optional<T> const* a)
{
  if (a->has_value()) {
    T r;
    log(&r, &a->value());
    *out = r;
  } else {
    *out = cuda::std::nullopt;
  }
}

/**
 * @brief Computes exponentiation.
 *
 * Scalar overloads support float and double inputs, and an optional overload propagates nulls.
 * @param out Destination for the computed value.
 * @param a Base value.
 * @param b Exponent value.
 */
__device__ inline void pow(float* out, float const* a, float const* b) { *out = ::powf(*a, *b); }

/**
 * @brief Computes exponentiation for double input.
 *
 * @param out Destination for the computed value.
 * @param a Base value.
 * @param b Exponent value.
 */
__device__ inline void pow(double* out, double const* a, double const* b) { *out = ::pow(*a, *b); }

/**
 * @brief Computes exponentiation for optional input.
 *
 * @tparam T Input and output type.
 * @param out Destination optional value.
 * @param a Optional base value.
 * @param b Optional exponent value.
 */
template <typename T>
__device__ void pow(cuda::std::optional<T>* out,
                    cuda::std::optional<T> const* a,
                    cuda::std::optional<T> const* b)
{
  if (a->has_value() && b->has_value()) {
    T r;
    pow(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = cuda::std::nullopt;
  }
}

/**
 * @brief Rounds to integral value using current rounding mode.
 *
 * Scalar overloads support float and double inputs, and an optional overload propagates nulls.
 * @param out Destination for the computed value.
 * @param a Input value.
 */
__device__ inline void rint(float* out, float const* a) { *out = ::rintf(*a); }

/**
 * @brief Rounds to integral value for double input.
 *
 * @param out Destination for the computed value.
 * @param a Input value.
 */
__device__ inline void rint(double* out, double const* a) { *out = ::rint(*a); }

/**
 * @brief Rounds to integral value for optional input.
 *
 * @tparam T Input and output type.
 * @param out Destination optional value.
 * @param a Optional input value.
 */
template <typename T>
__device__ void rint(cuda::std::optional<T>* out, cuda::std::optional<T> const* a)
{
  if (a->has_value()) {
    T r;
    rint(&r, &a->value());
    *out = r;
  } else {
    *out = cuda::std::nullopt;
  }
}

/**
 * @brief Computes square root.
 *
 * Scalar overloads support float and double inputs, and an optional overload propagates nulls.
 * @param out Destination for the computed value.
 * @param a Input value.
 */
__device__ inline void sqrt(float* out, float const* a) { *out = ::sqrtf(*a); }

/**
 * @brief Computes square root for double input.
 *
 * @param out Destination for the computed value.
 * @param a Input value.
 */
__device__ inline void sqrt(double* out, double const* a) { *out = ::sqrt(*a); }

/**
 * @brief Computes square root for optional input.
 *
 * @tparam T Input and output type.
 * @param out Destination optional value.
 * @param a Optional input value.
 */
template <typename T>
__device__ void sqrt(cuda::std::optional<T>* out, cuda::std::optional<T> const* a)
{
  if (a->has_value()) {
    T r;
    sqrt(&r, &a->value());
    *out = r;
  } else {
    *out = cuda::std::nullopt;
  }
}

}  // namespace ops
}  // namespace CUDF_EXPORT cudf
