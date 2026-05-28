/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/utilities/export.hpp>

#include <cuda/std/cmath>
#include <cuda/std/optional>

namespace CUDF_EXPORT cudf {
namespace ops {

/**
 * @brief Computes cube root
 *
 * @tparam T Value type.
 * @param out Result destination.
 * @param a Input value.
 */
template <typename T>
__device__ inline void cbrt(T* out, T const* a)
  requires(cuda::std::is_floating_point_v<T>)
{
  *out = cuda::std::cbrt(*a);
}

/**
 * @brief Computes cube root.
 *
 * @tparam T Value type.
 * @param out Result destination.
 * @param a Input value.
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
 * @tparam T Value type.
 * @param out Result destination.
 * @param a Input value.
 */
template <typename T>
__device__ inline void ceil(T* out, T const* a)
  requires(cuda::std::is_floating_point_v<T>)
{
  *out = cuda::std::ceil(*a);
}

/**
 * @brief Computes ceiling.
 *
 * @tparam R Decimal representation type.
 * @param out Result destination.
 * @param a Input value.
 */
template <typename R>
__device__ void ceil(numeric::decimal<R>* out, numeric::decimal<R> const* a)
{
  if (a->scale() >= 0) {
    *out = *a;
  } else {
    auto factor =
      numeric::detail::ipow<R, numeric::Radix::BASE_10>(-static_cast<int32_t>(a->scale()));
    auto div = a->value() / factor;
    auto rem = a->value() % factor;
    if (rem == 0) {
      *out = *a;
    } else {
      auto val = a->value() > 0 ? (div + 1) : div;
      *out     = numeric::decimal<R>{numeric::scaled_integer<R>{val * factor, a->scale()}};
    }
  }
}

/**
 * @brief Computes ceiling.
 *
 * @tparam T Value type.
 * @param out Result destination.
 * @param a Input value.
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
 * @tparam T Value type.
 * @param out Result destination.
 * @param a Input value.
 */
template <typename T>
__device__ inline void exp(T* out, T const* a)
  requires(cuda::std::is_floating_point_v<T>)
{
  *out = cuda::std::exp(*a);
}

/**
 * @brief Computes natural exponential.
 *
 * @tparam T Value type.
 * @param out Result destination.
 * @param a Input value.
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
 * @brief Computes floor of a value.
 *
 * @tparam T Value type.
 * @param out Result destination.
 * @param a Input value.
 */
template <typename T>
__device__ inline void floor(T* out, T const* a)
  requires(cuda::std::is_floating_point_v<T>)
{
  *out = cuda::std::floor(*a);
}

/**
 * @brief Computes floor of a value.
 *
 * @tparam R Decimal representation type.
 * @param out Result destination.
 * @param a Input value.
 */
template <typename R>
__device__ void floor(numeric::decimal<R>* out, numeric::decimal<R> const* a)
{
  if (a->scale() >= 0) {
    *out = *a;
  } else {
    auto factor =
      numeric::detail::ipow<R, numeric::Radix::BASE_10>(-static_cast<int32_t>(a->scale()));
    auto div = a->value() / factor;
    auto rem = a->value() % factor;
    if (rem == 0) {
      *out = *a;
    } else {
      auto val = a->value() > 0 ? div : (div - 1);
      *out     = numeric::decimal<R>{numeric::scaled_integer<R>{val * factor, a->scale()}};
    }
  }
}

/**
 * @brief Computes floor of a value.
 *
 * @tparam T Value type.
 * @param out Result destination.
 * @param a Input value.
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
 * @tparam T Value type.
 * @param out Result destination.
 * @param a Input value.
 */
template <typename T>
__device__ inline void log(T* out, T const* a)
  requires(cuda::std::is_floating_point_v<T>)
{
  *out = cuda::std::log(*a);
}

/**
 * @brief Computes natural logarithm.
 *
 * @tparam T Value type.
 * @param out Result destination.
 * @param a Input value.
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
 * @tparam T Value type.
 * @param out Result destination.
 * @param a Base value.
 * @param b Exponent value.
 */
template <typename T>
__device__ inline void pow(T* out, T const* a, T const* b)
  requires(cuda::std::is_floating_point_v<T>)
{
  *out = cuda::std::pow(*a, *b);
}

/**
 * @brief Computes exponentiation.
 *
 * @tparam T Value type.
 * @param out Result destination.
 * @param a Base value.
 * @param b Exponent value.
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
 * @brief Rounds to integral value.
 *
 * @tparam T Value type.
 * @param out Result destination.
 * @param a Input value.
 */
template <typename T>
__device__ inline void rint(T* out, T const* a)
  requires(cuda::std::is_floating_point_v<T>)
{
  *out = cuda::std::rint(*a);
}

/**
 * @brief Rounds to integral value.
 *
 * @tparam T Value type.
 * @param out Result destination.
 * @param a Input value.
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
 * @param out Result destination.
 * @param a Input value.
 */
template <typename T>
__device__ inline void sqrt(T* out, T const* a)
  requires(cuda::std::is_floating_point_v<T>)
{
  *out = cuda::std::sqrt(*a);
}

/**
 * @brief Computes square root.
 *
 * @tparam T Value type.
 * @param out Result destination.
 * @param a Input value.
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
