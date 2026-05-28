/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/utilities/export.hpp>

#include <cuda/std/cmath>
#include <cuda/std/optional>

namespace CUDF_EXPORT cudf {
namespace ops {

/**
 * @brief Computes inverse cosine.
 *
 * @tparam T Value type
 * @param out Result destination.
 * @param a Input value.
 */
template <typename T>
__device__ inline void arccos(T* out, T const* a)
  requires(cuda::std::is_floating_point_v<T>)
{
  *out = cuda::std::acos(*a);
}

/**
 * @brief Computes inverse cosine.
 *
 * @tparam T Value type
 * @param out Result destination.
 * @param a Input value.
 */
template <typename T>
__device__ void arccos(cuda::std::optional<T>* out, cuda::std::optional<T> const* a)
{
  if (a->has_value()) {
    T r;
    arccos(&r, &a->value());
    *out = r;
  } else {
    *out = cuda::std::nullopt;
  }
}

/**
 * @brief Computes inverse hyperbolic cosine.
 *
 * @tparam T Value type
 * @param out Result destination.
 * @param a Input value.
 */
template <typename T>
__device__ inline void arccosh(T* out, T const* a)
  requires(cuda::std::is_floating_point_v<T>)
{
  *out = cuda::std::acosh(*a);
}

/**
 * @brief Computes inverse hyperbolic cosine.
 *
 * @tparam T Value type
 * @param out Result destination.
 * @param a Input value.
 */
template <typename T>
__device__ void arccosh(cuda::std::optional<T>* out, cuda::std::optional<T> const* a)
{
  if (a->has_value()) {
    T r;
    arccosh(&r, &a->value());
    *out = r;
  } else {
    *out = cuda::std::nullopt;
  }
}

/**
 * @brief Computes inverse sine.
 *
 * @tparam T Value type
 * @param out Result destination.
 * @param a Input value.
 */
template <typename T>
__device__ inline void arcsin(T* out, T const* a)
  requires(cuda::std::is_floating_point_v<T>)
{
  *out = cuda::std::asin(*a);
}

/**
 * @brief Computes inverse sine.
 *
 * @tparam T Value type
 * @param out Result destination.
 * @param a Input value.
 */
template <typename T>
__device__ void arcsin(cuda::std::optional<T>* out, cuda::std::optional<T> const* a)
{
  if (a->has_value()) {
    T r;
    arcsin(&r, &a->value());
    *out = r;
  } else {
    *out = cuda::std::nullopt;
  }
}

/**
 * @brief Computes inverse hyperbolic sine.
 *
 * @tparam T Value type
 * @param out Result destination.
 * @param a Input value.
 */
template <typename T>
__device__ inline void arcsinh(T* out, T const* a)
  requires(cuda::std::is_floating_point_v<T>)
{
  *out = cuda::std::asinh(*a);
}

/**
 * @brief Computes inverse hyperbolic sine.
 *
 * @tparam T Value type
 * @param out Result destination.
 * @param a Input value.
 */
template <typename T>
__device__ void arcsinh(cuda::std::optional<T>* out, cuda::std::optional<T> const* a)
{
  if (a->has_value()) {
    T r;
    arcsinh(&r, &a->value());
    *out = r;
  } else {
    *out = cuda::std::nullopt;
  }
}

/**
 * @brief Computes inverse tangent.
 *
 * @tparam T Value type
 * @param out Result destination.
 * @param a Input value.
 */
template <typename T>
__device__ inline void arctan(T* out, T const* a)
  requires(cuda::std::is_floating_point_v<T>)
{
  *out = cuda::std::atan(*a);
}

/**
 * @brief Computes inverse tangent.
 *
 * @tparam T Value type
 * @param out Result destination.
 * @param a Input value.
 */
template <typename T>
__device__ void arctan(cuda::std::optional<T>* out, cuda::std::optional<T> const* a)
{
  if (a->has_value()) {
    T r;
    arctan(&r, &a->value());
    *out = r;
  } else {
    *out = cuda::std::nullopt;
  }
}

/**
 * @brief Computes inverse hyperbolic tangent.
 *
 * @tparam T Value type
 * @param out Result destination.
 * @param a Input value.
 */
template <typename T>
__device__ inline void arctanh(T* out, T const* a)
  requires(cuda::std::is_floating_point_v<T>)
{
  *out = cuda::std::atanh(*a);
}

/**
 * @brief Computes inverse hyperbolic tangent.
 *
 * @tparam T Value type
 * @param out Result destination.
 * @param a Input value.
 */
template <typename T>
__device__ void arctanh(cuda::std::optional<T>* out, cuda::std::optional<T> const* a)
{
  if (a->has_value()) {
    T r;
    arctanh(&r, &a->value());
    *out = r;
  } else {
    *out = cuda::std::nullopt;
  }
}

/**
 * @brief Computes cosine.
 *
 * @tparam T Value type
 * @param out Result destination.
 * @param a Input value.
 */
template <typename T>
__device__ inline void cos(T* out, T const* a)
  requires(cuda::std::is_floating_point_v<T>)
{
  *out = cuda::std::cos(*a);
}

/**
 * @brief Computes cosine.
 *
 * @tparam T Value type
 * @param out Result destination.
 * @param a Input value.
 */
template <typename T>
__device__ void cos(cuda::std::optional<T>* out, cuda::std::optional<T> const* a)
{
  if (a->has_value()) {
    T r;
    cos(&r, &a->value());
    *out = r;
  } else {
    *out = cuda::std::nullopt;
  }
}

/**
 * @brief Computes hyperbolic cosine.
 *
 * @tparam T Value type
 * @param out Result destination.
 * @param a Input value.
 */
template <typename T>
__device__ inline void cosh(T* out, T const* a)
  requires(cuda::std::is_floating_point_v<T>)
{
  *out = cuda::std::cosh(*a);
}

/**
 * @brief Computes hyperbolic cosine.
 *
 * @tparam T Value type
 * @param out Result destination.
 * @param a Input value.
 */
template <typename T>
__device__ void cosh(cuda::std::optional<T>* out, cuda::std::optional<T> const* a)
{
  if (a->has_value()) {
    T r;
    cosh(&r, &a->value());
    *out = r;
  } else {
    *out = cuda::std::nullopt;
  }
}

/**
 * @brief Computes sine.
 *
 * @tparam T Value type
 * @param out Result destination.
 * @param a Input value.
 */
template <typename T>
__device__ inline void sin(T* out, T const* a)
  requires(cuda::std::is_floating_point_v<T>)
{
  *out = cuda::std::sin(*a);
}

/**
 * @brief Computes sine.
 *
 * @tparam T Value type
 * @param out Result destination.
 * @param a Input value.
 */
template <typename T>
__device__ void sin(cuda::std::optional<T>* out, cuda::std::optional<T> const* a)
{
  if (a->has_value()) {
    T r;
    sin(&r, &a->value());
    *out = r;
  } else {
    *out = cuda::std::nullopt;
  }
}

/**
 * @brief Computes hyperbolic sine.
 *
 * @tparam T Value type
 * @param out Result destination.
 * @param a Input value.
 */
template <typename T>
__device__ inline void sinh(T* out, T const* a)
  requires(cuda::std::is_floating_point_v<T>)
{
  *out = cuda::std::sinh(*a);
}

/**
 * @brief Computes hyperbolic sine.
 *
 * @tparam T Value type
 * @param out Result destination.
 * @param a Input value.
 */
template <typename T>
__device__ void sinh(cuda::std::optional<T>* out, cuda::std::optional<T> const* a)
{
  if (a->has_value()) {
    T r;
    sinh(&r, &a->value());
    *out = r;
  } else {
    *out = cuda::std::nullopt;
  }
}

/**
 * @brief Computes tangent.
 *
 * @tparam T Value type
 * @param out Result destination.
 * @param a Input value.
 */
template <typename T>
__device__ inline void tan(T* out, T const* a)
  requires(cuda::std::is_floating_point_v<T>)
{
  *out = cuda::std::tan(*a);
}

/**
 * @brief Computes tangent.
 *
 * @tparam T Value type
 * @param out Result destination.
 * @param a Input value.
 */
template <typename T>
__device__ void tan(cuda::std::optional<T>* out, cuda::std::optional<T> const* a)
{
  if (a->has_value()) {
    T r;
    tan(&r, &a->value());
    *out = r;
  } else {
    *out = cuda::std::nullopt;
  }
}

/**
 * @brief Computes hyperbolic tangent.
 *
 * @tparam T Value type
 * @param out Result destination.
 * @param a Input value.
 */
template <typename T>
__device__ inline void tanh(T* out, T const* a)
  requires(cuda::std::is_floating_point_v<T>)
{
  *out = cuda::std::tanh(*a);
}

/**
 * @brief Computes hyperbolic tangent.
 *
 * @tparam T Value type
 * @param out Result destination.
 * @param a Input value.
 */
template <typename T>
__device__ void tanh(cuda::std::optional<T>* out, cuda::std::optional<T> const* a)
{
  if (a->has_value()) {
    T r;
    tanh(&r, &a->value());
    *out = r;
  } else {
    *out = cuda::std::nullopt;
  }
}

}  // namespace ops
}  // namespace CUDF_EXPORT cudf
