/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/operators/types.cuh>

namespace CUDF_EXPORT cudf {

namespace ops {

template <typename T>
  requires(cuda::std::is_signed_v<T> || cuda::std::is_floating_point_v<T>)
__device__ inline errc abs(T* out, T const* a)
{
  *out = (*a < 0) ? -*a : *a;
  return errc::OK;
}

template <typename T>
  requires(cuda::std::is_unsigned_v<T>)
__device__ inline errc abs(T* out, T const* a)
{
  *out = *a;
  return errc::OK;
}

template <typename R>
__device__ inline errc abs(decimal<R>* out, decimal<R> const* a)
{
  auto rep = a->value() < 0 ? -a->value() : a->value();
  *out     = decimal<R>{numeric::scaled_integer<R>{rep, a->scale()}};
  return errc::OK;
}

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

template <typename T>
__device__ inline errc add(T* out, T const* a, T const* b)
{
  *out = (*a + *b);
  return errc::OK;
}

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

template <typename T>
__device__ inline errc div(T* out, T const* a, T const* b)
{
  *out = (*a / *b);
  return errc::OK;
}

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

template <typename T>
__device__ inline errc mod(T* out, T const* a, T const* b)
{
  *out = (*a % *b);
  return errc::OK;
}

__device__ inline errc mod(float* out, float const* a, float const* b)
{
  *out = ::fmodf(*a, *b);
  return errc::OK;
}

__device__ inline errc mod(double* out, double const* a, double const* b)
{
  *out = ::fmod(*a, *b);
  return errc::OK;
}

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

template <typename T>
__device__ inline errc mul(T* out, T const* a, T const* b)
{
  *out = (*a * *b);
  return errc::OK;
}

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

template <typename T>
  requires(cuda::std::is_signed_v<T>)
__device__ inline errc neg(T* out, T const* a)
{
  *out = -(*a);
  return errc::OK;
}

template <typename Rep>
__device__ inline errc neg(decimal<Rep>* out, decimal<Rep> const* a)
{
  auto rep = -a->value();
  *out     = decimal<Rep>{numeric::scaled_integer<Rep>{rep, a->scale()}};
  return errc::OK;
}

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

template <typename T>
__device__ inline errc sub(T* out, T const* a, T const* b)
{
  *out = *a - *b;
  return errc::OK;
}

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

}  // namespace ops
}  // namespace CUDF_EXPORT cudf
