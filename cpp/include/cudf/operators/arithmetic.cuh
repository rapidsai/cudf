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
__device__ inline errc abs(numeric::fixed_point<R, numeric::Radix::BASE_10>* out,
                           numeric::fixed_point<R, numeric::Radix::BASE_10> const* a)
{
  auto rep = a->value() < 0 ? -a->value() : a->value();
  *out =
    numeric::fixed_point<R, numeric::Radix::BASE_10>{numeric::scaled_integer<R>{rep, a->scale()}};
  return errc::OK;
}

template <typename R, typename Ratio>
__device__ inline errc abs(cuda::std::chrono::duration<R, Ratio>* out,
                           cuda::std::chrono::duration<R, Ratio> const* a)
{
  auto rep = a->count() < 0 ? -a->count() : a->count();
  *out     = cuda::std::chrono::duration<R, Ratio>{rep};
  return errc::OK;
}

template <typename T>
__device__ inline errc abs(optional<T>* out, optional<T> const* a)
{
  if (a->is_valid()) {
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
  if (a->is_valid() && b->is_valid()) {
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
  if (a->is_valid() && b->is_valid()) {
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

template <>
__device__ inline errc mod<float>(float* out, float const* a, float const* b)
{
  *out = ::fmodf(*a, *b);
  return errc::OK;
}

template <>
__device__ inline errc mod<double>(double* out, double const* a, double const* b)
{
  *out = ::fmod(*a, *b);
  return errc::OK;
}

template <typename T>
__device__ inline errc mod(optional<T>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->is_valid() && b->is_valid()) {
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
  if (a->is_valid() && b->is_valid()) {
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
__device__ inline errc neg(numeric::fixed_point<Rep, numeric::Radix::BASE_10>* out,
                           numeric::fixed_point<Rep, numeric::Radix::BASE_10> const* a)
{
  auto rep = -a->value();
  *out     = numeric::fixed_point<Rep, numeric::Radix::BASE_10>{
    numeric::scaled_integer<Rep>{rep, a->scale()}};
  return errc::OK;
}

template <typename Rep, typename Ratio>
__device__ inline errc neg(cuda::std::chrono::duration<Rep, Ratio>* out,
                           cuda::std::chrono::duration<Rep, Ratio> const* a)
{
  auto rep = -a->count();
  *out     = cuda::std::chrono::duration<Rep, Ratio>{rep};
  return errc::OK;
}

template <typename T>
__device__ inline errc neg(optional<T>* out, optional<T> const* a)
{
  if (a->is_valid()) {
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
  if (a->is_valid() && b->is_valid()) {
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
