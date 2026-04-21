/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/operators/types.cuh>

namespace CUDF_EXPORT cudf {

namespace ops {

template <typename T>
__device__ inline errc cbrt(T* out, T const* a);

template <>
__device__ inline errc cbrt<float>(float* out, float const* a)
{
  *out = ::cbrtf(*a);
  return errc::OK;
}

template <>
__device__ inline errc cbrt<double>(double* out, double const* a)
{
  *out = ::cbrt(*a);
  return errc::OK;
}

template <typename T>
__device__ inline errc cbrt(optional<T>* out, optional<T> const* a)
{
  if (a->is_valid()) {
    T r;
    cbrt(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

template <typename T>
__device__ inline errc ceil(T* out, T const* a);

template <>
__device__ inline errc ceil<float>(float* out, float const* a)
{
  *out = ::ceilf(*a);
  return errc::OK;
}

template <>
__device__ inline errc ceil<double>(double* out, double const* a)
{
  *out = ::ceil(*a);
  return errc::OK;
}

template <typename T>
__device__ inline errc ceil(optional<T>* out, optional<T> const* a)
{
  if (a->is_valid()) {
    T r;
    ceil(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

template <typename T>
__device__ inline errc exp(T* out, T const* a);

template <>
__device__ inline errc exp<float>(float* out, float const* a)
{
  *out = ::expf(*a);
  return errc::OK;
}

template <>
__device__ inline errc exp<double>(double* out, double const* a)
{
  *out = ::exp(*a);
  return errc::OK;
}

template <typename T>
__device__ inline errc exp(optional<T>* out, optional<T> const* a)
{
  if (a->is_valid()) {
    T r;
    exp(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

template <typename T>
__device__ inline errc floor(T* out, T const* a);

template <>
__device__ inline errc floor<float>(float* out, float const* a)
{
  *out = ::floorf(*a);
  return errc::OK;
}

template <>
__device__ inline errc floor<double>(double* out, double const* a)
{
  *out = ::floor(*a);
  return errc::OK;
}

template <typename T>
__device__ inline errc floor(optional<T>* out, optional<T> const* a)
{
  if (a->is_valid()) {
    T r;
    floor(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

template <typename T>
__device__ inline errc log(T* out, T const* a);

template <>
__device__ inline errc log<float>(float* out, float const* a)
{
  *out = ::logf(*a);
  return errc::OK;
}

template <>
__device__ inline errc log<double>(double* out, double const* a)
{
  *out = ::log(*a);
  return errc::OK;
}

template <typename T>
__device__ inline errc log(optional<T>* out, optional<T> const* a)
{
  if (a->is_valid()) {
    T r;
    log(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

template <typename T>
__device__ inline errc pow(T* out, T const* a, T const* b);

template <>
__device__ inline errc pow<float>(float* out, float const* a, float const* b)
{
  *out = ::powf(*a, *b);
  return errc::OK;
}

template <>
__device__ inline errc pow<double>(double* out, double const* a, double const* b)
{
  *out = ::pow(*a, *b);
  return errc::OK;
}

template <typename T>
__device__ inline errc pow(optional<T>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->is_valid() && b->is_valid()) {
    T r;
    pow(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

template <typename T>
__device__ inline errc pymod(T* out, T const* a, T const* b)
{
  *out = (*a % *b + *b) % *b;
  return errc::OK;
}

template <>
__device__ inline errc pymod<float>(float* out, float const* a, float const* b)
{
  *out = ::fmodf(::fmodf(*a, *b) + *b, *b);
  return errc::OK;
}

template <>
__device__ inline errc pymod<double>(double* out, double const* a, double const* b)
{
  *out = ::fmod(::fmod(*a, *b) + *b, *b);
  return errc::OK;
}

template <typename T>
__device__ inline errc pymod(optional<T>* out, optional<T> const* a, optional<T> const* b)
{
  if (a->is_valid() && b->is_valid()) {
    T r;
    pymod(&r, &a->value(), &b->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

template <typename T>
__device__ inline errc rint(T* out, T const* a);

template <>
__device__ inline errc rint<float>(float* out, float const* a)
{
  *out = ::rintf(*a);
  return errc::OK;
}

template <>
__device__ inline errc rint<double>(double* out, double const* a)
{
  *out = ::rint(*a);
  return errc::OK;
}

template <typename T>
__device__ inline errc rint(optional<T>* out, optional<T> const* a)
{
  if (a->is_valid()) {
    T r;
    rint(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

}  // namespace ops
}  // namespace CUDF_EXPORT cudf
