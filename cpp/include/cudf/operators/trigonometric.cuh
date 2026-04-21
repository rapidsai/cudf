/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/operators/types.cuh>

namespace CUDF_EXPORT cudf {

namespace ops {

template <typename T>
__device__ inline errc arccos(T* out, T const* a);

template <>
__device__ inline errc arccos<float>(float* out, float const* a)
{
  *out = ::acosf(*a);
  return errc::OK;
}

template <>
__device__ inline errc arccos<double>(double* out, double const* a)
{
  *out = ::acos(*a);
  return errc::OK;
}

template <typename T>
__device__ inline errc arccos(optional<T>* out, optional<T> const* a)
{
  if (a->is_valid()) {
    T r;
    arccos(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

template <typename T>
__device__ inline errc arccosh(T* out, T const* a);

template <>
__device__ inline errc arccosh<float>(float* out, float const* a)
{
  *out = ::acoshf(*a);
  return errc::OK;
}

template <>
__device__ inline errc arccosh<double>(double* out, double const* a)
{
  *out = ::acosh(*a);
  return errc::OK;
}

template <typename T>
__device__ inline errc arccosh(optional<T>* out, optional<T> const* a)
{
  if (a->is_valid()) {
    T r;
    arccosh(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

template <typename T>
__device__ inline errc arcsin(T* out, T const* a);

template <>
__device__ inline errc arcsin<float>(float* out, float const* a)
{
  *out = ::asinf(*a);
  return errc::OK;
}

template <>
__device__ inline errc arcsin<double>(double* out, double const* a)
{
  *out = ::asin(*a);
  return errc::OK;
}

template <typename T>
__device__ inline errc arcsin(optional<T>* out, optional<T> const* a)
{
  if (a->is_valid()) {
    T r;
    arcsin(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

template <typename T>
__device__ inline errc arcsinh(T* out, T const* a);

template <>
__device__ inline errc arcsinh<float>(float* out, float const* a)
{
  *out = ::asinhf(*a);
  return errc::OK;
}

template <>
__device__ inline errc arcsinh<double>(double* out, double const* a)
{
  *out = ::asinh(*a);
  return errc::OK;
}

template <typename T>
__device__ inline errc arcsinh(optional<T>* out, optional<T> const* a)
{
  if (a->is_valid()) {
    T r;
    arcsinh(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

template <typename T>
__device__ inline errc arctan(T* out, T const* a);

template <>
__device__ inline errc arctan<float>(float* out, float const* a)
{
  *out = ::atanf(*a);
  return errc::OK;
}

template <>
__device__ inline errc arctan<double>(double* out, double const* a)
{
  *out = ::atan(*a);
  return errc::OK;
}

template <typename T>
__device__ inline errc arctan(optional<T>* out, optional<T> const* a)
{
  if (a->is_valid()) {
    T r;
    arctan(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

template <typename T>
__device__ inline errc arctanh(T* out, T const* a);

template <>
__device__ inline errc arctanh<float>(float* out, float const* a)
{
  *out = ::atanhf(*a);
  return errc::OK;
}

template <>
__device__ inline errc arctanh<double>(double* out, double const* a)
{
  *out = ::atanh(*a);
  return errc::OK;
}

template <typename T>
__device__ inline errc arctanh(optional<T>* out, optional<T> const* a)
{
  if (a->is_valid()) {
    T r;
    arctanh(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

template <typename T>
__device__ inline errc cos(T* out, T const* a);

template <>
__device__ inline errc cos<float>(float* out, float const* a)
{
  *out = ::cosf(*a);
  return errc::OK;
}

template <>
__device__ inline errc cos<double>(double* out, double const* a)
{
  *out = ::cos(*a);
  return errc::OK;
}

template <typename T>
__device__ inline errc cos(optional<T>* out, optional<T> const* a)
{
  if (a->is_valid()) {
    T r;
    cos(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

template <typename T>
__device__ inline errc cosh(T* out, T const* a);

template <>
__device__ inline errc cosh<float>(float* out, float const* a)
{
  *out = ::coshf(*a);
  return errc::OK;
}

template <>
__device__ inline errc cosh<double>(double* out, double const* a)
{
  *out = ::cosh(*a);
  return errc::OK;
}

template <typename T>
__device__ inline errc cosh(optional<T>* out, optional<T> const* a)
{
  if (a->is_valid()) {
    T r;
    cosh(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

template <typename T>
__device__ inline errc sin(T* out, T const* a);

template <>
__device__ inline errc sin<float>(float* out, float const* a)
{
  *out = ::sinf(*a);
  return errc::OK;
}

template <>
__device__ inline errc sin<double>(double* out, double const* a)
{
  *out = ::sin(*a);
  return errc::OK;
}

template <typename T>
__device__ inline errc sin(optional<T>* out, optional<T> const* a)
{
  if (a->is_valid()) {
    T r;
    sin(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

template <typename T>
__device__ inline errc sinh(T* out, T const* a);

template <>
__device__ inline errc sinh<float>(float* out, float const* a)
{
  *out = ::sinhf(*a);
  return errc::OK;
}

template <>
__device__ inline errc sinh<double>(double* out, double const* a)
{
  *out = ::sinh(*a);
  return errc::OK;
}

template <typename T>
__device__ inline errc sinh(optional<T>* out, optional<T> const* a)
{
  if (a->is_valid()) {
    T r;
    sinh(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

template <typename T>
__device__ inline errc tanh(T* out, T const* a);

template <>
__device__ inline errc tanh<float>(float* out, float const* a)
{
  *out = ::tanhf(*a);
  return errc::OK;
}

template <>
__device__ inline errc tanh<double>(double* out, double const* a)
{
  *out = ::tanh(*a);
  return errc::OK;
}

template <typename T>
__device__ inline errc tanh(optional<T>* out, optional<T> const* a)
{
  if (a->is_valid()) {
    T r;
    tanh(&r, &a->value());
    *out = r;
  } else {
    *out = nullopt;
  }
  return errc::OK;
}

}  // namespace ops
}  // namespace CUDF_EXPORT cudf
