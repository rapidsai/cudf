/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/utilities/export.hpp>

#include <cuda/std/optional>

namespace CUDF_EXPORT cudf {
namespace ops {

/**
 * @brief Computes inverse cosine.
 *
 * Scalar overloads support float and double inputs, and an optional overload propagates nulls.
 * @param out Destination for the computed value.
 * @param a Input value.
 */
__device__ inline void arccos(float* out, float const* a) { *out = ::acosf(*a); }

/**
 * @brief Computes inverse cosine for double input.
 *
 * @param out Destination for the computed value.
 * @param a Input value.
 */
__device__ inline void arccos(double* out, double const* a) { *out = ::acos(*a); }

/**
 * @brief Computes inverse cosine for optional input.
 *
 * @tparam T Input and output type.
 * @param out Destination optional value.
 * @param a Optional input value.
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
 * Scalar overloads support float and double inputs, and an optional overload propagates nulls.
 * @param out Destination for the computed value.
 * @param a Input value.
 */
__device__ inline void arccosh(float* out, float const* a) { *out = ::acoshf(*a); }

/**
 * @brief Computes inverse hyperbolic cosine for double input.
 *
 * @param out Destination for the computed value.
 * @param a Input value.
 */
__device__ inline void arccosh(double* out, double const* a) { *out = ::acosh(*a); }

/**
 * @brief Computes inverse hyperbolic cosine for optional input.
 *
 * @tparam T Input and output type.
 * @param out Destination optional value.
 * @param a Optional input value.
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
 * Scalar overloads support float and double inputs, and an optional overload propagates nulls.
 * @param out Destination for the computed value.
 * @param a Input value.
 */
__device__ inline void arcsin(float* out, float const* a) { *out = ::asinf(*a); }

/**
 * @brief Computes inverse sine for double input.
 *
 * @param out Destination for the computed value.
 * @param a Input value.
 */
__device__ inline void arcsin(double* out, double const* a) { *out = ::asin(*a); }

/**
 * @brief Computes inverse sine for optional input.
 *
 * @tparam T Input and output type.
 * @param out Destination optional value.
 * @param a Optional input value.
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
 * Scalar overloads support float and double inputs, and an optional overload propagates nulls.
 * @param out Destination for the computed value.
 * @param a Input value.
 */
__device__ inline void arcsinh(float* out, float const* a) { *out = ::asinhf(*a); }

/**
 * @brief Computes inverse hyperbolic sine for double input.
 *
 * @param out Destination for the computed value.
 * @param a Input value.
 */
__device__ inline void arcsinh(double* out, double const* a) { *out = ::asinh(*a); }

/**
 * @brief Computes inverse hyperbolic sine for optional input.
 *
 * @tparam T Input and output type.
 * @param out Destination optional value.
 * @param a Optional input value.
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
 * Scalar overloads support float and double inputs, and an optional overload propagates nulls.
 * @param out Destination for the computed value.
 * @param a Input value.
 */
__device__ inline void arctan(float* out, float const* a) { *out = ::atanf(*a); }

/**
 * @brief Computes inverse tangent for double input.
 *
 * @param out Destination for the computed value.
 * @param a Input value.
 */
__device__ inline void arctan(double* out, double const* a) { *out = ::atan(*a); }

/**
 * @brief Computes inverse tangent for optional input.
 *
 * @tparam T Input and output type.
 * @param out Destination optional value.
 * @param a Optional input value.
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
 * Scalar overloads support float and double inputs, and an optional overload propagates nulls.
 * @param out Destination for the computed value.
 * @param a Input value.
 */
__device__ inline void arctanh(float* out, float const* a) { *out = ::atanhf(*a); }

/**
 * @brief Computes inverse hyperbolic tangent for double input.
 *
 * @param out Destination for the computed value.
 * @param a Input value.
 */
__device__ inline void arctanh(double* out, double const* a) { *out = ::atanh(*a); }

/**
 * @brief Computes inverse hyperbolic tangent for optional input.
 *
 * @tparam T Input and output type.
 * @param out Destination optional value.
 * @param a Optional input value.
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
 * Scalar overloads support float and double inputs, and an optional overload propagates nulls.
 * @param out Destination for the computed value.
 * @param a Input value.
 */
__device__ inline void cos(float* out, float const* a) { *out = ::cosf(*a); }

/**
 * @brief Computes cosine for double input.
 *
 * @param out Destination for the computed value.
 * @param a Input value.
 */
__device__ inline void cos(double* out, double const* a) { *out = ::cos(*a); }

/**
 * @brief Computes cosine for optional input.
 *
 * @tparam T Input and output type.
 * @param out Destination optional value.
 * @param a Optional input value.
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
 * Scalar overloads support float and double inputs, and an optional overload propagates nulls.
 * @param out Destination for the computed value.
 * @param a Input value.
 */
__device__ inline void cosh(float* out, float const* a) { *out = ::coshf(*a); }

/**
 * @brief Computes hyperbolic cosine for double input.
 *
 * @param out Destination for the computed value.
 * @param a Input value.
 */
__device__ inline void cosh(double* out, double const* a) { *out = ::cosh(*a); }

/**
 * @brief Computes hyperbolic cosine for optional input.
 *
 * @tparam T Input and output type.
 * @param out Destination optional value.
 * @param a Optional input value.
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
 * Scalar overloads support float and double inputs, and an optional overload propagates nulls.
 * @param out Destination for the computed value.
 * @param a Input value.
 */
__device__ inline void sin(float* out, float const* a) { *out = ::sinf(*a); }

/**
 * @brief Computes sine for double input.
 *
 * @param out Destination for the computed value.
 * @param a Input value.
 */
__device__ inline void sin(double* out, double const* a) { *out = ::sin(*a); }

/**
 * @brief Computes sine for optional input.
 *
 * @tparam T Input and output type.
 * @param out Destination optional value.
 * @param a Optional input value.
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
 * Scalar overloads support float and double inputs, and an optional overload propagates nulls.
 * @param out Destination for the computed value.
 * @param a Input value.
 */
__device__ inline void sinh(float* out, float const* a) { *out = ::sinhf(*a); }

/**
 * @brief Computes hyperbolic sine for double input.
 *
 * @param out Destination for the computed value.
 * @param a Input value.
 */
__device__ inline void sinh(double* out, double const* a) { *out = ::sinh(*a); }

/**
 * @brief Computes hyperbolic sine for optional input.
 *
 * @tparam T Input and output type.
 * @param out Destination optional value.
 * @param a Optional input value.
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
 * Scalar overloads support float and double inputs, and an optional overload propagates nulls.
 * @param out Destination for the computed value.
 * @param a Input value.
 */
__device__ inline void tan(float* out, float const* a) { *out = ::tanf(*a); }

/**
 * @brief Computes tangent for double input.
 *
 * @param out Destination for the computed value.
 * @param a Input value.
 */
__device__ inline void tan(double* out, double const* a) { *out = ::tan(*a); }

/**
 * @brief Computes tangent for optional input.
 *
 * @tparam T Input and output type.
 * @param out Destination optional value.
 * @param a Optional input value.
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
 * Scalar overloads support float and double inputs, and an optional overload propagates nulls.
 * @param out Destination for the computed value.
 * @param a Input value.
 */
__device__ inline void tanh(float* out, float const* a) { *out = ::tanhf(*a); }

/**
 * @brief Computes hyperbolic tangent for double input.
 *
 * @param out Destination for the computed value.
 * @param a Input value.
 */
__device__ inline void tanh(double* out, double const* a) { *out = ::tanh(*a); }

/**
 * @brief Computes hyperbolic tangent for optional input.
 *
 * @tparam T Input and output type.
 * @param out Destination optional value.
 * @param a Optional input value.
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
