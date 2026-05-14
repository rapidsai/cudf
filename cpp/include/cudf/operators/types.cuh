/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/operators/error.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/wrappers/durations.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <cuda/std/chrono>
#include <cuda/std/limits>
#include <cuda/std/optional>
#include <cuda/std/type_traits>

namespace CUDF_EXPORT cudf {
namespace ops {

template <typename T>
using optional = cuda::std::optional<T>;

inline constexpr auto nullopt = cuda::std::nullopt;

template <typename R>
using decimal = numeric::fixed_point<R, numeric::Radix::BASE_10>;

template <typename R, typename Ratio>
using duration = cuda::std::chrono::duration<R, Ratio>;

namespace detail {

/**
 * @brief Computes an integer power of ten.
 *
 * @tparam T Integral exponent and return type.
 * @param exponent Exponent value.
 * @return Ten raised to @p exponent.
 */
template <typename T>
__device__ constexpr T ipow10(T exponent)
{
  if (exponent == 0) { return 1; }

  T extra  = 1;
  T square = 10;
  T n      = exponent;

  while (n > 1) {
    if ((n & 1) == 1) { extra *= square; }
    n >>= 1;
    square *= square;
  }

  return square * extra;
}

}  // namespace detail

/**
 * @brief Copies an input value to the output.
 *
 * @tparam T Value type.
 * @param out Destination value.
 * @param a Input value.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc identity(T* out, T const* a)
{
  *out = *a;
  return errc::OK;
}

/**
 * @brief Copies an optional input value to the output.
 *
 * @tparam T Value type.
 * @param out Destination optional value.
 * @param a Optional input value.
 * @return errc::OK.
 */
template <typename T>
__device__ inline errc identity(optional<T>* out, optional<T> const* a)
{
  *out = *a;
  return errc::OK;
}

}  // namespace ops
}  // namespace CUDF_EXPORT cudf
