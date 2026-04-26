/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/wrappers/durations.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <cuda/std/chrono>
#include <cuda/std/limits>
#include <cuda/std/optional>
#include <cuda/std/type_traits>

namespace CUDF_EXPORT cudf {
namespace ops {

enum errc : int { OK = 0, OVERFLOW = 1, DIVISION_BY_ZERO = 2 };

template <typename T>
using optional = cuda::std::optional<T>;

inline constexpr auto nullopt = cuda::std::nullopt;

template <typename R>
using decimal = numeric::fixed_point<R, numeric::Radix::BASE_10>;

template <typename R, typename Ratio>
using duration = cuda::std::chrono::duration<R, Ratio>;

namespace detail {

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
}  // namespace ops
}  // namespace CUDF_EXPORT cudf
