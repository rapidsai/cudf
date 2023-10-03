/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

/* TODO: This include of cuda_runtime.h is needed to work around
 * https://github.com/NVIDIA/libcudacxx/pull/476
 * which is resolved in CCCL 2.2.0. The macro definition order requires CUDA
 * headers to be included before libcudacxx headers.
 */
#include <cuda_runtime.h>

#include <cudf/types.hpp>

#include <cuda/std/type_traits>

namespace cudf {
namespace detail {

/**
 * @brief A function for integer exponentiation by squaring with a fixed base
 *
 * https://simple.wikipedia.org/wiki/Exponentiation_by_squaring <br>
 * Note: this is the iterative equivalent of the recursive definition (faster) <br>
 * Quick-bench: https://quick-bench.com/q/Wg7o7HYQC9FW5M0CO0wQAjSwP_Y
 *
 * @tparam ResultType The result type
 * @tparam Base The base to be used for exponentiation
 * @tparam Exponent The exponent type
 * @param exponent The exponent to be used for exponentiation
 * @return Result of `Base` to the power of `exponent`
 */
template <typename ResultType, int Base, typename Exponent>
CUDF_HOST_DEVICE constexpr inline ResultType int_pow(Exponent exponent)
{
  static_assert(cuda::std::is_integral_v<ResultType> and cuda::std::is_integral_v<Exponent>);

  if (exponent == 0) { return static_cast<ResultType>(1); }
  if constexpr (cuda::std::is_signed_v<Exponent>) {
    if (exponent < 0) {
      // Integer exponentiation with negative exponent is not possible.
      return 0;
    }
  }

  auto extra  = static_cast<ResultType>(1);
  auto square = static_cast<ResultType>(Base);

  if (exponent == 0) { return 1; }
  if (Base == 0) { return 0; }
  while (exponent > 1) {
    if (exponent & 1) {
      // The exponent is odd, so multiply by one factor of x.
      extra *= square;
      exponent -= 1;
    }
    // The exponent is even, so square x and divide the exponent y by 2.
    exponent /= 2;
    square *= square;
  }
  return square * extra;
}

/**
 * @brief A function for integer exponentiation by squaring
 *
 * https://simple.wikipedia.org/wiki/Exponentiation_by_squaring <br>
 * Note: this is the iterative equivalent of the recursive definition (faster) <br>
 * Quick-bench: https://quick-bench.com/q/Wg7o7HYQC9FW5M0CO0wQAjSwP_Y
 *
 * @tparam Base The base type
 * @tparam Exponent The exponent type
 * @param base The base to be used for exponentiation
 * @param exponent The exponent to be used for exponentiation
 * @return Result of `base` to the power of `exponent`
 */
template <typename Base, typename Exponent>
CUDF_HOST_DEVICE constexpr inline Base int_pow(Base base, Exponent exponent)
{
  static_assert(cuda::std::is_integral_v<Base> and cuda::std::is_integral_v<Exponent>);

  if constexpr (cuda::std::is_signed_v<Exponent>) {
    if (exponent < 0) {
      // Integer exponentiation with negative exponent is not possible.
      return 0;
    }
  }
  if (exponent == 0) { return 1; }
  if (base == 0) { return 0; }
  Base extra = 1;
  while (exponent > 1) {
    if (exponent & 1) {
      // The exponent is odd, so multiply by one factor of x.
      extra *= base;
      exponent -= 1;
    }
    // The exponent is even, so square x and divide the exponent y by 2.
    exponent /= 2;
    base *= base;
  }
  return base * extra;
}

}  // namespace detail

}  // namespace cudf
