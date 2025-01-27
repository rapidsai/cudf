/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

#include <cudf/types.hpp>

#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include <algorithm>
#include <string>

namespace CUDF_EXPORT numeric {
namespace detail {

template <typename T>
auto to_string(T value) -> std::string
{
  if constexpr (cuda::std::is_same_v<T, __int128_t>) {
    auto s          = std::string{};
    auto const sign = value < 0;
    if (sign) {
      value += 1;  // avoid overflowing if value == _int128_t lowest
      value *= -1;
      if (value == cuda::std::numeric_limits<__int128_t>::max())
        return "-170141183460469231731687303715884105728";
      value += 1;  // can add back the one, no need to avoid overflow anymore
    }
    while (value) {
      s.push_back("0123456789"[value % 10]);
      value /= 10;
    }
    if (sign) s.push_back('-');
    std::reverse(s.begin(), s.end());
    return s;
  } else {
    return std::to_string(value);
  }
  return std::string{};  // won't ever hit here, need to suppress warning though
}

template <typename T>
CUDF_HOST_DEVICE constexpr auto abs(T value)
{
  return value >= 0 ? value : -value;
}

template <typename T>
CUDF_HOST_DEVICE inline auto min(T lhs, T rhs)
{
  return lhs < rhs ? lhs : rhs;
}

template <typename T>
CUDF_HOST_DEVICE inline auto max(T lhs, T rhs)
{
  return lhs > rhs ? lhs : rhs;
}

template <typename BaseType>
CUDF_HOST_DEVICE constexpr auto exp10(int32_t exponent)
{
  BaseType value = 1;
  while (exponent > 0)
    value *= 10, --exponent;
  return value;
}

}  // namespace detail
}  // namespace CUDF_EXPORT numeric
