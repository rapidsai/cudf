/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

#include <cudf/utilities/export.hpp>
#include <cudf/utilities/traits.hpp>

#include <cuda/std/type_traits>

namespace CUDF_EXPORT numeric {
namespace detail {

/**
 * @brief Determine the number of significant bits in an integer
 *
 * @tparam T Type of input integer value. Must be either uint32_t, uint64_t, or __uint128_t
 * @param value The integer whose bits are being counted
 * @return The number of significant bits: the # of bits - # of leading zeroes
 */
template <typename T,
          CUDF_ENABLE_IF(std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t> ||
                         std::is_same_v<T, __uint128_t>)>
CUDF_HOST_DEVICE inline constexpr int count_significant_bits(T value)
{
#ifdef __CUDA_ARCH__
  if constexpr (std::is_same_v<T, uint64_t>) {
    return 64 - __clzll(static_cast<int64_t>(value));
  } else if constexpr (std::is_same_v<T, uint32_t>) {
    return 32 - __clz(static_cast<int32_t>(value));
  } else if constexpr (std::is_same_v<T, __uint128_t>) {
    // 128 bit type, must break up into high and low components
    auto const high_bits = static_cast<int64_t>(value >> 64);
    auto const low_bits  = static_cast<int64_t>(value);
    return 128 - (__clzll(high_bits) + static_cast<int>(high_bits == 0) * __clzll(low_bits));
  }
#else
  // Undefined behavior to call __builtin_clzll() with zero in gcc and clang
  if (value == 0) { return 0; }

  if constexpr (std::is_same_v<T, uint64_t>) {
    return 64 - __builtin_clzll(value);
  } else if constexpr (std::is_same_v<T, uint32_t>) {
    return 32 - __builtin_clz(value);
  } else if constexpr (std::is_same_v<T, __uint128_t>) {
    // 128 bit type, must break up into high and low components
    auto const high_bits = static_cast<uint64_t>(value >> 64);
    if (high_bits == 0) {
      return 64 - __builtin_clzll(static_cast<uint64_t>(value));
    } else {
      return 128 - __builtin_clzll(high_bits);
    }
  }
#endif
}

}  // namespace detail
}  // namespace CUDF_EXPORT numeric
