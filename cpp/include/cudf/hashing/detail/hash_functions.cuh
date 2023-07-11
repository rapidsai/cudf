/*
 * Copyright (c) 2017-2023, NVIDIA CORPORATION.
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

#include <cudf/utilities/traits.hpp>

#include <limits>

namespace cudf::hashing::detail {

template <typename K>
struct MurmurHash3_32;

template <typename Key>
using default_hash = MurmurHash3_32<Key>;

/**
 * Normalization of floating point NaNs, passthrough for all other values.
 */
template <typename T>
T __device__ inline normalize_nans(T const& key)
{
  if constexpr (cudf::is_floating_point<T>()) {
    if (std::isnan(key)) { return std::numeric_limits<T>::quiet_NaN(); }
  }
  return key;
}

/**
 * Normalization of floating point NaNs and zeros, passthrough for all other values.
 */
template <typename T>
T __device__ inline normalize_nans_and_zeros(T const& key)
{
  if constexpr (cudf::is_floating_point<T>()) {
    if (key == T{0.0}) { return T{0.0}; }
  }
  return normalize_nans(key);
}

__device__ inline uint32_t rotate_bits_left(uint32_t x, uint32_t r)
{
  // This function is equivalent to (x << r) | (x >> (32 - r))
  return __funnelshift_l(x, x, r);
}

__device__ inline uint32_t rotate_bits_right(uint32_t x, uint32_t r)
{
  // This function is equivalent to (x >> r) | (x << (32 - r))
  return __funnelshift_r(x, x, r);
}

__device__ inline uint64_t rotate_bits_right(uint64_t x, uint32_t r)
{
  return (x >> r) | (x << (64 - r));
}

/**
 * @brief  This hash function simply returns the value that is asked to be hash
 * reinterpreted as the result_type of the functor.
 */
template <typename Key>
struct IdentityHash {
  using result_type = uint32_t;
  IdentityHash()    = default;
  constexpr IdentityHash(uint32_t seed) : m_seed(seed) {}

  template <typename return_type = result_type>
  constexpr std::enable_if_t<!std::is_arithmetic_v<Key>, return_type> operator()(
    Key const& key) const
  {
    CUDF_UNREACHABLE("IdentityHash does not support this data type");
  }

  template <typename return_type = result_type>
  constexpr std::enable_if_t<std::is_arithmetic_v<Key>, return_type> operator()(
    Key const& key) const
  {
    return static_cast<result_type>(key);
  }

 private:
  uint32_t m_seed{0};
};

}  // namespace cudf::hashing::detail
