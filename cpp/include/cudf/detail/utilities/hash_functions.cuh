/*
 * Copyright (c) 2017-2021, NVIDIA CORPORATION.
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

#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/utilities/assert.cuh>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/types.hpp>

using hash_value_type = uint32_t;

namespace cudf {
namespace detail {

/**
 * Normalization of floating point NaNs and zeros, passthrough for all other values.
 */
template <typename T>
T __device__ inline normalize_nans_and_zeros(T const& key)
{
  if constexpr (is_floating_point<T>()) {
    if (isnan(key)) {
      return std::numeric_limits<T>::quiet_NaN();
    } else if (key == T{0.0}) {
      return T{0.0};
    }
  }
  return key;
}

/**
 * Modified GPU implementation of
 * https://johnnylee-sde.github.io/Fast-unsigned-integer-to-hex-string/
 * Copyright (c) 2015 Barry Clark
 * Licensed under the MIT license.
 * See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
 */
void __device__ inline uint32ToLowercaseHexString(uint32_t num, char* destination)
{
  // Transform 0xABCD1234 => 0x0000ABCD00001234 => 0x0B0A0D0C02010403
  uint64_t x = num;
  x          = ((x & 0xFFFF0000) << 16) | ((x & 0xFFFF));
  x          = ((x & 0xF0000000F) << 8) | ((x & 0xF0000000F0) >> 4) | ((x & 0xF0000000F00) << 16) |
      ((x & 0xF0000000F000) << 4);

  // Calculate a mask of ascii value offsets for bytes that contain alphabetical hex digits
  uint64_t offsets = (((x + 0x0606060606060606) >> 4) & 0x0101010101010101) * 0x27;

  x |= 0x3030303030303030;
  x += offsets;
  std::memcpy(destination, reinterpret_cast<uint8_t*>(&x), 8);
}

}  // namespace detail
}  // namespace cudf

// MurmurHash3_32 implementation from
// https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp
//-----------------------------------------------------------------------------
// MurmurHash3 was written by Austin Appleby, and is placed in the public
// domain. The author hereby disclaims copyright to this source code.
// Note - The x86 and x64 versions do _not_ produce the same results, as the
// algorithms are optimized for their respective platforms. You can still
// compile and run any of them on any platform, but your performance with the
// non-native version will be less than optimal.
template <typename Key>
struct MurmurHash3_32 {
  using argument_type = Key;
  using result_type   = hash_value_type;

  MurmurHash3_32() = default;
  constexpr MurmurHash3_32(uint32_t seed) : m_seed(seed) {}

  __device__ inline uint32_t rotl32(uint32_t x, int8_t r) const
  {
    return (x << r) | (x >> (32 - r));
  }

  __device__ inline uint32_t fmix32(uint32_t h) const
  {
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
  }

  /* Copyright 2005-2014 Daniel James.
   *
   * Use, modification and distribution is subject to the Boost Software
   * License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
   * http://www.boost.org/LICENSE_1_0.txt)
   */
  /**
   * @brief  Combines two hash values into a new single hash value. Called
   * repeatedly to create a hash value from several variables.
   * Taken from the Boost hash_combine function
   * https://www.boost.org/doc/libs/1_35_0/doc/html/boost/hash_combine_id241013.html
   *
   * @param lhs The first hash value to combine
   * @param rhs The second hash value to combine
   *
   * @returns A hash value that intelligently combines the lhs and rhs hash values
   */
  __device__ inline result_type hash_combine(result_type lhs, result_type rhs)
  {
    result_type combined{lhs};

    combined ^= rhs + 0x9e3779b9 + (combined << 6) + (combined >> 2);

    return combined;
  }

  result_type __device__ inline operator()(Key const& key) const { return compute(key); }

  // compute wrapper for floating point types
  template <typename T, std::enable_if_t<std::is_floating_point<T>::value>* = nullptr>
  hash_value_type __device__ inline compute_floating_point(T const& key) const
  {
    if (key == T{0.0}) {
      return compute(T{0.0});
    } else if (isnan(key)) {
      T nan = std::numeric_limits<T>::quiet_NaN();
      return compute(nan);
    } else {
      return compute(key);
    }
  }

  template <typename TKey>
  result_type __device__ inline compute(TKey const& key) const
  {
    constexpr int len         = sizeof(argument_type);
    uint8_t const* const data = reinterpret_cast<uint8_t const*>(&key);
    constexpr int nblocks     = len / 4;

    uint32_t h1           = m_seed;
    constexpr uint32_t c1 = 0xcc9e2d51;
    constexpr uint32_t c2 = 0x1b873593;
    //----------
    // body
    uint32_t const* const blocks = reinterpret_cast<uint32_t const*>(data + nblocks * 4);
    for (int i = -nblocks; i; i++) {
      uint32_t k1 = blocks[i];  // getblock32(blocks,i);
      k1 *= c1;
      k1 = rotl32(k1, 15);
      k1 *= c2;
      h1 ^= k1;
      h1 = rotl32(h1, 13);
      h1 = h1 * 5 + 0xe6546b64;
    }
    //----------
    // tail
    uint8_t const* tail = reinterpret_cast<uint8_t const*>(data + nblocks * 4);
    uint32_t k1         = 0;
    switch (len & 3) {
      case 3: k1 ^= tail[2] << 16;
      case 2: k1 ^= tail[1] << 8;
      case 1:
        k1 ^= tail[0];
        k1 *= c1;
        k1 = rotl32(k1, 15);
        k1 *= c2;
        h1 ^= k1;
    };
    //----------
    // finalization
    h1 ^= len;
    h1 = fmix32(h1);
    return h1;
  }

 private:
  uint32_t m_seed{cudf::DEFAULT_HASH_SEED};
};

template <>
hash_value_type __device__ inline MurmurHash3_32<bool>::operator()(bool const& key) const
{
  return this->compute(static_cast<uint8_t>(key));
}

/**
 * @brief Specialization of MurmurHash3_32 operator for strings.
 */
template <>
hash_value_type __device__ inline MurmurHash3_32<cudf::string_view>::operator()(
  cudf::string_view const& key) const
{
  auto const len        = key.size_bytes();
  uint8_t const* data   = reinterpret_cast<uint8_t const*>(key.data());
  int const nblocks     = len / 4;
  result_type h1        = m_seed;
  constexpr uint32_t c1 = 0xcc9e2d51;
  constexpr uint32_t c2 = 0x1b873593;
  auto getblock32       = [] __device__(uint32_t const* p, int i) -> uint32_t {
    // Individual byte reads for unaligned accesses (very likely)
    auto q = (uint8_t const*)(p + i);
    return q[0] | (q[1] << 8) | (q[2] << 16) | (q[3] << 24);
  };

  //----------
  // body
  uint32_t const* const blocks = reinterpret_cast<uint32_t const*>(data + nblocks * 4);
  for (int i = -nblocks; i; i++) {
    uint32_t k1 = getblock32(blocks, i);
    k1 *= c1;
    k1 = rotl32(k1, 15);
    k1 *= c2;
    h1 ^= k1;
    h1 = rotl32(h1, 13);
    h1 = h1 * 5 + 0xe6546b64;
  }
  //----------
  // tail
  uint8_t const* tail = reinterpret_cast<uint8_t const*>(data + nblocks * 4);
  uint32_t k1         = 0;
  switch (len & 3) {
    case 3: k1 ^= tail[2] << 16;
    case 2: k1 ^= tail[1] << 8;
    case 1:
      k1 ^= tail[0];
      k1 *= c1;
      k1 = rotl32(k1, 15);
      k1 *= c2;
      h1 ^= k1;
  };
  //----------
  // finalization
  h1 ^= len;
  h1 = fmix32(h1);
  return h1;
}

template <>
hash_value_type __device__ inline MurmurHash3_32<float>::operator()(float const& key) const
{
  return this->compute_floating_point(key);
}

template <>
hash_value_type __device__ inline MurmurHash3_32<double>::operator()(double const& key) const
{
  return this->compute_floating_point(key);
}

template <>
hash_value_type __device__ inline MurmurHash3_32<numeric::decimal32>::operator()(
  numeric::decimal32 const& key) const
{
  return this->compute(key.value());
}

template <>
hash_value_type __device__ inline MurmurHash3_32<numeric::decimal64>::operator()(
  numeric::decimal64 const& key) const
{
  return this->compute(key.value());
}

template <>
hash_value_type __device__ inline MurmurHash3_32<numeric::decimal128>::operator()(
  numeric::decimal128 const& key) const
{
  return this->compute(key.value());
}

template <>
hash_value_type __device__ inline MurmurHash3_32<cudf::list_view>::operator()(
  cudf::list_view const& key) const
{
  cudf_assert(false && "List column hashing is not supported");
  return 0;
}

template <>
hash_value_type __device__ inline MurmurHash3_32<cudf::struct_view>::operator()(
  cudf::struct_view const& key) const
{
  cudf_assert(false && "Direct hashing of struct_view is not supported");
  return 0;
}

template <typename Key>
struct SparkMurmurHash3_32 {
  using argument_type = Key;
  using result_type   = hash_value_type;

  SparkMurmurHash3_32() = default;
  constexpr SparkMurmurHash3_32(uint32_t seed) : m_seed(seed) {}

  __device__ inline uint32_t rotl32(uint32_t x, int8_t r) const
  {
    return (x << r) | (x >> (32 - r));
  }

  __device__ inline uint32_t fmix32(uint32_t h) const
  {
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
  }

  result_type __device__ inline operator()(Key const& key) const { return compute(key); }

  // compute wrapper for floating point types
  template <typename T, std::enable_if_t<std::is_floating_point<T>::value>* = nullptr>
  hash_value_type __device__ inline compute_floating_point(T const& key) const
  {
    if (isnan(key)) {
      T nan = std::numeric_limits<T>::quiet_NaN();
      return compute(nan);
    } else {
      return compute(key);
    }
  }

  template <typename TKey>
  result_type __device__ inline compute(TKey const& key) const
  {
    constexpr int len        = sizeof(TKey);
    int8_t const* const data = reinterpret_cast<int8_t const*>(&key);
    constexpr int nblocks    = len / 4;

    uint32_t h1           = m_seed;
    constexpr uint32_t c1 = 0xcc9e2d51;
    constexpr uint32_t c2 = 0x1b873593;
    //----------
    // body
    uint32_t const* const blocks = reinterpret_cast<uint32_t const*>(data + nblocks * 4);
    for (int i = -nblocks; i; i++) {
      uint32_t k1 = blocks[i];
      k1 *= c1;
      k1 = rotl32(k1, 15);
      k1 *= c2;
      h1 ^= k1;
      h1 = rotl32(h1, 13);
      h1 = h1 * 5 + 0xe6546b64;
    }
    //----------
    // byte by byte tail processing
    for (int i = nblocks * 4; i < len; i++) {
      int32_t k1 = data[i];
      k1 *= c1;
      k1 = rotl32(k1, 15);
      k1 *= c2;
      h1 ^= k1;
      h1 = rotl32(h1, 13);
      h1 = h1 * 5 + 0xe6546b64;
    }
    //----------
    // finalization
    h1 ^= len;
    h1 = fmix32(h1);
    return h1;
  }

 private:
  uint32_t m_seed{cudf::DEFAULT_HASH_SEED};
};

template <>
hash_value_type __device__ inline SparkMurmurHash3_32<bool>::operator()(bool const& key) const
{
  return this->compute<uint32_t>(key);
}

template <>
hash_value_type __device__ inline SparkMurmurHash3_32<int8_t>::operator()(int8_t const& key) const
{
  return this->compute<uint32_t>(key);
}

template <>
hash_value_type __device__ inline SparkMurmurHash3_32<uint8_t>::operator()(uint8_t const& key) const
{
  return this->compute<uint32_t>(key);
}

template <>
hash_value_type __device__ inline SparkMurmurHash3_32<int16_t>::operator()(int16_t const& key) const
{
  return this->compute<uint32_t>(key);
}

template <>
hash_value_type __device__ inline SparkMurmurHash3_32<uint16_t>::operator()(
  uint16_t const& key) const
{
  return this->compute<uint32_t>(key);
}

template <>
hash_value_type __device__ inline SparkMurmurHash3_32<numeric::decimal32>::operator()(
  numeric::decimal32 const& key) const
{
  return this->compute<uint64_t>(key.value());
}

template <>
hash_value_type __device__ inline SparkMurmurHash3_32<numeric::decimal64>::operator()(
  numeric::decimal64 const& key) const
{
  return this->compute<uint64_t>(key.value());
}

template <>
hash_value_type __device__ inline SparkMurmurHash3_32<numeric::decimal128>::operator()(
  numeric::decimal128 const& key) const
{
  return this->compute<__int128_t>(key.value());
}

template <>
hash_value_type __device__ inline SparkMurmurHash3_32<cudf::list_view>::operator()(
  cudf::list_view const& key) const
{
  cudf_assert(false && "List column hashing is not supported");
  return 0;
}

template <>
hash_value_type __device__ inline SparkMurmurHash3_32<cudf::struct_view>::operator()(
  cudf::struct_view const& key) const
{
  cudf_assert(false && "Direct hashing of struct_view is not supported");
  return 0;
}

/**
 * @brief Specialization of MurmurHash3_32 operator for strings.
 */
template <>
hash_value_type __device__ inline SparkMurmurHash3_32<cudf::string_view>::operator()(
  cudf::string_view const& key) const
{
  auto const len        = key.size_bytes();
  int8_t const* data    = reinterpret_cast<int8_t const*>(key.data());
  int const nblocks     = len / 4;
  result_type h1        = m_seed;
  constexpr uint32_t c1 = 0xcc9e2d51;
  constexpr uint32_t c2 = 0x1b873593;
  auto getblock32       = [] __device__(uint32_t const* p, int i) -> uint32_t {
    // Individual byte reads for unaligned accesses (very likely)
    auto q = (const uint8_t*)(p + i);
    return q[0] | (q[1] << 8) | (q[2] << 16) | (q[3] << 24);
  };

  //----------
  // body
  uint32_t const* const blocks = reinterpret_cast<uint32_t const*>(data + nblocks * 4);
  for (int i = -nblocks; i; i++) {
    uint32_t k1 = getblock32(blocks, i);
    k1 *= c1;
    k1 = rotl32(k1, 15);
    k1 *= c2;
    h1 ^= k1;
    h1 = rotl32(h1, 13);
    h1 = h1 * 5 + 0xe6546b64;
  }
  //----------
  // Spark's byte by byte tail processing
  for (int i = nblocks * 4; i < len; i++) {
    int32_t k1 = data[i];
    k1 *= c1;
    k1 = rotl32(k1, 15);
    k1 *= c2;
    h1 ^= k1;
    h1 = rotl32(h1, 13);
    h1 = h1 * 5 + 0xe6546b64;
  }
  //----------
  // finalization
  h1 ^= len;
  h1 = fmix32(h1);
  return h1;
}

template <>
hash_value_type __device__ inline SparkMurmurHash3_32<float>::operator()(float const& key) const
{
  return this->compute_floating_point(key);
}

template <>
hash_value_type __device__ inline SparkMurmurHash3_32<double>::operator()(double const& key) const
{
  return this->compute_floating_point(key);
}

/**
 * @brief  This hash function simply returns the value that is asked to be hash
 * reinterpreted as the result_type of the functor.
 */
template <typename Key>
struct IdentityHash {
  using result_type = hash_value_type;
  IdentityHash()    = default;
  constexpr IdentityHash(uint32_t seed) : m_seed(seed) {}

  /* Copyright 2005-2014 Daniel James.
   *
   * Use, modification and distribution is subject to the Boost Software
   * License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
   * http://www.boost.org/LICENSE_1_0.txt)
   */
  /**
   * @brief  Combines two hash values into a new single hash value. Called
   * repeatedly to create a hash value from several variables.
   * Taken from the Boost hash_combine function
   * https://www.boost.org/doc/libs/1_35_0/doc/html/boost/hash_combine_id241013.html
   *
   * @param lhs The first hash value to combine
   * @param rhs The second hash value to combine
   *
   * @returns A hash value that intelligently combines the lhs and rhs hash values
   */
  constexpr result_type hash_combine(result_type lhs, result_type rhs) const
  {
    result_type combined{lhs};

    combined ^= rhs + 0x9e3779b9 + (combined << 6) + (combined >> 2);

    return combined;
  }

  template <typename return_type = result_type>
  constexpr std::enable_if_t<!std::is_arithmetic<Key>::value, return_type> operator()(
    Key const& key) const
  {
    cudf_assert(false && "IdentityHash does not support this data type");
    return 0;
  }

  template <typename return_type = result_type>
  constexpr std::enable_if_t<std::is_arithmetic<Key>::value, return_type> operator()(
    Key const& key) const
  {
    return static_cast<result_type>(key);
  }

 private:
  uint32_t m_seed{cudf::DEFAULT_HASH_SEED};
};

template <typename Key>
using default_hash = MurmurHash3_32<Key>;
