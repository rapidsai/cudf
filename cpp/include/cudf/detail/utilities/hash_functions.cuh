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

#include <cstddef>

#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/utilities/assert.cuh>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/hashing.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/types.hpp>

#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/find.h>
#include <thrust/iterator/reverse_iterator.h>
#include <thrust/pair.h>
#include <thrust/reverse.h>

namespace cudf {
namespace detail {

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

// Swap the endianness of a 32 bit value
__device__ inline uint32_t swap_endian(uint32_t x)
{
  // The selector 0x0123 reverses the byte order
  return __byte_perm(x, 0, 0x0123);
}

// Swap the endianness of a 64 bit value
// There is no CUDA intrinsic for permuting bytes in 64 bit integers
__device__ inline uint64_t swap_endian(uint64_t x)
{
  // Reverse the endianness of each 32 bit section
  uint32_t low_bits  = swap_endian(static_cast<uint32_t>(x));
  uint32_t high_bits = swap_endian(static_cast<uint32_t>(x >> 32));
  // Reassemble a 64 bit result, swapping the low bits and high bits
  return (static_cast<uint64_t>(low_bits) << 32) | (static_cast<uint64_t>(high_bits));
};

template <int capacity, typename hash_step_callable>
struct hash_circular_buffer {
  uint8_t storage[capacity];
  uint8_t* cur;
  int available_space{capacity};
  hash_step_callable hash_step;

  __device__ inline hash_circular_buffer(hash_step_callable hash_step)
    : cur{storage}, hash_step{hash_step}
  {
  }

  __device__ inline void put(uint8_t const* in, int size)
  {
    int copy_start = 0;
    while (size >= available_space) {
      // The buffer will be filled by this chunk of data. Copy a chunk of the
      // data to fill the buffer and trigger a hash step.
      memcpy(cur, in + copy_start, available_space);
      hash_step(storage);
      size -= available_space;
      copy_start += available_space;
      cur             = storage;
      available_space = capacity;
    }
    // The buffer will not be filled by the remaining data. That is, `size >= 0
    // && size < capacity`. We copy the remaining data into the buffer but do
    // not trigger a hash step.
    memcpy(cur, in + copy_start, size);
    cur += size;
    available_space -= size;
  }

  __device__ inline void pad(int const space_to_leave)
  {
    if (space_to_leave > available_space) {
      memset(cur, 0x00, available_space);
      hash_step(storage);
      cur             = storage;
      available_space = capacity;
    }
    memset(cur, 0x00, available_space - space_to_leave);
    cur += available_space - space_to_leave;
    available_space = space_to_leave;
  }

  __device__ inline uint8_t const& operator[](int idx) const { return storage[idx]; }
};

// Get a uint8_t pointer to a column element and its size as a pair.
template <typename Element>
auto __device__ inline get_element_pointer_and_size(Element const& element)
{
  if constexpr (is_fixed_width<Element>() && !is_chrono<Element>()) {
    return thrust::make_pair(reinterpret_cast<uint8_t const*>(&element), sizeof(Element));
  } else {
    CUDF_UNREACHABLE("Unsupported type.");
  }
}

template <>
auto __device__ inline get_element_pointer_and_size(string_view const& element)
{
  return thrust::make_pair(reinterpret_cast<uint8_t const*>(element.data()), element.size_bytes());
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
  // Transform 0xABCD'1234 => 0x0000'ABCD'0000'1234 => 0x0B0A'0D0C'0201'0403
  uint64_t x = num;
  x          = ((x & 0xFFFF'0000u) << 16) | ((x & 0xFFFF));
  x          = ((x & 0x000F'0000'000Fu) << 8) | ((x & 0x00F0'0000'00F0u) >> 4) |
      ((x & 0x0F00'0000'0F00u) << 16) | ((x & 0xF000'0000'F000) << 4);

  // Calculate a mask of ascii value offsets for bytes that contain alphabetical hex digits
  uint64_t offsets = (((x + 0x0606'0606'0606'0606) >> 4) & 0x0101'0101'0101'0101) * 0x27;

  x |= 0x3030'3030'3030'3030;
  x += offsets;
  std::memcpy(destination, reinterpret_cast<uint8_t*>(&x), 8);
}

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
  using result_type = hash_value_type;

  constexpr MurmurHash3_32() = default;
  constexpr MurmurHash3_32(uint32_t seed) : m_seed(seed) {}

  [[nodiscard]] __device__ inline uint32_t fmix32(uint32_t h) const
  {
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
  }

  [[nodiscard]] __device__ inline uint32_t getblock32(std::byte const* data,
                                                      cudf::size_type offset) const
  {
    // Read a 4-byte value from the data pointer as individual bytes for safe
    // unaligned access (very likely for string types).
    auto const block = reinterpret_cast<uint8_t const*>(data + offset);
    return block[0] | (block[1] << 8) | (block[2] << 16) | (block[3] << 24);
  }

  [[nodiscard]] result_type __device__ inline operator()(Key const& key) const
  {
    return compute(detail::normalize_nans_and_zeros(key));
  }

  template <typename T>
  result_type __device__ inline compute(T const& key) const
  {
    return compute_bytes(reinterpret_cast<std::byte const*>(&key), sizeof(T));
  }

  result_type __device__ inline compute_remaining_bytes(std::byte const* data,
                                                        cudf::size_type len,
                                                        cudf::size_type tail_offset,
                                                        result_type h) const
  {
    // Process remaining bytes that do not fill a four-byte chunk.
    uint32_t k1 = 0;
    switch (len % 4) {
      case 3: k1 ^= std::to_integer<uint8_t>(data[tail_offset + 2]) << 16; [[fallthrough]];
      case 2: k1 ^= std::to_integer<uint8_t>(data[tail_offset + 1]) << 8; [[fallthrough]];
      case 1:
        k1 ^= std::to_integer<uint8_t>(data[tail_offset]);
        k1 *= c1;
        k1 = cudf::detail::rotate_bits_left(k1, rot_c1);
        k1 *= c2;
        h ^= k1;
    };
    return h;
  }

  result_type __device__ compute_bytes(std::byte const* data, cudf::size_type const len) const
  {
    constexpr cudf::size_type BLOCK_SIZE = 4;
    cudf::size_type const nblocks        = len / BLOCK_SIZE;
    cudf::size_type const tail_offset    = nblocks * BLOCK_SIZE;
    result_type h                        = m_seed;

    // Process all four-byte chunks.
    for (cudf::size_type i = 0; i < nblocks; i++) {
      uint32_t k1 = getblock32(data, i * BLOCK_SIZE);
      k1 *= c1;
      k1 = cudf::detail::rotate_bits_left(k1, rot_c1);
      k1 *= c2;
      h ^= k1;
      h = cudf::detail::rotate_bits_left(h, rot_c2);
      h = h * 5 + c3;
    }

    h = compute_remaining_bytes(data, len, tail_offset, h);

    // Finalize hash.
    h ^= len;
    h = fmix32(h);
    return h;
  }

 private:
  uint32_t m_seed{cudf::DEFAULT_HASH_SEED};
  static constexpr uint32_t c1     = 0xcc9e2d51;
  static constexpr uint32_t c2     = 0x1b873593;
  static constexpr uint32_t c3     = 0xe6546b64;
  static constexpr uint32_t rot_c1 = 15;
  static constexpr uint32_t rot_c2 = 13;
};

template <>
hash_value_type __device__ inline MurmurHash3_32<bool>::operator()(bool const& key) const
{
  return compute(static_cast<uint8_t>(key));
}

template <>
hash_value_type __device__ inline MurmurHash3_32<float>::operator()(float const& key) const
{
  return compute(detail::normalize_nans_and_zeros(key));
}

template <>
hash_value_type __device__ inline MurmurHash3_32<double>::operator()(double const& key) const
{
  return compute(detail::normalize_nans_and_zeros(key));
}

template <>
hash_value_type __device__ inline MurmurHash3_32<cudf::string_view>::operator()(
  cudf::string_view const& key) const
{
  auto const data = reinterpret_cast<std::byte const*>(key.data());
  auto const len  = key.size_bytes();
  return compute_bytes(data, len);
}

template <>
hash_value_type __device__ inline MurmurHash3_32<numeric::decimal32>::operator()(
  numeric::decimal32 const& key) const
{
  return compute(key.value());
}

template <>
hash_value_type __device__ inline MurmurHash3_32<numeric::decimal64>::operator()(
  numeric::decimal64 const& key) const
{
  return compute(key.value());
}

template <>
hash_value_type __device__ inline MurmurHash3_32<numeric::decimal128>::operator()(
  numeric::decimal128 const& key) const
{
  return compute(key.value());
}

template <>
hash_value_type __device__ inline MurmurHash3_32<cudf::list_view>::operator()(
  cudf::list_view const& key) const
{
  CUDF_UNREACHABLE("List column hashing is not supported");
}

template <>
hash_value_type __device__ inline MurmurHash3_32<cudf::struct_view>::operator()(
  cudf::struct_view const& key) const
{
  CUDF_UNREACHABLE("Direct hashing of struct_view is not supported");
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
  uint32_t m_seed{cudf::DEFAULT_HASH_SEED};
};

template <typename Key>
using default_hash = MurmurHash3_32<Key>;

}  // namespace detail
}  // namespace cudf
