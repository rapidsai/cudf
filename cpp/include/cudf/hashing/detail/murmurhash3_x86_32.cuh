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

#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/hashing.hpp>
#include <cudf/hashing/detail/hash_functions.cuh>
#include <cudf/lists/list_view.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/structs/struct_view.hpp>
#include <cudf/types.hpp>

#include <cstddef>

namespace cudf::hashing::detail {

// MurmurHash3_x86_32 implementation from
// https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp
//-----------------------------------------------------------------------------
// MurmurHash3 was written by Austin Appleby, and is placed in the public
// domain. The author hereby disclaims copyright to this source code.
// Note - The x86 and x64 versions do _not_ produce the same results, as the
// algorithms are optimized for their respective platforms. You can still
// compile and run any of them on any platform, but your performance with the
// non-native version will be less than optimal.
template <typename Key>
struct MurmurHash3_x86_32 {
  using result_type = hash_value_type;

  constexpr MurmurHash3_x86_32() = default;
  constexpr MurmurHash3_x86_32(uint32_t seed) : m_seed(seed) {}

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
    return compute(normalize_nans_and_zeros(key));
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
        k1 = rotate_bits_left(k1, rot_c1);
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
      k1 = rotate_bits_left(k1, rot_c1);
      k1 *= c2;
      h ^= k1;
      h = rotate_bits_left(h, rot_c2);
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
hash_value_type __device__ inline MurmurHash3_x86_32<bool>::operator()(bool const& key) const
{
  return compute(static_cast<uint8_t>(key));
}

template <>
hash_value_type __device__ inline MurmurHash3_x86_32<float>::operator()(float const& key) const
{
  return compute(normalize_nans_and_zeros(key));
}

template <>
hash_value_type __device__ inline MurmurHash3_x86_32<double>::operator()(double const& key) const
{
  return compute(normalize_nans_and_zeros(key));
}

template <>
hash_value_type __device__ inline MurmurHash3_x86_32<cudf::string_view>::operator()(
  cudf::string_view const& key) const
{
  auto const data = reinterpret_cast<std::byte const*>(key.data());
  auto const len  = key.size_bytes();
  return compute_bytes(data, len);
}

template <>
hash_value_type __device__ inline MurmurHash3_x86_32<numeric::decimal32>::operator()(
  numeric::decimal32 const& key) const
{
  return compute(key.value());
}

template <>
hash_value_type __device__ inline MurmurHash3_x86_32<numeric::decimal64>::operator()(
  numeric::decimal64 const& key) const
{
  return compute(key.value());
}

template <>
hash_value_type __device__ inline MurmurHash3_x86_32<numeric::decimal128>::operator()(
  numeric::decimal128 const& key) const
{
  return compute(key.value());
}

template <>
hash_value_type __device__ inline MurmurHash3_x86_32<cudf::list_view>::operator()(
  cudf::list_view const& key) const
{
  CUDF_UNREACHABLE("List column hashing is not supported");
}

template <>
hash_value_type __device__ inline MurmurHash3_x86_32<cudf::struct_view>::operator()(
  cudf::struct_view const& key) const
{
  CUDF_UNREACHABLE("Direct hashing of struct_view is not supported");
}

}  // namespace cudf::hashing::detail
