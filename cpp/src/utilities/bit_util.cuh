
/*
 * Copyright 2018 BlazingDB, Inc.
 *     Copyright 2018 Alexander Ocsa <alexander@blazingdb.com>
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

namespace gdf {
namespace util {

static constexpr int ValidSize = 32;
using ValidType = uint32_t;


// Instead of this function, use get_number_of_bytes_for_valid from gdf/utils.h
//__host__ __device__ __forceinline__
//  size_t
//  valid_size(size_t column_length)
//{
//  const size_t n_ints = (column_length / ValidSize) + ((column_length % ValidSize) ? 1 : 0);
//  return n_ints * sizeof(ValidType);
//}

// Instead of this function, use gdf_is_valid from gdf/utils.h
///__host__ __device__ __forceinline__ bool get_bit(const gdf_valid_type* const bits, size_t i)
///{
///  return  bits == nullptr? true :  bits[i >> size_t(3)] & (1 << (i & size_t(7)));
///}

__host__ __device__ __forceinline__
  uint8_t
  byte_bitmask(size_t i)
{
  static uint8_t kBitmask[8] = { 1, 2, 4, 8, 16, 32, 64, 128 };
  return kBitmask[i];
}

__host__ __device__ __forceinline__
  uint8_t
  flipped_bitmask(size_t i)
{
  static uint8_t kFlippedBitmask[] = { 254, 253, 251, 247, 239, 223, 191, 127 };
  return kFlippedBitmask[i];
}

__host__ __device__ __forceinline__ void turn_bit_on(uint8_t* const bits, size_t i)
{
  bits[i / 8] |= byte_bitmask(i % 8);
}

__host__ __device__ __forceinline__ void turn_bit_off(uint8_t* const bits, size_t i)
{
  bits[i / 8] &= flipped_bitmask(i % 8);
}

__host__ __device__ __forceinline__ size_t last_byte_index(size_t column_size)
{
  return (column_size + 8 - 1) / 8;
}

static inline std::string chartobin(gdf_valid_type c, int size = 8)
{
  std::string bin;
  bin.resize(size);
  bin[0] = 0;
  int i;
  for (i = size - 1; i >= 0; i--)
  {
      bin[i] = (c % 2) + '0';
      c /= 2;
  }
  return bin;
}

static inline std::string gdf_valid_to_str(gdf_valid_type *valid, size_t column_size)
{
  size_t n_bytes = get_number_of_bytes_for_valid(column_size);
  std::string response;
  for (size_t i = 0; i < n_bytes; i++)
  {
      size_t length = (n_bytes != i + 1) ? GDF_VALID_BITSIZE : (column_size - GDF_VALID_BITSIZE * (n_bytes - 1));
      auto result = chartobin(valid[i], length);
      response += std::string(result);
  }
  return response;
}

} // namespace util
} // namespace gdf
