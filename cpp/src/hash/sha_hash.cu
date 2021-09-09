/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/hashing.hpp>
#include <cudf/detail/utilities/hash_functions.cuh>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/table/table_device_view.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>

namespace cudf {
namespace detail {

namespace {

// SHA supported leaf data type check
bool sha_type_check(data_type dt)
{
  return !is_chrono(dt) && (is_fixed_width(dt) || (dt.id() == type_id::STRING));
}

}  // namespace

struct SHA1Hash {
  CUDA_DEVICE_CALLABLE uint32_t rotl32(uint32_t x, int8_t r) const
  {
    // Equivalent to (x << r) | (x >> (32 - r))
    return __funnelshift_l(x, x, r);
  }

  /**
   * @brief Core SHA-1 algorithm implementation. Processes a single 512-bit chunk,
   * updating the hash value so far. Does not zero out the buffer contents.
   */
  void __device__ hash_step(sha1_intermediate_data* hash_state) const
  {
    uint32_t A = hash_state->hash_value[0];
    uint32_t B = hash_state->hash_value[1];
    uint32_t C = hash_state->hash_value[2];
    uint32_t D = hash_state->hash_value[3];
    uint32_t E = hash_state->hash_value[4];

    uint32_t words[80];

    // The 512-bit message buffer fills the first 16 words.
    for (int i = 0; i < 16; i++) {
      uint32_t buffer_element_as_int;
      std::memcpy(&buffer_element_as_int, hash_state->buffer + (i * 4), 4);
      // Convert word representation from little-endian to big-endian.
      words[i] = __byte_perm(buffer_element_as_int, 0, 0x0123);
    }

    // The rest of the 80 words are generated from the first 16 words.
    for (int i = 16; i < 80; i++) {
      uint32_t temp = words[i - 3] ^ words[i - 8] ^ words[i - 14] ^ words[i - 16];
      words[i]      = rotl32(temp, 1);
    }

#pragma unroll
    for (int i = 0; i < 80; i++) {
      uint32_t F;
      uint32_t temp;
      uint32_t k;
      switch (i / 20) {
        case 0:
          F = D ^ (B & (C ^ D));
          k = 0x5a827999;
          break;
        case 1:
          F = B ^ C ^ D;
          k = 0x6ed9eba1;
          break;
        case 2:
          F = (B & C) | (B & D) | (C & D);
          k = 0x8f1bbcdc;
          break;
        case 3:
          F = B ^ C ^ D;
          k = 0xca62c1d6;
          break;
      }
      temp = rotl32(A, 5) + F + E + k + words[i];
      E    = D;
      D    = C;
      C    = rotl32(B, 30);
      B    = A;
      A    = temp;
    }

    hash_state->hash_value[0] += A;
    hash_state->hash_value[1] += B;
    hash_state->hash_value[2] += C;
    hash_state->hash_value[3] += D;
    hash_state->hash_value[4] += E;

    hash_state->buffer_length = 0;
  }

  /**
   * @brief Core SHA1 element processing function
   */
  template <typename TKey>
  void __device__ process(TKey const& key, sha1_intermediate_data* hash_state) const
  {
    uint32_t const len  = sizeof(TKey);
    uint8_t const* data = reinterpret_cast<uint8_t const*>(&key);
    hash_state->message_length += len;

    // 64 bytes are processed in each hash step
    constexpr int sha1_chunk_size = 64;
    if (hash_state->buffer_length + len < sha1_chunk_size) {
      std::memcpy(hash_state->buffer + hash_state->buffer_length, data, len);
      hash_state->buffer_length += len;
    } else {
      uint32_t copylen = sha1_chunk_size - hash_state->buffer_length;

      std::memcpy(hash_state->buffer + hash_state->buffer_length, data, copylen);
      hash_step(hash_state);

      while (len > sha1_chunk_size + copylen) {
        std::memcpy(hash_state->buffer, data + copylen, sha1_chunk_size);
        hash_step(hash_state);
        copylen += sha1_chunk_size;
      }

      std::memcpy(hash_state->buffer, data + copylen, len - copylen);
      hash_state->buffer_length = len - copylen;
    }
  }

  void __device__ finalize(sha1_intermediate_data* hash_state, char* result_location) const
  {
    // Message length in bits
    auto const full_length = (static_cast<uint64_t>(hash_state->message_length)) << 3;

    // Add a one bit flag to signal the end of the message
    thrust::fill_n(thrust::seq, hash_state->buffer + hash_state->buffer_length, 1, 0x80);

    // 64 bytes are processed in each hash step
    constexpr int sha1_chunk_size = 64;
    // 8 bytes for the total message length, appended to the end of the last chunk processed
    constexpr int message_length_size = 8;
    // 1 byte for the end of the message flag
    constexpr int end_of_message_size = 1;
    if (hash_state->buffer_length + message_length_size + end_of_message_size <= sha1_chunk_size) {
      // Fill the remainder of the buffer with zeros
      thrust::fill_n(
        thrust::seq,
        hash_state->buffer + hash_state->buffer_length + 1,
        (sha1_chunk_size - message_length_size - end_of_message_size - hash_state->buffer_length),
        0x00);
    } else {
      thrust::fill_n(thrust::seq,
                     hash_state->buffer + hash_state->buffer_length + 1,
                     (sha1_chunk_size - hash_state->buffer_length),
                     0x00);
      hash_step(hash_state);

      thrust::fill_n(thrust::seq, hash_state->buffer, sha1_chunk_size - message_length_size, 0x00);
    }

    // Convert the 64-bit message length from little-endian to big-endian.
    // There is currently no CUDA intrinsic for permuting bytes in 64 bit integers.
    auto uint64_swap_endian = [](uint64_t x) -> uint64_t {
      // Reverse the endianness of each 32 bit section
      uint32_t low_bits  = __byte_perm(x, 0, 0x123);
      uint32_t high_bits = __byte_perm(x >> 32, 0, 0x123);
      // Reassemble a 64 bit result
      uint64_t y = (static_cast<uint64_t>(low_bits) << 32) | (static_cast<uint64_t>(high_bits));
      return y;
    };

    auto const full_length_flipped = uint64_swap_endian(full_length);
    std::memcpy(hash_state->buffer + sha1_chunk_size - message_length_size,
                reinterpret_cast<uint8_t const*>(&full_length_flipped),
                message_length_size);
    hash_step(hash_state);
    // std::memcpy(hash_state->hash_value, hash_state->buffer, 160);

#pragma unroll
    for (int i = 0; i < 5; ++i) {
      // Convert word representation from big-endian to little-endian.
      uint32_t flipped = __byte_perm(hash_state->hash_value[i], 0, 0x0123);
      uint32ToLowercaseHexString(flipped, result_location + (8 * i));
    }
  }

  template <typename T, typename std::enable_if_t<is_chrono<T>()>* = nullptr>
  void __device__ operator()(column_device_view col,
                             size_type row_index,
                             sha1_intermediate_data* hash_state) const
  {
    cudf_assert(false && "SHA-1 Unsupported chrono type column");
  }

  template <typename T, typename std::enable_if_t<!is_fixed_width<T>()>* = nullptr>
  void __device__ operator()(column_device_view col,
                             size_type row_index,
                             sha1_intermediate_data* hash_state) const
  {
    cudf_assert(false && "SHA-1 Unsupported non-fixed-width type column");
  }

  template <typename T, typename std::enable_if_t<is_floating_point<T>()>* = nullptr>
  void __device__ operator()(column_device_view col,
                             size_type row_index,
                             sha1_intermediate_data* hash_state) const
  {
    T const& key = col.element<T>(row_index);
    if (isnan(key)) {
      T nan = std::numeric_limits<T>::quiet_NaN();
      process(nan, hash_state);
    } else if (key == T{0.0}) {
      process(T{0.0}, hash_state);
    } else {
      process(key, hash_state);
    }
  }

  template <typename T,
            typename std::enable_if_t<is_fixed_width<T>() && !is_floating_point<T>() &&
                                      !is_chrono<T>()>* = nullptr>
  void CUDA_DEVICE_CALLABLE operator()(column_device_view col,
                                       size_type row_index,
                                       sha1_intermediate_data* hash_state) const
  {
    process(col.element<T>(row_index), hash_state);
  }
};

template <>
void CUDA_DEVICE_CALLABLE SHA1Hash::operator()<string_view>(
  column_device_view col, size_type row_index, sha1_intermediate_data* hash_state) const
{
  string_view key     = col.element<string_view>(row_index);
  uint32_t const len  = static_cast<uint32_t>(key.size_bytes());
  uint8_t const* data = reinterpret_cast<uint8_t const*>(key.data());
  hash_state->message_length += len;

  // 64 bytes are processed in each hash step
  constexpr int sha1_chunk_size = 64;
  if (hash_state->buffer_length + len < sha1_chunk_size) {
    // If the buffer will not be filled by this data, we copy the new data into
    // the buffer but do not trigger a hash step yet.
    std::memcpy(hash_state->buffer + hash_state->buffer_length, data, len);
    hash_state->buffer_length += len;
  } else {
    // The buffer will be filled by this data. Copy a chunk of the data to fill
    // the buffer and trigger a hash step.
    uint32_t copylen = sha1_chunk_size - hash_state->buffer_length;
    std::memcpy(hash_state->buffer + hash_state->buffer_length, data, copylen);
    hash_step(hash_state);

    // Take buffer-sized chunks of the data and do a hash step on each chunk.
    while (len > sha1_chunk_size + copylen) {
      std::memcpy(hash_state->buffer, data + copylen, sha1_chunk_size);
      hash_step(hash_state);
      copylen += sha1_chunk_size;
    }

    // The remaining data chunk does not fill the buffer. We copy the data into
    // the buffer but do not trigger a hash step yet.
    std::memcpy(hash_state->buffer, data + copylen, len - copylen);
    hash_state->buffer_length = len - copylen;
  }
}

std::unique_ptr<column> sha1_hash(table_view const& input,
                                  cudaStream_t stream,
                                  rmm::mr::device_memory_resource* mr)
{
  if (input.num_columns() == 0 || input.num_rows() == 0) {
    // Return the SHA-1 hash of a zero-length input.
    const string_scalar string_160bit("da39a3ee5e6b4b0d3255bfef95601890afd80709");
    auto output = make_column_from_scalar(string_160bit, input.num_rows(), stream, mr);
    return output;
  }

  // Accepts string and fixed width columns.
  // TODO: Accept single layer list columns holding those types.
  CUDF_EXPECTS(
    std::all_of(input.begin(), input.end(), [](auto col) { return sha_type_check(col.type()); }),
    "SHA-1 unsupported column type");

  // Result column allocation and creation
  auto begin = thrust::make_constant_iterator(40);
  auto offsets_column =
    cudf::strings::detail::make_offsets_child_column(begin, begin + input.num_rows(), stream, mr);

  auto chars_column = strings::detail::create_chars_child_column(input.num_rows() * 40, stream, mr);
  auto chars_view   = chars_column->mutable_view();
  auto d_chars      = chars_view.data<char>();

  rmm::device_buffer null_mask{0, stream, mr};

  auto const device_input = table_device_view::create(input, stream);

  // Hash each row, hashing each element sequentially left to right
  thrust::for_each(rmm::exec_policy(stream),
                   thrust::make_counting_iterator(0),
                   thrust::make_counting_iterator(input.num_rows()),
                   [d_chars, device_input = *device_input] __device__(auto row_index) {
                     sha1_intermediate_data hash_state;
                     SHA1Hash hasher = SHA1Hash{};
                     for (int col_index = 0; col_index < device_input.num_columns(); col_index++) {
                       if (device_input.column(col_index).is_valid(row_index)) {
                         cudf::type_dispatcher<dispatch_storage_type>(
                           device_input.column(col_index).type(),
                           hasher,
                           device_input.column(col_index),
                           row_index,
                           &hash_state);
                       }
                     }
                     hasher.finalize(&hash_state, d_chars + (row_index * 40));
                   });

  return make_strings_column(
    input.num_rows(), std::move(offsets_column), std::move(chars_column), 0, std::move(null_mask));
}

std::unique_ptr<column> sha256_hash(table_view const& input,
                                    bool truncate_output,
                                    cudaStream_t stream,
                                    rmm::mr::device_memory_resource* mr)
{
  return nullptr;
}

std::unique_ptr<column> sha512_hash(table_view const& input,
                                    bool truncate_output,
                                    cudaStream_t stream,
                                    rmm::mr::device_memory_resource* mr)
{
  return nullptr;
}

}  // namespace detail
}  // namespace cudf
