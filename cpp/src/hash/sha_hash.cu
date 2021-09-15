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

#include <hash/hash_constants.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>
#include <type_traits>

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

CUDA_DEVICE_CALLABLE uint32_t rotate_bits_left(uint32_t x, int8_t r)
{
  // Equivalent to (x << r) | (x >> (32 - r))
  return __funnelshift_l(x, x, r);
}

CUDA_DEVICE_CALLABLE uint32_t rotate_bits_right(uint32_t x, int8_t r)
{
  // Equivalent to (x >> r) | (x << (32 - r))
  return __funnelshift_r(x, x, r);
}

// Swap the endianness of a 32 bit value
CUDA_DEVICE_CALLABLE uint32_t swap_endian(uint32_t x)
{
  // The selector 0x0123 reverses the byte order
  return __byte_perm(x, 0, 0x0123);
}

// Swap the endianness of a 64 bit value
// There is no CUDA intrinsic for permuting bytes in 64 bit integers
CUDA_DEVICE_CALLABLE uint64_t swap_endian(uint64_t x)
{
  // Reverse the endianness of each 32 bit section
  uint32_t low_bits  = swap_endian(static_cast<uint32_t>(x));
  uint32_t high_bits = swap_endian(static_cast<uint32_t>(x >> 32));
  // Reassemble a 64 bit result, swapping the low bits and high bits
  return (static_cast<uint64_t>(low_bits) << 32) | (static_cast<uint64_t>(high_bits));
};

}  // namespace

template <typename HasherT>
struct SHAHash {
  // Number of bytes used for the message length
  static constexpr auto message_length_size = 8;

  /**
   * @brief Execute SHA on input data chunks.
   *
   * This accepts arbitrary data, handles it as bytes, and calls the hash step
   * when the buffer is filled up to message_chunk_size bytes.
   */
  template <typename TKey, typename Hasher = HasherT>
  void CUDA_DEVICE_CALLABLE process(TKey const& key,
                                    typename Hasher::sha_intermediate_data* hash_state) const
  {
    uint32_t const len  = sizeof(TKey);
    uint8_t const* data = reinterpret_cast<uint8_t const*>(&key);
    hash_state->message_length += len;

    if (hash_state->buffer_length + len < Hasher::message_chunk_size) {
      std::memcpy(hash_state->buffer + hash_state->buffer_length, data, len);
      hash_state->buffer_length += len;
    } else {
      uint32_t copylen = Hasher::message_chunk_size - hash_state->buffer_length;

      std::memcpy(hash_state->buffer + hash_state->buffer_length, data, copylen);
      Hasher::hash_step(hash_state);

      while (len > Hasher::message_chunk_size + copylen) {
        std::memcpy(hash_state->buffer, data + copylen, Hasher::message_chunk_size);
        Hasher::hash_step(hash_state);
        copylen += Hasher::message_chunk_size;
      }

      std::memcpy(hash_state->buffer, data + copylen, len - copylen);
      hash_state->buffer_length = len - copylen;
    }
  }

  /**
   * @brief Finalize SHA element processing.
   *
   * This method fills the remainder of the message buffer with zeros, appends
   * the message length (in another step of the hash, if needed), and performs
   * the final hash step.
   */
  template <typename Hasher = HasherT>
  void CUDA_DEVICE_CALLABLE finalize(typename Hasher::sha_intermediate_data* hash_state,
                                     char* result_location)
  {
    // Message length in bits.
    uint64_t const message_length_in_bits = (static_cast<uint64_t>(hash_state->message_length))
                                            << 3;

    // Add a one bit flag to signal the end of the message
    constexpr uint8_t end_of_message = 0x80;
    // 1 byte for the end of the message flag
    constexpr int end_of_message_size = 1;

    thrust::fill_n(thrust::seq,
                   hash_state->buffer + hash_state->buffer_length,
                   end_of_message_size,
                   end_of_message);

    // SHA-512 uses a 128-bit message length instead of a 64-bit message length
    // but this code does not support messages with lengths exceeding UINT64_MAX
    // bits. We always pad the upper 64 bits with zeros.
    constexpr auto message_length_supported_size = sizeof(message_length_in_bits);

    if (hash_state->buffer_length + message_length_size + end_of_message_size <=
        Hasher::message_chunk_size) {
      // Fill the remainder of the buffer with zeros up to the space reserved
      // for the message length. The message length fits in this hash step.
      thrust::fill_n(thrust::seq,
                     hash_state->buffer + hash_state->buffer_length + 1,
                     (Hasher::message_chunk_size - message_length_supported_size -
                      end_of_message_size - hash_state->buffer_length),
                     0x00);
    } else {
      // Fill the remainder of the buffer with zeros. The message length doesn't
      // fit and will be processed in a subsequent hash step comprised of only
      // zeros followed by the message length.
      thrust::fill_n(thrust::seq,
                     hash_state->buffer + hash_state->buffer_length + end_of_message_size,
                     (Hasher::message_chunk_size - hash_state->buffer_length),
                     0x00);
      Hasher::hash_step(hash_state);

      thrust::fill_n(
        thrust::seq, hash_state->buffer, Hasher::message_chunk_size - message_length_size, 0x00);
    }

    // Convert the 64-bit message length from little-endian to big-endian.
    uint64_t const full_length_flipped = swap_endian(message_length_in_bits);
    std::memcpy(hash_state->buffer + Hasher::message_chunk_size - message_length_supported_size,
                reinterpret_cast<uint8_t const*>(&full_length_flipped),
                message_length_supported_size);
    Hasher::hash_step(hash_state);

    // Each byte in the word generates two bytes in the hexadecimal string digest.
    auto constexpr num_words = Hasher::digest_size / (2 * sizeof(typename Hasher::sha_word_type));
#pragma unroll
    for (int i = 0; i < num_words; i++) {
      // Convert word representation from big-endian to little-endian.
      typename Hasher::sha_word_type flipped = swap_endian(hash_state->hash_value[i]);
      if constexpr (std::is_same_v<typename Hasher::sha_word_type, uint32_t>) {
        uint32ToLowercaseHexString(flipped, result_location + (8 * i));
      } else if constexpr (std::is_same_v<typename Hasher::sha_word_type, uint64_t>) {
        uint32_t low_bits = static_cast<uint32_t>(flipped);
        uint32ToLowercaseHexString(low_bits, result_location + (16 * i));
        uint32_t high_bits = static_cast<uint32_t>(flipped >> 32);
        uint32ToLowercaseHexString(high_bits, result_location + (16 * i) + 8);
      } else {
        cudf_assert(false && "Unsupported SHA word type.");
      }
    }
  };

  template <typename T,
            typename Hasher                            = HasherT,
            typename std::enable_if_t<is_chrono<T>()>* = nullptr>
  void CUDA_DEVICE_CALLABLE operator()(column_device_view col,
                                       size_type row_index,
                                       typename Hasher::sha_intermediate_data* hash_state) const
  {
    cudf_assert(false && "SHA Unsupported chrono type column");
  }

  template <
    typename T,
    typename Hasher                                                                     = HasherT,
    typename std::enable_if_t<!is_fixed_width<T>() && !std::is_same_v<T, string_view>>* = nullptr>
  void CUDA_DEVICE_CALLABLE operator()(column_device_view col,
                                       size_type row_index,
                                       typename Hasher::sha_intermediate_data* hash_state) const
  {
    cudf_assert(false && "SHA Unsupported non-fixed-width type column");
  }

  template <typename T,
            typename Hasher                                    = HasherT,
            typename std::enable_if_t<is_floating_point<T>()>* = nullptr>
  void CUDA_DEVICE_CALLABLE operator()(column_device_view col,
                                       size_type row_index,
                                       typename Hasher::sha_intermediate_data* hash_state) const
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
            typename Hasher                             = HasherT,
            typename std::enable_if_t<is_fixed_width<T>() && !is_floating_point<T>() &&
                                      !is_chrono<T>()>* = nullptr>
  void CUDA_DEVICE_CALLABLE operator()(column_device_view col,
                                       size_type row_index,
                                       typename Hasher::sha_intermediate_data* hash_state) const
  {
    process(col.element<T>(row_index), hash_state);
  }

  template <typename T,
            typename Hasher                                            = HasherT,
            typename std::enable_if_t<std::is_same_v<T, string_view>>* = nullptr>
  void CUDA_DEVICE_CALLABLE operator()(column_device_view col,
                                       size_type row_index,
                                       typename Hasher::sha_intermediate_data* hash_state) const
  {
    string_view key     = col.element<string_view>(row_index);
    uint32_t const len  = static_cast<uint32_t>(key.size_bytes());
    uint8_t const* data = reinterpret_cast<uint8_t const*>(key.data());
    hash_state->message_length += len;

    if (hash_state->buffer_length + len < Hasher::message_chunk_size) {
      // If the buffer will not be filled by this data, we copy the new data into
      // the buffer but do not trigger a hash step yet.
      std::memcpy(hash_state->buffer + hash_state->buffer_length, data, len);
      hash_state->buffer_length += len;
    } else {
      // The buffer will be filled by this data. Copy a chunk of the data to fill
      // the buffer and trigger a hash step.
      uint32_t copylen = Hasher::message_chunk_size - hash_state->buffer_length;
      std::memcpy(hash_state->buffer + hash_state->buffer_length, data, copylen);
      Hasher::hash_step(hash_state);

      // Take buffer-sized chunks of the data and do a hash step on each chunk.
      while (len > Hasher::message_chunk_size + copylen) {
        std::memcpy(hash_state->buffer, data + copylen, Hasher::message_chunk_size);
        Hasher::hash_step(hash_state);
        copylen += Hasher::message_chunk_size;
      }

      // The remaining data chunk does not fill the buffer. We copy the data into
      // the buffer but do not trigger a hash step yet.
      std::memcpy(hash_state->buffer, data + copylen, len - copylen);
      hash_state->buffer_length = len - copylen;
    }
  }
};

struct SHA1Hash : SHAHash<SHA1Hash> {
  // Intermediate data type storing the hash state
  using sha_intermediate_data = sha1_intermediate_data;
  // The word type used by this hash function
  using sha_word_type = sha1_word_type;
  // Number of bytes processed in each hash step
  static constexpr auto message_chunk_size = 64;
  // Digest size in bytes
  static constexpr auto digest_size = 40;

  /**
   * @brief Core SHA-1 algorithm implementation. Processes a single 512-bit chunk,
   * updating the hash value so far. Does not zero out the buffer contents.
   */
  static void __device__ hash_step(sha_intermediate_data* hash_state)
  {
    sha_word_type A = hash_state->hash_value[0];
    sha_word_type B = hash_state->hash_value[1];
    sha_word_type C = hash_state->hash_value[2];
    sha_word_type D = hash_state->hash_value[3];
    sha_word_type E = hash_state->hash_value[4];

    sha_word_type words[80];

    // Word size in bytes
    constexpr auto word_size = sizeof(sha_word_type);

    // The 512-bit message buffer fills the first 16 words.
    for (int i = 0; i < 16; i++) {
      sha_word_type buffer_element_as_int;
      std::memcpy(&buffer_element_as_int, hash_state->buffer + (i * word_size), word_size);
      // Convert word representation from little-endian to big-endian.
      words[i] = swap_endian(buffer_element_as_int);
    }

    // The rest of the 80 words are generated from the first 16 words.
    for (int i = 16; i < 80; i++) {
      sha_word_type temp = words[i - 3] ^ words[i - 8] ^ words[i - 14] ^ words[i - 16];
      words[i]           = rotate_bits_left(temp, 1);
    }

    for (int i = 0; i < 80; i++) {
      sha_word_type F;
      sha_word_type temp;
      sha_word_type k;
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
      temp = rotate_bits_left(A, 5) + F + E + k + words[i];
      E    = D;
      D    = C;
      C    = rotate_bits_left(B, 30);
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
};

struct SHA256Hash : SHAHash<SHA256Hash> {
  using sha_intermediate_data = sha256_intermediate_data;
  using sha_word_type         = sha256_word_type;
  // Number of bytes processed in each hash step
  static constexpr auto message_chunk_size = 64;
  // Digest size in bytes
  static constexpr auto digest_size = 64;

  /**
   * @brief Core SHA-256 algorithm implementation. Processes a single 512-bit chunk,
   * updating the hash value so far. Does not zero out the buffer contents.
   */
  static void __device__ hash_step(sha_intermediate_data* hash_state)
  {
    sha_word_type A = hash_state->hash_value[0];
    sha_word_type B = hash_state->hash_value[1];
    sha_word_type C = hash_state->hash_value[2];
    sha_word_type D = hash_state->hash_value[3];
    sha_word_type E = hash_state->hash_value[4];
    sha_word_type F = hash_state->hash_value[5];
    sha_word_type G = hash_state->hash_value[6];
    sha_word_type H = hash_state->hash_value[7];

    sha_word_type words[64];

    // Word size in bytes
    constexpr auto word_size = sizeof(sha_word_type);

    // The 512-bit message buffer fills the first 16 words.
    for (int i = 0; i < 16; i++) {
      sha_word_type buffer_element_as_int;
      std::memcpy(&buffer_element_as_int, hash_state->buffer + (i * word_size), word_size);
      // Convert word representation from little-endian to big-endian.
      words[i] = swap_endian(buffer_element_as_int);
    }

    // The rest of the 64 words are generated from the first 16 words.
    for (int i = 16; i < 64; i++) {
      sha_word_type s0 = rotate_bits_right(words[i - 15], 7) ^
                         rotate_bits_right(words[i - 15], 18) ^ (words[i - 15] >> 3);
      sha_word_type s1 = rotate_bits_right(words[i - 2], 17) ^ rotate_bits_right(words[i - 2], 19) ^
                         (words[i - 2] >> 10);
      words[i] = words[i - 16] + s0 + words[i - 7] + s1;
    }

    for (int i = 0; i < 64; i++) {
      sha_word_type const s1 =
        rotate_bits_right(E, 6) ^ rotate_bits_right(E, 11) ^ rotate_bits_right(E, 25);
      sha_word_type const ch    = (E & F) ^ ((~E) & G);
      sha_word_type const temp1 = H + s1 + ch + sha256_hash_constants[i] + words[i];
      sha_word_type const s0 =
        rotate_bits_right(A, 2) ^ rotate_bits_right(A, 13) ^ rotate_bits_right(A, 22);
      sha_word_type const maj   = (A & B) ^ (A & C) ^ (B & C);
      sha_word_type const temp2 = s0 + maj;

      H = G;
      G = F;
      F = E;
      E = D + temp1;
      D = C;
      C = B;
      B = A;
      A = temp1 + temp2;
    }

    hash_state->hash_value[0] += A;
    hash_state->hash_value[1] += B;
    hash_state->hash_value[2] += C;
    hash_state->hash_value[3] += D;
    hash_state->hash_value[4] += E;
    hash_state->hash_value[5] += F;
    hash_state->hash_value[6] += G;
    hash_state->hash_value[7] += H;

    hash_state->buffer_length = 0;
  }
};

/**
 * @brief Call a SHA-1 or SHA-2 hash function on a table view.
 *
 * @tparam Hasher The struct used for computing SHA hashes.
 *
 * @param input The input table.
 * results.
 * @param empty_result A string representing the expected result for empty inputs.
 * @param stream CUDA stream on which memory may be allocated if the memory
 * resource supports streams.
 * @param mr Memory resource to use for the device memory allocation
 * @return A new column with the computed hash function result.
 */
template <typename Hasher>
std::unique_ptr<column> sha_hash(table_view const& input,
                                 string_scalar const& empty_result,
                                 cudaStream_t stream,
                                 rmm::mr::device_memory_resource* mr)
{
  if (input.num_columns() == 0 || input.num_rows() == 0) {
    // Return the hash of a zero-length input.
    auto output = make_column_from_scalar(empty_result, input.num_rows(), stream, mr);
    return output;
  }

  // Accepts string and fixed width columns.
  // TODO: Accept single layer list columns holding those types.
  CUDF_EXPECTS(
    std::all_of(input.begin(), input.end(), [](auto col) { return sha_type_check(col.type()); }),
    "SHA unsupported column type");

  // Result column allocation and creation
  auto begin = thrust::make_constant_iterator(Hasher::digest_size);
  auto offsets_column =
    cudf::strings::detail::make_offsets_child_column(begin, begin + input.num_rows(), stream, mr);

  auto chars_column =
    strings::detail::create_chars_child_column(input.num_rows() * Hasher::digest_size, stream, mr);
  auto chars_view = chars_column->mutable_view();
  auto d_chars    = chars_view.data<char>();

  rmm::device_buffer null_mask{0, stream, mr};

  auto const device_input = table_device_view::create(input, stream);

  // Hash each row, hashing each element sequentially left to right
  thrust::for_each(rmm::exec_policy(stream),
                   thrust::make_counting_iterator(0),
                   thrust::make_counting_iterator(input.num_rows()),
                   [d_chars, device_input = *device_input] __device__(auto row_index) {
                     typename Hasher::sha_intermediate_data hash_state;
                     Hasher hasher = Hasher{};
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
                     hasher.finalize(&hash_state, d_chars + (row_index * Hasher::digest_size));
                   });

  return make_strings_column(
    input.num_rows(), std::move(offsets_column), std::move(chars_column), 0, std::move(null_mask));
}

std::unique_ptr<column> sha1_hash(table_view const& input,
                                  cudaStream_t stream,
                                  rmm::mr::device_memory_resource* mr)
{
  string_scalar const empty_result("da39a3ee5e6b4b0d3255bfef95601890afd80709");
  return sha_hash<SHA1Hash>(input, empty_result, stream, mr);
}

std::unique_ptr<column> sha256_hash(table_view const& input,
                                    bool truncate_output,
                                    cudaStream_t stream,
                                    rmm::mr::device_memory_resource* mr)
{
  string_scalar const empty_result(
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  return sha_hash<SHA256Hash>(input, empty_result, stream, mr);
}

std::unique_ptr<column> sha512_hash(table_view const& input,
                                    bool truncate_output,
                                    cudaStream_t stream,
                                    rmm::mr::device_memory_resource* mr)
{
  string_scalar const empty_result(
    "cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce47d0d13c5d85f2b0ff8318d2877eec"
    "2f63b931bd47417a81a538327af927da3e");
  // return sha_hash<SHA512Hash, digest_size>(input, empty_result, stream, mr);
  return nullptr;
}

}  // namespace detail
}  // namespace cudf
