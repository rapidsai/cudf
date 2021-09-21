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
#include <cudf/types.hpp>

#include <hash/hash_constants.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>
#include <type_traits>

#include <thrust/fill.h>
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
  // This function is equivalent to (x << r) | (x >> (32 - r))
  return __funnelshift_l(x, x, r);
}

CUDA_DEVICE_CALLABLE uint32_t rotate_bits_right(uint32_t x, int8_t r)
{
  // This function is equivalent to (x >> r) | (x << (32 - r))
  return __funnelshift_r(x, x, r);
}

CUDA_DEVICE_CALLABLE uint64_t rotate_bits_right(uint64_t x, int8_t r)
{
  return (x >> r) | (x << (64 - r));
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

/**
 * @brief A CRTP helper function
 *
 * https://www.fluentcpp.com/2017/05/19/crtp-helper/
 *
 * Does two things:
 * 1. Makes "crtp" explicit in the inheritance structure of a CRTP base class.
 * 2. Avoids having to `static_cast` in a lot of places
 *
 * @tparam T The derived class in a CRTP hierarchy
 */
template <typename T>
struct crtp {
  CUDA_DEVICE_CALLABLE T& underlying() { return static_cast<T&>(*this); }
  CUDA_DEVICE_CALLABLE T const& underlying() const { return static_cast<T const&>(*this); }
};

template <typename HasherT>
struct SHAHash : public crtp<HasherT> {
  /**
   * @brief Execute SHA on input data chunks.
   *
   * This accepts arbitrary data, handles it as bytes, and calls the hash step
   * when the buffer is filled up to message_chunk_size bytes.
   */
  template <typename Hasher = HasherT>
  void CUDA_DEVICE_CALLABLE process(uint8_t const* data,
                                    uint32_t len,
                                    typename Hasher::sha_intermediate_data& hash_state) const
  {
    hash_state.message_length += len;

    if (hash_state.buffer_length + len < Hasher::message_chunk_size) {
      // The buffer will not be filled by this data. We copy the new data into
      // the buffer but do not trigger a hash step yet.
      std::memcpy(hash_state.buffer + hash_state.buffer_length, data, len);
      hash_state.buffer_length += len;
    } else {
      // The buffer will be filled by this data. Copy a chunk of the data to fill
      // the buffer and trigger a hash step.
      uint32_t copylen = Hasher::message_chunk_size - hash_state.buffer_length;
      std::memcpy(hash_state.buffer + hash_state.buffer_length, data, copylen);
      Hasher::hash_step(hash_state);

      // Take buffer-sized chunks of the data and do a hash step on each chunk.
      while (len > Hasher::message_chunk_size + copylen) {
        std::memcpy(hash_state.buffer, data + copylen, Hasher::message_chunk_size);
        Hasher::hash_step(hash_state);
        copylen += Hasher::message_chunk_size;
      }

      // The remaining data chunk does not fill the buffer. We copy the data into
      // the buffer but do not trigger a hash step yet.
      std::memcpy(hash_state.buffer, data + copylen, len - copylen);
      hash_state.buffer_length = len - copylen;
    }
  }

  template <typename TKey, typename Hasher = HasherT>
  void CUDA_DEVICE_CALLABLE process_key(TKey const& key,
                                        typename Hasher::sha_intermediate_data& hash_state) const
  {
    uint8_t const* data    = reinterpret_cast<uint8_t const*>(&key);
    uint32_t constexpr len = sizeof(TKey);
    process(data, len, hash_state);
  }

  /**
   * @brief Finalize SHA element processing.
   *
   * This method fills the remainder of the message buffer with zeros, appends
   * the message length (in another step of the hash, if needed), and performs
   * the final hash step.
   */
  template <typename Hasher = HasherT>
  void CUDA_DEVICE_CALLABLE finalize(typename Hasher::sha_intermediate_data& hash_state,
                                     char* result_location)
  {
    // Message length in bits.
    uint64_t const message_length_in_bits = (static_cast<uint64_t>(hash_state.message_length)) << 3;
    // Add a one bit flag to signal the end of the message
    uint8_t constexpr end_of_message = 0x80;
    // 1 byte for the end of the message flag
    uint32_t constexpr end_of_message_size = 1;

    thrust::fill_n(thrust::seq,
                   hash_state.buffer + hash_state.buffer_length,
                   end_of_message_size,
                   end_of_message);

    // SHA-512 uses a 128-bit message length instead of a 64-bit message length
    // but this code does not support messages with lengths exceeding UINT64_MAX
    // bits. We always pad the upper 64 bits with zeros.
    auto constexpr message_length_supported_size = sizeof(message_length_in_bits);

    if (hash_state.buffer_length + Hasher::message_length_size + end_of_message_size <=
        Hasher::message_chunk_size) {
      // Fill the remainder of the buffer with zeros up to the space reserved
      // for the message length. The message length fits in this hash step.
      thrust::fill(thrust::seq,
                   hash_state.buffer + hash_state.buffer_length + end_of_message_size,
                   hash_state.buffer + Hasher::message_chunk_size - message_length_supported_size,
                   0x00);
    } else {
      // Fill the remainder of the buffer with zeros. The message length doesn't
      // fit and will be processed in a subsequent hash step comprised of only
      // zeros followed by the message length.
      thrust::fill(thrust::seq,
                   hash_state.buffer + hash_state.buffer_length + end_of_message_size,
                   hash_state.buffer + Hasher::message_chunk_size,
                   0x00);
      Hasher::hash_step(hash_state);

      // Fill the entire message with zeros up to the final bytes reserved for
      // the message length.
      thrust::fill_n(thrust::seq,
                     hash_state.buffer,
                     Hasher::message_chunk_size - message_length_supported_size,
                     0x00);
    }

    // Convert the 64-bit message length from little-endian to big-endian.
    uint64_t const full_length_flipped = swap_endian(message_length_in_bits);
    std::memcpy(hash_state.buffer + Hasher::message_chunk_size - message_length_supported_size,
                reinterpret_cast<uint8_t const*>(&full_length_flipped),
                message_length_supported_size);
    Hasher::hash_step(hash_state);

    // Each byte in the word generates two bytes in the hexadecimal string digest.
    // SHA-224 and SHA-384 digests are truncated because their digest does not
    // include all of the hash values.
    auto constexpr num_words_to_copy =
      Hasher::digest_size / (2 * sizeof(typename Hasher::sha_word_type));
    for (int i = 0; i < num_words_to_copy; i++) {
      // Convert word representation from big-endian to little-endian.
      typename Hasher::sha_word_type flipped = swap_endian(hash_state.hash_value[i]);
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
                                       typename Hasher::sha_intermediate_data& hash_state) const
  {
    cudf_assert(false && "SHA Unsupported chrono type column");
  }

  template <
    typename T,
    typename Hasher                                                                     = HasherT,
    typename std::enable_if_t<!is_fixed_width<T>() && !std::is_same_v<T, string_view>>* = nullptr>
  void CUDA_DEVICE_CALLABLE operator()(column_device_view col,
                                       size_type row_index,
                                       typename Hasher::sha_intermediate_data& hash_state) const
  {
    cudf_assert(false && "SHA Unsupported non-fixed-width type column");
  }

  template <typename T,
            typename Hasher                                    = HasherT,
            typename std::enable_if_t<is_floating_point<T>()>* = nullptr>
  void CUDA_DEVICE_CALLABLE operator()(column_device_view col,
                                       size_type row_index,
                                       typename Hasher::sha_intermediate_data& hash_state) const
  {
    T const& key = col.element<T>(row_index);
    if (isnan(key)) {
      T nan = std::numeric_limits<T>::quiet_NaN();
      process_key(nan, hash_state);
    } else if (key == T{0.0}) {
      process_key(T{0.0}, hash_state);
    } else {
      process_key(key, hash_state);
    }
  }

  template <typename T,
            typename Hasher                             = HasherT,
            typename std::enable_if_t<is_fixed_width<T>() && !is_floating_point<T>() &&
                                      !is_chrono<T>()>* = nullptr>
  void CUDA_DEVICE_CALLABLE operator()(column_device_view col,
                                       size_type row_index,
                                       typename Hasher::sha_intermediate_data& hash_state) const
  {
    process_key(col.element<T>(row_index), hash_state);
  }

  template <typename T,
            typename Hasher                                            = HasherT,
            typename std::enable_if_t<std::is_same_v<T, string_view>>* = nullptr>
  void CUDA_DEVICE_CALLABLE operator()(column_device_view col,
                                       size_type row_index,
                                       typename Hasher::sha_intermediate_data& hash_state) const
  {
    string_view key     = col.element<string_view>(row_index);
    uint8_t const* data = reinterpret_cast<uint8_t const*>(key.data());
    uint32_t const len  = static_cast<uint32_t>(key.size_bytes());
    process(data, len, hash_state);
  }
};

/**
 * @brief Core SHA-1 algorithm implementation. Processes a single 512-bit chunk,
 * updating the hash value so far. Does not zero out the buffer contents.
 */
template <typename sha_intermediate_data>
void CUDA_DEVICE_CALLABLE sha1_hash_step(sha_intermediate_data& hash_state)
{
  uint32_t A = hash_state.hash_value[0];
  uint32_t B = hash_state.hash_value[1];
  uint32_t C = hash_state.hash_value[2];
  uint32_t D = hash_state.hash_value[3];
  uint32_t E = hash_state.hash_value[4];

  uint32_t words[80];

  // Word size in bytes
  auto constexpr word_size = sizeof(uint32_t);

  // The 512-bit message buffer fills the first 16 words.
  for (int i = 0; i < 16; i++) {
    uint32_t buffer_element_as_int;
    std::memcpy(&buffer_element_as_int, hash_state.buffer + (i * word_size), word_size);
    // Convert word representation from little-endian to big-endian.
    words[i] = swap_endian(buffer_element_as_int);
  }

  // The rest of the 80 words are generated from the first 16 words.
  for (int i = 16; i < 80; i++) {
    uint32_t temp = words[i - 3] ^ words[i - 8] ^ words[i - 14] ^ words[i - 16];
    words[i]      = rotate_bits_left(temp, 1);
  }

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
    temp = rotate_bits_left(A, 5) + F + E + k + words[i];
    E    = D;
    D    = C;
    C    = rotate_bits_left(B, 30);
    B    = A;
    A    = temp;
  }

  hash_state.hash_value[0] += A;
  hash_state.hash_value[1] += B;
  hash_state.hash_value[2] += C;
  hash_state.hash_value[3] += D;
  hash_state.hash_value[4] += E;

  hash_state.buffer_length = 0;
}

/**
 * @brief Core SHA-256 algorithm implementation. Processes a single 512-bit chunk,
 * updating the hash value so far. Does not zero out the buffer contents.
 */
template <typename sha_intermediate_data>
void CUDA_DEVICE_CALLABLE sha256_hash_step(sha_intermediate_data& hash_state)
{
  uint32_t A = hash_state.hash_value[0];
  uint32_t B = hash_state.hash_value[1];
  uint32_t C = hash_state.hash_value[2];
  uint32_t D = hash_state.hash_value[3];
  uint32_t E = hash_state.hash_value[4];
  uint32_t F = hash_state.hash_value[5];
  uint32_t G = hash_state.hash_value[6];
  uint32_t H = hash_state.hash_value[7];

  uint32_t words[64];

  // Word size in bytes
  auto constexpr word_size = sizeof(uint32_t);

  // The 512-bit message buffer fills the first 16 words.
  for (int i = 0; i < 16; i++) {
    uint32_t buffer_element_as_int;
    std::memcpy(&buffer_element_as_int, hash_state.buffer + (i * word_size), word_size);
    // Convert word representation from little-endian to big-endian.
    words[i] = swap_endian(buffer_element_as_int);
  }

  // The rest of the 64 words are generated from the first 16 words.
  for (int i = 16; i < 64; i++) {
    uint32_t s0 = rotate_bits_right(words[i - 15], 7) ^ rotate_bits_right(words[i - 15], 18) ^
                  (words[i - 15] >> 3);
    uint32_t s1 = rotate_bits_right(words[i - 2], 17) ^ rotate_bits_right(words[i - 2], 19) ^
                  (words[i - 2] >> 10);
    words[i] = words[i - 16] + s0 + words[i - 7] + s1;
  }

  for (int i = 0; i < 64; i++) {
    uint32_t const s1 =
      rotate_bits_right(E, 6) ^ rotate_bits_right(E, 11) ^ rotate_bits_right(E, 25);
    uint32_t const ch    = (E & F) ^ ((~E) & G);
    uint32_t const temp1 = H + s1 + ch + sha256_hash_constants[i] + words[i];
    uint32_t const s0 =
      rotate_bits_right(A, 2) ^ rotate_bits_right(A, 13) ^ rotate_bits_right(A, 22);
    uint32_t const maj   = (A & B) ^ (A & C) ^ (B & C);
    uint32_t const temp2 = s0 + maj;

    H = G;
    G = F;
    F = E;
    E = D + temp1;
    D = C;
    C = B;
    B = A;
    A = temp1 + temp2;
  }

  hash_state.hash_value[0] += A;
  hash_state.hash_value[1] += B;
  hash_state.hash_value[2] += C;
  hash_state.hash_value[3] += D;
  hash_state.hash_value[4] += E;
  hash_state.hash_value[5] += F;
  hash_state.hash_value[6] += G;
  hash_state.hash_value[7] += H;

  hash_state.buffer_length = 0;
}

/**
 * @brief Core SHA-512 algorithm implementation. Processes a single 1024-bit chunk,
 * updating the hash value so far. Does not zero out the buffer contents.
 */
template <typename sha_intermediate_data>
void CUDA_DEVICE_CALLABLE sha512_hash_step(sha_intermediate_data& hash_state)
{
  uint64_t A = hash_state.hash_value[0];
  uint64_t B = hash_state.hash_value[1];
  uint64_t C = hash_state.hash_value[2];
  uint64_t D = hash_state.hash_value[3];
  uint64_t E = hash_state.hash_value[4];
  uint64_t F = hash_state.hash_value[5];
  uint64_t G = hash_state.hash_value[6];
  uint64_t H = hash_state.hash_value[7];

  uint64_t words[80];

  // Word size in bytes
  auto constexpr word_size = sizeof(uint64_t);

  // The 512-bit message buffer fills the first 16 words.
  for (int i = 0; i < 16; i++) {
    uint64_t buffer_element_as_int;
    std::memcpy(&buffer_element_as_int, hash_state.buffer + (i * word_size), word_size);
    // Convert word representation from little-endian to big-endian.
    words[i] = swap_endian(buffer_element_as_int);
  }

  // The rest of the 80 words are generated from the first 16 words.
  for (int i = 16; i < 80; i++) {
    uint64_t s0 = rotate_bits_right(words[i - 15], 1) ^ rotate_bits_right(words[i - 15], 8) ^
                  (words[i - 15] >> 7);
    uint64_t s1 = rotate_bits_right(words[i - 2], 19) ^ rotate_bits_right(words[i - 2], 61) ^
                  (words[i - 2] >> 6);
    words[i] = words[i - 16] + s0 + words[i - 7] + s1;
  }

  for (int i = 0; i < 80; i++) {
    uint64_t const s1 =
      rotate_bits_right(E, 14) ^ rotate_bits_right(E, 18) ^ rotate_bits_right(E, 41);
    uint64_t const ch    = (E & F) ^ ((~E) & G);
    uint64_t const temp1 = H + s1 + ch + sha512_hash_constants[i] + words[i];
    uint64_t const s0 =
      rotate_bits_right(A, 28) ^ rotate_bits_right(A, 34) ^ rotate_bits_right(A, 39);
    uint64_t const maj   = (A & B) ^ (A & C) ^ (B & C);
    uint64_t const temp2 = s0 + maj;

    H = G;
    G = F;
    F = E;
    E = D + temp1;
    D = C;
    C = B;
    B = A;
    A = temp1 + temp2;
  }

  hash_state.hash_value[0] += A;
  hash_state.hash_value[1] += B;
  hash_state.hash_value[2] += C;
  hash_state.hash_value[3] += D;
  hash_state.hash_value[4] += E;
  hash_state.hash_value[5] += F;
  hash_state.hash_value[6] += G;
  hash_state.hash_value[7] += H;

  hash_state.buffer_length = 0;
}

struct SHA1Hash : SHAHash<SHA1Hash> {
  // Intermediate data type storing the hash state
  using sha_intermediate_data = sha1_intermediate_data;
  // The word type used by this hash function
  using sha_word_type = sha1_word_type;
  // Number of bytes processed in each hash step
  static constexpr auto message_chunk_size = 64;
  // Digest size in bytes
  static constexpr auto digest_size = 40;
  // Number of bytes used for the message length
  static constexpr auto message_length_size = 8;

  static void CUDA_DEVICE_CALLABLE hash_step(sha_intermediate_data& hash_state)
  {
    sha1_hash_step(hash_state);
  }

  sha_intermediate_data hash_state;
};

struct SHA224Hash : SHAHash<SHA224Hash> {
  // Intermediate data type storing the hash state
  using sha_intermediate_data = sha224_intermediate_data;
  // The word type used by this hash function
  using sha_word_type = sha256_word_type;
  // Number of bytes processed in each hash step
  static constexpr auto message_chunk_size = 64;
  // Digest size in bytes. This is truncated from SHA-256.
  static constexpr auto digest_size = 56;
  // Number of bytes used for the message length
  static constexpr auto message_length_size = 8;

  static void CUDA_DEVICE_CALLABLE hash_step(sha_intermediate_data& hash_state)
  {
    sha256_hash_step(hash_state);
  }

  sha_intermediate_data hash_state;
};

struct SHA256Hash : SHAHash<SHA256Hash> {
  // Intermediate data type storing the hash state
  using sha_intermediate_data = sha256_intermediate_data;
  // The word type used by this hash function
  using sha_word_type = sha256_word_type;
  // Number of bytes processed in each hash step
  static constexpr auto message_chunk_size = 64;
  // Digest size in bytes
  static constexpr auto digest_size = 64;
  // Number of bytes used for the message length
  static constexpr auto message_length_size = 8;

  static void CUDA_DEVICE_CALLABLE hash_step(sha_intermediate_data& hash_state)
  {
    sha256_hash_step(hash_state);
  }

  sha_intermediate_data hash_state;
};

struct SHA384Hash : SHAHash<SHA384Hash> {
  // Intermediate data type storing the hash state
  using sha_intermediate_data = sha384_intermediate_data;
  // The word type used by this hash function
  using sha_word_type = sha512_word_type;
  // Number of bytes processed in each hash step
  static constexpr auto message_chunk_size = 128;
  // Digest size in bytes. This is truncated from SHA-512.
  static constexpr auto digest_size = 96;
  // Number of bytes used for the message length
  static constexpr auto message_length_size = 16;

  static void CUDA_DEVICE_CALLABLE hash_step(sha_intermediate_data& hash_state)
  {
    sha512_hash_step(hash_state);
  }

  sha_intermediate_data hash_state;
};

struct SHA512Hash : SHAHash<SHA512Hash> {
  // Intermediate data type storing the hash state
  using sha_intermediate_data = sha512_intermediate_data;
  // The word type used by this hash function
  using sha_word_type = sha512_word_type;
  // Number of bytes processed in each hash step
  static constexpr auto message_chunk_size = 128;
  // Digest size in bytes
  static constexpr auto digest_size = 128;
  // Number of bytes used for the message length
  static constexpr auto message_length_size = 16;

  static void CUDA_DEVICE_CALLABLE hash_step(sha_intermediate_data& hash_state)
  {
    sha512_hash_step(hash_state);
  }

  sha_intermediate_data hash_state;
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
                                 rmm::cuda_stream_view stream,
                                 rmm::mr::device_memory_resource* mr)
{
  if (input.num_columns() == 0 || input.num_rows() == 0) {
    // Return the hash of a zero-length input.
    // TODO: This probably needs tested!
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
  auto d_chars    = chars_view.template data<char>();

  rmm::device_buffer null_mask{0, stream, mr};

  auto const device_input = table_device_view::create(input, stream);

  // Hash each row, hashing each element sequentially left to right
  thrust::for_each(rmm::exec_policy(stream),
                   thrust::make_counting_iterator(0),
                   thrust::make_counting_iterator(input.num_rows()),
                   [d_chars, device_input = *device_input] __device__(auto row_index) {
                     Hasher hasher = Hasher{};
                     for (int col_index = 0; col_index < device_input.num_columns(); col_index++) {
                       if (device_input.column(col_index).is_valid(row_index)) {
                         cudf::type_dispatcher<dispatch_storage_type>(
                           device_input.column(col_index).type(),
                           hasher,
                           device_input.column(col_index),
                           row_index,
                           hasher.hash_state);
                       }
                     }
                     auto const result_location = d_chars + (row_index * Hasher::digest_size);
                     hasher.finalize(hasher.hash_state, result_location);
                   });

  return make_strings_column(
    input.num_rows(), std::move(offsets_column), std::move(chars_column), 0, std::move(null_mask));
}

std::unique_ptr<column> sha1_hash(table_view const& input,
                                  rmm::cuda_stream_view stream,
                                  rmm::mr::device_memory_resource* mr)
{
  string_scalar const empty_result("da39a3ee5e6b4b0d3255bfef95601890afd80709");
  return sha_hash<SHA1Hash>(input, empty_result, stream, mr);
}

std::unique_ptr<column> sha224_hash(table_view const& input,
                                    rmm::cuda_stream_view stream,
                                    rmm::mr::device_memory_resource* mr)
{
  string_scalar const empty_result("d14a028c2a3a2bc9476102bb288234c415a2b01f828ea62ac5b3e42f");
  return sha_hash<SHA224Hash>(input, empty_result, stream, mr);
}

std::unique_ptr<column> sha256_hash(table_view const& input,
                                    rmm::cuda_stream_view stream,
                                    rmm::mr::device_memory_resource* mr)
{
  string_scalar const empty_result(
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  return sha_hash<SHA256Hash>(input, empty_result, stream, mr);
}

std::unique_ptr<column> sha384_hash(table_view const& input,
                                    rmm::cuda_stream_view stream,
                                    rmm::mr::device_memory_resource* mr)
{
  string_scalar const empty_result(
    "38b060a751ac96384cd9327eb1b1e36a21fdb71114be07434c0cc7bf63f6e1da274edebfe76f65fbd51ad2f14898b9"
    "5b");
  return sha_hash<SHA384Hash>(input, empty_result, stream, mr);
}

std::unique_ptr<column> sha512_hash(table_view const& input,
                                    rmm::cuda_stream_view stream,
                                    rmm::mr::device_memory_resource* mr)
{
  string_scalar const empty_result(
    "cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce47d0d13c5d85f2b0ff8318d2877eec"
    "2f63b931bd47417a81a538327af927da3e");
  return sha_hash<SHA512Hash>(input, empty_result, stream, mr);
}

}  // namespace detail
}  // namespace cudf
