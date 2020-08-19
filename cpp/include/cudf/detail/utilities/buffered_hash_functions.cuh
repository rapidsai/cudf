/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION.
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
 #include <cudf/strings/string_view.cuh>
 #include <hash/hash_constants.hpp>


 
namespace cudf {
    namespace detail {
    /**
     * Modified GPU implementation of
     * https://johnnylee-sde.github.io/Fast-unsigned-integer-to-hex-string/
     * Copyright (c) 2015 Barry Clark
     * Licensed under the MIT license.
     * See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
     */
    void CUDA_DEVICE_CALLABLE uint32ToLowercaseHexString(uint32_t num, char* destination)
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
      thrust::copy_n(thrust::seq, reinterpret_cast<uint8_t*>(&x), 8, destination);
    }
    
    struct MD5Hash {
      /**
       * @brief Core MD5 algorithm implementation. Processes a single 512-bit chunk,
       * updating the hash value so far. Does not zero out the buffer contents.
       */
      void __device__ hash_step(md5_intermediate_data* hash_state) const
      {
        uint32_t A = hash_state->hash_value[0];
        uint32_t B = hash_state->hash_value[1];
        uint32_t C = hash_state->hash_value[2];
        uint32_t D = hash_state->hash_value[3];
    
        for (unsigned int j = 0; j < 64; j++) {
          uint32_t F;
          uint32_t g;
          switch (j / 16) {
            case 0:
              F = (B & C) | ((~B) & D);
              g = j;
              break;
            case 1:
              F = (D & B) | ((~D) & C);
              g = (5 * j + 1) % 16;
              break;
            case 2:
              F = B ^ C ^ D;
              g = (3 * j + 5) % 16;
              break;
            case 3:
              F = C ^ (B | (~D));
              g = (7 * j) % 16;
              break;
          }
    
          uint32_t buffer_element_as_int;
          std::memcpy(&buffer_element_as_int, hash_state->buffer + g * 4, 4);
          F = F + A + md5_hash_constants[j] + buffer_element_as_int;
          A = D;
          D = C;
          C = B;
          B = B + __funnelshift_l(F, F, md5_shift_constants[((j / 16) * 4) + (j % 4)]);
        }
    
        hash_state->hash_value[0] += A;
        hash_state->hash_value[1] += B;
        hash_state->hash_value[2] += C;
        hash_state->hash_value[3] += D;
    
        hash_state->buffer_length = 0;
      }
    
      /**
       * @brief Core MD5 element processing function
       */
      template <typename TKey>
      void __device__ process(TKey const& key, md5_intermediate_data* hash_state) const
      {
        uint32_t const len  = sizeof(TKey);
        uint8_t const* data = reinterpret_cast<uint8_t const*>(&key);
        hash_state->message_length += len;
    
        // 64 bytes for the number of bytes processed in a given step
        constexpr int md5_chunk_size = 64;
        if (hash_state->buffer_length + len < md5_chunk_size) {
          thrust::copy_n(thrust::seq, data, len, hash_state->buffer + hash_state->buffer_length);
          hash_state->buffer_length += len;
        } else {
          uint32_t copylen = md5_chunk_size - hash_state->buffer_length;
    
          thrust::copy_n(thrust::seq, data, copylen, hash_state->buffer + hash_state->buffer_length);
          hash_step(hash_state);
    
          while (len > md5_chunk_size + copylen) {
            thrust::copy_n(thrust::seq, data + copylen, md5_chunk_size, hash_state->buffer);
            hash_step(hash_state);
            copylen += md5_chunk_size;
          }
    
          thrust::copy_n(thrust::seq, data + copylen, len - copylen, hash_state->buffer);
          hash_state->buffer_length = len - copylen;
        }
      }
    
      void __device__ finalize(md5_intermediate_data* hash_state, char* result_location) const
      {
        auto const full_length = (static_cast<uint64_t>(hash_state->message_length)) << 3;
        thrust::fill_n(thrust::seq, hash_state->buffer + hash_state->buffer_length, 1, 0x80);
    
        // 64 bytes for the number of bytes processed in a given step
        constexpr int md5_chunk_size = 64;
        // 8 bytes for the total message length, appended to the end of the last chunk processed
        constexpr int message_length_size = 8;
        // 1 byte for the end of the message flag
        constexpr int end_of_message_size = 1;
        if (hash_state->buffer_length + message_length_size + end_of_message_size <= md5_chunk_size) {
          thrust::fill_n(
            thrust::seq,
            hash_state->buffer + hash_state->buffer_length + 1,
            (md5_chunk_size - message_length_size - end_of_message_size - hash_state->buffer_length),
            0x00);
        } else {
          thrust::fill_n(thrust::seq,
                         hash_state->buffer + hash_state->buffer_length + 1,
                         (md5_chunk_size - hash_state->buffer_length),
                         0x00);
          hash_step(hash_state);
    
          thrust::fill_n(thrust::seq, hash_state->buffer, md5_chunk_size - message_length_size, 0x00);
        }
    
        thrust::copy_n(thrust::seq,
                       reinterpret_cast<uint8_t const*>(&full_length),
                       message_length_size,
                       hash_state->buffer + md5_chunk_size - message_length_size);
        hash_step(hash_state);
    
    #pragma unroll
        for (int i = 0; i < 4; ++i){
          uint32ToLowercaseHexString(hash_state->hash_value[i], result_location + (8 * i));
        }
      }
    
      template <typename T, typename std::enable_if_t<is_chrono<T>()>* = nullptr>
      void __device__ operator()(column_device_view col,
                                 size_type row_index,
                                 md5_intermediate_data* hash_state) const
      {
        release_assert(false && "MD5 Unsupported chrono type column");
      }
    
      template <typename T, typename std::enable_if_t<!is_fixed_width<T>()>* = nullptr>
      void __device__ operator()(column_device_view col,
                                 size_type row_index,
                                 md5_intermediate_data* hash_state) const
      {
        release_assert(false && "MD5 Unsupported non-fixed-width type column");
      }
    
      template <typename T, typename std::enable_if_t<is_floating_point<T>()>* = nullptr>
      void __device__ operator()(column_device_view col,
                                 size_type row_index,
                                 md5_intermediate_data* hash_state) const
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
                                           md5_intermediate_data* hash_state) const
      {
        process(col.element<T>(row_index), hash_state);
      }
    };
    
    template <>
    void CUDA_DEVICE_CALLABLE MD5Hash::operator()<string_view>(column_device_view col,
                                                               size_type row_index,
                                                               md5_intermediate_data* hash_state) const
    {
      string_view key     = col.element<string_view>(row_index);
      uint32_t const len  = static_cast<uint32_t>(key.size_bytes());
      uint8_t const* data = reinterpret_cast<uint8_t const*>(key.data());
    
      hash_state->message_length += len;
    
      if (hash_state->buffer_length + len < 64) {
        thrust::copy_n(thrust::seq, data, len, hash_state->buffer + hash_state->buffer_length);
        hash_state->buffer_length += len;
      } else {
        uint32_t copylen = 64 - hash_state->buffer_length;
        thrust::copy_n(thrust::seq, data, copylen, hash_state->buffer + hash_state->buffer_length);
        hash_step(hash_state);
    
        while (len > 64 + copylen) {
          thrust::copy_n(thrust::seq, data + copylen, 64, hash_state->buffer);
          hash_step(hash_state);
          copylen += 64;
        }
    
        thrust::copy_n(thrust::seq, data + copylen, len - copylen, hash_state->buffer);
        hash_state->buffer_length = len - copylen;
      }
    }
    
    
    struct SHA1Hash {
    
    
      CUDA_HOST_DEVICE_CALLABLE uint32_t rotl32(uint32_t x, int8_t r) const
      {
        return (x << r) | (x >> (32 - r));
      }
    
      /**
       * @brief Core SHA1 algorithm implementation. Processes a single 512-bit chunk,
       * updating the hash value so far. Does not zero out the buffer contents.
       */
      void __device__ hash_step(sha1_intermediate_data* hash_state) const {
        uint32_t temp_hash[5];
        thrust::copy_n(thrust::seq, hash_state->hash_value, 5, temp_hash);
        // temp_hash[0] = hash_state->hash_value[0];
        // temp_hash[1] = hash_state->hash_value[1];
        // temp_hash[2] = hash_state->hash_value[2];
        // temp_hash[3] = hash_state->hash_value[3];
        // temp_hash[4] = hash_state->hash_value[4];
    
        uint32_t words[80];
        for(int i = 0; i < 16; i++) {
          uint32_t buffer_element_as_int;
          std::memcpy(&buffer_element_as_int, hash_state->buffer + (i * 4), 4);
          // words[i] = buffer_element_as_int;

          words[i] = (buffer_element_as_int << 24) & 0xff000000;
          words[i] |= (buffer_element_as_int << 8) & 0xff0000;
          words[i] |= (buffer_element_as_int >> 8) & 0xff00;
          words[i] |= (buffer_element_as_int >> 24) & 0xff;
        }
        // std::memcpy(words, hash_state->buffer, 64);
        for(int i = 16; i < 80; i++) {
          uint32_t temp = words[i-3] ^ words[i-8] ^ words[i-14] ^ words[i-16];
          words[i] = rotl32(temp, 1);
          // words[i] = __funnelshift_l(temp, temp, 1);
        }
    
      // #pragma unroll
        for(int i = 0; i < 80; i++) {
          uint32_t F;
          uint32_t temp;
          uint32_t k;
          switch(i/20) {
            case 0:
              // F = (temp_hash[1] & temp_hash[2]) | ((~temp_hash[1]) & temp_hash[3]);
              F = ((temp_hash[1] & (temp_hash[2] ^ temp_hash[3])) ^ temp_hash[3]);
              k = 0x5a827999;
              break;
            case 1:
              F = temp_hash[1] ^ temp_hash[2] ^ temp_hash[3];
              k = 0x6ed9eba1;
              break;
            case 2:
              F =  (temp_hash[1] & temp_hash[2]) | (temp_hash[1] & temp_hash[3]) | (temp_hash[2] & temp_hash[3]);
              k = 0x8f1bbcdc;
              break;
            case 3:
              F =  temp_hash[1] ^ temp_hash[2] ^ temp_hash[3];
              k = 0xca62c1d6;
              break;
          }
          temp = rotl32(temp_hash[0], 5) + F +  temp_hash[4] + k + words[i];
          // temp = __funnelshift_l(temp_hash[0], temp_hash[0], 5) + F +  temp_hash[4] + k + words[i];
          temp_hash[4] = temp_hash[3];
          temp_hash[3] = temp_hash[2];
          // temp_hash[2] = __funnelshift_l(temp_hash[1], temp_hash[1], 30);
          temp_hash[2] = rotl32(temp_hash[1], 30);
          temp_hash[1] = temp_hash[0];
          temp_hash[0] = temp;
        }
    
      #pragma unroll
        for(int i = 0; i < 5; i++) {
          hash_state->hash_value[i] = hash_state->hash_value[i] + temp_hash[i];
        }
      }
      
      /**
       * @brief Core SHA1 element processing function
       */
      template <typename TKey>
      void __device__ process(TKey const& key, sha1_intermediate_data* hash_state) const {
        uint32_t const len  = sizeof(TKey);
        uint8_t const* data = reinterpret_cast<uint8_t const*>(&key);
        hash_state->message_length += len;
    
        // 64 bytes for the number of bytes processed in a given step
        constexpr int sha1_chunk_size = 64;
        if (hash_state->buffer_length + len < sha1_chunk_size) {
          thrust::copy_n(thrust::seq, data, len, hash_state->buffer + hash_state->buffer_length);
          hash_state->buffer_length += len;
        } else {
          uint32_t copylen = sha1_chunk_size - hash_state->buffer_length;
    
          thrust::copy_n(thrust::seq, data, copylen, hash_state->buffer + hash_state->buffer_length);
          hash_step(hash_state);
    
          while (len > sha1_chunk_size + copylen) {
            thrust::copy_n(thrust::seq, data + copylen, sha1_chunk_size, hash_state->buffer);
            hash_step(hash_state);
            copylen += sha1_chunk_size;
          }
    
          thrust::copy_n(thrust::seq, data + copylen, len - copylen, hash_state->buffer);
          hash_state->buffer_length = len - copylen;
        }
      }
    
    
    void __device__ finalize(sha1_intermediate_data* hash_state, char* result_location) const
    {
      auto const full_length = (static_cast<uint64_t>(hash_state->message_length)) << 3;
      thrust::fill_n(thrust::seq, hash_state->buffer + hash_state->buffer_length, 1, 0x80);
    
      // 64 bytes for the number of bytes processed in a given step
      constexpr int sha1_chunk_size = 64;
      // 8 bytes for the total message length, appended to the end of the last chunk processed
      constexpr int message_length_size = 8;
      // 1 byte for the end of the message flag
      constexpr int end_of_message_size = 1;
      if (hash_state->buffer_length + message_length_size + end_of_message_size <= sha1_chunk_size) {
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
    
      thrust::copy_n(thrust::seq,
                     reinterpret_cast<uint8_t const*>(&full_length),
                     message_length_size,
                     hash_state->buffer + sha1_chunk_size - message_length_size);
      hash_step(hash_state);
      // std::memcpy(hash_state->hash_value, hash_state->buffer, 160);
    
    
    #pragma unroll
      for (int i = 0; i < 5; ++i){
        

        uint32_t flipped = (hash_state->hash_value[i] << 24) & 0xff000000;
        flipped |= (hash_state->hash_value[i] << 8) & 0xff0000;
        flipped |= (hash_state->hash_value[i] >> 8) & 0xff00;
        flipped |= (hash_state->hash_value[i] >> 24) & 0xff;
        uint32ToLowercaseHexString(flipped, result_location + (8 * i));
        
        // uint32ToLowercaseHexString(hash_state->hash_value[i], result_location + (8 * i));
    
        // std::memcpy(hash_state->hash_value, hash_state->buffer + 64 + (i * 4), 4);
        // std::memcpy(hash_state->hash_value, &full_length, 8);
        // uint32ToLowercaseHexString(hash_state->hash_value[0], result_location + (8 * i));
      }
    }
    
    
    template <typename T, typename std::enable_if_t<is_chrono<T>()>* = nullptr>
    void __device__ operator()(column_device_view col,
                               size_type row_index,
                               sha1_intermediate_data* hash_state) const
    {
      release_assert(false && "MD5 Unsupported chrono type column");
    }
    
    template <typename T, typename std::enable_if_t<!is_fixed_width<T>()>* = nullptr>
    void __device__ operator()(column_device_view col,
                               size_type row_index,
                               sha1_intermediate_data* hash_state) const
    {
      release_assert(false && "MD5 Unsupported non-fixed-width type column");
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
    void CUDA_DEVICE_CALLABLE SHA1Hash::operator()<string_view>(column_device_view col,
                                                               size_type row_index,
                                                               sha1_intermediate_data* hash_state) const
    {
      string_view key     = col.element<string_view>(row_index);
      uint32_t const len  = static_cast<uint32_t>(key.size_bytes());
      uint8_t const* data = reinterpret_cast<uint8_t const*>(key.data());
    
      hash_state->message_length += len;
    
      if (hash_state->buffer_length + len < 64) {
        thrust::copy_n(thrust::seq, data, len, hash_state->buffer + hash_state->buffer_length);
        hash_state->buffer_length += len;
      } else {
        uint32_t copylen = 64 - hash_state->buffer_length;
        thrust::copy_n(thrust::seq, data, copylen, hash_state->buffer + hash_state->buffer_length);
        hash_step(hash_state);
    
        while (len > 64 + copylen) {
          thrust::copy_n(thrust::seq, data + copylen, 64, hash_state->buffer);
          hash_step(hash_state);
          copylen += 64;
        }
    
        thrust::copy_n(thrust::seq, data + copylen, len - copylen, hash_state->buffer);
        hash_state->buffer_length = len - copylen;
      }
    }
    
    
    }  // namespace detail
    }  // namespace cudf
