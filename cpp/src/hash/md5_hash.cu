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
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/utilities/hash_functions.cuh>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/traits.hpp>
#include <hash/hash_constants.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>

namespace cudf {

namespace detail {

namespace {
/**
 * @brief Core MD5 algorithm implementation. Processes a single 512-bit chunk,
 * updating the hash value so far. Does not zero out the buffer contents.
 */
void CUDA_DEVICE_CALLABLE md5_hash_step(md5_intermediate_data* hash_state)
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
    memcpy(&buffer_element_as_int, hash_state->buffer + g * 4, 4);
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
 *
 * This accepts arbitrary data, handles it as bytes, and calls the hash step
 * when the buffer is filled up to message_chunk_size bytes.
 */
void CUDA_DEVICE_CALLABLE md5_process_bytes(uint8_t const* data,
                                            uint32_t len,
                                            md5_intermediate_data* hash_state)
{
  hash_state->message_length += len;

  // 64 bytes are processed in each hash step
  uint32_t constexpr md5_chunk_size = 64;

  if (hash_state->buffer_length + len < md5_chunk_size) {
    // The buffer will not be filled by this data. We copy the new data into
    // the buffer but do not trigger a hash step yet.
    memcpy(hash_state->buffer + hash_state->buffer_length, data, len);
    hash_state->buffer_length += len;
  } else {
    // The buffer will be filled by this data. Copy a chunk of the data to fill
    // the buffer and trigger a hash step.
    uint32_t copylen = md5_chunk_size - hash_state->buffer_length;
    memcpy(hash_state->buffer + hash_state->buffer_length, data, copylen);
    md5_hash_step(hash_state);

    // Take buffer-sized chunks of the data and do a hash step on each chunk.
    while (len > md5_chunk_size + copylen) {
      memcpy(hash_state->buffer, data + copylen, md5_chunk_size);
      md5_hash_step(hash_state);
      copylen += md5_chunk_size;
    }

    // The remaining data chunk does not fill the buffer. We copy the data into
    // the buffer but do not trigger a hash step yet.
    memcpy(hash_state->buffer, data + copylen, len - copylen);
    hash_state->buffer_length = len - copylen;
  }
}

/**
 * @brief MD5 typed element processor.
 *
 * This accepts typed data, normalizes it, and performs processing on raw bytes.
 */
template <typename T>
void CUDA_DEVICE_CALLABLE md5_process(T const& key, md5_intermediate_data* hash_state)
{
  if constexpr (is_fixed_width<T>() && !is_chrono<T>()) {
    if constexpr (is_floating_point<T>()) {
      auto const normalized_key = normalize_nans_and_zeros_helper<T>(key);
      uint8_t const* data       = reinterpret_cast<uint8_t const*>(&normalized_key);
      uint32_t constexpr len    = sizeof(T);
      md5_process_bytes(data, len, hash_state);
    } else {
      uint8_t const* data    = reinterpret_cast<uint8_t const*>(&key);
      uint32_t constexpr len = sizeof(T);
      md5_process_bytes(data, len, hash_state);
    }
  } else if constexpr (std::is_same_v<T, string_view>) {
    uint8_t const* data = reinterpret_cast<uint8_t const*>(key.data());
    uint32_t len        = static_cast<uint32_t>(key.size_bytes());
    md5_process_bytes(data, len, hash_state);
  } else {
    cudf_assert(false && "Unsupported type for hash function.");
  }
}
// MD5 supported leaf data type check
bool md5_type_check(data_type dt)
{
  return !is_chrono(dt) && (is_fixed_width(dt) || (dt.id() == type_id::STRING));
}

struct MD5ListHasher {
  template <typename T,
            CUDF_ENABLE_IF((is_fixed_width<T>() && !is_chrono<T>()) ||
                           std::is_same_v<T, string_view>)>
  void CUDA_DEVICE_CALLABLE operator()(column_device_view data_col,
                                       size_type offset_begin,
                                       size_type offset_end,
                                       md5_intermediate_data* hash_state) const
  {
    for (size_type i = offset_begin; i < offset_end; i++) {
      if (data_col.is_valid(i)) { md5_process(data_col.element<T>(i), hash_state); }
    }
  }

  template <typename T,
            CUDF_ENABLE_IF((!is_fixed_width<T>() || is_chrono<T>()) &&
                           !std::is_same_v<T, string_view>)>
  void CUDA_DEVICE_CALLABLE
  operator()(column_device_view, size_type, size_type, md5_intermediate_data*) const
  {
    cudf_assert(false && "Unsupported type for hash function.");
  }
};

struct MD5Hash {
  void CUDA_DEVICE_CALLABLE finalize(md5_intermediate_data* hash_state, char* result_location) const
  {
    auto const full_length = (static_cast<uint64_t>(hash_state->message_length)) << 3;
    thrust::fill_n(thrust::seq, hash_state->buffer + hash_state->buffer_length, 1, 0x80);

    // 64 bytes are processed in each hash step
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
      md5_hash_step(hash_state);

      thrust::fill_n(thrust::seq, hash_state->buffer, md5_chunk_size - message_length_size, 0x00);
    }

    std::memcpy(hash_state->buffer + md5_chunk_size - message_length_size,
                reinterpret_cast<uint8_t const*>(&full_length),
                message_length_size);
    md5_hash_step(hash_state);

    for (int i = 0; i < 4; ++i)
      uint32ToLowercaseHexString(hash_state->hash_value[i], result_location + (8 * i));
  }

  template <typename T,
            CUDF_ENABLE_IF((is_fixed_width<T>() && !is_chrono<T>()) ||
                           std::is_same_v<T, string_view>)>
  void CUDA_DEVICE_CALLABLE operator()(column_device_view col,
                                       size_type row_index,
                                       md5_intermediate_data* hash_state) const
  {
    md5_process(col.element<T>(row_index), hash_state);
  }

  template <typename T,
            CUDF_ENABLE_IF((!is_fixed_width<T>() || is_chrono<T>()) &&
                           !std::is_same_v<T, string_view>)>
  void CUDA_DEVICE_CALLABLE operator()(column_device_view, size_type, md5_intermediate_data*) const
  {
    cudf_assert(false && "Unsupported type for hash function.");
  }
};

template <>
void CUDA_DEVICE_CALLABLE MD5Hash::operator()<list_view>(column_device_view col,
                                                         size_type row_index,
                                                         md5_intermediate_data* hash_state) const
{
  auto const data    = col.child(lists_column_view::child_column_index);
  auto const offsets = col.child(lists_column_view::offsets_column_index);

  if (data.type().id() == type_id::LIST) cudf_assert(false && "Nested list unsupported");

  auto const offset_begin = offsets.element<size_type>(row_index);
  auto const offset_end   = offsets.element<size_type>(row_index + 1);

  cudf::type_dispatcher(data.type(), MD5ListHasher{}, data, offset_begin, offset_end, hash_state);
}

}  // namespace

std::unique_ptr<column> md5_hash(table_view const& input,
                                 rmm::cuda_stream_view stream,
                                 rmm::mr::device_memory_resource* mr)
{
  if (input.num_columns() == 0 || input.num_rows() == 0) {
    // Return the MD5 hash of a zero-length input.
    string_scalar const string_128bit("d41d8cd98f00b204e9orig98ecf8427e");
    return make_column_from_scalar(string_128bit, input.num_rows(), stream, mr);
  }

  // Accepts string and fixed width columns, or single layer list columns holding those types
  CUDF_EXPECTS(std::all_of(input.begin(),
                           input.end(),
                           [](auto const& col) {
                             if (col.type().id() == type_id::LIST) {
                               return md5_type_check(lists_column_view(col).child().type());
                             }
                             return md5_type_check(col.type());
                           }),
               "Unsupported column type for hash function.");

  // Digest size in bytes
  auto constexpr digest_size = 32;
  // Result column allocation and creation
  auto begin = thrust::make_constant_iterator(digest_size);
  auto offsets_column =
    cudf::strings::detail::make_offsets_child_column(begin, begin + input.num_rows(), stream, mr);

  auto chars_column =
    strings::detail::create_chars_child_column(input.num_rows() * digest_size, stream, mr);
  auto chars_view = chars_column->mutable_view();
  auto d_chars    = chars_view.data<char>();

  rmm::device_buffer null_mask{0, stream, mr};

  auto const device_input = table_device_view::create(input, stream);

  // Hash each row, hashing each element sequentially left to right
  thrust::for_each(rmm::exec_policy(stream),
                   thrust::make_counting_iterator(0),
                   thrust::make_counting_iterator(input.num_rows()),
                   [d_chars, device_input = *device_input] __device__(auto row_index) {
                     md5_intermediate_data hash_state;
                     MD5Hash hasher = MD5Hash{};
                     for (auto const& col : device_input) {
                       if (col.is_valid(row_index)) {
                         cudf::type_dispatcher<dispatch_storage_type>(
                           col.type(), hasher, col, row_index, &hash_state);
                       }
                     }
                     hasher.finalize(&hash_state, d_chars + (row_index * digest_size));
                   });

  return make_strings_column(
    input.num_rows(), std::move(offsets_column), std::move(chars_column), 0, std::move(null_mask));
}

}  // namespace detail
}  // namespace cudf
