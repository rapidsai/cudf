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

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>

#include <iterator>

namespace cudf {

namespace detail {

namespace {

static const __device__ __constant__ uint32_t md5_shift_constants[16] = {
  7,
  12,
  17,
  22,
  5,
  9,
  14,
  20,
  4,
  11,
  16,
  23,
  6,
  10,
  15,
  21,
};

static const __device__ __constant__ uint32_t md5_hash_constants[64] = {
  0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
  0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
  0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
  0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
  0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
  0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
  0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
  0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391,
};

template <typename T, int capacity>
struct hash_circular_buffer {
  T storage[capacity];
  T* cur;

  CUDA_DEVICE_CALLABLE hash_circular_buffer() : cur(storage) {}

  CUDA_DEVICE_CALLABLE T* begin() { return storage; }
  CUDA_DEVICE_CALLABLE const T* begin() const { return storage; }

  CUDA_DEVICE_CALLABLE T* end() { return &storage[capacity]; }
  CUDA_DEVICE_CALLABLE const T* end() const { return &storage[capacity]; }

  CUDA_DEVICE_CALLABLE int size() const
  {
    return std::distance(begin(), static_cast<const T*>(cur));
  }

  CUDA_DEVICE_CALLABLE int available_space() const { return capacity - size(); }

  template <typename hash_step_callable>
  CUDA_DEVICE_CALLABLE void put(T const* in, int size, hash_step_callable hash_step)
  {
    int space      = available_space();
    int copy_start = 0;
    while (size >= space) {
      // The buffer will be filled by this chunk of data. Copy a chunk of the
      // data to fill the buffer and trigger a hash step.
      memcpy(cur, in + copy_start, space);
      hash_step();
      size -= space;
      copy_start += space;
      cur   = begin();
      space = available_space();
    }
    // The buffer will not be filled by the remaining data. That is, `size >= 0
    // && size < capacity`. We copy the remaining data into the buffer but do
    // not trigger a hash step.
    memcpy(cur, in + copy_start, size);
    cur += size;
  }

  template <typename hash_step_callable>
  CUDA_DEVICE_CALLABLE void pad(int space_to_leave, hash_step_callable hash_step)
  {
    int space = available_space();
    if (space_to_leave > space) {
      memset(cur, 0x00, space);
      hash_step();
      cur   = begin();
      space = available_space();
    }
    memset(cur, 0x00, space - space_to_leave);
    cur += space - space_to_leave;
  }

  CUDA_DEVICE_CALLABLE T& operator[](size_t idx) { return storage[idx]; }
  CUDA_DEVICE_CALLABLE const T& operator[](size_t idx) const { return storage[idx]; }
};

struct md5_hash_state {
  uint64_t message_length = 0;
  uint32_t hash_value[4]  = {0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476};
  hash_circular_buffer<uint8_t, 64> buffer;
};

/**
 * @brief Core MD5 algorithm implementation. Processes a single 512-bit chunk,
 * updating the hash value so far. Does not zero out the buffer contents.
 */
void CUDA_DEVICE_CALLABLE md5_hash_step(md5_hash_state* hash_state)
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
    memcpy(&buffer_element_as_int, &hash_state->buffer[g * 4], 4);
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
}

/**
 * @brief Core MD5 element processing function
 *
 * This accepts arbitrary data, handles it as bytes, and calls the hash step
 * when the buffer is filled up to message_chunk_size bytes.
 */
void CUDA_DEVICE_CALLABLE md5_process_bytes(uint8_t const* data,
                                            uint32_t len,
                                            md5_hash_state* hash_state)
{
  hash_state->message_length += len;
  auto hash_step = [hash_state]() { md5_hash_step(hash_state); };
  hash_state->buffer.put(data, len, hash_step);
}

template <typename Key>
auto CUDA_DEVICE_CALLABLE get_data(Key const& k)
{
  if constexpr (is_fixed_width<Key>() && !is_chrono<Key>()) {
    return thrust::make_pair(reinterpret_cast<uint8_t const*>(&k), sizeof(Key));
  } else {
    cudf_assert(false && "Unsupported type.");
  }
}

auto CUDA_DEVICE_CALLABLE get_data(string_view const& s)
{
  return thrust::make_pair(reinterpret_cast<uint8_t const*>(s.data()), s.size_bytes());
}

/**
 * @brief MD5 typed element processor.
 *
 * This accepts typed data, normalizes it, and performs processing on raw bytes.
 */
template <typename Key>
void CUDA_DEVICE_CALLABLE md5_process(Key const& key, md5_hash_state* hash_state)
{
  auto const normalized_key = normalize_nans_and_zeros(key);
  auto const [data, size]   = get_data(normalized_key);
  md5_process_bytes(data, size, hash_state);
}

struct MD5ListHasher {
  template <typename Key>
  void CUDA_DEVICE_CALLABLE operator()(column_device_view data_col,
                                       size_type offset_begin,
                                       size_type offset_end,
                                       md5_hash_state* hash_state) const
  {
    if constexpr ((is_fixed_width<Key>() && !is_chrono<Key>()) ||
                  std::is_same_v<Key, string_view>) {
      for (size_type i = offset_begin; i < offset_end; i++) {
        if (data_col.is_valid(i)) { md5_process(data_col.element<Key>(i), hash_state); }
      }
    } else {
      cudf_assert(false && "Unsupported type.");
    }
  }
};

struct MD5Hash {
  void CUDA_DEVICE_CALLABLE finalize(md5_hash_state* hash_state, char* result_location) const
  {
    // Add a one bit flag (10000000) to signal the end of the message
    uint8_t constexpr end_of_message = 0x80;
    // The message length is appended to the end of the last chunk processed
    uint64_t const message_length_in_bits = hash_state->message_length * 8;

    auto hash_step = [hash_state]() { md5_hash_step(hash_state); };
    hash_state->buffer.put(&end_of_message, sizeof(end_of_message), hash_step);
    hash_state->buffer.pad(sizeof(message_length_in_bits), hash_step);
    hash_state->buffer.put(reinterpret_cast<uint8_t const*>(&message_length_in_bits),
                           sizeof(message_length_in_bits),
                           hash_step);

    for (int i = 0; i < 4; ++i) {
      uint32ToLowercaseHexString(hash_state->hash_value[i], result_location + (8 * i));
    }
  }

  template <typename T,
            CUDF_ENABLE_IF((is_fixed_width<T>() && !is_chrono<T>()) ||
                           std::is_same_v<T, string_view>)>
  void CUDA_DEVICE_CALLABLE operator()(column_device_view col,
                                       size_type row_index,
                                       md5_hash_state* hash_state) const
  {
    md5_process(col.element<T>(row_index), hash_state);
  }

  template <typename T,
            CUDF_ENABLE_IF((!is_fixed_width<T>() || is_chrono<T>()) &&
                           !std::is_same_v<T, string_view>)>
  void CUDA_DEVICE_CALLABLE operator()(column_device_view, size_type, md5_hash_state*) const
  {
    cudf_assert(false && "Unsupported type for hash function.");
  }
};

template <>
void CUDA_DEVICE_CALLABLE MD5Hash::operator()<list_view>(column_device_view col,
                                                         size_type row_index,
                                                         md5_hash_state* hash_state) const
{
  auto const data    = col.child(lists_column_view::child_column_index);
  auto const offsets = col.child(lists_column_view::offsets_column_index);

  if (data.type().id() == type_id::LIST) cudf_assert(false && "Nested list unsupported");

  auto const offset_begin = offsets.element<size_type>(row_index);
  auto const offset_end   = offsets.element<size_type>(row_index + 1);

  cudf::type_dispatcher(data.type(), MD5ListHasher{}, data, offset_begin, offset_end, hash_state);
}

// MD5 supported leaf data type check
constexpr inline bool md5_leaf_type_check(data_type dt)
{
  return (is_fixed_width(dt) && !is_chrono(dt)) || dt.id() == type_id::STRING;
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
                               return md5_leaf_type_check(lists_column_view(col).child().type());
                             }
                             return md5_leaf_type_check(col.type());
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
                     md5_hash_state hash_state;
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
