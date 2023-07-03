/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/algorithm.cuh>
#include <cudf/detail/utilities/hash_functions.cuh>
#include <cudf/table/table_device_view.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/for_each.h>

namespace cudf {
namespace detail {

namespace {

using hash_value_type = thrust::pair<uint64_t, uint64_t>;

// MurmurHash3_x64_128 implementation from
// https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp
//-----------------------------------------------------------------------------
// MurmurHash3_64_128 was written by Austin Appleby, and is placed in the public
// domain. The author hereby disclaims copyright to this source code.
// Note - The x86 and x64 versions do _not_ produce the same results, as the
// algorithms are optimized for their respective platforms. You can still
// compile and run any of them on any platform, but your performance with the
// non-native version will be less than optimal.
template <typename Key>
struct MurmurHash3_64 {
  using result_type = hash_value_type;

  constexpr MurmurHash3_64() = default;
  constexpr MurmurHash3_64(uint64_t seed) : m_seed(seed) {}

  __device__ inline uint32_t getblock32(std::byte const* data, cudf::size_type offset) const
  {
    // Read a 4-byte value from the data pointer as individual bytes for safe
    // unaligned access (very likely for string types).
    auto block = reinterpret_cast<uint8_t const*>(data + offset);
    return block[0] | (block[1] << 8) | (block[2] << 16) | (block[3] << 24);
  }

  __device__ inline uint64_t getblock64(std::byte const* data, cudf::size_type offset) const
  {
    uint64_t result = getblock32(data, offset + 4);
    result          = result << 32;
    return result | getblock32(data, offset);
  }

  __device__ inline uint64_t fmix64(uint64_t k) const
  {
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdUL;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53UL;
    k ^= k >> 33;
    return k;
  }

  result_type __device__ inline operator()(Key const& key) const { return compute(key); }

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
    // Process remaining bytes that do not fill a 8-byte chunk.
    uint64_t k1     = 0;
    uint64_t k2     = 0;
    auto const tail = reinterpret_cast<uint8_t const*>(data) + tail_offset;
    switch (len & (BLOCK_SIZE - 1)) {
      case 15: k2 ^= static_cast<uint64_t>(tail[14]) << 48;
      case 14: k2 ^= static_cast<uint64_t>(tail[13]) << 40;
      case 13: k2 ^= static_cast<uint64_t>(tail[12]) << 32;
      case 12: k2 ^= static_cast<uint64_t>(tail[11]) << 24;
      case 11: k2 ^= static_cast<uint64_t>(tail[10]) << 16;
      case 10: k2 ^= static_cast<uint64_t>(tail[9]) << 8;
      case 9:
        k2 ^= static_cast<uint64_t>(tail[8]) << 0;
        k2 *= c2;
        k2 = rotate_bits_left(k2, 33);
        k2 *= c1;
        h.second ^= k2;

      case 8: k1 ^= static_cast<uint64_t>(tail[7]) << 56;
      case 7: k1 ^= static_cast<uint64_t>(tail[6]) << 48;
      case 6: k1 ^= static_cast<uint64_t>(tail[5]) << 40;
      case 5: k1 ^= static_cast<uint64_t>(tail[4]) << 32;
      case 4: k1 ^= static_cast<uint64_t>(tail[3]) << 24;
      case 3: k1 ^= static_cast<uint64_t>(tail[2]) << 16;
      case 2: k1 ^= static_cast<uint64_t>(tail[1]) << 8;
      case 1:
        k1 ^= static_cast<uint64_t>(tail[0]) << 0;
        k1 *= c1;
        k1 = rotate_bits_left(k1, 31);
        k1 *= c2;
        h.first ^= k1;
    };
    return h;
  }

  result_type __device__ compute_bytes(std::byte const* data, cudf::size_type const len) const
  {
    auto const nblocks = len / BLOCK_SIZE;
    uint64_t h1        = m_seed;
    uint64_t h2        = m_seed;

    // Process all four-byte chunks.
    for (cudf::size_type i = 0; i < nblocks; i++) {
      uint64_t k1 = getblock64(data, (i * BLOCK_SIZE));                     // 1st 8 bytes
      uint64_t k2 = getblock64(data, (i * BLOCK_SIZE) + (BLOCK_SIZE / 2));  // 2nd 8 bytes

      k1 *= c1;
      k1 = rotate_bits_left(k1, 31);
      k1 *= c2;

      h1 ^= k1;
      h1 = rotate_bits_left(h1, 27);
      h1 += h2;
      h1 = h1 * 5 + 0x52dce729;

      k2 *= c2;
      k2 = rotate_bits_left(k2, 33);
      k2 *= c1;

      h2 ^= k2;
      h2 = rotate_bits_left(h2, 31);
      h2 += h1;
      h2 = h2 * 5 + 0x38495ab5;
    }

    thrust::tie(h1, h2) = compute_remaining_bytes(data, len, nblocks * BLOCK_SIZE, {h1, h2});

    // Finalize hash.
    h1 ^= len;
    h2 ^= len;

    h1 += h2;
    h2 += h1;

    h1 = fmix64(h1);
    h2 = fmix64(h2);

    h1 += h2;
    h2 += h1;

    return {h1, h2};
  }

 private:
  uint64_t m_seed{};
  static constexpr uint32_t BLOCK_SIZE = 16;  // 2 x 64-bit = 16 bytes

  static constexpr uint64_t c1 = 0x87c37b91114253d5UL;
  static constexpr uint64_t c2 = 0x4cf5ad432745937fUL;
};

template <>
hash_value_type __device__ inline MurmurHash3_64<bool>::operator()(bool const& key) const
{
  return compute<uint8_t>(key);
}

template <>
hash_value_type __device__ inline MurmurHash3_64<float>::operator()(float const& key) const
{
  return compute(detail::normalize_nans(key));
}

template <>
hash_value_type __device__ inline MurmurHash3_64<double>::operator()(double const& key) const
{
  return compute(detail::normalize_nans(key));
}

template <>
hash_value_type __device__ inline MurmurHash3_64<cudf::string_view>::operator()(
  cudf::string_view const& key) const
{
  auto const data = reinterpret_cast<std::byte const*>(key.data());
  auto const len  = key.size_bytes();
  return compute_bytes(data, len);
}

template <>
hash_value_type __device__ inline MurmurHash3_64<numeric::decimal32>::operator()(
  numeric::decimal32 const& key) const
{
  return compute(key.value());
}

template <>
hash_value_type __device__ inline MurmurHash3_64<numeric::decimal64>::operator()(
  numeric::decimal64 const& key) const
{
  return compute(key.value());
}

template <>
hash_value_type __device__ inline MurmurHash3_64<numeric::decimal128>::operator()(
  numeric::decimal128 const& key) const
{
  return compute(key.value());
}

/**
 * @brief Computes the hash value of a row in the given table.
 *
 * @tparam Nullate A cudf::nullate type describing whether to check for nulls.
 */
template <typename Nullate>
class murmur_device_row_hasher {
 public:
  murmur_device_row_hasher(Nullate nulls,
                           table_device_view const& t,
                           uint64_t seed,
                           uint64_t* d_output1,
                           uint64_t* d_output2)
    : _check_nulls(nulls), _input(t), _seed(seed), _output1(d_output1), _output2(d_output2)
  {
  }

  /**
   * @brief Return the hash value of a row in the given table.
   *
   * @param row_index The row index to compute the hash value of
   * @return The hash value of the row
   */
  __device__ void operator()(size_type row_index) const noexcept
  {
    auto h = cudf::detail::accumulate(
      _input.begin(),
      _input.end(),
      hash_value_type{_seed, 0},
      [row_index, nulls = this->_check_nulls] __device__(auto hash, auto column) {
        return cudf::type_dispatcher(
          column.type(), element_hasher_adapter{}, column, row_index, nulls, hash);
      });
    _output1[row_index] = h.first;
    _output2[row_index] = h.second;
  }

  /**
   * @brief Computes the hash value of an element in the given column.
   */
  class element_hasher_adapter {
   public:
    template <typename T, CUDF_ENABLE_IF(column_device_view::has_element_accessor<T>())>
    __device__ hash_value_type operator()(column_device_view const& col,
                                          size_type row_index,
                                          Nullate const _check_nulls,
                                          hash_value_type const _seed) const noexcept
    {
      if (_check_nulls && col.is_null(row_index)) {
        return {std::numeric_limits<uint64_t>::max(), std::numeric_limits<uint64_t>::max()};
      }
      auto const hasher = MurmurHash3_64<T>{_seed.first};
      return hasher(col.element<T>(row_index));
    }

    template <typename T, CUDF_ENABLE_IF(not column_device_view::has_element_accessor<T>())>
    __device__ hash_value_type operator()(column_device_view const&,
                                          size_type,
                                          Nullate const,
                                          hash_value_type const) const noexcept
    {
      CUDF_UNREACHABLE("Unsupported type for MurmurHash64");
    }
  };

  Nullate const _check_nulls;
  table_device_view const _input;
  uint64_t const _seed;
  uint64_t* _output1;
  uint64_t* _output2;
};

}  // namespace

std::unique_ptr<table> murmur_hash3_64_128(table_view const& input,
                                           uint64_t seed,
                                           rmm::cuda_stream_view stream,
                                           rmm::mr::device_memory_resource* mr)
{
  auto output1 = make_numeric_column(
    data_type(type_id::UINT64), input.num_rows(), mask_state::UNALLOCATED, stream, mr);
  auto output2 = make_numeric_column(
    data_type(type_id::UINT64), input.num_rows(), mask_state::UNALLOCATED, stream, mr);

  if (!input.is_empty()) {
    bool const nullable   = has_nulls(input);
    auto const input_view = table_device_view::create(input, stream);
    auto d_output1        = output1->mutable_view().data<uint64_t>();
    auto d_output2        = output2->mutable_view().data<uint64_t>();

    // Compute the hash value for each row
    thrust::for_each_n(rmm::exec_policy(stream),
                       thrust::counting_iterator<size_type>(0),
                       input.num_rows(),
                       murmur_device_row_hasher(nullable, *input_view, seed, d_output1, d_output2));
  }

  std::vector<std::unique_ptr<column>> out_columns(2);
  out_columns.front() = std::move(output1);
  out_columns.back()  = std::move(output2);
  return std::make_unique<table>(std::move(out_columns));
}

}  // namespace detail

std::unique_ptr<table> murmur_hash3_64_128(table_view const& input,
                                           uint64_t seed,
                                           rmm::cuda_stream_view stream,
                                           rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::murmur_hash3_64_128(input, seed, stream, mr);
}

}  // namespace cudf
