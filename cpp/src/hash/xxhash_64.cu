/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/algorithm.cuh>
#include <cudf/hashing/detail/hash_functions.cuh>
#include <cudf/hashing/detail/hashing.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/tabulate.h>

namespace cudf {
namespace hashing {
namespace detail {

namespace {

using hash_value_type = uint64_t;

template <typename Key>
struct XXHash_64 {
  using result_type = hash_value_type;

  constexpr XXHash_64() = default;
  constexpr XXHash_64(hash_value_type seed) : m_seed(seed) {}

  __device__ inline uint32_t getblock32(std::byte const* data, std::size_t offset) const
  {
    // Read a 4-byte value from the data pointer as individual bytes for safe
    // unaligned access (very likely for string types).
    auto block = reinterpret_cast<uint8_t const*>(data + offset);
    return block[0] | (block[1] << 8) | (block[2] << 16) | (block[3] << 24);
  }

  __device__ inline uint64_t getblock64(std::byte const* data, std::size_t offset) const
  {
    uint64_t result = getblock32(data, offset + 4);
    result          = result << 32;
    return result | getblock32(data, offset);
  }

  result_type __device__ inline operator()(Key const& key) const { return compute(key); }

  template <typename T>
  result_type __device__ inline compute(T const& key) const
  {
    auto data = device_span<std::byte const>(reinterpret_cast<std::byte const*>(&key), sizeof(T));
    return compute_bytes(data);
  }

  result_type __device__ inline compute_remaining_bytes(device_span<std::byte const>& in,
                                                        std::size_t offset,
                                                        result_type h64) const
  {
    // remaining data can be processed in 8-byte chunks
    if ((in.size() % 32) >= 8) {
      for (; offset <= in.size() - 8; offset += 8) {
        uint64_t k1 = getblock64(in.data(), offset) * prime2;

        k1 = rotate_bits_left(k1, 31) * prime1;
        h64 ^= k1;
        h64 = rotate_bits_left(h64, 27) * prime1 + prime4;
      }
    }

    // remaining data can be processed in 4-byte chunks
    if ((in.size() % 8) >= 4) {
      for (; offset <= in.size() - 4; offset += 4) {
        h64 ^= (getblock32(in.data(), offset) & 0xfffffffful) * prime1;
        h64 = rotate_bits_left(h64, 23) * prime2 + prime3;
      }
    }

    // and the rest
    if (in.size() % 4) {
      while (offset < in.size()) {
        h64 ^= (std::to_integer<uint8_t>(in[offset]) & 0xff) * prime5;
        h64 = rotate_bits_left(h64, 11) * prime1;
        ++offset;
      }
    }
    return h64;
  }

  result_type __device__ compute_bytes(device_span<std::byte const>& in) const
  {
    uint64_t offset = 0;
    uint64_t h64;
    // data can be processed in 32-byte chunks
    if (in.size() >= 32) {
      auto limit  = in.size() - 32;
      uint64_t v1 = m_seed + prime1 + prime2;
      uint64_t v2 = m_seed + prime2;
      uint64_t v3 = m_seed;
      uint64_t v4 = m_seed - prime1;

      do {
        // pipeline 4*8byte computations
        v1 += getblock64(in.data(), offset) * prime2;
        v1 = rotate_bits_left(v1, 31);
        v1 *= prime1;
        offset += 8;
        v2 += getblock64(in.data(), offset) * prime2;
        v2 = rotate_bits_left(v2, 31);
        v2 *= prime1;
        offset += 8;
        v3 += getblock64(in.data(), offset) * prime2;
        v3 = rotate_bits_left(v3, 31);
        v3 *= prime1;
        offset += 8;
        v4 += getblock64(in.data(), offset) * prime2;
        v4 = rotate_bits_left(v4, 31);
        v4 *= prime1;
        offset += 8;
      } while (offset <= limit);

      h64 = rotate_bits_left(v1, 1) + rotate_bits_left(v2, 7) + rotate_bits_left(v3, 12) +
            rotate_bits_left(v4, 18);

      v1 *= prime2;
      v1 = rotate_bits_left(v1, 31);
      v1 *= prime1;
      h64 ^= v1;
      h64 = h64 * prime1 + prime4;

      v2 *= prime2;
      v2 = rotate_bits_left(v2, 31);
      v2 *= prime1;
      h64 ^= v2;
      h64 = h64 * prime1 + prime4;

      v3 *= prime2;
      v3 = rotate_bits_left(v3, 31);
      v3 *= prime1;
      h64 ^= v3;
      h64 = h64 * prime1 + prime4;

      v4 *= prime2;
      v4 = rotate_bits_left(v4, 31);
      v4 *= prime1;
      h64 ^= v4;
      h64 = h64 * prime1 + prime4;
    } else {
      h64 = m_seed + prime5;
    }

    h64 += in.size();

    h64 = compute_remaining_bytes(in, offset, h64);

    return finalize(h64);
  }

  constexpr __host__ __device__ std::uint64_t finalize(std::uint64_t h) const noexcept
  {
    h ^= h >> 33;
    h *= prime2;
    h ^= h >> 29;
    h *= prime3;
    h ^= h >> 32;
    return h;
  }

 private:
  hash_value_type m_seed{};
  static constexpr uint64_t prime1 = 0x9e3779b185ebca87ul;
  static constexpr uint64_t prime2 = 0xc2b2ae3d27d4eb4ful;
  static constexpr uint64_t prime3 = 0x165667b19e3779f9ul;
  static constexpr uint64_t prime4 = 0x85ebca77c2b2ae63ul;
  static constexpr uint64_t prime5 = 0x27d4eb2f165667c5ul;
};

template <>
hash_value_type __device__ inline XXHash_64<bool>::operator()(bool const& key) const
{
  return compute(static_cast<uint8_t>(key));
}

template <>
hash_value_type __device__ inline XXHash_64<float>::operator()(float const& key) const
{
  return compute(normalize_nans(key));
}

template <>
hash_value_type __device__ inline XXHash_64<double>::operator()(double const& key) const
{
  return compute(normalize_nans(key));
}

template <>
hash_value_type __device__ inline XXHash_64<cudf::string_view>::operator()(
  cudf::string_view const& key) const
{
  auto const len = key.size_bytes();
  auto data = device_span<std::byte const>(reinterpret_cast<std::byte const*>(key.data()), len);
  return compute_bytes(data);
}

template <>
hash_value_type __device__ inline XXHash_64<numeric::decimal32>::operator()(
  numeric::decimal32 const& key) const
{
  return compute(key.value());
}

template <>
hash_value_type __device__ inline XXHash_64<numeric::decimal64>::operator()(
  numeric::decimal64 const& key) const
{
  return compute(key.value());
}

template <>
hash_value_type __device__ inline XXHash_64<numeric::decimal128>::operator()(
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
class device_row_hasher {
 public:
  device_row_hasher(Nullate nulls, table_device_view const& t, hash_value_type seed)
    : _check_nulls(nulls), _table(t), _seed(seed)
  {
  }

  __device__ auto operator()(size_type row_index) const noexcept
  {
    return cudf::detail::accumulate(
      _table.begin(),
      _table.end(),
      _seed,
      [row_index, nulls = _check_nulls] __device__(auto hash, auto column) {
        return cudf::type_dispatcher(
          column.type(), element_hasher_adapter{}, column, row_index, nulls, hash);
      });
  }

  /**
   * @brief Computes the hash value of an element in the given column.
   */
  class element_hasher_adapter {
   public:
    template <typename T, CUDF_ENABLE_IF(column_device_view::has_element_accessor<T>())>
    __device__ hash_value_type operator()(column_device_view const& col,
                                          size_type const row_index,
                                          Nullate const _check_nulls,
                                          hash_value_type const _seed) const noexcept
    {
      if (_check_nulls && col.is_null(row_index)) {
        return std::numeric_limits<hash_value_type>::max();
      }
      auto const hasher = XXHash_64<T>{_seed};
      return hasher(col.element<T>(row_index));
    }

    template <typename T, CUDF_ENABLE_IF(not column_device_view::has_element_accessor<T>())>
    __device__ hash_value_type operator()(column_device_view const&,
                                          size_type const,
                                          Nullate const,
                                          hash_value_type const) const noexcept
    {
      CUDF_UNREACHABLE("Unsupported type for XXHash_64");
    }
  };

  Nullate const _check_nulls;
  table_device_view const _table;
  hash_value_type const _seed;
};

}  // namespace

std::unique_ptr<column> xxhash_64(table_view const& input,
                                  uint64_t seed,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr)
{
  auto output = make_numeric_column(data_type(type_to_id<hash_value_type>()),
                                    input.num_rows(),
                                    mask_state::UNALLOCATED,
                                    stream,
                                    mr);

  // Return early if there's nothing to hash
  if (input.num_columns() == 0 || input.num_rows() == 0) { return output; }

  bool const nullable   = has_nulls(input);
  auto const input_view = table_device_view::create(input, stream);
  auto output_view      = output->mutable_view();

  // Compute the hash value for each row
  thrust::tabulate(rmm::exec_policy(stream),
                   output_view.begin<hash_value_type>(),
                   output_view.end<hash_value_type>(),
                   device_row_hasher(nullable, *input_view, seed));

  return output;
}

}  // namespace detail

std::unique_ptr<column> xxhash_64(table_view const& input,
                                  uint64_t seed,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::xxhash_64(input, seed, stream, mr);
}

}  // namespace hashing
}  // namespace cudf
