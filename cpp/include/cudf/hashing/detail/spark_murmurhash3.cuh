/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/detail/row_operator/hashing.cuh>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/hashing/detail/hash_functions.cuh>
#include <cudf/strings/string_view.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda/functional>
#include <cuda/std/algorithm>
#include <cuda/std/cstddef>
#include <cuda/std/iterator>
#include <cuda/std/type_traits>
#include <thrust/find.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reverse.h>

namespace cudf::hashing::detail {

using spark_hash_value_type = int32_t;

template <typename Key, CUDF_ENABLE_IF(not cudf::is_nested<Key>())>
struct Spark_MurmurHash3_x86_32 {
  using result_type = spark_hash_value_type;

  CUDF_HOST_DEVICE constexpr Spark_MurmurHash3_x86_32() = delete;
  CUDF_HOST_DEVICE constexpr Spark_MurmurHash3_x86_32(result_type seed)
    : m_seed(static_cast<uint32_t>(seed))
  {
  }

  [[nodiscard]] __device__ inline uint32_t fmix32(uint32_t h) const
  {
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
  }

  [[nodiscard]] __device__ inline uint32_t getblock32(cuda::std::byte const* data,
                                                      cudf::size_type offset) const
  {
    // Read a 4-byte value from the data pointer as individual bytes for safe
    // unaligned access (very likely for string types).
    auto block = reinterpret_cast<uint8_t const*>(data + offset);
    return block[0] | (block[1] << 8) | (block[2] << 16) | (block[3] << 24);
  }

  [[nodiscard]] result_type __device__ inline operator()(Key const& key) const
  {
    return compute(key);
  }

  template <typename T>
  result_type __device__ inline compute(T const& key) const
  {
    return compute_bytes(reinterpret_cast<cuda::std::byte const*>(&key), sizeof(T));
  }

  result_type __device__ inline compute_remaining_bytes(cuda::std::byte const* data,
                                                        cudf::size_type len,
                                                        cudf::size_type tail_offset,
                                                        result_type h) const
  {
    // Process remaining bytes that do not fill a four-byte chunk using Spark's approach
    // (does not conform to normal MurmurHash3).
    for (auto i = tail_offset; i < len; i++) {
      // We require a two-step cast to get the k1 value from the byte. First,
      // we must cast to a signed int8_t. Then, the sign bit is preserved when
      // casting to uint32_t under 2's complement. Java preserves the sign when
      // casting byte-to-int, but C++ does not.
      uint32_t k1 = static_cast<uint32_t>(cuda::std::to_integer<int8_t>(data[i]));
      k1 *= c1;
      k1 = rotate_bits_left(k1, rot_c1);
      k1 *= c2;
      h ^= k1;
      h = rotate_bits_left(static_cast<uint32_t>(h), rot_c2);
      h = h * 5 + c3;
    }
    return h;
  }

  result_type __device__ compute_bytes(cuda::std::byte const* data, cudf::size_type const len) const
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
      h = rotate_bits_left(static_cast<uint32_t>(h), rot_c2);
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
spark_hash_value_type __device__ inline Spark_MurmurHash3_x86_32<bool>::operator()(
  bool const& key) const
{
  return compute<uint32_t>(key);
}

template <>
spark_hash_value_type __device__ inline Spark_MurmurHash3_x86_32<int8_t>::operator()(
  int8_t const& key) const
{
  return compute<uint32_t>(key);
}

template <>
spark_hash_value_type __device__ inline Spark_MurmurHash3_x86_32<uint8_t>::operator()(
  uint8_t const& key) const
{
  return compute<uint32_t>(key);
}

template <>
spark_hash_value_type __device__ inline Spark_MurmurHash3_x86_32<int16_t>::operator()(
  int16_t const& key) const
{
  return compute<uint32_t>(key);
}

template <>
spark_hash_value_type __device__ inline Spark_MurmurHash3_x86_32<uint16_t>::operator()(
  uint16_t const& key) const
{
  return compute<uint32_t>(key);
}

template <>
spark_hash_value_type __device__ inline Spark_MurmurHash3_x86_32<float>::operator()(
  float const& key) const
{
  return compute<float>(normalize_nans(key));
}

template <>
spark_hash_value_type __device__ inline Spark_MurmurHash3_x86_32<double>::operator()(
  double const& key) const
{
  return compute<double>(normalize_nans(key));
}

template <>
spark_hash_value_type __device__ inline Spark_MurmurHash3_x86_32<cudf::string_view>::operator()(
  cudf::string_view const& key) const
{
  auto const data = reinterpret_cast<cuda::std::byte const*>(key.data());
  auto const len  = key.size_bytes();
  return compute_bytes(data, len);
}

template <>
spark_hash_value_type __device__ inline Spark_MurmurHash3_x86_32<numeric::decimal32>::operator()(
  numeric::decimal32 const& key) const
{
  return compute<uint64_t>(key.value());
}

template <>
spark_hash_value_type __device__ inline Spark_MurmurHash3_x86_32<numeric::decimal64>::operator()(
  numeric::decimal64 const& key) const
{
  return compute<uint64_t>(key.value());
}

template <>
spark_hash_value_type __device__ inline Spark_MurmurHash3_x86_32<numeric::decimal128>::operator()(
  numeric::decimal128 const& key) const
{
  // Generates the Spark MurmurHash3 hash value, mimicking the conversion:
  // java.math.BigDecimal.valueOf(unscaled_value, _scale).unscaledValue().toByteArray()
  // https://github.com/apache/spark/blob/master/sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/expressions/hash.scala#L381
  __int128_t const val               = key.value();
  constexpr cudf::size_type key_size = sizeof(__int128_t);
  cuda::std::byte const* data        = reinterpret_cast<cuda::std::byte const*>(&val);

  // Small negative values start with 0xff..., small positive values start with 0x00...
  bool const is_negative           = val < 0;
  cuda::std::byte const zero_value = is_negative ? cuda::std::byte{0xff} : cuda::std::byte{0x00};

  // If the value can be represented with a shorter than 16-byte integer, the
  // leading bytes of the little-endian value are truncated and are not hashed.
  auto const reverse_begin = cuda::std::reverse_iterator(data + key_size);
  auto const reverse_end   = cuda::std::reverse_iterator(data);
  auto const first_nonzero_byte =
    thrust::find_if_not(thrust::seq,
                        reverse_begin,
                        reverse_end,
                        [zero_value](cuda::std::byte const& v) { return v == zero_value; })
      .base();
  // Max handles special case of 0 and -1 which would shorten to 0 length otherwise
  cudf::size_type length =
    cuda::std::max(1, static_cast<cudf::size_type>(cuda::std::distance(data, first_nonzero_byte)));

  // Preserve the 2's complement sign bit by adding a byte back on if necessary.
  // e.g. 0x0000ff would shorten to 0x00ff. The 0x00 byte is retained to
  // preserve the sign bit, rather than leaving an "f" at the front which would
  // change the sign bit. However, 0x00007f would shorten to 0x7f. No extra byte
  // is needed because the leftmost bit matches the sign bit. Similarly for
  // negative values: 0xffff00 --> 0xff00 and 0xffff80 --> 0x80.
  if ((length < key_size) && (is_negative ^ bool(data[length - 1] & cuda::std::byte{0x80}))) {
    ++length;
  }

  // Convert to big endian by reversing the range of nonzero bytes. Only those bytes are hashed.
  __int128_t big_endian_value = 0;
  auto big_endian_data        = reinterpret_cast<cuda::std::byte*>(&big_endian_value);
  thrust::reverse_copy(thrust::seq, data, data + length, big_endian_data);
  return compute_bytes(big_endian_data, length);
}

/**
 * @brief Computes the hash value of a row in the given table.
 *
 * This functor uses Spark conventions for Murmur hashing, which differs from
 * the Murmur implementation used in the rest of libcudf. These differences
 * include:
 * - Serially using the output hash as an input seed for the next item
 * - Ignorance of null values
 *
 * The serial use of hashes as seeds means that data of different nested types
 * can exhibit hash collisions. For example, a row of an integer column
 * containing a 1 will have the same hash as a lists column of integers
 * containing a list of [1] and a struct column of a single integer column
 * containing a struct of {1}.
 *
 * As a consequence of ignoring null values, inputs like [1], [1, null], and
 * [null, 1] have the same hash (an expected hash collision). This kind of
 * collision can also occur across a table of nullable columns and with nulls
 * in structs ({1, null} and {null, 1} have the same hash). The seed value (the
 * previous element's hash value) is returned as the hash if an element is
 * null.
 *
 * For additional differences such as special tail processing and decimal type
 * handling, refer to the Spark_MurmurHash3_x86_32 functor.
 *
 * @tparam hash_function Hash functor to use for hashing elements. Must be Spark_MurmurHash3_x86_32.
 * @tparam Nullate A cudf::nullate type describing whether to check for nulls.
 */
template <template <typename> class hash_function, typename Nullate>
class spark_murmur_device_row_hasher {
  friend class cudf::detail::row::hash::row_hasher;  ///< Allow row_hasher to access private
                                                     ///< members.

 public:
  /**
   * @brief Return the hash value of a row in the given table.
   *
   * @param row_index The row index to compute the hash value of
   * @return The hash value of the row
   */
  __device__ auto operator()(size_type row_index) const noexcept
  {
    return cudf::detail::accumulate(
      _table.begin(),
      _table.end(),
      _seed,
      cuda::proclaim_return_type<spark_hash_value_type>(
        [row_index, nulls = this->_check_nulls] __device__(auto hash, auto column) {
          return cudf::type_dispatcher(
            column.type(), element_hasher_adapter<hash_function>{nulls, hash}, column, row_index);
        }));
  }

 private:
  /**
   * @brief Computes the hash value of an element in the given column.
   *
   * When the column is non-nested, this is a simple wrapper around the element_hasher.
   * When the column is nested, this uses a seed value to serially compute each
   * nested element, with the output hash becoming the seed for the next value.
   * This requires constructing a new hash functor for each nested element,
   * using the new seed from the previous element's hash. The hash of a null
   * element is the input seed (the previous element's hash).
   */
  template <template <typename> class hash_fn>
  class element_hasher_adapter {
   public:
    using hash_functor = cudf::detail::row::hash::element_hasher<hash_fn, Nullate>;
    using result_type  = typename hash_functor::result_type;

    __device__ element_hasher_adapter(Nullate check_nulls, result_type seed) noexcept
      : _check_nulls(check_nulls), _seed(seed)
    {
    }

    template <typename T, CUDF_ENABLE_IF(not cudf::is_nested<T>())>
    __device__ spark_hash_value_type operator()(column_device_view const& col,
                                                size_type row_index) const noexcept
    {
      auto const hasher = hash_functor{_check_nulls, _seed, _seed};
      return hasher.template operator()<T>(col, row_index);
    }

    template <typename T, CUDF_ENABLE_IF(cudf::is_nested<T>())>
    __device__ spark_hash_value_type operator()(column_device_view const& col,
                                                size_type row_index) const noexcept
    {
      column_device_view curr_col = col.slice(row_index, 1);
      while (curr_col.type().id() == type_id::STRUCT || curr_col.type().id() == type_id::LIST) {
        if (curr_col.type().id() == type_id::STRUCT) {
          if (curr_col.num_child_columns() == 0) { return _seed; }
          // Non-empty structs are assumed to be decomposed and contain only one child
          curr_col = cudf::structs_column_device_view(curr_col).get_sliced_child(0);
        } else if (curr_col.type().id() == type_id::LIST) {
          curr_col = cudf::lists_column_device_view(curr_col).get_sliced_child();
        }
      }

      return cudf::detail::accumulate(
        thrust::counting_iterator(0),
        thrust::counting_iterator(curr_col.size()),
        _seed,
        [curr_col, nulls = this->_check_nulls] __device__(auto hash, auto element_index) {
          auto const hasher = hash_functor{nulls, hash, hash};
          return cudf::type_dispatcher<cudf::detail::dispatch_void_if_nested>(
            curr_col.type(), hasher, curr_col, element_index);
        });
    }

    Nullate const _check_nulls;  ///< Whether to check for nulls
    result_type const _seed;     ///< The seed to use for hashing, also returned for null elements
  };

  using result_type = typename element_hasher_adapter<hash_function>::result_type;

  CUDF_HOST_DEVICE spark_murmur_device_row_hasher(Nullate check_nulls,
                                                  table_device_view t,
                                                  result_type seed = DEFAULT_HASH_SEED) noexcept
    : _check_nulls{check_nulls}, _table{t}, _seed(seed)
  {
    // Error out if passed an unsupported hash_function
    static_assert(
      cuda::std::is_same_v<Spark_MurmurHash3_x86_32<int>, hash_function<int>>,
      "spark_murmur_device_row_hasher only supports the Spark_MurmurHash3_x86_32 hash function");
  }

  Nullate const _check_nulls;
  table_device_view const _table;
  result_type const _seed;
};

/**
 * @brief Throws if `input` contains a nested shape unsupported by Spark MurmurHash3.
 *
 * @param input Table to validate
 */
void check_spark_murmurhash3_compatibility(table_view const& input);

}  // namespace cudf::hashing::detail
