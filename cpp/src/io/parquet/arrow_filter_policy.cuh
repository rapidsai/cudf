/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/hashing/detail/xxhash_64.cuh>
#include <cudf/types.hpp>

#include <cuda/std/algorithm>
#include <cuda/std/limits>

#include <cstdint>

namespace cudf::io::parquet::detail {

/**
 * @brief A policy that defines how Arrow Block-Split Bloom Filter generates and stores a key's
 * fingerprint.
 *
 * Reference:
 * https://github.com/apache/arrow/blob/be1dcdb96b030639c0b56955c4c62f9d6b03f473/cpp/src/parquet/bloom_filter.cc#L219-L230
 *
 * @tparam Key The type of the values to generate a fingerprint for.
 */
template <class Key>
class arrow_filter_policy {
 public:
  using hasher =
    cudf::hashing::detail::XXHash_64<Key>;  ///< 64-bit XXHash hasher for Arrow bloom filter policy
  using word_type        = std::uint32_t;   ///< uint32_t for Arrow bloom filter policy
  using key_type         = Key;             ///< Hash function input type
  using hash_result_type = std::uint64_t;   ///< hash function output type

  static constexpr std::uint32_t bits_set_per_block = 8;  ///< bits set per Arrow filter block
  static constexpr std::uint32_t words_per_block    = 8;  ///< words per Arrow filter block

  static constexpr std::uint32_t bytes_per_filter_block =
    32;  ///< Number of bytes in one Arrow filter block
  static constexpr std::uint32_t max_arrow_filter_bytes =
    128 * 1024 * 1024;  ///< Max bytes in Arrow bloom filter
  static constexpr std::uint32_t max_filter_blocks =
    (max_arrow_filter_bytes /
     bytes_per_filter_block);  ///< Max sub-filter blocks allowed in Arrow bloom filter

  /**
   * @brief Constructs the `arrow_filter_policy` object.
   *
   * @note The number of filter blocks with Arrow policy must be in the
   * range of [1, 4194304]. If the bloom filter is constructed with a larger
   * number of blocks, only the first 4194304 (128MB) blocks will be used.
   *
   * @param hash Hash function used to generate a key's fingerprint
   */
  CUDF_HOST_DEVICE constexpr arrow_filter_policy(hasher hash = {}) : hash_{hash} {}

  /**
   * @brief Generates the hash value for a given key.
   *
   * @param key The key to hash
   *
   * @return The hash value of the key
   */
  __device__ constexpr hash_result_type hash(key_type const& key) const { return hash_(key); }

  /**
   * @brief Determines the filter block a key is added into.
   *
   * @note The number of filter blocks with Arrow policy must be in the
   * range of [1, 4194304]. Passing a larger `num_blocks` will still
   * upperbound the number of blocks used to the mentioned range.
   *
   * @tparam Extent Size type that is used to determine the number of blocks in the filter
   *
   * @param hash Hash value of the key
   * @param num_blocks Number of block in the filter
   *
   * @return The block index for the given key's hash value
   */
  template <class Extent>
  __device__ constexpr auto block_index(hash_result_type hash, Extent num_blocks) const
  {
    constexpr auto hash_bits = cuda::std::numeric_limits<word_type>::digits;
    auto const max_blocks    = cuda::std::min<Extent>(num_blocks, max_filter_blocks);
    // Make sure we are only contained within the `max_filter_blocks` blocks
    return static_cast<word_type>(((hash >> hash_bits) * max_blocks) >> hash_bits) % max_blocks;
  }

  /**
   * @brief Determines the fingerprint pattern for a word/segment within the filter block for a
   * given key's hash value.
   *
   * @param hash Hash value of the key
   * @param word_index Target word/segment within the filter block
   *
   * @return The bit pattern for the word/segment in the filter block
   */
  __device__ constexpr word_type word_pattern(hash_result_type hash, std::uint32_t word_index) const
  {
    constexpr std::uint32_t salts[words_per_block] = {0x47b6137bU,
                                                      0x44974d91U,
                                                      0x8824ad5bU,
                                                      0xa2b7289dU,
                                                      0x705495c7U,
                                                      0x2df1424bU,
                                                      0x9efc4947U,
                                                      0x5c6bfb31U};
    word_type const key                            = static_cast<word_type>(hash);
    auto const salt                                = salts[word_index];
    return word_type{1} << ((key * salt) >> 27);
  }

 private:
  hasher hash_;
};

}  // namespace cudf::io::parquet::detail
