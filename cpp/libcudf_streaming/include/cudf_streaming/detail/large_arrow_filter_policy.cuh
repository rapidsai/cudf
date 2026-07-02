/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuco/bloom_filter_policies.cuh>
#include <cuda/std/limits>

#include <cstdint>

namespace cudf_streaming::detail {

/**
 * @brief Arrow block-split bloom filter policy without Arrow's 128 MiB storage-size limit.
 *
 * This policy preserves Arrow's hash and word-pattern generation. Unlike
 * `cuco::arrow_filter_policy`, it uses the full filter extent when selecting a block instead of
 * clamping the extent to 4,194,304 blocks.
 *
 * @tparam Key The type of values to generate fingerprints for.
 * @tparam XXHash64 64-bit XXHash implementation used for fingerprint generation.
 */
template <class Key, template <typename> class XXHash64>
class large_arrow_filter_policy {
  using arrow_policy_type = cuco::arrow_filter_policy<Key, XXHash64>;

 public:
  using hasher           = typename arrow_policy_type::hasher;
  using word_type        = typename arrow_policy_type::word_type;
  using key_type         = typename arrow_policy_type::key_type;
  using hash_result_type = typename arrow_policy_type::hash_result_type;

  static constexpr auto bits_set_per_block = arrow_policy_type::bits_set_per_block;
  static constexpr auto words_per_block    = arrow_policy_type::words_per_block;

  /**
   * @brief Construct a policy using the given hash function.
   *
   * @param hash Hash function used to generate a key's fingerprint.
   */
  __host__ __device__ constexpr large_arrow_filter_policy(hasher hash = {}) : arrow_policy_{hash} {}

  /**
   * @brief Generate the hash value for a key.
   *
   * @param key Key to hash.
   * @return Hash value for the key.
   */
  __device__ constexpr hash_result_type hash(key_type const& key) const
  {
    return arrow_policy_.hash(key);
  }

  /**
   * @brief Determine the filter block selected by a hash value.
   *
   * @tparam Extent Size type used for the number of filter blocks.
   * @param hash Hash value for the key.
   * @param num_blocks Number of blocks in the filter.
   * @return Index of the selected filter block.
   */
  template <class Extent>
  __host__ __device__ constexpr auto block_index(hash_result_type hash, Extent num_blocks) const
  {
    using size_type          = typename Extent::value_type;
    constexpr auto hash_bits = cuda::std::numeric_limits<word_type>::digits;

    auto const block_hash = hash >> hash_bits;
    auto const blocks     = static_cast<size_type>(num_blocks);

    // Compute (block_hash * blocks) >> 32 without overflowing when size_type is 64 bits.
    auto const low_mask   = hash_result_type{cuda::std::numeric_limits<word_type>::max()};
    auto const blocks_low = blocks & low_mask;
    if constexpr (sizeof(size_type) > sizeof(word_type)) {
      auto const blocks_high = blocks >> hash_bits;
      return static_cast<size_type>(block_hash * blocks_high +
                                    ((block_hash * blocks_low) >> hash_bits));
    } else {
      return static_cast<size_type>((block_hash * blocks_low) >> hash_bits);
    }
  }

  /**
   * @brief Determine the fingerprint pattern for a word within a filter block.
   *
   * @param hash Hash value for the key.
   * @param word_index Target word within the filter block.
   * @return Bit pattern for the target word.
   */
  __device__ constexpr word_type word_pattern(hash_result_type hash, std::uint32_t word_index) const
  {
    return arrow_policy_.word_pattern(hash, word_index);
  }

 private:
  arrow_policy_type arrow_policy_;
};

}  // namespace cudf_streaming::detail
