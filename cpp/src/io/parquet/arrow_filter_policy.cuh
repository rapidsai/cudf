/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <cuco/hash_functions.cuh>
#include <cuda/functional>
#include <cuda/std/bit>
#include <cuda/std/limits>

#include <cstdint>
#include <nv/target>

namespace cuco {

/**
 * @brief A policy that defines how Arrow Block-Split Bloom Filter generates and stores a key's
 * fingerprint.
 *
 * Reference:
 * https://github.com/apache/arrow/blob/be1dcdb96b030639c0b56955c4c62f9d6b03f473/cpp/src/parquet/bloom_filter.cc#L219-L230
 *
 * Example:
 * @code{.cpp}
 * template <typename KeyType, std::uint32_t NUM_FILTER_BLOCKS>
 * void bulk_insert_and_eval_arrow_policy_bloom_filter(device_vector<KeyType> const& positive_keys,
 *                                                 device_vector<KeyType> const& negative_keys)
 * {
 *     using policy_type = cuco::arrow_filter_policy<KeyType, cuco::xxhash_64>;
 *
 *     // Warn or throw if the number of filter blocks is greater than maximum used by Arrow policy.
 *     static_assert(NUM_FILTER_BLOCKS <= policy_type::max_filter_blocks, "NUM_FILTER_BLOCKS must be
 *                                                                         in range: [1, 4194304]");
 *
 *     // Create a bloom filter with Arrow policy
 *     cuco::bloom_filter<KeyType, cuco::extent<size_t>,
 *         cuda::thread_scope_device, policy_type> filter{NUM_FILTER_BLOCKS};
 *
 *     // Add positive keys to the bloom filter
 *     filter.add(positive_keys.begin(), positive_keys.end());
 *
 *     auto const num_tp = positive_keys.size();
 *     auto const num_tn = negative_keys.size();
 *
 *     // Vectors to store query results.
 *     thrust::device_vector<bool> true_positive_result(num_tp, false);
 *     thrust::device_vector<bool> true_negative_result(num_tn, false);
 *
 *     // Query the bloom filter for the inserted keys.
 *     filter.contains(positive_keys.begin(), positive_keys.end(), true_positive_result.begin());
 *
 *     // We should see a true-positive rate of 1.
 *     float true_positive_rate = float(thrust::count(thrust::device,
 *          true_positive_result.begin(), true_positive_result.end(), true)) / float(num_tp);
 *
 *     // Query the bloom filter for the non-inserted keys.
 *     filter.contains(negative_keys.begin(), negative_keys.end(), true_negative_result.begin());
 *
 *     // We may see a false-positive rate > 0 depending on the number of bits in the
 *     // filter and the number of hashes used per key.
 *     float false_positive_rate = float(thrust::count(thrust::device,
 *          true_negative_result.begin(), true_negative_result.end(), true)) / float(num_tn);
 * }
 * @endcode
 *
 * @tparam Key The type of the values to generate a fingerprint for.
 * @tparam XXHash64 64-bit XXHash hasher implementation for fingerprint generation.
 */
template <class Key, template <typename> class XXHash64>
class arrow_filter_policy {
 public:
  using hasher          = XXHash64<Key>;  ///< 64-bit XXHash hasher for Arrow bloom filter policy
  using word_type       = std::uint32_t;  ///< uint32_t for Arrow bloom filter policy
  using key_type        = Key;            ///< Hash function input type
  using hash_value_type = std::uint64_t;  ///< hash function output type

  static constexpr uint32_t bits_set_per_block = 8;  ///< hardcoded bits set per Arrow filter block
  static constexpr uint32_t words_per_block    = 8;  ///< hardcoded words per Arrow filter block

  static constexpr std::uint32_t bytes_per_filter_block =
    32;  ///< Number of bytes in one Arrow filter block
  static constexpr std::uint32_t max_arrow_filter_bytes =
    128 * 1024 * 1024;  ///< Max bytes in Arrow bloom filter
  static constexpr std::uint32_t max_filter_blocks =
    (max_arrow_filter_bytes /
     bytes_per_filter_block);  ///< Max sub-filter blocks allowed in Arrow bloom filter

 private:
  // Arrow's block-based bloom filter algorithm needs these eight odd SALT values to calculate
  // eight indexes of bit to set, one bit in each 32-bit (uint32_t) word.
  __device__ static constexpr cuda::std::array<std::uint32_t, 8> SALT()
  {
    return {0x47b6137bU,
            0x44974d91U,
            0x8824ad5bU,
            0xa2b7289dU,
            0x705495c7U,
            0x2df1424bU,
            0x9efc4947U,
            0x5c6bfb31U};
  }

 public:
  /**
   * @brief Constructs the `arrow_filter_policy` object.
   *
   * @note The number of filter blocks with Arrow policy must be in the
   * range of [1, 4194304]. If the bloom filter is constructed with a larger
   * number of blocks, only the first 4194304 (128MB) blocks will be used.
   *
   * @param hash Hash function used to generate a key's fingerprint
   */
  __host__ __device__ constexpr arrow_filter_policy(hasher hash = {}) : hash_{hash} {}

  /**
   * @brief Generates the hash value for a given key.
   *
   * @param key The key to hash
   *
   * @return The hash value of the key
   */
  __device__ constexpr hash_value_type hash(key_type const& key) const { return hash_(key); }

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
  __device__ constexpr auto block_index(hash_value_type hash, Extent num_blocks) const
  {
    constexpr auto hash_bits = cuda::std::numeric_limits<word_type>::digits;
    // TODO: assert if num_blocks > max_filter_blocks
    auto const max_blocks = cuda::std::min<Extent>(num_blocks, max_filter_blocks);
    // Make sure we are only contained withing the `max_filter_blocks` blocks
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
  __device__ constexpr word_type word_pattern(hash_value_type hash, std::uint32_t word_index) const
  {
    // SALT array to calculate bit indexes for the current word
    auto constexpr salt = SALT();
    word_type const key = static_cast<word_type>(hash);
    return word_type{1} << ((key * salt[word_index]) >> 27);
  }

 private:
  hasher hash_;
};

}  // namespace cuco
