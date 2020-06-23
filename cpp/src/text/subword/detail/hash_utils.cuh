/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <rmm/thrust_rmm_allocator.h>

namespace nvtext {
namespace detail {

// PRIME is constant for both levels and a and b are constant for
// the first level hash.
constexpr uint64_t PRIME = 281474976710677;

/**
 * @brief This does a multiply mod 48 without overflow for the sdbm hash "pop" method.
 *
 * This method computes the bottom 48 bits of the result of multiplying two numbers
 * respecting the restrictions described in params.
 *
 * It works by splitting num into reps 16 bit chunks and performing reps multiplies.
 * The result of all of those multiplies are added together.
 *
 * @param num_48bit A multiplicand that is at most 48 bits.
 * @param num Any 64 bit number to multiply by num_48bit mod 2**48
 * @return (num_48bit * num) mod 2**48
 */
__device__ __forceinline__ uint64_t mul_mod_48(uint64_t num_48bit, uint64_t num)
{
  constexpr uint64_t mask          = (1ULL << 48) - 1;
  constexpr uint8_t bit_chunk_size = 16;

  uint64_t result = 0;
#pragma unroll
  for (uint8_t i = 0; i < sizeof(num) / 2; ++i) {
    auto const shift_amt = bit_chunk_size * i;
    auto const bottom_16 = static_cast<uint16_t>(num >> shift_amt);
    // update result
    result = result + ((num_48bit * bottom_16) << shift_amt);
    result &= mask;
  }
  return result;
}

/**
 * @brief Computes the sdbm hash for the sequence starting at sequence_start up to length sequences.
 *
 * A start value for the sdbm hash can optionally be given. This is useful when checking if elements
 * starting with "##" exist in the table since we can pass in the hash of "##" as the start value.
 *
 * @param sequence_start Code points to hash
 * @param length Number of code points to has
 * @param start_value Initializes the hash computation.
 * @return The sdbm hash of all elements in range `[sequence_start, sequence_start + length)`
 */
__device__ __forceinline__ uint64_t sdbm_hash(uint32_t* sequence_start,
                                              uint32_t length,
                                              uint64_t start_value = 0)
{
  // This expression computes h_{i} = (65599*h{i-1} + new_val) mod 2^48 and was obtained from here:
  // http://www.cse.yorku.ca/~oz/hash.html

  constexpr uint64_t mask = (1ULL << 48) - 1;
  uint64_t hash_value     = start_value;

  for (int i = 0; i < length; ++i) {
    hash_value = ((hash_value << 6) + (hash_value << 16) - hash_value) & mask;
    hash_value = (hash_value + (sequence_start[i] & mask)) & mask;
  }

  return hash_value;
}

/**
 * @brief Removes the last value added to the hash.
 *
 * If we have `current_hash = sdbm_hash("dog")` then, `prev_sdbm_hash(current_hash, cp(g))`
 * returns the `sdbm_hash("do")` where it is assumed cp returns the unicode code point for a
 * given letter.
 *
 * @return The hash value before that new value was added.
 */
__device__ __forceinline__ uint64_t prev_sdbm_hash(uint64_t current_hash, uint32_t last_val)
{
  constexpr uint64_t mask = (1ULL << 48) - 1;
  // Multiplicative inverse of 65599 under mod 2**48
  constexpr uint64_t mod_inverse = 24320495251391;
  uint64_t prev_hash = mul_mod_48(mod_inverse, current_hash) - mul_mod_48(mod_inverse, last_val);
  return prev_hash & mask;
}

/**
 * @brief The hash function used for accesses to the table.
 *
 * This is a universal hash function with parameters chosen to achieve perfect hashing.
 *
 * Algorithm is `((a*k + b) % PRIME) % table_size` where PRIME is globally defined
 * as 281474976710677
 *
 * @return The computed hash value.
 */
__device__ __forceinline__ uint32_t hash(uint64_t key, uint64_t a, uint64_t b, uint32_t table_size)
{
  return ((a * key + b) % PRIME) % table_size;
}

/**
 * @brief Retrieves the value associated with key in the hash table.
 *
 * If there is no value in the table with the input key, -1 is returned.
 *
 * This method will ALWAYS return the correct value if a key is in the table however, some
 * code point sequences may hash to the same key in which case an incorrect value is returned.
 * This is unlikely and the times this occurs are unlikely to affect the model's performance.
 *
 * @param key The key to search for in the hash table
 * @param hash_table A pointer to the flattened hash table
 * @param bin_coefficients A pointer to the hashing parameters for each bin in the hash table.
 * @param bin_offsets A pointer to the start of each bin in the hash table.
 * @return -1 if key is not in the hash table. If the key is in the table returns an index in
 *         [0, vocab_size) indicating the index for the token in the bert model.
 */
__device__ __forceinline__ int retrieve(uint64_t key,
                                        uint32_t outer_table_a,
                                        uint32_t outer_table_b,
                                        uint16_t num_bins,
                                        uint64_t* hash_table,
                                        uint64_t* bin_coefficients,
                                        uint16_t* bin_offsets)
{
  auto const hash_bin        = hash(key, outer_table_a, outer_table_b, num_bins);
  auto const bin_params      = bin_coefficients[hash_bin];
  auto const start_ht_offset = bin_offsets[hash_bin];

  // The shift constants are due to how the hash coefficients are packed and are
  // obtained from the python script perfect_hash.py which generates the expected
  // tables.
  auto const inner_bin_a = bin_params >> 16;
  auto const inner_bin_b = (bin_params >> 9) & ((1 << 7) - 1);
  auto const bin_size    = static_cast<uint8_t>(bin_params);

  auto const inner_offset = hash(key, inner_bin_a, inner_bin_b, bin_size);
  auto const kv_pair      = hash_table[start_ht_offset + inner_offset];

  auto const expected_key = kv_pair >> 16;
  // extract value from encoded key-value
  int value = kv_pair & ((1 << 16) - 1);
  return key == expected_key ? value : -1;
}

/**
 * @brief Loads the hashing information from hash_data_file AFTER running the perfect_hash.py python
 * script on the bert vocabulary file.
 *
 * The file includes values from a two level hashing scheme based on the FKS perfect hashing
 * algorithm.
 *
 * This function reads the text file of values and loads them into the parameter objects.
 *
 *
 * @param hash_data_file The path to the file containing the hashing information after the python
 *        script has been stored.
 * @param device_hash_table[out] A pointer to the flattened hash table values.
 * @param device_bin_coefficients A pointer to the data containing the hashing parameters
 *        for each hash bin on the GPU.
 * @param device_bin_offsets A pointer to the data containing the start index of each bin in
 *        the flattened hash table.
 * @param unk_tok_id The token id for unknown tokens
 * @param first_tok_id The token id for the starting of the first string
 * @param sep_tok_id The token id for string separators
 * @param outer_table_a The outer hash value parameter A
 * @param outer_table_b The outer hash value parameter B
 * @param num_bins The number of bins for the outer hash
 */
void transfer_hash_info_to_device(const std::string hash_data_file,
                                  rmm::device_vector<uint64_t>& device_hash_table,
                                  rmm::device_vector<uint64_t>& device_bin_coefficients,
                                  rmm::device_vector<uint16_t>& device_bin_offsets,
                                  uint16_t& unk_tok_id,
                                  uint16_t& first_tok_id,
                                  uint16_t& sep_tok_id,
                                  uint32_t& outer_table_a,
                                  uint32_t& outer_table_b,
                                  uint16_t& num_bins);

}  // namespace detail
}  // namespace nvtext
