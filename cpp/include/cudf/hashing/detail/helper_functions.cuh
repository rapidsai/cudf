/*
 * Copyright (c) 2017-2024, NVIDIA CORPORATION.
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

#include <cudf/types.hpp>

#include <thrust/pair.h>

constexpr int64_t DEFAULT_HASH_TABLE_OCCUPANCY = 50;

/**
 * @brief  Compute requisite size of hash table.
 *
 * Computes the number of entries required in a hash table to satisfy
 * inserting a specified number of keys to achieve the specified hash table
 * occupancy.
 *
 * @param num_keys_to_insert The number of keys that will be inserted
 * @param desired_occupancy The desired occupancy percentage, e.g., 50 implies a
 * 50% occupancy
 * @return size_t The size of the hash table that will satisfy the desired
 * occupancy for the specified number of insertions
 */
inline size_t compute_hash_table_size(cudf::size_type num_keys_to_insert,
                                      uint32_t desired_occupancy = DEFAULT_HASH_TABLE_OCCUPANCY)
{
  assert(desired_occupancy != 0);
  assert(desired_occupancy <= 100);
  double const grow_factor{100.0 / desired_occupancy};

  // Calculate size of hash map based on the desired occupancy
  size_t hash_table_size{static_cast<size_t>(std::ceil(num_keys_to_insert * grow_factor))};

  return hash_table_size;
}
