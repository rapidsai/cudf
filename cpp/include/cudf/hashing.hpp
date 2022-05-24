/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

namespace cudf {

using hash_value_type = uint32_t;  ///< Type of hash value

/**
 * @addtogroup column_hash
 * @{
 * @file
 */

/**
 *  @brief Identifies the hash function to be used
 */
enum class hash_id {
  HASH_IDENTITY = 0,    ///< Identity hash function that simply returns the key to be hashed
  HASH_MURMUR3,         ///< Murmur3 hash function
  HASH_SERIAL_MURMUR3,  ///< Serial Murmur3 hash function
  HASH_SPARK_MURMUR3,   ///< Spark Murmur3 hash function
  HASH_MD5              ///< MD5 hash function
};

/**
 * @brief The default seed value for hash functions
 */
static constexpr uint32_t DEFAULT_HASH_SEED = 0;

/**
 * @brief Computes the hash value of each row in the input set of columns.
 *
 * @param input The table of columns to hash
 * @param hash_function The hash function enum to use
 * @param seed Optional seed value to use for the hash function
 * @param mr Device memory resource used to allocate the returned column's device memory
 *
 * @returns A column where each row is the hash of a column from the input
 */
std::unique_ptr<column> hash(
  table_view const& input,
  hash_id hash_function               = hash_id::HASH_MURMUR3,
  uint32_t seed                       = DEFAULT_HASH_SEED,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/** @} */  // end of group
}  // namespace cudf
