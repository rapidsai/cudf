/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cudf/hashing.hpp>

namespace cudf {
namespace detail {

/** --------------------------------------------------------------------------*
 * @brief Partitions rows from the input table into multiple output tables.
 *
 * Partitions rows of `input` into `num_partitions` bins based on the hash
 * value of the columns specified by `columns_to_hash`. Rows partitioned into
 * the same bin are grouped together into a new table. Returns a vector
 * containing `num_partitions` new tables.
 *
 * @throw std::out_of_range if index is `columns_to_hash` is invalid
 *
 * @param input The table to partition
 * @param columns_to_hash Indices of input columns to hash
 * @param num_partitions The number of partitions to use
 * @param mr Optional resource to use for device memory allocation
 * @param stream Optional stream to use for allocations and copies
 *
 * @returns A vector of tables partitioned from the input
 * -------------------------------------------------------------------------**/
std::vector<std::unique_ptr<experimental::table>>
hash_partition(table_view const& input,
               std::vector<size_type> const& columns_to_hash,
               int num_partitions,
               rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
               cudaStream_t stream = 0);

/** --------------------------------------------------------------------------*
 * @brief Computes the hash value of each row in the input set of columns.
 *
 * @param input The table of columns to hash
 * @param initial_hash Optional vector of initial hash values for each column.
 * If this vector is empty then each element will be hashed as-is.
 * @param mr Optional resource to use for device memory allocation
 * @param stream Optional stream to use for allocations and copies
 *
 * @returns A column where each row is the hash of a column from the input
 * -------------------------------------------------------------------------**/
std::unique_ptr<column> hash(table_view const& input,
                             std::vector<uint32_t> const& initial_hash = {},
                             rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                             cudaStream_t stream = 0);

}  // namespace detail
}  // namespace cudf
