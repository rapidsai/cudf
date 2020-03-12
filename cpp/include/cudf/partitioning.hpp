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

#include <cudf/types.hpp>
#include <memory>
#include <vector>

namespace cudf {
namespace experimental {
/**
 * @brief Partitions rows from the input table into multiple output tables.
 *
 * Partitions rows of `input` into `num_partitions` bins based on the hash
 * value of the columns specified by `columns_to_hash`. Rows partitioned into
 * the same bin are grouped consecutively in the output table. Returns a vector
 * of row offsets to the start of each partition in the output table.
 *
 * @throw std::out_of_range if index is `columns_to_hash` is invalid
 *
 * @param input The table to partition
 * @param columns_to_hash Indices of input columns to hash
 * @param num_partitions The number of partitions to use
 * @param mr Optional resource to use for device memory allocation
 *
 * @returns An output table and a vector of row offsets to each partition
 */
std::pair<std::unique_ptr<experimental::table>, std::vector<size_type>>
hash_partition(
    table_view const& input, std::vector<size_type> const& columns_to_hash,
    int num_partitions,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

}  // namespace experimental
}  // namespace cudf
