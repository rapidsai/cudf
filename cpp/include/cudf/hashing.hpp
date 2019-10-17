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

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

namespace cudf {

enum class hash_func {
  HASH_MURMUR3=0, ///< Murmur3 hash function
  HASH_IDENTITY,  ///< Identity hash function that simply returns the key to be hashed
};

/** --------------------------------------------------------------------------*
 * @brief Computes the hash values of the rows in the specified columns of the 
 * input columns and bins the hash values into the desired number of partitions. 
 * Rearranges the input columns such that rows with hash values in the same bin 
 * are contiguous.
 * 
 * @param[in] input The input set of columns
 * @param[in] columns_to_hash Indices of the columns in the input set to hash
 * @param[in] num_partitions The number of partitions to rearrange the input rows into
 * @param[in] hash The hash function to use
 * 
 * @returns A pair of <partioned output, partition offsets>, where the partitioned
 * output is a new table with rows from the input reordered into partitions and
 * the partion offsets are a vector of offsets to the first index of each partition
 * ----------------------------------------------------------------------------**/
std::pair<std::unique_ptr<table>, std::vector<size_type>>
hash_partition(table_view const& input,
               std::vector<size_type> const& columns_to_hash,
               int num_partitions,
               hash_func hash,
               device_memory_resource* mr = rmm::get_default_resource());

/** --------------------------------------------------------------------------*
 * @brief Computes the hash value of each row in the input set of columns.
 *
 * @param[in] input The input set of columns whose rows will be hashed
 * @param[in] hash The hash function to use
 * @param[in] initial_hash_values Optional array in device memory specifying an initial hash value for each column
 * that will be combined with the hash of every element in the column. If this argument is `nullptr`,
 * then each element will be hashed as-is.
 * @param[out] output The hash value of each row of the input
 *
 * @returns A column of has
 * ----------------------------------------------------------------------------**/
std::unique_ptr<column> hash(table_view const& input,
                             hash_func hash,
                             std::vector<size_type> const& initial_hash_values,
                             device_memory_resource* mr = rmm::get_default_resource());

}  // namespace cudf
