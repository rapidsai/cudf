/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <vector>
#include <cudf/cudf.h>
#include <cudf/types.hpp>

namespace cudf {
namespace experimental {

/**
 * @brief Round-robin partition.
 * 
 * Returns a new table with rows re-arranged into partition groups and
 * a vector of row offsets to the start of each partition in the output table.
 * Rows are assigned partitions based on their row index in the table,
 * based on the following formula:
 * partition = (row_index + start_partition) % num_partitions
 *
 * Example 1:
 * input:
 * table => col 1 {0, ..., 12}
 * num_partitions = 3
 * start_partition = 0
 * 
 * output: pair<table, partition_offsets>
 * table => col 1 {0,3,6,9,12,1,4,7,10,2,5,8,11}
 * partition_offsets => {0,5,9}
 *
 * Example 2:
 * input:
 * table => col 1 {0, ..., 12}
 * num_partitions = 3
 * start_partition = 1
 * 
 * output: pair<table, partition_offsets>
 * table => col 1 {2,5,8,11,0,3,6,9,12,1,4,7,10}
 * partition_offsets => {0,4,9}
 * 
 * Example 3:
 * input:
 * table => col 1 {0, ..., 10}
 * num_partitions = 3
 * start_partition = 0
 * 
 * output: pair<table, partition_offsets>
 * table => col 1 {0,3,6,9,1,4,7,10,2,5,8}
 * partition_offsets => {0,4,8}
 *
 * Example 4:
 * input:
 * table => col 1 {0, ..., 10}
 * num_partitions = 3
 * start_partition = 1
 * 
 * output: pair<table, partition_offsets>
 * table => col 1 {2,5,8,0,3,6,9,1,4,7,10}
 * partition_offsets => {0,3,7}
 *
 * Example 5:
 * input:
 * table => col 1 {0, ..., 10}
 * num_partitions = 3
 * start_partition = 2
 * 
 * output: pair<table, partition_offsets>
 * table => col 1 {1,4,7,10,2,5,8,0,3,6,9}
 * partition_offsets => {0,4,7}
 *
 * @Param[in] input The input table to be round-robin partitioned
 * @Param[in] num_partitions Number of partitions for the table
 * @Param[in] start_partition Index of the 1st partition
 * @Param[in] mr Device memory allocator
 *
 * @Returns A std::pair consisting of an unique_ptr to the partitioned table and the partition offsets for each partition within the table
 */
std::pair<std::unique_ptr<cudf::experimental::table>, std::vector<cudf::size_type>>
round_robin_partition(table_view const& input,
                      cudf::size_type num_partitions,
                      cudf::size_type start_partition = 0,
                      rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());
  
}  // namespace experimental
}  // namespace cudf
