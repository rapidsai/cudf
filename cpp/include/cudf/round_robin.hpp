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
 * in a round robin fashion.
 *
 * @throws cudf::logic_error if num_partitions <= 1 
 * @throws cudf::logic_error if start_partition >= num_partitions.
 *
 * A good analogy for the algorithm is dealing out cards:
 *
 *  1. The deck of cards is represented as the rows in the table.
 *  2. The number of partitions is the number of players being dealt cards.
 *  3. the start_partition indicates which player starts getting cards first.
 *
 * The algorithm has two outcomes:
 *
 *  (a) Another deck of cards formed by stacking each 
 *      player's cards back into a deck again, 
 *      preserving the order of cards dealt to each player, 
 *      starting with player 0.
 *  (b) A vector into the output deck indicating where a player's cards start.
 *
 * A player's deck (partition) is the range of cards starting 
 * at the corresponding offset and ending at the next player's 
 * starting offset or the last card in the deck if it's the last player.
 *
 * When num_partitions > nrows, we have more players than decks. 
 * We start dealing to the first indicated player and continuing 
 * around the players until we run out of cards before we run out of players. 
 * Players that did not get any cards are represented by
 * offset[i] == offset[i+1] or
 * offset[i] == table.num_rows() if i == num_partitions-1
 * meaning there are no cards (rows) in their deck (partition).
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
 * Example 6:
 * input:
 * table => col 1 {0, ..., 10}
 * num_partitions = 15 > num_rows = 11
 * start_partition = 2
 * 
 * output: pair<table, partition_offsets>
 * table => col 1 {0,1,2,3,4,5,6,7,8,9,10}
 * partition_offsets => {0,0,0,1,2,3,4,5,6,7,8,9,10,11,11}
 *
 * Example 7:
 * input:
 * table => col 1 {0, ..., 10}
 * num_partitions = 15 > num_rows = 11
 * start_partition = 10
 * 
 * output: pair<table, partition_offsets>
 * table => col 1 {5,6,7,8,9,10,0,1,2,3,4}
 * partition_offsets => {0,1,2,3,4,5,6,6,6,6,6,7,8,9,10}
 *
 * Example 8:
 * input:
 * table => col 1 {0, ..., 10}
 * num_partitions = 15 > num_rows = 11
 * start_partition = 14
 * 
 * output: pair<table, partition_offsets>
 * table => col 1 {1,2,3,4,5,6,7,8,9,10,0}
 * partition_offsets => {0,1,2,3,4,5,6,7,8,9,10,10,10,10,10}
 *
 * Example 9:
 * input:
 * table => col 1 {0, ..., 10}
 * num_partitions = 11 == num_rows = 11
 * start_partition = 2
 * 
 * output: pair<table, partition_offsets>
 * table => col 1 {9,10,0,1,2,3,4,5,6,7,8}
 * partition_offsets => {0,1,2,3,4,5,6,7,8,9,10}
 *
 * @param[in] input The input table to be round-robin partitioned
 * @param[in] num_partitions Number of partitions for the table
 * @param[in] start_partition Index of the 1st partition
 * @param[in] mr Device memory allocator
 *
 * @return A std::pair consisting of an unique_ptr to the partitioned table 
 * and the partition offsets for each partition within the table.
 */
std::pair<std::unique_ptr<cudf::experimental::table>, std::vector<cudf::size_type>>
round_robin_partition(table_view const& input,
                      cudf::size_type num_partitions,
                      cudf::size_type start_partition = 0,
                      rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());
  
}  // namespace experimental
}  // namespace cudf
