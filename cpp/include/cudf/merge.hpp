/*
 * Copyright (c) 2018-2019, NVIDIA CORPORATION.
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
 * @brief Merge sorted tables.
 * 
 * Merges two sorted tables into one sorted table
 * containing data from both tables.
 *
 * Example 1:
 * input:
 * table 1 => col 1 {0, 1, 2, 3}
 *            col 2 {4, 5, 6, 7}
 * table 2 => col 1 {1, 2}
 *            col 2 {8, 9}
 * output:
 * table => col 1 {0, 1, 1, 2, 2, 3}
 *          col 2 {4, 5, 8, 6, 9, 7}
 *
 * Example 2: 
 * input:
 * table 1 => col 0 {1, 0}
 *            col 1 {'c', 'b'}
 *            col 2 {RED, GREEN}
 *
 *
 * table 2 => col 0 {1}
 *            col 1 {'a'}
 *            col 2 {NULL}
 *
 *  with key_cols[] = {0,1}
 *  and  asc_desc[] = {ASC, ASC};
 *
 *  Lex-sorting is on columns {0,1}; hence, lex-sorting of ((L0 x L1) V (R0 x R1)) is:
 *  (0,'b', GREEN), (1,'a', NULL), (1,'c', RED)
 *
 *  (third column, the "color", just "goes along for the ride"; 
 *   meaning is permutted according to the data movements dictated 
 *   by lexicographic ordering of columns 0 and 1);
 *
 *   with result columns:
 *
 *   Res0 = {0,1,1}
 *   Res1 = {'b', 'a', 'c'}
 *   Res2 = {GREEN, NULL, RED}
 *
 * @Param[in] left_table A sorted table to be merged
 * @Param[in] right_table A sorted table to be merged
 * @Param[in] key_cols Indices of left_cols and right_cols to be used
 *                     for comparison criteria
 * @Param[in] column_order Sort order types of columns indexed by key_cols
 * @Param[in] null_precedence Array indicating the order of nulls with respect to non-nulls for the indexing columns (key_cols)
 *
 * @Returns A table containing sorted data from left_table and right_table
 */
std::unique_ptr<cudf::experimental::table> merge(table_view const& left_table,
                                                 table_view const& right_table,
                                                 std::vector<cudf::size_type> const& key_cols,
                                                 std::vector<cudf::order> const& column_order,
                                                 std::vector<cudf::null_order> const& null_precedence = {},
                                                 rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

}  // namespace experimental
}  // namespace cudf

