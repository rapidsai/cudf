/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.
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
/**
 * @addtogroup column_merge
 * @{
 * @file
 */

/**
 * @brief Merge a set of sorted tables.
 *
 * Merges sorted tables into one sorted table
 * containing data from all tables.
 *
 * ```
 * Example 1:
 * input:
 * table 1 => col 1 {0, 1, 2, 3}
 *            col 2 {4, 5, 6, 7}
 * table 2 => col 1 {1, 2}
 *            col 2 {8, 9}
 * table 3 => col 1 {2, 4}
 *            col 2 {8, 9}
 * output:
 * table => col 1 {0, 1, 1, 2, 2, 2, 3, 4}
 *          col 2 {4, 5, 8, 6, 8, 9, 7, 9}
 * ```
 * ```
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
 * ```
 *
 * @throws cudf::logic_error if tables in `tables_to_merge` have different
 * number of columns
 * @throws cudf::logic_error if tables in `tables_to_merge` have columns with
 * mismatched types
 * @throws cudf::logic_error if `key_cols` is empty
 * @throws cudf::logic_error if `key_cols` size is larger than the number of
 * columns in `tables_to_merge` tables
 * @throws cudf::logic_error if `key_cols` size and `column_order` size mismatches
 *
 * @param[in] tables_to_merge Non-empty list of tables to be merged
 * @param[in] key_cols Indices of left_cols and right_cols to be used
 *                     for comparison criteria
 * @param[in] column_order Sort order types of columns indexed by key_cols
 * @param[in] null_precedence Array indicating the order of nulls with respect
 * to non-nulls for the indexing columns (key_cols)
 *
 * @returns A table containing sorted data from all input tables
 */
std::unique_ptr<cudf::table> merge(
  std::vector<table_view> const& tables_to_merge,
  std::vector<cudf::size_type> const& key_cols,
  std::vector<cudf::order> const& column_order,
  std::vector<cudf::null_order> const& null_precedence = {},
  rmm::mr::device_memory_resource* mr                  = rmm::mr::get_current_device_resource());

/** @} */  // end of group
}  // namespace cudf
