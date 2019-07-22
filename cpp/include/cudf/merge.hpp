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

#ifndef MERGE_HPP
#define MERGE_HPP

#include <vector>
#include "cudf.h"
#include "types.hpp"

namespace cudf {
/**
 * @brief Merge sorted tables.
 * 
 * Merges two sorted tables (including null values) into one sorted table
 * containing data from both tables.
 *
 * Example:
 * input:
 * table 1 => col 1 {0, 1, 2, 3}
 *            col 2 {4, 5, 6, 7}
 * table 2 => col 1 {1, 2}
 *            col 2 {8, 9}
 * output:
 * table => col 1 {0, 1, 1, 2, 2, 3}
 *          col 2 {4, 5, 8, 6, 9, 7}
 *
 * @Param[in] left_table A sorted table to be merged
 * @Param[in] right_table A sorted table to be merged
 * @Param[in] key_cols Indices of left_cols and right_cols to be used
 *                     for comparison criteria
 * @Param[in] asc_desc Sort order types of columns indexed by key_cols
 * @Param[in] nulls_are_smallest Flag indicating is nulls are to be treated as the smallest value
 *
 * @Returns A table containing sorted data from left_table and right_table
 */
table merge(table const& left_table,
            table const& right_table,
            std::vector<gdf_size_type> const& key_cols,
            std::vector<order_by_type> const& asc_desc,
            bool nulls_are_smallest = false);

}  // namespace cudf

#endif  // MERGE_HPP
