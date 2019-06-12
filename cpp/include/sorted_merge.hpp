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

#ifndef SORTED_MERGE_HPP
#define SORTED_MERGE_HPP

#include <vector>
#include "cudf.h"
#include "types.hpp"
#include "rmm/thrust_rmm_allocator.h"

namespace cudf {
/**
 * @brief Merge sorted tables.
 *
 * @Param[in] left_table A sorted table to be merged
 * @Param[in] right_table A sorted table to be merged
 * @Param[in] sort_by_cols Indices of left_cols and right_cols to be used
 *                         for comparison criteria
 * @Param[in] asc_desc Sort order types of columns indexed by sort_by_cols
 * @Param[out] output_cols Merged columns
 *
 * @Returns A table containing sorted data from left_table and right_table
 */
table sorted_merge(table const& left_table,
                   table const& right_table,
                   std::vector<gdf_size_type> const& sort_by_cols,
                   rmm::device_vector<int8_t> const& asc_desc);

}  // namespace cudf

#endif  // SORTED_MERGE_HPP
