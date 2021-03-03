/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/stream_compaction.hpp>

namespace cudf {
namespace lists {
/**
 * @addtogroup lists_drop_duplicates
 * @{
 * @file
 */

/**
 * @brief Create a new column by removing duplicated elements from each list in the given
 * lists_column
 *
 * Given an `input` list_column_view, the list elements in the column are copied to an output column
 * such that their duplicated entries are dropped. Adopted from stream compaction, the definition of
 * uniqueness depends on the value of @p keep:
 * - KEEP_FIRST: only the first of a sequence of duplicate entries is copied
 * - KEEP_LAST: only the last of a sequence of duplicate entries is copied
 * - KEEP_NONE: all duplicate entries are dropped out
 *
 * @param[in] lists_column    the input lists_column_view
 * @param[in] keep            keep first entry, last entry, or no entries if duplicates found
 * @param[in] nulls_equal     flag to denote nulls are considered equal
 * @param[in] mr              Device memory resource used to allocate the returned column
 *
 *  * @code{.pseudo}
 * input = { {1, 1, 2, 1, 3}, {4}, {5, 6, 6, 6, 5} }
 * if keep is KEEP_FIRST, the output will be { {1, 2, 3}, {4}, {5, 6} }
 * if keep is KEEP_LAST, the output will be  { {2, 1, 3}, {4}, {6, 5} }
 * if keep is KEEP_NONE, the output will be  { {2, 3}, {4}, {} }
 * @endcode
 *
 * @return A list column with lists having unique entries as specified by the `keep` policy.
 */
std::unique_ptr<column> drop_list_duplicates(
  lists_column_view const& lists_column,
  duplicate_keep_option keep,
  null_equality nulls_equal           = null_equality::EQUAL,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/** @} */  // end of group
}  // namespace lists
}  // namespace cudf
