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
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/stream_compaction.hpp>

#include <optional>

namespace cudf {
namespace lists {
/**
 * @addtogroup lists_drop_duplicates
 * @{
 * @file
 */

/*
 * @brief Flag to specify which entry to keep when removing the duplicate entries from a repeated
 * sequence.
 */
enum class keep_policy {
  UNDEFINED,  ///< An arbitrary entry at an unknown position in the repeated sequence will be kept.
  FIRST,      ///< Keep the first entry (all duplicate entries after it will be removed).
  LAST        ///< Keep the last entry (all duplicate entries before it will be removed).
};

/**
 * @brief Create new lists columns by extracting the key list entries and their corresponding value
 * entries from the given lists columns such that only the unique list entries in the `keys` column
 * will be copied.
 *
 * In some cases, there is only a need to remove duplicates entries from one input lists column. In
 * such situations, the input values lists column can be ignored.
 *
 * If the `values` lists column is given, the users are responsible to have the keys-values columns
 * having the same number of entries in each row. Otherwise, the results will be undefined.
 *
 * Given a pair of keys-values lists columns, each list entry in the keys column corresponds to a
 * list entry in the values column (i.e., the lists at each row index in both keys and values
 * columns have the same size). The entries in both columns are copied into an output pair of keys
 * and values lists columns (respectively), in a way such that the repeated key entries (and their
 * corresponding value entries) are dropped out to keep only the entries with unique keys.
 *
 * The order of entries within each list of the output lists columns are not guaranteed to be
 * preserved as in the input. In the current implementation, entries in the output keys lists are
 * sorted by ascending order (nulls last), but this is not guaranteed in future implementation.
 *
 * @throw cudf::logic_error if the child column of the input keys column contains nested type other
 * than struct.
 *
 * @param keys The input keys lists column to check for uniqueness.
 * @param values The optional values lists column in which each list entry corresponds to a list
 *        entry in the keys column.
 * @param nulls_equal Flag to specify whether null key entries should be considered equal.
 * @param nans_equal Flag to specify whether NaN key entries should be considered as equal value
 *        (only applicable for floating point data column).
 * @param keep_entry Flag to specify which entry will be kept when removing duplicate entries in the
 *        repeated sequence. This is only relevant when the values lists column is given.
 * @param mr Device resource used to allocate memory.
 *
 * @code{.pseudo}
 * input  = { {1, 1, 2, 1, 3}, {4}, NULL, {}, {NULL, NULL, NULL, 5, 6, 6, 6, 5} }
 * output = { {1, 2, 3}, {4}, NULL, {}, {5, 6, NULL} }
 * @endcode
 *
 * @return A pair of pointers storing to the columns resulted from removing duplicate key entries
 *         and their corresponding values entries from the input lists columns. If the input values
 *         column is missing, its corresponding output will be a null pointer.
 */
std::pair<std::unique_ptr<column>, std::unique_ptr<column>> drop_list_duplicates(
  lists_column_view const& keys,
  std::optional<lists_column_view> const& values,
  null_equality nulls_equal           = null_equality::EQUAL,
  nan_equality nans_equal             = nan_equality::UNEQUAL,
  keep_policy keep_entry              = keep_policy::UNDEFINED,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/** @} */  // end of group
}  // namespace lists
}  // namespace cudf
