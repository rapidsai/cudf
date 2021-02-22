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

namespace cudf {
namespace lists {
/**
 * @addtogroup lists_sort
 * @{
 * @file
 */

/**
 * @brief Segmented sort of the elements within a list in each row of a list column.
 *
 * `source_column` with depth 1 is only supported.
 *
 * * @code{.pseudo}
 * source_column            : [{4, 2, 3, 1}, {1, 2, NULL, 4}, {-10, 10, 0}]
 *
 * Ascending,  Null After   : [{1, 2, 3, 4}, {1, 2, 4, NULL}, {-10, 0, 10}]
 * Ascending,  Null Before  : [{1, 2, 3, 4}, {NULL, 1, 2, 4}, {-10, 0, 10}]
 * Descending, Null After   : [{4, 3, 2, 1}, {NULL, 4, 2, 1}, {10, 0, -10}]
 * Descending, Null Before  : [{4, 3, 2, 1}, {4, 2, 1, NULL}, {10, 0, -10}]
 * @endcode
 *
 * @param source_column View of the list column of numeric types to sort
 * @param column_order The desired sort order
 * @param null_precedence The desired order of null compared to other elements in the list
 * @param mr Device memory resource to allocate any returned objects
 * @return list column with elements in each list sorted.
 *
 */
std::unique_ptr<column> sort_lists(
  lists_column_view const& source_column,
  order column_order,
  null_order null_precedence,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/** @} */  // end of group
}  // namespace lists
}  // namespace cudf
