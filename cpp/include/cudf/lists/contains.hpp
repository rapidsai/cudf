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

namespace cudf {
namespace lists {
/**
 * @addtogroup lists_contains
 * @{
 * @file
 */

/**
 * @brief Create a column of bool values indicating whether the specified scalar
 * is an element of each row of a list column.
 *
 * The output column has as many elements as the input `lists` column.
 * Output `column[i]` is set to true if the lists row `lists[i]` contains the value
 * specified in `search_key`. Otherwise, it is set to false.
 *
 * Output `column[i]` is set to null if one or more of the following are true:
 *   1. The search key `search_key` is null
 *   2. The list row `lists[i]` is null
 *   3. The list row `lists[i]` does not contain the search key, and contains at least
 *      one null.
 *
 * @param lists Lists column whose `n` rows are to be searched
 * @param search_key The scalar key to be looked up in each list row
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return std::unique_ptr<column> BOOL8 column of `n` rows with the result of the lookup
 */
std::unique_ptr<column> contains(
  cudf::lists_column_view const& lists,
  cudf::scalar const& search_key,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Create a column of bool values indicating whether the list rows of the first
 * column contain the corresponding values in the second column
 *
 * The output column has as many elements as the input `lists` column.
 * Output `column[i]` is set to true if the lists row `lists[i]` contains the value
 * in `search_keys[i]`. Otherwise, it is set to false.
 *
 * Output `column[i]` is set to null if one or more of the following are true:
 *   1. The row `search_keys[i]` is null
 *   2. The list row `lists[i]` is null
 *   3. The list row `lists[i]` does not contain the `search_keys[i]`, and contains at least
 *      one null.
 *
 * @param lists Lists column whose `n` rows are to be searched
 * @param search_keys Column of elements to be looked up in each list row
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return std::unique_ptr<column> BOOL8 column of `n` rows with the result of the lookup
 */
std::unique_ptr<column> contains(
  cudf::lists_column_view const& lists,
  cudf::column_view const& search_keys,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/** @} */  // end of group
}  // namespace lists
}  // namespace cudf
