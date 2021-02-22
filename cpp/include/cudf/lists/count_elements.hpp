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
 * @addtogroup lists_elements
 * @{
 * @file
 */

/**
 * @brief Returns a numeric column containing the number of rows in
 * each list element in the given lists column.
 *
 * The output column will have the same number of rows as the
 * input lists column. Each `output[i]` will be `input[i].size()`.
 *
 * @code{.pseudo}
 * l = { {1, 2, 3}, {4}, {5, 6} }
 * r = count_elements(l)
 * r is now {3, 1, 2}
 * @endcode
 *
 * Any null input element will result in a corresponding null entry
 * in the output column.
 *
 * @param input Input lists column.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New INT32 column with the number of elements for each row.
 */
std::unique_ptr<column> count_elements(
  lists_column_view const& input,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/** @} */  // end of lists_elements group

}  // namespace lists
}  // namespace cudf
