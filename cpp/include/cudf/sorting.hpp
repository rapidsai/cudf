/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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
namespace exp {

/**---------------------------------------------------------------------------*
 * @brief Computes the row indices that would produce `input`  in a
 * lexicographical sorted order.
 *
 * @param input The table to sort
 * @param column_order The desired sort order for each column. Size must be
 * equal to `input.num_columns()` or empty. If empty, all columns will be sorted
 * in ascending order.
 * @param null_precedence The size of a NULL value in comparison to all other
 * values
 * @return std::unique_ptr<column> A non-nullable column of INT32 elements
 * containing the permuted row indices of `input` if it were sorted
 *---------------------------------------------------------------------------**/
std::unique_ptr<column> sorted_order(
    table_view input, std::vector<order> const& column_order,
    null_order null_precedence,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());
}  // namespace exp
}  // namespace cudf