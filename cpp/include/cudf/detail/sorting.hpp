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
namespace experimental {
namespace detail {

/**---------------------------------------------------------------------------*
 * @brief Computes the row indices that would produce `input`  in a
 * lexicographical sorted order.
 *
 * @param input The table to sort
 * @param column_order The desired sort order for each column. Size must be
 * equal to `input.num_columns()` or empty. If empty, all columns will be sorted
 * in ascending order.
 * @param null_precedence The desired order of null compared to other elements
 * for each column.  Size must be equal to `input.num_columns()` or empty.
 * If empty, all columns will be sorted in `null_order::BEFORE`.
 * @param[in] mr Optional, The resource to use for all allocations
 * @param[in] stream Optional CUDA stream on which to execute kernels
 * @return std::unique_ptr<column> A non-nullable column of INT32 elements
 * containing the permuted row indices of `input` if it were sorted
 *---------------------------------------------------------------------------**/
std::unique_ptr<column> sorted_order(
    table_view input, std::vector<order> const& column_order = {},
    std::vector<null_order> const& null_precedence = {},
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
    cudaStream_t stream = 0);

/**
 * @copydoc cudf::experimental::sort_by_key
 *
 * @param[in] stream Optional CUDA stream on which to execute kernels
 */
std::unique_ptr<table> sort_by_key(
    table_view const& values, table_view const& keys,
    std::vector<order> const& column_order = {},
    std::vector<null_order> const& null_precedence = {},
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
    cudaStream_t stream = 0);

}  // namespace detail
}  // namespace experimental
}  // namespace cudf
