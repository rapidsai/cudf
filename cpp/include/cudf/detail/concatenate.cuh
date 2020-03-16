/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cudf/concatenate.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/table/table_view.hpp>

#include <vector>

namespace cudf {
namespace detail {

void concatenate_masks(
    rmm::device_vector<column_device_view> const& d_views,
    rmm::device_vector<size_type> const& d_offsets,
    bitmask_type * dest_mask,
    size_type output_size,
    cudaStream_t stream);

/**---------------------------------------------------------------------------*
 * @brief Concatenates `views[i]`'s bitmask from the bits
 * `[views[i].offset(), views[i].offset() + views[i].size())` for all elements
 * views[i] in views into an array
 *
 * @param views Vector of column views whose bitmask needs to be copied
 * @param dest_mask Pointer to array that contains the combined bitmask
 * of the column views
 * @param stream stream on which all memory allocations and copies
 * will be performed
 *---------------------------------------------------------------------------**/
void concatenate_masks(
    std::vector<column_view> const &views,
    bitmask_type * dest_mask,
    cudaStream_t stream);

/**---------------------------------------------------------------------------*
 * @brief Concatenates multiple columns into a single column.
 *
 * @throws cudf::logic_error
 * If types of the input columns mismatch
 *
 * @param columns_to_concat The column views to be concatenated into a single
 * column
 * @param mr Optional The resource to use for all allocations
 * @param stream Optional The stream on which to execute all allocations and copies
 * @return Unique pointer to a single table having all the rows from the
 * elements of `columns_to_concat` respectively in the same order.
 *---------------------------------------------------------------------------**/
 std::unique_ptr<column>
 concatenate(std::vector<column_view> const& columns_to_concat,
             rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
             cudaStream_t stream = 0);
 
}  // namespace detail
}  // namespace cudf
