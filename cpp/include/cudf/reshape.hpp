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

#include <memory>
#include <cudf/column/column.hpp>
#include <cudf/table/table_view.hpp>

namespace cudf {

namespace experimental {

/**
 * @brief Interleave columns of a table in to a single column.
 *
 * Converts the column major table @p in into a row major column.
 * Example:
 * ```
 * in = [[A1, A2, A3], [B1, B2, B3]]
 * return = [A1, B1, A2, B2, A3, B3]
 * ```
 *
 * @note: The dtype of all columns in @p in should be the same.
 *
 * @param[in]        in table containing columns to interleave.
 * @param[in] mr     Optional resource to use for device memory
 * @param[in] stream Optional CUDA stream on which to execute kernels.
 *
 * @return The interleaved columns as a single column
 */
std::unique_ptr<column>
interleave_columns(table_view const& in,
                   rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource(),
                   cudaStream_t stream = 0);

/*
 * @brief Constructs a new table with "rows" from @in stacked @p count times.
 *
 * When @p count = 0, the returned table is `empty_like(in)`.
 *
 * in     = [[8, 4, 7], [5, 2, 3]]
 * count  = 2
 * return = [[8, 4, 7, 8, 4, 7], [5, 2, 3, 5, 2, 3]]
 *
 * @param[in] in     Table containing "rows" columns to tile in to new table.
 * @param[in] count  Number of times to tile "rows". Must be non-negative.
 * @param[in] mr     Optional resource to use for device memory
 * @param[in] stream Optional CUDA stream on which to execute kernels.
 *
 * @return           The table containing the tiled "rows".
 */
std::unique_ptr<table>
tile(table_view const& in,
     size_type count,
     rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource(),
     cudaStream_t stream = 0);

} // namespace experimental

} // namespace cudf
