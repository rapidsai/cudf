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

#include <cudf/column/column.hpp>
#include <cudf/table/table_view.hpp>
#include <memory>
#include "cudf/types.hpp"

namespace cudf {
/**
 * @addtogroup column_reshape
 * @{
 */

/**
 * @brief Interleave columns of a table into a single column.
 *
 * Converts the column major table `input` into a row major column.
 * Example:
 * ```
 * in     = [[A1, A2, A3], [B1, B2, B3]]
 * return = [A1, B1, A2, B2, A3, B3]
 * ```
 *
 * @throws cudf::logic_error if input contains no columns.
 * @throws cudf::logic_error if input columns dtypes are not identical.
 *
 * @param[in] input Table containing columns to interleave.
 *
 * @return The interleaved columns as a single column
 */
std::unique_ptr<column> interleave_columns(
  table_view const& input,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Repeats the rows from `input` table `count` times to form a new table.
 *
 * `output.num_columns() == input.num_columns()`
 * `output.num_rows() == input.num_rows() * count`
 *
 * ```
 * input  = [[8, 4, 7], [5, 2, 3]]
 * count  = 2
 * return = [[8, 4, 7, 8, 4, 7], [5, 2, 3, 5, 2, 3]]
 * ```
 *
 * @param[in] input Table containing rows to be repeated.
 * @param[in] count Number of times to tile "rows". Must be non-negative.
 *
 * @return The table containing the tiled "rows".
 */
std::unique_ptr<table> tile(
  table_view const& input,
  size_type count,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/** @} */  // end of group
}  // namespace cudf
