/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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
#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>

#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <memory>

namespace CUDF_EXPORT cudf {
/**
 * @addtogroup column_reshape
 * @{
 * @file
 * @brief Column APIs for interleave and tile
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
 * @param[in] input Table containing columns to interleave
 * @param[in] mr Device memory resource used to allocate the returned column's device memory
 *
 * @return The interleaved columns as a single column
 */
std::unique_ptr<column> interleave_columns(
  table_view const& input,
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

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
 * @param[in] input Table containing rows to be repeated
 * @param[in] count Number of times to tile "rows". Must be non-negative
 * @param[in] mr Device memory resource used to allocate the returned table's device memory
 *
 * @return The table containing the tiled "rows"
 */
std::unique_ptr<table> tile(
  table_view const& input,
  size_type count,
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * @brief Configures whether byte casting flips endianness
 */
enum class flip_endianness : bool { NO, YES };

/**
 * @brief Converts a column's elements to lists of bytes
 *
 * ```
 * input<int32>  = [8675, 309]
 * configuration = flip_endianness::YES
 * return        = [[0x00, 0x00, 0x21, 0xe3], [0x00, 0x00, 0x01, 0x35]]
 * ```
 *
 * @param input_column Column to be converted to lists of bytes
 * @param endian_configuration Whether to retain or flip the endianness of the elements
 * @param mr Device memory resource used to allocate the returned column's device memory
 *
 * @return The column containing the lists of bytes
 */
std::unique_ptr<column> byte_cast(
  column_view const& input_column,
  flip_endianness endian_configuration,
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/** @} */  // end of group

}  // namespace CUDF_EXPORT cudf
