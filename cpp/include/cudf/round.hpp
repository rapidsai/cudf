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

namespace cudf {

/**
 * @addtogroup transformation_unaryops
 * @{
 * @file
 * @brief Column APIs for round
 */

/**
 * @brief Different rounding methods for `cudf::round`
 */
enum class rounding_method : int32_t { HALF_UP };

/**
 * @brief Rounds all the values in a column to the specified number of decimal places.
 *
 * `cudf::round` currently supports HALF_UP rounding for integer and floating point numbers.
 *
 * Example:
 * ```
 * column_view col; // contains { 1.729, 17.29, 172.9, 1729 };
 *
 * auto result1 = cudf::round(col);     // yields { 2,   17,   173,   1729 }
 * auto result2 = cudf::round(col, 1);  // yields { 1.7, 17.3, 172.9, 1729 }
 * auto result3 = cudf::round(col, -1); // yields { 0,   20,   170,   1730 }
 * ```
 *
 * Info on HALF_UP rounding: https://en.wikipedia.org/wiki/Rounding#Round_half_up
 *
 * @param input          Column of values to be rounded
 * @param decimal_places Number of decimal places to round to (default 0). If negative, this
 * specifies the number of positions to the left of the decimal point.
 * @param method         Rounding method
 * @param mr             Device memory resource used to allocate the returned column's device memory
 *
 * @return std::unique_ptr<column> Column with each of the values rounded
 */
std::unique_ptr<column> round(
  column_view const& input,
  int32_t decimal_places              = 0,
  rounding_method method              = rounding_method::HALF_UP,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/** @} */  // end of group
}  // namespace cudf
