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
#include <cudf/utilities/export.hpp>

#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

namespace CUDF_EXPORT cudf {

/**
 * @addtogroup transformation_unaryops
 * @{
 * @file
 * @brief Column APIs for round
 */

/**
 * @brief Different rounding methods for `cudf::round`
 *
 * Info on HALF_EVEN rounding: https://en.wikipedia.org/wiki/Rounding#Rounding_half_to_even
 * Info on HALF_UP   rounding: https://en.wikipedia.org/wiki/Rounding#Rounding_half_away_from_zero
 * Note: HALF_UP means up in MAGNITUDE: Away from zero! Because of how Java and python define it
 */
enum class rounding_method : int32_t { HALF_UP, HALF_EVEN };

/**
 * @brief Rounds all the values in a column to the specified number of decimal places.
 *
 * `cudf::round` currently supports HALF_UP and HALF_EVEN rounding for integer, floating point and
 * `decimal32` and `decimal64` numbers. For `decimal32` and `decimal64` numbers, negated
 * `numeric::scale` is equivalent to `decimal_places`.
 *
 * Example:
 * ```
 * using namespace cudf;
 *
 * column_view a; // contains { 1.729, 17.29, 172.9, 1729 };
 *
 * auto result1 = round(a);     // { 2,   17,   173,   1729 }
 * auto result2 = round(a, 1);  // { 1.7, 17.3, 172.9, 1729 }
 * auto result3 = round(a, -1); // { 0,   20,   170,   1730 }
 *
 * column_view b; // contains { 1.5, 2.5, 1.35, 1.45, 15, 25 };
 *
 * auto result4 = round(b,  0, rounding_method::HALF_EVEN); // { 2,   2,   1,   1,   15, 25};
 * auto result5 = round(b,  1, rounding_method::HALF_EVEN); // { 1.5, 2.5, 1.4, 1.4, 15, 25};
 * auto result6 = round(b, -1, rounding_method::HALF_EVEN); // { 0,   0,   0,   0,   20, 20};
 * ```
 *
 * @param input          Column of values to be rounded
 * @param decimal_places Number of decimal places to round to (default 0). If negative, this
 * specifies the number of positions to the left of the decimal point.
 * @param method         Rounding method
 * @param mr             Device memory resource used to allocate the returned column's device memory
 *
 * @return Column with each of the values rounded
 */
std::unique_ptr<column> round(
  column_view const& input,
  int32_t decimal_places            = 0,
  rounding_method method            = rounding_method::HALF_UP,
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/** @} */  // end of group
}  // namespace CUDF_EXPORT cudf
