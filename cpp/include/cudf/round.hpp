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
 * @addtogroup transformation_binaryops
 * @{
 * @file
 * @brief Column APIs for round
 */

/**
 * @brief Options for rounding with `cudf::round`
 */
enum class round_option : int32_t { HALF_UP, HALF_EVEN };

/**
 * @brief Rounds all the values in a column to the specified `decimal_places`
 *
 * `cudf::round` currently supports HALF_UP rounding for integer and floating point numbers.
 * When `decimal_places` is positive, it is rounding to the corresponding number of decimal places.
 * When `decimal_places` is negative, it is effectively rounding by removing `decimal_places`
 * number(s) left of the decimal point.
 *
 * Info of HALF_UP rounding: https://en.wikipedia.org/wiki/Rounding#Round_half_up
 *
 * @param input          Column of values to be rounded
 * @param decimal_places Number of decimal places to round to
 * @param round          Rounding option
 * @param mr             Device memory resource used to allocate the returned column's device memory
 *
 * @return std::unique_ptr<column> Column with each of the values rounded
 */
std::unique_ptr<column> round(
  column_view const& input,
  int32_t decimal_places              = 0,
  round_option round                  = round_option::HALF_UP,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/** @} */  // end of group
}  // namespace cudf
