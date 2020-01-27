/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

namespace cudf {
namespace experimental {

/**
 * @brief  Applies a fixed-size rolling window function to the values in a column.
 *
 * This function aggregates values in a window around each element i of the input column, and
 * invalidates the bit mask for element i if there are not enough observations. The window size is
 * static (the same for each element). This matches Pandas' API for DataFrame.rolling with a few
 * notable differences:
 * - instead of the center flag it uses a two-part window to allow for more flexible windows.
 *   The total window size = `preceding_window + following_window + 1`. Element `i` uses elements
 *   `[i-preceding_window, i+following_window]` to do the window computation.
 * - instead of storing NA/NaN for output rows that do not meet the minimum number of observations
 *   this function updates the valid bitmask of the column to indicate which elements are valid.
 * 
 * The returned column for `op == COUNT` always has `INT32` type. All other operators return a 
 * column of the same type as the input. Therefore it is suggested to convert integer column types
 * (especially low-precision integers) to `FLOAT32` or `FLOAT64` before doing a rolling `MEAN`.
 *
 * @param[in] input_col The input column
 * @param[in] preceding_window The static rolling window size in the backward direction.
 * @param[in] following_window The static rolling window size in the forward direction.
 * @param[in] min_periods Minimum number of observations in window required to have a value,
 *                        otherwise element `i` is null.
 * @param[in] op The rolling window aggregation type (SUM, MAX, MIN, etc.)
 *
 * @returns   A nullable output column containing the rolling window results
 **/
std::unique_ptr<column> rolling_window(column_view const& input,
                                       size_type preceding_window,
                                       size_type following_window,
                                       size_type min_periods,
                                       std::unique_ptr<aggregation> const& aggr,
                                       rmm::mr::device_memory_resource* mr =
                                        rmm::mr::get_default_resource());

/**
 * @brief  Applies a variable-size rolling window function to the values in a column.
 *
 * This function aggregates values in a window around each element i of the input column, and
 * invalidates the bit mask for element i if there are not enough observations. The window size is
 * dynamic (varying for each element). This matches Pandas' API for DataFrame.rolling with a few
 * notable differences:
 * - instead of the center flag it uses a two-part window to allow for more flexible windows.
 *   The total window size = `preceding_window + following_window + 1`. Element `i` uses elements
 *   `[i-preceding_window, i+following_window]` to do the window computation.
 * - instead of storing NA/NaN for output rows that do not meet the minimum number of observations
 *   this function updates the valid bitmask of the column to indicate which elements are valid.
 * - support for dynamic rolling windows, i.e. window size can be specified for each element using
 *   an additional array.
 * 
 * The returned column for `op == COUNT` always has INT32 type. All other operators return a 
 * column of the same type as the input. Therefore it is suggested to convert integer column types
 * (especially low-precision integers) to `FLOAT32` or `FLOAT64` before doing a rolling `MEAN`.
 * 
 * @throws cudf::logic_error if window column type is not INT32
 *
 * @param[in] input_col The input column
 * @param[in] preceding_window A non-nullable column of INT32 window sizes in the forward direction.
 *                             `preceding_window[i]` specifies preceding window size for element `i`.
 * @param[in] following_window A non-nullable column of INT32 window sizes in the backward direction.
 *                             `following_window[i]` specifies following window size for element `i`.
 * @param[in] min_periods Minimum number of observations in window required to have a value,
 *                        otherwise element `i` is null.
 * @param[in] op The rolling window aggregation type (sum, max, min, etc.)
 *
 * @returns   A nullable output column containing the rolling window results
 **/
std::unique_ptr<column> rolling_window(column_view const& input,
                                       column_view const& preceding_window,
                                       column_view const& following_window,
                                       size_type min_periods,
                                       std::unique_ptr<aggregation> const& aggr,
                                       rmm::mr::device_memory_resource* mr =
                                        rmm::mr::get_default_resource());

} // namespace experimental 
} // namespace cudf
