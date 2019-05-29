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

#ifndef ROLLING_HPP
#define ROLLING_HPP

#include "cudf.h"

// basic aggregation classes from groupby
// TODO: once #1478 is merged we need to update to use device_atomics.cuh instead
#include <groupby/aggregation_operations.hpp>

namespace cudf {
/* --------------------------------------------------------------------------*
 * @brief  Computes the rolling window function of the values in a column.
 *
 * This function aggregates values in a window around each element i of the input
 * column, and invalidates the bit mask for element i if there are not enough observations. The
 * window size and the number of observations can be static or dynamic (varying for
 * each element). This matches Pandas' API for DataFrame.rolling with a few notable
 * differences:
 * - instead of the center flag it uses the forward window size to allow for
 *   more flexible windows. The total window size = window + forward_window.
 *   Element i uses elements [i-window+1, i+forward_window] to do the window
 *   computation.
 * - instead of storing NA/NaN for output rows that do not meet the minimum
 *   number of observations this function updates the valid bitmask of the column
 *   to indicate which elements are valid.
 * - support for dynamic rolling windows, i.e. window size or number of
 *   observations can be specified for each element using an additional array.
 * This function is asynchronous with respect to the CPU, i.e. the call will
 * return before the operation is completed on the GPU. If the stream is
 * specified then the operation is asynchronous with respect to other CUDA
 * streams.
 *
 * @param[in] input_col The input column
 * @param[in] window The static rolling window size. If window_col = NULL, 
 *                output_col[i] accumulates values from input_col[i-window+1] to 
 *                input_col[i] inclusive
 * @param[in] min_periods Minimum number of observations in window required to
 *                have a value, otherwise 0 is stored in the valid bit mask for
 *                element i. If min_periods_col != NULL, then minimum number of
 *                observations for element i is obtained from min_periods_col[i]
 * @param[in] forward_window The static window size in the forward direction. If 
 *                forward_window_col = NULL, output_col[i] accumulates values from
 *                input_col[i] to input_col[i+forward_window] inclusive
 * @param[in] agg_type The rolling window aggregtion type (sum, max, min, etc.)
 * @param[in] window_col The window size values, window_col[i] specifies window
 *                size for element i. If window_col = NULL, then window is used as 
 *                the static window size for all elements
 * @param[in] min_periods_col The minimum number of observation values,
 *                min_periods_col[i] specifies minimum number of observations for 
 *                element i. If min_periods_col = NULL, then min_periods is used as 
 *                the static value for all elements
 * @param[in] forward_window_col The forward window size values,
 *                forward_window_col[i] specifies forward window size for element i.
 *                If forward_window_col = NULL, then forward_window is used as the
 *                static forward window size for aill elements
 * @param[in] stream Optional CUDA stream on which to execute kernels
 *
 * @returns   gdf_column The output column
 *
 * --------------------------------------------------------------------------*/
gdf_column* rolling_window(const gdf_column &input_col,
                           gdf_size_type window,
                           gdf_size_type min_periods,
                           gdf_size_type forward_window,
                           gdf_agg_op agg_type,
                           const gdf_size_type *window_col,
                           const gdf_size_type *min_periods_col,
                           const gdf_size_type *forward_window_col,
			   cudaStream_t stream = 0);

// helper funcitons - used in the implementation and tests
namespace rolling {
  // return true if ColumnType is arithmetic type or
  // AggOp is min_op/max_op/count_op for wrapper (non-arithmetic) types
  template <typename ColumnType, template <typename AggType> class AggOp>
  static constexpr bool is_supported()
  {
    return std::is_arithmetic<ColumnType>::value ||
           std::is_same<AggOp<ColumnType>, min_op<ColumnType>>::value ||
           std::is_same<AggOp<ColumnType>, max_op<ColumnType>>::value ||
           std::is_same<AggOp<ColumnType>, count_op<ColumnType>>::value;
  }

  // store functor
  template <typename ColumnType, bool average>
  struct store_output_functor
  {
    CUDA_HOST_DEVICE_CALLABLE void operator()(ColumnType &out, ColumnType &val, gdf_size_type count)
    {
      out = val;
    }
  };

  // partial specialization for AVG
  template <typename ColumnType>
  struct store_output_functor<ColumnType, true>
  {
    CUDA_HOST_DEVICE_CALLABLE void operator()(ColumnType &out, ColumnType &val, gdf_size_type count)
    {
      out = val / count;
    }
  };
}  // namespace cudf::rolling

}  // namespace cudf

#endif  // ROLLING_HPP
