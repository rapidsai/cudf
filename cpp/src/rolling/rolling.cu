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

#include "utilities/nvtx/nvtx_utils.h"
#include "utilities/type_dispatcher.hpp"
#include "utilities/bit_util.cuh"
#include "bitmask/bit_mask.cuh"
#include "rmm/thrust_rmm_allocator.h"
#include <cudf.h>
#include <cub/cub.cuh>
#include <memory>
#include <stdio.h>
#include <algorithm>

#include <rolling.hpp>

// allocate column
#include <io/utilities/wrapper_utils.hpp>

// basic aggregation classes from groupby
// TODO: once #1478 is merged we need to update to use device_atomics.cuh instead
#include <groupby/aggregation_operations.hpp>

namespace
{
  // return true if ColumnType is arithmetic type or
  // AggOp is min_op or max_op for wrapper (non-arithmetic) types
  template <typename ColumnType, template <typename AggType> class AggOp>
  static constexpr bool is_supported()
  {
    return std::is_arithmetic<ColumnType>::value ||
           std::is_same<AggOp<ColumnType>, min_op<ColumnType>>::value ||
           std::is_same<AggOp<ColumnType>, max_op<ColumnType>>::value;
  }

  // store functor
  template <typename ColumnType, bool average>
  struct store_output_functor
  {
    CUDA_DEVICE_CALLABLE void operator()(ColumnType &out, ColumnType &val, gdf_size_type count)
    {
      out = val;
    }
  };

  // partial specialization for AVG
  template <typename ColumnType>
  struct store_output_functor<ColumnType, true>
  {
    CUDA_DEVICE_CALLABLE void operator()(ColumnType &out, ColumnType &val, gdf_size_type count)
    {
      out = val / count;
    }
  };


/**
 * @brief Computes the rolling window function
 *
 * @tparam ColumnType  Datatype of values pointed to by the pointers
 * @tparam agg_op  A functor that defines the aggregation operation
 * @tparam average Perform average across all valid elements in the window
 * @param nrows[in]  Number of rows in input table
 * @param out_col[out]  Pointers to pre-allocated output column's data
 * @param out_cols_valid[out]  Pointers to the pre-allocated validity mask of
 * 		  the output column
 * @param in_col[in]  Pointers to input column's data
 * @param in_cols_valid[in]  Pointers to the validity mask of the input column
 * @param window[in]  The static rolling window size, accumulates from
 *                in_col[i-window+1] to in_col[i] inclusive
 * @param min_periods[in]  Minimum number of observations in window required to
 *                have a value, otherwise 0 is stored in the valid bit mask
 * @param forward_window[in]  The static rolling window size in the forward
 *                direction, accumulates from in_col[i] to
 *                in_col[i+forward_window] inclusive
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
 *                static forward window size for all elements

 */
template <typename ColumnType, template <typename AggType> class agg_op, bool average>
__global__
void gpu_rolling(gdf_size_type nrows,
		 ColumnType * const __restrict__ out_col, bit_mask::bit_mask_t * const __restrict__ out_col_valid,
		 ColumnType const * const __restrict__ in_col, bit_mask::bit_mask_t const * const __restrict__ in_col_valid,
		 gdf_size_type window,
		 gdf_size_type min_periods,
		 gdf_size_type forward_window,
		 const gdf_size_type *window_col,
		 const gdf_size_type *min_periods_col,
		 const gdf_size_type *forward_window_col)
{
  // we're going to be using bit utils a lot in the kernel
  using namespace bit_mask;

  gdf_size_type i = blockIdx.x * blockDim.x + threadIdx.x;
  gdf_size_type stride = blockDim.x * gridDim.x;

  agg_op<ColumnType> op;

  auto active_threads = __ballot_sync(0xffffffff, i < nrows);
  while(i < nrows)
  {
    ColumnType val = agg_op<ColumnType>::IDENTITY;
    volatile gdf_size_type count = 0;	// declare this as volatile to avoid some compiler optimizations that lead to incorrect results on some systems (bug will be filed to investigate)

    // dynamic window handling
    if (window_col != nullptr) window = window_col[i];
    if (min_periods_col != nullptr) min_periods = max(min_periods_col[i], 1);	// at least one observation is required
    if (forward_window_col != nullptr) forward_window = forward_window_col[i];

    // compute bounds
    gdf_size_type start_index = max((gdf_size_type)0, i - window + 1);
    gdf_size_type end_index = min(nrows, i + forward_window + 1);       // exclusive

    // aggregate
    // TODO: We should explore using shared memory to avoid redundant loads.
    //       This might require separating the kernel into a special version
    //       for dynamic and static sizes.
    for (gdf_index_type j = start_index; j < end_index; j++) {
      bool const input_is_valid{is_valid(in_col_valid, j)};
      if (input_is_valid) {
        val = op(in_col[j], val);
        count++;
      }
    }

    // check if we have enough input samples
    bool output_is_valid = (count >= min_periods);

    // set the mask
    bit_mask_t const result_mask{__ballot_sync(active_threads, output_is_valid)};
    gdf_index_type const out_mask_location = gdf::util::detail::bit_container_index<bit_mask_t, gdf_index_type>(i);

    // only one thread writes the mask
    if (0 == threadIdx.x % warpSize)
      out_col_valid[out_mask_location] = result_mask;

    // store the output value, one per thread
    if (output_is_valid)
      store_output_functor<ColumnType, average>{}(out_col[i], val, count);

    // process next element 
    i += stride;
    active_threads = __ballot_sync(active_threads, i < nrows);
  }
}

struct rolling_window_launcher
{
  /**
   * @brief Uses SFINAE to instantiate only for supported type combos
   */
  template<typename ColumnType, template <typename AggType> class agg_op, bool average, class... TArgs,
	   typename std::enable_if_t<is_supported<ColumnType, agg_op>(), std::nullptr_t> = nullptr>
  void dispatch_aggregation_type(gdf_size_type nrows, cudaStream_t stream, TArgs... FArgs)
  {
    PUSH_RANGE("CUDF_ROLLING", GDF_ORANGE);

    gdf_size_type block = 256;
    gdf_size_type grid = (nrows + block-1) / block;

    gpu_rolling<ColumnType, agg_op, average><<<grid, block, 0, stream>>>(nrows, FArgs...);
    CUDA_CHECK_LAST();

    POP_RANGE();
  }

  /**
   * @brief If we cannot perform aggregation on this type then throw an error
   */
  template<typename ColumnType, template <typename AggType> class agg_op, bool average, class... TArgs,
	   typename std::enable_if_t<!is_supported<ColumnType, agg_op>(), std::nullptr_t> = nullptr>
  void dispatch_aggregation_type(gdf_size_type nrows, cudaStream_t stream, TArgs... FArgs)
  {
    CUDF_FAIL("Unsupported column type/operation combo. Only `min` and `max` are supported for non-arithmetic types for aggregations.");
  }

  /**
   * @brief Helper function for gdf_rolling. Deduces the type of the
   * aggregation column and type and calls another function to invoke the
   * rolling window kernel.
   */
  template <typename ColumnType>
  void operator()(gdf_size_type nrows,
		  gdf_agg_op agg_type,
		  void *out_col_data_ptr, gdf_valid_type *out_col_valid_ptr,
		  void *in_col_data_ptr, gdf_valid_type *in_col_valid_ptr,
		  gdf_size_type window,
		  gdf_size_type min_periods,
		  gdf_size_type forward_window,
		  const gdf_size_type *window_col,
		  const gdf_size_type *min_periods_col,
		  const gdf_size_type *forward_window_col,
		  cudaStream_t stream)
  {
    ColumnType *typed_out_data = static_cast<ColumnType*>(out_col_data_ptr);
    bit_mask::bit_mask_t *typed_out_valid = reinterpret_cast<bit_mask::bit_mask_t*>(out_col_valid_ptr);
    const ColumnType *typed_in_data = static_cast<const ColumnType*>(in_col_data_ptr);
    const bit_mask::bit_mask_t *typed_in_valid = reinterpret_cast<const bit_mask::bit_mask_t*>(in_col_valid_ptr);

    // TODO: We should consolidate our aggregation enums for reductions, scans,
    //       groupby and rolling. @harrism suggested creating
    //       aggregate_dispatcher that works like type_dispatcher.
    switch (agg_type) {
    case GDF_SUM:
      dispatch_aggregation_type<ColumnType, sum_op, false>(nrows, stream,
							   typed_out_data, typed_out_valid,
							   typed_in_data, typed_in_valid,
							   window, min_periods, forward_window,
							   window_col, min_periods_col, forward_window_col);
      break;
    case GDF_MIN:
      dispatch_aggregation_type<ColumnType, min_op, false>(nrows, stream,
							   typed_out_data, typed_out_valid,
							   typed_in_data, typed_in_valid,
							   window, min_periods, forward_window,
							   window_col, min_periods_col, forward_window_col);
      break;
    case GDF_MAX:
      dispatch_aggregation_type<ColumnType, max_op, false>(nrows, stream,
							   typed_out_data, typed_out_valid,
							   typed_in_data, typed_in_valid,
							   window, min_periods, forward_window,
							   window_col, min_periods_col, forward_window_col);
      break;
    case GDF_COUNT:
      dispatch_aggregation_type<ColumnType, count_op, false>(nrows, stream,
							   typed_out_data, typed_out_valid,
							   typed_in_data, typed_in_valid,
							   window, min_periods, forward_window,
							   window_col, min_periods_col, forward_window_col);
      break;
    case GDF_AVG:
      dispatch_aggregation_type<ColumnType, sum_op, true>(nrows, stream,
							   typed_out_data, typed_out_valid,
							   typed_in_data, typed_in_valid,
							   window, min_periods, forward_window,
							   window_col, min_periods_col, forward_window_col);
      break;
    default:
      // TODO: need a nice way to convert enums to strings, same would be useful for groupbys
      CUDF_FAIL("Aggregation function " + std::to_string(agg_type) + " is not implemented");
    }
  }
};

}  // anonymous namespace

namespace cudf {

// see rolling.hpp for declaration
gdf_column* rolling_window(const gdf_column &input_col,
                           gdf_size_type window,
                           gdf_size_type min_periods,
                           gdf_size_type forward_window,
                           gdf_agg_op agg_type,
                           const gdf_size_type *window_col,
                           const gdf_size_type *min_periods_col,
                           const gdf_size_type *forward_window_col,
			   cudaStream_t stream)
{
  CUDF_EXPECTS((window >= 0) && (min_periods >= 0) && (forward_window >= 0), "Window size and min periods must be non-negative");

  // Use the column wrapper class from io/utilities to quickly create a column
  gdf_column_wrapper output_col(input_col.size,
				input_col.dtype,
				gdf_dtype_extra_info{TIME_UNIT_NONE},
				"");

  // If there are no rows in the input, return successfully
  if (input_col.size == 0)
    return output_col.release();

  // Allocate memory for the output column
  CUDF_EXPECTS(output_col.allocate() == GDF_SUCCESS, "Cannot allocate the output column");

  // At least one observation is required to procure a valid output
  min_periods = std::max(min_periods, 1);

  // Launch type dispatcher
  cudf::type_dispatcher(input_col.dtype,
                        rolling_window_launcher{},
                        input_col.size, agg_type,
                        output_col->data, output_col->valid,
			input_col.data, input_col.valid,
                        window, min_periods, forward_window,
			window_col, min_periods_col, forward_window_col,
			stream);

  // Release the gdf pointer from the wrapper class
  return output_col.release();
}

}  // namespace cudf
