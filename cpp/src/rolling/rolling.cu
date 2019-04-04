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
#include "rmm/thrust_rmm_allocator.h"
#include <bitmask/legacy_bitmask.hpp>
#include <cudf.h>
#include <cub/cub.cuh>
#include <memory>
#include <stdio.h>
#include <algorithm>

namespace
{

/**
 * @brief Computes the rolling window function
 *
 * @tparam ColumnType  Datatype of values pointed to by the pointers
 * @param in_col[in]  Pointers to input column's data
 * @param in_cols_valid[in]  Pointers to the validity mask of the input column
 * @param out_col[out]  Pointers to pre-allocated output column's data
 * @param out_cols_valid[out]  Pointers to the pre-allocated validity mask of
 * 		  the output column
 * @param nrows[in]  Number of rows in input table
 * @param window[in]  The static rolling window size, accumulates from
 *                in_col[i-window+1] to in_col[i] inclusive
 * @param min_periods[in]  Minimum number of observations in window required to
 *                have a value, otherwise 0 is stored in the valid bit mask
 * @param forward_window[in]  The static rolling window size in the forward
 *                direction, accumulates from in_col[i] to
 *                in_col[i+forward_window] inclusive
 */
template <typename ColumnType>
__global__
void gpu_rolling(ColumnType *in_col, gdf_valid_type *in_col_valid,
		 ColumnType *out_col, gdf_valid_type *out_col_valid,
		 gdf_size_type nrows, gdf_size_type window, gdf_size_type min_periods, gdf_size_type forward_window)
{
  using MaskType = uint32_t;
  constexpr uint32_t BITS_PER_MASK{sizeof(MaskType) * 8};

  gdf_size_type i = blockIdx.x * blockDim.x + threadIdx.x;
  gdf_size_type stride = blockDim.x * gridDim.x;

  // TODO: need a more generic way to handle aggregators (check groupby)
  // TODO: currently only sum is supported

  auto active_threads = __ballot_sync(0xffffffff, i < nrows);
  while(i < nrows)
  {
    ColumnType val = 0;
    gdf_size_type count = 0;

    // compute bounds
    gdf_size_type start_index = max((gdf_size_type)0, i - window + 1);
    gdf_size_type end_index = min(nrows, i + forward_window + 1);       // exclusive

    // aggregate
    for (size_t j = start_index; j < end_index; j++) {
      bool const input_is_valid{gdf_is_valid(in_col_valid, j)};
      if (input_is_valid) {
        val = val + in_col[j];
        count++;
      }
    }
  
    out_col[i] = val;

    // set the mask
    bool output_is_valid = (count >= min_periods);
    MaskType const result_mask{__ballot_sync(active_threads, output_is_valid)};
    gdf_index_type const out_mask_location = i / BITS_PER_MASK;
    MaskType* const __restrict__ out_mask32 = reinterpret_cast<MaskType*>(out_col_valid);
    // only one thread writes the mask
    if (0 == threadIdx.x % warpSize)
      out_mask32[out_mask_location] = result_mask;

    // process next element 
    i += stride;
    active_threads = __ballot_sync(active_threads, i < nrows);
  }
}

struct launch_kernel
{
  /**
   * @brief Uses SFINAE to instantiate only for arithmetic types
   */
  template<typename ColumnType, class... TArgs,
	   typename std::enable_if_t<std::is_arithmetic<ColumnType>::value, std::nullptr_t> = nullptr> 
  gdf_error dispatch_aggregation_type(gdf_size_type nrows, TArgs... FArgs)
  {
    PUSH_RANGE("CUDF_ROLLING", GDF_ORANGE);

    gdf_size_type block = 256;
    gdf_size_type grid = (nrows + block-1) / block;

    gpu_rolling<ColumnType><<<grid, block>>>(FArgs...); 

    cudaDeviceSynchronize();
    CUDA_CHECK_LAST();

    POP_RANGE();

    return GDF_SUCCESS;
  };

  /**
   * @brief If we cannot perform aggregation on this type then throw an error
   */
  template<typename ColumnType, class... TArgs,
	   typename std::enable_if_t<!std::is_arithmetic<ColumnType>::value, std::nullptr_t> = nullptr> 
  gdf_error dispatch_aggregation_type(gdf_size_type nrows, TArgs... FArgs)
  {
    return GDF_UNSUPPORTED_DTYPE;
  }

  /**
   * @brief Helper function for gdf_rolling. Deduces the type of the
   * aggregation column and calls another function to perform the rolling window.
   */
  template <typename ColumnType>
  gdf_error operator()(
    void *in_col_data_ptr, gdf_valid_type *in_col_valid_ptr,
    void *out_col_data_ptr, gdf_valid_type *out_col_valid_ptr,
    gdf_size_type nrows, gdf_size_type window, gdf_size_type min_periods, gdf_size_type forward_window)
  {
    return dispatch_aggregation_type<ColumnType>(nrows,
					  	 reinterpret_cast<ColumnType*>(in_col_data_ptr), in_col_valid_ptr,
			      		  	 reinterpret_cast<ColumnType*>(out_col_data_ptr), out_col_valid_ptr,
		 	      		  	 nrows, window, min_periods, forward_window);
  }
};

} // end of anonymous namespace

gdf_error gdf_rolling_window(gdf_column *output_col,
                             const gdf_column *input_col,
                             gdf_size_type window,
                             gdf_size_type min_periods,
                             gdf_size_type forward_window,
                             gdf_agg_op agg_type,
                             const gdf_size_type *window_col,
                             const gdf_size_type *min_periods_col,
                             const gdf_size_type *forward_window_col)
{
  // Make sure the inputs are not null
  GDF_REQUIRE((nullptr != input_col) && (nullptr != output_col), GDF_DATASET_EMPTY)

  // If there are no rows in the input, return successfully
  GDF_REQUIRE(input_col->size > 0, GDF_SUCCESS)

  // Check datatype homogeneity
  GDF_REQUIRE(output_col->dtype == input_col->dtype, GDF_DTYPE_MISMATCH)

  // TODO: support dynamic window sizes
  GDF_REQUIRE((nullptr == window_col) && (nullptr == min_periods_col) && (nullptr == forward_window_col), GDF_NOTIMPLEMENTED_ERROR)

  return cudf::type_dispatcher(input_col->dtype,
                               launch_kernel{},
                               input_col->data, input_col->valid,
                               output_col->data, output_col->valid,
                               input_col->size, window, min_periods, forward_window);
}

