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

#include <cudf/utilities/nvtx_utils.hpp>
#include <cudf/utilities/legacy/type_dispatcher.hpp>
#include <utilities/legacy/bit_util.cuh>
#include <bitmask/legacy/bit_mask.cuh>
#include <rmm/thrust_rmm_allocator.h>
#include <cudf/cudf.h>
#include <cub/cub.cuh>
#include <memory>
#include <stdio.h>
#include <algorithm>

#include <cudf/legacy/copying.hpp>
#include <cudf/legacy/rolling.hpp>
#include <rolling/legacy/rolling_detail.hpp>

// allocate column
#include <io/utilities/wrapper_utils.hpp>

#include <jit/launcher.h>
#include <jit/legacy/type.h>
#include <jit/parser.h>
#include "jit/code/code.h"
#include "jit/util/type.h"

#include <types.h.jit>
#include <types.hpp.jit>

#include <rmm/device_scalar.hpp>
#include <cudf/detail/utilities/cuda.cuh>

namespace
{
/**
 * @brief Computes the rolling window function
 *
 * @tparam ColumnType  Datatype of values pointed to by the pointers
 * @tparam agg_op  A functor that defines the aggregation operation
 * @tparam average Perform average across all valid elements in the window
 * @param nrows[in]  Number of rows in input table
 * @param output_valid_count[in]  Number of valid rows in the output
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
template <typename ColumnType, class agg_op, bool average, cudf::size_type block_size>
__launch_bounds__ (block_size)
__global__
void gpu_rolling(cudf::size_type nrows,
                 gdf_size_type * __restrict__ const output_valid_count,
                 ColumnType * const __restrict__ out_col, 
                 bit_mask::bit_mask_t * const __restrict__ out_col_valid,
                 ColumnType const * const __restrict__ in_col, 
                 bit_mask::bit_mask_t const * const __restrict__ in_col_valid,
                 cudf::size_type window,
                 cudf::size_type min_periods,
                 cudf::size_type forward_window,
                 const cudf::size_type *window_col,
                 const cudf::size_type *min_periods_col,
                 const cudf::size_type *forward_window_col)
{
  // we're going to be using bit utils a lot in the kernel
  using namespace bit_mask;
  const bool is_nullable = (in_col_valid != nullptr);

  cudf::size_type i = blockIdx.x * blockDim.x + threadIdx.x;
  cudf::size_type stride = blockDim.x * gridDim.x;

  agg_op op;

  gdf_size_type warp_valid_count{0};

  auto active_threads = __ballot_sync(0xffffffff, i < nrows);
  while(i < nrows)
  {
    ColumnType val = agg_op::template identity<ColumnType>();
    volatile cudf::size_type count = 0;	// declare this as volatile to avoid some compiler optimizations that lead to incorrect results for CUDA 10.0 and below (fixed in CUDA 10.1)

    // dynamic window handling
    if (window_col != nullptr) window = window_col[i];
    if (min_periods_col != nullptr) min_periods = max(min_periods_col[i], 1);	// at least one observation is required
    if (forward_window_col != nullptr) forward_window = forward_window_col[i];

    // compute bounds
    cudf::size_type start_index = max((cudf::size_type)0, i - window + 1);
    cudf::size_type end_index = min(nrows, i + forward_window + 1);       // exclusive

    // aggregate
    // TODO: We should explore using shared memory to avoid redundant loads.
    //       This might require separating the kernel into a special version
    //       for dynamic and static sizes.
    for (cudf::size_type j = start_index; j < end_index; j++) {
      if (!is_nullable || is_valid(in_col_valid, j)) {
        val = op(in_col[j], val);
        count++;
      }
    }

    // check if we have enough input samples
    bool output_is_valid = (count >= min_periods);

    // set the mask
    bit_mask_t const result_mask{__ballot_sync(active_threads, output_is_valid)};
    cudf::size_type const out_mask_location = cudf::util::detail::bit_container_index<bit_mask_t, cudf::size_type>(i);

    // only one thread writes the mask
    if (0 == threadIdx.x % warpSize){
      out_col_valid[out_mask_location] = result_mask;
      warp_valid_count += __popc(result_mask);
    }

    // store the output value, one per thread
    if (output_is_valid)
      cudf::detail::store_output_functor<ColumnType, average>{}(out_col[i], val, count);

    // process next element
    i += stride;
    active_threads = __ballot_sync(active_threads, i < nrows);
  }

  // sum the valid counts across the whole block  
  gdf_size_type block_valid_count = cudf::experimental::detail::single_lane_block_sum_reduce<block_size, 0>(warp_valid_count);
  if(threadIdx.x == 0){
    atomicAdd(output_valid_count, block_valid_count);
  }  
}

struct rolling_window_launcher
{
  /**
   * @brief Uses SFINAE to instantiate only for supported type combos
   */
  template<typename ColumnType, class agg_op, bool average, class... TArgs,
     typename std::enable_if_t<cudf::detail::is_supported<ColumnType, agg_op>(), std::nullptr_t> = nullptr>
  void dispatch_aggregation_type(cudf::size_type nrows, gdf_size_type& null_count, cudaStream_t stream, TArgs... FArgs)
  {
    cudf::nvtx::range_push("CUDF_ROLLING", cudf::nvtx::color::ORANGE);

    constexpr cudf::size_type block = 256;
    cudf::size_type grid = (nrows + block-1) / block;

    rmm::device_scalar<gdf_size_type> device_valid_count{0, stream};

    gpu_rolling<ColumnType, agg_op, average, block><<<grid, block, 0, stream>>>(nrows, device_valid_count.data(), FArgs...);

    null_count = nrows - device_valid_count.value();

    // check the stream for debugging
    CHECK_CUDA(stream);

    cudf::nvtx::range_pop();
  }

  /**
   * @brief If we cannot perform aggregation on this type then throw an error
   */
  template<typename ColumnType, class agg_op, bool average, class... TArgs,
     typename std::enable_if_t<!cudf::detail::is_supported<ColumnType, agg_op>(), std::nullptr_t> = nullptr>
  void dispatch_aggregation_type(cudf::size_type nrows, gdf_size_type& null_count, cudaStream_t stream, TArgs... FArgs)
  {
    CUDF_FAIL("Unsupported column type/operation combo. Only `min` and `max` are supported for non-arithmetic types for aggregations.");
  }

  /**
   * @brief Helper function for gdf_rolling. Deduces the type of the
   * aggregation column and type and calls another function to invoke the
   * rolling window kernel.
   */
  template <typename ColumnType>
  void operator()(cudf::size_type nrows,
      gdf_size_type &null_count,
      gdf_agg_op agg_type,
      void *out_col_data_ptr, cudf::valid_type *out_col_valid_ptr,
      void *in_col_data_ptr, cudf::valid_type *in_col_valid_ptr,
      cudf::size_type window,
      cudf::size_type min_periods,
      cudf::size_type forward_window,
      const cudf::size_type *window_col,
      const cudf::size_type *min_periods_col,
      const cudf::size_type *forward_window_col,
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
      dispatch_aggregation_type<ColumnType, cudf::DeviceSum, false>(nrows, null_count, stream,
                typed_out_data, typed_out_valid,
                typed_in_data, typed_in_valid,
                window, min_periods, forward_window,
                window_col, min_periods_col, forward_window_col);
      break;
    case GDF_MIN:
      dispatch_aggregation_type<ColumnType, cudf::DeviceMin, false>(nrows, null_count, stream,
                 typed_out_data, typed_out_valid,
                 typed_in_data, typed_in_valid,
                 window, min_periods, forward_window,
                 window_col, min_periods_col, forward_window_col);
      break;
    case GDF_MAX:
      dispatch_aggregation_type<ColumnType, cudf::DeviceMax, false>(nrows, null_count, stream,
                 typed_out_data, typed_out_valid,
                 typed_in_data, typed_in_valid,
                 window, min_periods, forward_window,
                 window_col, min_periods_col, forward_window_col);
      break;
    case GDF_COUNT:
      dispatch_aggregation_type<ColumnType, cudf::DeviceCount, false>(nrows, null_count, stream,
                 typed_out_data, typed_out_valid,
                 typed_in_data, typed_in_valid,
                 window, min_periods, forward_window,
                 window_col, min_periods_col, forward_window_col);
      break;
    case GDF_AVG:
      dispatch_aggregation_type<ColumnType, cudf::DeviceSum, true>(nrows, null_count, stream,
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
                           cudf::size_type window,
                           cudf::size_type min_periods,
                           cudf::size_type forward_window,
                           gdf_agg_op agg_type,
                           const cudf::size_type *window_col,
                           const cudf::size_type *min_periods_col,
                           const cudf::size_type *forward_window_col)
{
  CUDF_EXPECTS((window >= 0) && (min_periods >= 0) && (forward_window >= 0), "Window size and min periods must be non-negative");

  // Use the column wrapper class from io/utilities to quickly create a column
  gdf_column_wrapper output_col(input_col.size,
                                input_col.dtype,
                                input_col.dtype_info,
                                input_col.col_name == nullptr ? "" : std::string(input_col.col_name));

  // If there are no rows in the input, return successfully
  if (input_col.size == 0)
    return output_col.release();

  // Allocate memory for the output column
  output_col.allocate();

  // At least one observation is required to procure a valid output
  min_periods = std::max(min_periods, 1);

  // always use the default stream for now
  cudaStream_t stream = NULL;

  // Launch type dispatcher
  cudf::type_dispatcher(input_col.dtype,
                        rolling_window_launcher{},
                        input_col.size, output_col->null_count, agg_type,
                        output_col->data, output_col->valid,
                        input_col.data, input_col.valid,
                        window, min_periods, forward_window,
                        window_col, min_periods_col, forward_window_col,
                        stream);

  // Release the gdf pointer from the wrapper class
  return output_col.release();
}

gdf_column rolling_window(gdf_column const& input,
                           cudf::size_type window,
                           cudf::size_type min_periods,
                           cudf::size_type forward_window,
                           const std::string& user_defined_aggregator,
                           gdf_agg_op agg_op,
                           gdf_dtype output_type,
                           cudf::size_type const* window_col,
                           cudf::size_type const* min_periods_col,
                           cudf::size_type const* forward_window_col)
{
  CUDF_EXPECTS((window >= 0) && (min_periods >= 0) && (forward_window >= 0), "Window size and min periods must be non-negative");

  gdf_column output = allocate_column(output_type, input.size, true);

  // If there are no rows in the input, return successfully
  if (input.size == 0)
    return output;

  if (input.null_count > 0) {
    CUDF_FAIL("Currently the UDF version of rolling window"
        " does NOT support inputs with nulls.");
  }

  // At least one observation is required to procure a valid output
  min_periods = std::max(min_periods, 1);

  std::string hash = "prog_rolling." 
    + std::to_string(std::hash<std::string>{}(user_defined_aggregator));
  
  std::string cuda_source;
  switch(agg_op){            
    case GDF_NUMBA_GENERIC_AGG_OPS:
      cuda_source = 
        cudf::jit::parse_single_function_ptx(
          user_defined_aggregator, 
          cudf::rolling::jit::get_function_name(agg_op), 
          cudf::jit::getTypeName(output_type), {0, 5} // {0, 5} means the first and sixth args are pointers.
        ) + cudf::rolling::jit::code::kernel;
      break; 
    case GDF_CUDA_GENERIC_AGG_OPS:
      cuda_source = 
        cudf::jit::parse_single_function_cuda(
          user_defined_aggregator, 
          cudf::rolling::jit::get_function_name(agg_op) 
        ) + cudf::rolling::jit::code::kernel;
      break;
    default:
      CUDF_FAIL("Unsupported UDF type.");
  }
  
  // Launch the jitify kernel
  cudf::jit::launcher(
    hash, cuda_source,
    { cudf::rolling::jit::code::operation_h , cudf_types_h, cudf_types_hpp },
    { "-std=c++14" }, nullptr
  ).set_kernel_inst(
    "gpu_rolling", // name of the kernel we are launching
    { cudf::jit::getTypeName(output.dtype), // list of template arguments
      cudf::jit::getTypeName(input.dtype),
      cudf::rolling::jit::get_operator_name(agg_op) } 
  ).launch(
    output.size,
    output.data,
    output.valid,
    input.data,
    input.valid,
    window,
    min_periods,
    forward_window,
    window_col,
    min_periods_col,
    forward_window_col
  );

  set_null_count(output);

  return output;
}

}  // namespace cudf
