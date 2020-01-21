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

namespace cudf {
namespace rolling {
namespace jit {
namespace code {

const char* kernel =
R"***(
#include <cudf/types.h>
#include <cudf/types.hpp>
#include "operation.h"

template <typename OutType, typename InType, class agg_op>
__global__
void gpu_rolling(cudf::size_type nrows,
                 OutType* __restrict__ out_col, 
                 unsigned int* __restrict__ out_col_valid,
                 InType const* const __restrict__ in_col, 
                 unsigned int const* const __restrict__ in_col_valid,
                 cudf::size_type window,
                 cudf::size_type min_periods,
                 cudf::size_type forward_window,
                 const cudf::size_type *window_col,
                 const cudf::size_type *min_periods_col,
                 const cudf::size_type *forward_window_col)
{
  constexpr int warp_size = 32;

  cudf::size_type i = blockIdx.x * blockDim.x + threadIdx.x;
  cudf::size_type stride = blockDim.x * gridDim.x;

  auto active_threads = __ballot_sync(0xffffffff, i < nrows);
  while(i < nrows)
  {
    OutType val;
    volatile cudf::size_type count = 0;	// declare this as volatile to avoid some compiler optimizations that lead to incorrect results for CUDA 10.0 and below (fixed in CUDA 10.1)

    // dynamic window handling
    if (window_col != nullptr) window = window_col[i];
    if (min_periods_col != nullptr) min_periods = max(min_periods_col[i], 1);	// at least one observation is required
    if (forward_window_col != nullptr) forward_window = forward_window_col[i];

    // compute bounds
    cudf::size_type start_index = max((cudf::size_type)0, i - window + 1);
    cudf::size_type end_index = min(nrows, i + forward_window + 1);       // exclusive

    // aggregate
    count = end_index - start_index;
    val = agg_op::template operate<OutType, InType>(in_col, start_index, count);

    // check if we have enough input samples
    bool output_is_valid = (count >= min_periods);

    // set the mask
    const unsigned int result_mask = __ballot_sync(active_threads, output_is_valid);
    const cudf::size_type out_mask_location = i / warp_size;

    // only one thread writes the mask
    if (0 == threadIdx.x % warp_size){
      out_col_valid[out_mask_location] = result_mask;
    }

    // store the output value, one per thread
    out_col[i] = val;

    // process next element 
    i += stride;
    active_threads = __ballot_sync(active_threads, i < nrows);
  }
}
)***";

} // namespace code
} // namespace jit
} // namespace rolling
} // namespace cudf
