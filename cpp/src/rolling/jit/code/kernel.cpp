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
namespace experimental {
namespace rolling {
namespace jit {
namespace code {



const char* kernel_headers = 
R"***(
#include <cudf/types.hpp>
)***";

const char* kernel =
R"***(
#include "operation.h"
template <typename OutType, typename InType, class agg_op, bool static_window>
__global__
void gpu_rolling_new(cudf::size_type nrows,
                 OutType* __restrict__ out_col, 
                 cudf::valid_type* __restrict__ out_col_valid,
                 InType const* const __restrict__ in_col, 
                 cudf::valid_type const* const __restrict__ in_col_valid,
                 cudf::size_type * __restrict__ output_valid_count,
                 cudf::size_type const* __restrict__ preceding_window_begin,
                 cudf::size_type const* __restrict__ following_window_begin,
                 cudf::size_type min_periods)
{
  constexpr int warp_size{32};
  cudf::size_type i = blockIdx.x * blockDim.x + threadIdx.x;
  cudf::size_type stride = blockDim.x * gridDim.x;

  cudf::size_type warp_valid_count{0};

  auto active_threads = __ballot_sync(0xffffffff, i < nrows);
  while(i < nrows)
  {
    // declare this as volatile to avoid some compiler optimizations that lead to incorrect results
    // for CUDA 10.0 and below (fixed in CUDA 10.1)
    volatile cudf::size_type count = 0;

    cudf::size_type preceding_window = preceding_window_begin[static_window ? 0 : i];
    cudf::size_type following_window = following_window_begin[static_window ? 0 : i];

        // compute bounds
    cudf::size_type start_index = max(0, i - preceding_window);
    cudf::size_type end_index = min(nrows, i + following_window + 1);

    // aggregate
    // TODO: We should explore using shared memory to avoid redundant loads.
    //       This might require separating the kernel into a special version
    //       for dynamic and static sizes.
    count = end_index - start_index;
    OutType val = agg_op::template operate<OutType, InType>(in_col, start_index, count);

    // check if we have enough input samples
    bool output_is_valid = (count >= min_periods);

    // set the mask
    const unsigned int result_mask = __ballot_sync(active_threads, output_is_valid);
    const cudf::size_type out_mask_location = i / warp_size;

    // only one thread writes the mask
    if (0 == threadIdx.x % warp_size) {
      out_col_valid[out_mask_location] = result_mask;
      warp_valid_count += __popc(result_mask);
    }

    // store the output value, one per thread
    out_col[i] = val;

    // process next element 
    i += stride;
    active_threads = __ballot_sync(active_threads, i < nrows);
  }

  // TODO: likely faster to do a single_lane_block_reduce and a single
  // atomic per block but that requires jitifying all of that code...
  if(0 == threadIdx.x % warp_size) {
    atomicAdd(output_valid_count, warp_valid_count);
  }
}
)***";

} // namespace code
} // namespace jit
} // namespace rolling
} // namespace experimental
} // namespace cudf
