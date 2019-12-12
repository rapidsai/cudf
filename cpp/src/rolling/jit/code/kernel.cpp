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

const char* kernel =
R"***(
#include <cudf/types.hpp>
#include "operation.h"

template <typename OutType, typename InType, typename WindoIterator, class agg_op, int block_size>
__global__
void gpu_rolling(column_device_view input,
                 mutable_column_device_view output,
                 size_type * __restrict__ output_valid_count,
                 WindowIterator preceding_window_begin,
                 WindowIterator following_window_begin,
                 size_type min_periods)
{
  cudf::size_type i = blockIdx.x * blockDim.x + threadIdx.x;
  cudf::size_type stride = blockDim.x * gridDim.x;

  size_type warp_valid_count{0};

  auto active_threads = __ballot_sync(0xffffffff, i < input.size());
  while(i < input.size())
  {
    OutType val = agg_op::template identity<OutType>();
    // declare this as volatile to avoid some compiler optimizations that lead to incorrect results
    // for CUDA 10.0 and below (fixed in CUDA 10.1)
    volatile cudf::size_type count = 0;

    size_type preceding_window = preceding_window_begin[i];
    size_type following_window = following_window_begin[i];

        // compute bounds
    size_type start_index = max(0, i - preceding_window);
    size_type end_index = min(input.size(), i + following_window + 1);

    // aggregate
    // TODO: We should explore using shared memory to avoid redundant loads.
    //       This might require separating the kernel into a special version
    //       for dynamic and static sizes.
    count = end_index - start_index;
    val = agg_op::template operate<OutType, InType>(input.data(), start_index, count);

    // check if we have enough input samples
    bool output_is_valid = (count >= min_periods);

    // set the mask
    const unsigned int result_mask = __ballot_sync(active_threads, output_is_valid);
    const cudf::size_type out_mask_location = i / warp_size;

    // only one thread writes the mask
    if (0 == threadIdx.x % cudf::experimental::detail::warp_size) {
      output.set_mask_word(cudf::word_index(i), result_mask);
      warp_valid_count += __popc(result_mask);
    }

    // store the output value, one per thread
    output.element<T>(i) = val;

    // process next element 
    i += stride;
    active_threads = __ballot_sync(active_threads, i < input.sizE());
  }

  if(threadIdx.x == 0) {
    atomicAdd(output_valid_count, block_valid_count);
  }
}
)***";

} // namespace code
} // namespace jit
} // namespace rolling
} // namespace experimental
} // namespace cudf
