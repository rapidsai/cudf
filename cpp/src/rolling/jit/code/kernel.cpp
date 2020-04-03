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

namespace cudf {
namespace experimental {
namespace rolling {
namespace jit {
namespace code {



const char* kernel_headers = 
R"***(
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
)***";

const char* kernel =
R"***(
#include "operation.h"

template<class T>
T minimum(T a, T b){
  return a < b ? a : b;
}

struct preceding_window_wrapper {

  const cudf::size_type *d_group_offsets;
  const cudf::size_type *d_group_labels;
  cudf::size_type window;
  
  cudf::size_type operator[](cudf::size_type idx){
    auto group_label = d_group_labels[idx];
    auto group_start = d_group_offsets[group_label];
    return minimum(window, idx - group_start + 1); // Preceding includes current row.
  }

};

struct following_window_wrapper {

  const cudf::size_type *d_group_offsets;
  const cudf::size_type *d_group_labels;
  cudf::size_type window;
  
  cudf::size_type operator[](cudf::size_type idx){
    auto group_label = d_group_labels[idx];
    auto group_end = d_group_offsets[group_label+1]; // Cannot fall off the end, since offsets is capped with `input.size()`.
    return minimum(window, (group_end - 1) - idx);
  }

};

template<class Iter, class T>
cudf::size_type __device__ upper_bound(Iter iter, cudf::size_type first, cudf::size_type last, const T& value)
{
    cudf::size_type current;
    cudf::size_type count, step;
    count = last - first;

    while (count > 0) {
      current = first;
      step = count / 2;
      current += step;
      if (!(value < iter[current])) {
        first = ++current;
        count -= step + 1;
      }
      else
        count = step;
    }
    return first;
}

template<class Iter, class T>
cudf::size_type __device__ lower_bound(Iter iter, cudf::size_type first, cudf::size_type last, const T& value)
{
    cudf::size_type current;
    cudf::size_type count, step;
    count = last - first;

    while (count > 0) {
      current = first;
      step = count / 2;
      current += step;
      if (iter[current] < value) {
        first = ++current;
        count -= step + 1;
      }
      else
        count = step;
    }
    return first;
}

template<class T>
struct timestamp_preceding_window_wrapper {
  
  const cudf::size_type *d_group_offsets;
  const cudf::size_type *d_group_labels;
  const T *d_timestamps;
  cudf::size_type window_in_days;
  T mult_factor;

  cudf::size_type __device__ operator[](cudf::size_type idx) {
    auto group_label = d_group_labels[idx];
    auto group_start = d_group_offsets[group_label];
    auto lower_bound_ = d_timestamps[idx] - window_in_days*mult_factor;
    
    return idx - lower_bound(d_timestamps, group_start, idx, lower_bound_) + 1;
    // Add 1, for `preceding` to account for current row.
  } 
};

template<class T>
struct timestamp_following_window_wrapper {
  
  const cudf::size_type *d_group_offsets;
  const cudf::size_type *d_group_labels;
  const T *d_timestamps;
  cudf::size_type window_in_days;
  T mult_factor;

  cudf::size_type __device__ operator[](cudf::size_type idx) {
    auto group_label = d_group_labels[idx];
    auto group_end = d_group_offsets[group_label+1]; // Cannot fall off the end, since offsets is capped with `input.size()`.
    auto upper_bound_ = d_timestamps[idx] + window_in_days*mult_factor;

    return upper_bound(d_timestamps, idx, group_end, upper_bound_) - idx - 1;
  }
};

template <typename WindowType>
cudf::size_type __device__ get_window(WindowType window, cudf::size_type index) { return window[index]; }

template <>
cudf::size_type __device__ get_window(cudf::size_type window, cudf::size_type index) { return window; }

template <typename InType, typename OutType, class agg_op, typename PrecedingWindowType, typename FollowingWindowType>
__global__
void gpu_rolling_new(cudf::size_type nrows,
                 InType const* const __restrict__ in_col, 
                 cudf::bitmask_type const* const __restrict__ in_col_valid,
                 OutType* __restrict__ out_col, 
                 cudf::bitmask_type* __restrict__ out_col_valid,
                 cudf::size_type * __restrict__ output_valid_count,
                 PrecedingWindowType preceding_window_begin,
                 FollowingWindowType following_window_begin,
                 cudf::size_type min_periods)
{
  cudf::size_type i = blockIdx.x * blockDim.x + threadIdx.x;
  cudf::size_type stride = blockDim.x * gridDim.x;

  cudf::size_type warp_valid_count{0};

  auto active_threads = __ballot_sync(0xffffffff, i < nrows);
  while(i < nrows)
  {
    // declare this as volatile to avoid some compiler optimizations that lead to incorrect results
    // for CUDA 10.0 and below (fixed in CUDA 10.1)
    volatile cudf::size_type count = 0;

    cudf::size_type preceding_window = get_window(preceding_window_begin, i);
    cudf::size_type following_window = get_window(following_window_begin, i);

    // compute bounds
    cudf::size_type start = min(nrows, max(0, i - preceding_window + 1));
    cudf::size_type end = min(nrows, max(0, i + following_window + 1));
    cudf::size_type start_index = min(start, end);
    cudf::size_type end_index = max(start, end);

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

    // store the output value, one per thread
    if (output_is_valid) {
      out_col[i] = val;
    }

    // only one thread writes the mask
    if (0 == cudf::intra_word_index(i)) {
      out_col_valid[cudf::word_index(i)] = result_mask;
      warp_valid_count += __popc(result_mask);
    }

    // process next element 
    i += stride;
    active_threads = __ballot_sync(active_threads, i < nrows);
  }

  // TODO: likely faster to do a single_lane_block_reduce and a single
  // atomic per block but that requires jitifying single_lane_block_reduce...
  if(0 == cudf::intra_word_index(threadIdx.x)) {
    atomicAdd(output_valid_count, warp_valid_count);
  }
}
)***";

} // namespace code
} // namespace jit
} // namespace rolling
} // namespace experimental
} // namespace cudf
