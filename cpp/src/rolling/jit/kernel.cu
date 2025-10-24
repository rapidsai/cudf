/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>

#include <rolling/detail/rolling_jit.hpp>
#include <rolling/jit/operation.hpp>

namespace cudf {
namespace rolling {
namespace jit {

template <typename WindowType>
cudf::size_type __device__ get_window(WindowType window, cudf::thread_index_type index)
{
  return window[index];
}

template <>
cudf::size_type __device__ get_window(cudf::size_type window, cudf::thread_index_type index)
{
  return window;
}

template <typename InType,
          typename OutType,
          class agg_op,
          typename PrecedingWindowType,
          typename FollowingWindowType>
CUDF_KERNEL void gpu_rolling_new(cudf::size_type nrows,
                                 InType const* const __restrict__ in_col,
                                 cudf::bitmask_type const* const __restrict__ in_col_valid,
                                 OutType* __restrict__ out_col,
                                 cudf::bitmask_type* __restrict__ out_col_valid,
                                 cudf::size_type* __restrict__ output_valid_count,
                                 PrecedingWindowType preceding_window_begin,
                                 FollowingWindowType following_window_begin,
                                 cudf::size_type min_periods)
{
  auto i            = cudf::detail::grid_1d::global_thread_id();
  auto const stride = cudf::detail::grid_1d::grid_stride();

  cudf::size_type warp_valid_count{0};

  auto active_threads = __ballot_sync(0xffff'ffffu, i < nrows);
  while (i < nrows) {
    int64_t const preceding_window = get_window(preceding_window_begin, i);
    int64_t const following_window = get_window(following_window_begin, i);

    // compute bounds
    auto const start = static_cast<cudf::size_type>(
      min(static_cast<int64_t>(nrows), max(int64_t{0}, i - preceding_window + 1)));
    auto const end = static_cast<cudf::size_type>(
      min(static_cast<int64_t>(nrows), max(int64_t{0}, i + following_window + 1)));
    auto const start_index = min(start, end);
    auto const end_index   = max(start, end);

    // aggregate
    // TODO: We should explore using shared memory to avoid redundant loads.
    //       This might require separating the kernel into a special version
    //       for dynamic and static sizes.
    cudf::size_type count = end_index - start_index;
    OutType val           = agg_op::template operate<OutType, InType>(in_col, start_index, count);

    // check if we have enough input samples
    bool const output_is_valid = (count >= min_periods);

    // set the mask
    unsigned int const result_mask = __ballot_sync(active_threads, output_is_valid);

    // store the output value, one per thread
    if (output_is_valid) { out_col[i] = val; }

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
  if (0 == cudf::intra_word_index(threadIdx.x)) { atomicAdd(output_valid_count, warp_valid_count); }
}

}  // namespace jit
}  // namespace rolling
}  // namespace cudf
