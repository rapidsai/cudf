/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// Ported from cuco's open_addressing count_each kernel.

#pragma once

#include "kernels_common.cuh"

#include <cudf/detail/utilities/grid_1d.cuh>

#include <rmm/cuda_stream_view.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cudf::detail {

template <bool IsOuter, typename Ref>
CUDF_KERNEL void __launch_bounds__(DEFAULT_JOIN_BLOCK_SIZE)
  count_each_kernel(probe_key_type const* __restrict__ keys,
                    cuda::std::int64_t n,
                    size_type* __restrict__ output,
                    Ref ref)
{
  auto constexpr cg_size = DEFAULT_JOIN_CG_SIZE;

  auto idx          = grid_1d::global_thread_id() / cg_size;
  auto const stride = grid_1d::grid_stride() / cg_size;

  while (idx < n) {
    auto const key = keys[idx];
    if constexpr (cg_size == 1) {
      auto const cnt = ref.count(key);
      if constexpr (IsOuter) {
        output[idx] = (cnt == 0) ? size_type{1} : cnt;
      } else {
        output[idx] = cnt;
      }
    } else {
      auto const tile =
        cooperative_groups::tiled_partition<cg_size>(cooperative_groups::this_thread_block());
      if constexpr (IsOuter) {
        auto temp_count = static_cast<size_type>(ref.count(tile, key));
        if (tile.all(temp_count == 0) and tile.thread_rank() == 0) { ++temp_count; }
        auto const cnt =
          cooperative_groups::reduce(tile, temp_count, cooperative_groups::plus<size_type>());
        if (tile.thread_rank() == 0) { output[idx] = cnt; }
      } else {
        auto const cnt = cooperative_groups::reduce(tile,
                                                    static_cast<size_type>(ref.count(tile, key)),
                                                    cooperative_groups::plus<size_type>());
        if (tile.thread_rank() == 0) { output[idx] = cnt; }
      }
    }
    idx += stride;
  }
}

template <bool IsOuter, typename Ref>
void launch_count_each(probe_key_type const* keys,
                       cuda::std::int64_t n,
                       size_type* output,
                       Ref ref,
                       rmm::cuda_stream_view stream)
{
  if (n == 0) { return; }

  auto const config =
    grid_1d{static_cast<thread_index_type>(n * DEFAULT_JOIN_CG_SIZE), DEFAULT_JOIN_BLOCK_SIZE};

  count_each_kernel<IsOuter>
    <<<config.num_blocks, config.num_threads_per_block, 0, stream.value()>>>(keys, n, output, ref);
}

}  // namespace cudf::detail
