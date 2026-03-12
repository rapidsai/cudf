/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cudf/column/column_device_view_base.cuh>
#include <cudf/types.hpp>

#include <cuda/ptx>

namespace cudf {
namespace jit {

__device__ inline bool warp_elect(unsigned int mask)
{
#if __CUDA_ARCH__ >= 900
  // use elect.sync
  return cuda::ptx::elect_sync(mask);
#else
  // fallback: manually elect a leader (e.g., the first active thread)
  int leader = __ffs(mask) - 1;
  int lane   = (threadIdx.x & 31);
  return (lane == leader);
#endif
}

template <typename Out>
__device__ void warp_compact_validity(mutable_column_device_view_core const* outcols,
                                      size_type row,
                                      bool is_valid)
{
  if constexpr (!Out::may_be_nullable) {
    return;
  } else {
    auto active    = __activemask();
    auto null_word = __ballot_sync(active, is_valid);
    // use warp-elect to make sure we only issue one memory transaction per warp
    if (warp_elect(active)) { Out::set_null_mask_word(outcols, row / 32, null_word); }
  }
}

}  // namespace jit
}  // namespace cudf
