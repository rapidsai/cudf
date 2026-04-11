/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cudf/column/column_device_view_base.cuh>
#include <cudf/types.hpp>

#include <cuda/ptx>
#include <cuda/std/bit>

namespace cudf {
namespace jit {

__device__ inline bool warp_elect(unsigned int mask)
{
#if __CUDA_ARCH__ >= 900
  // use elect.sync
  return cuda::ptx::elect_sync(mask);
#else
  // fallback: manually elect a leader (e.g., the first active thread)
  int leader = mask == 0 ? 0 : cuda::std::countr_zero(mask);
  int lane   = (threadIdx.x & 31);
  return (lane == leader);
#endif
}

template <typename Out>
__device__ void warp_compact_validity(unsigned int active_mask,
                                      mutable_column_device_view_core const* outcols,
                                      size_type row,
                                      bool is_valid)
{
  if constexpr (!Out::may_be_nullable) {
    return;
  } else {
    auto null_word = __ballot_sync(active_mask, is_valid);
    // use warp-elect to make sure we only issue one memory transaction per warp
    if (warp_elect(active_mask)) { Out::set_null_mask_word(outcols, row / 32, null_word); }
  }
}

}  // namespace jit
}  // namespace cudf
