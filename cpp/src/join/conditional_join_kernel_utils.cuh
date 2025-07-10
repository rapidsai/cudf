/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

#pragma once

#include <cudf/types.hpp>

#include <cub/util_ptx.cuh>
#include <cuda/atomic>
#include <cuda/std/cstddef>

namespace cudf {
namespace detail {
// TODO(lamarrr): remove
namespace jit {
inline constexpr size_type warp_size{32};
}

/**
 * @brief Adds a pair of indices to the shared memory cache
 *
 * @param[in] first The first index in the pair
 * @param[in] second The second index in the pair
 * @param[in,out] current_idx_shared Pointer to shared index that determines
 * where in the shared memory cache the pair will be written
 * @param[in] warp_id The ID of the warp of the calling the thread
 * @param[out] joined_shared_l Pointer to the shared memory cache for left indices
 * @param[out] joined_shared_r Pointer to the shared memory cache for right indices
 */
__inline__ __device__ void add_pair_to_cache(size_type const first,
                                             size_type const second,
                                             std::size_t* current_idx_shared,
                                             int const warp_id,
                                             size_type* joined_shared_l,
                                             size_type* joined_shared_r)
{
  cuda::atomic_ref<std::size_t, cuda::thread_scope_block> ref{*(current_idx_shared + warp_id)};
  std::size_t my_current_idx = ref.fetch_add(1, cuda::memory_order_relaxed);
  // It's guaranteed to fit into the shared cache
  joined_shared_l[my_current_idx] = first;
  joined_shared_r[my_current_idx] = second;
}

__inline__ __device__ void add_left_to_cache(size_type const first,
                                             std::size_t* current_idx_shared,
                                             int const warp_id,
                                             size_type* joined_shared_l)
{
  cuda::atomic_ref<std::size_t, cuda::thread_scope_block> ref{*(current_idx_shared + warp_id)};
  std::size_t my_current_idx      = ref.fetch_add(1, cuda::memory_order_relaxed);
  joined_shared_l[my_current_idx] = first;
}

template <int num_warps, cudf::size_type output_cache_size>
__device__ void flush_output_cache(unsigned int const activemask,
                                   std::size_t const max_size,
                                   int const warp_id,
                                   int const lane_id,
                                   std::size_t* current_idx,
                                   std::size_t current_idx_shared[num_warps],
                                   size_type join_shared_l[num_warps][output_cache_size],
                                   size_type join_shared_r[num_warps][output_cache_size],
                                   size_type* join_output_l,
                                   size_type* join_output_r)
{
  // count how many active threads participating here which could be less than warp_size
  int const num_threads     = __popc(activemask);
  std::size_t output_offset = 0;

  if (0 == lane_id) {
    cuda::atomic_ref<std::size_t, cuda::thread_scope_device> ref{*current_idx};
    output_offset = ref.fetch_add(current_idx_shared[warp_id], cuda::memory_order_relaxed);
  }

  // No warp sync is necessary here because we are assuming that ShuffleIndex
  // is internally using post-CUDA 9.0 synchronization-safe primitives
  // (__shfl_sync instead of __shfl). __shfl is technically not guaranteed to
  // be safe by the compiler because it is not required by the standard to
  // converge divergent branches before executing.
  output_offset = cub::ShuffleIndex<detail::jit::warp_size>(output_offset, 0, activemask);

  for (std::size_t shared_out_idx = static_cast<std::size_t>(lane_id);
       shared_out_idx < current_idx_shared[warp_id];
       shared_out_idx += num_threads) {
    std::size_t thread_offset = output_offset + shared_out_idx;
    if (thread_offset < max_size) {
      join_output_l[thread_offset] = join_shared_l[warp_id][shared_out_idx];
      join_output_r[thread_offset] = join_shared_r[warp_id][shared_out_idx];
    }
  }
}

template <int num_warps, cudf::size_type output_cache_size>
__device__ void flush_output_cache(unsigned int const activemask,
                                   std::size_t const max_size,
                                   int const warp_id,
                                   int const lane_id,
                                   std::size_t* current_idx,
                                   std::size_t current_idx_shared[num_warps],
                                   size_type join_shared_l[num_warps][output_cache_size],
                                   size_type* join_output_l)
{
  int const num_threads     = __popc(activemask);
  std::size_t output_offset = 0;

  if (0 == lane_id) {
    cuda::atomic_ref<std::size_t, cuda::thread_scope_device> ref{*current_idx};
    output_offset = ref.fetch_add(current_idx_shared[warp_id], cuda::memory_order_relaxed);
  }

  output_offset = cub::ShuffleIndex<detail::jit::warp_size>(output_offset, 0, activemask);

  for (std::size_t shared_out_idx = static_cast<std::size_t>(lane_id);
       shared_out_idx < current_idx_shared[warp_id];
       shared_out_idx += num_threads) {
    std::size_t thread_offset = output_offset + shared_out_idx;
    if (thread_offset < max_size) {
      join_output_l[thread_offset] = join_shared_l[warp_id][shared_out_idx];
    }
  }
}

}  // namespace detail
}  // namespace cudf
