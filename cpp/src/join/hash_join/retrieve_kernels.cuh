/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// Hash join retrieve kernel using prefix-scan offsets.  Each CG knows
// exactly where to write — no atomics, no shared-memory buffering.
// Uses cuco ref public APIs: storage_ref(), probing_scheme(), empty_key_sentinel(), key_eq().

#pragma once

#include "kernels_common.cuh"

#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cooperative_groups.h>
#include <cuco/detail/utils.cuh>
#include <thrust/scan.h>

namespace cudf::detail {

template <bool IsOuter, typename Ref>
CUDF_KERNEL void __launch_bounds__(DEFAULT_JOIN_BLOCK_SIZE)
  retrieve_kernel(probe_key_type const* __restrict__ input_probe,
                  size_type const* __restrict__ offsets,
                  cuda::std::int64_t n,
                  size_type* __restrict__ left_output,
                  size_type* __restrict__ right_output,
                  Ref ref)
{
  namespace cg = cooperative_groups;

  auto constexpr cg_size     = Ref::cg_size;
  auto constexpr bucket_size = Ref::bucket_size;
  auto const empty_sentinel  = ref.empty_key_sentinel();
  auto const key_equal       = ref.key_eq();

  auto const tile   = cg::tiled_partition<cg_size>(cg::this_thread_block());
  auto idx          = grid_1d::global_thread_id() / cg_size;
  auto const stride = grid_1d::grid_stride() / cg_size;

  while (idx < n) {
    auto const probe_key  = input_probe[idx];
    auto const left_index = probe_key.second;
    auto write_pos        = static_cast<size_type>(offsets[idx]);

    auto probing_iter = ref.probing_scheme().template make_iterator<bucket_size>(
      tile, probe_key, ref.storage_ref().extent());
    auto const init_probing_idx = *probing_iter;

    bool running                      = true;
    [[maybe_unused]] bool found_match = false;

    while (tile.any(running)) {
      if (running) {
        auto const bucket_slots = ref.storage_ref()[*probing_iter];

        bool equals[bucket_size];
        for (int i = 0; i < bucket_size; ++i) {
          equals[i] = false;
          if (running) {
            if (bucket_slots[i] == empty_sentinel) {
              running = false;
            } else if (key_equal(probe_key, bucket_slots[i])) {
              equals[i] = true;
            }
          }
        }

        tile.sync();
        running = tile.all(running);

        for (int i = 0; i < bucket_size; ++i) {
          auto const match_mask  = tile.ballot(equals[i]);
          auto const num_matches = __popc(match_mask);

          if (equals[i]) {
            auto const lane_offset =
              cuco::detail::count_least_significant_bits(match_mask, tile.thread_rank());
            left_output[write_pos + lane_offset]  = left_index;
            right_output[write_pos + lane_offset] = bucket_slots[i].second;
            if constexpr (IsOuter) { found_match = true; }
          }

          if (tile.thread_rank() == 0) { write_pos += num_matches; }
          write_pos = tile.shfl(write_pos, 0);
        }
      }

      ++probing_iter;
      if (*probing_iter == init_probing_idx) { running = false; }
    }

    if constexpr (IsOuter) {
      if (!found_match && tile.thread_rank() == 0) {
        left_output[write_pos]  = left_index;
        right_output[write_pos] = JoinNoMatch;
      }
    }

    idx += stride;
  }
}

template <bool IsOuter, typename Ref>
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
launch_retrieve(probe_key_type const* keys,
                cuda::std::int64_t n,
                size_type const* match_counts,
                Ref ref,
                rmm::cuda_stream_view stream,
                rmm::device_async_resource_ref mr)
{
  if (n == 0) {
    return std::pair(std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr),
                     std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr));
  }

  // Exclusive scan of match counts to get per-row output offsets.
  rmm::device_uvector<size_type> offsets(n, stream);
  thrust::exclusive_scan(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                         match_counts,
                         match_counts + n,
                         offsets.begin());

  // Total output size = last offset + last count.  Batch both D2H copies.
  size_type last_offset     = 0;
  size_type last_count      = 0;
  void* const dsts[]        = {&last_offset, &last_count};
  void const* const srcs[]  = {offsets.data() + n - 1, match_counts + n - 1};
  std::size_t const sizes[] = {sizeof(size_type), sizeof(size_type)};
  CUDF_CUDA_TRY(cudf::detail::memcpy_batch_async(dsts, srcs, sizes, 2, stream));
  stream.synchronize();
  auto const total_output = static_cast<std::size_t>(last_offset) + last_count;

  if (total_output == 0) {
    return std::pair(std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr),
                     std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr));
  }

  auto left_indices  = std::make_unique<rmm::device_uvector<size_type>>(total_output, stream, mr);
  auto right_indices = std::make_unique<rmm::device_uvector<size_type>>(total_output, stream, mr);

  auto const config =
    grid_1d{static_cast<thread_index_type>(n * DEFAULT_JOIN_CG_SIZE), DEFAULT_JOIN_BLOCK_SIZE};

  retrieve_kernel<IsOuter><<<config.num_blocks, config.num_threads_per_block, 0, stream.value()>>>(
    keys, offsets.data(), n, left_indices->data(), right_indices->data(), ref);

  return std::pair(std::move(left_indices), std::move(right_indices));
}

}  // namespace cudf::detail
