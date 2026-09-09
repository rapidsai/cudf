/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/hashing.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>

#include <cuco/pair.cuh>
#include <cuda/atomic>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/stream>

namespace cudf::groupby::detail::hash {

/// One open-addressed slot: the first input row that claimed the slot.
using hash_table_entry_type = size_type;

/// Where an input row landed: the slot owned by its key and its rank among the rows of that slot.
using build_position_type = cuco::pair<cuda::std::uint32_t, size_type>;

/// Slot recorded for rows that are excluded from the groupby because their keys contain nulls.
constexpr cuda::std::uint32_t hash_csr_no_slot =
  cuda::std::numeric_limits<cuda::std::uint32_t>::max();

constexpr thread_index_type hash_csr_block_size = 256;

/// Device view of the linearly probed open-addressed table that maps each distinct key to a slot.
struct hash_csr_table_ref {
  hash_table_entry_type* entries;
  cuda::std::uint32_t capacity;

  /**
   * @brief Returns the slot owned by the key of `row`, claiming an empty slot when the key is new.
   *
   * Most groupby rows repeat a key that is already in the table, so each slot is read before any
   * attempt to claim it and the compare-and-swap only runs on empty slots.
   */
  template <typename Equal>
  __device__ cuda::std::uint32_t insert_or_find(size_type row,
                                                hash_value_type hash,
                                                Equal const& equal) const
  {
    auto slot = static_cast<cuda::std::uint32_t>(hash % capacity);
    for (cuda::std::uint32_t step = 0; step < capacity; ++step) {
      auto entry_ref =
        cuda::atomic_ref<hash_table_entry_type, cuda::thread_scope_device>{entries[slot]};
      auto current = entry_ref.load(cuda::memory_order_relaxed);
      if (current == cudf::detail::CUDF_SIZE_TYPE_SENTINEL &&
          entry_ref.compare_exchange_strong(current, row, cuda::memory_order_relaxed)) {
        return slot;
      }
      if (equal(row, current)) { return slot; }
      slot = slot + 1 == capacity ? 0 : slot + 1;
    }
    return capacity;
  }
};

/**
 * @brief Inserts every valid row into the table and, when `positions` is given, records the slot
 * of each row and its rank within that slot.
 *
 * Ranks are handed out by one atomic per distinct slot per warp: lanes that landed in the same
 * slot combine their increments, which keeps low-cardinality inputs from serializing on a few
 * counters.
 */
template <typename Equal, typename Hasher>
CUDF_KERNEL void hash_csr_build_kernel(size_type num_rows,
                                       bitmask_type const* valid_rows,
                                       build_position_type* positions,
                                       size_type* slot_counts,
                                       hash_csr_table_ref table,
                                       Equal equal,
                                       Hasher hasher)
{
  auto const lane   = static_cast<cuda::std::uint32_t>(threadIdx.x % cudf::detail::warp_size);
  auto const stride = cudf::detail::grid_1d::grid_stride();
  // Every lane of a warp runs the same number of iterations so the warp-wide match and shuffle
  // below always see a converged warp; lanes past the last row simply do not participate.
  for (auto first_row = cudf::detail::grid_1d::global_thread_id() - lane; first_row < num_rows;
       first_row += stride) {
    auto const row = first_row + lane;
    auto const is_active =
      row < num_rows &&
      (valid_rows == nullptr || cudf::bit_is_set(valid_rows, static_cast<size_type>(row)));
    auto slot = hash_csr_no_slot;
    if (is_active) {
      auto const index = static_cast<size_type>(row);
      slot             = table.insert_or_find(index, hasher(index), equal);
    }
    // Without aggregations only the distinct keys matter, and the table alone provides them.
    if (positions == nullptr) { continue; }

    auto const active_mask = __ballot_sync(0xffff'ffffu, is_active);
    if (is_active) {
      auto const peers  = __match_any_sync(active_mask, slot);
      auto const leader = __ffs(static_cast<int>(peers)) - 1;
      size_type first_rank{};
      if (lane == static_cast<cuda::std::uint32_t>(leader)) {
        first_rank =
          cuda::atomic_ref<size_type, cuda::thread_scope_device>{slot_counts[slot]}.fetch_add(
            static_cast<size_type>(__popc(static_cast<int>(peers))), cuda::memory_order_relaxed);
      }
      first_rank = __shfl_sync(peers, first_rank, leader);
      auto const rank_in_warp =
        static_cast<size_type>(__popc(static_cast<int>(peers & ((1u << lane) - 1u))));
      positions[row] = {slot, first_rank + rank_in_warp};
    } else if (row < num_rows) {
      positions[row] = {hash_csr_no_slot, cudf::detail::CUDF_SIZE_TYPE_SENTINEL};
    }
  }
}

/// Scatters each valid row to its group: the start offset of its slot plus its rank in the slot.
CUDF_KERNEL void hash_csr_fill_kernel(size_type num_rows,
                                      build_position_type const* positions,
                                      size_type const* slot_offsets,
                                      size_type* grouped_rows)
{
  auto const stride = cudf::detail::grid_1d::grid_stride();
  for (auto row = cudf::detail::grid_1d::global_thread_id(); row < num_rows; row += stride) {
    auto const position = positions[row];
    if (position.first == hash_csr_no_slot) { continue; }
    grouped_rows[slot_offsets[position.first] + position.second] = static_cast<size_type>(row);
  }
}

template <typename Equal, typename Hasher>
void launch_hash_csr_build_kernel(size_type num_rows,
                                  bitmask_type const* valid_rows,
                                  build_position_type* positions,
                                  size_type* slot_counts,
                                  hash_csr_table_ref table,
                                  Equal equal,
                                  Hasher hasher,
                                  cuda::stream_ref stream)
{
  if (num_rows == 0) { return; }
  auto const config = cudf::detail::grid_1d{num_rows, hash_csr_block_size};
  hash_csr_build_kernel<<<config.num_blocks, config.num_threads_per_block, 0, stream.get()>>>(
    num_rows, valid_rows, positions, slot_counts, table, equal, hasher);
  CUDF_CUDA_TRY(cudaGetLastError());
}

inline void launch_hash_csr_fill_kernel(size_type num_rows,
                                        build_position_type const* positions,
                                        size_type const* slot_offsets,
                                        size_type* grouped_rows,
                                        cuda::stream_ref stream)
{
  if (num_rows == 0) { return; }
  auto const config = cudf::detail::grid_1d{num_rows, hash_csr_block_size};
  hash_csr_fill_kernel<<<config.num_blocks, config.num_threads_per_block, 0, stream.get()>>>(
    num_rows, positions, slot_offsets, grouped_rows);
  CUDF_CUDA_TRY(cudaGetLastError());
}

}  // namespace cudf::groupby::detail::hash
