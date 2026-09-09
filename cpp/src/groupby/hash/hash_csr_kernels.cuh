/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/hashing.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>

#include <cuco/pair.cuh>
#include <cuda/atomic>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/utility>
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

/// Inputs with at least this many rows size the table from an estimate of the number of distinct
/// keys instead of the row count; smaller inputs get a table for every row right away.
constexpr size_type hash_csr_min_rows_to_estimate = 1 << 21;
/// Keys wider than this (in bytes per row) always get a table for every row: every probe of an
/// occupied slot compares keys, which costs more than the smaller table saves for them.
constexpr size_type hash_csr_max_estimated_key_bytes = 32;
/// Slots of the table the distinct keys of a sample of the rows are counted in.
constexpr cuda::std::uint32_t hash_csr_sample_capacity = 1u << 20;
/// Fewest slots of a table sized from an estimate, so that a handful of hot slots still spread
/// over many cache lines.
constexpr cuda::std::uint32_t hash_csr_min_estimated_capacity = 1u << 18;
/// A probe this long is vanishingly unlikely at a load factor of one half, so it means the table
/// is (nearly) full and the build has to restart with a larger one.
constexpr cuda::std::uint32_t hash_csr_max_probes = 256;

/// Device view of the linearly probed open-addressed table that maps each distinct key to a slot.
struct hash_csr_table_ref {
  hash_table_entry_type* entries;
  cuda::std::uint32_t capacity;
  cuda::std::uint32_t max_probes;  ///< Probes after which the table is declared full

  /**
   * @brief Returns the slot owned by the key of `row`, claiming an empty slot when the key is new.
   *
   * Most groupby rows repeat a key that is already in the table, so each slot is read before any
   * attempt to claim it and the compare-and-swap only runs on empty slots.
   *
   * @return The slot and whether `row` claimed it, or `capacity` when `max_probes` slots were
   * probed without success
   */
  template <typename Equal>
  __device__ cuda::std::pair<cuda::std::uint32_t, bool> insert_or_find(size_type row,
                                                                       hash_value_type hash,
                                                                       Equal const& equal) const
  {
    auto slot = static_cast<cuda::std::uint32_t>(hash % capacity);
    for (cuda::std::uint32_t step = 0; step < max_probes; ++step) {
      // A claimed slot never changes, so the first look may come from L1 (block scope). Only an
      // empty value can be stale; it is checked again at device scope before the slot is claimed,
      // so that a hot slot cached as empty costs one L2 load rather than a failed atomic.
      auto current =
        cuda::atomic_ref<hash_table_entry_type, cuda::thread_scope_block>{entries[slot]}.load(
          cuda::memory_order_relaxed);
      if (current == cudf::detail::CUDF_SIZE_TYPE_SENTINEL) {
        auto entry_ref =
          cuda::atomic_ref<hash_table_entry_type, cuda::thread_scope_device>{entries[slot]};
        current = entry_ref.load(cuda::memory_order_relaxed);
        if (current == cudf::detail::CUDF_SIZE_TYPE_SENTINEL &&
            entry_ref.compare_exchange_strong(current, row, cuda::memory_order_relaxed)) {
          return {slot, true};
        }
      }
      if (equal(row, current)) { return {slot, false}; }
      slot = slot + 1 == capacity ? 0 : slot + 1;
    }
    return {capacity, false};
  }
};

/**
 * @brief Inserts every valid row into the table and, when `positions` is given, records the slot
 * of each row and its rank within that slot.
 *
 * Ranks are handed out by one atomic per distinct slot per warp: lanes that landed in the same
 * slot combine their increments, which keeps low-cardinality inputs from serializing on a few
 * counters.
 *
 * When `overflow` is given, a row whose probe runs out of `table.max_probes` slots sets it and
 * the rows still to come are skipped: the caller then rebuilds with a larger table.
 */
template <typename Equal, typename Hasher>
CUDF_KERNEL void hash_csr_build_kernel(size_type num_rows,
                                       bitmask_type const* valid_rows,
                                       build_position_type* positions,
                                       size_type* slot_counts,
                                       hash_csr_table_ref table,
                                       Equal equal,
                                       Hasher hasher,
                                       int* overflow)
{
  auto const lane    = static_cast<cuda::std::uint32_t>(threadIdx.x % cudf::detail::warp_size);
  auto const stride  = cudf::detail::grid_1d::grid_stride();
  auto const is_full = [overflow] {
    return overflow != nullptr && cuda::atomic_ref<int, cuda::thread_scope_block>{*overflow}.load(
                                    cuda::memory_order_relaxed) != 0;
  };
  // Every lane of a warp runs the same number of iterations so the warp-wide match and shuffle
  // below always see a converged warp; lanes past the last row simply do not participate.
  for (auto first_row = cudf::detail::grid_1d::global_thread_id() - lane; first_row < num_rows;
       first_row += stride) {
    auto const row = first_row + lane;
    auto const is_active =
      row < num_rows &&
      (valid_rows == nullptr || cudf::bit_is_set(valid_rows, static_cast<size_type>(row)));
    auto slot = hash_csr_no_slot;
    if (is_active && !is_full()) {
      auto const index = static_cast<size_type>(row);
      slot             = table.insert_or_find(index, hasher(index), equal).first;
      if (slot == table.capacity) {
        cuda::atomic_ref<int, cuda::thread_scope_device>{*overflow}.store(
          1, cuda::memory_order_relaxed);
        slot = hash_csr_no_slot;
      }
    }
    // Without aggregations only the distinct keys matter, and the table alone provides them.
    if (positions == nullptr) { continue; }

    auto const has_slot    = slot != hash_csr_no_slot;
    auto const active_mask = __ballot_sync(0xffff'ffffu, has_slot);
    if (has_slot) {
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

/**
 * @brief Inserts every `stride`-th valid row into the table and counts the sampled rows and the
 * slots they claim.
 *
 * Rows are sampled one at a time rather than in runs, since neighboring rows often share a key.
 * `counts[0]` receives the number of valid sampled rows and `counts[1]` the number of distinct
 * keys among them.
 */
template <typename Equal, typename Hasher>
CUDF_KERNEL void hash_csr_sample_kernel(size_type num_rows,
                                        size_type stride,
                                        bitmask_type const* valid_rows,
                                        hash_csr_table_ref table,
                                        Equal equal,
                                        Hasher hasher,
                                        size_type* counts)
{
  auto const row = cudf::detail::grid_1d::global_thread_id() * stride;
  auto const is_valid =
    row < num_rows &&
    (valid_rows == nullptr || cudf::bit_is_set(valid_rows, static_cast<size_type>(row)));
  auto is_new = false;
  if (is_valid) {
    auto const index = static_cast<size_type>(row);
    is_new           = table.insert_or_find(index, hasher(index), equal).second;
  }
  // Every thread of the block reaches both counts, which is what they require.
  auto const num_valid = __syncthreads_count(is_valid);
  auto const num_new   = __syncthreads_count(is_new);
  if (threadIdx.x == 0) {
    cuda::atomic_ref<size_type, cuda::thread_scope_device>{counts[0]}.fetch_add(
      num_valid, cuda::memory_order_relaxed);
    cuda::atomic_ref<size_type, cuda::thread_scope_device>{counts[1]}.fetch_add(
      num_new, cuda::memory_order_relaxed);
  }
}

/// Scatters each valid row to its group: the start offset of its slot plus its rank in the slot.
CUDF_KERNEL void hash_csr_fill_kernel(size_type num_rows,
                                      build_position_type const* positions,
                                      size_type const* slot_offsets,
                                      size_type* grouped_rows)
{
  static_assert(sizeof(build_position_type) == sizeof(unsigned long long));
  auto const stride = cudf::detail::grid_1d::grid_stride();
  for (auto row = cudf::detail::grid_1d::global_thread_id(); row < num_rows; row += stride) {
    // The positions are read once, so they stream past the caches and leave them to the slot
    // offsets, which every row looks up, and to the grouped rows, which the aggregations read next.
    auto const packed = __ldcs(reinterpret_cast<unsigned long long const*>(positions) + row);
    auto const slot   = static_cast<cuda::std::uint32_t>(packed & 0xffff'ffffu);
    auto const rank   = static_cast<size_type>(packed >> 32);
    if (slot == hash_csr_no_slot) { continue; }
    grouped_rows[slot_offsets[slot] + rank] = static_cast<size_type>(row);
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
                                  int* overflow,
                                  cuda::stream_ref stream)
{
  if (num_rows == 0) { return; }
  auto const config = cudf::detail::grid_1d{num_rows, hash_csr_block_size};
  hash_csr_build_kernel<<<config.num_blocks, config.num_threads_per_block, 0, stream.get()>>>(
    num_rows, valid_rows, positions, slot_counts, table, equal, hasher, overflow);
  CUDF_CUDA_TRY(cudaGetLastError());
}

template <typename Equal, typename Hasher>
void launch_hash_csr_sample_kernel(size_type num_rows,
                                   size_type stride,
                                   bitmask_type const* valid_rows,
                                   hash_csr_table_ref table,
                                   Equal equal,
                                   Hasher hasher,
                                   size_type* counts,
                                   cuda::stream_ref stream)
{
  auto const num_samples = cudf::util::div_rounding_up_safe(num_rows, stride);
  if (num_samples == 0) { return; }
  auto const config = cudf::detail::grid_1d{num_samples, hash_csr_block_size};
  hash_csr_sample_kernel<<<config.num_blocks, config.num_threads_per_block, 0, stream.get()>>>(
    num_rows, stride, valid_rows, table, equal, hasher, counts);
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
