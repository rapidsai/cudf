/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/hashing.hpp>
#include <cudf/join/join.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <cooperative_groups.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_scan.cuh>
#include <cuda/atomic>
#include <cuda/std/algorithm>
#include <cuda/std/cstdint>
#include <cuda/stream>

#include <climits>
#include <limits>

namespace cudf::detail {

/**
 * @brief A single append-only page in a dynamic CSR bucket.
 *
 * Pages hold up to 32 contiguous row references. Batches are grouped by resolved key-map slot
 * before append, so every allocated page has a contiguous slice in the fixed-capacity value
 * backing store and page allocation remains bounded by the configured row capacity.
 */
struct dynamic_csr_segment {
  cuda::std::uint32_t next;
  size_type begin;
  size_type length;
};

struct dynamic_csr_slot {
  cuda::std::uint32_t head;
  cuda::std::uint32_t tail;
  size_type size;
};

constexpr auto dynamic_csr_no_segment   = std::numeric_limits<cuda::std::uint32_t>::max();
constexpr auto dynamic_csr_invalid_slot = dynamic_csr_no_segment;
constexpr auto dynamic_csr_empty_hash   = std::numeric_limits<hash_value_type>::max();
// A key slot is claimed before its representative row can be published. Keep that transient
// state distinct from both an empty slot and a usable row hash so colliding inserters never call
// the row comparator with an uninitialized representative.
constexpr auto dynamic_csr_pending_hash = dynamic_csr_empty_hash - 1;

struct dynamic_csr_key_slot {
  hash_value_type hash;
  cuda::std::uint64_t representative;
};

/** @brief Open-addressed key map whose equality is supplied by the batch-aware caller. */
struct dynamic_csr_key_map_ref {
  dynamic_csr_key_slot* entries;
  cuda::std::uint32_t capacity;
  cuda::std::uint32_t capacity_mask;

  template <typename Equal>
  __device__ size_type find_or_insert(hash_value_type hash,
                                      cuda::std::uint64_t representative,
                                      Equal equal) const
  {
    // Reserve empty and pending state values. Mapping the two largest hashes to the same usable
    // hash is safe because equality remains the authority within a probe sequence.
    auto const normalized_hash =
      hash >= dynamic_csr_pending_hash ? dynamic_csr_pending_hash - 1 : hash;
    for (cuda::std::uint32_t step = 0; step < capacity; ++step) {
      // The map capacity is a power of two, so this avoids an expensive 64-bit modulo in every
      // probe step and keeps the full-hash probe sequence intact.
      auto const slot = (static_cast<cuda::std::uint32_t>(normalized_hash) + step) & capacity_mask;
      auto entry_hash =
        cuda::atomic_ref<hash_value_type, cuda::thread_scope_device>{entries[slot].hash};
      while (true) {
        auto current = entry_hash.load(cuda::memory_order_acquire);
        if (current == dynamic_csr_pending_hash) {
          // The claiming thread publishes the representative with a release store below.
          continue;
        }
        if (current == dynamic_csr_empty_hash) {
          auto expected = dynamic_csr_empty_hash;
          if (entry_hash.compare_exchange_strong(
                expected, dynamic_csr_pending_hash, cuda::memory_order_acq_rel)) {
            cuda::atomic_ref<cuda::std::uint64_t, cuda::thread_scope_device>{
              entries[slot].representative}
              .store(representative, cuda::memory_order_release);
            // The hash is the publication state. A reader may not decode the representative
            // until this fence and the following release store have made its payload visible.
            __threadfence();
            entry_hash.store(normalized_hash, cuda::memory_order_release);
            return slot;
          }
          // Another thread changed this exact slot. Reinspect it before moving onward so an
          // equal key cannot be assigned a second CSR chain.
          continue;
        }
        auto const stored_representative =
          cuda::atomic_ref<cuda::std::uint64_t, cuda::thread_scope_device>{
            entries[slot].representative}
            .load(cuda::memory_order_acquire);
        if (current == normalized_hash && equal(representative, stored_representative)) {
          return slot;
        }
        break;
      }
    }
    return capacity;
  }

  template <typename Equal>
  __device__ size_type find(hash_value_type hash, cuda::std::uint64_t probe, Equal equal) const
  {
    auto const normalized_hash =
      hash >= dynamic_csr_pending_hash ? dynamic_csr_pending_hash - 1 : hash;
    for (cuda::std::uint32_t step = 0; step < capacity; ++step) {
      auto const slot = (static_cast<cuda::std::uint32_t>(normalized_hash) + step) & capacity_mask;
      auto entry_hash =
        cuda::atomic_ref<hash_value_type, cuda::thread_scope_device>{entries[slot].hash};
      auto current = entry_hash.load(cuda::memory_order_acquire);
      while (current == dynamic_csr_pending_hash) {
        current = entry_hash.load(cuda::memory_order_acquire);
      }
      if (current == dynamic_csr_empty_hash) { return capacity; }
      auto const stored_representative =
        cuda::atomic_ref<cuda::std::uint64_t, cuda::thread_scope_device>{
          entries[slot].representative}
          .load(cuda::memory_order_acquire);
      if (current == normalized_hash && equal(probe, stored_representative)) { return slot; }
    }
    return capacity;
  }

  /**
   * @brief Warp-cooperative lookup over consecutive linear-probe windows.
   *
   * A tile loads a 32-entry window coalesced, then uses ballots to select the first matching
   * hash/equality pair or the first empty entry. Full row equality still resolves same-hash
   * collision chains; the hash is only a fast rejection filter.
   */
  template <typename Equal>
  __device__ size_type find_cooperative(hash_value_type hash,
                                        cuda::std::uint64_t probe,
                                        Equal equal,
                                        cooperative_groups::thread_block_tile<32> const& tile) const
  {
    auto const normalized_hash =
      hash >= dynamic_csr_pending_hash ? dynamic_csr_pending_hash - 1 : hash;
    auto const lane = static_cast<cuda::std::uint32_t>(tile.thread_rank());
    for (cuda::std::uint32_t base = 0; base < capacity; base += 32) {
      auto const step  = base + lane;
      auto const index = (static_cast<cuda::std::uint32_t>(normalized_hash) + step) & capacity_mask;
      auto entry_hash =
        cuda::atomic_ref<hash_value_type, cuda::thread_scope_device>{entries[index].hash};
      auto current = entry_hash.load(cuda::memory_order_acquire);
      while (current == dynamic_csr_pending_hash) {
        current = entry_hash.load(cuda::memory_order_acquire);
      }
      auto const empty = current == dynamic_csr_empty_hash;
      auto const stored_representative =
        empty ? cuda::std::uint64_t{0}
              : cuda::atomic_ref<cuda::std::uint64_t, cuda::thread_scope_device>{entries[index]
                                                                                   .representative}
                  .load(cuda::memory_order_acquire);
      auto const matches    = current == normalized_hash && equal(probe, stored_representative);
      auto const match_mask = tile.ballot(matches);
      auto const empty_mask = tile.ballot(empty);
      auto const event_mask = match_mask | empty_mask;
      if (event_mask != 0) {
        auto const first = static_cast<cuda::std::uint32_t>(__ffs(event_mask) - 1);
        return (match_mask & (cuda::std::uint32_t{1} << first)) != 0
                 ? static_cast<size_type>(
                     (static_cast<cuda::std::uint32_t>(normalized_hash) + base + first) &
                     capacity_mask)
                 : capacity;
      }
    }
    return capacity;
  }
};

inline void throw_if_dynamic_key_map_exhausted(int const* exhausted_flag, cuda::stream_ref stream)
{
  int exhausted{};
  cudf::detail::cuda_memcpy_async(
    cudf::host_span<int>{&exhausted, 1}, cudf::device_span<int const>{exhausted_flag, 1}, stream);
  stream.sync();
  CUDF_EXPECTS(exhausted == 0,
               "streaming_hash_join segmented dynamic key map exhausted its capacity",
               std::overflow_error);
}

/**
 * @brief Device view of an append-only, segmented CSR.
 *
 * `bucket` is intentionally a hash bucket rather than a distinct-key slot. A 32-bit row hash is
 * not sufficient to establish row equality; the streaming join's existing comparator remains the
 * authority for that. This allows the representation to be built while preserving collision
 * correctness and makes the bucket chains available to a subsequent comparator-aware probe
 * kernel.
 */
struct segmented_dynamic_csr_ref {
  dynamic_csr_slot const* slots;
  dynamic_csr_segment const* segments;
  cuda::std::uint64_t const* values;

  [[nodiscard]] __device__ size_type size(size_type bucket) const { return slots[bucket].size; }

  /** @brief Return the build-row reference at a slot-local ordinal. */
  [[nodiscard]] __device__ cuda::std::uint64_t at(size_type bucket, size_type ordinal) const
  {
    auto segment = slots[bucket].head;
    while (segment != dynamic_csr_no_segment) {
      auto const descriptor = segments[segment];
      if (ordinal < descriptor.length) { return values[descriptor.begin + ordinal]; }
      ordinal -= descriptor.length;
      segment = descriptor.next;
    }
    return cuda::std::uint64_t{0};  // Unreachable when ordinal < size(bucket).
  }
};

/**
 * @brief Materialize packed build references for already-resolved segmented slots.
 *
 * The persistent-map probe produces `slots` and per-probe output offsets; this kernel deliberately
 * keeps descriptor traversal separate from key comparison so hash collisions are resolved before
 * a slot reaches it.
 */
CUDF_KERNEL void segmented_dynamic_csr_retrieve_kernel(size_type output_size,
                                                       size_type num_probe_rows,
                                                       size_type const* probe_slots,
                                                       cuda::std::int64_t const* offsets,
                                                       segmented_dynamic_csr_ref csr,
                                                       cuda::std::uint64_t* row_refs,
                                                       size_type* left_indices)
{
  for (auto output = grid_1d::global_thread_id(); output < output_size;
       output += grid_1d::grid_stride()) {
    // Find the final offset no greater than `output`. A linear search here makes a distinct-key
    // join quadratic; the binary search preserves the CSR layout with logarithmic lookup work.
    auto lower = size_type{0};
    auto upper = num_probe_rows;
    while (lower + 1 < upper) {
      auto const middle = lower + (upper - lower) / 2;
      if (offsets[middle] <= static_cast<cuda::std::int64_t>(output)) {
        lower = middle;
      } else {
        upper = middle;
      }
    }
    auto const probe     = lower;
    left_indices[output] = probe;
    row_refs[output] = csr.at(probe_slots[probe], static_cast<size_type>(output - offsets[probe]));
  }
}

CUDF_KERNEL void unpack_segmented_dynamic_refs_kernel(size_type output_size,
                                                      cuda::std::uint64_t const* row_refs,
                                                      size_type* batch_indices,
                                                      size_type* row_indices)
{
  for (auto output = grid_1d::global_thread_id(); output < output_size;
       output += grid_1d::grid_stride()) {
    auto const ref        = row_refs[output];
    batch_indices[output] = static_cast<size_type>(ref >> 32);
    row_indices[output]   = static_cast<size_type>(ref);
  }
}

template <typename Hasher, typename Equal>
CUDF_KERNEL void segmented_dynamic_csr_probe_count_kernel(size_type num_rows,
                                                          size_type* probe_slots,
                                                          size_type* match_counts,
                                                          segmented_dynamic_csr_ref csr,
                                                          dynamic_csr_key_map_ref key_map,
                                                          Hasher hasher,
                                                          Equal equal)
{
  auto const block = cooperative_groups::this_thread_block();
  auto const tile  = cooperative_groups::tiled_partition<32>(block);
  auto row         = static_cast<size_type>(
    (static_cast<cuda::std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / 32);
  auto const stride =
    static_cast<size_type>((static_cast<cuda::std::uint64_t>(gridDim.x) * blockDim.x) / 32);
  for (; row < num_rows; row += stride) {
    auto const index = static_cast<size_type>(row);
    // The probe reference carries its local row in the low word; equality's left table is the
    // probe table, while the map representative selects the appropriate stored build batch.
    auto const slot =
      key_map.find_cooperative(hasher(index), static_cast<cuda::std::uint64_t>(index), equal, tile);
    if (tile.thread_rank() == 0) {
      probe_slots[index]  = slot == key_map.capacity ? JoinNoMatch : slot;
      match_counts[index] = slot == key_map.capacity ? size_type{0} : csr.size(slot);
    }
  }
}

template <typename Hasher, typename Equal>
void launch_segmented_dynamic_csr_probe_count_kernel(size_type num_rows,
                                                     size_type* probe_slots,
                                                     size_type* match_counts,
                                                     segmented_dynamic_csr_ref csr,
                                                     dynamic_csr_key_map_ref key_map,
                                                     Hasher hasher,
                                                     Equal equal,
                                                     cuda::stream_ref stream)
{
  if (num_rows == 0) { return; }
  constexpr auto threads = 256;
  auto const blocks      = static_cast<unsigned int>(
    (static_cast<cuda::std::uint64_t>(num_rows) * 32 + threads - 1) / threads);
  segmented_dynamic_csr_probe_count_kernel<<<blocks, threads, 0, stream.get()>>>(
    num_rows, probe_slots, match_counts, csr, key_map, hasher, equal);
  CUDF_CUDA_TRY(cudaGetLastError());
}

template <typename Hasher, typename Equal>
CUDF_KERNEL void resolve_dynamic_csr_slots_kernel(size_type num_rows,
                                                  bitmask_type const* valid_rows,
                                                  size_type batch_id,
                                                  cuda::std::uint32_t* row_slots,
                                                  cuda::std::uint64_t* row_refs,
                                                  int* key_map_exhausted,
                                                  Hasher hasher,
                                                  dynamic_csr_key_map_ref key_map,
                                                  Equal equal)
{
  auto const stride = grid_1d::grid_stride();
  for (auto row = grid_1d::global_thread_id(); row < num_rows; row += stride) {
    auto const index = static_cast<size_type>(row);
    if (valid_rows != nullptr && !cudf::bit_is_set(valid_rows, index)) {
      row_slots[index] = dynamic_csr_invalid_slot;
      row_refs[index]  = cuda::std::uint64_t{0};
      continue;
    }
    auto const row_ref =
      (static_cast<cuda::std::uint64_t>(static_cast<cuda::std::uint32_t>(batch_id)) << 32) |
      static_cast<cuda::std::uint32_t>(index);
    auto const bucket = key_map.find_or_insert(hasher(index), row_ref, equal);
    if (bucket == key_map.capacity) {
      // Do not silently lose a build row if a future sizing policy violates the map-capacity
      // invariant. The host reports this at the next externally visible operation.
      cuda::atomic_ref<int, cuda::thread_scope_device>{*key_map_exhausted}.store(
        1, cuda::memory_order_relaxed);
      row_slots[index] = dynamic_csr_invalid_slot;
      row_refs[index]  = cuda::std::uint64_t{0};
      continue;
    }
    row_slots[index] = static_cast<cuda::std::uint32_t>(bucket);
    row_refs[index]  = row_ref;
  }
}

/**
 * @brief Append a run of sorted row references as fixed-size pages.
 *
 * Runs are created after the map lookup, so a page consumes only its actual row references.
 * This is what keeps fixed 32-row pages within the configured row capacity even when every key
 * is distinct: the total number of descriptors is at most the number of valid input rows.
 */
CUDF_KERNEL void append_dynamic_csr_runs_kernel(size_type max_runs,
                                                cuda::std::uint32_t const* unique_slots,
                                                size_type const* run_ends,
                                                int const* num_runs,
                                                size_type base_row,
                                                cuda::std::uint64_t const* sorted_refs,
                                                cuda::std::uint64_t* values,
                                                dynamic_csr_slot* slots,
                                                dynamic_csr_segment* segments,
                                                cuda::std::uint32_t* next_segment)
{
  constexpr auto page_size = size_type{32};
  for (auto run = grid_1d::global_thread_id(); run < max_runs; run += grid_1d::grid_stride()) {
    auto const run_index         = static_cast<size_type>(run);
    values[base_row + run_index] = sorted_refs[run_index];
    if (run_index >= static_cast<size_type>(*num_runs)) { continue; }
    auto const bucket = unique_slots[run_index];
    // Invalid (null-unequal or failed-map) rows were sorted last and are not part of the CSR.
    if (bucket == dynamic_csr_invalid_slot) { continue; }
    auto const run_begin  = run_index == 0 ? size_type{0} : run_ends[run_index - 1];
    auto const run_length = run_ends[run_index] - run_begin;
    auto const page_count = (run_length + page_size - 1) / page_size;
    auto const first_page =
      cuda::atomic_ref<cuda::std::uint32_t, cuda::thread_scope_device>{*next_segment}.fetch_add(
        static_cast<cuda::std::uint32_t>(page_count), cuda::memory_order_relaxed);

    // Initialize the complete local chain before publishing its first page at the bucket tail.
    for (auto page = size_type{0}; page < page_count; ++page) {
      auto const page_id = first_page + static_cast<cuda::std::uint32_t>(page);
      auto const length  = cuda::std::min(page_size, run_length - page * page_size);
      segments[page_id]  = {page + 1 == page_count ? dynamic_csr_no_segment : page_id + 1,
                           base_row + run_begin + page * page_size,
                           length};
    }
    __threadfence();

    // Publish the fully initialized payload before exposing this page at the tail. Inserts may be
    // issued from independent streams, so the link operation must also establish publication.
    auto const last_page = first_page + static_cast<cuda::std::uint32_t>(page_count - 1);
    auto const old_tail =
      cuda::atomic_ref<cuda::std::uint32_t, cuda::thread_scope_device>{slots[bucket].tail}.exchange(
        last_page, cuda::memory_order_acq_rel);
    if (old_tail == dynamic_csr_no_segment) {
      cuda::atomic_ref<cuda::std::uint32_t, cuda::thread_scope_device>{slots[bucket].head}.store(
        first_page, cuda::memory_order_release);
    } else {
      cuda::atomic_ref<cuda::std::uint32_t, cuda::thread_scope_device>{segments[old_tail].next}
        .store(first_page, cuda::memory_order_release);
    }
    cuda::atomic_ref<size_type, cuda::thread_scope_device>{slots[bucket].size}.fetch_add(
      run_length, cuda::memory_order_relaxed);
  }
}

CUDF_KERNEL void initialize_dynamic_csr_slots_kernel(dynamic_csr_slot* slots, size_type num_buckets)
{
  for (auto i = grid_1d::global_thread_id(); i < num_buckets; i += grid_1d::grid_stride()) {
    slots[i] = {dynamic_csr_no_segment, dynamic_csr_no_segment, size_type{0}};
  }
}

/**
 * @brief Fixed-capacity, append-only backing store for segmented dynamic CSR experiments.
 */
class segmented_dynamic_csr {
 public:
  segmented_dynamic_csr(size_type max_rows,
                        size_type num_buckets,
                        cuda::stream_ref stream,
                        rmm::device_async_resource_ref mr)
    : _num_buckets{num_buckets},
      _mr{mr},
      _slots{static_cast<std::size_t>(num_buckets), stream, mr},
      _segments{static_cast<std::size_t>(max_rows), stream, mr},
      _values{static_cast<std::size_t>(max_rows), stream, mr},
      _next_segment{1, stream, mr},
      _key_map_exhausted{1, stream, mr}
  {
    CUDF_CUDA_TRY(
      cudaMemsetAsync(_slots.data(), 0xff, _slots.size() * sizeof(dynamic_csr_slot), stream.get()));
    CUDF_CUDA_TRY(cudaMemsetAsync(_key_map_exhausted.data(), 0, sizeof(int), stream.get()));
    CUDF_CUDA_TRY(
      cudaMemsetAsync(_next_segment.data(), 0, sizeof(cuda::std::uint32_t), stream.get()));
    // The size field is overwritten by memset above, so initialize the complete slot state in a
    // tiny kernel-free host staging copy is not possible. The first append resets its slot size;
    // the constructor therefore uses a dedicated initialization kernel below.
    initialize(stream);
  }

  template <typename Hasher, typename Equal>
  void append(size_type num_rows,
              bitmask_type const* valid_rows,
              size_type batch_id,
              size_type base_row,
              Hasher hasher,
              dynamic_csr_key_map_ref key_map,
              Equal equal,
              cuda::stream_ref stream)
  {
    if (num_rows == 0) { return; }
    auto const threads = 256;
    auto const blocks  = static_cast<unsigned int>((num_rows + threads - 1) / threads);
    auto row_slots =
      rmm::device_uvector<cuda::std::uint32_t>{static_cast<std::size_t>(num_rows), stream, _mr};
    auto row_refs =
      rmm::device_uvector<cuda::std::uint64_t>{static_cast<std::size_t>(num_rows), stream, _mr};
    resolve_dynamic_csr_slots_kernel<<<blocks, threads, 0, stream.get()>>>(
      num_rows,
      valid_rows,
      batch_id,
      row_slots.data(),
      row_refs.data(),
      _key_map_exhausted.data(),
      hasher,
      key_map,
      equal);
    CUDF_CUDA_TRY(cudaGetLastError());

    auto sorted_slots =
      rmm::device_uvector<cuda::std::uint32_t>{static_cast<std::size_t>(num_rows), stream, _mr};
    auto sorted_refs =
      rmm::device_uvector<cuda::std::uint64_t>{static_cast<std::size_t>(num_rows), stream, _mr};
    std::size_t sort_bytes{};
    CUDF_CUDA_TRY(cub::DeviceRadixSort::SortPairs(nullptr,
                                                  sort_bytes,
                                                  row_slots.data(),
                                                  sorted_slots.data(),
                                                  row_refs.data(),
                                                  sorted_refs.data(),
                                                  num_rows,
                                                  0,
                                                  sizeof(cuda::std::uint32_t) * CHAR_BIT,
                                                  stream.get()));
    auto sort_temp = rmm::device_buffer{sort_bytes, stream, _mr};
    CUDF_CUDA_TRY(cub::DeviceRadixSort::SortPairs(sort_temp.data(),
                                                  sort_bytes,
                                                  row_slots.data(),
                                                  sorted_slots.data(),
                                                  row_refs.data(),
                                                  sorted_refs.data(),
                                                  num_rows,
                                                  0,
                                                  sizeof(cuda::std::uint32_t) * CHAR_BIT,
                                                  stream.get()));
    auto unique_slots =
      rmm::device_uvector<cuda::std::uint32_t>{static_cast<std::size_t>(num_rows), stream, _mr};
    auto run_lengths =
      rmm::device_uvector<size_type>{static_cast<std::size_t>(num_rows), stream, _mr};
    auto num_runs = rmm::device_uvector<int>{1, stream, _mr};
    std::size_t encode_bytes{};
    CUDF_CUDA_TRY(cub::DeviceRunLengthEncode::Encode(nullptr,
                                                     encode_bytes,
                                                     sorted_slots.data(),
                                                     unique_slots.data(),
                                                     run_lengths.data(),
                                                     num_runs.data(),
                                                     num_rows,
                                                     stream.get()));
    auto encode_temp = rmm::device_buffer{encode_bytes, stream, _mr};
    CUDF_CUDA_TRY(cub::DeviceRunLengthEncode::Encode(encode_temp.data(),
                                                     encode_bytes,
                                                     sorted_slots.data(),
                                                     unique_slots.data(),
                                                     run_lengths.data(),
                                                     num_runs.data(),
                                                     num_rows,
                                                     stream.get()));
    auto run_ends = rmm::device_uvector<size_type>{static_cast<std::size_t>(num_rows), stream, _mr};
    std::size_t scan_bytes{};
    CUDF_CUDA_TRY(cub::DeviceScan::InclusiveSum(
      nullptr, scan_bytes, run_lengths.data(), run_ends.data(), num_rows, stream.get()));
    auto scan_temp = rmm::device_buffer{scan_bytes, stream, _mr};
    CUDF_CUDA_TRY(cub::DeviceScan::InclusiveSum(
      scan_temp.data(), scan_bytes, run_lengths.data(), run_ends.data(), num_rows, stream.get()));
    append_dynamic_csr_runs_kernel<<<blocks, threads, 0, stream.get()>>>(num_rows,
                                                                         unique_slots.data(),
                                                                         run_ends.data(),
                                                                         num_runs.data(),
                                                                         base_row,
                                                                         sorted_refs.data(),
                                                                         _values.data(),
                                                                         _slots.data(),
                                                                         _segments.data(),
                                                                         _next_segment.data());
    CUDF_CUDA_TRY(cudaGetLastError());
  }

  [[nodiscard]] segmented_dynamic_csr_ref ref() const
  {
    return {_slots.data(), _segments.data(), _values.data()};
  }

  [[nodiscard]] size_type num_buckets() const { return _num_buckets; }

  void throw_if_key_map_exhausted(cuda::stream_ref stream) const
  {
    throw_if_dynamic_key_map_exhausted(_key_map_exhausted.data(), stream);
  }

 private:
  void initialize(cuda::stream_ref stream)
  {
    auto const threads = 256;
    auto const blocks  = static_cast<unsigned int>((_num_buckets + threads - 1) / threads);
    initialize_dynamic_csr_slots_kernel<<<blocks, threads, 0, stream.get()>>>(_slots.data(),
                                                                              _num_buckets);
    CUDF_CUDA_TRY(cudaGetLastError());
  }

  size_type _num_buckets;
  rmm::device_async_resource_ref _mr;
  rmm::device_uvector<dynamic_csr_slot> _slots;
  rmm::device_uvector<dynamic_csr_segment> _segments;
  rmm::device_uvector<cuda::std::uint64_t> _values;
  rmm::device_uvector<cuda::std::uint32_t> _next_segment;
  rmm::device_uvector<int> _key_map_exhausted;
};

}  // namespace cudf::detail
