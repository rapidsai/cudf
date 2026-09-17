/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/types.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/atomic>
#include <cuda/std/cstdint>
#include <cuda/stream>

#include <limits>

namespace cudf::detail {

/**
 * @brief A single append-only page in a dynamic CSR bucket.
 *
 * The first implementation deliberately uses one row per page. It is a correctness-oriented
 * baseline for the linked-page representation: a later grouped append kernel can replace this
 * with geometrically-sized pages without changing the device lookup contract.
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

constexpr auto dynamic_csr_no_segment = std::numeric_limits<cuda::std::uint32_t>::max();
constexpr auto dynamic_csr_empty_hash = std::numeric_limits<hash_value_type>::max();
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
  size_type capacity;

  template <typename Equal>
  __device__ size_type find_or_insert(hash_value_type hash,
                                      cuda::std::uint64_t representative,
                                      Equal equal) const
  {
    // Reserve empty and pending state values. Mapping the two largest hashes to the same usable
    // hash is safe because equality remains the authority within a probe sequence.
    auto const normalized_hash =
      hash >= dynamic_csr_pending_hash ? dynamic_csr_pending_hash - 1 : hash;
    for (size_type step = 0; step < capacity; ++step) {
      // `size_type` is signed. Do the modulo in an unsigned wide type before converting so row
      // hashes with the high bit set cannot produce a negative, out-of-bounds slot index.
      auto const slot = static_cast<size_type>((static_cast<cuda::std::uint64_t>(normalized_hash) +
                                                static_cast<cuda::std::uint64_t>(step)) %
                                               static_cast<cuda::std::uint64_t>(capacity));
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
    for (size_type step = 0; step < capacity; ++step) {
      auto const slot = static_cast<size_type>((static_cast<cuda::std::uint64_t>(normalized_hash) +
                                                static_cast<cuda::std::uint64_t>(step)) %
                                               static_cast<cuda::std::uint64_t>(capacity));
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
};

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
  for (auto row = grid_1d::global_thread_id(); row < num_rows; row += grid_1d::grid_stride()) {
    auto const index = static_cast<size_type>(row);
    // The probe reference carries its local row in the low word; equality's left table is the
    // probe table, while the map representative selects the appropriate stored build batch.
    auto const slot = key_map.find(hasher(index), static_cast<cuda::std::uint64_t>(index), equal);
    probe_slots[index]  = slot == key_map.capacity ? JoinNoMatch : slot;
    match_counts[index] = slot == key_map.capacity ? size_type{0} : csr.size(slot);
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
  auto const blocks      = static_cast<unsigned int>((num_rows + threads - 1) / threads);
  segmented_dynamic_csr_probe_count_kernel<<<blocks, threads, 0, stream.get()>>>(
    num_rows, probe_slots, match_counts, csr, key_map, hasher, equal);
  CUDF_CUDA_TRY(cudaGetLastError());
}

template <typename Hasher, typename Equal>
CUDF_KERNEL void append_dynamic_csr_rows_kernel(size_type num_rows,
                                                bitmask_type const* valid_rows,
                                                size_type batch_id,
                                                size_type base_row,
                                                size_type num_buckets,
                                                dynamic_csr_slot* slots,
                                                dynamic_csr_segment* segments,
                                                cuda::std::uint64_t* values,
                                                Hasher hasher,
                                                dynamic_csr_key_map_ref key_map,
                                                Equal equal)
{
  auto const stride = grid_1d::grid_stride();
  for (auto row = grid_1d::global_thread_id(); row < num_rows; row += stride) {
    auto const index = static_cast<size_type>(row);
    if (valid_rows != nullptr && !cudf::bit_is_set(valid_rows, index)) { continue; }

    auto const segment_id = static_cast<cuda::std::uint32_t>(base_row + index);
    auto const row_ref =
      (static_cast<cuda::std::uint64_t>(static_cast<cuda::std::uint32_t>(batch_id)) << 32) |
      static_cast<cuda::std::uint32_t>(index);
    auto const bucket = key_map.find_or_insert(hasher(index), row_ref, equal);
    if (bucket == num_buckets) { continue; }
    values[base_row + index] = row_ref;
    segments[segment_id]     = {dynamic_csr_no_segment, base_row + index, size_type{1}};

    // Publish the fully initialized payload before exposing this page at the tail. Inserts may be
    // issued from independent streams, so the link operation must also establish publication.
    auto const old_tail =
      cuda::atomic_ref<cuda::std::uint32_t, cuda::thread_scope_device>{slots[bucket].tail}.exchange(
        segment_id, cuda::memory_order_acq_rel);
    if (old_tail == dynamic_csr_no_segment) {
      cuda::atomic_ref<cuda::std::uint32_t, cuda::thread_scope_device>{slots[bucket].head}.store(
        segment_id, cuda::memory_order_release);
    } else {
      cuda::atomic_ref<cuda::std::uint32_t, cuda::thread_scope_device>{segments[old_tail].next}
        .store(segment_id, cuda::memory_order_release);
    }
    cuda::atomic_ref<size_type, cuda::thread_scope_device>{slots[bucket].size}.fetch_add(
      size_type{1}, cuda::memory_order_relaxed);
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
      _slots{static_cast<std::size_t>(num_buckets), stream, mr},
      _segments{static_cast<std::size_t>(max_rows), stream, mr},
      _values{static_cast<std::size_t>(max_rows), stream, mr}
  {
    CUDF_CUDA_TRY(
      cudaMemsetAsync(_slots.data(), 0xff, _slots.size() * sizeof(dynamic_csr_slot), stream.get()));
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
    append_dynamic_csr_rows_kernel<<<blocks, threads, 0, stream.get()>>>(num_rows,
                                                                         valid_rows,
                                                                         batch_id,
                                                                         base_row,
                                                                         _num_buckets,
                                                                         _slots.data(),
                                                                         _segments.data(),
                                                                         _values.data(),
                                                                         hasher,
                                                                         key_map,
                                                                         equal);
    CUDF_CUDA_TRY(cudaGetLastError());
  }

  [[nodiscard]] segmented_dynamic_csr_ref ref() const
  {
    return {_slots.data(), _segments.data(), _values.data()};
  }

  [[nodiscard]] size_type num_buckets() const { return _num_buckets; }

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
  rmm::device_uvector<dynamic_csr_slot> _slots;
  rmm::device_uvector<dynamic_csr_segment> _segments;
  rmm::device_uvector<cuda::std::uint64_t> _values;
};

}  // namespace cudf::detail
