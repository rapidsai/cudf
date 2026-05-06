/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "groupby/common/utils.hpp"
#include "groupby/hash/helpers.cuh"

#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/detail/row_operator/hashing.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/groupby.hpp>
#include <cudf/hashing/detail/default_hash.cuh>
#include <cudf/table/table.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <cuco/static_set.cuh>
#include <cuda/atomic>
#include <cuda/std/functional>
#include <cuda/std/utility>

#include <functional>
#include <memory>
#include <vector>

namespace cudf::groupby {

/*
 * Companion location for a stored dense ID: which compacted batch table the key
 * lives in (`first`) and the row index within that table (`second`).  Packed into
 * one 8-byte pair so the comparator does a single load instead of two.
 */
using key_location_t = cuda::std::pair<size_type, size_type>;

using streaming_probing_scheme_t =
  cuco::linear_probing<detail::hash::GROUPBY_CG_SIZE,
                       cudf::hashing::detail::default_hash<size_type>>;

using streaming_set_t = cuco::static_set<cudf::size_type,
                                         cuco::extent<int64_t>,
                                         cuda::thread_scope_device,
                                         cuda::std::equal_to<size_type>,
                                         streaming_probing_scheme_t,
                                         rmm::mr::polymorphic_allocator<char>,
                                         cuco::storage<detail::hash::GROUPBY_BUCKET_SIZE>>;

/*
 * N-table comparator for the persistent hash set.
 *
 * Slot values < max_groups are "stored" dense IDs resolved via the companion
 * vector key_loc[id] = {batch_id, row_within_batch} to a (compacted_batch_table,
 * row) location.  Slot values >= max_groups are transient batch values: row index
 * = value - max_groups in the current batch table.  The transient encoding lives
 * only for the duration of one probe_and_insert call; new keys are rewritten to
 * dense IDs before the next batch's insertion.
 *
 * Cross-comparators are pre-built as device_row_comparator(batch, compacted[k])
 * and stored in a device array.  Self-comparisons use batch_self_eq.
 */
template <typename RowEqT>
struct n_table_comparator {
  RowEqT batch_self_eq;           ///< Self-comparator on the current batch table
  RowEqT const* cross_eqs;        ///< Device array [num_compacted_batches]: batch vs compacted[k]
  key_location_t const* key_loc;  ///< {batch_id, row_in_compacted} per dense ID
  size_type max_groups;           ///< Threshold: idx >= max_groups is a transient batch value

  __device__ bool operator()(size_type lhs, size_type rhs) const noexcept
  {
    bool const lhs_is_batch = (lhs >= max_groups);
    bool const rhs_is_batch = (rhs >= max_groups);

    if (lhs_is_batch && rhs_is_batch) { return batch_self_eq(lhs - max_groups, rhs - max_groups); }
    if (lhs_is_batch) {
      auto const loc = key_loc[rhs];
      return cross_eqs[loc.first](lhs - max_groups, loc.second);
    }
    if (rhs_is_batch) {
      auto const loc = key_loc[lhs];
      return cross_eqs[loc.first](rhs - max_groups, loc.second);
    }
    return lhs == rhs;
  }
};

/*
 * insert_and_find on the main set for each batch row.  Returns the resident slot's
 * value (consumed by `thrust::transform` as `target_indices[batch_idx]`) and writes
 * two side-output arrays:
 *   inserted_flags[batch_idx] = whether this row won the CAS (used as the stencil
 *                               for the subsequent count + copy_if of winners)
 *   slot_offsets[batch_idx]   = offset of the resident slot from `base`, or
 *                               CUDF_SIZE_TYPE_SENTINEL for null/excluded rows
 *
 * If the batch produces no new keys, target_indices is already final and no further
 * pass is required.  If new keys exist, Pass 2 rewrites the affected slots and the
 * caller re-reads target_indices via slot_offsets in a single transform.
 */
template <typename SetRef>
struct insert_and_map_fn {
  mutable SetRef set_ref;
  bitmask_type const* row_bitmask;
  size_type max_groups;
  size_type const* base;
  bool* inserted_flags;
  size_type* slot_offsets;

  __device__ size_type operator()(size_type batch_idx) const
  {
    if (row_bitmask && !cudf::bit_is_set(row_bitmask, batch_idx)) {
      inserted_flags[batch_idx] = false;
      slot_offsets[batch_idx]   = cudf::detail::CUDF_SIZE_TYPE_SENTINEL;
      return cudf::detail::CUDF_SIZE_TYPE_SENTINEL;
    }
    auto const [iter, inserted] = set_ref.insert_and_find(max_groups + batch_idx);
    inserted_flags[batch_idx]   = inserted;
    slot_offsets[batch_idx]     = static_cast<size_type>(iter - base);
    return *iter;
  }
};

/*
 * Per-row hash producer used to populate the precomputed batch hash cache.
 * Returns 0 (a dummy value never read) for rows excluded by the null bitmask
 * under `null_policy::EXCLUDE`, avoiding wasted hash work for rows that won't
 * probe the set.  Cuco only consults the cache for rows whose `insert_and_map_fn`
 * actually called `insert_and_find`, so the dummy is invisible to it.
 */
template <typename RowHasher>
struct conditional_hash_fn {
  RowHasher row_hasher;
  bitmask_type const* row_bitmask;

  __device__ hash_value_type operator()(size_type i) const noexcept
  {
    if (row_bitmask && !cudf::bit_is_set(row_bitmask, i)) { return hash_value_type{0}; }
    return row_hasher(i);
  }
};

/*
 * Hasher backed by a precomputed cache, indexed by `idx - offset`.
 * In streaming_groupby `offset = max_groups`, so transient batch values
 * `max_groups + batch_idx` resolve to `cache[batch_idx]`.  Caching is faster
 * than inlining the row_hasher inside cuco's probe at high cardinality
 * (measured: ~10% throughput drop without the cache) — the row_hasher's
 * dispatch + nullable check inflates the cuco insert kernel.
 */
struct offset_cache_hasher {
  hash_value_type const* cache;
  size_type offset;
  __device__ hash_value_type operator()(size_type idx) const noexcept
  {
    return cache[idx - offset];
  }
};

/*
 * For each newly discovered key (dense rank `r` within this batch):
 *   1. Rewrite its slot from transient encoding to its dense ID.  Pass 2 runs
 *      after Pass 1's kernel completes and each slot has a single writer, so a
 *      relaxed atomic store is sufficient (no CAS needed).
 *   2. Write the (batch_id, row) pair to the companion vector at the dense ID.
 */
struct finalize_new_key_fn {
  size_type const*
    batch_local_indices;            ///< batch-local row indices of new keys [new_distinct_count]
  size_type* base;                  ///< hash set storage base
  size_type const* slot_offsets;    ///< slot offset per batch row [batch_size]
  key_location_t* key_loc;          ///< {batch_id, row_in_compacted} per dense ID
  size_type batch_id;               ///< the index of this batch in _compacted_batches
  size_type distinct_count_before;  ///< _distinct_count prior to this batch

  __device__ void operator()(size_type r) const
  {
    auto const dense_id    = distinct_count_before + r;
    auto const batch_local = batch_local_indices[r];

    cuda::atomic_ref<size_type, cuda::thread_scope_device>{*(base + slot_offsets[batch_local])}
      .store(dense_id, cuda::std::memory_order_relaxed);

    key_loc[dense_id] = key_location_t{batch_id, r};
  }
};

/*
 * Read the dense ID at each batch row's resident slot, post-Pass-2.
 * Returns CUDF_SIZE_TYPE_SENTINEL for null/excluded rows (slot_offsets sentinel).
 */
struct read_target_indices_fn {
  size_type const* base;
  size_type const* slot_offsets;

  __device__ size_type operator()(size_type i) const
  {
    auto const off = slot_offsets[i];
    if (off == cudf::detail::CUDF_SIZE_TYPE_SENTINEL) {
      return cudf::detail::CUDF_SIZE_TYPE_SENTINEL;
    }
    return *(base + off);
  }
};

template <bool has_nested_columns>
auto build_cross_comparators(
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> const& preprocessed_batch,
  std::vector<std::shared_ptr<cudf::detail::row::equality::preprocessed_table>> const&
    preprocessed_batches,
  cudf::nullate::DYNAMIC has_null,
  rmm::cuda_stream_view stream)
{
  using eq_t = cudf::detail::row::equality::device_row_comparator<
    has_nested_columns,
    cudf::nullate::DYNAMIC,
    cudf::detail::row::equality::nan_equal_physical_equality_comparator>;

  auto const n       = static_cast<size_type>(preprocessed_batches.size());
  auto const temp_mr = cudf::get_current_device_resource_ref();

  std::vector<eq_t> h_eqs;
  h_eqs.reserve(n);
  for (size_type k = 0; k < n; ++k) {
    auto const cross_cmp = cudf::detail::row::equality::two_table_comparator{
      preprocessed_batch, preprocessed_batches[k]};
    auto const adapter = cross_cmp.equal_to<has_nested_columns>(has_null, null_equality::EQUAL);
    h_eqs.push_back(adapter.comparator);
  }

  return cudf::detail::make_device_uvector_async(h_eqs, stream, temp_mr);
}

/// The impl struct for streaming_groupby. Defined in impl.cu.
struct streaming_groupby::impl {
  std::vector<size_type> _key_indices;
  std::vector<streaming_aggregation_request> _requests_clone;
  size_type _max_groups;
  null_policy _null_handling;

  bool _initialized{false};
  /*
   * Number of distinct keys accumulated so far.  Also serves as the high-water
   * mark of dense IDs in the persistent hash set: stored slot values are in
   * [0, _distinct_count).
   */
  size_type _distinct_count{0};
  bool _has_nullable_keys{false};
  bool _has_nested_keys{false};

  // -- Compacted batch storage --
  std::vector<std::unique_ptr<table>> _compacted_batches;
  std::vector<std::shared_ptr<cudf::detail::row::equality::preprocessed_table>>
    _preprocessed_batches;

  /// Companion vector indexed by dense ID, sized to max_groups.
  /// Each entry is {batch_id, row_in_compacted_batch}.
  std::unique_ptr<rmm::device_uvector<key_location_t>> _key_loc;

  std::vector<size_type> _request_first_agg_offset;
  std::vector<aggregation::Kind> _agg_kinds;
  std::vector<std::unique_ptr<aggregation>> _agg_objects;
  std::vector<int8_t> _is_agg_intermediate;
  bool _has_compound_aggs{false};

  /*
   * Aggregation results table, pre-allocated to max_groups rows.
   * Indexed by dense ID (== row index).
   */
  std::unique_ptr<table> _agg_results;
  /*
   * Cached mutable_table_device_view of `_agg_results`.  `_agg_results` is allocated
   * once at initialize() and never resized, so this device-side descriptor can be
   * built once and reused on every aggregate() / merge() call rather than rebuilt
   * (which requires a host-to-device copy of the column metadata).
   * Uses std::function as the type-erased deleter because the natural deleter
   * returned by `create` is an unnamed lambda type.
   */
  std::unique_ptr<mutable_table_device_view, std::function<void(mutable_table_device_view*)>>
    _d_agg_results;
  std::vector<size_type> _value_col_indices;
  rmm::device_uvector<aggregation::Kind> _d_agg_kinds;

  std::unique_ptr<streaming_set_t> _key_set;

  [[nodiscard]] size_type num_keys() const { return static_cast<size_type>(_key_indices.size()); }
  [[nodiscard]] bool has_state() const { return _initialized && _distinct_count > 0; }

  impl(host_span<size_type const> key_indices,
       host_span<streaming_aggregation_request const> requests,
       size_type max_groups,
       null_policy null_handling);

  void initialize(table_view const& data, rmm::cuda_stream_view stream);
  void create_key_set(rmm::cuda_stream_view stream);
  void update_nullable_state(table_view const& batch_keys);

  struct batch_insert_result {
    rmm::device_uvector<size_type> target_indices;
    size_type new_insertions;
    rmm::device_buffer bitmask_buffer;
  };

  batch_insert_result probe_and_insert(table_view const& batch_keys, rmm::cuda_stream_view stream);

  /*
   * Template implementation of probe_and_insert, split by has_nested.
   * Defined in insert.cuh, instantiated in insert.cu and insert_nested.cu.
   */
  template <bool has_nested>
  batch_insert_result probe_and_insert_impl(table_view const& batch_keys,
                                            rmm::cuda_stream_view stream);

  void do_aggregate(table_view const& data, rmm::cuda_stream_view stream);

  [[nodiscard]] std::unique_ptr<table> gather_agg_results(rmm::cuda_stream_view stream,
                                                          rmm::device_async_resource_ref mr) const;
  [[nodiscard]] std::unique_ptr<table> gather_distinct_keys(
    rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const;

  [[nodiscard]] std::pair<std::unique_ptr<table>, std::vector<aggregation_result>> do_finalize(
    rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const;

  void do_merge(impl const& other, rmm::cuda_stream_view stream);
};

}  // namespace cudf::groupby
