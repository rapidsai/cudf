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
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <cuco/static_set.cuh>
#include <cuda/std/functional>

#include <memory>
#include <vector>

namespace cudf::groupby {

// The actual comparator and hasher are always rebinded per-batch via
// rebind_key_eq / rebind_hash_function, so the types used here are never invoked.
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

/// N-table comparator for the persistent hash set.
///
/// Indices >= num_stored are "batch rows" (encoded as num_stored + batch_idx).
/// Indices < num_stored are "stored groups" resolved via companion vectors
/// key_batch[idx] and key_row[idx] to (compacted_batch_table, row_within_batch).
///
/// Cross-comparators are pre-built as device_row_comparator(batch, compacted[k])
/// and stored in a device array.  Self-comparisons use batch_self_eq.
template <typename RowEqT>
struct n_table_comparator {
  RowEqT batch_self_eq;        ///< Self-comparator on the current batch table
  RowEqT const* cross_eqs;     ///< Device array [num_compacted_batches]: batch vs compacted[k]
  size_type const* key_batch;  ///< Companion vector: compacted batch index per stored entry
  size_type const* key_row;    ///< Companion vector: row index within compacted batch
  size_type num_stored;        ///< Threshold: idx >= num_stored is a batch row

  __device__ bool operator()(size_type lhs, size_type rhs) const noexcept
  {
    bool const lhs_is_batch = (lhs >= num_stored);
    bool const rhs_is_batch = (rhs >= num_stored);

    if (lhs_is_batch && rhs_is_batch) { return batch_self_eq(lhs - num_stored, rhs - num_stored); }
    if (lhs_is_batch) { return cross_eqs[key_batch[rhs]](lhs - num_stored, key_row[rhs]); }
    if (rhs_is_batch) { return cross_eqs[key_batch[lhs]](rhs - num_stored, key_row[lhs]); }
    return lhs == rhs;
  }
};

/// insert_and_find on the main set for each batch row.
/// Produces target_indices (stored encoded index) and insertion flags.
template <typename SetRef>
struct insert_and_map_fn {
  mutable SetRef set_ref;
  bitmask_type const* row_bitmask;
  size_type offset;
  bool* inserted_flags;

  __device__ size_type operator()(size_type batch_idx) const
  {
    if (row_bitmask && !cudf::bit_is_set(row_bitmask, batch_idx)) {
      inserted_flags[batch_idx] = false;
      return cudf::detail::CUDF_SIZE_TYPE_SENTINEL;
    }
    auto const [iter, inserted] = set_ref.insert_and_find(offset + batch_idx);
    inserted_flags[batch_idx]   = inserted;
    return *iter;
  }
};

/// Hasher that uses a precomputed hash cache with an offset.
/// Given index `idx`, returns `cache[idx - offset]`.
struct offset_cache_hasher {
  hash_value_type const* cache;
  size_type offset;
  __device__ hash_value_type operator()(size_type idx) const noexcept
  {
    return cache[idx - offset];
  }
};

/// For each newly discovered key, scatter companion vectors and record the
/// encoded index for finalization.  Combines two operations in a single pass.
struct scatter_new_key_metadata {
  size_type const* batch_local_indices;  ///< batch-local row indices of new keys
  size_type* key_batch;                  ///< companion: batch id at encoded index
  size_type* key_row;                    ///< companion: row within compacted batch
  size_type* encoded_indices;            ///< encoded_indices[dense_group] = encoded index
  size_type encoding_offset;             ///< num_stored: added to batch-local index to get encoded
  size_type batch_id;
  size_type num_groups_before;
  __device__ void operator()(size_type idx) const
  {
    auto const encoded_idx                   = batch_local_indices[idx] + encoding_offset;
    key_batch[encoded_idx]                   = batch_id;
    key_row[encoded_idx]                     = idx;
    encoded_indices[num_groups_before + idx] = encoded_idx;
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
  size_type _distinct_count{0};
  /// High-water mark of encoded indices consumed.  Encoded batch indices are
  /// [_num_stored, _num_stored + batch_size).  After each batch, _num_stored
  /// advances by batch_size (the full encoding range, not just distinct count).
  /// The sum of all batch sizes must not exceed max_groups.
  size_type _num_stored{0};
  bool _has_nullable_keys{false};
  bool _has_nested_keys{false};

  // -- Compacted batch storage --
  std::vector<std::unique_ptr<table>> _compacted_batches;
  std::vector<std::shared_ptr<cudf::detail::row::equality::preprocessed_table>>
    _preprocessed_batches;

  /// Companion vectors indexed by encoded index (sparse, up to max_groups).
  std::unique_ptr<rmm::device_uvector<size_type>> _key_batch;
  std::unique_ptr<rmm::device_uvector<size_type>> _key_row;

  /// For each dense group g, _encoded_indices[g] is the encoded index
  /// where that group's aggregation results live in _agg_results.
  /// Used at finalization to gather sparse agg results into dense output.
  std::unique_ptr<rmm::device_uvector<size_type>> _encoded_indices;

  std::vector<size_type> _request_first_agg_offset;
  std::vector<aggregation::Kind> _agg_kinds;
  std::vector<std::unique_ptr<aggregation>> _agg_objects;
  std::vector<int8_t> _is_agg_intermediate;
  bool _has_compound_aggs{false};

  /// Sparse agg results table, pre-allocated to max_groups rows.
  /// Indexed by encoded index (not dense group index).
  std::unique_ptr<table> _agg_results;
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

  /// Template implementation of probe_and_insert, split by has_nested.
  /// Defined in insert.cuh, instantiated in insert.cu and insert_nested.cu.
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
