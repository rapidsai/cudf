/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common/utils.hpp"
#include "hash/extract_single_pass_aggs.hpp"
#include "hash/hash_compound_agg_finalizer.hpp"
#include "hash/helpers.cuh"
#include "hash/output_utils.hpp"
#include "hash/single_pass_functors.cuh"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/aggregation/device_aggregators.cuh>
#include <cudf/detail/aggregation/result_cache.hpp>
#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/detail/row_operator/hashing.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/groupby.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/atomic>
#include <cuda/iterator>
#include <cuda/std/type_traits>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <algorithm>
#include <memory>
#include <vector>

namespace cudf::groupby {
namespace {

// Whether key columns have null masks depends on whether any input batch had nullable keys.
// When keys are non-nullable, we skip null mask allocation and use null-unaware row operators.

/// Probe the hash set for each batch row. Encodes batch row i as (offset + i)
/// before calling find. Returns the stored group index if found, SENTINEL otherwise.
template <typename SetRef>
struct probe_target_indices_fn {
  SetRef set_ref;
  bitmask_type const* row_bitmask;  // batch-local bitmask (may be nullptr)
  size_type offset;                 // num_accumulated: batch row i → offset + i

  __device__ size_type operator()(size_type batch_idx) const
  {
    if (row_bitmask && !cudf::bit_is_set(row_bitmask, batch_idx)) {
      return cudf::detail::CUDF_SIZE_TYPE_SENTINEL;
    }
    auto ref_copy = set_ref;
    auto const it = ref_copy.find(offset + batch_idx);
    if (it == set_ref.end()) { return cudf::detail::CUDF_SIZE_TYPE_SENTINEL; }
    return *it;
  }
};

/// Hasher that uses a precomputed hash cache with an offset.
/// Given index `idx`, returns `cache[idx - offset]`.
/// Used to hash batch rows encoded as (num_accumulated + batch_idx).
struct offset_cache_hasher {
  hash_value_type const* cache;
  size_type offset;
  __device__ hash_value_type operator()(size_type idx) const noexcept
  {
    return cache[idx - offset];
  }
};

/// Comparator for probe phase: compares a stored accumulated-table index (lhs)
/// against a batch-encoded index (rhs = num_accumulated + batch_row).
/// Wraps two_table_comparator's device comparator.
template <typename CrossEq>
struct probe_comparator {
  CrossEq cross_eq;
  size_type num_accumulated;

  __device__ bool operator()(size_type lhs, size_type rhs) const noexcept
  {
    // lhs: stored accumulated index [0, num_accumulated)
    // rhs: probe batch index [num_accumulated, ...)
    // Swap if needed (cuco may call either order).
    if (rhs < num_accumulated) {
      // rhs is accumulated, lhs is batch
      return cross_eq(cudf::detail::row::lhs_index_type{rhs},
                      cudf::detail::row::rhs_index_type{lhs - num_accumulated});
    }
    return cross_eq(cudf::detail::row::lhs_index_type{lhs},
                    cudf::detail::row::rhs_index_type{rhs - num_accumulated});
  }
};

/**
 * @brief Element aggregator for merging intermediate results.
 *
 * Like cudf::detail::element_aggregator but for COUNT_VALID/COUNT_ALL it adds
 * the source count value instead of incrementing by 1. Also handles
 * SUM_OF_SQUARES by adding directly (the source already contains squared sums,
 * so we must NOT square again).
 */
struct merge_element_aggregator {
  template <typename Source, aggregation::Kind k>
  __device__ void operator()(mutable_column_device_view target,
                             size_type target_index,
                             column_device_view source,
                             size_type source_index) const noexcept
  {
    if constexpr (!cudf::detail::is_valid_aggregation<Source, k>()) {
      return;  // Unsupported type/kind combination.
    } else {
      if constexpr (k != aggregation::COUNT_ALL) {
        if (source.is_null(source_index)) { return; }
      }
      if constexpr (!(k == aggregation::COUNT_VALID || k == aggregation::COUNT_ALL)) {
        if (target.is_null(target_index)) { target.set_valid(target_index); }
      }

      if constexpr (k == aggregation::COUNT_VALID || k == aggregation::COUNT_ALL) {
        // Merge counts by addition (not increment-by-1).
        using Target = cudf::detail::target_type_t<Source, k>;
        cudf::detail::atomic_add(&target.element<Target>(target_index),
                                 source.element<Target>(source_index));
      } else if constexpr (k == aggregation::SUM_OF_SQUARES) {
        // Source already contains the sum of squared values; just add.
        using Target = cudf::detail::target_type_t<Source, k>;
        cudf::detail::atomic_add(&target.element<Target>(target_index),
                                 static_cast<Target>(source.element<Source>(source_index)));
      } else {
        // SUM, MIN, MAX, PRODUCT: standard update_target_element works correctly.
        cudf::detail::update_target_element<Source, k>{}(
          target, target_index, source, source_index);
      }
    }
  }
};

/**
 * @brief Functor to merge intermediate aggregation results from a source table into a target table,
 * using pre-computed target indices.
 */
struct merge_single_pass_aggs_fn {
  size_type const* target_indices;
  aggregation::Kind const* aggs;
  table_device_view source_values;
  mutable_table_device_view target_values;

  __device__ void operator()(int64_t idx) const
  {
    auto const num_rows       = source_values.num_rows();
    auto const source_row_idx = static_cast<size_type>(idx % num_rows);
    if (auto const target_row_idx = target_indices[source_row_idx];
        target_row_idx != cudf::detail::CUDF_SIZE_TYPE_SENTINEL) {
      auto const col_idx     = static_cast<size_type>(idx / num_rows);
      auto const& source_col = source_values.column(col_idx);
      auto const& target_col = target_values.column(col_idx);
      cudf::detail::dispatch_type_and_aggregation(source_col.type(),
                                                  aggs[col_idx],
                                                  merge_element_aggregator{},
                                                  target_col,
                                                  target_row_idx,
                                                  source_col,
                                                  source_row_idx);
    }
  }
};

constexpr bool is_supported_streaming_agg(aggregation::Kind k)
{
  switch (k) {
    case aggregation::SUM:
    case aggregation::SUM_OF_SQUARES:
    case aggregation::PRODUCT:
    case aggregation::MIN:
    case aggregation::MAX:
    case aggregation::COUNT_VALID:
    case aggregation::COUNT_ALL:
    case aggregation::MEAN:
    case aggregation::M2:
    case aggregation::STD:
    case aggregation::VARIANCE: return true;
    default: return false;
  }
}

void validate_requests(host_span<streaming_aggregation_request const> requests)
{
  for (auto const& req : requests) {
    for (auto const& agg : req.aggregations) {
      CUDF_EXPECTS(is_supported_streaming_agg(agg->kind),
                   "Unsupported aggregation kind for streaming groupby. "
                   "Only hash-based aggregations (SUM, PRODUCT, MIN, MAX, COUNT, MEAN, "
                   "SUM_OF_SQUARES, M2, VARIANCE, STD) are supported.",
                   std::invalid_argument);
    }
  }
}

std::vector<aggregation_request> build_aggregation_requests(
  host_span<streaming_aggregation_request const> requests, table_view const& data)
{
  std::vector<aggregation_request> result;
  for (auto const& req : requests) {
    aggregation_request ar;
    ar.values = data.column(req.column_index);
    for (auto const& agg : req.aggregations) {
      ar.aggregations.push_back(std::unique_ptr<groupby_aggregation>{
        dynamic_cast<groupby_aggregation*>(agg->clone().release())});
    }
    result.push_back(std::move(ar));
  }
  return result;
}

rmm::device_buffer compute_row_bitmask(table_view const& keys, rmm::cuda_stream_view stream)
{
  if (keys.num_columns() == 0 || !cudf::has_nulls(keys)) { return rmm::device_buffer{}; }
  auto result = cudf::bitmask_and(keys, stream);
  if (result.second == 0) { return rmm::device_buffer{}; }
  return std::move(result.first);
}

/// Self-comparator wrapper that strips an offset before comparing.
/// Used for deduplicating batch rows encoded as (num_accumulated + batch_idx).
template <typename SelfEq>
struct offset_self_comparator {
  SelfEq self_eq;
  size_type offset;
  __device__ bool operator()(size_type lhs, size_type rhs) const noexcept
  {
    return self_eq(lhs - offset, rhs - offset);
  }
};

/// Translates virtual batch indices to dense accumulated indices via a lookup table.
struct apply_virtual_to_dense {
  size_type const* virtual_to_dense;
  size_type num_accumulated;
  __device__ size_type operator()(size_type target) const
  {
    if (target >= num_accumulated) { return virtual_to_dense[target - num_accumulated]; }
    return target;
  }
};

/// Subtracts a fixed offset from a value.
struct subtract_offset {
  size_type offset;
  __device__ size_type operator()(size_type v) const { return v - offset; }
};

/// For SENTINEL batch rows, insert_and_find on the temp set to deduplicate.
/// For non-SENTINEL rows, keep the existing target. Counts unique new insertions.
template <typename SetRef>
struct dedup_new_keys_fn {
  size_type const* targets;
  SetRef temp_ref;
  size_type num_accumulated;
  bitmask_type const* row_bitmask;
  size_type* new_count;

  __device__ size_type operator()(size_type batch_idx, size_type cur_target) const
  {
    if (cur_target != cudf::detail::CUDF_SIZE_TYPE_SENTINEL) { return cur_target; }
    // Skip null-excluded rows.
    if (row_bitmask && !cudf::bit_is_set(row_bitmask, batch_idx)) {
      return cudf::detail::CUDF_SIZE_TYPE_SENTINEL;
    }
    // This is a new key — deduplicate within the batch using the temp set.
    auto const encoded_idx      = num_accumulated + batch_idx;
    auto ref_copy               = temp_ref;
    auto const [iter, inserted] = ref_copy.insert_and_find(encoded_idx);
    if (inserted) {
      cuda::atomic_ref<size_type, cuda::thread_scope_device>{*new_count}.fetch_add(
        1, cuda::memory_order_relaxed);
    }
    return *iter;  // encoded index of the first occurrence
  }
};

/// Inserts a single key into a hash set ref.
template <typename SetRef>
struct insert_single_key {
  SetRef ref;
  __device__ void operator()(size_type idx) const
  {
    auto ref_copy = ref;
    ref_copy.insert(idx);
  }
};

/// Scatter dense group indices into a batch-sized lookup array.
/// deduped_combined[k] is a combined-table index (num_accumulated + batch_row).
/// We set batch_to_dense[batch_row] = num_accumulated + k.
struct scatter_dense_index {
  size_type const* deduped_combined;
  size_type* batch_to_dense;
  size_type num_accumulated;
  __device__ void operator()(size_type k) const
  {
    auto const batch_row      = deduped_combined[k] - num_accumulated;
    batch_to_dense[batch_row] = num_accumulated + k;
  }
};

}  // namespace

struct streaming_groupby::impl {
  std::vector<size_type> _key_indices;  ///< Column indices in each batch that form the grouping key
  std::vector<streaming_aggregation_request>
    _requests_clone;                         ///< Deep copy of user-supplied aggregation requests
  std::vector<size_type> _aggs_per_request;  ///< Number of user-facing aggs per request (for
                                             ///< splitting results in finalize)
  size_type _max_groups;       ///< Upper bound on distinct groups the hash set is sized for
  null_policy _null_handling;  ///< Whether rows with null keys are excluded from aggregation

  bool _initialized{false};  ///< Whether initialize() has been called (deferred until first batch
                             ///< for type inference)
  size_type _num_unique_keys{0};  ///< Running count of distinct keys inserted into the hash set
  bool _has_nullable_keys{
    false};  ///< Whether any input batch had nullable key columns (for output)

  /// Accumulated unique keys table — grows via concatenation as new unique keys are discovered.
  /// Replaces the old fixed-capacity staging area. Supports all column types including
  /// variable-width (strings, lists, structs).
  std::unique_ptr<table> _accumulated_keys;

  std::vector<size_type> _request_first_agg_offset;  ///< Per-request column offset into flattened
                                                     ///< agg results (for compound finalization)
  std::vector<aggregation::Kind> _agg_kinds;         ///< Expanded simple agg kinds after compound
                                                     ///< expansion/dedup (one per results column)
  std::vector<std::unique_ptr<aggregation>>
    _agg_objects;  ///< Owning copies of expanded simple agg objects, parallel to `_agg_kinds`
  std::vector<int8_t> _is_agg_intermediate;  ///< 1 if agg is an intermediate from compound
                                             ///< expansion, 0 if user-requested
  bool _has_compound_aggs{
    false};  ///< Whether any user-facing agg is compound (requires finalization)

  std::unique_ptr<table> _agg_results;  ///< Pre-allocated `_max_groups`-row table accumulating
                                        ///< intermediate agg results across batches
  std::vector<size_type>
    _value_col_indices;  ///< Cached expanded-agg-column to data-column-index mapping (avoids
                         ///< per-batch extract_single_pass_aggs)

  ///< Persistent cuco::static_set mapping key rows (by virtual index) to group indices.
  ///< Sized for `_max_groups` entries (with load-factor headroom). Created once on the first
  ///< batch; the stored comparator/hasher become stale but are never used directly — device
  ///< refs are rebound with fresh row operators before every operation.
  std::unique_ptr<detail::hash::global_set_t> _key_set;

  [[nodiscard]] size_type num_keys() const { return static_cast<size_type>(_key_indices.size()); }

  [[nodiscard]] bool has_state() const { return _initialized && _num_unique_keys > 0; }

  impl(host_span<size_type const> key_indices,
       host_span<streaming_aggregation_request const> requests,
       size_type max_groups,
       null_policy null_handling)
    : _max_groups{max_groups}, _null_handling{null_handling}
  {
    CUDF_EXPECTS(max_groups > 0, "max_groups must be positive.", std::invalid_argument);
    if (!key_indices.empty()) { _key_indices.assign(key_indices.begin(), key_indices.end()); }
    validate_requests(requests);

    for (auto const& req : requests) {
      _aggs_per_request.push_back(static_cast<size_type>(req.aggregations.size()));
      streaming_aggregation_request clone;
      clone.column_index = req.column_index;
      for (auto const& agg : req.aggregations) {
        clone.aggregations.push_back(std::unique_ptr<groupby_aggregation>{
          dynamic_cast<groupby_aggregation*>(agg->clone().release())});
      }
      _requests_clone.push_back(std::move(clone));
    }
  }

  void initialize(table_view const& data, rmm::cuda_stream_view stream)
  {
    // No fixed-width restriction — accumulated keys table grows dynamically.
    // Create an empty accumulated keys table with the correct schema.
    std::vector<std::unique_ptr<column>> empty_cols;
    for (auto idx : _key_indices) {
      auto const& key_col = data.column(idx);
      empty_cols.push_back(cudf::empty_like(key_col));
    }
    _accumulated_keys = std::make_unique<table>(std::move(empty_cols));

    auto agg_requests = build_aggregation_requests(_requests_clone, data);
    auto [values_view, agg_kinds_hv, agg_objects, is_intermediate, has_compound] =
      detail::hash::extract_single_pass_aggs(agg_requests, stream);

    _agg_kinds.assign(agg_kinds_hv.begin(), agg_kinds_hv.end());
    _agg_objects         = std::move(agg_objects);
    _is_agg_intermediate = std::move(is_intermediate);
    _has_compound_aggs   = has_compound;

    _agg_results = detail::hash::create_results_table(_max_groups,
                                                      values_view,
                                                      _agg_kinds,
                                                      _is_agg_intermediate,
                                                      stream,
                                                      cudf::get_current_device_resource_ref());

    // Build the cached mapping from each expanded agg column to its source data column index.
    _value_col_indices.reserve(values_view.num_columns());
    for (size_type c = 0; c < values_view.num_columns(); ++c) {
      auto const* col_head = values_view.column(c).head();
      for (size_type j = 0; j < static_cast<size_type>(agg_requests.size()); ++j) {
        if (agg_requests[j].values.head() == col_head) {
          _value_col_indices.push_back(_requests_clone[j].column_index);
          break;
        }
      }
    }

    // Compute per-request first-column offset into the flattened single-pass agg table.
    size_type offset = 0;
    for (auto const& req : _requests_clone) {
      _request_first_agg_offset.push_back(offset);
      auto const values_type = data.column(req.column_index).type();
      for (auto const& agg : req.aggregations) {
        auto const& ga = dynamic_cast<groupby_aggregation const&>(*agg);
        offset +=
          static_cast<size_type>(detail::hash::get_simple_aggregations(ga, values_type).size());
      }
    }

    _initialized = true;
  }

  void create_key_set(table_view const& dummy_key_table, rmm::cuda_stream_view stream)
  {
    // Create the hash set with a dummy comparator/hasher — they are always rebound
    // before any operation. We just need the correct types.
    auto preprocessed =
      cudf::detail::row::hash::preprocessed_table::create(dummy_key_table, stream);
    auto const comparator = cudf::detail::row::equality::self_comparator{preprocessed};
    auto const row_hash   = cudf::detail::row::hash::row_hasher{std::move(preprocessed)};

    auto const has_null    = cudf::nullate::DYNAMIC{false};
    auto const d_row_hash  = row_hash.device_hasher(has_null);
    auto const d_row_equal = comparator.equal_to<false>(has_null, null_equality::EQUAL);

    _key_set = std::make_unique<detail::hash::global_set_t>(
      cuco::extent<int64_t>{static_cast<int64_t>(_max_groups)},
      cudf::detail::CUCO_DESIRED_LOAD_FACTOR,
      cuco::empty_key{cudf::detail::CUDF_SIZE_TYPE_SENTINEL},
      d_row_equal,
      detail::hash::probing_scheme_t{detail::hash::row_hasher_with_cache_t{d_row_hash}},
      cuco::thread_scope_device,
      cuco::storage<detail::hash::GROUPBY_BUCKET_SIZE>{},
      rmm::mr::polymorphic_allocator<char>{},
      stream.value());
  }

  /// Check and lazily update nullable state based on incoming batch keys.
  void update_nullable_state(table_view const& batch_keys)
  {
    if (_has_nullable_keys) return;
    for (size_type c = 0; c < num_keys(); ++c) {
      if (batch_keys.column(c).nullable()) {
        _has_nullable_keys = true;
        return;
      }
    }
  }

  // Compute target indices for a batch of rows by inserting into the hash set.
  struct batch_insert_result {
    rmm::device_uvector<size_type> target_indices;
    size_type new_insertions;
    rmm::device_buffer bitmask_buffer;
  };

  /**
   * @brief Two-phase probe-then-insert for batch keys.
   *
   * Phase 1 — Probe: Build a temporary combined table [accumulated | batch] and use
   *   self_comparator + find() to look up each batch row. Returns the dense group
   *   index for existing keys, SENTINEL for new keys.
   *
   * Phase 2 — Deduplicate new keys within the batch using insert_and_find on the
   *   combined table, gather unique new keys, concatenate onto accumulated table,
   *   and insert their dense indices into the main hash set.
   *
   * The main hash set only ever stores dense accumulated indices [0, num_unique).
   */
  batch_insert_result probe_and_insert(table_view const& batch_keys, rmm::cuda_stream_view stream)
  {
    auto const batch_size      = batch_keys.num_rows();
    auto const num_accumulated = _num_unique_keys;
    auto const temp_mr         = cudf::get_current_device_resource_ref();
    auto const has_null        = cudf::nullate::DYNAMIC{_has_nullable_keys};

    // Preprocess batch and accumulated tables independently — no concatenation.
    auto pp_batch = cudf::detail::row::hash::preprocessed_table::create(batch_keys, stream);
    auto const batch_hasher_obj = cudf::detail::row::hash::row_hasher{pp_batch};
    auto const d_batch_hash     = batch_hasher_obj.device_hasher(has_null);

    // Precompute batch hash values into a cache — O(batch_size).
    rmm::device_uvector<hash_value_type> batch_hash_cache(batch_size, stream, temp_mr);
    thrust::transform(rmm::exec_policy_nosync(stream),
                      cuda::counting_iterator<size_type>(0),
                      cuda::counting_iterator<size_type>(batch_size),
                      batch_hash_cache.begin(),
                      d_batch_hash);

    // Batch-local bitmask for null exclusion.
    auto const skip_rows_with_nulls = _has_nullable_keys && _null_handling == null_policy::EXCLUDE;
    auto bitmask_buffer =
      skip_rows_with_nulls ? compute_row_bitmask(batch_keys, stream) : rmm::device_buffer{};
    auto const* batch_bitmask =
      skip_rows_with_nulls ? static_cast<bitmask_type const*>(bitmask_buffer.data()) : nullptr;

    rmm::device_uvector<size_type> target_indices(batch_size, stream, temp_mr);

    // Phase 1: Probe the main set using two_table_comparator(accumulated, batch).
    // The hasher uses the precomputed batch hash cache with offset encoding.
    // find() only hashes probe keys and compares probe vs stored — never stored vs stored.
    if (num_accumulated > 0) {
      auto const accumulated_view = _accumulated_keys->view();
      auto pp_accumulated =
        cudf::detail::row::hash::preprocessed_table::create(accumulated_view, stream);

      auto const cross_cmp =
        cudf::detail::row::equality::two_table_comparator{pp_accumulated, pp_batch};
      auto const d_cross_eq = cross_cmp.equal_to<false>(has_null, null_equality::EQUAL);

      using cross_eq_t      = decltype(d_cross_eq);
      auto const probe_eq   = probe_comparator<cross_eq_t>{d_cross_eq, num_accumulated};
      auto const probe_hash = offset_cache_hasher{batch_hash_cache.data(), num_accumulated};

      auto find_ref =
        _key_set->ref(cuco::op::find).rebind_key_eq(probe_eq).rebind_hash_function(probe_hash);

      thrust::transform(
        rmm::exec_policy_nosync(stream),
        cuda::counting_iterator<size_type>(0),
        cuda::counting_iterator<size_type>(batch_size),
        target_indices.begin(),
        probe_target_indices_fn<decltype(find_ref)>{find_ref, batch_bitmask, num_accumulated});
    } else {
      // No accumulated keys — all batch rows are new (SENTINEL).
      // dedup_new_keys_fn will handle null exclusion via bitmask check.
      thrust::fill(rmm::exec_policy_nosync(stream),
                   target_indices.begin(),
                   target_indices.end(),
                   cudf::detail::CUDF_SIZE_TYPE_SENTINEL);
    }

    // Count how many batch rows are new (SENTINEL) after probe.
    // Early exit if no new keys — skip temp set allocation entirely.
    auto const num_sentinel = thrust::count(rmm::exec_policy_nosync(stream),
                                            target_indices.begin(),
                                            target_indices.end(),
                                            cudf::detail::CUDF_SIZE_TYPE_SENTINEL);
    if (num_sentinel == 0) { return {std::move(target_indices), 0, std::move(bitmask_buffer)}; }

    // Phase 2: Deduplicate new keys within the batch using a temp set.
    // The temp set uses batch-only row operators with batch-local indices
    // encoded as (num_accumulated + batch_idx) so the offset_cache_hasher works.
    auto const batch_self_cmp = cudf::detail::row::equality::self_comparator{pp_batch};
    auto const d_batch_eq     = batch_self_cmp.equal_to<false>(has_null, null_equality::EQUAL);

    // Wrap batch self-comparator to handle offset encoding:
    // temp set stores (num_accumulated + batch_idx), comparator strips offset.
    auto const dedup_eq   = offset_self_comparator{d_batch_eq, num_accumulated};
    auto const dedup_hash = offset_cache_hasher{batch_hash_cache.data(), num_accumulated};

    // Create temp set sized to num_sentinel (upper bound on unique new keys),
    // not batch_size. This dramatically reduces memory at low cardinality.
    auto temp_set = detail::hash::global_set_t(
      cuco::extent<int64_t>{static_cast<int64_t>(num_sentinel)},
      cudf::detail::CUCO_DESIRED_LOAD_FACTOR,
      cuco::empty_key{cudf::detail::CUDF_SIZE_TYPE_SENTINEL},
      d_batch_eq,
      detail::hash::probing_scheme_t{detail::hash::row_hasher_with_cache_t{d_batch_hash}},
      cuco::thread_scope_device,
      cuco::storage<detail::hash::GROUPBY_BUCKET_SIZE>{},
      rmm::mr::polymorphic_allocator<char>{},
      stream.value());

    auto temp_iaf_ref = temp_set.ref(cuco::op::insert_and_find)
                          .rebind_key_eq(dedup_eq)
                          .rebind_hash_function(dedup_hash);

    auto d_new_count = cudf::detail::device_scalar<size_type>(0, stream, temp_mr);
    thrust::transform(
      rmm::exec_policy_nosync(stream),
      cuda::counting_iterator<size_type>(0),
      cuda::counting_iterator<size_type>(batch_size),
      target_indices.begin(),
      target_indices.begin(),
      dedup_new_keys_fn<decltype(temp_iaf_ref)>{
        target_indices.begin(), temp_iaf_ref, num_accumulated, batch_bitmask, d_new_count.data()});

    auto const num_new_unique = d_new_count.value(stream);

    if (num_new_unique > 0) {
      // Retrieve deduplicated encoded indices from temp set.
      rmm::device_uvector<size_type> deduped_encoded(num_new_unique, stream, temp_mr);
      temp_set.retrieve_all(deduped_encoded.begin(), stream.value());

      // Sort for deterministic dense-index assignment.
      thrust::sort(rmm::exec_policy_nosync(stream), deduped_encoded.begin(), deduped_encoded.end());

      // Convert to batch-local indices for gather.
      rmm::device_uvector<size_type> new_batch_indices(num_new_unique, stream, temp_mr);
      thrust::transform(rmm::exec_policy_nosync(stream),
                        deduped_encoded.begin(),
                        deduped_encoded.end(),
                        new_batch_indices.begin(),
                        subtract_offset{num_accumulated});

      // Gather new unique keys from the batch.
      auto new_keys = cudf::detail::gather(batch_keys,
                                           new_batch_indices,
                                           out_of_bounds_policy::DONT_CHECK,
                                           cudf::negative_index_policy::NOT_ALLOWED,
                                           stream,
                                           temp_mr);

      // Concatenate onto accumulated keys — only when new keys found.
      if (_accumulated_keys->num_rows() == 0) {
        _accumulated_keys = std::move(new_keys);
      } else {
        auto const views  = std::vector<table_view>{_accumulated_keys->view(), new_keys->view()};
        _accumulated_keys = cudf::concatenate(views, stream, temp_mr);
      }

      // Build batch_to_dense lookup: for encoded index (num_accumulated + j),
      // batch_to_dense[j] = num_accumulated + k (the dense group index).
      rmm::device_uvector<size_type> batch_to_dense(batch_size, stream, temp_mr);
      thrust::fill(rmm::exec_policy_nosync(stream),
                   batch_to_dense.begin(),
                   batch_to_dense.end(),
                   cudf::detail::CUDF_SIZE_TYPE_SENTINEL);

      thrust::for_each_n(
        rmm::exec_policy_nosync(stream),
        cuda::counting_iterator<size_type>(0),
        num_new_unique,
        scatter_dense_index{deduped_encoded.data(), batch_to_dense.data(), num_accumulated});

      // Remap targets >= num_accumulated to dense accumulated indices.
      thrust::transform(rmm::exec_policy_nosync(stream),
                        target_indices.begin(),
                        target_indices.end(),
                        target_indices.begin(),
                        apply_virtual_to_dense{batch_to_dense.data(), num_accumulated});

      // Insert new dense indices [num_accumulated, num_accumulated + num_new_unique)
      // into the main hash set with the UPDATED accumulated table's row operators.
      auto const updated_view = _accumulated_keys->view();
      auto pp_updated = cudf::detail::row::hash::preprocessed_table::create(updated_view, stream);
      auto const cmp_updated  = cudf::detail::row::equality::self_comparator{pp_updated};
      auto const hash_updated = cudf::detail::row::hash::row_hasher{std::move(pp_updated)};
      auto const d_hash_upd   = hash_updated.device_hasher(has_null);
      auto const d_equal_upd  = cmp_updated.equal_to<false>(has_null, null_equality::EQUAL);
      auto const hasher_upd   = detail::hash::row_hasher_with_cache_t{d_hash_upd};

      auto insert_ref =
        _key_set->ref(cuco::op::insert).rebind_key_eq(d_equal_upd).rebind_hash_function(hasher_upd);
      thrust::for_each(rmm::exec_policy_nosync(stream),
                       cuda::counting_iterator<size_type>(num_accumulated),
                       cuda::counting_iterator<size_type>(num_accumulated + num_new_unique),
                       insert_single_key<decltype(insert_ref)>{insert_ref});

      _num_unique_keys += num_new_unique;
    }

    return {std::move(target_indices), num_new_unique, std::move(bitmask_buffer)};
  }

  void check_unique_key_count()
  {
    CUDF_EXPECTS(_num_unique_keys <= _max_groups,
                 "Unique keys (" + std::to_string(_num_unique_keys) + ") exceeded max_groups (" +
                   std::to_string(_max_groups) + ").");
  }

  void do_aggregate(table_view const& data, rmm::cuda_stream_view stream)
  {
    auto const batch_size = data.num_rows();
    if (batch_size == 0) { return; }

    CUDF_EXPECTS(batch_size <= _max_groups,
                 "Batch size (" + std::to_string(batch_size) + ") exceeds max_groups (" +
                   std::to_string(_max_groups) + ").",
                 std::invalid_argument);

    if (!_initialized) { initialize(data, stream); }

    std::vector<column_view> batch_key_cols;
    batch_key_cols.reserve(num_keys());
    for (auto idx : _key_indices) {
      batch_key_cols.push_back(data.column(idx));
    }
    auto const batch_keys = table_view{batch_key_cols};

    update_nullable_state(batch_keys);

    if (!_key_set) { create_key_set(batch_keys, stream); }

    auto result = probe_and_insert(batch_keys, stream);

    // Build values_view using cached column index mapping.
    std::vector<column_view> value_cols;
    value_cols.reserve(_value_col_indices.size());
    for (auto idx : _value_col_indices) {
      value_cols.push_back(data.column(idx));
    }
    auto const values_view = table_view{value_cols};

    auto const d_values    = table_device_view::create(values_view, stream);
    auto d_results_ptr     = mutable_table_device_view::create(*_agg_results, stream);
    auto const d_agg_kinds = cudf::detail::make_device_uvector_async(
      _agg_kinds, stream, cudf::get_current_device_resource_ref());

    auto const num_agg_cols = static_cast<int64_t>(_agg_kinds.size());
    thrust::for_each_n(
      rmm::exec_policy_nosync(stream),
      cuda::counting_iterator<int64_t>(0),
      static_cast<int64_t>(batch_size) * num_agg_cols,
      detail::hash::compute_single_pass_aggs_dense_output_fn{
        result.target_indices.begin(), d_agg_kinds.data(), *d_values, *d_results_ptr});

    // Check after aggregation so internal state remains consistent if this throws.
    check_unique_key_count();
  }

  [[nodiscard]] std::pair<std::unique_ptr<table>, std::vector<aggregation_result>> do_finalize(
    rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const
  {
    CUDF_EXPECTS(_initialized, "Cannot finalize streaming_groupby with no accumulated data.");

    // The accumulated keys table contains exactly the unique keys in group-index order.
    // No need to extract populated keys from the hash set — just use the accumulated table.
    auto keys_output = std::make_unique<table>(_accumulated_keys->view(), stream, mr);

    // Gather aggregation results in the same order as accumulated keys.
    // The hash set maps key rows to indices [0, num_unique_keys), which is exactly
    // the row order of _accumulated_keys. So we gather agg results at indices [0, N).
    auto const num_unique = _num_unique_keys;
    rmm::device_uvector<size_type> gather_map(
      num_unique, stream, cudf::get_current_device_resource_ref());
    thrust::sequence(
      rmm::exec_policy_nosync(stream), gather_map.begin(), gather_map.end(), size_type{0});

    auto agg_gathered = cudf::detail::gather(_agg_results->view(),
                                             gather_map,
                                             out_of_bounds_policy::DONT_CHECK,
                                             cudf::negative_index_policy::NOT_ALLOWED,
                                             stream,
                                             mr);

    if (_has_compound_aggs) {
      cudf::detail::result_cache cache(_agg_kinds.size());

      std::vector<aggregation_request> agg_requests_fin;
      for (size_t i = 0; i < _requests_clone.size(); ++i) {
        aggregation_request ar;
        ar.values = agg_gathered->view().column(_request_first_agg_offset[i]);
        for (auto const& agg : _requests_clone[i].aggregations) {
          ar.aggregations.push_back(std::unique_ptr<groupby_aggregation>{
            dynamic_cast<groupby_aggregation*>(agg->clone().release())});
        }
        agg_requests_fin.push_back(std::move(ar));
      }

      auto [values_view_fin,
            agg_kinds_fin,
            agg_objects_fin,
            is_intermediate_fin,
            has_compound_fin] = detail::hash::extract_single_pass_aggs(agg_requests_fin, stream);

      detail::hash::finalize_output(values_view_fin, agg_objects_fin, agg_gathered, &cache, stream);

      for (auto const& req : agg_requests_fin) {
        auto const finalizer =
          detail::hash::hash_compound_agg_finalizer(req.values, &cache, nullptr, stream, mr);
        for (auto const& agg : req.aggregations) {
          cudf::detail::aggregation_dispatcher(agg->kind, finalizer, *agg);
        }
      }

      return {std::move(keys_output),
              detail::extract_results(
                host_span<aggregation_request const>{agg_requests_fin}, cache, stream, mr)};
    }

    auto released_cols = agg_gathered->release();
    std::vector<aggregation_result> results;
    size_type col_idx = 0;
    for (auto num_aggs : _aggs_per_request) {
      aggregation_result agg_result;
      for (size_type a = 0; a < num_aggs; ++a) {
        agg_result.results.push_back(std::move(released_cols[col_idx++]));
      }
      results.push_back(std::move(agg_result));
    }
    return {std::move(keys_output), std::move(results)};
  }

  void do_merge(impl const& other, rmm::cuda_stream_view stream)
  {
    if (!other._initialized || !other.has_state()) { return; }
    CUDF_EXPECTS(_initialized,
                 "Cannot merge into an uninitialized streaming_groupby. "
                 "Call aggregate() at least once before merge().");
    CUDF_EXPECTS(other._num_unique_keys <= _max_groups,
                 "Merge source unique keys (" + std::to_string(other._num_unique_keys) +
                   ") exceeds max_groups (" + std::to_string(_max_groups) + ").",
                 std::invalid_argument);

    auto const default_mr = cudf::get_current_device_resource_ref();

    // The other's accumulated keys are already the unique keys in group-index order.
    auto const other_key_view   = other._accumulated_keys->view();
    auto const num_other_groups = other._num_unique_keys;
    if (num_other_groups == 0) { return; }

    // Gather other's agg results at indices [0, num_other_groups).
    rmm::device_uvector<size_type> other_gather_map(num_other_groups, stream, default_mr);
    thrust::sequence(
      rmm::exec_policy_nosync(stream), other_gather_map.begin(), other_gather_map.end());

    auto other_aggs = cudf::detail::gather(other._agg_results->view(),
                                           other_gather_map,
                                           out_of_bounds_policy::DONT_CHECK,
                                           cudf::negative_index_policy::NOT_ALLOWED,
                                           stream,
                                           default_mr);

    // Treat the other's unique keys as a "batch" and insert into this object.
    update_nullable_state(other_key_view);

    if (!_key_set) { create_key_set(other_key_view, stream); }

    auto result = probe_and_insert(other_key_view, stream);

    // Merge aggregation values using target indices.
    auto const d_source    = table_device_view::create(other_aggs->view(), stream);
    auto d_target          = mutable_table_device_view::create(*_agg_results, stream);
    auto const d_agg_kinds = cudf::detail::make_device_uvector_async(
      _agg_kinds, stream, cudf::get_current_device_resource_ref());

    auto const num_agg_cols = static_cast<int64_t>(_agg_kinds.size());
    thrust::for_each_n(rmm::exec_policy_nosync(stream),
                       cuda::counting_iterator<int64_t>(0),
                       static_cast<int64_t>(num_other_groups) * num_agg_cols,
                       merge_single_pass_aggs_fn{
                         result.target_indices.begin(), d_agg_kinds.data(), *d_source, *d_target});

    check_unique_key_count();
  }
};

streaming_groupby::streaming_groupby(host_span<size_type const> key_indices,
                                     host_span<streaming_aggregation_request const> requests,
                                     size_type max_groups,
                                     null_policy null_handling)
  : _impl{std::make_unique<impl>(key_indices, requests, max_groups, null_handling)}
{
}

streaming_groupby::~streaming_groupby() = default;

streaming_groupby::streaming_groupby(streaming_groupby&&) noexcept = default;

streaming_groupby& streaming_groupby::operator=(streaming_groupby&&) noexcept = default;

void streaming_groupby::aggregate(table_view const& data,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  _impl->do_aggregate(data, stream);
}

void streaming_groupby::merge(streaming_groupby const& other,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  _impl->do_merge(*other._impl, stream);
}

std::pair<std::unique_ptr<table>, std::vector<aggregation_result>> streaming_groupby::finalize(
  rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const
{
  CUDF_FUNC_RANGE();
  return _impl->do_finalize(stream, mr);
}

}  // namespace cudf::groupby
