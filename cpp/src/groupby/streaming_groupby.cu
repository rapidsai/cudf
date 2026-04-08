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

#include <cuda/iterator>
#include <cuda/std/functional>
#include <thrust/copy.h>
#include <thrust/for_each.h>
#include <thrust/gather.h>
#include <thrust/transform.h>

#include <algorithm>
#include <memory>
#include <vector>

namespace cudf::groupby {
namespace {

// Concrete device_row_comparator type used throughout streaming groupby.
using row_eq_t = detail::hash::row_comparator_t;

// ---------------------------------------------------------------------------
// Functors
// ---------------------------------------------------------------------------

/// N-table comparator for the persistent hash set.
///
/// Indices >= num_stored are "batch rows" (encoded as num_stored + batch_idx).
/// Indices < num_stored are "stored groups" resolved via companion vectors
/// key_batch[idx] and key_row[idx] to (compacted_batch_table, row_within_batch).
///
/// Cross-comparators are pre-built as device_row_comparator(batch, compacted[k])
/// and stored in a device array.  Self-comparisons use batch_self_eq.
struct n_table_comparator {
  row_eq_t batch_self_eq;      ///< Self-comparator on the current batch table
  row_eq_t const* cross_eqs;   ///< Device array [num_compacted_batches]: batch vs compacted[k]
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
  SetRef set_ref;
  bitmask_type const* row_bitmask;
  size_type offset;
  bool* inserted_flags;

  __device__ size_type operator()(size_type batch_idx) const
  {
    if (row_bitmask && !cudf::bit_is_set(row_bitmask, batch_idx)) {
      inserted_flags[batch_idx] = false;
      return cudf::detail::CUDF_SIZE_TYPE_SENTINEL;
    }
    auto ref_copy               = set_ref;
    auto const [iter, inserted] = ref_copy.insert_and_find(offset + batch_idx);
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

/// Adds a fixed offset to a value.
struct offset_adder {
  size_type offset;
  __device__ size_type operator()(size_type v) const { return v + offset; }
};

/// For each newly discovered key k, scatter companion vectors and record the
/// encoded index for finalization.  Combines two operations in a single pass.
struct scatter_new_key_metadata {
  size_type const* deduped_encoded;  ///< deduped_encoded[k] = encoded index
  size_type* key_batch;              ///< companion: batch id at encoded index
  size_type* key_row;                ///< companion: row within compacted batch
  size_type* group_encoded;          ///< group_encoded[dense_group] = encoded index
  size_type batch_id;
  size_type num_groups_before;
  __device__ void operator()(size_type k) const
  {
    auto const enc                       = deduped_encoded[k];
    key_batch[enc]                       = batch_id;
    key_row[enc]                         = k;
    group_encoded[num_groups_before + k] = enc;
  }
};

/**
 * @brief Element aggregator for merging intermediate results.
 */
struct merge_element_aggregator {
  template <typename Source, aggregation::Kind k>
  __device__ void operator()(mutable_column_device_view target,
                             size_type target_index,
                             column_device_view source,
                             size_type source_index) const noexcept
  {
    if constexpr (!cudf::detail::is_valid_aggregation<Source, k>()) {
      return;
    } else {
      if constexpr (k != aggregation::COUNT_ALL) {
        if (source.is_null(source_index)) { return; }
      }
      if constexpr (!(k == aggregation::COUNT_VALID || k == aggregation::COUNT_ALL)) {
        if (target.is_null(target_index)) { target.set_valid(target_index); }
      }

      if constexpr (k == aggregation::COUNT_VALID || k == aggregation::COUNT_ALL) {
        using Target = cudf::detail::target_type_t<Source, k>;
        cudf::detail::atomic_add(&target.element<Target>(target_index),
                                 source.element<Target>(source_index));
      } else if constexpr (k == aggregation::SUM_OF_SQUARES) {
        using Target = cudf::detail::target_type_t<Source, k>;
        cudf::detail::atomic_add(&target.element<Target>(target_index),
                                 static_cast<Target>(source.element<Source>(source_index)));
      } else {
        cudf::detail::update_target_element<Source, k>{}(
          target, target_index, source, source_index);
      }
    }
  }
};

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

void validate_requests(host_span<streaming_aggregation_request const> requests)
{
  for (auto const& req : requests) {
    for (auto const& agg : req.aggregations) {
      CUDF_EXPECTS(detail::is_hash_aggregation(agg->kind),
                   "Unsupported aggregation kind for streaming groupby. "
                   "Only hash-based aggregations are supported.",
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

/// Compute a combined null bitmask for multi-column keys.
/// Returns {buffer, raw_pointer} where pointer is null if no nulls.
std::pair<rmm::device_buffer, bitmask_type const*> compute_row_bitmask(table_view const& keys,
                                                                       rmm::cuda_stream_view stream)
{
  if (keys.num_columns() == 0 || !cudf::has_nulls(keys)) {
    return {rmm::device_buffer{0, stream}, nullptr};
  }
  // Single-column fast path: reuse the column's null mask directly.
  if (keys.num_columns() == 1) {
    auto const& col = keys.column(0);
    if (col.offset() == 0) { return {rmm::device_buffer{0, stream}, col.null_mask()}; }
    auto buf = cudf::copy_bitmask(col, stream);
    auto ptr = static_cast<bitmask_type const*>(buf.data());
    return {std::move(buf), ptr};
  }
  auto [buf, null_count] = cudf::bitmask_and(keys, stream);
  if (null_count == 0) { return {rmm::device_buffer{0, stream}, nullptr}; }
  return {std::move(buf), static_cast<bitmask_type const*>(buf.data())};
}

// ---------------------------------------------------------------------------
// Helper: build N cross-comparators on host and copy to device
// ---------------------------------------------------------------------------

rmm::device_uvector<row_eq_t> build_cross_comparators(
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> const& pp_batch,
  std::vector<std::shared_ptr<cudf::detail::row::equality::preprocessed_table>> const& pp_batches,
  cudf::nullate::DYNAMIC has_null,
  rmm::cuda_stream_view stream)
{
  auto const n       = static_cast<size_type>(pp_batches.size());
  auto const temp_mr = cudf::get_current_device_resource_ref();

  std::vector<row_eq_t> h_eqs;
  h_eqs.reserve(n);
  for (size_type k = 0; k < n; ++k) {
    auto const cross_cmp =
      cudf::detail::row::equality::two_table_comparator{pp_batch, pp_batches[k]};
    auto const adapter = cross_cmp.equal_to<false>(has_null, null_equality::EQUAL);
    h_eqs.push_back(adapter.comparator);
  }

  return cudf::detail::make_device_uvector_async(h_eqs, stream, temp_mr);
}

}  // namespace

// ===========================================================================
// streaming_groupby::impl
// ===========================================================================

struct streaming_groupby::impl {
  std::vector<size_type> _key_indices;
  std::vector<streaming_aggregation_request> _requests_clone;
  std::vector<size_type> _aggs_per_request;
  size_type _max_groups;
  null_policy _null_handling;

  bool _initialized{false};
  size_type _num_unique_keys{0};
  /// High-water mark of encoded indices consumed.  Encoded batch indices are
  /// [_num_stored, _num_stored + batch_size).  After each batch, _num_stored
  /// advances by batch_size (the full encoding range, not just num_unique).
  size_type _num_stored{0};
  bool _has_nullable_keys{false};

  // -- Compacted batch storage --
  std::vector<std::unique_ptr<table>> _compacted_batches;
  std::vector<std::shared_ptr<cudf::detail::row::equality::preprocessed_table>> _pp_batches;

  /// Companion vectors indexed by ENCODED INDEX (sparse, up to 2*max_groups).
  std::unique_ptr<rmm::device_uvector<size_type>> _key_batch;
  std::unique_ptr<rmm::device_uvector<size_type>> _key_row;

  /// For each dense group g, _group_encoded_indices[g] is the encoded index
  /// where that group's aggregation results live in _agg_results.
  /// Used at finalization to gather sparse agg results into dense output.
  std::unique_ptr<rmm::device_uvector<size_type>> _group_encoded_indices;

  std::vector<size_type> _request_first_agg_offset;
  std::vector<aggregation::Kind> _agg_kinds;
  std::vector<std::unique_ptr<aggregation>> _agg_objects;
  std::vector<int8_t> _is_agg_intermediate;
  bool _has_compound_aggs{false};

  /// Capacity of sparse agg results / companion vectors (2 * max_groups).
  size_type _sparse_capacity{0};

  /// Sparse agg results table, pre-allocated to _sparse_capacity rows.
  /// Indexed by encoded index (not dense group index).
  std::unique_ptr<table> _agg_results;
  std::vector<size_type> _value_col_indices;
  rmm::device_uvector<aggregation::Kind> _d_agg_kinds;

  std::unique_ptr<detail::hash::global_set_t> _key_set;

  [[nodiscard]] size_type num_keys() const { return static_cast<size_type>(_key_indices.size()); }
  [[nodiscard]] bool has_state() const { return _initialized && _num_unique_keys > 0; }

  impl(host_span<size_type const> key_indices,
       host_span<streaming_aggregation_request const> requests,
       size_type max_groups,
       null_policy null_handling)
    : _max_groups{max_groups},
      _null_handling{null_handling},
      _d_agg_kinds{0, cudf::get_default_stream(), cudf::get_current_device_resource_ref()}
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
    auto const mr = cudf::get_current_device_resource_ref();

    auto agg_requests = build_aggregation_requests(_requests_clone, data);
    auto [values_view, agg_kinds_hv, agg_objects, is_intermediate, has_compound] =
      detail::hash::extract_single_pass_aggs(agg_requests, stream);

    _agg_kinds.assign(agg_kinds_hv.begin(), agg_kinds_hv.end());
    _agg_objects         = std::move(agg_objects);
    _is_agg_intermediate = std::move(is_intermediate);
    _has_compound_aggs   = has_compound;

    // Sparse agg results: 2 * max_groups to accommodate encoded index range.
    _sparse_capacity =
      static_cast<size_type>(std::min(static_cast<int64_t>(_max_groups) * 2,
                                      static_cast<int64_t>(std::numeric_limits<size_type>::max())));
    _agg_results = detail::hash::create_results_table(
      _sparse_capacity, values_view, _agg_kinds, _is_agg_intermediate, stream, mr);

    _d_agg_kinds = cudf::detail::make_device_uvector_async(_agg_kinds, stream, mr);

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

    // Companion vectors: 2 * max_groups to accommodate encoded index range.
    _key_batch = std::make_unique<rmm::device_uvector<size_type>>(_sparse_capacity, stream, mr);
    _key_row   = std::make_unique<rmm::device_uvector<size_type>>(_sparse_capacity, stream, mr);

    // Group-to-encoded-index map: sized to max_groups (one entry per unique key).
    _group_encoded_indices =
      std::make_unique<rmm::device_uvector<size_type>>(_max_groups, stream, mr);

    _initialized = true;
  }

  void create_key_set(table_view const& dummy_key_table, rmm::cuda_stream_view stream)
  {
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

  struct batch_insert_result {
    rmm::device_uvector<size_type> target_indices;
    size_type new_insertions;
    rmm::device_buffer bitmask_buffer;
  };

  /**
   * @brief Unified insert_and_find on the main set for all batches.
   *
   * Each batch row is encoded as (num_stored + batch_idx).  The n_table_comparator
   * resolves stored entries via companion vectors and batch entries via the batch
   * table.  insert_and_find naturally deduplicates both across batches (existing
   * keys) and within the batch (intra-batch duplicates).
   *
   * Target indices are the raw encoded indices stored in the set — sparse.
   * Aggregation writes to agg_results[encoded_index].  Densification happens
   * only at finalization.
   */
  batch_insert_result probe_and_insert(table_view const& batch_keys, rmm::cuda_stream_view stream)
  {
    auto const batch_size = batch_keys.num_rows();
    auto const num_stored = _num_stored;
    auto const temp_mr    = cudf::get_current_device_resource_ref();
    auto const has_null   = cudf::nullate::DYNAMIC{_has_nullable_keys};

    CUDF_EXPECTS(static_cast<int64_t>(num_stored) + batch_size <= _sparse_capacity,
                 "Encoded index range (" + std::to_string(num_stored) + " + " +
                   std::to_string(batch_size) + ") exceeds sparse capacity (" +
                   std::to_string(_sparse_capacity) +
                   "). Use smaller batches or increase max_groups.",
                 std::overflow_error);

    // Preprocess batch for row operators.
    auto pp_batch = cudf::detail::row::hash::preprocessed_table::create(batch_keys, stream);
    auto const batch_hasher_obj = cudf::detail::row::hash::row_hasher{pp_batch};
    auto const d_batch_hash     = batch_hasher_obj.device_hasher(has_null);

    // Precompute batch hash values — O(batch_size).
    rmm::device_uvector<hash_value_type> batch_hash_cache(batch_size, stream, temp_mr);
    thrust::transform(rmm::exec_policy_nosync(stream),
                      cuda::counting_iterator<size_type>(0),
                      cuda::counting_iterator<size_type>(batch_size),
                      batch_hash_cache.begin(),
                      d_batch_hash);

    // Batch-local bitmask for null exclusion.
    auto const skip_rows_with_nulls = _has_nullable_keys && _null_handling == null_policy::EXCLUDE;
    auto [bitmask_buffer, batch_bitmask] = skip_rows_with_nulls
                                             ? compute_row_bitmask(batch_keys, stream)
                                             : std::pair<rmm::device_buffer, bitmask_type const*>{
                                                 rmm::device_buffer{0, stream}, nullptr};

    // Build comparator.  When num_stored == 0 and _pp_batches is empty,
    // the n_table_comparator only enters the batch-self branch.
    auto const batch_self_cmp = cudf::detail::row::equality::self_comparator{pp_batch};
    auto const batch_self_eq  = batch_self_cmp.equal_to<false>(has_null, null_equality::EQUAL);

    auto d_cross_eqs = build_cross_comparators(pp_batch, _pp_batches, has_null, stream);

    auto const eq = n_table_comparator{
      batch_self_eq, d_cross_eqs.data(), _key_batch->data(), _key_row->data(), num_stored};
    auto const hasher = offset_cache_hasher{batch_hash_cache.data(), num_stored};

    auto iaf_ref =
      _key_set->ref(cuco::op::insert_and_find).rebind_key_eq(eq).rebind_hash_function(hasher);

    // insert_and_find for all batch rows.
    rmm::device_uvector<size_type> target_indices(batch_size, stream, temp_mr);
    rmm::device_uvector<bool> inserted_flags(batch_size, stream, temp_mr);

    thrust::transform(rmm::exec_policy_nosync(stream),
                      cuda::counting_iterator<size_type>(0),
                      cuda::counting_iterator<size_type>(batch_size),
                      target_indices.begin(),
                      insert_and_map_fn{iaf_ref, batch_bitmask, num_stored, inserted_flags.data()});

    // Count newly inserted keys.
    auto const num_new_unique = static_cast<size_type>(thrust::count(
      rmm::exec_policy_nosync(stream), inserted_flags.begin(), inserted_flags.end(), true));

    if (num_new_unique > 0) {
      // Stream compact: get the batch-local indices of newly inserted keys.
      rmm::device_uvector<size_type> batch_local_indices(num_new_unique, stream, temp_mr);
      thrust::copy_if(rmm::exec_policy_nosync(stream),
                      cuda::counting_iterator<size_type>(0),
                      cuda::counting_iterator<size_type>(batch_size),
                      inserted_flags.begin(),
                      batch_local_indices.begin(),
                      cuda::std::identity{});

      // Build encoded indices from batch-local: encoded = num_stored + batch_local.
      rmm::device_uvector<size_type> deduped_encoded(num_new_unique, stream, temp_mr);
      thrust::transform(rmm::exec_policy_nosync(stream),
                        batch_local_indices.begin(),
                        batch_local_indices.end(),
                        deduped_encoded.begin(),
                        offset_adder{num_stored});

      // Gather compacted unique keys from the batch.
      auto compacted = cudf::detail::gather(batch_keys,
                                            batch_local_indices,
                                            out_of_bounds_policy::DONT_CHECK,
                                            cudf::negative_index_policy::NOT_ALLOWED,
                                            stream,
                                            temp_mr);

      auto pp_compacted =
        cudf::detail::row::hash::preprocessed_table::create(compacted->view(), stream);

      // Store the compacted batch.
      auto const new_batch_id = static_cast<size_type>(_compacted_batches.size());
      _compacted_batches.push_back(std::move(compacted));
      _pp_batches.push_back(pp_compacted);

      // Scatter companion vectors and record encoded indices in a single pass.
      thrust::for_each_n(rmm::exec_policy_nosync(stream),
                         cuda::counting_iterator<size_type>(0),
                         num_new_unique,
                         scatter_new_key_metadata{deduped_encoded.data(),
                                                  _key_batch->data(),
                                                  _key_row->data(),
                                                  _group_encoded_indices->data(),
                                                  new_batch_id,
                                                  _num_unique_keys});

      _num_unique_keys += num_new_unique;
    }

    // Advance encoding offset by the full batch size — the entire range
    // [num_stored, num_stored + batch_size) is consumed, even for rows that
    // were duplicates or null-excluded.  The next batch must encode after this range.
    _num_stored += batch_size;

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

    auto const d_values = table_device_view::create(values_view, stream);
    auto d_results_ptr  = mutable_table_device_view::create(*_agg_results, stream);

    auto const num_agg_cols = static_cast<int64_t>(_agg_kinds.size());
    thrust::for_each_n(
      rmm::exec_policy_nosync(stream),
      cuda::counting_iterator<int64_t>(0),
      static_cast<int64_t>(batch_size) * num_agg_cols,
      detail::hash::compute_single_pass_aggs_dense_output_fn{
        result.target_indices.begin(), _d_agg_kinds.data(), *d_values, *d_results_ptr});

    check_unique_key_count();
  }

  [[nodiscard]] std::unique_ptr<table> gather_all_unique_keys(
    rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const
  {
    if (_compacted_batches.empty()) { return std::make_unique<table>(); }
    if (_compacted_batches.size() == 1) {
      return std::make_unique<table>(_compacted_batches[0]->view(), stream, mr);
    }
    std::vector<table_view> views;
    views.reserve(_compacted_batches.size());
    for (auto const& batch : _compacted_batches) {
      views.push_back(batch->view());
    }
    return cudf::concatenate(views, stream, mr);
  }

  [[nodiscard]] std::pair<std::unique_ptr<table>, std::vector<aggregation_result>> do_finalize(
    rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const
  {
    CUDF_EXPECTS(_initialized, "Cannot finalize streaming_groupby with no accumulated data.");

    auto keys_output = gather_all_unique_keys(stream, mr);

    // Gather sparse agg results at the encoded indices for each dense group.
    // _group_encoded_indices[g] is the encoded index for group g.
    auto const num_unique = _num_unique_keys;
    auto agg_gathered =
      cudf::detail::gather(_agg_results->view(),
                           device_span<size_type const>{_group_encoded_indices->data(),
                                                        static_cast<std::size_t>(num_unique)},
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

    auto const mr = cudf::get_current_device_resource_ref();

    auto other_keys             = other.gather_all_unique_keys(stream, mr);
    auto const other_key_view   = other_keys->view();
    auto const num_other_groups = other._num_unique_keys;
    if (num_other_groups == 0) { return; }

    // Gather other's sparse agg results using its encoded indices.
    auto other_aggs =
      cudf::detail::gather(other._agg_results->view(),
                           device_span<size_type const>{other._group_encoded_indices->data(),
                                                        static_cast<std::size_t>(num_other_groups)},
                           out_of_bounds_policy::DONT_CHECK,
                           cudf::negative_index_policy::NOT_ALLOWED,
                           stream,
                           mr);

    update_nullable_state(other_key_view);

    if (!_key_set) { create_key_set(other_key_view, stream); }

    auto result = probe_and_insert(other_key_view, stream);

    // Merge aggregation values using target indices (sparse encoded).
    auto const d_source = table_device_view::create(other_aggs->view(), stream);
    auto d_target       = mutable_table_device_view::create(*_agg_results, stream);

    auto const num_agg_cols = static_cast<int64_t>(_agg_kinds.size());
    thrust::for_each_n(rmm::exec_policy_nosync(stream),
                       cuda::counting_iterator<int64_t>(0),
                       static_cast<int64_t>(num_other_groups) * num_agg_cols,
                       merge_single_pass_aggs_fn{
                         result.target_indices.begin(), _d_agg_kinds.data(), *d_source, *d_target});

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
