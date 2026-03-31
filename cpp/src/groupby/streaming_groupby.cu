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
#include <thrust/for_each.h>
#include <thrust/transform.h>

#include <algorithm>
#include <memory>
#include <vector>

namespace cudf::groupby {
namespace {

// Key columns always have null masks (created with ALL_NULL), so the row
// comparator and hasher must always be null-aware.
auto constexpr has_null = cudf::nullate::DYNAMIC{true};

template <typename SetRef>
struct compute_target_indices_fn {
  SetRef set_ref;
  bitmask_type const* row_bitmask;
  size_type staging_offset;
  size_type* new_insertion_count;  // Optional: atomic counter for new insertions (may be nullptr).

  __device__ size_type operator()(size_type batch_idx) const
  {
    auto const global_idx = staging_offset + batch_idx;
    if (row_bitmask && !cudf::bit_is_set(row_bitmask, global_idx)) {
      return cudf::detail::CUDF_SIZE_TYPE_SENTINEL;
    }
    auto ref_copy               = set_ref;
    auto const [iter, inserted] = ref_copy.insert_and_find(global_idx);
    if (new_insertion_count && inserted) {
      cuda::atomic_ref<size_type, cuda::thread_scope_device>{*new_insertion_count}.fetch_add(
        1, cuda::memory_order_relaxed);
    }
    return *iter;
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
  size_type _staging_end{0};      ///< Write cursor into key table; next batch starts here
  bool _has_nullable_keys{
    false};  ///< Whether any input batch had nullable key columns (for output)

  ///< Fixed-capacity key table (`_max_groups` rows) storing all key rows from all batches
  ///< (including duplicates). Backs the hash set's row equality comparator — stored indices
  ///< reference rows in this table, so duplicate slots cannot be reclaimed.
  std::vector<std::unique_ptr<column>> _key_columns;

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

  ///< Persistent cuco::static_set mapping key rows (by key table index) to group indices.
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
    for (auto idx : _key_indices) {
      auto const& key_col = data.column(idx);
      CUDF_EXPECTS(cudf::is_fixed_width(key_col.type()),
                   "Streaming groupby only supports fixed-width key columns.",
                   std::invalid_argument);
      _key_columns.push_back(make_fixed_width_column(key_col.type(),
                                                     _max_groups,
                                                     mask_state::ALL_NULL,
                                                     stream,
                                                     cudf::get_current_device_resource_ref()));
    }

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
    // This avoids re-calling build_aggregation_requests + extract_single_pass_aggs per batch.
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

    // Compute per-request first-column offset into the flattened single-pass agg table,
    // needed to retrieve the correct intermediate column for compound agg finalization.
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

  void create_key_set(table_view const& key_table, rmm::cuda_stream_view stream)
  {
    auto preprocessed_keys = cudf::detail::row::hash::preprocessed_table::create(key_table, stream);
    auto const comparator  = cudf::detail::row::equality::self_comparator{preprocessed_keys};
    auto const row_hash    = cudf::detail::row::hash::row_hasher{std::move(preprocessed_keys)};

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

  void copy_keys_to_staging(table_view const& batch_keys,
                            size_type staging_offset,
                            rmm::cuda_stream_view stream)
  {
    for (size_type c = 0; c < num_keys(); ++c) {
      if (batch_keys.column(c).nullable()) { _has_nullable_keys = true; }
      auto dst = _key_columns[c]->mutable_view();
      cudf::copy_range_in_place(
        batch_keys.column(c), dst, 0, batch_keys.column(c).size(), staging_offset, stream);
    }
  }

  /// Key table view that always includes null masks (for row operators).
  [[nodiscard]] table_view key_table_view(size_type num_rows) const
  {
    std::vector<column_view> views;
    views.reserve(_key_columns.size());
    for (auto const& col : _key_columns) {
      views.push_back(cudf::slice(col->view(), {0, num_rows})[0]);
    }
    return table_view{views};
  }

  /// Key table view that only includes null masks if input data was actually nullable (for output).
  [[nodiscard]] table_view key_table_output_view(size_type num_rows) const
  {
    std::vector<column_view> views;
    views.reserve(_key_columns.size());
    for (auto const& col : _key_columns) {
      if (_has_nullable_keys) {
        views.push_back(cudf::slice(col->view(), {0, num_rows})[0]);
      } else {
        views.push_back(column_view{col->type(), num_rows, col->view().head(), nullptr, 0});
      }
    }
    return table_view{views};
  }

  // Compute target indices for a batch of rows by inserting into the hash set.
  // Returns the target indices, new insertion count, and the row bitmask buffer (kept alive).
  struct batch_insert_result {
    rmm::device_uvector<size_type> target_indices;
    size_type new_insertions;
    rmm::device_buffer bitmask_buffer;
  };

  batch_insert_result compute_batch_target_indices(size_type batch_size,
                                                   size_type staging_offset,
                                                   table_view const& full_key_view,
                                                   rmm::cuda_stream_view stream)
  {
    auto preprocessed_keys =
      cudf::detail::row::hash::preprocessed_table::create(full_key_view, stream);
    auto const comparator = cudf::detail::row::equality::self_comparator{preprocessed_keys};
    auto const row_hash   = cudf::detail::row::hash::row_hasher{std::move(preprocessed_keys)};

    auto const d_row_hash = row_hash.device_hasher(has_null);

    auto const skip_rows_with_nulls =
      cudf::has_nulls(full_key_view) && _null_handling == null_policy::EXCLUDE;
    auto row_bitmask_buffer =
      skip_rows_with_nulls ? compute_row_bitmask(full_key_view, stream) : rmm::device_buffer{};
    auto const* row_bitmask =
      skip_rows_with_nulls ? static_cast<bitmask_type const*>(row_bitmask_buffer.data()) : nullptr;

    rmm::device_uvector<size_type> target_indices(batch_size, stream);

    // Device counter for new insertions — avoids expensive post-hoc hash set scan.
    auto d_counter = cudf::detail::device_scalar<size_type>(0, stream);

    auto const hasher      = detail::hash::row_hasher_with_cache_t{d_row_hash};
    auto const d_row_equal = comparator.equal_to<false>(has_null, null_equality::EQUAL);
    auto set_ref           = _key_set->ref(cuco::op::insert_and_find)
                     .rebind_key_eq(d_row_equal)
                     .rebind_hash_function(hasher);
    thrust::transform(
      rmm::exec_policy_nosync(stream),
      cuda::counting_iterator<size_type>(0),
      cuda::counting_iterator<size_type>(batch_size),
      target_indices.begin(),
      compute_target_indices_fn{set_ref, row_bitmask, staging_offset, d_counter.data()});

    auto const new_insertions = d_counter.value(stream);
    return {std::move(target_indices), new_insertions, std::move(row_bitmask_buffer)};
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

    CUDF_EXPECTS(static_cast<int64_t>(_staging_end) + batch_size <= _max_groups,
                 "Key table capacity exceeded: staging_end (" + std::to_string(_staging_end) +
                   ") + batch_size (" + std::to_string(batch_size) + ") > max_groups (" +
                   std::to_string(_max_groups) + ").",
                 std::overflow_error);

    if (!_initialized) { initialize(data, stream); }

    std::vector<column_view> batch_key_cols;
    batch_key_cols.reserve(num_keys());
    for (auto idx : _key_indices) {
      batch_key_cols.push_back(data.column(idx));
    }
    auto const batch_keys = table_view{batch_key_cols};

    auto const staging_offset = _staging_end;
    copy_keys_to_staging(batch_keys, staging_offset, stream);
    _staging_end += batch_size;

    auto const visible_rows  = _staging_end;
    auto const full_key_view = key_table_view(visible_rows);

    if (!_key_set) { create_key_set(full_key_view, stream); }

    auto result = compute_batch_target_indices(batch_size, staging_offset, full_key_view, stream);
    _num_unique_keys += result.new_insertions;

    // Build values_view using cached column index mapping (avoids per-batch
    // build_aggregation_requests + extract_single_pass_aggs overhead).
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

    auto const visible_rows = _staging_end;

    auto populated = detail::hash::extract_populated_keys(
      *_key_set, visible_rows, stream, cudf::get_current_device_resource_ref());

    auto keys_gathered = cudf::detail::gather(key_table_output_view(visible_rows),
                                              populated,
                                              out_of_bounds_policy::DONT_CHECK,
                                              cudf::negative_index_policy::NOT_ALLOWED,
                                              stream,
                                              mr);

    auto agg_gathered = cudf::detail::gather(_agg_results->view(),
                                             populated,
                                             out_of_bounds_policy::DONT_CHECK,
                                             cudf::negative_index_policy::NOT_ALLOWED,
                                             stream,
                                             mr);

    if (_has_compound_aggs) {
      cudf::detail::result_cache cache(_agg_kinds.size());

      // Build finalization requests with the gathered intermediate column at the first
      // agg offset so the compound finalizer sees the correct input type.
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

      return {std::move(keys_gathered),
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
    return {std::move(keys_gathered), std::move(results)};
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

    // Gather other's populated keys and raw intermediate agg results.
    auto const other_visible  = other._staging_end;
    auto const other_key_view = other.key_table_output_view(other_visible);

    auto const default_mr = cudf::get_current_device_resource_ref();
    auto other_populated =
      detail::hash::extract_populated_keys(*other._key_set, other_visible, stream, default_mr);

    auto const num_other_groups = static_cast<size_type>(other_populated.size());
    if (num_other_groups == 0) { return; }

    auto other_keys = cudf::detail::gather(other_key_view,
                                           other_populated,
                                           out_of_bounds_policy::DONT_CHECK,
                                           cudf::negative_index_policy::NOT_ALLOWED,
                                           stream,
                                           default_mr);

    auto other_aggs = cudf::detail::gather(other._agg_results->view(),
                                           other_populated,
                                           out_of_bounds_policy::DONT_CHECK,
                                           cudf::negative_index_policy::NOT_ALLOWED,
                                           stream,
                                           default_mr);

    CUDF_EXPECTS(static_cast<int64_t>(_staging_end) + num_other_groups <= _max_groups,
                 "Key table capacity exceeded during merge: staging_end (" +
                   std::to_string(_staging_end) + ") + merge groups (" +
                   std::to_string(num_other_groups) + ") > max_groups (" +
                   std::to_string(_max_groups) + ").",
                 std::overflow_error);

    auto const staging_offset = _staging_end;
    copy_keys_to_staging(other_keys->view(), staging_offset, stream);
    _staging_end += num_other_groups;

    auto const visible_rows  = _staging_end;
    auto const full_key_view = key_table_view(visible_rows);

    if (!_key_set) { create_key_set(full_key_view, stream); }

    auto result =
      compute_batch_target_indices(num_other_groups, staging_offset, full_key_view, stream);
    _num_unique_keys += result.new_insertions;

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

    // Check after merge aggregation so internal state remains consistent if this throws.
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
