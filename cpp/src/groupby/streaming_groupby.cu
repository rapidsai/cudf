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
#include <cudf/detail/gather.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/detail/row_operator/hashing.cuh>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
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
#include <cuda/std/type_traits>
#include <thrust/for_each.h>
#include <thrust/transform.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <vector>

namespace cudf::groupby {
namespace {

struct copy_null_mask_fn {
  bitmask_type* dst_mask;
  bitmask_type const* src_mask;
  size_type offset;

  __device__ void operator()(size_type i) const
  {
    if (!cudf::bit_is_set(src_mask, i)) { cudf::clear_bit(dst_mask, offset + i); }
  }
};

template <typename SetRef>
struct compute_target_indices_fn {
  SetRef set_ref;
  bitmask_type const* row_bitmask;
  size_type staging_offset;

  __device__ size_type operator()(size_type batch_idx) const
  {
    auto const global_idx = staging_offset + batch_idx;
    if (row_bitmask && !cudf::bit_is_set(row_bitmask, global_idx)) {
      return cudf::detail::CUDF_SIZE_TYPE_SENTINEL;
    }
    auto ref_copy = set_ref;
    return *ref_copy.insert_and_find(global_idx).first;
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
  std::vector<size_type> _key_indices;
  std::vector<streaming_aggregation_request> _requests_clone;
  std::vector<size_type> _aggs_per_request;
  size_type _max_groups;
  null_policy _null_handling;

  bool _initialized{false};
  size_type _num_unique_keys{0};
  size_type _staging_end{0};
  size_type _staging_capacity{0};
  bool _has_nullable_keys{false};

  std::vector<std::unique_ptr<column>> _key_columns;
  std::vector<size_type> _request_first_agg_offset;

  std::vector<aggregation::Kind> _agg_kinds;
  std::vector<std::unique_ptr<aggregation>> _agg_objects;
  std::vector<int8_t> _is_agg_intermediate;
  bool _has_compound_aggs{false};

  std::unique_ptr<table> _agg_results;

  // Only the non-nullable set is used since streaming_groupby requires fixed-width keys
  // (no nested columns). The equality comparator is rebound per-batch with correct null
  // awareness, so flat nullable columns are handled correctly.
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

  void initialize(table_view const& data,
                  rmm::cuda_stream_view stream,
                  rmm::device_async_resource_ref mr)
  {
    _staging_capacity =
      static_cast<size_type>(std::min(static_cast<int64_t>(_max_groups) * 2,
                                      static_cast<int64_t>(std::numeric_limits<size_type>::max())));

    for (auto idx : _key_indices) {
      auto const& key_col = data.column(idx);
      CUDF_EXPECTS(cudf::is_fixed_width(key_col.type()),
                   "Streaming groupby only supports fixed-width key columns.",
                   std::invalid_argument);
      _key_columns.push_back(make_fixed_width_column(
        key_col.type(), _staging_capacity, mask_state::ALL_NULL, stream, mr));
    }

    auto agg_requests = build_aggregation_requests(_requests_clone, data);
    auto [values_view, agg_kinds_hv, agg_objects, is_intermediate, has_compound] =
      detail::hash::extract_single_pass_aggs(agg_requests, stream);

    _agg_kinds.assign(agg_kinds_hv.begin(), agg_kinds_hv.end());
    _agg_objects         = std::move(agg_objects);
    _is_agg_intermediate = std::move(is_intermediate);
    _has_compound_aggs   = has_compound;

    _agg_results = detail::hash::create_results_table(
      _max_groups, values_view, _agg_kinds, _is_agg_intermediate, stream, mr);

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

  void ensure_staging_capacity(size_type additional_rows,
                               rmm::cuda_stream_view stream,
                               rmm::device_async_resource_ref mr)
  {
    auto const needed64 = static_cast<int64_t>(_staging_end) + additional_rows;
    if (needed64 <= _staging_capacity) { return; }

    auto const new_cap64    = std::max(static_cast<int64_t>(_staging_capacity) * 2, needed64);
    auto const new_capacity = static_cast<size_type>(
      std::min(new_cap64, static_cast<int64_t>(std::numeric_limits<size_type>::max())));

    for (auto& col : _key_columns) {
      auto new_col =
        make_fixed_width_column(col->type(), new_capacity, mask_state::ALL_NULL, stream, mr);
      auto new_view       = new_col->mutable_view();
      auto const old_view = col->view();

      // Copy existing data.
      auto const elem_size = cudf::size_of(col->type());
      CUDF_CUDA_TRY(cudf::detail::memcpy_async(
        new_view.head(), old_view.head(), elem_size * _staging_end, stream));

      // Copy existing null mask bits.
      if (_staging_end > 0) {
        auto const mask_bytes =
          static_cast<size_t>(num_bitmask_words(_staging_end)) * sizeof(bitmask_type);
        CUDF_CUDA_TRY(cudf::detail::memcpy_async(
          new_view.null_mask(), old_view.null_mask(), mask_bytes, stream));
      }

      col = std::move(new_col);
    }

    _staging_capacity = new_capacity;

    // The hash set's stored comparator references the old preprocessed_table which points
    // to the old column buffers. Destroy and let do_aggregate recreate it.
    if (_key_set) { _key_set.reset(); }
  }

  void create_key_set(table_view const& key_table, rmm::cuda_stream_view stream)
  {
    auto preprocessed_keys = cudf::detail::row::hash::preprocessed_table::create(key_table, stream);
    auto const comparator  = cudf::detail::row::equality::self_comparator{preprocessed_keys};
    auto const row_hash    = cudf::detail::row::hash::row_hasher{std::move(preprocessed_keys)};
    auto const has_null    = cudf::nullate::DYNAMIC{cudf::has_nested_nulls(key_table)};
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

  // Re-insert existing staging rows [0, count) into a freshly created key set.
  void reinsert_existing_keys(size_type count,
                              table_view const& full_key_view,
                              bitmask_type const* row_bitmask,
                              rmm::cuda_stream_view stream)
  {
    auto preprocessed_keys =
      cudf::detail::row::hash::preprocessed_table::create(full_key_view, stream);
    auto const comparator  = cudf::detail::row::equality::self_comparator{preprocessed_keys};
    auto const row_hash    = cudf::detail::row::hash::row_hasher{std::move(preprocessed_keys)};
    auto const has_null    = cudf::nullate::DYNAMIC{cudf::has_nested_nulls(full_key_view)};
    auto const d_row_hash  = row_hash.device_hasher(has_null);
    auto const d_row_equal = comparator.equal_to<false>(has_null, null_equality::EQUAL);
    auto const hasher      = detail::hash::row_hasher_with_cache_t{d_row_hash};

    auto set_ref = _key_set->ref(cuco::op::insert_and_find)
                     .rebind_key_eq(d_row_equal)
                     .rebind_hash_function(hasher);

    // Insert all existing rows; the target indices are discarded (we only care about
    // repopulating the set). Reuse compute_target_indices_fn with staging_offset=0.
    rmm::device_uvector<size_type> discard(count, stream);
    thrust::transform(rmm::exec_policy_nosync(stream),
                      cuda::counting_iterator<size_type>(0),
                      cuda::counting_iterator<size_type>(count),
                      discard.begin(),
                      compute_target_indices_fn<decltype(set_ref)>{set_ref, row_bitmask, 0});
  }

  void copy_keys_to_staging(table_view const& batch_keys,
                            size_type staging_offset,
                            rmm::cuda_stream_view stream)
  {
    // Collect temp bitmask buffers so they survive async kernels launched below.
    std::vector<rmm::device_buffer> temp_masks;

    for (size_type c = 0; c < num_keys(); ++c) {
      auto const& src = batch_keys.column(c);
      auto& dst       = _key_columns[c];

      auto dst_view        = dst->mutable_view();
      auto const elem_size = cudf::size_of(src.type());
      CUDF_CUDA_TRY(cudf::detail::memcpy_async(
        static_cast<uint8_t*>(dst_view.head()) + elem_size * staging_offset,
        src.head<uint8_t>() + elem_size * src.offset(),
        elem_size * src.size(),
        stream));

      cudf::set_null_mask(
        dst_view.null_mask(), staging_offset, staging_offset + src.size(), true, stream);

      if (src.nullable()) {
        _has_nullable_keys = true;
        temp_masks.push_back(cudf::copy_bitmask(src, stream));
        auto const* temp_ptr = static_cast<bitmask_type const*>(temp_masks.back().data());
        thrust::for_each_n(rmm::exec_policy_nosync(stream),
                           cuda::counting_iterator<size_type>(0),
                           src.size(),
                           copy_null_mask_fn{dst_view.null_mask(), temp_ptr, staging_offset});
      }
    }
  }

  [[nodiscard]] table_view key_table_view(size_type num_rows) const
  {
    std::vector<column_view> views;
    views.reserve(_key_columns.size());
    for (auto const& col : _key_columns) {
      auto const& cv = col->view();
      if (_has_nullable_keys) {
        views.push_back(cudf::slice(cv, {0, num_rows})[0]);
      } else {
        views.push_back(column_view{cv.type(), num_rows, cv.head(), nullptr, 0});
      }
    }
    return table_view{views};
  }

  // Compute target indices for a batch of rows by inserting into the hash set.
  // Returns the target indices and the row bitmask buffer (to keep it alive).
  std::pair<rmm::device_uvector<size_type>, rmm::device_buffer> compute_batch_target_indices(
    size_type batch_size,
    size_type staging_offset,
    table_view const& full_key_view,
    rmm::cuda_stream_view stream)
  {
    auto preprocessed_keys =
      cudf::detail::row::hash::preprocessed_table::create(full_key_view, stream);
    auto const comparator = cudf::detail::row::equality::self_comparator{preprocessed_keys};
    auto const row_hash   = cudf::detail::row::hash::row_hasher{std::move(preprocessed_keys)};
    auto const has_null   = cudf::nullate::DYNAMIC{cudf::has_nested_nulls(full_key_view)};
    auto const d_row_hash = row_hash.device_hasher(has_null);

    auto const skip_rows_with_nulls =
      cudf::has_nulls(full_key_view) && _null_handling == null_policy::EXCLUDE;
    auto row_bitmask_buffer =
      skip_rows_with_nulls ? compute_row_bitmask(full_key_view, stream) : rmm::device_buffer{};
    auto const* row_bitmask =
      skip_rows_with_nulls ? static_cast<bitmask_type const*>(row_bitmask_buffer.data()) : nullptr;

    rmm::device_uvector<size_type> target_indices(batch_size, stream);
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
      compute_target_indices_fn<decltype(set_ref)>{set_ref, row_bitmask, staging_offset});

    return {std::move(target_indices), std::move(row_bitmask_buffer)};
  }

  void update_unique_key_count(size_type visible_rows, rmm::cuda_stream_view stream)
  {
    auto const populated = detail::hash::extract_populated_keys(
      *_key_set, visible_rows, stream, cudf::get_current_device_resource_ref());
    _num_unique_keys = static_cast<size_type>(populated.size());

    CUDF_EXPECTS(_num_unique_keys <= _max_groups,
                 "Unique keys (" + std::to_string(_num_unique_keys) + ") exceeded max_groups (" +
                   std::to_string(_max_groups) + ").");
  }

  void do_aggregate(table_view const& data,
                    rmm::cuda_stream_view stream,
                    rmm::device_async_resource_ref mr)
  {
    auto const batch_size = data.num_rows();
    if (batch_size == 0) { return; }

    CUDF_EXPECTS(batch_size <= _max_groups,
                 "Batch size (" + std::to_string(batch_size) + ") exceeds max_groups (" +
                   std::to_string(_max_groups) + ").",
                 std::invalid_argument);

    if (!_initialized) { initialize(data, stream, mr); }

    std::vector<column_view> batch_key_cols;
    batch_key_cols.reserve(num_keys());
    for (auto idx : _key_indices) {
      batch_key_cols.push_back(data.column(idx));
    }
    auto const batch_keys = table_view{batch_key_cols};

    ensure_staging_capacity(batch_size, stream, mr);

    auto const staging_offset = _staging_end;
    copy_keys_to_staging(batch_keys, staging_offset, stream);
    _staging_end += batch_size;

    auto const visible_rows  = _staging_end;
    auto const full_key_view = key_table_view(visible_rows);

    bool const set_was_reset = !_key_set;
    if (!_key_set) { create_key_set(full_key_view, stream); }

    // If the set was recreated (e.g., after staging buffer growth), re-insert old keys.
    if (set_was_reset && staging_offset > 0) {
      auto const skip_rows_with_nulls =
        cudf::has_nulls(full_key_view) && _null_handling == null_policy::EXCLUDE;
      auto const bitmask_buf =
        skip_rows_with_nulls ? compute_row_bitmask(full_key_view, stream) : rmm::device_buffer{};
      auto const* bitmask =
        skip_rows_with_nulls ? static_cast<bitmask_type const*>(bitmask_buf.data()) : nullptr;
      reinsert_existing_keys(staging_offset, full_key_view, bitmask, stream);
    }

    auto [target_indices, bitmask_buf] =
      compute_batch_target_indices(batch_size, staging_offset, full_key_view, stream);

    auto agg_requests         = build_aggregation_requests(_requests_clone, data);
    auto [values_view,
          agg_kinds_batch,
          agg_objects_batch,
          is_intermediate_batch,
          has_compound_batch] = detail::hash::extract_single_pass_aggs(agg_requests, stream);

    auto const d_values    = table_device_view::create(values_view, stream);
    auto d_results_ptr     = mutable_table_device_view::create(*_agg_results, stream);
    auto const d_agg_kinds = cudf::detail::make_device_uvector_async(
      _agg_kinds, stream, cudf::get_current_device_resource_ref());

    auto const num_agg_cols = static_cast<int64_t>(_agg_kinds.size());
    thrust::for_each_n(rmm::exec_policy_nosync(stream),
                       cuda::counting_iterator<int64_t>(0),
                       static_cast<int64_t>(batch_size) * num_agg_cols,
                       detail::hash::compute_single_pass_aggs_dense_output_fn{
                         target_indices.begin(), d_agg_kinds.data(), *d_values, *d_results_ptr});

    update_unique_key_count(visible_rows, stream);
  }

  [[nodiscard]] std::pair<std::unique_ptr<table>, std::vector<aggregation_result>> do_finalize(
    rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const
  {
    CUDF_EXPECTS(_initialized, "Cannot finalize streaming_groupby with no accumulated data.");

    auto const visible_rows  = _staging_end;
    auto const full_key_view = key_table_view(visible_rows);

    auto populated = detail::hash::extract_populated_keys(*_key_set, visible_rows, stream, mr);

    auto keys_gathered = cudf::detail::gather(full_key_view,
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

  void do_merge(impl const& other, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
  {
    if (!other._initialized || !other.has_state()) { return; }
    CUDF_EXPECTS(_initialized,
                 "Cannot merge into an uninitialized streaming_groupby. "
                 "Call aggregate() at least once before merge().");

    // Gather other's populated keys and raw intermediate agg results.
    auto const other_visible  = other._staging_end;
    auto const other_key_view = other.key_table_view(other_visible);

    auto other_populated =
      detail::hash::extract_populated_keys(*other._key_set, other_visible, stream, mr);

    auto const num_other_groups = static_cast<size_type>(other_populated.size());
    if (num_other_groups == 0) { return; }

    auto other_keys = cudf::detail::gather(other_key_view,
                                           other_populated,
                                           out_of_bounds_policy::DONT_CHECK,
                                           cudf::negative_index_policy::NOT_ALLOWED,
                                           stream,
                                           mr);

    auto other_aggs = cudf::detail::gather(other._agg_results->view(),
                                           other_populated,
                                           out_of_bounds_policy::DONT_CHECK,
                                           cudf::negative_index_policy::NOT_ALLOWED,
                                           stream,
                                           mr);

    // Copy other's keys into our staging buffer and insert into hash set.
    ensure_staging_capacity(num_other_groups, stream, mr);

    auto const staging_offset = _staging_end;
    copy_keys_to_staging(other_keys->view(), staging_offset, stream);
    _staging_end += num_other_groups;

    auto const visible_rows  = _staging_end;
    auto const full_key_view = key_table_view(visible_rows);

    bool const set_was_reset = !_key_set;
    if (!_key_set) { create_key_set(full_key_view, stream); }

    if (set_was_reset && staging_offset > 0) {
      auto const skip_nulls =
        cudf::has_nulls(full_key_view) && _null_handling == null_policy::EXCLUDE;
      auto const bitmask_buf =
        skip_nulls ? compute_row_bitmask(full_key_view, stream) : rmm::device_buffer{};
      auto const* bitmask =
        skip_nulls ? static_cast<bitmask_type const*>(bitmask_buf.data()) : nullptr;
      reinsert_existing_keys(staging_offset, full_key_view, bitmask, stream);
    }

    auto [target_indices, bitmask_buf] =
      compute_batch_target_indices(num_other_groups, staging_offset, full_key_view, stream);

    // Merge intermediates using merge-aware aggregation (adds counts, doesn't re-square, etc.)
    auto const d_source    = table_device_view::create(other_aggs->view(), stream);
    auto d_target          = mutable_table_device_view::create(*_agg_results, stream);
    auto const d_agg_kinds = cudf::detail::make_device_uvector_async(
      _agg_kinds, stream, cudf::get_current_device_resource_ref());

    auto const num_agg_cols = static_cast<int64_t>(_agg_kinds.size());
    thrust::for_each_n(
      rmm::exec_policy_nosync(stream),
      thrust::make_counting_iterator(int64_t{0}),
      static_cast<int64_t>(num_other_groups) * num_agg_cols,
      merge_single_pass_aggs_fn{target_indices.begin(), d_agg_kinds.data(), *d_source, *d_target});

    update_unique_key_count(visible_rows, stream);
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
  _impl->do_aggregate(data, stream, mr);
}

void streaming_groupby::merge(streaming_groupby const& other,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  _impl->do_merge(*other._impl, stream, mr);
}

std::pair<std::unique_ptr<table>, std::vector<aggregation_result>> streaming_groupby::finalize(
  rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const
{
  CUDF_FUNC_RANGE();
  return _impl->do_finalize(stream, mr);
}

}  // namespace cudf::groupby
