/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common.cuh"
#include "groupby/hash/extract_single_pass_aggs.hpp"
#include "groupby/hash/hash_compound_agg_finalizer.hpp"
#include "groupby/hash/output_utils.hpp"

#include <cudf/column/column_view.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/aggregation/result_cache.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace cudf::groupby {

namespace {

void validate_requests(host_span<streaming_aggregation_request const> requests)
{
  for (auto const& req : requests) {
    CUDF_EXPECTS(req.aggregation != nullptr,
                 "streaming_aggregation_request must have a non-null aggregation.",
                 std::invalid_argument);
    CUDF_EXPECTS(detail::is_hash_aggregation(req.aggregation->kind) &&
                   req.aggregation->kind != aggregation::ARGMIN &&
                   req.aggregation->kind != aggregation::ARGMAX,
                 "Unsupported aggregation kind for streaming groupby. "
                 "ARGMIN/ARGMAX are not supported because row indices are batch-local.",
                 std::invalid_argument);
  }
}

std::vector<aggregation_request> build_aggregation_requests(
  host_span<streaming_aggregation_request const> requests, table_view const& data)
{
  std::vector<aggregation_request> result;
  for (auto const& req : requests) {
    aggregation_request ar;
    ar.values = data.column(req.column_index);
    ar.aggregations.push_back(std::unique_ptr<groupby_aggregation>{
      dynamic_cast<groupby_aggregation*>(req.aggregation->clone().release())});
    result.push_back(std::move(ar));
  }
  return result;
}

}  // namespace

streaming_groupby::impl::impl(host_span<size_type const> key_indices,
                              host_span<streaming_aggregation_request const> requests,
                              size_type max_groups,
                              null_policy null_handling)
  : _max_groups{max_groups},
    _null_handling{null_handling},
    _d_agg_kinds{0, rmm::cuda_stream_default, cudf::get_current_device_resource_ref()}
{
  CUDF_EXPECTS(max_groups > 0, "max_groups must be positive.", std::invalid_argument);
  if (!key_indices.empty()) { _key_indices.assign(key_indices.begin(), key_indices.end()); }
  validate_requests(requests);

  for (auto const& req : requests) {
    streaming_aggregation_request clone;
    clone.column_index = req.column_index;
    clone.aggregation  = std::unique_ptr<groupby_aggregation>{
      dynamic_cast<groupby_aggregation*>(req.aggregation->clone().release())};
    _requests_clone.push_back(std::move(clone));
  }
}

void streaming_groupby::impl::initialize(table_view const& data, rmm::cuda_stream_view stream)
{
  auto const mr = cudf::get_current_device_resource_ref();

  // Detect nested key columns (struct, list) for comparator template dispatch.
  std::vector<column_view> key_cols;
  key_cols.reserve(_key_indices.size());
  for (auto idx : _key_indices) {
    key_cols.push_back(data.column(idx));
  }
  _has_nested_keys = cudf::detail::has_nested_columns(table_view{key_cols});

  auto agg_requests = build_aggregation_requests(_requests_clone, data);
  auto [values_view, agg_kinds_hv, agg_objects, is_intermediate, has_compound] =
    detail::hash::extract_single_pass_aggs(agg_requests, stream);

  _agg_kinds.assign(agg_kinds_hv.begin(), agg_kinds_hv.end());
  _agg_objects         = std::move(agg_objects);
  _is_agg_intermediate = std::move(is_intermediate);
  _has_compound_aggs   = has_compound;

  // Reject aggregation kinds that are unsupported in streaming after decomposition.
  for (auto k : _agg_kinds) {
    CUDF_EXPECTS(k != aggregation::ARGMIN && k != aggregation::ARGMAX,
                 "Streaming groupby does not support MIN/MAX on variable-width types "
                 "(internally decomposed to ARGMIN/ARGMAX).",
                 std::invalid_argument);
    CUDF_EXPECTS(k != aggregation::SUM_WITH_OVERFLOW,
                 "Streaming groupby does not support SUM_WITH_OVERFLOW "
                 "(struct intermediate cannot be merged across batches).",
                 std::invalid_argument);
  }

  _agg_results = detail::hash::create_results_table(
    _max_groups, values_view, _agg_kinds, _is_agg_intermediate, stream, mr);

  _d_agg_kinds = cudf::detail::make_device_uvector_async(_agg_kinds, stream, mr);

  // Build column-index mapping from expanded agg columns to source data columns.
  // Match by data pointer AND offset to handle sliced columns correctly.
  _value_col_indices.reserve(values_view.num_columns());
  for (size_type i = 0; i < values_view.num_columns(); ++i) {
    auto const& col = values_view.column(i);
    bool found      = false;
    for (size_type j = 0; j < static_cast<size_type>(agg_requests.size()); ++j) {
      auto const& req_col = agg_requests[j].values;
      if (req_col.head() == col.head() && req_col.offset() == col.offset()) {
        _value_col_indices.push_back(_requests_clone[j].column_index);
        found = true;
        break;
      }
    }
    CUDF_EXPECTS(found, "Internal error: expanded agg column not found in requests.");
  }

  size_type offset = 0;
  for (auto const& req : _requests_clone) {
    _request_first_agg_offset.push_back(offset);
    auto const values_type = data.column(req.column_index).type();
    auto const& ga         = dynamic_cast<groupby_aggregation const&>(*req.aggregation);
    offset += static_cast<size_type>(detail::hash::get_simple_aggregations(ga, values_type).size());
  }

  // Companion vectors: max_groups to accommodate encoded index range.
  _key_batch = std::make_unique<rmm::device_uvector<size_type>>(_max_groups, stream, mr);
  _key_row   = std::make_unique<rmm::device_uvector<size_type>>(_max_groups, stream, mr);

  // Group-to-encoded-index map: sized to max_groups (one entry per distinct key).
  _encoded_indices = std::make_unique<rmm::device_uvector<size_type>>(_max_groups, stream, mr);

  _initialized = true;
}

void streaming_groupby::impl::create_key_set(rmm::cuda_stream_view stream)
{
  _key_set = std::make_unique<streaming_set_t>(
    cuco::extent<int64_t>{static_cast<int64_t>(_max_groups)},
    cudf::detail::CUCO_DESIRED_LOAD_FACTOR,
    cuco::empty_key{cudf::detail::CUDF_SIZE_TYPE_SENTINEL},
    cuda::std::equal_to<size_type>{},
    streaming_probing_scheme_t{cudf::hashing::detail::default_hash<size_type>{}},
    cuco::thread_scope_device,
    cuco::storage<detail::hash::GROUPBY_BUCKET_SIZE>{},
    rmm::mr::polymorphic_allocator<char>{},
    stream.value());
}

void streaming_groupby::impl::update_nullable_state(table_view const& batch_keys)
{
  if (_has_nullable_keys) return;
  for (size_type c = 0; c < num_keys(); ++c) {
    if (batch_keys.column(c).nullable()) {
      _has_nullable_keys = true;
      return;
    }
  }
}

std::unique_ptr<table> streaming_groupby::impl::gather_agg_results(
  rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const
{
  return cudf::detail::gather(
    _agg_results->view(),
    device_span<size_type const>{_encoded_indices->data(),
                                 static_cast<std::size_t>(_distinct_count)},
    out_of_bounds_policy::DONT_CHECK,
    cudf::negative_index_policy::NOT_ALLOWED,
    stream,
    mr);
}

std::unique_ptr<table> streaming_groupby::impl::gather_distinct_keys(
  rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const
{
  if (_compacted_batches.empty()) { return std::make_unique<table>(); }
  if (_compacted_batches.size() == 1) {
    return std::make_unique<table>(_compacted_batches[0]->view(), stream, mr);
  }
  std::vector<table_view> distinct_keys(_compacted_batches.size());
  std::transform(_compacted_batches.begin(),
                 _compacted_batches.end(),
                 distinct_keys.begin(),
                 [](auto const& batch) { return batch->view(); });
  return cudf::concatenate(distinct_keys, stream, mr);
}

std::pair<std::unique_ptr<table>, std::vector<aggregation_result>>
streaming_groupby::impl::do_finalize(rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr) const
{
  CUDF_EXPECTS(_initialized, "Cannot finalize streaming_groupby with no accumulated data.");

  auto keys         = gather_distinct_keys(stream, mr);
  auto agg_gathered = gather_agg_results(stream, mr);

  if (_has_compound_aggs) {
    cudf::detail::result_cache cache(_agg_kinds.size());

    std::vector<aggregation_request> agg_requests_fin;
    for (size_t i = 0; i < _requests_clone.size(); ++i) {
      aggregation_request ar;
      ar.values = agg_gathered->view().column(_request_first_agg_offset[i]);
      ar.aggregations.push_back(std::unique_ptr<groupby_aggregation>{
        dynamic_cast<groupby_aggregation*>(_requests_clone[i].aggregation->clone().release())});
      agg_requests_fin.push_back(std::move(ar));
    }

    auto [values_view_fin, agg_kinds_fin, agg_objects_fin, is_intermediate_fin, has_compound_fin] =
      detail::hash::extract_single_pass_aggs(agg_requests_fin, stream);

    detail::hash::finalize_output(values_view_fin, agg_objects_fin, agg_gathered, &cache, stream);

    std::for_each(agg_requests_fin.begin(), agg_requests_fin.end(), [&](auto const& req) {
      auto const finalizer =
        detail::hash::hash_compound_agg_finalizer(req.values, &cache, nullptr, stream, mr);
      std::for_each(req.aggregations.begin(), req.aggregations.end(), [&](auto const& agg) {
        cudf::detail::aggregation_dispatcher(agg->kind, finalizer, *agg);
      });
    });

    return {std::move(keys),
            detail::extract_results(
              host_span<aggregation_request const>{agg_requests_fin}, cache, stream, mr)};
  }

  auto released_cols = agg_gathered->release();
  std::vector<aggregation_result> results;
  results.reserve(_requests_clone.size());
  for (auto& col : released_cols) {
    aggregation_result agg_result;
    agg_result.results.push_back(std::move(col));
    results.push_back(std::move(agg_result));
  }
  return {std::move(keys), std::move(results)};
}

streaming_groupby::impl::batch_insert_result streaming_groupby::impl::probe_and_insert(
  table_view const& batch_keys, rmm::cuda_stream_view stream)
{
  if (_has_nested_keys) {
    return probe_and_insert_impl<true>(batch_keys, stream);
  } else {
    return probe_and_insert_impl<false>(batch_keys, stream);
  }
}

// Constructor, destructor, and move ops require full impl definition.
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

// Private member functions defined here (requires full impl definition).
// The public API wrappers in streaming_groupby.cpp call these.
void streaming_groupby::do_aggregate(table_view const& data, rmm::cuda_stream_view stream)
{
  _impl->do_aggregate(data, stream);
}

void streaming_groupby::do_merge(streaming_groupby const& other, rmm::cuda_stream_view stream)
{
  _impl->do_merge(*other._impl, stream);
}

std::pair<std::unique_ptr<table>, std::vector<aggregation_result>> streaming_groupby::do_finalize(
  rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const
{
  return _impl->do_finalize(stream, mr);
}

}  // namespace cudf::groupby
