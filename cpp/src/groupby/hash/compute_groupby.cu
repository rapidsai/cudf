/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compute_groupby.hpp"
#include "compute_single_pass_aggs.hpp"
#include "hash_compound_agg_finalizer.hpp"
#include "helpers.cuh"
#include "output_utils.hpp"

#include <cudf/detail/aggregation/aggregation.cuh>
#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuco/static_set.cuh>
#include <thrust/tabulate.h>

namespace cudf::groupby::detail::hash {

namespace {

// The number of columns in the keys table that will trigger caching of row hashes.
// This is a heuristic to reduce memory read when the keys table is hashed twice.
constexpr int HASH_CACHING_THRESHOLD = 4;

int count_nested_columns(column_view const& input)
{
  if (!is_nested(input.type())) { return 1; }

  // Count the current column too.
  return 1 + std::accumulate(
               input.child_begin(), input.child_end(), 0, [](int count, column_view const& child) {
                 return count + count_nested_columns(child);
               });
}

}  // namespace

template <typename Equal, typename Hash>
std::unique_ptr<table> compute_groupby(table_view const& keys,
                                       host_span<aggregation_request const> requests,
                                       bool skip_rows_with_nulls,
                                       Equal const& d_row_equal,
                                       Hash const& d_row_hash,
                                       cudf::detail::result_cache* cache,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  auto const num_keys = keys.num_rows();

  [[maybe_unused]] auto const [row_bitmask_data, row_bitmask] =
    [&]() -> std::pair<rmm::device_buffer, bitmask_type const*> {
    if (!skip_rows_with_nulls) { return {rmm::device_buffer{0, stream}, nullptr}; }

    if (keys.num_columns() == 1) {
      auto const& keys_col = keys.column(0);
      // Only use the input null mask directly if the keys table was not sliced.
      if (keys_col.offset() == 0) { return {rmm::device_buffer{0, stream}, keys_col.null_mask()}; }
      // If the keys table was sliced, we need to copy the null mask to ensure its first bit aligns
      // with the first row of the keys table.
      auto null_mask_data  = cudf::copy_bitmask(keys_col, stream);
      auto const null_mask = static_cast<bitmask_type const*>(null_mask_data.data());
      return {std::move(null_mask_data), null_mask};
    }

    auto [null_mask_data, null_count] = cudf::bitmask_and(keys, stream);
    if (null_count == 0) { return {rmm::device_buffer{0, stream}, nullptr}; }

    auto const null_mask = static_cast<bitmask_type const*>(null_mask_data.data());
    return {std::move(null_mask_data), null_mask};
  }();

  auto const cached_hashes = [&]() -> rmm::device_uvector<hash_value_type> {
    auto const num_columns =
      std::accumulate(keys.begin(), keys.end(), 0, [](int count, column_view const& col) {
        return count + count_nested_columns(col);
      });

    if (num_columns <= HASH_CACHING_THRESHOLD) {
      return rmm::device_uvector<hash_value_type>{0, stream};
    }

    rmm::device_uvector<hash_value_type> hashes(num_keys, stream);
    thrust::tabulate(rmm::exec_policy_nosync(stream),
                     hashes.begin(),
                     hashes.end(),
                     [d_row_hash, row_bitmask] __device__(size_type const idx) {
                       if (!row_bitmask || cudf::bit_is_set(row_bitmask, idx)) {
                         return d_row_hash(idx);
                       }
                       return hash_value_type{0};  // dummy value, as it will be unused
                     });
    return hashes;
  }();

  auto set =
    cuco::static_set{cuco::extent<int64_t>{static_cast<int64_t>(num_keys)},
                     cudf::detail::CUCO_DESIRED_LOAD_FACTOR,  // 50% load factor
                     cuco::empty_key{cudf::detail::CUDF_SIZE_TYPE_SENTINEL},
                     d_row_equal,
                     probing_scheme_t{row_hasher_with_cache_t{d_row_hash, cached_hashes.data()}},
                     cuco::thread_scope_device,
                     cuco::storage<GROUPBY_BUCKET_SIZE>{},
                     rmm::mr::polymorphic_allocator<char>{},
                     stream.value()};

  auto const gather_keys = [&](auto const& gather_map) {
    return cudf::detail::gather(keys,
                                gather_map,
                                out_of_bounds_policy::DONT_CHECK,
                                cudf::detail::negative_index_policy::NOT_ALLOWED,
                                stream,
                                mr);
  };

  // In case of no requests, we still need to generate a set of unique keys.
  if (requests.empty()) {
    thrust::for_each_n(
      rmm::exec_policy_nosync(stream),
      thrust::make_counting_iterator(0),
      num_keys,
      [set_ref = set.ref(cuco::op::insert), row_bitmask] __device__(size_type const idx) mutable {
        if (!row_bitmask || cudf::bit_is_set(row_bitmask, idx)) { set_ref.insert(idx); }
      });

    rmm::device_uvector<size_type> unique_key_indices(num_keys, stream);
    auto const keys_end       = set.retrieve_all(unique_key_indices.begin(), stream.value());
    auto const key_gather_map = device_span<size_type const>{
      unique_key_indices.data(),
      static_cast<std::size_t>(cuda::std::distance(unique_key_indices.begin(), keys_end))};
    return gather_keys(key_gather_map);
  }

  // Compute all single pass aggs first.
  auto const [key_gather_map, has_compound_aggs] =
    compute_single_pass_aggs(set, row_bitmask, requests, cache, stream, mr);

  if (has_compound_aggs) {
    for (auto const& request : requests) {
      auto const& agg_v = request.aggregations;
      auto const& col   = request.values;

      // The map to find the target output index for each input row is not always available due to
      // minimizing overhead. As such, there is no way for the finalizers to perform additional
      // aggregation operations. They can only compute their output using the previously computed
      // single-pass aggregations with linear transformations such as addition/multiplication (e.g.
      // for variance/stddev). In the future, if there are more compound aggregations that require
      // additional aggregation steps, we can revisit this design.
      auto finalizer = hash_compound_agg_finalizer(col, cache, row_bitmask, stream, mr);
      for (auto&& agg : agg_v) {
        agg->finalize(finalizer);
      }
    }
  }

  return gather_keys(key_gather_map);
}

template std::unique_ptr<table> compute_groupby<row_comparator_t, row_hash_t>(
  table_view const& keys,
  host_span<aggregation_request const> requests,
  bool skip_rows_with_nulls,
  row_comparator_t const& d_row_equal,
  row_hash_t const& d_row_hash,
  cudf::detail::result_cache* cache,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

template std::unique_ptr<table> compute_groupby<nullable_row_comparator_t, row_hash_t>(
  table_view const& keys,
  host_span<aggregation_request const> requests,
  bool skip_rows_with_nulls,
  nullable_row_comparator_t const& d_row_equal,
  row_hash_t const& d_row_hash,
  cudf::detail::result_cache* cache,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);
}  // namespace cudf::groupby::detail::hash
