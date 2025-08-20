/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "compute_aggregations.hpp"
#include "compute_groupby.hpp"
#include "helpers.cuh"
#include "sparse_to_dense_results.hpp"

#include <cudf/detail/aggregation/aggregation.cuh>
#include <cudf/detail/aggregation/result_cache.hpp>
#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/groupby.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuco/static_set.cuh>
#include <thrust/tabulate.h>

#include <memory>

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
  // convert to int64_t to avoid potential overflow with large `keys`
  auto const num_keys = static_cast<int64_t>(keys.num_rows());

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

  // Cache of sparse results where the location of aggregate value in each
  // column is indexed by the hash set
  cudf::detail::result_cache sparse_results(requests.size());

  auto set = cuco::static_set{
    cuco::extent<int64_t>{num_keys},
    cudf::detail::CUCO_DESIRED_LOAD_FACTOR,  // 50% load factor
    cuco::empty_key{cudf::detail::CUDF_SIZE_TYPE_SENTINEL},
    d_row_equal,
    probing_scheme_t{row_hasher_with_cache_t{d_row_hash, cached_hashes.data()}},
    cuco::thread_scope_device,
    cuco::storage<GROUPBY_BUCKET_SIZE>{},
    cudf::detail::cuco_allocator<char>{rmm::mr::polymorphic_allocator<char>{}, stream},
    stream.value()};

  // Compute all single pass aggs first
  auto gather_map =
    compute_aggregations(num_keys, row_bitmask, set, requests, &sparse_results, stream);

  // Compact all results from sparse_results and insert into cache
  sparse_to_dense_results(
    requests, &sparse_results, cache, gather_map, set.ref(cuco::find), row_bitmask, stream, mr);

  return cudf::detail::gather(keys,
                              gather_map,
                              out_of_bounds_policy::DONT_CHECK,
                              cudf::detail::negative_index_policy::NOT_ALLOWED,
                              stream,
                              mr);
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
