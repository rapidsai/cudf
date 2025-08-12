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
// This is a heuristic to reduce memory read when the keys table is hashes twice.
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

  [[maybe_unused]] auto const [cached_hashes, cached_hashes_data] =
    [&]() -> std::pair<rmm::device_uvector<hash_value_type>, hash_value_type const*> {
    auto const num_columns =
      std::accumulate(keys.begin(), keys.end(), 0, [](int count, column_view const& col) {
        return count + count_nested_columns(col);
      });

    if (num_columns <= HASH_CACHING_THRESHOLD) {
      return {rmm::device_uvector<hash_value_type>{0, stream}, nullptr};
    }

    rmm::device_uvector<hash_value_type> hashes(num_keys, stream);
    thrust::tabulate(rmm::exec_policy_nosync(stream),
                     hashes.begin(),
                     hashes.end(),
                     [d_row_hash] __device__(size_type const idx) { return d_row_hash(idx); });
    auto hashes_data = hashes.data();
    return {std::move(hashes), hashes_data};
  }();

  // Cache of sparse results where the location of aggregate value in each
  // column is indexed by the hash set
  cudf::detail::result_cache sparse_results(requests.size());

  auto set = cuco::static_set{
    cuco::extent<int64_t>{num_keys},
    cudf::detail::CUCO_DESIRED_LOAD_FACTOR,  // 50% load factor
    cuco::empty_key{cudf::detail::CUDF_SIZE_TYPE_SENTINEL},
    d_row_equal,
    probing_scheme_t{row_hasher_with_cache_t{d_row_hash, cached_hashes_data}},
    cuco::thread_scope_device,
    cuco::storage<GROUPBY_BUCKET_SIZE>{},
    cudf::detail::cuco_allocator<char>{rmm::mr::polymorphic_allocator<char>{}, stream},
    stream.value()};

  auto row_bitmask =
    skip_rows_with_nulls
      ? cudf::bitmask_and(keys, stream, cudf::get_current_device_resource_ref()).first
      : rmm::device_buffer{};

  // Compute all single pass aggs first
  auto gather_map = compute_aggregations(num_keys,
                                         skip_rows_with_nulls,
                                         static_cast<bitmask_type*>(row_bitmask.data()),
                                         set,
                                         requests,
                                         &sparse_results,
                                         stream);

  // Compact all results from sparse_results and insert into cache
  sparse_to_dense_results(requests,
                          &sparse_results,
                          cache,
                          gather_map,
                          set.ref(cuco::find),
                          static_cast<bitmask_type*>(row_bitmask.data()),
                          stream,
                          mr);

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
