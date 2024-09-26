/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include "compute_single_pass_aggs.hpp"
#include "helpers.cuh"
#include "sparse_to_dense_results.hpp"
#include "var_hash_functor.cuh"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/aggregation/result_cache.hpp>
#include <cudf/detail/binaryop.hpp>
#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/unary.hpp>
#include <cudf/groupby.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <cuco/static_set.cuh>

#include <memory>

namespace cudf::groupby::detail::hash {

/**
 * @brief Computes groupby using hash table.
 *
 * First, we create a hash table that stores the indices of unique rows in
 * `keys`. The upper limit on the number of values in this map is the number
 * of rows in `keys`.
 *
 * To store the results of aggregations, we create temporary sparse columns
 * which have the same size as input value columns. Using the hash map, we
 * determine the location within the sparse column to write the result of the
 * aggregation into.
 *
 * The sparse column results of all aggregations are stored into the cache
 * `sparse_results`. This enables the use of previously calculated results in
 * other aggregations.
 *
 * All the aggregations which can be computed in a single pass are computed
 * first, in a combined kernel. Then using these results, aggregations that
 * require multiple passes, will be computed.
 *
 * Finally, using the hash map, we generate a vector of indices of populated
 * values in sparse result columns. Then, for each aggregation originally
 * requested in `requests`, we gather sparse results into a column of dense
 * results using the aforementioned index vector. Dense results are stored into
 * the in/out parameter `cache`.
 */
template <typename Equal>
std::unique_ptr<table> compute_groupby(table_view const& keys,
                                       host_span<aggregation_request const> requests,
                                       cudf::detail::result_cache* cache,
                                       bool skip_key_rows_with_nulls,
                                       Equal const& d_row_equal,
                                       row_hash_t const& d_row_hash,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  // convert to int64_t to avoid potential overflow with large `keys`
  auto const num_keys = static_cast<int64_t>(keys.num_rows());

  // Cache of sparse results where the location of aggregate value in each
  // column is indexed by the hash set
  cudf::detail::result_cache sparse_results(requests.size());

  auto const set = cuco::static_set{
    cuco::extent<int64_t>{num_keys},
    cudf::detail::CUCO_DESIRED_LOAD_FACTOR,  // 50% occupancy
    cuco::empty_key{cudf::detail::CUDF_SIZE_TYPE_SENTINEL},
    d_row_equal,
    probing_scheme_t{d_row_hash},
    cuco::thread_scope_device,
    cuco::storage<GROUPBY_WINDOW_SIZE>{},
    cudf::detail::cuco_allocator<char>{rmm::mr::polymorphic_allocator<char>{}, stream},
    stream.value()};

  // Compute all single pass aggs first
  auto gather_map = compute_single_pass_aggs(
    keys, requests, &sparse_results, set, skip_key_rows_with_nulls, stream);

  // Compact all results from sparse_results and insert into cache
  sparse_to_dense_results(keys,
                          requests,
                          &sparse_results,
                          cache,
                          gather_map,
                          set.ref(cuco::find),
                          skip_key_rows_with_nulls,
                          stream,
                          mr);

  return cudf::detail::gather(keys,
                              gather_map,
                              out_of_bounds_policy::DONT_CHECK,
                              cudf::detail::negative_index_policy::NOT_ALLOWED,
                              stream,
                              mr);
}

template std::unique_ptr<table> compute_groupby<row_comparator_t>(
  table_view const& keys,
  host_span<aggregation_request const> requests,
  cudf::detail::result_cache* cache,
  bool skip_key_rows_with_nulls,
  row_comparator_t const& d_row_equal,
  row_hash_t const& d_row_hash,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

template std::unique_ptr<table> compute_groupby<nullable_row_comparator_t>(
  table_view const& keys,
  host_span<aggregation_request const> requests,
  cudf::detail::result_cache* cache,
  bool skip_key_rows_with_nulls,
  nullable_row_comparator_t const& d_row_equal,
  row_hash_t const& d_row_hash,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

}  // namespace cudf::groupby::detail::hash
