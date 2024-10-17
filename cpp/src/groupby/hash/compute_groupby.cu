/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include "compute_groupby.hpp"
#include "compute_single_pass_aggs.hpp"
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
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <cuco/static_set.cuh>

#include <iterator>
#include <memory>

namespace cudf::groupby::detail::hash {
template <typename SetType>
rmm::device_uvector<size_type> extract_populated_keys(SetType const& key_set,
                                                      size_type num_keys,
                                                      rmm::cuda_stream_view stream)
{
  rmm::device_uvector<size_type> populated_keys(num_keys, stream);
  auto const keys_end = key_set.retrieve_all(populated_keys.begin(), stream.value());

  populated_keys.resize(std::distance(populated_keys.begin(), keys_end), stream);
  return populated_keys;
}

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

  // Cache of sparse results where the location of aggregate value in each
  // column is indexed by the hash set
  cudf::detail::result_cache sparse_results(requests.size());

  auto const set = cuco::static_set{
    num_keys,
    cudf::detail::CUCO_DESIRED_LOAD_FACTOR,  // 50% load factor
    cuco::empty_key{cudf::detail::CUDF_SIZE_TYPE_SENTINEL},
    d_row_equal,
    probing_scheme_t{d_row_hash},
    cuco::thread_scope_device,
    cuco::storage<GROUPBY_WINDOW_SIZE>{},
    cudf::detail::cuco_allocator<char>{rmm::mr::polymorphic_allocator<char>{}, stream},
    stream.value()};

  auto row_bitmask =
    skip_rows_with_nulls
      ? cudf::bitmask_and(keys, stream, cudf::get_current_device_resource_ref()).first
      : rmm::device_buffer{};

  // Compute all single pass aggs first
  compute_single_pass_aggs(num_keys,
                           skip_rows_with_nulls,
                           static_cast<bitmask_type*>(row_bitmask.data()),
                           set.ref(cuco::insert_and_find),
                           requests,
                           &sparse_results,
                           stream);

  // Extract the populated indices from the hash set and create a gather map.
  // Gathering using this map from sparse results will give dense results.
  auto gather_map = extract_populated_keys(set, keys.num_rows(), stream);

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

template rmm::device_uvector<size_type> extract_populated_keys<global_set_t>(
  global_set_t const& key_set, size_type num_keys, rmm::cuda_stream_view stream);

template rmm::device_uvector<size_type> extract_populated_keys<nullable_global_set_t>(
  nullable_global_set_t const& key_set, size_type num_keys, rmm::cuda_stream_view stream);

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
