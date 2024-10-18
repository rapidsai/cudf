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

#include "hash_compound_agg_finalizer.hpp"
#include "helpers.cuh"

#include <cudf/detail/aggregation/result_cache.hpp>
#include <cudf/groupby.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

namespace cudf::groupby::detail::hash {
template <typename SetRef>
void sparse_to_dense_results(host_span<aggregation_request const> requests,
                             cudf::detail::result_cache* sparse_results,
                             cudf::detail::result_cache* dense_results,
                             device_span<size_type const> gather_map,
                             SetRef set,
                             bitmask_type const* row_bitmask,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr)
{
  for (auto const& request : requests) {
    auto const& agg_v = request.aggregations;
    auto const& col   = request.values;

    // Given an aggregation, this will get the result from sparse_results and
    // convert and return dense, compacted result
    auto finalizer = hash_compound_agg_finalizer(
      col, sparse_results, dense_results, gather_map, set, row_bitmask, stream, mr);
    for (auto&& agg : agg_v) {
      agg->finalize(finalizer);
    }
  }
}

template void sparse_to_dense_results<hash_set_ref_t<cuco::find_tag>>(
  host_span<aggregation_request const> requests,
  cudf::detail::result_cache* sparse_results,
  cudf::detail::result_cache* dense_results,
  device_span<size_type const> gather_map,
  hash_set_ref_t<cuco::find_tag> set,
  bitmask_type const* row_bitmask,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

template void sparse_to_dense_results<nullable_hash_set_ref_t<cuco::find_tag>>(
  host_span<aggregation_request const> requests,
  cudf::detail::result_cache* sparse_results,
  cudf::detail::result_cache* dense_results,
  device_span<size_type const> gather_map,
  nullable_hash_set_ref_t<cuco::find_tag> set,
  bitmask_type const* row_bitmask,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);
}  // namespace cudf::groupby::detail::hash
