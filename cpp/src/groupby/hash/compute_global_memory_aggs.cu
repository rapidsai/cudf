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

#include "compute_global_memory_aggs.hpp"
#include "create_output.hpp"
#include "single_pass_functors.cuh"

#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/for_each.h>

namespace cudf::groupby::detail::hash {

template <typename SetType>
std::tuple<std::unique_ptr<table>, rmm::device_uvector<size_type>> compute_global_memory_aggs(
  bitmask_type const* row_bitmask,
  table_view const& values,
  SetType const& key_set,
  host_span<aggregation::Kind const> h_agg_kinds,
  device_span<aggregation::Kind const> d_agg_kinds,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const num_rows = values.num_rows();
  auto target_indices =
    compute_key_indices(row_bitmask, key_set.ref(cuco::op::insert_and_find), num_rows, stream);
  auto [unique_key_indices, key_transform_map] = extract_populated_keys(key_set, num_rows, stream);
  transform_key_indices(target_indices, key_transform_map, stream);
  key_transform_map = rmm::device_uvector<size_type>{0, stream};  // done, free up memory

  auto const d_values = table_device_view::create(values, stream);
  auto agg_results    = create_results_table(
    static_cast<size_type>(unique_key_indices.size()), values, h_agg_kinds, stream, mr);
  auto d_results_ptr = mutable_table_device_view::create(*agg_results, stream);

  thrust::for_each_n(rmm::exec_policy_nosync(stream),
                     thrust::make_counting_iterator(int64_t{0}),
                     num_rows * static_cast<int64_t>(h_agg_kinds.size()),
                     compute_single_pass_aggs_fn{
                       target_indices.begin(), d_agg_kinds.data(), *d_values, *d_results_ptr});

  return {std::move(agg_results), std::move(unique_key_indices)};
}

template std::tuple<std::unique_ptr<table>, rmm::device_uvector<size_type>>
compute_global_memory_aggs<global_set_t>(bitmask_type const* row_bitmask,
                                         table_view const& values,
                                         global_set_t const& key_set,
                                         host_span<aggregation::Kind const> h_agg_kinds,
                                         device_span<aggregation::Kind const> d_agg_kinds,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr);

template std::tuple<std::unique_ptr<table>, rmm::device_uvector<size_type>>
compute_global_memory_aggs<nullable_global_set_t>(bitmask_type const* row_bitmask,
                                                  table_view const& values,
                                                  nullable_global_set_t const& key_set,
                                                  host_span<aggregation::Kind const> h_agg_kinds,
                                                  device_span<aggregation::Kind const> d_agg_kinds,
                                                  rmm::cuda_stream_view stream,
                                                  rmm::device_async_resource_ref mr);

}  // namespace cudf::groupby::detail::hash
