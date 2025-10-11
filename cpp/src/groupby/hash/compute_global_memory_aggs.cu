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
#include "output_utils.hpp"
#include "single_pass_functors.cuh"

#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/for_each.h>
#include <thrust/tabulate.h>

namespace cudf::groupby::detail::hash {

namespace {

/**
 * @brief Compute and return an array mapping each input row to its corresponding key index in
 * the input keys table.
 *
 * @tparam SetType Type of the key hash set
 * @param row_bitmask Bitmask indicating which rows in the input keys table are valid
 * @param set_ref Key hash set
 * @param num_rows Number of rows in the input keys table
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return A device vector mapping each input row to its key index
 */
template <typename SetRef>
rmm::device_uvector<size_type> compute_matching_keys(bitmask_type const* row_bitmask,
                                                     SetRef set_ref,
                                                     size_type num_rows,
                                                     rmm::cuda_stream_view stream)
{
  // Mapping from each row in the input key/value into the indices of the key.
  rmm::device_uvector<size_type> key_indices(num_rows, stream);

  // Need to set to sentinel value for rows that are null (if any).
  // The sentinel value will then be used to identify null rows instead of using the bitmask.
  thrust::tabulate(rmm::exec_policy_nosync(stream),
                   key_indices.begin(),
                   key_indices.end(),
                   [set_ref, row_bitmask] __device__(size_type const idx) mutable {
                     if (!row_bitmask || cudf::bit_is_set(row_bitmask, idx)) {
                       return *set_ref.insert_and_find(idx).first;
                     }
                     return cudf::detail::CUDF_SIZE_TYPE_SENTINEL;
                   });
  return key_indices;
}

}  // namespace

template <typename SetType>
std::pair<std::unique_ptr<table>, rmm::device_uvector<size_type>> compute_global_memory_aggs(
  bitmask_type const* row_bitmask,
  table_view const& values,
  SetType const& key_set,
  host_span<aggregation::Kind const> h_agg_kinds,
  device_span<aggregation::Kind const> d_agg_kinds,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const num_rows = values.num_rows();
  auto matching_keys =
    compute_matching_keys(row_bitmask, key_set.ref(cuco::op::insert_and_find), num_rows, stream);
  auto [unique_keys, key_transform_map] = extract_populated_keys(key_set, num_rows, stream, mr);
  auto const target_indices = compute_target_indices(matching_keys, key_transform_map, stream);
  matching_keys     = rmm::device_uvector<size_type>{0, stream};  // done, free up memory early
  key_transform_map = rmm::device_uvector<size_type>{0, stream};  // done, free up memory early

  auto const d_values = table_device_view::create(values, stream);
  auto agg_results    = create_results_table(
    static_cast<size_type>(unique_keys.size()), values, h_agg_kinds, stream, mr);
  auto d_results_ptr = mutable_table_device_view::create(*agg_results, stream);

  thrust::for_each_n(rmm::exec_policy_nosync(stream),
                     thrust::make_counting_iterator(int64_t{0}),
                     num_rows * static_cast<int64_t>(h_agg_kinds.size()),
                     compute_single_pass_aggs_fn{
                       target_indices.begin(), d_agg_kinds.data(), *d_values, *d_results_ptr});

  return {std::move(agg_results), std::move(unique_keys)};
}

template std::pair<std::unique_ptr<table>, rmm::device_uvector<size_type>>
compute_global_memory_aggs<global_set_t>(bitmask_type const* row_bitmask,
                                         table_view const& values,
                                         global_set_t const& key_set,
                                         host_span<aggregation::Kind const> h_agg_kinds,
                                         device_span<aggregation::Kind const> d_agg_kinds,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr);

template std::pair<std::unique_ptr<table>, rmm::device_uvector<size_type>>
compute_global_memory_aggs<nullable_global_set_t>(bitmask_type const* row_bitmask,
                                                  table_view const& values,
                                                  nullable_global_set_t const& key_set,
                                                  host_span<aggregation::Kind const> h_agg_kinds,
                                                  device_span<aggregation::Kind const> d_agg_kinds,
                                                  rmm::cuda_stream_view stream,
                                                  rmm::device_async_resource_ref mr);

}  // namespace cudf::groupby::detail::hash
