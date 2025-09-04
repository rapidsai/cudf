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

#pragma once

#include "compute_global_memory_aggs.hpp"
#include "compute_mapping_indices.hpp"
#include "compute_shared_memory_aggs.hpp"
#include "compute_single_pass_aggs.hpp"
#include "create_output.hpp"
#include "flatten_single_pass_aggs.hpp"
#include "helpers.cuh"
#include "single_pass_functors.cuh"

#include <cudf/detail/utilities/cuda.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/table/table_device_view.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuco/static_set.cuh>
#include <thrust/for_each.h>

namespace cudf::groupby::detail::hash {

template <typename SetType>
std::pair<rmm::device_uvector<size_type>, bool> compute_single_pass_aggs(
  SetType& global_set,
  bitmask_type const* row_bitmask,
  host_span<aggregation_request const> requests,
  cudf::detail::result_cache* cache,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  // Collect the single-pass aggregations that can be processed in this function.
  // The compound aggregations that require multiple passes will be handled separately later on.
  auto const [spass_values, spass_agg_kinds, spass_aggs, has_compound_aggs] =
    flatten_single_pass_aggs(requests, stream);
  auto const d_spass_agg_kinds = cudf::detail::make_device_uvector_async(
    spass_agg_kinds, stream, rmm::mr::get_current_device_resource());
  auto const num_rows = spass_values.num_rows();

  // Grid size used for both index mapping and shared memory aggregation kernels.
  auto const grid_size = [&] {
    auto const max_blocks_mapping =
      max_active_blocks_mapping_kernel<typename SetType::ref_type<cuco::insert_and_find_tag>>();
    auto const max_blocks_aggs = max_active_blocks_shmem_aggs_kernel();
    // We launch the same grid size for both kernels, thus we need to take the minimum of the two.
    auto const max_blocks    = std::min(max_blocks_mapping, max_blocks_aggs);
    auto const max_grid_size = max_blocks * cudf::detail::num_multiprocessors();
    auto const num_blocks    = cudf::util::div_rounding_up_safe(num_rows, GROUPBY_BLOCK_SIZE);

    std::string const s = "Exception: max block mapping: " + std::to_string(max_blocks_mapping) +
                          ", max block aggs: " + std::to_string(max_blocks_aggs) +
                          ", max grid size: " + std::to_string(max_grid_size) +
                          ", num blocks: " + std::to_string(num_blocks) + "\n";

    if (num_blocks <= 0 || max_grid_size <= 0) { throw std::runtime_error(s.c_str()); }

    return std::min(max_grid_size, num_blocks);
  }();

  // Just to make sure everything is fine, since zero grid_size is zero only if the input is empty,
  // which should have already been handled before reaching here.
  CUDF_EXPECTS(grid_size > 0, "Invalid grid size computation.");

  auto const [can_run_by_shared_mem_kernel, available_shmem_size] =
    is_shared_memory_compatible(spass_agg_kinds, spass_values, grid_size);

  // Performs naive global memory aggregations when the workload is not compatible with shared
  // memory, such as when aggregating dictionary columns, when there is insufficient dynamic
  // shared memory for shared memory aggregations, or when SUM_WITH_OVERFLOW aggregations are
  // present.
  auto const run_aggs_by_global_mem_kernel = [&] {
    auto [spass_results, unique_key_indices] = compute_global_memory_aggs(
      row_bitmask, spass_values, global_set, spass_agg_kinds, d_spass_agg_kinds, stream, mr);
    collect_output_to_cache(spass_values, spass_aggs, spass_results, cache, stream);
    return std::pair{std::move(unique_key_indices), has_compound_aggs};
  };
  if (!can_run_by_shared_mem_kernel) { return run_aggs_by_global_mem_kernel(); }

  // Maps from the global row index of the input table to its block-wise rank.
  rmm::device_uvector<size_type> local_mapping_indices(num_rows, stream);
  // Maps from the block-wise rank to the row index of result table.
  rmm::device_uvector<size_type> global_mapping_indices(grid_size * GROUPBY_SHM_MAX_ELEMENTS,
                                                        stream);
  // Initialize it with a sentinel value, so later we can identify which ones are unused and which
  // ones need to be updated.
  thrust::uninitialized_fill(rmm::exec_policy_nosync(stream),
                             global_mapping_indices.begin(),
                             global_mapping_indices.end(),
                             cudf::detail::CUDF_SIZE_TYPE_SENTINEL);
  // Compute the cardinality (the number of unique keys) encounter by each thread block.
  rmm::device_uvector<size_type> block_cardinality(grid_size, stream);

  auto set_ref_insert = global_set.ref(cuco::op::insert_and_find);
  compute_mapping_indices(grid_size,
                          num_rows,
                          set_ref_insert,
                          row_bitmask,
                          local_mapping_indices.data(),
                          global_mapping_indices.data(),
                          block_cardinality.data(),
                          stream);

  // For the thread blocks that need fallback to the code path using global memory aggregation,
  // we need to collect these block ids.
  auto const [num_fallback_blocks, fallback_blocks] =
    find_fallback_blocks(grid_size, block_cardinality.data(), stream);

  // If all blocks fallback, just run everything using the global memory aggregation code path.
  if (num_fallback_blocks == grid_size) { return run_aggs_by_global_mem_kernel(); }

  // Maps from each row to the index of its corresponding aggregation result.
  // This is only used when there are fallback blocks.
  auto target_indices = rmm::device_uvector<size_type>(num_fallback_blocks ? num_rows : 0, stream);

  if (num_fallback_blocks) {
    // We compute number of jumping steps when using all blocks (`num_strides`).
    // Then, we use only `num_fallback_blocks` to jump the same number of steps.
    auto const fallback_stride   = GROUPBY_BLOCK_SIZE * num_fallback_blocks;
    auto const full_stride       = GROUPBY_BLOCK_SIZE * grid_size;
    auto const num_strides       = util::div_rounding_up_safe(num_rows, full_stride);
    auto const num_fallback_rows = num_fallback_blocks * GROUPBY_BLOCK_SIZE * num_strides;

    // Find key indices for each input row, only for the rows processed by the fallback blocks.
    // This also insert all the missing key indices into the global set so we can have a complete
    // set of keys for the final output.
    thrust::for_each_n(rmm::exec_policy_nosync(stream),
                       thrust::make_counting_iterator(0),
                       num_fallback_rows,
                       [num_rows,
                        full_stride,
                        fallback_stride,
                        set_ref_insert,
                        target_indices  = target_indices.begin(),
                        fallback_blocks = fallback_blocks.begin(),
                        row_bitmask] __device__(auto const idx) mutable {
                         auto const idx_in_stride = idx % fallback_stride;
                         auto const thread_rank   = idx_in_stride % GROUPBY_BLOCK_SIZE;
                         auto const block_idx = fallback_blocks[idx_in_stride / GROUPBY_BLOCK_SIZE];
                         auto const row_idx   = full_stride * (idx / fallback_stride) +
                                              GROUPBY_BLOCK_SIZE * block_idx + thread_rank;
                         if (row_idx >= num_rows) { return; }
                         if (!row_bitmask || cudf::bit_is_set(row_bitmask, row_idx)) {
                           target_indices[row_idx] = *set_ref_insert.insert_and_find(row_idx).first;
                         }
                       });
  }

  auto [unique_key_indices, key_transform_map] =
    extract_populated_keys(global_set, num_rows, stream);

  // If there are fallback blocks, we need to transform the target indices for the fallback kernel.
  if (num_fallback_blocks) { transform_key_indices(target_indices, key_transform_map, stream); }

  // Now, update the target indices for computing aggregations using the shared memory kernel.
  thrust::for_each_n(
    rmm::exec_policy_nosync(stream),
    thrust::make_counting_iterator(0),
    grid_size * GROUPBY_BLOCK_SIZE,
    [key_transform_map      = key_transform_map.begin(),
     global_mapping_indices = global_mapping_indices.begin()] __device__(auto const idx) {
      auto const block_id    = idx / GROUPBY_BLOCK_SIZE;
      auto const thread_rank = idx % GROUPBY_BLOCK_SIZE;
      auto const mapping_idx = block_id * GROUPBY_SHM_MAX_ELEMENTS + thread_rank;
      auto const old_idx     = global_mapping_indices[mapping_idx];
      if (old_idx != cudf::detail::CUDF_SIZE_TYPE_SENTINEL) {
        global_mapping_indices[mapping_idx] = key_transform_map[old_idx];
      }
    });
  key_transform_map = rmm::device_uvector<size_type>{0, stream};  // done, free up memory

  auto const d_spass_values = table_device_view::create(spass_values, stream);
  auto spass_results        = create_results_table(
    static_cast<size_type>(unique_key_indices.size()), spass_values, spass_agg_kinds, stream, mr);
  auto d_results_ptr = mutable_table_device_view::create(*spass_results, stream);

  compute_shared_memory_aggs(grid_size,
                             available_shmem_size,
                             num_rows,
                             row_bitmask,
                             local_mapping_indices.data(),
                             global_mapping_indices.data(),
                             block_cardinality.data(),
                             *d_spass_values,
                             *d_results_ptr,
                             d_spass_agg_kinds.data(),
                             stream);

  // The shared memory groupby is designed so that each thread block can handle up to 128 unique
  // keys. When a block reaches this cardinality limit, shared memory becomes insufficient to store
  // the temporary aggregation results. In these situations, we must fallback to a global memory
  // aggregator to process the remaining aggregation requests.
  if (num_fallback_blocks) {
    // We only execute this kernel for the fallback blocks.
    auto const fallback_stride   = GROUPBY_BLOCK_SIZE * num_fallback_blocks;
    auto const full_stride       = GROUPBY_BLOCK_SIZE * grid_size;
    auto const num_strides       = util::div_rounding_up_safe(num_rows, full_stride);
    auto const num_fallback_rows = num_fallback_blocks * GROUPBY_BLOCK_SIZE * num_strides;

    thrust::for_each_n(rmm::exec_policy_nosync(stream),
                       thrust::make_counting_iterator(int64_t{0}),
                       static_cast<int64_t>(num_fallback_rows) * spass_values.num_columns(),
                       global_memory_fallback_fn{target_indices.begin(),
                                                 d_spass_agg_kinds.data(),
                                                 *d_spass_values,
                                                 *d_results_ptr,
                                                 fallback_blocks.data(),
                                                 fallback_stride,
                                                 full_stride,
                                                 num_strides,
                                                 num_fallback_rows,
                                                 num_rows});
  }

  collect_output_to_cache(spass_values, spass_aggs, spass_results, cache, stream);
  return {std::move(unique_key_indices), has_compound_aggs};
}
}  // namespace cudf::groupby::detail::hash
