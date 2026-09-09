/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "compute_global_memory_aggs.hpp"
#include "compute_mapping_indices.hpp"
#include "compute_shared_memory_aggs.hpp"
#include "compute_single_pass_aggs.hpp"
#include "extract_single_pass_aggs.hpp"
#include "helpers.cuh"
#include "output_utils.hpp"

#include <cudf/detail/utilities/cuda.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/table/table_device_view.cuh>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuco/static_set.cuh>
#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/transform.h>

#include <algorithm>
#include <cstddef>
#include <limits>

namespace cudf::groupby::detail::hash {

template <typename SetType>
std::pair<rmm::device_uvector<size_type>, bool> compute_single_pass_aggs(
  SetType& global_set,
  bitmask_type const* row_bitmask,
  std::span<aggregation_request const> requests,
  cudf::detail::result_cache* cache,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  // Collect the single-pass aggregations that can be processed in this function.
  // The compound aggregations that require multiple passes will be handled separately later on.
  auto const [values, agg_kinds, aggs, is_agg_intermediate, has_compound_aggs] =
    extract_single_pass_aggs(requests, stream);
  auto const d_agg_kinds = cudf::detail::make_device_uvector_async(
    agg_kinds, stream, cudf::get_current_device_resource_ref());
  auto const num_rows = values.num_rows();

  // Performs naive global memory aggregations when the workload is not compatible with shared
  // memory, such as when aggregating dictionary columns, when there is insufficient dynamic
  // shared memory for shared memory aggregations, or when SUM_OVERFLOW aggregations are
  // present.
  auto const run_aggs_by_global_mem_kernel = [&] {
    auto [agg_results, unique_key_indices] = compute_global_memory_aggs(
      row_bitmask, values, global_set, agg_kinds, d_agg_kinds, is_agg_intermediate, stream, mr);
    finalize_output(values, aggs, agg_results, cache, stream);
    return std::pair{std::move(unique_key_indices), has_compound_aggs};
  };

  // Grid size used for both index mapping and shared memory aggregation kernels.
  auto const grid_size = [&] {
    auto const max_blocks_mapping =
      max_active_blocks_mapping_kernel<typename SetType::ref_type<cuco::insert_and_find_tag>>();
    auto const max_blocks_aggs = max_active_blocks_shmem_aggs_kernel();
    // We launch the same grid size for both kernels, thus we need to take the minimum of the two.
    auto const max_blocks    = std::min(max_blocks_mapping, max_blocks_aggs);
    auto const max_grid_size = max_blocks * cudf::detail::num_multiprocessors();
    auto const num_blocks    = cudf::util::div_rounding_up_safe(num_rows, GROUPBY_BLOCK_SIZE);
    return std::min(max_grid_size, num_blocks);
  }();

  // grid_size is zero means the shared memory kernel cannot be launched, since input cannot be
  // empty: empty input should already been handled before reaching here.
  if (grid_size <= 0) { return run_aggs_by_global_mem_kernel(); }

  auto const can_use_shared_mem_kernel =
    is_shared_memory_compatible(agg_kinds, values, grid_size).first;

  if (!can_use_shared_mem_kernel) { return run_aggs_by_global_mem_kernel(); }

  // Build the ordinary sparse row-to-key mapping once. Unlike the previous shared-memory path,
  // cardinality does not select an entirely different mapper or require a host-side fallback.
  rmm::device_uvector<size_type> matching_keys(num_rows, stream);
  auto mapper_set_ref = global_set.ref(cuco::op::insert_and_find);
  thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    cuda::counting_iterator<size_type>{0},
                    cuda::counting_iterator<size_type>{num_rows},
                    matching_keys.begin(),
                    [mapper_set_ref, row_bitmask] __device__(size_type const idx) mutable {
                      if (!row_bitmask || cudf::bit_is_set(row_bitmask, idx)) {
                        return *mapper_set_ref.insert_and_find(idx).first;
                      }
                      return cudf::detail::CUDF_SIZE_TYPE_SENTINEL;
                    });

  auto unique_keys           = extract_populated_keys(global_set, num_rows, stream, mr);
  auto const num_output_rows = static_cast<size_type>(unique_keys.size());
  auto key_transform_map     = compute_key_transform_map(
    num_rows, unique_keys, stream, cudf::get_current_device_resource_ref());
  auto agg_results =
    create_results_table(num_output_rows, values, agg_kinds, is_agg_intermediate, stream, mr);

  auto const d_spass_values = table_device_view::create(values, stream);
  auto d_results_ptr        = mutable_table_device_view::create(*agg_results, stream);

  if (num_output_rows > 0) {
    constexpr size_type block_size                   = 256;
    constexpr std::size_t block_private_budget_bytes = 16U * 1024U * 1024U;
    auto const launch_grid_size = cudf::util::div_rounding_up_safe(num_rows, block_size);
    auto const num_replicas =
      std::min(launch_grid_size, static_cast<size_type>(cudf::detail::num_multiprocessors()));

    // alloc_size includes result data and validity masks. Multiplying the one-replica allocation
    // is a conservative bound for the combined allocation because its buffers are rounded only
    // once. Division performs the budget test without overflowing size_t.
    auto const per_replica_bytes = agg_results->alloc_size();
    auto const private_rows_fit =
      num_output_rows <= std::numeric_limits<size_type>::max() / num_replicas;
    auto const private_bytes_fit =
      per_replica_bytes <= block_private_budget_bytes / static_cast<std::size_t>(num_replicas);

    if (private_rows_fit && private_bytes_fit) {
      auto block_private_results = create_results_table(num_output_rows * num_replicas,
                                                        values,
                                                        agg_kinds,
                                                        is_agg_intermediate,
                                                        stream,
                                                        cudf::get_current_device_resource_ref());
      auto d_block_private_results =
        mutable_table_device_view::create(*block_private_results, stream);
      compute_block_private_aggs(launch_grid_size,
                                 block_size,
                                 num_replicas,
                                 num_rows,
                                 matching_keys.data(),
                                 key_transform_map.data(),
                                 *d_spass_values,
                                 *d_results_ptr,
                                 *d_block_private_results,
                                 num_output_rows,
                                 d_agg_kinds.data(),
                                 stream);
    } else {
      // Reuse the mapping and dense output even when partials exceed the memory budget. This avoids
      // both a second mapping pass and the num_rows-sized sparse result table/gather.
      compute_mapped_global_aggs(launch_grid_size,
                                 block_size,
                                 num_rows,
                                 matching_keys.data(),
                                 key_transform_map.data(),
                                 *d_spass_values,
                                 *d_results_ptr,
                                 num_output_rows,
                                 d_agg_kinds.data(),
                                 stream);
    }
  }

  finalize_output(values, aggs, agg_results, cache, stream);
  return {std::move(unique_keys), has_compound_aggs};
}
}  // namespace cudf::groupby::detail::hash
