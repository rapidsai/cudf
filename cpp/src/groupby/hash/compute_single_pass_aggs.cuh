/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
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
  auto const [values, agg_kinds, aggs, has_compound_aggs] =
    extract_single_pass_aggs(requests, stream);
  auto const d_agg_kinds = cudf::detail::make_device_uvector_async(
    agg_kinds, stream, cudf::get_current_device_resource_ref());
  auto const num_rows = values.num_rows();

  // Performs naive global memory aggregations when the workload is not compatible with shared
  // memory, such as when aggregating dictionary columns, when there is insufficient dynamic
  // shared memory for shared memory aggregations, or when SUM_WITH_OVERFLOW aggregations are
  // present.
  auto const run_aggs_by_global_mem_kernel = [&] {
    auto [agg_results, unique_key_indices] = compute_global_memory_aggs(
      row_bitmask, values, global_set, agg_kinds, d_agg_kinds, stream, mr);
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
    auto const num_blocks    = util::div_rounding_up_safe(num_rows, GROUPBY_BLOCK_SIZE);
    return std::min(max_grid_size, num_blocks);
  }();

  // grid_size is zero means the shared memory kernel cannot be launched, since input cannot be
  // empty: empty input should already been handled before reaching here.
  if (grid_size <= 0) { return run_aggs_by_global_mem_kernel(); }

  auto const [can_use_shared_mem_kernel, available_shmem_size] =
    is_shared_memory_compatible(agg_kinds, values, grid_size);

  if (!can_use_shared_mem_kernel) { return run_aggs_by_global_mem_kernel(); }

  // `grid_stride` is the number of rows processed by all blocks at one step (except the last step,
  // which may have fewer rows). Blocks will process by iterations, each iteration contains at least
  // one step and at most `num_strides` steps. This iterative approach is necessary to guarantee
  // that the block will not encounter too many unique keys such that the shared memory hash set
  // cannot hold.
  auto const grid_stride = GROUPBY_BLOCK_SIZE * grid_size;
  auto const num_strides = util::div_rounding_up_safe(num_rows, grid_stride);

  // Maps from the global row index of the input table to its block-wise thread rank.
  rmm::device_uvector<size_type> local_mapping_indices(num_rows, stream);
  // Maps from the block-wise thread rank to the row index of the output keys table,
  // need to store a separate map for each iteration.
  rmm::device_uvector<size_type> global_mapping_indices(
    num_strides * grid_size * GROUPBY_SHM_MAX_ELEMENTS, stream);

  // Some positions in `global_mapping_indices` will be unused.
  // We just initialize them with a sentinel value so later on we know to ignore them.
  // thrust::uninitialized_fill(rmm::exec_policy_nosync(stream),
  //                            global_mapping_indices.begin(),
  //                            global_mapping_indices.end(),
  //                            cudf::detail::CUDF_SIZE_TYPE_SENTINEL);

  // The upper-bound row index for each block to stop processing each iteration.
  rmm::device_uvector<size_type> block_row_ends(num_strides * grid_size, stream);

  // Compute the cardinality (the number of unique keys) encounter by each thread block.
  // The cardinality is not allowed to exceed `GROUPBY_SHM_MAX_ELEMENTS`.
  // Whenever the cardinality exceeds `GROUPBY_CARDINALITY_THRESHOLD` (it is still less than
  // `GROUPBY_SHM_MAX_ELEMENTS`), we push the upper-bound of the row index processed by the current
  // block into `block_row_ends` and reset the block's data (including cardinality) to start a new
  // iteration.
  rmm::device_uvector<size_type> block_cardinality(num_strides * grid_size, stream);

  compute_mapping_indices(grid_size,
                          num_rows,
                          global_set.ref(cuco::op::insert_and_find),
                          row_bitmask,
                          local_mapping_indices.data(),
                          global_mapping_indices.data(),
                          block_cardinality.data(),
                          block_row_ends.data(),
                          stream);

  auto unique_keys = extract_populated_keys(global_set, num_rows, stream, mr);

  // Now, update the target indices for computing aggregations using the shared memory kernel.
  // {
  //   auto const transform_map = compute_key_transform_map(
  //     num_rows, unique_keys, stream, cudf::get_current_device_resource_ref());
  //   thrust::transform(rmm::exec_policy_nosync(stream),
  //                     global_mapping_indices.begin(),
  //                     global_mapping_indices.end(),
  //                     global_mapping_indices.begin(),
  //                     [transform_map = transform_map.begin()] __device__(auto const idx) {
  //                       return idx != cudf::detail::CUDF_SIZE_TYPE_SENTINEL ? transform_map[idx]
  //                                                                           : idx;
  //                     });
  // }

  auto const d_spass_values = table_device_view::create(values, stream);
  auto agg_results =
    create_results_table(static_cast<size_type>(unique_keys.size()), values, agg_kinds, stream, mr);
  auto d_results_ptr = mutable_table_device_view::create(*agg_results, stream);

  auto const transform_map = compute_key_transform_map(
    num_rows, unique_keys, stream, cudf::get_current_device_resource_ref());
  compute_shared_memory_aggs(grid_size,
                             available_shmem_size,
                             num_rows,
                             local_mapping_indices.data(),
                             global_mapping_indices.data(),
                             transform_map.data(),
                             block_cardinality.data(),
                             block_row_ends.data(),
                             *d_spass_values,
                             *d_results_ptr,
                             d_agg_kinds.data(),
                             stream);

  finalize_output(values, aggs, agg_results, cache, stream);
  return {std::move(unique_keys), has_compound_aggs};
}
}  // namespace cudf::groupby::detail::hash
