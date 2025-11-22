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
    auto const num_blocks    = cudf::util::div_rounding_up_safe(num_rows, GROUPBY_BLOCK_SIZE);
    return std::min(max_grid_size, num_blocks);
  }();

  // grid_size is zero means the shared memory kernel cannot be launched, since input cannot be
  // empty: empty input should already been handled before reaching here.
  if (grid_size <= 0) { return run_aggs_by_global_mem_kernel(); }

  auto const [can_use_shared_mem_kernel, available_shmem_size] =
    is_shared_memory_compatible(agg_kinds, values, grid_size);

  if (!can_use_shared_mem_kernel) { return run_aggs_by_global_mem_kernel(); }

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

  // Maps from each row of the values table to the row index of its corresponding key in the input
  // keys table. This is only used when there are fallback blocks.
  auto matching_keys = [&] {
    if (num_fallback_blocks == 0) { return rmm::device_uvector<size_type>{0, stream}; }
    auto matching_keys = rmm::device_uvector<size_type>(num_rows, stream);
    thrust::uninitialized_fill(rmm::exec_policy_nosync(stream),
                               matching_keys.begin(),
                               matching_keys.end(),
                               cudf::detail::CUDF_SIZE_TYPE_SENTINEL);
    return matching_keys;
  }();

  if (num_fallback_blocks > 0) {
    auto const full_stride       = GROUPBY_BLOCK_SIZE * grid_size;
    auto const num_strides       = util::div_rounding_up_safe(num_rows, full_stride);
    auto const fallback_stride   = GROUPBY_BLOCK_SIZE * num_fallback_blocks;
    auto const num_fallback_rows = fallback_stride * num_strides;

    // This kernel finds the matching keys for the rows processed by the fallback blocks.
    // It also inserts all the missing keys (which were not inserted due to short-circuit in the
    // mapping indices kernel) into the global set, so we can have a complete set of keys for the
    // final output.
    //
    // Because the number of blocks is relatively small, we have to divide the input into
    // multiple "full" segments such that each thread processes one row for each segment in a
    // sequential manner. The length of each segment is computed as
    // `full_stride = grid_size * GROUPBY_BLOCK_SIZE`, and the number of segments is given by
    // `num_strides = round_up(num_rows / full_stride)`.
    //
    // For the fallback blocks, the number of rows processed by them is computed as
    // `num_fallback_rows = num_fallback_blocks * GROUPBY_BLOCK_SIZE * num_strides`.
    // This kernel has grid size of `num_fallback_rows`, and we want to map from the range
    // [0, num_fallback_rows) to the (non-contiguous) ranges of rows processed exactly by these
    // fallback blocks. In order to do so, we also divide such thread index range into the same
    // number of segments but with shorter length (computed by `fallback_stride`), referred to as
    // the "short" segments. It is guaranteed that "short" segments are always mapped one-to-one to
    // the "full" segments.
    thrust::for_each_n(rmm::exec_policy_nosync(stream),
                       thrust::make_counting_iterator(0),
                       num_fallback_rows,
                       [num_rows,
                        full_stride,
                        fallback_stride,
                        set_ref_insert,
                        matching_keys   = matching_keys.begin(),
                        fallback_blocks = fallback_blocks.begin(),
                        row_bitmask] __device__(auto const idx) mutable {
                         // Local index within the corresponding "full" segment.
                         // Since "short" segments map one-to-one to the "full" segment, here we
                         // use the "short" segment length `fallback_stride`.
                         auto const idx_in_stride = idx % fallback_stride;

                         // Rank of the thread within the corresponding fallback block.
                         auto const thread_rank = idx_in_stride % GROUPBY_BLOCK_SIZE;

                         // The index of the fallback block that the current thread is processing.
                         auto const block_idx = fallback_blocks[idx_in_stride / GROUPBY_BLOCK_SIZE];

                         // Compute the row index processed by the corresponding fallback block.
                         // Here, `full_stride * (idx / fallback_stride)` is the start offset of the
                         // current "full" segment, `GROUPBY_BLOCK_SIZE * block_idx` is the start
                         // offset of the corresponding fallback block within the "full" segment.
                         auto const row_idx = full_stride * (idx / fallback_stride) +
                                              GROUPBY_BLOCK_SIZE * block_idx + thread_rank;

                         if (row_idx >= num_rows) { return; }
                         if (!row_bitmask || cudf::bit_is_set(row_bitmask, row_idx)) {
                           matching_keys[row_idx] = *set_ref_insert.insert_and_find(row_idx).first;
                         }
                       });
  }

  auto unique_keys       = extract_populated_keys(global_set, num_rows, stream, mr);
  auto key_transform_map = compute_key_transform_map(
    num_rows, unique_keys, stream, cudf::get_current_device_resource_ref());

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
  // Find the target indices for computing aggregations using the shared memory kernel.
  // This is only used when there are fallback blocks.
  auto const target_indices = [&] {
    if (num_fallback_blocks == 0) { return rmm::device_uvector<size_type>{0, stream}; }
    return compute_target_indices(
      matching_keys, key_transform_map, stream, cudf::get_current_device_resource_ref());
  }();
  matching_keys     = rmm::device_uvector<size_type>{0, stream};  // done, free up memory early
  key_transform_map = rmm::device_uvector<size_type>{0, stream};  // done, free up memory early

  auto const d_spass_values = table_device_view::create(values, stream);
  auto agg_results =
    create_results_table(static_cast<size_type>(unique_keys.size()), values, agg_kinds, stream, mr);
  auto d_results_ptr = mutable_table_device_view::create(*agg_results, stream);

  compute_shared_memory_aggs(grid_size,
                             available_shmem_size,
                             num_rows,
                             row_bitmask,
                             local_mapping_indices.data(),
                             global_mapping_indices.data(),
                             block_cardinality.data(),
                             *d_spass_values,
                             *d_results_ptr,
                             d_agg_kinds.data(),
                             stream);

  // The shared memory groupby is designed so that each thread block can handle up to 128 unique
  // keys. When a block reaches this cardinality limit, shared memory becomes insufficient to store
  // the temporary aggregation results. In these situations, we must fallback to a global memory
  // aggregator to process the remaining aggregation requests.
  if (num_fallback_blocks > 0) {
    // We only execute this kernel for the fallback blocks.
    auto const fallback_stride   = GROUPBY_BLOCK_SIZE * num_fallback_blocks;
    auto const full_stride       = GROUPBY_BLOCK_SIZE * grid_size;
    auto const num_strides       = util::div_rounding_up_safe(num_rows, full_stride);
    auto const num_fallback_rows = num_fallback_blocks * GROUPBY_BLOCK_SIZE * num_strides;

    thrust::for_each_n(rmm::exec_policy_nosync(stream),
                       thrust::make_counting_iterator(int64_t{0}),
                       static_cast<int64_t>(num_fallback_rows) * values.num_columns(),
                       global_memory_fallback_fn{target_indices.begin(),
                                                 d_agg_kinds.data(),
                                                 *d_spass_values,
                                                 *d_results_ptr,
                                                 fallback_blocks.data(),
                                                 fallback_stride,
                                                 full_stride,
                                                 num_strides,
                                                 num_fallback_rows,
                                                 num_rows});
  }

  finalize_output(values, aggs, agg_results, cache, stream);
  return {std::move(unique_keys), has_compound_aggs};
}
}  // namespace cudf::groupby::detail::hash
