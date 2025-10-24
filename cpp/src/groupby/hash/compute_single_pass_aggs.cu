/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compute_single_pass_aggs.cuh"
#include "compute_single_pass_aggs.hpp"

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <cub/device/device_select.cuh>

namespace cudf::groupby::detail::hash {

std::pair<bool, size_type> is_shared_memory_compatible(host_span<aggregation::Kind const> agg_kinds,
                                                       table_view const& values,
                                                       size_type grid_size)
{
  // If any aggregation has values type is dictionary, or the aggregation is SUM_WITH_OVERFLOW,
  // we should always use global memory code path.
  for (std::size_t i = 0; i < agg_kinds.size(); ++i) {
    if (is_dictionary(values.column(i).type()) || agg_kinds[i] == aggregation::SUM_WITH_OVERFLOW) {
      return {false, 0};
    }
  }

  auto const available_shmem_size = get_available_shared_memory_size(grid_size);
  auto const offsets_buffer_size  = compute_shmem_offsets_size(values.num_columns()) * 2;
  auto const data_buffer_size     = available_shmem_size - offsets_buffer_size;

  auto const can_run_by_shared_mem_kernel =
    std::all_of(values.begin(), values.end(), [&](auto const& col) {
      // Ensure there is enough buffer space to store local aggregations up to the max
      // cardinality for shared memory aggregations
      auto const size = type_dispatcher<dispatch_storage_type>(col.type(), size_of_functor{});
      return data_buffer_size >= size * GROUPBY_CARDINALITY_THRESHOLD;
    });
  return {can_run_by_shared_mem_kernel, available_shmem_size};
}

std::pair<size_type, rmm::device_uvector<size_type>> find_fallback_blocks(
  size_type grid_size, size_type const* block_cardinality, rmm::cuda_stream_view stream)
{
  rmm::device_uvector<size_type> fallback_block_ids(grid_size, stream);
  rmm::device_scalar<size_type> d_num_fallback_blocks(stream);

  std::size_t storage_bytes = 0;
  auto const select_pred    = [block_cardinality] __device__(auto const idx) {
    return block_cardinality[idx] >= GROUPBY_CARDINALITY_THRESHOLD;
  };
  auto const exec_copy_if = [&](auto const storage_ptr) {
    cub::DeviceSelect::If(storage_ptr,
                          storage_bytes,
                          thrust::make_counting_iterator(0),
                          fallback_block_ids.begin(),
                          d_num_fallback_blocks.data(),
                          grid_size,
                          select_pred,
                          stream.value());
  };

  exec_copy_if(nullptr);
  rmm::device_buffer tmp_storage(storage_bytes, stream);
  exec_copy_if(tmp_storage.data());

  auto const num_fallback_blocks = d_num_fallback_blocks.value(stream);
  if (num_fallback_blocks > 0) { fallback_block_ids.resize(num_fallback_blocks, stream); }

  return {num_fallback_blocks,
          num_fallback_blocks > 0 ? std::move(fallback_block_ids)
                                  : rmm::device_uvector<size_type>{0, stream}};
}

template std::pair<rmm::device_uvector<size_type>, bool> compute_single_pass_aggs<global_set_t>(
  global_set_t& global_set,
  bitmask_type const* row_bitmask,
  host_span<aggregation_request const> requests,
  cudf::detail::result_cache* cache,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

}  // namespace cudf::groupby::detail::hash
