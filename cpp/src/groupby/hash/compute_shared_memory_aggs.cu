/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compute_shared_memory_aggs.hpp"
#include "global_memory_aggregator.cuh"
#include "helpers.cuh"
#include "shared_memory_aggregator.cuh"
#include "single_pass_functors.cuh"

#include <cudf/aggregation.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/cuda.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cooperative_groups.h>
#include <cuda/std/cstddef>

namespace cudf::groupby::detail::hash {
namespace {
/// Shared memory data alignment
CUDF_HOST_DEVICE size_type constexpr ALIGNMENT = 8;

// Allocates shared memory required for output columns. Exits if there is insufficient memory to
// perform shared memory aggregation for the current output column.
__device__ void setup_shmem_buffers(size_type& col_start,
                                    size_type& col_end,
                                    mutable_table_device_view output_values,
                                    size_type output_size,
                                    size_type* shmem_agg_res_offsets,
                                    size_type* shmem_agg_mask_offsets,
                                    size_type cardinality,
                                    size_type total_agg_size)
{
  col_start                 = col_end;
  size_type bytes_allocated = 0;

  auto const valid_col_size =
    util::round_up_safe(static_cast<size_type>(sizeof(bool) * cardinality), ALIGNMENT);

  while (bytes_allocated < total_agg_size && col_end < output_size) {
    auto const col_idx = col_end;
    auto const next_col_size =
      util::round_up_safe(type_dispatcher<dispatch_storage_type>(
                            output_values.column(col_idx).type(), size_of_functor{}) *
                            cardinality,
                          ALIGNMENT);
    auto const next_col_total_size = next_col_size + valid_col_size;
    if (bytes_allocated + next_col_total_size > total_agg_size) { break; }

    shmem_agg_res_offsets[col_end]  = bytes_allocated;
    shmem_agg_mask_offsets[col_end] = bytes_allocated + next_col_size;
    bytes_allocated += next_col_total_size;
    ++col_end;
  }
}

// Each block initialize its own shared memory aggregation results
__device__ void initialize_shmem_aggregations(cooperative_groups::thread_block const& block,
                                              size_type col_start,
                                              size_type col_end,
                                              mutable_table_device_view output_values,
                                              cuda::std::byte* shmem_agg_storage,
                                              size_type const* shmem_agg_res_offsets,
                                              size_type const* shmem_agg_mask_offsets,
                                              size_type cardinality,
                                              aggregation::Kind const* d_agg_kinds)
{
  for (auto col_idx = col_start; col_idx < col_end; col_idx++) {
    for (auto idx = block.thread_rank(); idx < cardinality; idx += block.num_threads()) {
      auto const target = shmem_agg_storage + shmem_agg_res_offsets[col_idx];
      auto const target_mask =
        reinterpret_cast<bool*>(shmem_agg_storage + shmem_agg_mask_offsets[col_idx]);
      cudf::detail::dispatch_type_and_aggregation(output_values.column(col_idx).type(),
                                                  d_agg_kinds[col_idx],
                                                  initialize_shmem{},
                                                  target,
                                                  target_mask,
                                                  idx);
    }
  }
}

__device__ void aggregate_to_shmem(cooperative_groups::thread_block const& block,
                                   size_type row_start,
                                   size_type row_end,
                                   size_type col_start,
                                   size_type col_end,
                                   table_device_view source,
                                   size_type const* local_mapping_index,
                                   cuda::std::byte* shmem_agg_storage,
                                   size_type const* shmem_agg_res_offsets,
                                   size_type const* shmem_agg_mask_offsets,
                                   aggregation::Kind const* d_agg_kinds,
                                   size_type agg_location_offset)
{
  // Aggregates global memory sources to shared memory targets
  for (auto source_idx = block.thread_rank() + row_start; source_idx < row_end;
       source_idx += cudf::detail::grid_1d::grid_stride()) {
    auto target_idx = local_mapping_index[source_idx];
    if (target_idx == cudf::detail::CUDF_SIZE_TYPE_SENTINEL) { continue; }  // null row
    target_idx += agg_location_offset;

    for (auto col_idx = col_start; col_idx < col_end; col_idx++) {
      auto const source_col = source.column(col_idx);
      auto const target     = shmem_agg_storage + shmem_agg_res_offsets[col_idx];
      auto const target_mask =
        reinterpret_cast<bool*>(shmem_agg_storage + shmem_agg_mask_offsets[col_idx]);
      cudf::detail::dispatch_type_and_aggregation(source_col.type(),
                                                  d_agg_kinds[col_idx],
                                                  shmem_element_aggregator{},
                                                  target,
                                                  target_mask,
                                                  target_idx,
                                                  source_col,
                                                  source_idx);
    }
  }
}

__device__ void update_aggs_shmem_to_gmem(cooperative_groups::thread_block const& block,
                                          size_type iter,
                                          size_type col_start,
                                          size_type col_end,
                                          table_device_view input_values,
                                          mutable_table_device_view target,
                                          size_type cardinality,
                                          size_type num_agg_locations,
                                          size_type const* global_mapping_index,
                                          size_type const* transform_map,
                                          cuda::std::byte* shmem_agg_storage,
                                          size_type const* agg_res_offsets,
                                          size_type const* agg_mask_offsets,
                                          aggregation::Kind const* d_agg_kinds)
{
  auto const block_data_offset =
    static_cast<int64_t>(GROUPBY_SHM_MAX_ELEMENTS) * (gridDim.x * iter + blockIdx.x);

  // Aggregates shared memory sources to global memory targets
  for (auto idx = block.thread_rank(); idx < num_agg_locations; idx += block.num_threads()) {
    auto const target_idx =
      transform_map[global_mapping_index[block_data_offset + (idx % cardinality)]];
    for (auto col_idx = col_start; col_idx < col_end; col_idx++) {
      auto const target_col = target.column(col_idx);
      auto const source     = shmem_agg_storage + agg_res_offsets[col_idx];
      auto const source_mask =
        reinterpret_cast<bool*>(shmem_agg_storage + agg_mask_offsets[col_idx]);
      cudf::detail::dispatch_type_and_aggregation(input_values.column(col_idx).type(),
                                                  d_agg_kinds[col_idx],
                                                  gmem_element_aggregator{},
                                                  target_col,
                                                  target_idx,
                                                  input_values.column(col_idx),
                                                  source,
                                                  source_mask,
                                                  idx);
    }
  }
}

/* Takes the local_mapping_index and global_mapping_index to compute
 * pre (shared) and final (global) aggregates*/
CUDF_KERNEL void single_pass_shmem_aggs_kernel(size_type num_rows,
                                               size_type const* local_mapping_index,
                                               size_type const* global_mapping_index,
                                               size_type const* transform_map,
                                               size_type const* block_cardinality,
                                               size_type const* block_row_ends,
                                               table_device_view input_values,
                                               mutable_table_device_view output_values,
                                               aggregation::Kind const* d_agg_kinds,
                                               size_type total_agg_size,
                                               size_type offsets_size)
{
  __shared__ size_type col_start;
  __shared__ size_type col_end;
  __shared__ size_type row_start;
  __shared__ size_type row_end;
  __shared__ size_type cardinality;

  extern __shared__ cuda::std::byte shmem_agg_storage[];

  auto const shmem_agg_res_offsets =
    reinterpret_cast<size_type*>(shmem_agg_storage + total_agg_size);
  auto const shmem_agg_mask_offsets =
    reinterpret_cast<size_type*>(shmem_agg_storage + total_agg_size + offsets_size);
  auto const block = cooperative_groups::this_thread_block();

  auto iter = 0;
  if (block.thread_rank() == 0) { row_start = blockIdx.x * GROUPBY_BLOCK_SIZE; }
  block.sync();

  while (true) {
    if (block.thread_rank() == 0) {
      auto const block_data_idx = gridDim.x * iter + blockIdx.x;
      cardinality               = block_cardinality[block_data_idx];
      row_end                   = block_row_ends[block_data_idx];
      col_start                 = 0;
      col_end                   = 0;
    }
    block.sync();

    auto constexpr min_shmem_agg_locations = 32;
    auto const multiplication_factor       = min_shmem_agg_locations / cardinality;
    auto const num_agg_locations           = cuda::std::max(multiplication_factor, 1) * cardinality;
    auto const agg_location_offset =
      multiplication_factor > 1 ? (block.thread_rank() % multiplication_factor) * cardinality : 0;

    auto const num_cols = output_values.num_columns();
    while (col_end < num_cols) {
      block.sync();
      if (block.thread_rank() == 0) {
        setup_shmem_buffers(col_start,
                            col_end,
                            output_values,
                            num_cols,
                            shmem_agg_res_offsets,
                            shmem_agg_mask_offsets,
                            num_agg_locations,
                            total_agg_size);
      }
      block.sync();

      initialize_shmem_aggregations(block,
                                    col_start,
                                    col_end,
                                    output_values,
                                    shmem_agg_storage,
                                    shmem_agg_res_offsets,
                                    shmem_agg_mask_offsets,
                                    num_agg_locations,
                                    d_agg_kinds);
      block.sync();

      aggregate_to_shmem(block,
                         row_start,
                         row_end,
                         col_start,
                         col_end,
                         input_values,
                         local_mapping_index,
                         shmem_agg_storage,
                         shmem_agg_res_offsets,
                         shmem_agg_mask_offsets,
                         d_agg_kinds,
                         agg_location_offset);
      block.sync();

      update_aggs_shmem_to_gmem(block,
                                iter,
                                col_start,
                                col_end,
                                input_values,
                                output_values,
                                cardinality,
                                num_agg_locations,
                                global_mapping_index,
                                transform_map,
                                shmem_agg_storage,
                                shmem_agg_res_offsets,
                                shmem_agg_mask_offsets,
                                d_agg_kinds);
      block.sync();
    }  // while (col_end < num_cols)

    ++iter;
    if (row_end + cudf::detail::grid_1d::grid_stride() >= num_rows) { break; }
    if (block.thread_rank() == 0) { row_start = row_end; }
    block.sync();
  }  // while (true)
}

}  // namespace

size_type get_available_shared_memory_size(size_type grid_size)
{
  auto const active_blocks_per_sm =
    util::div_rounding_up_safe(grid_size, cudf::detail::num_multiprocessors());

  size_t dynamic_shmem_size = 0;
  CUDF_CUDA_TRY(cudaOccupancyAvailableDynamicSMemPerBlock(
    &dynamic_shmem_size, single_pass_shmem_aggs_kernel, active_blocks_per_sm, GROUPBY_BLOCK_SIZE));
  return util::round_down_safe(static_cast<size_type>(0.5 * dynamic_shmem_size), ALIGNMENT);
}

int32_t max_active_blocks_shmem_aggs_kernel()
{
  int32_t max_active_blocks{-1};
  CUDF_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &max_active_blocks, single_pass_shmem_aggs_kernel, GROUPBY_BLOCK_SIZE, 0));
  return max_active_blocks;
}

void compute_shared_memory_aggs(size_type grid_size,
                                size_type available_shmem_size,
                                size_type num_input_rows,
                                size_type const* local_mapping_index,
                                size_type const* global_mapping_index,
                                size_type const* transform_map,
                                size_type const* block_cardinality,
                                size_type const* block_row_ends,
                                table_device_view const& input_values,
                                mutable_table_device_view const& output_values,
                                aggregation::Kind const* d_agg_kinds,
                                rmm::cuda_stream_view stream)
{
  // For each aggregation, need one offset determining where the aggregation is
  // performed, another indicating the validity of the aggregation.
  // The rest of shmem is utilized for the actual arrays in shmem.
  auto const offsets_size   = compute_shmem_offsets_size(output_values.num_columns());
  auto const shmem_agg_size = available_shmem_size - offsets_size * 2;

  single_pass_shmem_aggs_kernel<<<grid_size, GROUPBY_BLOCK_SIZE, available_shmem_size, stream>>>(
    num_input_rows,
    local_mapping_index,
    global_mapping_index,
    transform_map,
    block_cardinality,
    block_row_ends,
    input_values,
    output_values,
    d_agg_kinds,
    shmem_agg_size,
    offsets_size);
}
}  // namespace cudf::groupby::detail::hash
