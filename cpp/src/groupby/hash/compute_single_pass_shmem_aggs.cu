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

#include "compute_single_pass_shmem_aggs.hpp"
#include "create_sparse_results_table.hpp"
#include "global_memory_aggregator.cuh"
#include "helpers.cuh"
#include "shared_memory_aggregator.cuh"
#include "single_pass_functors.cuh"

#include <cudf/aggregation.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/cuda.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cooperative_groups.h>

#include <cstddef>

namespace cudf::groupby::detail::hash {
namespace {
__device__ void calculate_columns_to_aggregate(cudf::size_type& col_start,
                                               cudf::size_type& col_end,
                                               cudf::mutable_table_device_view output_values,
                                               cudf::size_type output_size,
                                               std::byte** s_aggregates_pointer,
                                               bool** s_aggregates_valid_pointer,
                                               std::byte* shared_set_aggregates,
                                               cudf::size_type cardinality,
                                               cudf::size_type total_agg_size)
{
  col_start                       = col_end;
  cudf::size_type bytes_allocated = 0;

  auto const valid_col_size = round_to_multiple_of_8(sizeof(bool) * cardinality);

  while (bytes_allocated < total_agg_size && col_end < output_size) {
    auto const next_col_size =
      round_to_multiple_of_8(sizeof(output_values.column(col_end).type()) * cardinality);
    auto const next_col_total_size = next_col_size + valid_col_size;

    if (bytes_allocated + next_col_total_size > total_agg_size) { break; }

    s_aggregates_pointer[col_end] = shared_set_aggregates + bytes_allocated;
    s_aggregates_valid_pointer[col_end] =
      reinterpret_cast<bool*>(shared_set_aggregates + bytes_allocated + next_col_size);

    bytes_allocated += next_col_total_size;
    ++col_end;
  }
}

__device__ void initialize_shmem_aggregations(cudf::size_type col_start,
                                              cudf::size_type col_end,
                                              cudf::mutable_table_device_view output_values,
                                              std::byte** s_aggregates_pointer,
                                              bool** s_aggregates_valid_pointer,
                                              cudf::size_type cardinality,
                                              cudf::aggregation::Kind const* d_agg_kinds)
{
  for (auto col_idx = col_start; col_idx < col_end; col_idx++) {
    for (auto idx = threadIdx.x; idx < cardinality; idx += blockDim.x) {
      cudf::detail::dispatch_type_and_aggregation(output_values.column(col_idx).type(),
                                                  d_agg_kinds[col_idx],
                                                  initialize_shmem{},
                                                  s_aggregates_pointer[col_idx],
                                                  idx,
                                                  s_aggregates_valid_pointer[col_idx]);
    }
  }
}

__device__ void compute_pre_aggregrations(cudf::size_type col_start,
                                          cudf::size_type col_end,
                                          bitmask_type const* row_bitmask,
                                          bool skip_rows_with_nulls,
                                          cudf::table_device_view input_values,
                                          cudf::size_type num_input_rows,
                                          cudf::size_type* local_mapping_index,
                                          std::byte** s_aggregates_pointer,
                                          bool** s_aggregates_valid_pointer,
                                          cudf::aggregation::Kind const* d_agg_kinds)
{
  // TODO grid_1d utility
  for (auto cur_idx = blockDim.x * blockIdx.x + threadIdx.x; cur_idx < num_input_rows;
       cur_idx += blockDim.x * gridDim.x) {
    if (not skip_rows_with_nulls or cudf::bit_is_set(row_bitmask, cur_idx)) {
      auto map_idx = local_mapping_index[cur_idx];

      for (auto col_idx = col_start; col_idx < col_end; col_idx++) {
        auto input_col = input_values.column(col_idx);

        cudf::detail::dispatch_type_and_aggregation(input_col.type(),
                                                    d_agg_kinds[col_idx],
                                                    shmem_element_aggregator{},
                                                    s_aggregates_pointer[col_idx],
                                                    map_idx,
                                                    s_aggregates_valid_pointer[col_idx],
                                                    input_col,
                                                    cur_idx);
      }
    }
  }
}

__device__ void compute_final_aggregations(cudf::size_type col_start,
                                           cudf::size_type col_end,
                                           cudf::table_device_view input_values,
                                           cudf::mutable_table_device_view output_values,
                                           cudf::size_type cardinality,
                                           cudf::size_type* global_mapping_index,
                                           std::byte** s_aggregates_pointer,
                                           bool** s_aggregates_valid_pointer,
                                           cudf::aggregation::Kind const* d_agg_kinds)
{
  for (auto cur_idx = threadIdx.x; cur_idx < cardinality; cur_idx += blockDim.x) {
    auto out_idx = global_mapping_index[blockIdx.x * GROUPBY_SHM_MAX_ELEMENTS + cur_idx];
    for (auto col_idx = col_start; col_idx < col_end; col_idx++) {
      auto output_col = output_values.column(col_idx);

      cudf::detail::dispatch_type_and_aggregation(input_values.column(col_idx).type(),
                                                  d_agg_kinds[col_idx],
                                                  gmem_element_aggregator{},
                                                  output_col,
                                                  out_idx,
                                                  input_values.column(col_idx),
                                                  s_aggregates_pointer[col_idx],
                                                  cur_idx,
                                                  s_aggregates_valid_pointer[col_idx]);
    }
  }
}

/* Takes the local_mapping_index and global_mapping_index to compute
 * pre (shared) and final (global) aggregates*/
CUDF_KERNEL void single_pass_shmem_aggs_kernel(cudf::size_type num_rows,
                                               bitmask_type const* row_bitmask,
                                               bool skip_rows_with_nulls,
                                               cudf::size_type* local_mapping_index,
                                               cudf::size_type* global_mapping_index,
                                               cudf::size_type* block_cardinality,
                                               cudf::table_device_view input_values,
                                               cudf::mutable_table_device_view output_values,
                                               cudf::aggregation::Kind const* d_agg_kinds,
                                               cudf::size_type total_agg_size,
                                               cudf::size_type pointer_size)
{
  auto const block       = cooperative_groups::this_thread_block();
  auto const cardinality = block_cardinality[block.group_index().x];
  if (cardinality >= GROUPBY_CARDINALITY_THRESHOLD) { return; }

  auto const num_cols = output_values.num_columns();

  __shared__ cudf::size_type col_start;
  __shared__ cudf::size_type col_end;
  extern __shared__ std::byte shared_set_aggregates[];
  std::byte** s_aggregates_pointer =
    reinterpret_cast<std::byte**>(shared_set_aggregates + total_agg_size);
  bool** s_aggregates_valid_pointer =
    reinterpret_cast<bool**>(shared_set_aggregates + total_agg_size + pointer_size);

  if (block.thread_rank() == 0) {
    col_start = 0;
    col_end   = 0;
  }
  block.sync();

  while (col_end < num_cols) {
    if (block.thread_rank() == 0) {
      calculate_columns_to_aggregate(col_start,
                                     col_end,
                                     output_values,
                                     num_cols,
                                     s_aggregates_pointer,
                                     s_aggregates_valid_pointer,
                                     shared_set_aggregates,
                                     cardinality,
                                     total_agg_size);
    }
    block.sync();
    initialize_shmem_aggregations(col_start,
                                  col_end,
                                  output_values,
                                  s_aggregates_pointer,
                                  s_aggregates_valid_pointer,
                                  cardinality,
                                  d_agg_kinds);
    block.sync();
    compute_pre_aggregrations(col_start,
                              col_end,
                              row_bitmask,
                              skip_rows_with_nulls,
                              input_values,
                              num_rows,
                              local_mapping_index,
                              s_aggregates_pointer,
                              s_aggregates_valid_pointer,
                              d_agg_kinds);
    block.sync();
    compute_final_aggregations(col_start,
                               col_end,
                               input_values,
                               output_values,
                               cardinality,
                               global_mapping_index,
                               s_aggregates_pointer,
                               s_aggregates_valid_pointer,
                               d_agg_kinds);
    block.sync();
  }
}

constexpr size_t get_previous_multiple_of_8(size_t number) { return number / 8 * 8; }

}  // namespace

size_t available_shared_memory_size(cudf::size_type grid_size)
{
  auto const active_blocks_per_sm =
    cudf::util::div_rounding_up_safe(grid_size, cudf::detail::num_multiprocessors());

  size_t dynamic_shmem_size = 0;
  CUDF_CUDA_TRY(cudaOccupancyAvailableDynamicSMemPerBlock(
    &dynamic_shmem_size, single_pass_shmem_aggs_kernel, active_blocks_per_sm, GROUPBY_BLOCK_SIZE));
  return get_previous_multiple_of_8(0.5 * dynamic_shmem_size);
}

size_t shmem_agg_pointer_size(cudf::size_type num_cols) { return sizeof(void*) * num_cols; }

void compute_single_pass_shmem_aggs(cudf::size_type grid_size,
                                    cudf::size_type num_input_rows,
                                    bitmask_type const* row_bitmask,
                                    bool skip_rows_with_nulls,
                                    cudf::size_type* local_mapping_index,
                                    cudf::size_type* global_mapping_index,
                                    cudf::size_type* block_cardinality,
                                    cudf::table_device_view input_values,
                                    cudf::mutable_table_device_view output_values,
                                    cudf::aggregation::Kind const* d_agg_kinds,
                                    rmm::cuda_stream_view stream)
{
  auto const shmem_size = available_shared_memory_size(grid_size);
  // For each aggregation, need two pointers to arrays in shmem
  // One where the aggregation is performed, one indicating the validity of the aggregation
  auto const shmem_pointer_size = shmem_agg_pointer_size(output_values.num_columns());
  // The rest of shmem is utilized for the actual arrays in shmem
  CUDF_EXPECTS(shmem_size > shmem_pointer_size * 2,
               "No enough space for shared memory aggregations");
  auto const shmem_agg_size = shmem_size - shmem_pointer_size * 2;
  single_pass_shmem_aggs_kernel<<<grid_size, GROUPBY_BLOCK_SIZE, shmem_size, stream>>>(
    num_input_rows,
    row_bitmask,
    skip_rows_with_nulls,
    local_mapping_index,
    global_mapping_index,
    block_cardinality,
    input_values,
    output_values,
    d_agg_kinds,
    shmem_agg_size,
    shmem_pointer_size);
}
}  // namespace cudf::groupby::detail::hash
