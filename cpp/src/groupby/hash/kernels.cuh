/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include "cudf/types.hpp"
#include "helpers.cuh"
#include "single_pass_functors.cuh"

#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>

namespace cudf::groupby::detail::hash {

__device__ void calculate_columns_to_aggregate(int& col_start,
                                               int& col_end,
                                               cudf::mutable_table_device_view output_values,
                                               int num_input_cols,
                                               std::byte** s_aggregates_pointer,
                                               bool** s_aggregates_valid_pointer,
                                               std::byte* shared_set_aggregates,
                                               cudf::size_type cardinality,
                                               int total_agg_size)
{
  if (threadIdx.x == 0) {
    col_start           = col_end;
    int bytes_allocated = 0;
    int valid_col_size  = round_to_multiple_of_8(sizeof(bool) * cardinality);
    while ((bytes_allocated < total_agg_size) && (col_end < num_input_cols)) {
      int next_col_size =
        round_to_multiple_of_8(sizeof(output_values.column(col_end).type()) * cardinality);
      int next_col_total_size = valid_col_size + next_col_size;
      if (bytes_allocated + next_col_total_size > total_agg_size) { break; }
      s_aggregates_pointer[col_end] = shared_set_aggregates + bytes_allocated;
      s_aggregates_valid_pointer[col_end] =
        reinterpret_cast<bool*>(shared_set_aggregates + bytes_allocated + next_col_size);
      bytes_allocated += next_col_total_size;
      col_end++;
    }
  }
}

__device__ void initialize_shared_memory_aggregates(int col_start,
                                                    int col_end,
                                                    cudf::mutable_table_device_view output_values,
                                                    std::byte** s_aggregates_pointer,
                                                    bool** s_aggregates_valid_pointer,
                                                    cudf::size_type cardinality,
                                                    cudf::aggregation::Kind const* aggs)
{
  for (auto col_idx = col_start; col_idx < col_end; col_idx++) {
    for (auto idx = threadIdx.x; idx < cardinality; idx += blockDim.x) {
      cudf::detail::dispatch_type_and_aggregation(output_values.column(col_idx).type(),
                                                  aggs[col_idx],
                                                  initialize_shmem{},
                                                  s_aggregates_pointer[col_idx],
                                                  idx,
                                                  s_aggregates_valid_pointer[col_idx]);
    }
  }
}

__device__ void compute_pre_aggregrates(int col_start,
                                        int col_end,
                                        cudf::table_device_view input_values,
                                        cudf::size_type num_input_rows,
                                        cudf::size_type* local_mapping_index,
                                        std::byte** s_aggregates_pointer,
                                        bool** s_aggregates_valid_pointer,
                                        cudf::aggregation::Kind const* aggs)
{
  for (auto cur_idx = blockDim.x * blockIdx.x + threadIdx.x; cur_idx < num_input_rows;
       cur_idx += blockDim.x * gridDim.x) {
    auto map_idx = local_mapping_index[cur_idx];

    for (auto col_idx = col_start; col_idx < col_end; col_idx++) {
      auto input_col = input_values.column(col_idx);

      cudf::detail::dispatch_type_and_aggregation(input_col.type(),
                                                  aggs[col_idx],
                                                  shmem_element_aggregator{},
                                                  s_aggregates_pointer[col_idx],
                                                  map_idx,
                                                  s_aggregates_valid_pointer[col_idx],
                                                  input_col,
                                                  cur_idx);
    }
  }
}

__device__ void compute_final_aggregates(int col_start,
                                         int col_end,
                                         cudf::table_device_view input_values,
                                         cudf::mutable_table_device_view output_values,
                                         cudf::size_type cardinality,
                                         cudf::size_type* global_mapping_index,
                                         std::byte** s_aggregates_pointer,
                                         bool** s_aggregates_valid_pointer,
                                         cudf::aggregation::Kind const* aggs)
{
  for (auto cur_idx = threadIdx.x; cur_idx < cardinality; cur_idx += blockDim.x) {
    auto out_idx = global_mapping_index[blockIdx.x * GROUPBY_SHM_MAX_ELEMENTS + cur_idx];
    for (auto col_idx = col_start; col_idx < col_end; col_idx++) {
      auto output_col = output_values.column(col_idx);

      cudf::detail::dispatch_type_and_aggregation(input_values.column(col_idx).type(),
                                                  aggs[col_idx],
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
CUDF_KERNEL void compute_aggs_kernel(cudf::size_type num_rows,
                                     cudf::size_type* local_mapping_index,
                                     cudf::size_type* global_mapping_index,
                                     cudf::size_type* block_cardinality,
                                     cudf::table_device_view input_values,
                                     cudf::mutable_table_device_view output_values,
                                     cudf::aggregation::Kind const* aggs,
                                     int total_agg_size,
                                     int pointer_size)
{
  auto const block       = cooperative_groups::this_thread_block();
  auto const cardinality = block_cardinality[block.group_index().x];
  if (cardinality >= GROUPBY_CARDINALITY_THRESHOLD) { return; }

  auto const num_cols = output_values.num_columns();

  __shared__ int col_start;
  __shared__ int col_end;
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
    calculate_columns_to_aggregate(col_start,
                                   col_end,
                                   output_values,
                                   num_cols,
                                   s_aggregates_pointer,
                                   s_aggregates_valid_pointer,
                                   shared_set_aggregates,
                                   cardinality,
                                   total_agg_size);
    block.sync();
    initialize_shared_memory_aggregates(col_start,
                                        col_end,
                                        output_values,
                                        s_aggregates_pointer,
                                        s_aggregates_valid_pointer,
                                        cardinality,
                                        aggs);
    block.sync();
    compute_pre_aggregrates(col_start,
                            col_end,
                            input_values,
                            num_rows,
                            local_mapping_index,
                            s_aggregates_pointer,
                            s_aggregates_valid_pointer,
                            aggs);
    block.sync();
    compute_final_aggregates(col_start,
                             col_end,
                             input_values,
                             output_values,
                             cardinality,
                             global_mapping_index,
                             s_aggregates_pointer,
                             s_aggregates_valid_pointer,
                             aggs);
    block.sync();
  }
}

}  // namespace cudf::groupby::detail::hash
