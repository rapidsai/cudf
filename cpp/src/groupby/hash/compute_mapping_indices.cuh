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
#pragma once

#include "compute_mapping_indices.hpp"
#include "helpers.cuh"

#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/cuda.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cooperative_groups.h>
#include <cuco/static_set_ref.cuh>
#include <cuda/std/atomic>
#include <cuda/std/utility>

#include <algorithm>

namespace cudf::groupby::detail::hash {
template <typename SetType>
__device__ void find_local_mapping(cooperative_groups::thread_block const& block,
                                   cudf::size_type idx,
                                   cudf::size_type num_input_rows,
                                   SetType shared_set,
                                   bitmask_type const* row_bitmask,
                                   bool skip_rows_with_nulls,
                                   cudf::size_type* cardinality,
                                   cudf::size_type* local_mapping_index,
                                   cudf::size_type* shared_set_indices)
{
  auto const is_valid_input =
    idx < num_input_rows and (not skip_rows_with_nulls or cudf::bit_is_set(row_bitmask, idx));
  auto const [result_idx, inserted] = [&]() {
    if (is_valid_input) {
      auto const result      = shared_set.insert_and_find(idx);
      auto const matched_idx = *result.first;
      auto const inserted    = result.second;
      // inserted a new element
      if (result.second) {
        auto const shared_set_index          = atomicAdd(cardinality, 1);
        shared_set_indices[shared_set_index] = idx;
        local_mapping_index[idx]             = shared_set_index;
      }
      return cuda::std::pair{matched_idx, inserted};
    }
    return cuda::std::pair{0, false};  // dummy values
  }();
  // Syncing the thread block is needed so that updates in `local_mapping_index` are visible to all
  // threads in the thread block.
  block.sync();
  if (is_valid_input) {
    // element was already in set
    if (!inserted) { local_mapping_index[idx] = local_mapping_index[result_idx]; }
  }
}

template <typename SetRef>
__device__ void find_global_mapping(cooperative_groups::thread_block const& block,
                                    cudf::size_type cardinality,
                                    SetRef global_set,
                                    cudf::size_type* shared_set_indices,
                                    cudf::size_type* global_mapping_index)
{
  // for all unique keys in shared memory hash set, stores their matches in
  // global hash set to `global_mapping_index`
  for (auto idx = block.thread_rank(); idx < cardinality; idx += block.num_threads()) {
    auto const input_idx = shared_set_indices[idx];
    global_mapping_index[block.group_index().x * GROUPBY_SHM_MAX_ELEMENTS + idx] =
      *global_set.insert_and_find(input_idx).first;
  }
}

/*
 * @brief Inserts keys into the shared memory hash set, and stores the block-wise rank for a given
 * row index in `local_mapping_index`. If the number of unique keys found in a threadblock exceeds
 * `GROUPBY_CARDINALITY_THRESHOLD`, the threads in that block will exit without updating
 * `global_set` or setting `global_mapping_index`. Else, we insert the unique keys found to the
 * global hash set, and save the row index of the global sparse table in `global_mapping_index`.
 */
template <class SetRef>
CUDF_KERNEL void mapping_indices_kernel(cudf::size_type num_input_rows,
                                        SetRef global_set,
                                        bitmask_type const* row_bitmask,
                                        bool skip_rows_with_nulls,
                                        cudf::size_type* local_mapping_index,
                                        cudf::size_type* global_mapping_index,
                                        cudf::size_type* block_cardinality,
                                        cuda::std::atomic_flag* needs_global_memory_fallback)
{
  __shared__ cudf::size_type shared_set_indices[GROUPBY_SHM_MAX_ELEMENTS];

  // Shared set initialization
  __shared__ cuco::bucket<cudf::size_type, GROUPBY_BUCKET_SIZE> buckets[bucket_extent.value()];

  auto raw_set = cuco::static_set_ref{
    cuco::empty_key<cudf::size_type>{cudf::detail::CUDF_SIZE_TYPE_SENTINEL},
    global_set.key_eq(),
    probing_scheme_t{global_set.hash_function()},
    cuco::thread_scope_block,
    cuco::bucket_storage_ref<cudf::size_type, GROUPBY_BUCKET_SIZE, decltype(bucket_extent)>{
      bucket_extent, buckets}};
  auto shared_set = raw_set.rebind_operators(cuco::insert_and_find);

  auto const block = cooperative_groups::this_thread_block();
  shared_set.initialize(block);

  __shared__ cudf::size_type cardinality;
  if (block.thread_rank() == 0) { cardinality = 0; }
  block.sync();

  auto const stride = cudf::detail::grid_1d::grid_stride();

  for (auto idx = cudf::detail::grid_1d::global_thread_id();
       idx - block.thread_rank() < num_input_rows;
       idx += stride) {
    find_local_mapping(block,
                       idx,
                       num_input_rows,
                       shared_set,
                       row_bitmask,
                       skip_rows_with_nulls,
                       &cardinality,
                       local_mapping_index,
                       shared_set_indices);

    block.sync();

    if (cardinality >= GROUPBY_CARDINALITY_THRESHOLD) {
      if (block.thread_rank() == 0) { needs_global_memory_fallback->test_and_set(); }
      break;
    }
  }

  // Insert unique keys from shared to global hash set if block-cardinality
  // doesn't exceed the threshold upper-limit
  if (cardinality < GROUPBY_CARDINALITY_THRESHOLD) {
    find_global_mapping(block, cardinality, global_set, shared_set_indices, global_mapping_index);
  }

  if (block.thread_rank() == 0) { block_cardinality[block.group_index().x] = cardinality; }
}

template <class SetRef>
cudf::size_type max_occupancy_grid_size(cudf::size_type n)
{
  cudf::size_type max_active_blocks{-1};
  CUDF_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &max_active_blocks, mapping_indices_kernel<SetRef>, GROUPBY_BLOCK_SIZE, 0));
  auto const grid_size  = max_active_blocks * cudf::detail::num_multiprocessors();
  auto const num_blocks = cudf::util::div_rounding_up_safe(n, GROUPBY_BLOCK_SIZE);
  return std::min(grid_size, num_blocks);
}

template <class SetRef>
void compute_mapping_indices(cudf::size_type grid_size,
                             cudf::size_type num,
                             SetRef global_set,
                             bitmask_type const* row_bitmask,
                             bool skip_rows_with_nulls,
                             cudf::size_type* local_mapping_index,
                             cudf::size_type* global_mapping_index,
                             cudf::size_type* block_cardinality,
                             cuda::std::atomic_flag* needs_global_memory_fallback,
                             rmm::cuda_stream_view stream)
{
  mapping_indices_kernel<<<grid_size, GROUPBY_BLOCK_SIZE, 0, stream>>>(
    num,
    global_set,
    row_bitmask,
    skip_rows_with_nulls,
    local_mapping_index,
    global_mapping_index,
    block_cardinality,
    needs_global_memory_fallback);
}
}  // namespace cudf::groupby::detail::hash
