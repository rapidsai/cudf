/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "compute_mapping_indices.hpp"
#include "helpers.cuh"

#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/utilities/cuda.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cooperative_groups.h>
#include <cuco/static_set_ref.cuh>
#include <cuda/std/atomic>

namespace cudf::groupby::detail::hash {
template <typename SetType>
__device__ void find_local_mapping(cooperative_groups::thread_block const& block,
                                   size_type idx,
                                   size_type num_input_rows,
                                   SetType shared_set,
                                   bitmask_type const* row_bitmask,
                                   size_type* cardinality,
                                   size_type* local_mapping_indices,
                                   size_type* shared_set_indices)
{
  auto const is_valid_input =
    idx < num_input_rows && (!row_bitmask || cudf::bit_is_set(row_bitmask, idx));

  auto const [inserted_idx, inserted] = [&] {
    if (!is_valid_input) { return cuda::std::pair{0, false}; }
    auto const [matched_it, inserted] = shared_set.insert_and_find(idx);
    if (inserted) {
      auto const ref_cardinality =
        cuda::atomic_ref<size_type, cuda::thread_scope_block>{*cardinality};
      auto const shared_set_index = ref_cardinality.fetch_add(1, cuda::std::memory_order_relaxed);
      shared_set_indices[shared_set_index] = idx;
      local_mapping_indices[idx]           = shared_set_index;
    }
    return cuda::std::pair{*matched_it, inserted};
  }();
  // Syncing the thread block is needed so that updates in `local_mapping_indices` are visible to
  // all threads in the thread block.
  block.sync();
  if (is_valid_input) {
    if (!inserted) { local_mapping_indices[idx] = local_mapping_indices[inserted_idx]; }
  } else if (idx < num_input_rows) {
    // Store a sentinel value, so later on we can use it to identify which rows are invalid without
    // using the validity bitmask.
    local_mapping_indices[idx] = cudf::detail::CUDF_SIZE_TYPE_SENTINEL;
  }
}

template <typename SetRef>
__device__ void find_global_mapping(cooperative_groups::thread_block const& block,
                                    size_type iter,
                                    size_type cardinality,
                                    SetRef global_set,
                                    size_type* shared_set_indices,
                                    size_type* global_mapping_indices)
{
  auto const block_data_offset =
    static_cast<int64_t>(GROUPBY_SHM_MAX_ELEMENTS) * (gridDim.x * iter + blockIdx.x);

  // For all unique keys in shared memory hash set, stores their matches in global hash set to
  // `global_mapping_indices`.
  for (auto idx = block.thread_rank(); idx < cardinality; idx += block.num_threads()) {
    auto const input_idx                            = shared_set_indices[idx];
    auto const key_idx                              = *global_set.insert_and_find(input_idx).first;
    global_mapping_indices[block_data_offset + idx] = key_idx;
  }
}

/*
 * @brief Inserts keys into the shared memory hash set, and stores the block-wise rank for a given
 * row index in `local_mapping_indices`. If the number of unique keys found in a threadblock exceeds
 * `GROUPBY_CARDINALITY_THRESHOLD`, we insert the unique keys found to the global hash set, and
 * reset the shared memory for the block to start a new iteration.
 */
template <class SetRef>
CUDF_KERNEL void mapping_indices_kernel(size_type num_input_rows,
                                        SetRef global_set,
                                        bitmask_type const* row_bitmask,
                                        size_type* local_mapping_indices,
                                        size_type* global_mapping_indices,
                                        size_type* block_cardinality,
                                        size_type* block_row_ends)
{
  __shared__ size_type shared_set_indices[GROUPBY_SHM_MAX_ELEMENTS];

  // Data buffer for the shared memory hash set.
  __shared__ size_type tmp_storage[valid_extent.value()];

  auto raw_set = cuco::static_set_ref{
    cuco::empty_key<size_type>{cudf::detail::CUDF_SIZE_TYPE_SENTINEL},
    global_set.key_eq(),
    probing_scheme_t{global_set.hash_function()},
    cuco::thread_scope_block,
    cuco::bucket_storage_ref<size_type, GROUPBY_BUCKET_SIZE, decltype(valid_extent)>{valid_extent,
                                                                                     tmp_storage}};
  auto shared_set  = raw_set.rebind_operators(cuco::insert_and_find);
  auto const block = cooperative_groups::this_thread_block();
  shared_set.initialize(block);

  __shared__ size_type cardinality;
  if (block.thread_rank() == 0) { cardinality = 0; }
  block.sync();

  auto const grid_stride = cudf::detail::grid_1d::grid_stride();
  size_type iter         = 0;  // number of times the block resets

  // All threads in the block will participate in the loop, and sync.
  for (auto idx = cudf::detail::grid_1d::global_thread_id();
       idx - block.thread_rank() < num_input_rows;
       idx += grid_stride) {
    block.sync();
    find_local_mapping(block,
                       idx,
                       num_input_rows,
                       shared_set,
                       row_bitmask,
                       &cardinality,
                       local_mapping_indices,
                       shared_set_indices);
    block.sync();

    // The iteration ends here. Flush data and reset the block.
    if (auto const is_last_iteration = idx + grid_stride - block.thread_rank() >= num_input_rows;
        is_last_iteration || cardinality > GROUPBY_CARDINALITY_THRESHOLD) {
      if (!is_last_iteration) { shared_set.initialize(block); }
      find_global_mapping(
        block, iter, cardinality, global_set, shared_set_indices, global_mapping_indices);
      block.sync();  // to isolate cardinality from being modified below
      if (block.thread_rank() == 0) {
        auto const block_data_idx         = gridDim.x * iter + blockIdx.x;
        block_cardinality[block_data_idx] = cardinality;
        block_row_ends[block_data_idx] =
          cuda::std::min(static_cast<size_type>(idx + GROUPBY_BLOCK_SIZE), num_input_rows);
        cardinality = 0;
      }
      ++iter;
      block.sync();
    }
  }
}

template <class SetRef>
int32_t max_active_blocks_mapping_kernel()
{
  int32_t max_active_blocks{-1};
  CUDF_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &max_active_blocks, mapping_indices_kernel<SetRef>, GROUPBY_BLOCK_SIZE, 0));
  return max_active_blocks;
}

template <class SetRef>
void compute_mapping_indices(size_type grid_size,
                             size_type num_rows,
                             SetRef global_set,
                             bitmask_type const* row_bitmask,
                             size_type* local_mapping_indices,
                             size_type* global_mapping_indices,
                             size_type* block_cardinality,
                             size_type* block_row_ends,
                             rmm::cuda_stream_view stream)
{
  mapping_indices_kernel<<<grid_size, GROUPBY_BLOCK_SIZE, 0, stream>>>(num_rows,
                                                                       global_set,
                                                                       row_bitmask,
                                                                       local_mapping_indices,
                                                                       global_mapping_indices,
                                                                       block_cardinality,
                                                                       block_row_ends);
}
}  // namespace cudf::groupby::detail::hash
