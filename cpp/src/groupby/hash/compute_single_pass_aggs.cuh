/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include "compute_aggregations.hpp"
#include "compute_single_pass_aggs.hpp"
#include "create_sparse_results_table.hpp"
#include "flatten_single_pass_aggs.hpp"
#include "helpers.cuh"
#include "single_pass_functors.cuh"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/aggregation/result_cache.hpp>
#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/cuda.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/groupby.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <cooperative_groups.h>
#include <cuco/static_set.cuh>

#include <unordered_set>

namespace cudf::groupby::detail::hash {
namespace {
template <typename SetType>
// TODO pass block
__device__ void find_local_mapping(cudf::size_type cur_idx,
                                   cudf::size_type num_input_rows,
                                   SetType shared_set,
                                   bitmask_type const* row_bitmask,
                                   bool skip_rows_with_nulls,
                                   cudf::size_type* cardinality,
                                   cudf::size_type* local_mapping_index,
                                   cudf::size_type* shared_set_indices)
{
  cudf::size_type result_idx;
  // TODO: un-init
  bool inserted;
  if (cur_idx < num_input_rows and
      (not skip_rows_with_nulls or cudf::bit_is_set(row_bitmask, cur_idx))) {
    auto const result = shared_set.insert_and_find(cur_idx);
    result_idx        = *result.first;
    inserted          = result.second;
    // inserted a new element
    if (result.second) {
      auto const shared_set_index          = atomicAdd(cardinality, 1);
      shared_set_indices[shared_set_index] = cur_idx;
      local_mapping_index[cur_idx]         = shared_set_index;
    }
  }
  // Syncing the thread block is needed so that updates in `local_mapping_index` are visible to all
  // threads in the thread block.
  __syncthreads();
  if (cur_idx < num_input_rows and
      (not skip_rows_with_nulls or cudf::bit_is_set(row_bitmask, cur_idx))) {
    // element was already in set
    if (!inserted) { local_mapping_index[cur_idx] = local_mapping_index[result_idx]; }
  }
}

template <typename SetType>
__device__ void find_global_mapping(cudf::size_type cur_idx,
                                    SetType global_set,
                                    cudf::size_type* shared_set_indices,
                                    cudf::size_type* global_mapping_index)
{
  auto const input_idx = shared_set_indices[cur_idx];
  global_mapping_index[blockIdx.x * GROUPBY_SHM_MAX_ELEMENTS + cur_idx] =
    *global_set.insert_and_find(input_idx).first;
}

/*
 * Inserts keys into the shared memory hash set, and stores the row index of the local
 * pre-aggregate table in `local_mapping_index`. If the number of unique keys found in a
 * threadblock exceeds `GROUPBY_CARDINALITY_THRESHOLD`, the threads in that block will exit without
 * updating `global_set` or setting `global_mapping_index`. Else, we insert the unique keys found to
 * the global hash set, and save the row index of the global sparse table in `global_mapping_index`.
 */
template <class SetRef, typename GlobalSetType, class WindowExtent>
CUDF_KERNEL void compute_mapping_indices(GlobalSetType global_set,
                                         cudf::size_type num_input_rows,
                                         WindowExtent window_extent,
                                         bitmask_type const* row_bitmask,
                                         bool skip_rows_with_nulls,
                                         cudf::size_type* local_mapping_index,
                                         cudf::size_type* global_mapping_index,
                                         cudf::size_type* block_cardinality,
                                         bool* direct_aggregations)
{
  // TODO: indices inserted in each shared memory set
  __shared__ cudf::size_type shared_set_indices[GROUPBY_SHM_MAX_ELEMENTS];

  // Shared set initialization
  __shared__ typename SetRef::window_type windows[window_extent.value()];
  auto storage     = SetRef::storage_ref_type(window_extent, windows);
  auto shared_set  = SetRef(cuco::empty_key<cudf::size_type>{cudf::detail::CUDF_SIZE_TYPE_SENTINEL},
                           global_set.key_eq(),
                           probing_scheme_t{global_set.hash_function()},
                            {},
                           storage);
  auto const block = cooperative_groups::this_thread_block();
  shared_set.initialize(block);

  auto shared_insert_ref = std::move(shared_set).with(cuco::insert_and_find);

  __shared__ cudf::size_type cardinality;
  if (block.thread_rank() == 0) { cardinality = 0; }
  block.sync();

  auto const stride = cudf::detail::grid_1d::grid_stride();

  for (auto cur_idx = cudf::detail::grid_1d::global_thread_id();
       cur_idx - block.thread_rank() < num_input_rows;
       cur_idx += stride) {
    find_local_mapping(cur_idx,
                       num_input_rows,
                       shared_insert_ref,
                       row_bitmask,
                       skip_rows_with_nulls,
                       &cardinality,
                       local_mapping_index,
                       shared_set_indices);

    block.sync();

    if (cardinality >= GROUPBY_CARDINALITY_THRESHOLD) {
      if (block.thread_rank() == 0) { *direct_aggregations = true; }
      break;
    }

    block.sync();
  }

  // Insert unique keys from shared to global hash set
  if (cardinality < GROUPBY_CARDINALITY_THRESHOLD) {
    for (auto cur_idx = block.thread_rank(); cur_idx < cardinality;
         cur_idx += block.num_threads()) {
      find_global_mapping(cur_idx, global_set, shared_set_indices, global_mapping_index);
    }
  }

  if (block.thread_rank() == 0) { block_cardinality[block.group_index().x] = cardinality; }
}

template <typename Kernel>
int max_occupancy_grid_size(Kernel kernel, cudf::size_type n)
{
  int max_active_blocks{-1};
  CUDF_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &max_active_blocks, kernel, GROUPBY_BLOCK_SIZE, 0));
  auto const grid_size  = max_active_blocks * cudf::detail::num_multiprocessors();
  auto const num_blocks = cudf::util::div_rounding_up_safe(n, GROUPBY_BLOCK_SIZE);
  return std::min(grid_size, num_blocks);
}
}  // namespace

/**
 * @brief Computes all aggregations from `requests` that require a single pass
 * over the data and stores the results in `sparse_results`
 */
template <typename SetType>
rmm::device_uvector<cudf::size_type> compute_single_pass_aggs(
  cudf::table_view const& keys,
  cudf::host_span<cudf::groupby::aggregation_request const> requests,
  cudf::detail::result_cache* sparse_results,
  SetType& global_set,
  bool skip_rows_with_nulls,
  rmm::cuda_stream_view stream)
{
  // GROUPBY_SHM_MAX_ELEMENTS with 0.7 occupancy
  auto constexpr shared_set_capacity =
    static_cast<std::size_t>(static_cast<double>(GROUPBY_SHM_MAX_ELEMENTS) * 1.43);
  using extent_type            = cuco::extent<cudf::size_type, shared_set_capacity>;
  using shared_set_type        = cuco::static_set<cudf::size_type,
                                           extent_type,
                                           cuda::thread_scope_block,
                                           typename SetType::key_equal,
                                           probing_scheme_t,
                                           cuco::cuda_allocator<cudf::size_type>,
                                           cuco::storage<GROUPBY_WINDOW_SIZE>>;
  using shared_set_ref_type    = typename shared_set_type::ref_type<>;
  auto constexpr window_extent = cuco::make_window_extent<shared_set_ref_type>(extent_type{});

  auto const num_input_rows = keys.num_rows();

  auto row_bitmask =
    skip_rows_with_nulls
      ? cudf::detail::bitmask_and(keys, stream, cudf::get_current_device_resource_ref()).first
      : rmm::device_buffer{};

  auto global_set_ref  = global_set.ref(cuco::op::insert_and_find);
  auto const grid_size = max_occupancy_grid_size(
    compute_mapping_indices<shared_set_ref_type, decltype(global_set_ref), decltype(window_extent)>,
    num_input_rows);
  // 'local_mapping_index' maps from the global row index of the input table to the row index of
  // the local pre-aggregate table
  rmm::device_uvector<cudf::size_type> local_mapping_index(num_input_rows, stream);
  // 'global_mapping_index' maps from  the local pre-aggregate table to the row index of
  // global aggregate table
  rmm::device_uvector<cudf::size_type> global_mapping_index(grid_size * GROUPBY_SHM_MAX_ELEMENTS,
                                                            stream);
  rmm::device_uvector<cudf::size_type> block_cardinality(grid_size, stream);
  rmm::device_scalar<bool> direct_aggregations(false, stream);
  compute_mapping_indices<shared_set_ref_type>
    <<<grid_size, GROUPBY_BLOCK_SIZE, 0, stream>>>(global_set_ref,
                                                   num_input_rows,
                                                   window_extent,
                                                   static_cast<bitmask_type*>(row_bitmask.data()),
                                                   skip_rows_with_nulls,
                                                   local_mapping_index.data(),
                                                   global_mapping_index.data(),
                                                   block_cardinality.data(),
                                                   direct_aggregations.data());
  stream.synchronize();

  // 'populated_keys' contains inserted row_indices (keys) of global hash set
  rmm::device_uvector<cudf::size_type> populated_keys(keys.num_rows(), stream);

  // flatten the aggs to a table that can be operated on by aggregate_row
  auto const [flattened_values, agg_kinds, aggs] = flatten_single_pass_aggs(requests);
  auto const d_agg_kinds                         = cudf::detail::make_device_uvector_async(
    agg_kinds, stream, rmm::mr::get_current_device_resource());

  // prepare to launch kernel to do the actual aggregation
  auto d_values = table_device_view::create(flattened_values, stream);

  cudf::table sparse_table =
    compute_aggregations<SetType>(grid_size,
                                  num_input_rows,
                                  static_cast<bitmask_type*>(row_bitmask.data()),
                                  skip_rows_with_nulls,
                                  local_mapping_index.data(),
                                  global_mapping_index.data(),
                                  block_cardinality.data(),
                                  *d_values,
                                  flattened_values,
                                  d_agg_kinds.data(),
                                  agg_kinds,
                                  direct_aggregations.value(stream),
                                  global_set,
                                  populated_keys,
                                  stream);
  auto d_sparse_table = mutable_table_device_view::create(sparse_table, stream);

  if (direct_aggregations.value(stream)) {
    auto const stride = GROUPBY_BLOCK_SIZE * grid_size;
    thrust::for_each_n(rmm::exec_policy(stream),
                       thrust::make_counting_iterator(0),
                       keys.num_rows(),
                       compute_direct_aggregates{global_set_ref,
                                                 *d_values,
                                                 *d_sparse_table,
                                                 d_agg_kinds.data(),
                                                 block_cardinality.data(),
                                                 stride,
                                                 static_cast<bitmask_type*>(row_bitmask.data()),
                                                 skip_rows_with_nulls});
    extract_populated_keys(global_set, populated_keys, stream);
  }

  // Add results back to sparse_results cache
  auto sparse_result_cols = sparse_table.release();
  for (size_t i = 0; i < aggs.size(); i++) {
    // Note that the cache will make a copy of this temporary aggregation
    sparse_results->add_result(
      flattened_values.column(i), *aggs[i], std::move(sparse_result_cols[i]));
  }

  return populated_keys;
}
}  // namespace cudf::groupby::detail::hash
