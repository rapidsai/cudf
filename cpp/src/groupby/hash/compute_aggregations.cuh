/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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
#include "compute_global_memory_aggs.hpp"
#include "compute_mapping_indices.hpp"
#include "compute_shared_memory_aggs.hpp"
#include "create_results_table.hpp"
#include "flatten_single_pass_aggs.hpp"
#include "hash_compound_agg_finalizer.hpp"
#include "helpers.cuh"
#include "single_pass_functors.cuh"

#include <cudf/detail/aggregation/result_cache.hpp>
#include <cudf/detail/utilities/cuda.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/groupby.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_select.cuh>
#include <cuco/static_set.cuh>
#include <cuda/std/atomic>
#include <thrust/for_each.h>
#include <thrust/gather.h>
#include <thrust/tabulate.h>
#include <thrust/uninitialized_fill.h>

namespace cudf::groupby::detail::hash {

/**
 * @brief Computes all aggregations from `requests` that require a single pass
 * over the data and stores the results in `sparse_results`
 */
template <typename SetType>
std::pair<rmm::device_uvector<size_type>, rmm::device_uvector<size_type>> compute_aggregations(
  int64_t num_rows,
  bitmask_type const* row_bitmask,
  SetType& global_set,
  host_span<aggregation_request const> requests,
  cudf::detail::result_cache* cache,
  rmm::cuda_stream_view stream)
{
  // Collect the single-pass aggregations that can be processed separately before we can calculate
  // the compound aggregations.
  auto [spass_values, spass_agg_kinds, spass_aggs] = flatten_single_pass_aggs(requests, stream);
  auto const d_spass_agg_kinds                     = cudf::detail::make_device_uvector_async(
    spass_agg_kinds, stream, rmm::mr::get_current_device_resource());

  auto const grid_size = [&] {
    auto const max_blocks_mapping =
      max_active_blocks_mapping_kernel<typename SetType::ref_type<cuco::insert_and_find_tag>>();
    auto const max_blocks_aggs = max_active_blocks_shmem_aggs_kernel();
    // We launch the same grid size for both kernels, thus we need to take the minimum of the two.
    auto const max_blocks    = std::min(max_blocks_mapping, max_blocks_aggs);
    auto const max_grid_size = max_blocks * cudf::detail::num_multiprocessors();
    auto const num_blocks =
      cudf::util::div_rounding_up_safe(static_cast<size_type>(num_rows), GROUPBY_BLOCK_SIZE);
    return std::min(max_grid_size, num_blocks);
  }();
  auto const available_shmem_size = get_available_shared_memory_size(grid_size);
  auto const offsets_buffer_size  = compute_shmem_offsets_size(spass_values.num_columns()) * 2;
  auto const data_buffer_size     = available_shmem_size - offsets_buffer_size;

  // Check if any aggregation is SUM_WITH_OVERFLOW, which should always use global memory
  auto const has_sum_with_overflow =
    std::any_of(spass_agg_kinds.begin(), spass_agg_kinds.end(), [](aggregation::Kind k) {
      return k == aggregation::SUM_WITH_OVERFLOW;
    });

  auto const is_shared_memory_compatible =
    !has_sum_with_overflow &&
    std::all_of(requests.begin(), requests.end(), [&](aggregation_request const& request) {
      if (is_dictionary(request.values.type())) { return false; }
      // Ensure there is enough buffer space to store local aggregations up to the max cardinality
      // for shared memory aggregations
      auto const size =
        type_dispatcher<dispatch_storage_type>(request.values.type(), size_of_functor{});
      return data_buffer_size >= size * GROUPBY_CARDINALITY_THRESHOLD;
    });

  // 'populated_keys' contains inserted row_indices (keys) of global hash set
  rmm::device_uvector<cudf::size_type> populated_keys(num_rows, stream);

  // TODO
  rmm::device_uvector<cudf::size_type> key_indices(num_rows, stream);
  thrust::uninitialized_fill(rmm::exec_policy_nosync(stream),
                             key_indices.begin(),
                             key_indices.end(),
                             cudf::detail::CUDF_SIZE_TYPE_SENTINEL);

  auto global_set_ref = global_set.ref(cuco::op::insert_and_find);

  // Performs naive global memory aggregations when the workload is not compatible with shared
  // memory, such as when aggregating dictionary columns, when there is insufficient dynamic
  // shared memory for shared memory aggregations, or when SUM_WITH_OVERFLOW aggregations are
  // present.
  if (!is_shared_memory_compatible) {
    thrust::tabulate(rmm::exec_policy_nosync(stream),
                     key_indices.begin(),
                     key_indices.end(),
                     [global_set_ref, row_bitmask] __device__(auto const idx) mutable {
                       if (!row_bitmask || cudf::bit_is_set(row_bitmask, idx)) {
                         return *global_set_ref.insert_and_find(idx).first;
                       }
                       return cudf::detail::CUDF_SIZE_TYPE_SENTINEL;
                     });

    extract_populated_keys(global_set, populated_keys, stream);
    find_output_indices(key_indices, populated_keys, stream);

    // TODO
    compute_global_memory_aggs(num_rows,
                               static_cast<size_type>(populated_keys.size()),
                               key_indices.begin(),
                               row_bitmask,
                               spass_values,
                               d_spass_agg_kinds.data(),
                               spass_agg_kinds,
                               spass_aggs,
                               cache,
                               stream);
    return std::pair{std::move(populated_keys), std::move(key_indices)};
  }

  // 'local_mapping_index' maps from the global row index of the input table to its block-wise rank
  rmm::device_uvector<cudf::size_type> local_mapping_index(num_rows, stream);
  // 'global_mapping_index' maps from the block-wise rank to the row index of global aggregate table
  rmm::device_uvector<cudf::size_type> global_mapping_index(grid_size * GROUPBY_SHM_MAX_ELEMENTS,
                                                            stream);
  thrust::uninitialized_fill(rmm::exec_policy_nosync(stream),
                             global_mapping_index.begin(),
                             global_mapping_index.end(),
                             cudf::detail::CUDF_SIZE_TYPE_SENTINEL);

  rmm::device_uvector<cudf::size_type> block_cardinality(grid_size, stream);

  // Flag indicating whether a global memory aggregation fallback is required or not
  rmm::device_scalar<cuda::std::atomic_flag> needs_global_memory_fallback(stream);
  CUDF_CUDA_TRY(cudaMemsetAsync(
    needs_global_memory_fallback.data(), 0, sizeof(cuda::std::atomic_flag), stream.value()));

  compute_mapping_indices(grid_size,
                          num_rows,
                          global_set_ref,
                          row_bitmask,
                          local_mapping_index.data(),
                          global_mapping_index.data(),
                          block_cardinality.data(),
                          needs_global_memory_fallback.data(),
                          stream);

  // For the thread blocks that need fallback to the code path using global memory aggregation,
  // we need to collect these block ids.
  rmm::device_uvector<cudf::size_type> fallback_block_ids(grid_size, stream);
  rmm::device_scalar<cudf::size_type> d_num_fallback_blocks(stream);
  size_type num_fallback_blocks = 0;
  {
    auto const select_cond = [block_cardinality =
                                block_cardinality.begin()] __device__(auto const idx) {
      if (block_cardinality[idx] >= GROUPBY_CARDINALITY_THRESHOLD) {}
      return block_cardinality[idx] >= GROUPBY_CARDINALITY_THRESHOLD;
    };

    size_t storage_bytes = 0;
    cub::DeviceSelect::If(nullptr,
                          storage_bytes,
                          thrust::make_counting_iterator(0),
                          fallback_block_ids.begin(),
                          d_num_fallback_blocks.data(),
                          grid_size,
                          select_cond,
                          stream.value());
    rmm::device_buffer tmp_storage(storage_bytes, stream);
    cub::DeviceSelect::If(tmp_storage.data(),
                          storage_bytes,
                          thrust::make_counting_iterator(0),
                          fallback_block_ids.begin(),
                          d_num_fallback_blocks.data(),
                          grid_size,
                          select_cond,
                          stream.value());
  }

  cuda::std::atomic_flag h_needs_fallback;
  // Cannot use `device_scalar::value` as it requires a copy constructor, which
  // `atomic_flag` doesn't have.
  CUDF_CUDA_TRY(cudaMemcpyAsync(&h_needs_fallback,
                                needs_global_memory_fallback.data(),
                                sizeof(cuda::std::atomic_flag),
                                cudaMemcpyDefault,
                                stream.value()));

  CUDF_CUDA_TRY(cudaMemcpyAsync(&num_fallback_blocks,
                                d_num_fallback_blocks.data(),
                                sizeof(cudf::size_type),
                                cudaMemcpyDefault,
                                stream.value()));

  stream.synchronize();
  auto const needs_fallback = h_needs_fallback.test();

  // TODO
  if (needs_fallback) {
    // TODO: this is a repeat code
    if (num_fallback_blocks == grid_size) {
      thrust::tabulate(rmm::exec_policy_nosync(stream),
                       key_indices.begin(),
                       key_indices.end(),
                       [global_set_ref, row_bitmask] __device__(auto const idx) mutable {
                         if (!row_bitmask || cudf::bit_is_set(row_bitmask, idx)) {
                           return *global_set_ref.insert_and_find(idx).first;
                         }
                         return cudf::detail::CUDF_SIZE_TYPE_SENTINEL;
                       });

      extract_populated_keys(global_set, populated_keys, stream);
      find_output_indices(key_indices, populated_keys, stream);

      // TODO
      compute_global_memory_aggs(num_rows,
                                 static_cast<size_type>(populated_keys.size()),
                                 key_indices.begin(),
                                 row_bitmask,
                                 spass_values,
                                 d_spass_agg_kinds.data(),
                                 spass_agg_kinds,
                                 spass_aggs,
                                 cache,
                                 stream);
      return std::pair{std::move(populated_keys), std::move(key_indices)};
    }

    auto const num_strides =
      util::div_rounding_up_safe(static_cast<size_type>(num_rows), GROUPBY_BLOCK_SIZE * grid_size);

#if 1
    thrust::for_each_n(rmm::exec_policy_nosync(stream),
                       thrust::make_counting_iterator(0),
                       GROUPBY_BLOCK_SIZE * num_fallback_blocks * num_strides,
                       [num_rows    = static_cast<size_type>(num_rows),
                        full_stride = GROUPBY_BLOCK_SIZE * grid_size,
                        stride      = GROUPBY_BLOCK_SIZE * num_fallback_blocks,
                        global_set_ref,
                        key_indices        = key_indices.begin(),
                        fallback_block_ids = fallback_block_ids.begin(),
                        row_bitmask] __device__(auto const idx) mutable {
                         auto const idx_in_stride = idx % stride;
                         auto const thread_rank   = idx_in_stride % GROUPBY_BLOCK_SIZE;
                         auto const block_idx =
                           fallback_block_ids[idx_in_stride / GROUPBY_BLOCK_SIZE];
                         auto const row_idx = full_stride * (idx / stride) +
                                              GROUPBY_BLOCK_SIZE * block_idx + thread_rank;

                         if (row_idx >= num_rows) { return; }

                         if (!row_bitmask || cudf::bit_is_set(row_bitmask, row_idx)) {
                           key_indices[row_idx] = *global_set_ref.insert_and_find(row_idx).first;
                         }
                       });

#else
    thrust::tabulate(rmm::exec_policy_nosync(stream),
                     key_indices.begin(),
                     key_indices.end(),
                     [global_set_ref, row_bitmask] __device__(auto const idx) mutable {
                       if (!row_bitmask || cudf::bit_is_set(row_bitmask, idx)) {
                         return *global_set_ref.insert_and_find(idx).first;
                       }
                       return cudf::detail::CUDF_SIZE_TYPE_SENTINEL;
                     });

#endif
  }
  extract_populated_keys(global_set, populated_keys, stream);

  // TODO: update global_mapping_index with the new indices
  auto const new_key_indices = find_output_indices(key_indices, populated_keys, stream);

  thrust::for_each_n(
    rmm::exec_policy_nosync(stream),
    thrust::make_counting_iterator(0),
    grid_size * GROUPBY_BLOCK_SIZE,
    [new_key_indices      = new_key_indices.begin(),
     global_mapping_index = global_mapping_index.begin()] __device__(auto const idx) {
      auto const block_id    = idx / GROUPBY_BLOCK_SIZE;
      auto const thread_rank = idx % GROUPBY_BLOCK_SIZE;
      auto const mapping_idx = block_id * GROUPBY_SHM_MAX_ELEMENTS + thread_rank;
      auto const old_idx     = global_mapping_index[mapping_idx];
      if (old_idx != cudf::detail::CUDF_SIZE_TYPE_SENTINEL) {
        global_mapping_index[mapping_idx] = new_key_indices[old_idx];
      }
    });

  // make table that will hold sparse results
  cudf::table result_table = create_results_table(
    static_cast<size_type>(populated_keys.size()), spass_values, spass_agg_kinds, stream);

  // prepare to launch kernel to do the actual aggregation
  auto d_values       = table_device_view::create(spass_values, stream);
  auto d_sparse_table = mutable_table_device_view::create(result_table, stream);

  compute_shared_memory_aggs(grid_size,
                             available_shmem_size,
                             num_rows,
                             row_bitmask,
                             local_mapping_index.data(),
                             global_mapping_index.data(),
                             block_cardinality.data(),
                             *d_values,
                             *d_sparse_table,
                             d_spass_agg_kinds.data(),
                             stream);

  // The shared memory groupby is designed so that each thread block can handle up to 128 unique
  // keys. When a block reaches this cardinality limit, shared memory becomes insufficient to store
  // the temporary aggregation results. In these situations, we must fall back to a global memory
  // aggregator to process the remaining aggregation requests.
  if (needs_fallback) {
    auto const num_strides =
      util::div_rounding_up_safe(static_cast<size_type>(num_rows), GROUPBY_BLOCK_SIZE * grid_size);
    auto const full_stride         = GROUPBY_BLOCK_SIZE * grid_size;
    auto const stride              = GROUPBY_BLOCK_SIZE * num_fallback_blocks;
    auto const num_processing_rows = GROUPBY_BLOCK_SIZE * num_fallback_blocks * num_strides;
    thrust::for_each_n(rmm::exec_policy_nosync(stream),
                       thrust::make_counting_iterator(int64_t{0}),
                       static_cast<int64_t>(num_processing_rows) * spass_values.num_columns(),
                       global_memory_fallback_fn{key_indices.begin(),
                                                 *d_values,
                                                 *d_sparse_table,
                                                 d_spass_agg_kinds.data(),
                                                 fallback_block_ids.data(),
                                                 stride,
                                                 num_strides,
                                                 full_stride,
                                                 num_processing_rows,
                                                 static_cast<size_type>(num_rows)});
  }

  // Add results back to sparse_results cache
  auto result_cols = result_table.release();
  for (size_t i = 0; i < spass_aggs.size(); i++) {
    // Note that the cache will make a copy of this temporary aggregation
    cache->add_result(spass_values.column(i), *spass_aggs[i], std::move(result_cols[i]));
  }

  return std::pair{std::move(populated_keys), std::move(key_indices)};
}
}  // namespace cudf::groupby::detail::hash
