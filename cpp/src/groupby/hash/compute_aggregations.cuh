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

#include "compute_aggregations.hpp"
#include "compute_global_memory_aggs.hpp"
#include "compute_mapping_indices.hpp"
#include "compute_shared_memory_aggs.hpp"
#include "create_sparse_results_table.hpp"
#include "flatten_single_pass_aggs.hpp"
#include "helpers.cuh"
#include "single_pass_functors.cuh"

#include <cudf/detail/aggregation/result_cache.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/groupby.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuco/static_set.cuh>
#include <cuda/std/atomic>
#include <thrust/for_each.h>

#include <algorithm>
#include <memory>
#include <vector>

namespace cudf::groupby::detail::hash {
/**
 * @brief Computes all aggregations from `requests` that require a single pass
 * over the data and stores the results in `sparse_results`
 */
template <typename SetType>
rmm::device_uvector<cudf::size_type> compute_aggregations(
  int64_t num_rows,
  bool skip_rows_with_nulls,
  bitmask_type const* row_bitmask,
  SetType& global_set,
  cudf::host_span<cudf::groupby::aggregation_request const> requests,
  cudf::detail::result_cache* sparse_results,
  rmm::cuda_stream_view stream)
{
  // flatten the aggs to a table that can be operated on by aggregate_row
  auto [flattened_values, agg_kinds, aggs] = flatten_single_pass_aggs(requests);
  auto const d_agg_kinds                   = cudf::detail::make_device_uvector_async(
    agg_kinds, stream, rmm::mr::get_current_device_resource());

  auto const grid_size =
    max_occupancy_grid_size<typename SetType::ref_type<cuco::insert_and_find_tag>>(num_rows);
  auto const available_shmem_size = get_available_shared_memory_size(grid_size);
  auto const has_sufficient_shmem =
    available_shmem_size > (compute_shmem_offsets_size(flattened_values.num_columns()) * 2);
  auto const has_dictionary_request = std::any_of(
    requests.begin(), requests.end(), [](cudf::groupby::aggregation_request const& request) {
      return cudf::is_dictionary(request.values.type());
    });
  auto const is_shared_memory_compatible = !has_dictionary_request and has_sufficient_shmem;

  // Performs naive global memory aggregations when the workload is not compatible with shared
  // memory, such as when aggregating dictionary columns or when there is insufficient dynamic
  // shared memory for shared memory aggregations.
  if (!is_shared_memory_compatible) {
    return compute_global_memory_aggs(num_rows,
                                      skip_rows_with_nulls,
                                      row_bitmask,
                                      flattened_values,
                                      d_agg_kinds.data(),
                                      agg_kinds,
                                      global_set,
                                      aggs,
                                      sparse_results,
                                      stream);
  }

  // 'populated_keys' contains inserted row_indices (keys) of global hash set
  rmm::device_uvector<cudf::size_type> populated_keys(num_rows, stream);
  // 'local_mapping_index' maps from the global row index of the input table to its block-wise rank
  rmm::device_uvector<cudf::size_type> local_mapping_index(num_rows, stream);
  // 'global_mapping_index' maps from the block-wise rank to the row index of global aggregate table
  rmm::device_uvector<cudf::size_type> global_mapping_index(grid_size * GROUPBY_SHM_MAX_ELEMENTS,
                                                            stream);
  rmm::device_uvector<cudf::size_type> block_cardinality(grid_size, stream);

  // Flag indicating whether a global memory aggregation fallback is required or not
  rmm::device_scalar<cuda::std::atomic_flag> needs_global_memory_fallback(stream);

  auto global_set_ref = global_set.ref(cuco::op::insert_and_find);

  compute_mapping_indices(grid_size,
                          num_rows,
                          global_set_ref,
                          row_bitmask,
                          skip_rows_with_nulls,
                          local_mapping_index.data(),
                          global_mapping_index.data(),
                          block_cardinality.data(),
                          needs_global_memory_fallback.data(),
                          stream);

  cuda::std::atomic_flag h_needs_fallback;
  // Cannot use `device_scalar::value` as it requires a copy constructor, which
  // `atomic_flag` doesn't have.
  CUDF_CUDA_TRY(cudaMemcpyAsync(&h_needs_fallback,
                                needs_global_memory_fallback.data(),
                                sizeof(cuda::std::atomic_flag),
                                cudaMemcpyDefault,
                                stream.value()));
  stream.synchronize();
  auto const needs_fallback = h_needs_fallback.test();

  // make table that will hold sparse results
  cudf::table sparse_table = create_sparse_results_table(flattened_values,
                                                         d_agg_kinds.data(),
                                                         agg_kinds,
                                                         needs_fallback,
                                                         global_set,
                                                         populated_keys,
                                                         stream);
  // prepare to launch kernel to do the actual aggregation
  auto d_values       = table_device_view::create(flattened_values, stream);
  auto d_sparse_table = mutable_table_device_view::create(sparse_table, stream);

  compute_shared_memory_aggs(grid_size,
                             available_shmem_size,
                             num_rows,
                             row_bitmask,
                             skip_rows_with_nulls,
                             local_mapping_index.data(),
                             global_mapping_index.data(),
                             block_cardinality.data(),
                             *d_values,
                             *d_sparse_table,
                             d_agg_kinds.data(),
                             stream);

  // The shared memory groupby is designed so that each thread block can handle up to 128 unique
  // keys. When a block reaches this cardinality limit, shared memory becomes insufficient to store
  // the temporary aggregation results. In these situations, we must fall back to a global memory
  // aggregator to process the remaining aggregation requests.
  if (needs_fallback) {
    auto const stride = GROUPBY_BLOCK_SIZE * grid_size;
    thrust::for_each_n(rmm::exec_policy_nosync(stream),
                       thrust::counting_iterator{0},
                       num_rows,
                       global_memory_fallback_fn{global_set_ref,
                                                 *d_values,
                                                 *d_sparse_table,
                                                 d_agg_kinds.data(),
                                                 block_cardinality.data(),
                                                 stride,
                                                 row_bitmask,
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
