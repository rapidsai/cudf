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

#include "compute_global_memory_aggs.hpp"
#include "create_sparse_results_table.hpp"
#include "flatten_single_pass_aggs.hpp"
#include "helpers.cuh"
#include "single_pass_functors.cuh"

#include <cudf/detail/aggregation/result_cache.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/groupby.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuco/static_set.cuh>
#include <thrust/for_each.h>

#include <memory>
#include <vector>

namespace cudf::groupby::detail::hash {
void compute_global_memory_aggs(cudf::size_type num_rows,
                                bool skip_rows_with_nulls,
                                bitmask_type const* row_bitmask,
                                cudf::table_view const& flattened_values,
                                cudf::aggregation::Kind const* d_agg_kinds,
                                host_span<cudf::aggregation::Kind const> agg_kinds,
                                cudf::device_span<cudf::size_type const> populated_keys,
                                cudf::size_type const* key_indices,
                                std::vector<std::unique_ptr<aggregation>>& aggregations,
                                cudf::detail::result_cache* sparse_results,
                                rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();

  auto constexpr uses_global_memory_aggs = true;

  // make table that will hold sparse results
  cudf::table sparse_table = create_sparse_results_table(
    flattened_values, d_agg_kinds, agg_kinds, uses_global_memory_aggs, populated_keys, stream);

  // prepare to launch kernel to do the actual aggregation
  auto d_values       = table_device_view::create(flattened_values, stream);
  auto d_sparse_table = mutable_table_device_view::create(sparse_table, stream);

  thrust::for_each_n(
    rmm::exec_policy_nosync(stream),
    thrust::make_counting_iterator(int64_t{0}),
    num_rows * static_cast<int64_t>(flattened_values.num_columns()),
    hash::compute_single_pass_aggs_fn{
      key_indices, *d_values, *d_sparse_table, d_agg_kinds, row_bitmask, skip_rows_with_nulls});

  // Add results back to sparse_results cache
  auto sparse_result_cols = sparse_table.release();
  for (size_t i = 0; i < aggregations.size(); i++) {
    // Note that the cache will make a copy of this temporary aggregation
    sparse_results->add_result(
      flattened_values.column(i), *aggregations[i], std::move(sparse_result_cols[i]));
  }
}
}  // namespace cudf::groupby::detail::hash
