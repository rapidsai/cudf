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

#include "compute_global_memory_aggs.hpp"
#include "create_results_table.hpp"
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

namespace cudf::groupby::detail::hash {
void compute_global_memory_aggs(size_type num_rows,
                                size_type num_output,
                                size_type const* key_indices,
                                bitmask_type const* row_bitmask,
                                table_view const& flattened_values,
                                aggregation::Kind const* d_agg_kinds,
                                host_span<aggregation::Kind const> agg_kinds,
                                std::vector<std::unique_ptr<aggregation>>& aggregations,
                                cudf::detail::result_cache* cache,
                                rmm::cuda_stream_view stream)
{
  cudf::scoped_range r("global mem");

  // make table that will hold sparse results
  cudf::table result_table = create_results_table(num_output, flattened_values, agg_kinds, stream);

  // prepare to launch kernel to do the actual aggregation
  auto d_values       = table_device_view::create(flattened_values, stream);
  auto d_result_table = mutable_table_device_view::create(result_table, stream);

  // TODO: change to spass_values
  thrust::for_each_n(
    rmm::exec_policy_nosync(stream),
    thrust::make_counting_iterator(int64_t{0}),
    static_cast<int64_t>(num_rows) * flattened_values.num_columns(),
    compute_single_pass_aggs_fn{key_indices, *d_values, *d_result_table, d_agg_kinds});

  // Add results back to sparse_results cache
  auto result_cols = result_table.release();
  for (size_t i = 0; i < aggregations.size(); i++) {
    // Note that the cache will make a copy of this temporary aggregation
    cache->add_result(flattened_values.column(i), *aggregations[i], std::move(result_cols[i]));
  }

  stream.synchronize();
}
}  // namespace cudf::groupby::detail::hash
