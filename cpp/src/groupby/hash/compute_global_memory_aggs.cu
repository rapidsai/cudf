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

#include "compute_global_memory_aggs.cuh"
#include "compute_global_memory_aggs.hpp"

namespace cudf::groupby::detail::hash {
template rmm::device_uvector<cudf::size_type> compute_global_memory_aggs<global_set_t>(
  cudf::size_type num_rows,
  bool skip_rows_with_nulls,
  bitmask_type const* row_bitmask,
  cudf::table_view const& flattened_values,
  cudf::aggregation::Kind const* d_agg_kinds,
  host_span<cudf::aggregation::Kind const> agg_kinds,
  global_set_t& global_set,
  std::vector<std::unique_ptr<aggregation>>& aggregations,
  cudf::detail::result_cache* sparse_results,
  rmm::cuda_stream_view stream);
}  // namespace cudf::groupby::detail::hash
