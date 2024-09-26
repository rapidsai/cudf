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

#include "compute_single_pass_aggs.cuh"
#include "helpers.cuh"

namespace cudf {
namespace groupby {
namespace detail {
namespace hash {

using global_set_t = cuco::static_set<cudf::size_type,
                                      cuco::extent<int64_t>,
                                      cuda::thread_scope_device,
                                      nullable_row_comparator_t,
                                      probing_scheme_t,
                                      cudf::detail::cuco_allocator<char>,
                                      cuco::storage<GROUPBY_WINDOW_SIZE>>;

template void extract_populated_keys<global_set_t>(
  global_set_t const& key_set,
  rmm::device_uvector<cudf::size_type>& populated_keys,
  rmm::cuda_stream_view stream);

template auto create_sparse_results_table<global_set_t>(
  cudf::table_view const& flattened_values,
  cudf::aggregation::Kind const* d_agg_kinds,
  std::vector<cudf::aggregation::Kind> aggs,
  bool direct_aggregations,
  global_set_t const& global_set,
  rmm::device_uvector<cudf::size_type>& populated_keys,
  rmm::cuda_stream_view stream);

template rmm::device_uvector<cudf::size_type> compute_single_pass_aggs<global_set_t>(
  cudf::table_view const& keys,
  cudf::host_span<cudf::groupby::aggregation_request const> requests,
  cudf::detail::result_cache* sparse_results,
  global_set_t& global_set,
  bool skip_rows_with_nulls,
  rmm::cuda_stream_view stream);

}  // namespace hash
}  // namespace detail
}  // namespace groupby
}  // namespace cudf
