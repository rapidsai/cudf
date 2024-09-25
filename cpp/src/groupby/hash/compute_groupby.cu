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

#include "compute_groupby.cuh"

namespace cudf::groupby::detail::hash {

template void sparse_to_dense_results<hash_set_ref_t>(table_view const& keys,
                                                      host_span<aggregation_request const> requests,
                                                      cudf::detail::result_cache* sparse_results,
                                                      cudf::detail::result_cache* dense_results,
                                                      device_span<size_type const> gather_map,
                                                      hash_set_ref_t set,
                                                      bool skip_key_rows_with_nulls,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::device_async_resource_ref mr);

template std::unique_ptr<table> compute_groupby<row_comparator_t>(
  table_view const& keys,
  host_span<aggregation_request const> requests,
  cudf::detail::result_cache* cache,
  bool skip_key_rows_with_nulls,
  row_comparator_t const& d_row_equal,
  row_hash_t const& d_row_hash,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

}  // namespace cudf::groupby::detail::hash
