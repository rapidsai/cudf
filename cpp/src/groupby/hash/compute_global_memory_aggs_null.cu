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

#include "compute_global_memory_aggs.cuh"

namespace cudf::groupby::detail::hash {

template std::pair<std::unique_ptr<table>, rmm::device_uvector<size_type>>
compute_global_memory_aggs<nullable_global_set_t>(bitmask_type const* row_bitmask,
                                                  table_view const& values,
                                                  nullable_global_set_t const& key_set,
                                                  host_span<aggregation::Kind const> h_agg_kinds,
                                                  device_span<aggregation::Kind const> d_agg_kinds,
                                                  rmm::cuda_stream_view stream,
                                                  rmm::device_async_resource_ref mr);

}  // namespace cudf::groupby::detail::hash
