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

#include "compute_single_pass_aggs.cuh"
#include "compute_single_pass_aggs.hpp"

namespace cudf::groupby::detail::hash {
template std::pair<rmm::device_uvector<size_type>, bool>
compute_single_pass_aggs<nullable_global_set_t>(nullable_global_set_t& global_set,
                                                bitmask_type const* row_bitmask,
                                                host_span<aggregation_request const> requests,
                                                cudf::detail::result_cache* cache,
                                                rmm::cuda_stream_view stream,
                                                rmm::device_async_resource_ref mr);
}  // namespace cudf::groupby::detail::hash
