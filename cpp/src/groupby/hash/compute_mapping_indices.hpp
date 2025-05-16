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

#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuda/std/atomic>

namespace cudf::groupby::detail::hash {
/*
 * @brief Computes the maximum number of active blocks of the given kernel that can be executed on
 * the underlying device
 */
template <class SetRef>
[[nodiscard]] cudf::size_type max_occupancy_grid_size(cudf::size_type n);

template <class SetRef>
void compute_mapping_indices(cudf::size_type grid_size,
                             cudf::size_type num,
                             SetRef global_set,
                             bitmask_type const* row_bitmask,
                             bool skip_rows_with_nulls,
                             cudf::size_type* local_mapping_index,
                             cudf::size_type* global_mapping_index,
                             cudf::size_type* block_cardinality,
                             cuda::std::atomic_flag* needs_global_memory_fallback,
                             rmm::cuda_stream_view stream);
}  // namespace cudf::groupby::detail::hash
