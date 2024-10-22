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

#include "compute_mapping_indices.cuh"
#include "compute_mapping_indices.hpp"

namespace cudf::groupby::detail::hash {
template cudf::size_type max_occupancy_grid_size<hash_set_ref_t<cuco::insert_and_find_tag>>(
  cudf::size_type n);

template void compute_mapping_indices<hash_set_ref_t<cuco::insert_and_find_tag>>(
  cudf::size_type grid_size,
  cudf::size_type num,
  hash_set_ref_t<cuco::insert_and_find_tag> global_set,
  bitmask_type const* row_bitmask,
  bool skip_rows_with_nulls,
  cudf::size_type* local_mapping_index,
  cudf::size_type* global_mapping_index,
  cudf::size_type* block_cardinality,
  cuda::std::atomic_flag* needs_global_memory_fallback,
  rmm::cuda_stream_view stream);
}  // namespace cudf::groupby::detail::hash
