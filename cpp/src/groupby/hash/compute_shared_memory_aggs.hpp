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

#include <cudf/aggregation.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf::groupby::detail::hash {
std::size_t get_available_shared_memory_size(cudf::size_type grid_size);

std::size_t constexpr compute_shmem_offsets_size(cudf::size_type num_cols)
{
  return sizeof(cudf::size_type) * num_cols;
}

void compute_shared_memory_aggs(cudf::size_type grid_size,
                                std::size_t available_shmem_size,
                                cudf::size_type num_input_rows,
                                bitmask_type const* row_bitmask,
                                bool skip_rows_with_nulls,
                                cudf::size_type* local_mapping_index,
                                cudf::size_type* global_mapping_index,
                                cudf::size_type* block_cardinality,
                                cudf::table_device_view input_values,
                                cudf::mutable_table_device_view output_values,
                                cudf::aggregation::Kind const* d_agg_kinds,
                                rmm::cuda_stream_view stream);
}  // namespace cudf::groupby::detail::hash
