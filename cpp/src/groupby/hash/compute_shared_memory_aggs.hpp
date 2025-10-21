/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/aggregation.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf::groupby::detail::hash {
size_type get_available_shared_memory_size(cudf::size_type grid_size);

size_type constexpr compute_shmem_offsets_size(cudf::size_type num_cols)
{
  return static_cast<size_type>(sizeof(cudf::size_type) * num_cols);
}

void compute_shared_memory_aggs(cudf::size_type grid_size,
                                cudf::size_type available_shmem_size,
                                cudf::size_type num_input_rows,
                                bitmask_type const* row_bitmask,
                                cudf::size_type* local_mapping_index,
                                cudf::size_type* global_mapping_index,
                                cudf::size_type* block_cardinality,
                                cudf::table_device_view input_values,
                                cudf::mutable_table_device_view output_values,
                                cudf::aggregation::Kind const* d_agg_kinds,
                                rmm::cuda_stream_view stream);
}  // namespace cudf::groupby::detail::hash
