/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/aggregation.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/types.hpp>

#include <cuda/stream>

#include <cstdint>

namespace cudf::groupby::detail::hash {

/*
 * @brief Computes the maximum number of active blocks of the shared memory aggregation kernel that
 * can be executed on the underlying device.
 */
int32_t max_active_blocks_shmem_aggs_kernel();

size_type get_available_shared_memory_size(size_type grid_size);

size_type constexpr compute_shmem_offsets_size(size_type num_cols)
{
  return static_cast<size_type>(sizeof(size_type) * num_cols);
}

void compute_shared_memory_aggs(size_type grid_size,
                                size_type available_shmem_size,
                                size_type num_input_rows,
                                bitmask_type const* row_bitmask,
                                size_type* local_mapping_index,
                                size_type* global_mapping_index,
                                size_type* block_cardinality,
                                table_device_view input_values,
                                mutable_table_device_view output_values,
                                aggregation::Kind const* d_agg_kinds,
                                cuda::stream_ref stream);

/**
 * @brief Aggregates a sparse-to-dense row mapping into replica-private global-memory partials,
 * then reduces the partials into the final output.
 */
void compute_block_private_aggs(size_type grid_size,
                                size_type block_size,
                                size_type num_replicas,
                                size_type num_input_rows,
                                size_type const* matching_keys,
                                size_type const* key_transform_map,
                                table_device_view input_values,
                                mutable_table_device_view output_values,
                                mutable_table_device_view private_values,
                                size_type num_output_rows,
                                aggregation::Kind const* d_agg_kinds,
                                cuda::stream_ref stream);

/** @brief Aggregates a sparse-to-dense row mapping directly into the final dense output. */
void compute_mapped_global_aggs(size_type grid_size,
                                size_type block_size,
                                size_type num_input_rows,
                                size_type const* matching_keys,
                                size_type const* key_transform_map,
                                table_device_view input_values,
                                mutable_table_device_view output_values,
                                size_type num_output_rows,
                                aggregation::Kind const* d_agg_kinds,
                                cuda::stream_ref stream);
}  // namespace cudf::groupby::detail::hash
