/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuda/std/atomic>

namespace cudf::groupby::detail::hash {

/*
 * @brief Computes the maximum number of active blocks of the shared memory aggregation kernel that
 * can be executed on the underlying device.
 */
int32_t max_active_blocks_shmem_aggs_kernel();

/*
 * @brief Computes the maximum number of active blocks of the mapping indices kernel that can be
 * executed on the underlying device.
 */
template <class SetRef>
[[nodiscard]] int32_t max_active_blocks_mapping_kernel();

template <class SetRef>
void compute_mapping_indices(cudf::size_type grid_size,
                             cudf::size_type num,
                             SetRef global_set,
                             bitmask_type const* row_bitmask,
                             cudf::size_type* local_mapping_index,
                             cudf::size_type* global_mapping_index,
                             cudf::size_type* block_cardinality,
                             cuda::std::atomic_flag* needs_global_memory_fallback,
                             rmm::cuda_stream_view stream);
}  // namespace cudf::groupby::detail::hash
