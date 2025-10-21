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
 * @brief Computes the maximum number of active blocks of the mapping indices kernel that can be
 * executed on the underlying device.
 */
template <class SetRef>
[[nodiscard]] int32_t max_active_blocks_mapping_kernel();

template <class SetRef>
void compute_mapping_indices(size_type grid_size,
                             size_type num_rows,
                             SetRef global_set,
                             bitmask_type const* row_bitmask,
                             size_type* local_mapping_index,
                             size_type* global_mapping_index,
                             size_type* block_cardinality,
                             rmm::cuda_stream_view stream);
}  // namespace cudf::groupby::detail::hash
