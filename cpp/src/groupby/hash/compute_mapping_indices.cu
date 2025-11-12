/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compute_mapping_indices.cuh"
#include "compute_mapping_indices.hpp"

namespace cudf::groupby::detail::hash {
template int32_t max_active_blocks_mapping_kernel<hash_set_ref_t<cuco::insert_and_find_tag>>();

template void compute_mapping_indices<hash_set_ref_t<cuco::insert_and_find_tag>>(
  size_type grid_size,
  size_type num_rows,
  hash_set_ref_t<cuco::insert_and_find_tag> global_set,
  bitmask_type const* row_bitmask,
  size_type* local_mapping_index,
  size_type* global_mapping_index,
  size_type* block_cardinality,
  rmm::cuda_stream_view stream);
}  // namespace cudf::groupby::detail::hash
