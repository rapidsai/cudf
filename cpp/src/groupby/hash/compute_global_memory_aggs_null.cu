/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compute_global_memory_aggs.cuh"
#include "compute_global_memory_aggs.hpp"

namespace cudf::groupby::detail::hash {
template rmm::device_uvector<cudf::size_type> compute_global_memory_aggs<nullable_global_set_t>(
  cudf::size_type num_rows,
  bitmask_type const* row_bitmask,
  cudf::table_view const& flattened_values,
  cudf::aggregation::Kind const* d_agg_kinds,
  host_span<cudf::aggregation::Kind const> agg_kinds,
  nullable_global_set_t& global_set,
  std::vector<std::unique_ptr<aggregation>>& aggregations,
  cudf::detail::result_cache* sparse_results,
  rmm::cuda_stream_view stream);
}  // namespace cudf::groupby::detail::hash
