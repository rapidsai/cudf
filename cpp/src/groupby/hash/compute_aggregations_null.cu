/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compute_aggregations.cuh"
#include "compute_aggregations.hpp"

namespace cudf::groupby::detail::hash {
template rmm::device_uvector<cudf::size_type> compute_aggregations<nullable_global_set_t>(
  int64_t num_rows,
  bitmask_type const* row_bitmask,
  nullable_global_set_t& global_set,
  cudf::host_span<cudf::groupby::aggregation_request const> requests,
  cudf::detail::result_cache* sparse_results,
  rmm::cuda_stream_view stream);
}  // namespace cudf::groupby::detail::hash
