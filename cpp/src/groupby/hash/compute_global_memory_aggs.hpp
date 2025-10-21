/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/detail/aggregation/result_cache.hpp>
#include <cudf/groupby.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <memory>
#include <vector>

namespace cudf::groupby::detail::hash {
template <typename SetType>
rmm::device_uvector<cudf::size_type> compute_global_memory_aggs(
  cudf::size_type num_rows,
  bitmask_type const* row_bitmask,
  cudf::table_view const& flattened_values,
  cudf::aggregation::Kind const* d_agg_kinds,
  host_span<cudf::aggregation::Kind const> agg_kinds,
  SetType& global_set,
  std::vector<std::unique_ptr<aggregation>>& aggregations,
  cudf::detail::result_cache* sparse_results,
  rmm::cuda_stream_view stream);
}  // namespace cudf::groupby::detail::hash
