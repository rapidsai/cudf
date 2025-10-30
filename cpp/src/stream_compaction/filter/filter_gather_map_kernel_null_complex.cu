/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "filter_gather_map_kernel.cuh"
#include "filter_gather_map_kernel.hpp"

namespace cudf::detail {
template void launch_filter_gather_map_kernel<true, true>(
  cudf::table_device_view const& left_table,
  cudf::table_device_view const& right_table,
  cudf::device_span<cudf::size_type const> left_indices,
  cudf::device_span<cudf::size_type const> right_indices,
  cudf::ast::detail::expression_device_view device_expression_data,
  cudf::null_policy null_handling,
  cudf::detail::grid_1d const& config,
  std::size_t shmem_per_block,
  bool* output_flags,
  rmm::cuda_stream_view stream);
}  // namespace cudf::detail
