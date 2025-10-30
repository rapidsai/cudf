/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/ast/detail/expression_parser.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf::detail {
/**
 * @brief Maximum number of threads per block for filter gather map kernels
 */
size_type constexpr MAX_BLOCK_SIZE = 256;

/**
 * @brief Launches the appropriate filter gather map kernel based on template parameters
 *
 * This function selects and launches the appropriate kernel implementation based on whether
 * the expression may evaluate to null and whether the expression contains complex types.
 *
 * @tparam has_nulls Indicates whether the expression may evaluate to null
 * @tparam has_complex_type Indicates whether the expression may contain complex types
 *
 * @param left_table Device view of the left table
 * @param right_table Device view of the right table
 * @param left_indices Device span of left table indices
 * @param right_indices Device span of right table indices
 * @param device_expression_data Device data required to evaluate the expression
 * @param null_handling Policy for handling null indices (INCLUDE or EXCLUDE)
 * @param config Grid configuration for kernel launch
 * @param shmem_per_block Amount of shared memory to allocate per block
 * @param output_flags Output array to mark valid index pairs
 * @param stream CUDA stream on which to launch the kernel
 */
template <bool has_nulls, bool has_complex_type>
void launch_filter_gather_map_kernel(
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
