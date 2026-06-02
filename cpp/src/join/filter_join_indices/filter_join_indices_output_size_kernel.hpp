/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "join/filter_join_indices/filter_join_indices_kernel.hpp"

#include <cudf/ast/detail/expression_parser.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/join/join.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cstddef>

namespace cudf::detail {

/**
 * @brief Launches a kernel that counts the per-join-kind output size for `filter_join_indices`.
 *
 * For INNER_JOIN this is the number of pairs whose predicate evaluates to true.
 * For LEFT_JOIN this is the number of input pairs whose predicate evaluates to true
 * (including pre-existing unmatched pairs that are preserved); additionally,
 * `left_passing_marks[left_row_index]` is set to `true` for every left row that
 * contributes to that count (used by the host code to derive the number of left
 * rows that need a synthetic JoinNoMatch entry).
 * For FULL_JOIN this is the number of failed matched pairs (predicate false and
 * both indices valid), which is added on top of `left_indices.size()` host-side.
 *
 * The kernel avoids materializing a per-pair boolean buffer; it folds the count
 * directly into `count_out` via atomic increments.
 *
 * @tparam has_nulls Indicates whether the expression may evaluate to null
 * @tparam has_complex_type Indicates whether the expression may contain complex types
 *
 * @param[in] left_table Device view of the left table
 * @param[in] right_table Device view of the right table
 * @param[in] left_indices Device span of left table indices
 * @param[in] right_indices Device span of right table indices
 * @param[in] device_expression_data Device data required to evaluate the expression
 * @param[in] config Grid configuration for kernel launch
 * @param[in] shmem_per_block Amount of shared memory to allocate per block
 * @param[in] join_kind The join kind. Must be INNER_JOIN, LEFT_JOIN, or FULL_JOIN.
 * @param[out] count_out Atomic counter for the per-kind count described above
 * @param[out] left_passing_marks Byte buffer of size `left_table.num_rows()` used by LEFT_JOIN
 *             to mark left rows whose entries contribute to `count_out`. Must be zero-initialized
 *             before the kernel launch and may be `nullptr` for INNER_JOIN and FULL_JOIN.
 * @param[in] stream CUDA stream on which to launch the kernel
 */
template <bool has_nulls, bool has_complex_type>
void launch_filter_output_size_kernel(
  cudf::table_device_view const& left_table,
  cudf::table_device_view const& right_table,
  cudf::device_span<cudf::size_type const> left_indices,
  cudf::device_span<cudf::size_type const> right_indices,
  cudf::ast::detail::expression_device_view device_expression_data,
  cudf::detail::grid_1d const& config,
  std::size_t shmem_per_block,
  cudf::join_kind join_kind,
  std::size_t* count_out,
  bool* left_passing_marks,
  rmm::cuda_stream_view stream);

}  // namespace cudf::detail
