/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "join_common_utils.cuh"
#include "join_common_utils.hpp"
#include "mixed_join_common_utils.cuh"

#include <cudf/ast/detail/expression_parser.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuco/pair.cuh>

namespace CUDF_EXPORT cudf {
namespace detail {

/**
 * @brief Performs a join using the combination of a hash lookup to identify
 * equal rows between one pair of tables and the evaluation of an expression
 * containing an arbitrary expression.
 *
 * This method probes the hash table with each row in the probe table using a
 * custom equality comparator that also checks that the conditional expression
 * evaluates to true between the left/right tables when a match is found
 * between probe and build rows.
 *
 * @tparam has_nulls Whether or not the inputs may contain nulls.
 *
 * @param[in] left_table The left table
 * @param[in] right_table The right table
 * @param[in] join_type The type of join to be performed
 * @param[in] equality_probe The equality comparator used when probing the hash table.
 * @param[in] hash_table_storage Device span of the hash table storage
 * @param[in] input_pairs Precomputed input pairs for probing
 * @param[in] hash_indices Precomputed hash indices for efficient probing
 * @param[out] join_output_l The left result of the join operation
 * @param[out] join_output_r The right result of the join operation
 * @param[in] device_expression_data Container of device data required to evaluate the desired
 * expression.
 * @param[in] join_result_offsets Prefix sum of matches_per_row to get output offsets
 * @param[in] swap_tables If true, the kernel was launched with one thread per right row and
 * the kernel needs to internally loop over left rows. Otherwise, loop over right rows.
 */
template <bool has_nulls>
void launch_mixed_join(
  table_device_view left_table,
  table_device_view right_table,
  bool is_outer_join,
  bool swap_tables,
  row_equality equality_probe,
  cudf::device_span<cuco::pair<hash_value_type, cudf::size_type>> hash_table_storage,
  cuco::pair<hash_value_type, cudf::size_type> const* input_pairs,
  cuda::std::pair<uint32_t, uint32_t> const* hash_indices,
  cudf::ast::detail::expression_device_view device_expression_data,
  size_type* join_output_l,
  size_type* join_output_r,
  cudf::size_type const* join_result_offsets,
  detail::grid_1d config,
  int64_t shmem_size_per_block,
  rmm::cuda_stream_view stream);

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
