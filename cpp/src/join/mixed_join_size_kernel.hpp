/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "mixed_join_common_utils.cuh"

#include <cudf/ast/detail/expression_evaluator.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuco/pair.cuh>
#include <cuda/std/utility>

namespace CUDF_EXPORT cudf {
namespace detail {

/**
 * @brief Computes the output size of joining the left table to the right table.
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
 * @param[in] is_outer_join Whether this is an outer join
 * @param[in] swap_tables If true, the kernel was launched with one thread per right row and
 * the kernel needs to internally loop over left rows. Otherwise, loop over right rows.
 * @param[in] equality_probe The equality comparator used when probing the hash table.
 * @param[in] hash_table_storage Device span of the hash table storage
 * @param[in] input_pairs Precomputed input pairs for probing
 * @param[in] hash_indices Precomputed hash indices for efficient probing
 * @param[in] device_expression_data Container of device data required to evaluate the desired
 * expression.
 * @param[out] matches_per_row The number of matches in one pair of
 * equality/conditional tables for each row in the other pair of tables. If
 * swap_tables is true, matches_per_row corresponds to the right_table,
 * otherwise it corresponds to the left_table. Note that corresponding swap of
 * left/right tables to determine which is the build table and which is the
 * probe table has already happened on the host.
 * @param[in] config Grid configuration for the kernel launch
 * @param[in] shmem_size_per_block Shared memory size per block
 * @param[in] stream CUDA stream to use
 */

template <bool has_nulls>
void launch_mixed_join_count(
  table_device_view left_table,
  table_device_view right_table,
  bool is_outer_join,
  bool swap_tables,
  row_equality equality_probe,
  cudf::device_span<cuco::pair<hash_value_type, cudf::size_type>> hash_table_storage,
  cuco::pair<hash_value_type, cudf::size_type> const* input_pairs,
  cuda::std::pair<uint32_t, uint32_t> const* hash_indices,
  ast::detail::expression_device_view device_expression_data,
  cudf::device_span<cudf::size_type> matches_per_row,
  detail::grid_1d config,
  int64_t shmem_size_per_block,
  rmm::cuda_stream_view stream);

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
