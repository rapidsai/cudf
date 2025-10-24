/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "join_common_utils.hpp"
#include "mixed_join_common_utils.cuh"

#include <cudf/ast/detail/expression_evaluator.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/span.hpp>

#include <cuda/std/utility>

namespace cudf::detail {

/**
 * @brief Standalone count implementation using precomputed hash indices
 *
 * This implementation provides essential count functionality for mixed joins
 * using precomputed probe indices and step sizes.
 */
template <bool has_nulls>
__device__ __forceinline__ auto standalone_count(
  pair_expression_equality<has_nulls> const& key_equal,
  cudf::device_span<cuco::pair<hash_value_type, cudf::size_type>> hash_table_storage,
  cuco::pair<hash_value_type, cudf::size_type> const& probe_key,
  cuda::std::pair<hash_value_type, hash_value_type> const& hash_idx,
  bool is_outer_join) noexcept
{
  cudf::size_type count = 0;
  auto prober = hash_table_prober<has_nulls>{key_equal, hash_table_storage, probe_key, hash_idx};

  while (true) {
    auto const result = prober.probe_current_bucket();
    count += result.match_count();

    // Exit if we find an empty slot
    if (result.has_empty_slot()) {
      // Handle outer join logic: non-matching rows are counted as 1 match
      if (is_outer_join && count == 0) { return 1; }
      return count;
    }

    prober.advance();
  }
}

template <bool has_nulls>
CUDF_KERNEL void __launch_bounds__(DEFAULT_JOIN_BLOCK_SIZE) mixed_join_count(
  table_device_view left_table,
  table_device_view right_table,
  bool is_outer_join,
  bool swap_tables,
  row_equality equality_probe,
  cudf::device_span<cuco::pair<hash_value_type, cudf::size_type>> hash_table_storage,
  cuco::pair<hash_value_type, cudf::size_type> const* input_pairs,
  cuda::std::pair<hash_value_type, hash_value_type> const* hash_indices,
  ast::detail::expression_device_view device_expression_data,
  cudf::device_span<cudf::size_type> matches_per_row)
{
  // The (required) extern storage of the shared memory array leads to
  // conflicting declarations between different templates. The easiest
  // workaround is to declare an arbitrary (here char) array type then cast it
  // after the fact to the appropriate type.
  extern __shared__ char raw_intermediate_storage[];
  auto intermediate_storage =
    reinterpret_cast<cudf::ast::detail::IntermediateDataType<has_nulls>*>(raw_intermediate_storage);
  auto thread_intermediate_storage =
    intermediate_storage + (threadIdx.x * device_expression_data.num_intermediates);

  auto const start_idx                 = cudf::detail::grid_1d::global_thread_id();
  auto const stride                    = cudf::detail::grid_1d::grid_stride();
  cudf::size_type const left_num_rows  = left_table.num_rows();
  cudf::size_type const right_num_rows = right_table.num_rows();
  auto const outer_num_rows            = (swap_tables ? right_num_rows : left_num_rows);

  auto const evaluator = cudf::ast::detail::expression_evaluator<has_nulls>{
    left_table, right_table, device_expression_data};

  // Figure out the number of elements for this key.
  // TODO: Address asymmetry in operator.
  auto count_equality = pair_expression_equality<has_nulls>{
    evaluator, thread_intermediate_storage, swap_tables, equality_probe};

  for (auto outer_row_index = start_idx; outer_row_index < outer_num_rows;
       outer_row_index += stride) {
    auto const& probe_key = input_pairs[outer_row_index];
    auto const& hash_idx  = hash_indices[outer_row_index];

    auto match_count =
      standalone_count(count_equality, hash_table_storage, probe_key, hash_idx, is_outer_join);

    matches_per_row[outer_row_index] = match_count;
  }
}

template <bool has_nulls>
void launch_mixed_join_count(
  table_device_view left_table,
  table_device_view right_table,
  bool is_outer_join,
  bool swap_tables,
  row_equality equality_probe,
  cudf::device_span<cuco::pair<hash_value_type, cudf::size_type>> hash_table_storage,
  cuco::pair<hash_value_type, cudf::size_type> const* input_pairs,
  cuda::std::pair<hash_value_type, hash_value_type> const* hash_indices,
  ast::detail::expression_device_view device_expression_data,
  cudf::device_span<cudf::size_type> matches_per_row,
  detail::grid_1d config,
  int64_t shmem_size_per_block,
  rmm::cuda_stream_view stream)
{
  mixed_join_count<has_nulls>
    <<<config.num_blocks, config.num_threads_per_block, shmem_size_per_block, stream.value()>>>(
      left_table,
      right_table,
      is_outer_join,
      swap_tables,
      equality_probe,
      hash_table_storage,
      input_pairs,
      hash_indices,
      device_expression_data,
      matches_per_row);
}

}  // namespace cudf::detail
