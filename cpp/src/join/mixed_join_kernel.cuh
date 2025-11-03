/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "join_common_utils.hpp"
#include "mixed_join_common_utils.cuh"
#include "mixed_join_kernel.hpp"

#include <cudf/ast/detail/expression_evaluator.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/span.hpp>

namespace cudf {
namespace detail {

/**
 * @brief Optimized retrieve implementation using precomputed matches per row
 *
 * This implementation uses precomputed match counts to avoid expensive atomic
 * operations and directly fills output arrays based on known match positions.
 *
 * @tparam is_outer Boolean flag indicating whether outer join semantics should be used
 * @tparam has_nulls Whether the input tables may contain nulls
 */
template <bool is_outer, bool has_nulls>
__device__ __forceinline__ void retrieve_matches(
  cudf::device_span<cuco::pair<hash_value_type, cudf::size_type>> hash_table_storage,
  pair_expression_equality<has_nulls> const& key_equal,
  cuco::pair<hash_value_type, cudf::size_type> const& probe_key,
  cuda::std::pair<hash_value_type, hash_value_type> const& hash_idx,
  cudf::size_type* probe_output,
  cudf::size_type* match_output) noexcept
{
  auto const probe_row_index = probe_key.second;
  cudf::size_type output_idx = 0;
  bool found_match           = false;
  auto prober = hash_table_prober<has_nulls>{key_equal, hash_table_storage, probe_key, hash_idx};

  while (true) {
    auto const result       = prober.probe_current_bucket();
    auto const bucket_slots = prober.get_bucket_slots();

    if (result.first_slot_equals_) {
      probe_output[output_idx] = probe_row_index;
      match_output[output_idx] = bucket_slots.first.second;
      output_idx++;
      found_match = true;
    }

    if (result.second_slot_equals_) {
      probe_output[output_idx] = probe_row_index;
      match_output[output_idx] = bucket_slots.second.second;
      output_idx++;
      found_match = true;
    }

    // Exit if we find an empty slot
    if (result.has_empty_slot()) { break; }

    prober.advance();
  }

  // Handle outer join logic for non-matching rows
  if constexpr (is_outer) {
    if (not found_match) {
      probe_output[0] = probe_row_index;
      match_output[0] = cudf::JoinNoMatch;
    }
  }
}

template <bool has_nulls>
CUDF_KERNEL void __launch_bounds__(DEFAULT_JOIN_BLOCK_SIZE)
  mixed_join(table_device_view left_table,
             table_device_view right_table,
             bool is_outer_join,
             bool swap_tables,
             row_equality equality_probe,
             cudf::device_span<cuco::pair<hash_value_type, cudf::size_type>> hash_table_storage,
             cuco::pair<hash_value_type, cudf::size_type> const* input_pairs,
             cuda::std::pair<hash_value_type, hash_value_type> const* hash_indices,
             cudf::ast::detail::expression_device_view device_expression_data,
             size_type* join_output_l,
             size_type* join_output_r,
             cudf::size_type const* join_result_offsets)
{
  // Normally the casting of a shared memory array is used to create multiple
  // arrays of different types from the shared memory buffer, but here it is
  // used to circumvent conflicts between arrays of different types between
  // different template instantiations due to the extern specifier.
  extern __shared__ char raw_intermediate_storage[];
  cudf::ast::detail::IntermediateDataType<has_nulls>* intermediate_storage =
    reinterpret_cast<cudf::ast::detail::IntermediateDataType<has_nulls>*>(raw_intermediate_storage);
  auto thread_intermediate_storage =
    &intermediate_storage[threadIdx.x * device_expression_data.num_intermediates];

  cudf::size_type const left_num_rows  = left_table.num_rows();
  cudf::size_type const right_num_rows = right_table.num_rows();
  auto const outer_num_rows            = (swap_tables ? right_num_rows : left_num_rows);

  auto const start_idx = cudf::detail::grid_1d::global_thread_id();
  auto const stride    = cudf::detail::grid_1d::grid_stride();

  auto const evaluator = cudf::ast::detail::expression_evaluator<has_nulls>{
    left_table, right_table, device_expression_data};

  auto const equality = pair_expression_equality<has_nulls>{
    evaluator, thread_intermediate_storage, swap_tables, equality_probe};

  // Process each row and write matches to precomputed output positions
  for (auto outer_row_index = start_idx; outer_row_index < outer_num_rows;
       outer_row_index += stride) {
    auto const& probe_key    = input_pairs[outer_row_index];
    auto const& hash_idx     = hash_indices[outer_row_index];
    auto const output_offset = join_result_offsets[outer_row_index];

    if (is_outer_join) {
      retrieve_matches<true>(
        hash_table_storage,
        equality,
        probe_key,
        hash_idx,
        swap_tables ? join_output_r + output_offset : join_output_l + output_offset,
        swap_tables ? join_output_l + output_offset : join_output_r + output_offset);
    } else {
      retrieve_matches<false>(
        hash_table_storage,
        equality,
        probe_key,
        hash_idx,
        swap_tables ? join_output_r + output_offset : join_output_l + output_offset,
        swap_tables ? join_output_l + output_offset : join_output_r + output_offset);
    }
  }
}

template <bool has_nulls>
void launch_mixed_join(
  table_device_view left_table,
  table_device_view right_table,
  bool is_outer_join,
  bool swap_tables,
  row_equality equality_probe,
  cudf::device_span<cuco::pair<hash_value_type, cudf::size_type>> hash_table_storage,
  cuco::pair<hash_value_type, cudf::size_type> const* input_pairs,
  cuda::std::pair<hash_value_type, hash_value_type> const* hash_indices,
  cudf::ast::detail::expression_device_view device_expression_data,
  size_type* join_output_l,
  size_type* join_output_r,
  cudf::size_type const* join_result_offsets,
  detail::grid_1d config,
  int64_t shmem_size_per_block,
  rmm::cuda_stream_view stream)
{
  mixed_join<has_nulls>
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
      join_output_l,
      join_output_r,
      join_result_offsets);
}

}  // namespace detail

}  // namespace cudf
