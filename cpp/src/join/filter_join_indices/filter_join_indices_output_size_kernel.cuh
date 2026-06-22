/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "join/filter_join_indices/filter_join_indices_output_size_kernel.hpp"
#include "join/join_common_utils.hpp"

#include <cudf/ast/detail/expression_evaluator.cuh>
#include <cudf/ast/detail/expression_parser.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/join/join.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cub/block/block_reduce.cuh>
#include <cuda/atomic>
#include <cuda/std/cstddef>

#include <cstddef>

namespace cudf::detail {

/**
 * @brief Counts the per-join-kind output size of `filter_join_indices` without materializing
 *        a per-pair boolean buffer.
 *
 * Each thread accumulates a private partial count, the block aggregates with CUB, and each
 * block adds its block-sum to `*count_out` exactly once via `cuda::atomic_ref`. For LEFT_JOIN,
 * `left_passing_marks[left_row_index]` is additionally set to `true` for every left row that
 * contributes to the count, which lets the host derive the number of synthetic JoinNoMatch
 * entries.
 */
template <bool has_nulls, bool has_complex_type>
CUDF_KERNEL __launch_bounds__(DEFAULT_JOIN_BLOCK_SIZE) void filter_join_indices_output_size_kernel(
  cudf::table_device_view left_table,
  cudf::table_device_view right_table,
  cudf::device_span<cudf::size_type const> left_indices,
  cudf::device_span<cudf::size_type const> right_indices,
  cudf::ast::detail::expression_device_view device_expression_data,
  cudf::join_kind join_kind,
  std::size_t* count_out,
  bool* left_passing_marks)
{
  extern __shared__ char raw_intermediate_storage[];
  auto* intermediate_storage =
    reinterpret_cast<cudf::ast::detail::IntermediateDataType<has_nulls>*>(raw_intermediate_storage);
  auto thread_intermediate_storage =
    &intermediate_storage[threadIdx.x * device_expression_data.num_intermediates];

  using BlockReduce = cub::BlockReduce<cuda::std::size_t, DEFAULT_JOIN_BLOCK_SIZE>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  auto const tid    = cudf::detail::grid_1d::global_thread_id();
  auto const stride = cudf::detail::grid_1d::grid_stride();

  auto evaluator = cudf::ast::detail::expression_evaluator<has_nulls, has_complex_type>{
    left_table, right_table, device_expression_data};

  cuda::std::size_t thread_local_count = 0;

  for (auto i = tid; i < static_cast<cudf::thread_index_type>(left_indices.size()); i += stride) {
    auto const left_row_index  = left_indices[i];
    auto const right_row_index = right_indices[i];

    bool const has_non_match =
      (left_row_index == cudf::JoinNoMatch || right_row_index == cudf::JoinNoMatch);

    bool predicate_pass = false;
    bool both_valid     = false;
    if (has_non_match) {
      // Outer-join unmatched pair: treat as passing so it is preserved in the output count.
      predicate_pass = true;
    } else if (left_row_index >= 0 && left_row_index < left_table.num_rows() &&
               right_row_index >= 0 && right_row_index < right_table.num_rows()) {
      auto result = cudf::ast::detail::value_expression_result<bool, has_nulls>{};
      evaluator.evaluate(result, left_row_index, right_row_index, 0, thread_intermediate_storage);
      predicate_pass = result.is_valid() && result.value();
      both_valid     = true;
    }

    switch (join_kind) {
      case cudf::join_kind::INNER_JOIN:
        if (predicate_pass) { ++thread_local_count; }
        break;
      case cudf::join_kind::LEFT_JOIN:
        if (predicate_pass) {
          ++thread_local_count;
          // Mark the left row as "passing" so the host can derive how many left rows need a
          // synthetic JoinNoMatch entry. For matched-passing pairs and for pre-existing
          // (left, JoinNoMatch) entries from upstream hash_join.left_join the left index is a
          // valid row index in [0, left_table.num_rows()).
          if (left_row_index >= 0 && left_row_index < left_table.num_rows()) {
            left_passing_marks[left_row_index] = true;
          }
        }
        break;
      case cudf::join_kind::FULL_JOIN:
        // Count failed matches: predicate false AND both indices valid.
        if (both_valid && !predicate_pass) { ++thread_local_count; }
        break;
      default: break;
    }
  }

  cuda::std::size_t const block_sum = BlockReduce(temp_storage).Sum(thread_local_count);

  if (threadIdx.x == 0) {
    cuda::atomic_ref<cuda::std::size_t, cuda::thread_scope_device> count_ref{*count_out};
    count_ref.fetch_add(block_sum, cuda::memory_order_relaxed);
  }
}

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
  rmm::cuda_stream_view stream)
{
  filter_join_indices_output_size_kernel<has_nulls, has_complex_type>
    <<<config.num_blocks, config.num_threads_per_block, shmem_per_block, stream.value()>>>(
      left_table,
      right_table,
      left_indices,
      right_indices,
      device_expression_data,
      join_kind,
      count_out,
      left_passing_marks);
  CUDF_CUDA_TRY(cudaGetLastError());
}

}  // namespace cudf::detail
