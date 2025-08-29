/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "join_common_utils.cuh"
#include "join_common_utils.hpp"
#include "mixed_join_common_utils.cuh"

#include <cudf/ast/detail/expression_evaluator.cuh>
#include <cudf/ast/detail/expression_parser.hpp>
#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/atomic>

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
  cuda::std::pair<cudf::size_type, cudf::size_type> const& hash_idx,
  join_kind join_type) noexcept
{
  cudf::size_type count = 0;
  auto const extent     = hash_table_storage.size();
  auto const* data      = hash_table_storage.data();
  auto probe_idx        = static_cast<std::size_t>(hash_idx.first);   // initial probe index
  auto const step       = static_cast<std::size_t>(hash_idx.second);  // step size

  while (true) {
    auto const bucket_slots =
      *reinterpret_cast<cuda::std::array<cuco::pair<hash_value_type, cudf::size_type>, 2> const*>(
        data + probe_idx);

    // Check for empty slots and key equality
    auto const first_slot_is_empty  = bucket_slots[0].second == cudf::detail::JoinNoneValue;
    auto const second_slot_is_empty = bucket_slots[1].second == cudf::detail::JoinNoneValue;
    auto const first_slot_equals =
      (not first_slot_is_empty and key_equal(probe_key, bucket_slots[0]));
    auto const second_slot_equals =
      (not second_slot_is_empty and key_equal(probe_key, bucket_slots[1]));

    count += (first_slot_equals + second_slot_equals);

    // Exit if we find an empty slot
    if (first_slot_is_empty or second_slot_is_empty) {
      // Handle outer join logic: non-matching rows are counted as 1 match
      if ((join_type == join_kind::LEFT_JOIN || join_type == join_kind::FULL_JOIN) && count == 0) {
        return 1;
      }
      return count;
    }

    // Move to next bucket using precomputed step
    probe_idx = (probe_idx + step) % extent;

    // Detect full cycle completion
    if (probe_idx == static_cast<std::size_t>(hash_idx.first)) {
      // Handle outer join logic: non-matching rows are counted as 1 match
      if ((join_type == join_kind::LEFT_JOIN || join_type == join_kind::FULL_JOIN) && count == 0) {
        return 1;
      }
      return count;
    }
  }
}

template <bool has_nulls>
CUDF_KERNEL void __launch_bounds__(DEFAULT_JOIN_BLOCK_SIZE) compute_mixed_join_output_size(
  table_device_view left_table,
  table_device_view right_table,
  join_kind join_type,
  row_equality equality_probe,
  cudf::device_span<cuco::pair<hash_value_type, cudf::size_type>> hash_table_storage,
  cuco::pair<hash_value_type, cudf::size_type> const* input_pairs,
  cuda::std::pair<cudf::size_type, cudf::size_type> const* hash_indices,
  ast::detail::expression_device_view device_expression_data,
  bool swap_tables,
  size_t* d_total_count)
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

  using BlockReduce = cub::BlockReduce<size_t, DEFAULT_JOIN_BLOCK_SIZE>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

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

  // Thread-local count
  size_t per_thread_count = 0;

  for (auto outer_row_index = start_idx; outer_row_index < outer_num_rows;
       outer_row_index += stride) {
    auto const& probe_key = input_pairs[outer_row_index];
    auto const& hash_idx  = hash_indices[outer_row_index];

    auto match_count =
      standalone_count(count_equality, hash_table_storage, probe_key, hash_idx, join_type);

    per_thread_count += match_count;
  }

  size_t per_block_count = BlockReduce(temp_storage).Sum(per_thread_count);

  if (threadIdx.x == 0) {
    cuda::atomic_ref<size_t, cuda::thread_scope_device> counter_ref(*d_total_count);
    counter_ref.fetch_add(per_block_count, cuda::memory_order_relaxed);
  }
}

template <bool has_nulls>
std::size_t launch_compute_mixed_join_output_size(
  table_device_view left_table,
  table_device_view right_table,
  join_kind join_type,
  row_equality equality_probe,
  cudf::device_span<cuco::pair<hash_value_type, cudf::size_type>> hash_table_storage,
  cuco::pair<hash_value_type, cudf::size_type> const* input_pairs,
  cuda::std::pair<cudf::size_type, cudf::size_type> const* hash_indices,
  ast::detail::expression_device_view device_expression_data,
  bool swap_tables,
  detail::grid_1d const& config,
  int64_t shmem_size_per_block,
  rmm::cuda_stream_view stream)
{
  cudf::detail::device_scalar<std::size_t> d_total_count{
    0, stream, cudf::get_current_device_resource_ref()};

  compute_mixed_join_output_size<has_nulls>
    <<<config.num_blocks, config.num_threads_per_block, shmem_size_per_block, stream.value()>>>(
      left_table,
      right_table,
      join_type,
      equality_probe,
      hash_table_storage,
      input_pairs,
      hash_indices,
      device_expression_data,
      swap_tables,
      d_total_count.data());

  return d_total_count.value(stream);
}

}  // namespace cudf::detail
