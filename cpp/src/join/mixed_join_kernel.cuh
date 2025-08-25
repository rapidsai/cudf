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

#include "cudf/types.hpp"
#include "join_common_utils.cuh"
#include "join_common_utils.hpp"
#include "mixed_join_common_utils.cuh"
#include "mixed_join_kernel.hpp"

#include <cudf/ast/detail/expression_evaluator.cuh>
#include <cudf/ast/detail/expression_parser.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/atomic>

namespace cudf::detail {

/**
 * @brief Optimized standalone retrieve implementation for hash table probing
 *
 * This implementation uses precomputed hash indices and storage references
 * for efficient mixed join operations with minimal overhead.
 *
 * @tparam is_outer Boolean flag indicating whether outer join semantics should be used
 */
template <bool is_outer, bool has_nulls>
__device__ __forceinline__ void retrieve(
  cooperative_groups::thread_block const& block,
  cudf::device_span<cuco::pair<hash_value_type, cudf::size_type>> hash_table_storage,
  pair_expression_equality<has_nulls> const& key_equal,
  cuco::pair<hash_value_type, cudf::size_type> const* input_probe_begin,
  cuco::pair<hash_value_type, cudf::size_type> const* input_probe_end,
  cuda::std::pair<cudf::size_type, cudf::size_type> const* input_hash_begin,
  cudf::size_type* output_probe,
  cudf::size_type* output_match,
  cuda::atomic<size_t, cuda::thread_scope_device>& atomic_counter) noexcept
{
  using size_type                   = cudf::size_type;
  static constexpr auto bucket_size = 2;
  static constexpr auto block_size  = DEFAULT_JOIN_BLOCK_SIZE;
  namespace cg                      = cooperative_groups;

  auto const n = cuda::std::distance(input_probe_begin, input_probe_end);

  auto constexpr flushing_cg_size = cudf::detail::warp_size;

  auto constexpr num_flushing_cgs     = block_size / flushing_cg_size;
  auto constexpr max_matches_per_step = flushing_cg_size * bucket_size;
  auto constexpr buffer_size          = max_matches_per_step + flushing_cg_size;

  auto const flushing_cg = cg::tiled_partition<flushing_cg_size>(block);

  auto const flushing_cg_id = flushing_cg.meta_group_rank();
  auto const stride         = block_size;
  auto idx                  = threadIdx.x;

  __shared__ cudf::size_type probe_output_buffer[num_flushing_cgs][buffer_size];
  __shared__ cudf::size_type match_output_buffer[num_flushing_cgs][buffer_size];
  __shared__ cudf::size_type flushing_cg_counter[num_flushing_cgs];

  if (flushing_cg.thread_rank() == 0) { flushing_cg_counter[flushing_cg_id] = 0; }
  flushing_cg.sync();

  auto flush_output_buffer = [&](auto const& tile) {
    size_type offset = 0;
    auto const count = flushing_cg_counter[flushing_cg_id];
    auto const rank  = tile.thread_rank();
    if (rank == 0) { offset = atomic_counter.fetch_add(count, cuda::memory_order_relaxed); }
    offset = tile.shfl(offset, 0);

    for (auto i = rank; i < count; i += tile.size()) {
      *(output_probe + offset + i) = probe_output_buffer[flushing_cg_id][i];
      *(output_match + offset + i) = match_output_buffer[flushing_cg_id][i];
    }
  };

  while (flushing_cg.any(idx < n)) {
    bool active_flag = idx < n;
    auto const active_flushing_cg =
      cg::binary_partition<flushing_cg_size>(flushing_cg, active_flag);

    if (active_flag) {
      auto const& probe_key = *(input_probe_begin + idx);
      auto const& hash_idx  = *(input_hash_begin + idx);

      auto const extent     = hash_table_storage.size();
      auto current_slot_idx = static_cast<std::size_t>(hash_idx.first);
      auto const step       = static_cast<std::size_t>(hash_idx.second);

      bool running                      = true;
      [[maybe_unused]] bool found_match = false;

      while (active_flushing_cg.any(running)) {
        if (running) {
          auto const bucket_slots = *reinterpret_cast<
            cuda::std::array<cuco::pair<hash_value_type, cudf::size_type>, 2> const*>(
            hash_table_storage.data() + current_slot_idx);

          auto const first_slot_is_empty  = bucket_slots[0].second == cudf::detail::JoinNoneValue;
          auto const second_slot_is_empty = bucket_slots[1].second == cudf::detail::JoinNoneValue;
          auto const first_equals =
            (not first_slot_is_empty and key_equal(probe_key, bucket_slots[0]));
          auto const second_equals =
            (not second_slot_is_empty and key_equal(probe_key, bucket_slots[1]));

          if (first_equals or second_equals) {
            if constexpr (is_outer) { found_match = true; }

            cudf::size_type num_matches = (first_equals ? 1 : 0) + (second_equals ? 1 : 0);
            cuda::atomic_ref<cudf::size_type, cuda::thread_scope_block> counter_ref{
              flushing_cg_counter[flushing_cg_id]};
            cudf::size_type output_idx =
              counter_ref.fetch_add(num_matches, cuda::memory_order_relaxed);

            auto const probe_row_index = probe_key.second;

            if (first_equals) {
              probe_output_buffer[flushing_cg_id][output_idx] = probe_row_index;
              match_output_buffer[flushing_cg_id][output_idx] = bucket_slots[0].second;
              if (second_equals) {
                probe_output_buffer[flushing_cg_id][output_idx + 1] = probe_row_index;
                match_output_buffer[flushing_cg_id][output_idx + 1] = bucket_slots[1].second;
              }
            } else if (second_equals) {
              probe_output_buffer[flushing_cg_id][output_idx] = probe_row_index;
              match_output_buffer[flushing_cg_id][output_idx] = bucket_slots[1].second;
            }
          }

          if (first_slot_is_empty or second_slot_is_empty) {
            running = false;

            if constexpr (is_outer) {
              if (not found_match) {
                cuda::atomic_ref<cudf::size_type, cuda::thread_scope_block> counter_ref{
                  flushing_cg_counter[flushing_cg_id]};
                auto const output_idx      = counter_ref.fetch_add(1, cuda::memory_order_relaxed);
                auto const probe_row_index = probe_key.second;
                probe_output_buffer[flushing_cg_id][output_idx] = probe_row_index;
                match_output_buffer[flushing_cg_id][output_idx] = cudf::detail::JoinNoneValue;
              }
            }
          }
        }

        active_flushing_cg.sync();
        if (flushing_cg_counter[flushing_cg_id] > (buffer_size - max_matches_per_step)) {
          flush_output_buffer(active_flushing_cg);
          active_flushing_cg.sync();

          if (active_flushing_cg.thread_rank() == 0) { flushing_cg_counter[flushing_cg_id] = 0; }
          active_flushing_cg.sync();
        }

        current_slot_idx = (current_slot_idx + step) % extent;
        if (current_slot_idx == static_cast<std::size_t>(hash_idx.first)) { running = false; }
      }
    }

    idx += stride;
  }

  flushing_cg.sync();
  if (flushing_cg_counter[flushing_cg_id] > 0) { flush_output_buffer(flushing_cg); }
}

template <bool has_nulls>
CUDF_KERNEL void __launch_bounds__(DEFAULT_JOIN_BLOCK_SIZE)
  mixed_join(table_device_view left_table,
             table_device_view right_table,
             join_kind join_type,
             row_equality equality_probe,
             cudf::device_span<cuco::pair<hash_value_type, cudf::size_type>> hash_table_storage,
             cuco::pair<hash_value_type, cudf::size_type> const* input_pairs,
             cuda::std::pair<cudf::size_type, cudf::size_type> const* hash_indices,
             size_type* join_output_l,
             size_type* join_output_r,
             cudf::ast::detail::expression_device_view device_expression_data,
             bool swap_tables)
{
  extern __shared__ char raw_intermediate_storage[];
  auto intermediate_storage =
    reinterpret_cast<cudf::ast::detail::IntermediateDataType<has_nulls>*>(raw_intermediate_storage);
  auto thread_intermediate_storage =
    &intermediate_storage[threadIdx.x * device_expression_data.num_intermediates];

  cudf::size_type const left_num_rows  = left_table.num_rows();
  cudf::size_type const right_num_rows = right_table.num_rows();
  auto const outer_num_rows            = (swap_tables ? right_num_rows : left_num_rows);

  auto const evaluator = cudf::ast::detail::expression_evaluator<has_nulls>{
    left_table, right_table, device_expression_data};
  auto const equality = pair_expression_equality<has_nulls>{
    evaluator, thread_intermediate_storage, swap_tables, equality_probe};

  namespace cg = cooperative_groups;

  auto const block = cg::this_thread_block();
  cuda::atomic<size_t, cuda::thread_scope_device> counter_ref{0};

  auto const block_begin_offset = block.group_index().x * DEFAULT_JOIN_BLOCK_SIZE;
  auto const block_end_offset   = cuda::std::min(
    outer_num_rows, static_cast<cudf::size_type>(block_begin_offset + DEFAULT_JOIN_BLOCK_SIZE));

  if (block_begin_offset < block_end_offset) {
    if (join_type == join_kind::LEFT_JOIN || join_type == join_kind::FULL_JOIN) {
      retrieve<true>(block,
                     hash_table_storage,
                     equality,
                     input_pairs + block_begin_offset,
                     input_pairs + block_end_offset,
                     hash_indices + block_begin_offset,
                     swap_tables ? join_output_r : join_output_l,
                     swap_tables ? join_output_l : join_output_r,
                     counter_ref);
    } else {
      retrieve<false>(block,
                      hash_table_storage,
                      equality,
                      input_pairs + block_begin_offset,
                      input_pairs + block_end_offset,
                      hash_indices + block_begin_offset,
                      swap_tables ? join_output_r : join_output_l,
                      swap_tables ? join_output_l : join_output_r,
                      counter_ref);
    }
  }
}

template <bool has_nulls>
void launch_mixed_join(
  table_device_view left_table,
  table_device_view right_table,
  join_kind join_type,
  row_equality equality_probe,
  cudf::device_span<cuco::pair<hash_value_type, cudf::size_type>> hash_table_storage,
  cuco::pair<hash_value_type, cudf::size_type> const* input_pairs,
  cuda::std::pair<cudf::size_type, cudf::size_type> const* hash_indices,
  size_type* join_output_l,
  size_type* join_output_r,
  cudf::ast::detail::expression_device_view device_expression_data,
  bool swap_tables,
  detail::grid_1d const& config,
  int64_t shmem_size_per_block,
  rmm::cuda_stream_view stream)
{
  mixed_join<has_nulls>
    <<<config.num_blocks, config.num_threads_per_block, shmem_size_per_block, stream.value()>>>(
      left_table,
      right_table,
      join_type,
      equality_probe,
      hash_table_storage,
      input_pairs,
      hash_indices,
      join_output_l,
      join_output_r,
      device_expression_data,
      swap_tables);
}

}  // namespace cudf::detail
