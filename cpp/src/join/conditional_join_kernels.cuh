/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <join/join_common_utils.cuh>
#include <join/join_common_utils.hpp>

#include <cudf/ast/detail/expression_evaluator.cuh>
#include <cudf/ast/detail/expression_parser.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/table/table_device_view.cuh>

#include <cub/cub.cuh>

namespace cudf {
namespace detail {

/**
 * @brief Computes the output size of joining the left table to the right table.
 *
 * This method uses a nested loop to iterate over the left and right tables and count the number of
 * matches according to a boolean expression.
 *
 * @tparam block_size The number of threads per block for this kernel
 * @tparam has_nulls Whether or not the inputs may contain nulls.
 *
 * @param[in] left_table The left table
 * @param[in] right_table The right table
 * @param[in] join_type The type of join to be performed
 * @param[in] device_expression_data Container of device data required to evaluate the desired
 * expression.
 * @param[in] swap_tables If true, the kernel was launched with one thread per right row and
 * the kernel needs to internally loop over left rows. Otherwise, loop over right rows.
 * @param[out] output_size The resulting output size
 */
template <int block_size, bool has_nulls>
__global__ void compute_conditional_join_output_size(
  table_device_view left_table,
  table_device_view right_table,
  join_kind join_type,
  ast::detail::expression_device_view device_expression_data,
  bool const swap_tables,
  std::size_t* output_size)
{
  // The (required) extern storage of the shared memory array leads to
  // conflicting declarations between different templates. The easiest
  // workaround is to declare an arbitrary (here char) array type then cast it
  // after the fact to the appropriate type.
  extern __shared__ char raw_intermediate_storage[];
  cudf::ast::detail::IntermediateDataType<has_nulls>* intermediate_storage =
    reinterpret_cast<cudf::ast::detail::IntermediateDataType<has_nulls>*>(raw_intermediate_storage);
  auto thread_intermediate_storage =
    &intermediate_storage[threadIdx.x * device_expression_data.num_intermediates];

  std::size_t thread_counter{0};
  cudf::size_type const start_idx      = threadIdx.x + blockIdx.x * block_size;
  cudf::size_type const stride         = block_size * gridDim.x;
  cudf::size_type const left_num_rows  = left_table.num_rows();
  cudf::size_type const right_num_rows = right_table.num_rows();
  auto const outer_num_rows            = (swap_tables ? right_num_rows : left_num_rows);
  auto const inner_num_rows            = (swap_tables ? left_num_rows : right_num_rows);

  auto evaluator = cudf::ast::detail::expression_evaluator<has_nulls>(
    left_table, right_table, device_expression_data);

  for (cudf::size_type outer_row_index = start_idx; outer_row_index < outer_num_rows;
       outer_row_index += stride) {
    bool found_match = false;
    for (cudf::size_type inner_row_index = 0; inner_row_index < inner_num_rows; inner_row_index++) {
      auto output_dest           = cudf::ast::detail::value_expression_result<bool, has_nulls>();
      auto const left_row_index  = swap_tables ? inner_row_index : outer_row_index;
      auto const right_row_index = swap_tables ? outer_row_index : inner_row_index;
      evaluator.evaluate(
        output_dest, left_row_index, right_row_index, 0, thread_intermediate_storage);
      if (output_dest.is_valid() && output_dest.value()) {
        if ((join_type != join_kind::LEFT_ANTI_JOIN) &&
            !(join_type == join_kind::LEFT_SEMI_JOIN && found_match)) {
          ++thread_counter;
        }
        found_match = true;
      }
    }
    if ((join_type == join_kind::LEFT_JOIN || join_type == join_kind::LEFT_ANTI_JOIN ||
         join_type == join_kind::FULL_JOIN) &&
        (!found_match)) {
      ++thread_counter;
    }
  }

  using BlockReduce = cub::BlockReduce<cudf::size_type, block_size>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  std::size_t block_counter = BlockReduce(temp_storage).Sum(thread_counter);

  // Add block counter to global counter
  if (threadIdx.x == 0) atomicAdd(output_size, block_counter);
}

/**
 * @brief Performs a join conditioned on a predicate to find all matching rows
 * between the left and right tables and generate the output for the desired
 * Join operation.
 *
 * @tparam block_size The number of threads per block for this kernel
 * @tparam output_cache_size The side of the shared memory buffer to cache join
 * output results
 * @tparam has_nulls Whether or not the inputs may contain nulls.
 *
 * @param[in] left_table The left table
 * @param[in] right_table The right table
 * @param[in] join_type The type of join to be performed
 * @param[out] join_output_l The left result of the join operation
 * @param[out] join_output_r The right result of the join operation
 * @param[in,out] current_idx A global counter used by threads to coordinate
 * writes to the global output
 * @param device_expression_data Container of device data required to evaluate the desired
 * expression.
 * @param[in] max_size The maximum size of the output
 * @param[in] swap_tables If true, the kernel was launched with one thread per right row and
 * the kernel needs to internally loop over left rows. Otherwise, loop over right rows.
 */
template <cudf::size_type block_size, cudf::size_type output_cache_size, bool has_nulls>
__global__ void conditional_join(table_device_view left_table,
                                 table_device_view right_table,
                                 join_kind join_type,
                                 cudf::size_type* join_output_l,
                                 cudf::size_type* join_output_r,
                                 cudf::size_type* current_idx,
                                 cudf::ast::detail::expression_device_view device_expression_data,
                                 cudf::size_type const max_size,
                                 bool const swap_tables)
{
  constexpr int num_warps = block_size / detail::warp_size;
  __shared__ cudf::size_type current_idx_shared[num_warps];
  __shared__ cudf::size_type join_shared_l[num_warps][output_cache_size];
  __shared__ cudf::size_type join_shared_r[num_warps][output_cache_size];

  // Normally the casting of a shared memory array is used to create multiple
  // arrays of different types from the shared memory buffer, but here it is
  // used to circumvent conflicts between arrays of different types between
  // different template instantiations due to the extern specifier.
  extern __shared__ char raw_intermediate_storage[];
  cudf::ast::detail::IntermediateDataType<has_nulls>* intermediate_storage =
    reinterpret_cast<cudf::ast::detail::IntermediateDataType<has_nulls>*>(raw_intermediate_storage);
  auto thread_intermediate_storage =
    &intermediate_storage[threadIdx.x * device_expression_data.num_intermediates];

  int const warp_id                    = threadIdx.x / detail::warp_size;
  int const lane_id                    = threadIdx.x % detail::warp_size;
  cudf::size_type const left_num_rows  = left_table.num_rows();
  cudf::size_type const right_num_rows = right_table.num_rows();
  auto const outer_num_rows            = (swap_tables ? right_num_rows : left_num_rows);
  auto const inner_num_rows            = (swap_tables ? left_num_rows : right_num_rows);

  if (0 == lane_id) { current_idx_shared[warp_id] = 0; }

  __syncwarp();

  cudf::size_type outer_row_index = threadIdx.x + blockIdx.x * block_size;

  unsigned int const activemask = __ballot_sync(0xffff'ffffu, outer_row_index < outer_num_rows);

  auto evaluator = cudf::ast::detail::expression_evaluator<has_nulls>(
    left_table, right_table, device_expression_data);

  if (outer_row_index < outer_num_rows) {
    bool found_match = false;
    for (size_type inner_row_index(0); inner_row_index < inner_num_rows; ++inner_row_index) {
      auto output_dest           = cudf::ast::detail::value_expression_result<bool, has_nulls>();
      auto const left_row_index  = swap_tables ? inner_row_index : outer_row_index;
      auto const right_row_index = swap_tables ? outer_row_index : inner_row_index;
      evaluator.evaluate(
        output_dest, left_row_index, right_row_index, 0, thread_intermediate_storage);

      if (output_dest.is_valid() && output_dest.value()) {
        // If the rows are equal, then we have found a true match
        // In the case of left anti joins we only add indices from left after
        // the loop if we have found _no_ matches from the right.
        // In the case of left semi joins we only add the first match (note
        // that the current logic relies on the fact that we process all right
        // table rows for a single left table row on a single thread so that no
        // synchronization of found_match is required).
        if ((join_type != join_kind::LEFT_ANTI_JOIN) &&
            !(join_type == join_kind::LEFT_SEMI_JOIN && found_match)) {
          add_pair_to_cache(left_row_index,
                            right_row_index,
                            current_idx_shared,
                            warp_id,
                            join_shared_l[warp_id],
                            join_shared_r[warp_id]);
        }
        found_match = true;
      }

      __syncwarp(activemask);

      // flush output cache if next iteration does not fit
      auto const do_flush   = current_idx_shared[warp_id] + detail::warp_size >= output_cache_size;
      auto const flush_mask = __ballot_sync(activemask, do_flush);
      if (do_flush) {
        flush_output_cache<num_warps, output_cache_size>(flush_mask,
                                                         max_size,
                                                         warp_id,
                                                         lane_id,
                                                         current_idx,
                                                         current_idx_shared,
                                                         join_shared_l,
                                                         join_shared_r,
                                                         join_output_l,
                                                         join_output_r);
        __syncwarp(flush_mask);
        if (0 == lane_id) { current_idx_shared[warp_id] = 0; }
      }
      __syncwarp(activemask);
    }

    // Left, left anti, and full joins all require saving left columns that
    // aren't present in the right.
    if ((join_type == join_kind::LEFT_JOIN || join_type == join_kind::LEFT_ANTI_JOIN ||
         join_type == join_kind::FULL_JOIN) &&
        (!found_match)) {
      // TODO: This code assumes that swap_tables is false for all join
      // kinds aside from inner joins. Once the code is generalized to handle
      // other joins we'll want to switch the variable in the line below back
      // to the left_row_index, but for now we can assume that they are
      // equivalent inside this conditional.
      add_pair_to_cache(outer_row_index,
                        static_cast<cudf::size_type>(JoinNoneValue),
                        current_idx_shared,
                        warp_id,
                        join_shared_l[warp_id],
                        join_shared_r[warp_id]);
    }

    __syncwarp(activemask);

    // final flush of output cache
    auto const do_flush   = current_idx_shared[warp_id] > 0;
    auto const flush_mask = __ballot_sync(activemask, do_flush);
    if (do_flush) {
      flush_output_cache<num_warps, output_cache_size>(flush_mask,
                                                       max_size,
                                                       warp_id,
                                                       lane_id,
                                                       current_idx,
                                                       current_idx_shared,
                                                       join_shared_l,
                                                       join_shared_r,
                                                       join_output_l,
                                                       join_output_r);
    }
  }
}

}  // namespace detail

}  // namespace cudf
