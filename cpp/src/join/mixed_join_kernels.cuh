/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <join/hash_join.cuh>
#include <join/join_common_utils.cuh>
#include <join/join_common_utils.hpp>
#include <join/mixed_join_common_utils.cuh>

#include <cudf/ast/detail/expression_evaluator.cuh>
#include <cudf/ast/detail/expression_parser.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/span.hpp>

#include <cooperative_groups.h>

#include <cub/cub.cuh>
#include <thrust/iterator/discard_iterator.h>

namespace cudf {
namespace detail {

namespace cg = cooperative_groups;

template <cudf::size_type block_size, bool has_nulls>
__launch_bounds__(block_size) __global__
  void mixed_join(table_device_view left_table,
                  table_device_view right_table,
                  table_device_view probe,
                  table_device_view build,
                  row_equality const equality_probe,
                  join_kind const join_type,
                  cudf::detail::mixed_multimap_type::device_view hash_table_view,
                  size_type* join_output_l,
                  size_type* join_output_r,
                  cudf::ast::detail::expression_device_view device_expression_data,
                  cudf::size_type const* join_result_offsets,
                  bool const swap_tables)
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

  cudf::size_type outer_row_index = threadIdx.x + blockIdx.x * block_size;

  auto evaluator = cudf::ast::detail::expression_evaluator<has_nulls>(
    left_table, right_table, device_expression_data);

  row_hash hash_probe{nullate::DYNAMIC{has_nulls}, probe};
  auto const empty_key_sentinel = hash_table_view.get_empty_key_sentinel();
  make_pair_function pair_func{hash_probe, empty_key_sentinel};

  if (outer_row_index < outer_num_rows) {
    // Figure out the number of elements for this key.
    cg::thread_block_tile<1> this_thread = cg::this_thread();
    // Figure out the number of elements for this key.
    auto query_pair = pair_func(outer_row_index);
    auto equality   = pair_expression_equality<has_nulls>{
      evaluator, thread_intermediate_storage, swap_tables, equality_probe};

    auto probe_key_begin       = thrust::make_discard_iterator();
    auto probe_value_begin     = swap_tables ? join_output_r + join_result_offsets[outer_row_index]
                                             : join_output_l + join_result_offsets[outer_row_index];
    auto contained_key_begin   = thrust::make_discard_iterator();
    auto contained_value_begin = swap_tables ? join_output_l + join_result_offsets[outer_row_index]
                                             : join_output_r + join_result_offsets[outer_row_index];

    if (join_type == join_kind::LEFT_JOIN || join_type == join_kind::FULL_JOIN) {
      hash_table_view.pair_retrieve_outer(this_thread,
                                          query_pair,
                                          probe_key_begin,
                                          probe_value_begin,
                                          contained_key_begin,
                                          contained_value_begin,
                                          equality);
    } else {
      hash_table_view.pair_retrieve(this_thread,
                                    query_pair,
                                    probe_key_begin,
                                    probe_value_begin,
                                    contained_key_begin,
                                    contained_value_begin,
                                    equality);
    }
  }
}

/**
 * @brief Computes the output size of joining the left table to the right table.
 *
 * This method probes the hash table with each row in the probe table using a
 * custom equality comparator that also checks that the conditional expression
 * evaluates to true between the left/right tables when a match is found
 * between probe and build rows.
 *
 * @tparam block_size The number of threads per block for this kernel
 * @tparam has_nulls Whether or not the inputs may contain nulls.
 *
 * @param[in] left_table The left table
 * @param[in] right_table The right table
 * @param[in] probe The table with which to probe the hash table for matches.
 * @param[in] build The table with which the hash table was built.
 * @param[in] equality_probe The equality comparator used when probing the hash table.
 * @param[in] join_type The type of join to be performed
 * @param[in] hash_table_view The hash table built from `build`.
 * @param[in] device_expression_data Container of device data required to evaluate the desired
 * expression.
 * @param[in] swap_tables If true, the kernel was launched with one thread per right row and
 * the kernel needs to internally loop over left rows. Otherwise, loop over right rows.
 * @param[out] output_size The resulting output size
 * @param[out] matches_per_row The number of matches in one pair of
 * equality/conditional tables for each row in the other pair of tables. If
 * swap_tables is true, matches_per_row corresponds to the right_table,
 * otherwise it corresponds to the left_table. Note that corresponding swap of
 * left/right tables to determine which is the build table and which is the
 * probe table has already happened on the host.
 */
template <int block_size, bool has_nulls>
__global__ void compute_mixed_join_output_size(
  table_device_view left_table,
  table_device_view right_table,
  table_device_view probe,
  table_device_view build,
  row_equality const equality_probe,
  join_kind const join_type,
  cudf::detail::mixed_multimap_type::device_view hash_table_view,
  ast::detail::expression_device_view device_expression_data,
  bool const swap_tables,
  std::size_t* output_size,
  cudf::device_span<cudf::size_type> matches_per_row);

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
 * @tparam block_size The number of threads per block for this kernel
 * @tparam has_nulls Whether or not the inputs may contain nulls.
 *
 * @param[in] left_table The left table
 * @param[in] right_table The right table
 * @param[in] probe The table with which to probe the hash table for matches.
 * @param[in] build The table with which the hash table was built.
 * @param[in] equality_probe The equality comparator used when probing the hash table.
 * @param[in] join_type The type of join to be performed
 * @param[in] hash_table_view The hash table built from `build`.
 * @param[out] join_output_l The left result of the join operation
 * @param[out] join_output_r The right result of the join operation
 * @param[in] device_expression_data Container of device data required to evaluate the desired
 * expression.
 * @param[in] join_result_offsets The starting indices in join_output[l|r]
 * where the matches for each row begin. Equivalent to a prefix sum of
 * matches_per_row.
 * @param[in] swap_tables If true, the kernel was launched with one thread per right row and
 * the kernel needs to internally loop over left rows. Otherwise, loop over right rows.
 */
template <cudf::size_type block_size, bool has_nulls>
__global__ void mixed_join(table_device_view left_table,
                           table_device_view right_table,
                           table_device_view probe,
                           table_device_view build,
                           row_equality const equality_probe,
                           join_kind const join_type,
                           cudf::detail::mixed_multimap_type::device_view hash_table_view,
                           size_type* join_output_l,
                           size_type* join_output_r,
                           cudf::ast::detail::expression_device_view device_expression_data,
                           cudf::size_type const* join_result_offsets,
                           bool const swap_tables);

}  // namespace detail

}  // namespace cudf
