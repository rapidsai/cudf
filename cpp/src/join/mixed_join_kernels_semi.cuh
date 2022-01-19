/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
#include <join/mixed_join_common_utils.cuh>

#include <cudf/ast/detail/expression_evaluator.cuh>
#include <cudf/ast/detail/expression_parser.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/span.hpp>

#include <cub/cub.cuh>

namespace cudf {
namespace detail {
namespace cg = cooperative_groups;

/**
 * @brief Equality comparator for cuco::static_map queries.
 *
 * This equality comparator is designed for use with cuco::static_map's APIs. A
 * probe hit indicates that the hashes of the keys are equal, at which point
 * this comparator checks whether the keys themselves are equal (using the
 * provided equality_probe) and then evaluates the conditional expression
 */
template <bool has_nulls>
struct single_expression_equality : expression_equality<has_nulls> {
  using expression_equality<has_nulls>::expression_equality;

  // The parameters are build/probe rather than left/right because the operator
  // is called by cuco's kernels with parameters in this order (note that this
  // is an implementation detail that we should eventually stop relying on by
  // defining operators with suitable heterogeneous typing). Rather than
  // converting to left/right semantics, we can operate directly on build/probe
  // until we get to the expression evaluator, which needs to convert back to
  // left/right semantics because the conditional expression need not be
  // commutative.
  // TODO: The input types should really be size_type.
  __device__ __forceinline__ bool operator()(hash_value_type const build_row_index,
                                             hash_value_type const probe_row_index) const noexcept
  {
    auto output_dest = cudf::ast::detail::value_expression_result<bool, has_nulls>();
    // Two levels of checks:
    // 1. The contents of the columns involved in the equality condition are equal.
    // 2. The predicate evaluated on the relevant columns (already encoded in the evaluator)
    // evaluates to true.
    if (this->equality_probe(probe_row_index, build_row_index)) {
      auto const lrow_idx = this->swap_tables ? build_row_index : probe_row_index;
      auto const rrow_idx = this->swap_tables ? probe_row_index : build_row_index;
      this->evaluator.evaluate(output_dest,
                               static_cast<size_type>(lrow_idx),
                               static_cast<size_type>(rrow_idx),
                               0,
                               this->thread_intermediate_storage);
      return (output_dest.is_valid() && output_dest.value());
    }
    return false;
  }
};

/**
 * @brief Computes the output size of joining the left table to the right table for semi/anti joins.
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
__global__ void compute_mixed_join_output_size_semi(
  table_device_view left_table,
  table_device_view right_table,
  table_device_view probe,
  table_device_view build,
  row_equality const equality_probe,
  join_kind const join_type,
  cudf::detail::semi_map_type::device_view hash_table_view,
  ast::detail::expression_device_view device_expression_data,
  bool const swap_tables,
  std::size_t* output_size,
  cudf::device_span<cudf::size_type> matches_per_row)
{
  // The (required) extern storage of the shared memory array leads to
  // conflicting declarations between different templates. The easiest
  // workaround is to declare an arbitrary (here char) array type then cast it
  // after the fact to the appropriate type.
  extern __shared__ char raw_intermediate_storage[];
  cudf::ast::detail::IntermediateDataType<has_nulls>* intermediate_storage =
    reinterpret_cast<cudf::ast::detail::IntermediateDataType<has_nulls>*>(raw_intermediate_storage);
  auto thread_intermediate_storage =
    intermediate_storage + (threadIdx.x * device_expression_data.num_intermediates);

  std::size_t thread_counter{0};
  cudf::size_type const start_idx      = threadIdx.x + blockIdx.x * block_size;
  cudf::size_type const stride         = block_size * gridDim.x;
  cudf::size_type const left_num_rows  = left_table.num_rows();
  cudf::size_type const right_num_rows = right_table.num_rows();
  auto const outer_num_rows            = (swap_tables ? right_num_rows : left_num_rows);

  auto evaluator = cudf::ast::detail::expression_evaluator<has_nulls>(
    left_table, right_table, device_expression_data);
  row_hash hash_probe{nullate::DYNAMIC{has_nulls}, probe};
  // TODO: Address asymmetry in operator.
  auto equality = single_expression_equality<has_nulls>{
    evaluator, thread_intermediate_storage, swap_tables, equality_probe};

  for (cudf::size_type outer_row_index = start_idx; outer_row_index < outer_num_rows;
       outer_row_index += stride) {
    matches_per_row[outer_row_index] =
      ((join_type == join_kind::LEFT_ANTI_JOIN) !=
       (hash_table_view.contains(outer_row_index, hash_probe, equality)));
    thread_counter += matches_per_row[outer_row_index];
  }

  using BlockReduce = cub::BlockReduce<cudf::size_type, block_size>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  std::size_t block_counter = BlockReduce(temp_storage).Sum(thread_counter);

  // Add block counter to global counter
  if (threadIdx.x == 0) atomicAdd(output_size, block_counter);
}

/**
 * @brief Performs a semi/anti join using the combination of a hash lookup to
 * identify equal rows between one pair of tables and the evaluation of an
 * expression containing an arbitrary expression.
 *
 * This method probes the hash table with each row in the probe table using a
 * custom equality comparator that also checks that the conditional expression
 * evaluates to true between the left/right tables when a match is found
 * between probe and build rows.
 *
 * @tparam block_size The number of threads per block for this kernel
 * @tparam output_cache_size The side of the shared memory buffer to cache join
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
 * @param[in] device_expression_data Container of device data required to evaluate the desired
 * expression.
 * @param[in] join_result_offsets The starting indices in join_output[l|r]
 * where the matches for each row begin. Equivalent to a prefix sum of
 * matches_per_row.
 * @param[in] swap_tables If true, the kernel was launched with one thread per right row and
 * the kernel needs to internally loop over left rows. Otherwise, loop over right rows.
 */
template <cudf::size_type block_size,
          cudf::size_type output_cache_size,
          bool has_nulls,
          typename OutputIt1>
__global__ void mixed_join_semi(table_device_view left_table,
                                table_device_view right_table,
                                table_device_view probe,
                                table_device_view build,
                                row_equality const equality_probe,
                                join_kind const join_type,
                                cudf::detail::semi_map_type::device_view hash_table_view,
                                OutputIt1 join_output_l,
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

  if (outer_row_index < outer_num_rows) {
    // Figure out the number of elements for this key.
    auto equality = single_expression_equality<has_nulls>{
      evaluator, thread_intermediate_storage, swap_tables, equality_probe};

    if ((join_type == join_kind::LEFT_ANTI_JOIN) !=
        (hash_table_view.contains(outer_row_index, hash_probe, equality))) {
      *(join_output_l + join_result_offsets[outer_row_index]) = outer_row_index;
    }
  }
}

}  // namespace detail

}  // namespace cudf
