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

template <cudf::size_type block_size, bool has_nulls>
__launch_bounds__(block_size) __global__
  void mixed_join_semi(table_device_view left_table,
                       table_device_view right_table,
                       table_device_view probe,
                       table_device_view build,
                       row_equality const equality_probe,
                       join_kind const join_type,
                       cudf::detail::semi_map_type::device_view hash_table_view,
                       size_type* join_output_l,
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

template __global__ void mixed_join_semi<DEFAULT_JOIN_BLOCK_SIZE, true>(
  table_device_view left_table,
  table_device_view right_table,
  table_device_view probe,
  table_device_view build,
  row_equality const equality_probe,
  join_kind const join_type,
  cudf::detail::semi_map_type::device_view hash_table_view,
  size_type* join_output_l,
  cudf::ast::detail::expression_device_view device_expression_data,
  cudf::size_type const* join_result_offsets,
  bool const swap_tables);

template __global__ void mixed_join_semi<DEFAULT_JOIN_BLOCK_SIZE, false>(
  table_device_view left_table,
  table_device_view right_table,
  table_device_view probe,
  table_device_view build,
  row_equality const equality_probe,
  join_kind const join_type,
  cudf::detail::semi_map_type::device_view hash_table_view,
  size_type* join_output_l,
  cudf::ast::detail::expression_device_view device_expression_data,
  cudf::size_type const* join_result_offsets,
  bool const swap_tables);

}  // namespace detail

}  // namespace cudf
