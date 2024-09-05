/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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
#include "mixed_join_kernel.hpp"

#include <cudf/ast/detail/expression_evaluator.cuh>
#include <cudf/ast/detail/expression_parser.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/span.hpp>

#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <thrust/iterator/discard_iterator.h>

namespace cudf {
namespace detail {

namespace cg = cooperative_groups;

#pragma GCC diagnostic ignored "-Wattributes"

template <cudf::size_type block_size, bool has_nulls>
CUDF_KERNEL void __launch_bounds__(block_size)
  mixed_join(table_device_view left_table,
             table_device_view right_table,
             table_device_view probe,
             table_device_view build,
             row_hash const hash_probe,
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

template <bool has_nulls>
void launch_mixed_join(table_device_view left_table,
                       table_device_view right_table,
                       table_device_view probe,
                       table_device_view build,
                       row_hash const hash_probe,
                       row_equality const equality_probe,
                       join_kind const join_type,
                       cudf::detail::mixed_multimap_type::device_view hash_table_view,
                       size_type* join_output_l,
                       size_type* join_output_r,
                       cudf::ast::detail::expression_device_view device_expression_data,
                       cudf::size_type const* join_result_offsets,
                       bool const swap_tables,
                       detail::grid_1d const config,
                       int64_t shmem_size_per_block,
                       rmm::cuda_stream_view stream)
{
  mixed_join<DEFAULT_JOIN_BLOCK_SIZE, has_nulls>
    <<<config.num_blocks, config.num_threads_per_block, shmem_size_per_block, stream.value()>>>(
      left_table,
      right_table,
      probe,
      build,
      hash_probe,
      equality_probe,
      join_type,
      hash_table_view,
      join_output_l,
      join_output_r,
      device_expression_data,
      join_result_offsets,
      swap_tables);
}

}  // namespace detail

}  // namespace cudf
