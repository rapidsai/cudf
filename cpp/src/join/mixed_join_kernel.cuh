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

#include <thrust/iterator/transform_output_iterator.h>

namespace cudf::detail {

struct output_fn {
  __device__ constexpr cudf::size_type operator()(
    cuco::pair<hash_value_type, cudf::size_type> const& slot) const
  {
    return slot.second;
  }
};

template <cudf::size_type block_size, bool has_nulls>
CUDF_KERNEL void __launch_bounds__(block_size)
  mixed_join(table_device_view left_table,
             table_device_view right_table,
             table_device_view probe,
             table_device_view build,
             row_hash const hash_probe,
             row_equality const equality_probe,
             join_kind const join_type,
             cudf::detail::mixed_join_hash_table_ref_t<cuco::retrieve_tag> const& hash_table_ref,
             thrust::transform_output_iterator<output_fn, size_type*> join_output_l,
             thrust::transform_output_iterator<output_fn, size_type*> join_output_r,
             size_t* size,
             cudf::ast::detail::expression_device_view device_expression_data,
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

  auto evaluator = cudf::ast::detail::expression_evaluator<has_nulls>(
    left_table, right_table, device_expression_data);
  auto equality = pair_expression_equality<has_nulls>{
    evaluator, thread_intermediate_storage, swap_tables, equality_probe};
  auto retrieve_ref = hash_table_ref.rebind_key_eq(equality);

  auto const pair_iter =
    thrust::make_transform_iterator(thrust::counting_iterator(0), pair_fn{hash_probe});

  namespace cg = cooperative_groups;

  auto const block = cg::this_thread_block();
  auto counter_ref = cuda::atomic<size_t, cuda::thread_scope_device>(*size);

  auto const block_begin_offset = block.group_index().x * block_size;
  auto const block_end_offset =
    cuda::std::min(outer_num_rows, static_cast<cudf::size_type>(block_begin_offset + block_size));

  if (block_begin_offset < block_end_offset) {
    if (join_type == join_kind::LEFT_JOIN || join_type == join_kind::FULL_JOIN) {
      retrieve_ref.template retrieve_outer<
        block_size,
        thrust::transform_iterator<pair_fn<cudf::detail::row_hash>,
                                   thrust::counting_iterator<cudf::size_type>>,
        thrust::transform_output_iterator<output_fn, size_type*>,
        thrust::transform_output_iterator<output_fn, size_type*>,
        cuda::atomic<size_t, cuda::thread_scope_device>>(block,
                                                         pair_iter + block_begin_offset,
                                                         pair_iter + block_end_offset,
                                                         join_output_l,
                                                         join_output_r,
                                                         &counter_ref);
    } else {
      retrieve_ref
        .template retrieve<block_size,
                           thrust::transform_iterator<pair_fn<cudf::detail::row_hash>,
                                                      thrust::counting_iterator<cudf::size_type>>,
                           thrust::transform_output_iterator<output_fn, size_type*>,
                           thrust::transform_output_iterator<output_fn, size_type*>,
                           cuda::atomic<size_t, cuda::thread_scope_device>>(
          block,
          pair_iter + block_begin_offset,
          pair_iter + block_end_offset,
          join_output_l,
          join_output_r,
          &counter_ref);
    }
  }
}

template <bool has_nulls>
void launch_mixed_join(
  table_device_view left_table,
  table_device_view right_table,
  table_device_view probe,
  table_device_view build,
  row_hash const hash_probe,
  row_equality const equality_probe,
  join_kind const join_type,
  cudf::detail::mixed_join_hash_table_ref_t<cuco::retrieve_tag> const& hash_table_ref,
  size_type* join_output_l,
  size_type* join_output_r,
  cudf::ast::detail::expression_device_view device_expression_data,
  bool const swap_tables,
  detail::grid_1d const config,
  int64_t shmem_size_per_block,
  rmm::cuda_stream_view stream)
{
  cudf::detail::device_scalar<size_t> size(0, stream);

  mixed_join<DEFAULT_JOIN_BLOCK_SIZE, has_nulls>
    <<<config.num_blocks, config.num_threads_per_block, shmem_size_per_block, stream.value()>>>(
      left_table,
      right_table,
      probe,
      build,
      hash_probe,
      equality_probe,
      join_type,
      hash_table_ref,
      thrust::make_transform_output_iterator(join_output_l, output_fn{}),
      thrust::make_transform_output_iterator(join_output_r, output_fn{}),
      size.data(),
      device_expression_data,
      swap_tables);
}

}  // namespace cudf::detail
