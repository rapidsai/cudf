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

#include "hash_join.cuh"
#include "join_common_utils.hpp"
#include "join_kernels.cuh"

#include <cudf/ast/detail/transform.cuh>
#include <cudf/ast/nodes.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/table/table.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <thrust/optional.h>

#include <algorithm>

namespace cudf {
namespace detail {

/**
 * @brief Computes the join operation between two tables and returns the
 * output indices of left and right table as a combined table
 *
 * @param left  Table of left columns to join
 * @param right Table of right  columns to join
 * @param flip_join_indices Flag that indicates whether the left and right
 * tables have been flipped, meaning the output indices should also be flipped
 * @param JoinKind The type of join to be performed
 * @param compare_nulls Controls whether null join-key values should match or not.
 * @param stream CUDA stream used for device memory operations and kernel launches
 *
 * @return Join output indices vector pair
 */
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
get_conditional_join_indices(table_view const& left,
                             table_view const& right,
                             bool flip_join_indices,
                             join_kind JoinKind,
                             ast::expression binary_pred,
                             null_equality compare_nulls,
                             rmm::cuda_stream_view stream,
                             rmm::mr::device_memory_resource* mr)
{
  // The `right` table is always used for the inner loop. We want to use the smaller table
  // for the inner loop. Thus, if `left` is smaller than `right`, swap `left/right`.
  // TODO: I'm not confident that this will give the correct result. An
  // arbitrary conditional join on two tables can use, for instance, different
  // columns from each table, in which case flipping the left and right tables
  // will not respect the table references encoded in the predicate.
  if ((JoinKind == join_kind::INNER_JOIN) && (right.num_rows() > left.num_rows())) {
    return get_conditional_join_indices(
      right, left, true, JoinKind, binary_pred, compare_nulls, stream, mr);
  }

  // We can immediately filter out cases where the right table is empty. In
  // some cases, we return all the rows of the left table with a corresponding
  // null index for the right table; in others, we return an empty output.
  if (right.num_rows() == 0) {
    switch (JoinKind) {
      // Left, left anti, and full (which are effectively left because we are
      // guaranteed that left has more rows than right) all return a all the
      // row indices from left with a corresponding NULL from the right.
      case join_kind::LEFT_JOIN:
      case join_kind::LEFT_ANTI_JOIN:
      case join_kind::FULL_JOIN: return get_trivial_left_join_indices(left, stream);
      // Inner and left semi joins return empty output because no matches can exist.
      case join_kind::INNER_JOIN:
      case join_kind::LEFT_SEMI_JOIN:
        return std::make_pair(std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr),
                              std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr));
    }
  }

  // Prepare output column. Whether or not the output column is nullable is
  // determined by whether any of the columns in the input table are nullable.
  // If none of the input columns actually contain nulls, we can still use the
  // non-nullable version of the expression evaluation code path for
  // performance, so we capture that information as well.
  auto nullable =
    std::any_of(left.begin(), left.end(), [](column_view c) { return c.nullable(); }) ||
    std::any_of(left.begin(), left.end(), [](column_view c) { return c.nullable(); });
  auto has_nulls =
    nullable &&
    (std::any_of(
       left.begin(), left.end(), [](column_view c) { return c.nullable() && c.has_nulls(); }) ||
     std::any_of(
       right.begin(), right.end(), [](column_view c) { return c.nullable() && c.has_nulls(); }));

  auto const plan = ast::detail::ast_plan{binary_pred, left, right, has_nulls, stream, mr};
  CUDF_EXPECTS(plan.output_type().id() == type_id::BOOL8,
               "The expression must produce a boolean output.");

  auto left_table  = table_device_view::create(left, stream);
  auto right_table = table_device_view::create(right, stream);

  // Allocate storage for the counter used to get the size of the join output
  rmm::device_scalar<size_type> size(0, stream, mr);
  constexpr int block_size{DEFAULT_JOIN_BLOCK_SIZE};
  detail::grid_1d config(left_table->num_rows(), block_size);

  // Determine number of output rows without actually building the output to simply
  // find what the size of the output will be.
  join_kind KernelJoinKind = JoinKind == join_kind::FULL_JOIN ? join_kind::LEFT_JOIN : JoinKind;
  if (has_nulls) {
    compute_conditional_join_output_size<block_size, true>
      <<<config.num_blocks,
         config.num_threads_per_block,
         plan.dev_plan.shmem_per_thread,
         stream.value()>>>(*left_table, *right_table, KernelJoinKind, plan.dev_plan, size.data());
  } else {
    compute_conditional_join_output_size<block_size, false>
      <<<config.num_blocks,
         config.num_threads_per_block,
         plan.dev_plan.shmem_per_thread,
         stream.value()>>>(*left_table, *right_table, KernelJoinKind, plan.dev_plan, size.data());
  }
  CHECK_CUDA(stream.value());

  size_type join_size = size.value(stream);

  // If the output size will be zero, we can return immediately.
  if (join_size == 0) {
    return std::make_pair(std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr),
                          std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr));
  }

  rmm::device_scalar<size_type> write_index(0, stream);

  auto left_indices  = std::make_unique<rmm::device_uvector<size_type>>(join_size, stream, mr);
  auto right_indices = std::make_unique<rmm::device_uvector<size_type>>(join_size, stream, mr);

  const auto& join_output_l = flip_join_indices ? right_indices->data() : left_indices->data();
  const auto& join_output_r = flip_join_indices ? left_indices->data() : right_indices->data();
  if (has_nulls) {
    conditional_join<block_size, DEFAULT_JOIN_CACHE_SIZE, true>
      <<<config.num_blocks,
         config.num_threads_per_block,
         plan.dev_plan.shmem_per_thread,
         stream.value()>>>(*left_table,
                           *right_table,
                           KernelJoinKind,
                           join_output_l,
                           join_output_r,
                           write_index.data(),
                           plan.dev_plan,
                           join_size);
  } else {
    conditional_join<block_size, DEFAULT_JOIN_CACHE_SIZE, false>
      <<<config.num_blocks,
         config.num_threads_per_block,
         plan.dev_plan.shmem_per_thread,
         stream.value()>>>(*left_table,
                           *right_table,
                           KernelJoinKind,
                           join_output_l,
                           join_output_r,
                           write_index.data(),
                           plan.dev_plan,
                           join_size);
  }

  CHECK_CUDA(stream.value());

  auto join_indices = std::make_pair(std::move(left_indices), std::move(right_indices));

  // For full joins, get the indices in the right table that were not joined to
  // by any row in the left table.
  if (JoinKind == join_kind::FULL_JOIN) {
    auto complement_indices = detail::get_left_join_indices_complement(
      join_indices.second, left.num_rows(), right.num_rows(), stream, mr);
    join_indices = detail::concatenate_vector_pairs(join_indices, complement_indices, stream);
  }
  return join_indices;
}

}  // namespace detail

}  // namespace cudf
