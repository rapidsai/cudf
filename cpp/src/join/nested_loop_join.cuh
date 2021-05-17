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

#include <iostream>

namespace cudf {
namespace detail {
/**
 * @brief Gives an estimate of the size of the join output produced when
 * joining two tables together.
 *
 * @throw cudf::logic_error if JoinKind is not INNER_JOIN or LEFT_JOIN
 *
 * @param left The left hand table
 * @param right The right hand table
 * @param JoinKind The type of join to be performed
 * @param compare_nulls Controls whether null join-key values should match or not.
 * @param stream CUDA stream used for device memory operations and kernel launches
 *
 * @return An estimate of the size of the output of the join operation
 */
size_type estimate_nested_loop_join_output_size(
  table_device_view left,
  table_device_view right,
  join_kind JoinKind,
  null_equality compare_nulls,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  const size_type left_num_rows{left.num_rows()};
  const size_type right_num_rows{right.num_rows()};

  if (right_num_rows == 0) {
    // If the right table is empty, we know exactly how large the output
    // will be for the different types of joins and can return immediately
    switch (JoinKind) {
      // Inner join with an empty table will have no output
      case join_kind::INNER_JOIN: return 0;

      // Left join with an empty table will have an output of NULL rows
      // equal to the number of rows in the left table
      case join_kind::LEFT_JOIN: return left_num_rows;

      default: CUDF_FAIL("Unsupported join type");
    }
  }

  // Allocate storage for the counter used to get the size of the join output
  size_type h_size_estimate{0};
  rmm::device_scalar<size_type> size_estimate(0, stream, mr);

  CHECK_CUDA(stream.value());

  constexpr int block_size{DEFAULT_JOIN_BLOCK_SIZE};
  int numBlocks{-1};

  CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &numBlocks, compute_nested_loop_join_output_size<block_size>, block_size, 0));

  int dev_id{-1};
  CUDA_TRY(cudaGetDevice(&dev_id));

  int num_sms{-1};
  CUDA_TRY(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id));

  size_estimate.set_value_zero(stream);

  row_equality equality{left, right, compare_nulls == null_equality::EQUAL};
  // Determine number of output rows without actually building the output to simply
  // find what the size of the output will be.
  compute_nested_loop_join_output_size<block_size>
    <<<numBlocks * num_sms, block_size, 0, stream.value()>>>(
      left, right, JoinKind, equality, size_estimate.data());
  CHECK_CUDA(stream.value());

  h_size_estimate = size_estimate.value(stream);

  return h_size_estimate;
}

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
get_predicate_join_indices(table_view const& left,
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
  if ((JoinKind == join_kind::INNER_JOIN) && (right.num_rows() > left.num_rows())) {
    return get_predicate_join_indices(
      right, left, true, JoinKind, binary_pred, compare_nulls, stream, mr);
  }
  // Trivial left join case - exit early
  if ((JoinKind == join_kind::LEFT_JOIN) && (right.num_rows() == 0)) {
    return get_trivial_left_join_indices(left, stream);
  }

  auto left_table  = table_device_view::create(left, stream);
  auto right_table = table_device_view::create(right, stream);

  size_type estimated_size = estimate_nested_loop_join_output_size(
    *left_table, *right_table, JoinKind, compare_nulls, stream, mr);

  // If the estimated output size is zero, return immediately
  if (estimated_size == 0) {
    return std::make_pair(std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr),
                          std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr));
  }

  // TODO: The AST code's linearizer data path_only uses the table of the
  // expression for determining the data type of a column reference, so for now
  // we can reuse the same linearizer for convenience and assume that the left
  // and right tables have all the same data types. We will eventually have to
  // relax this assumption to provide reasonable error checking.
  auto const expr_linearizer   = ast::detail::linearizer(binary_pred, left);  // Linearize the AST
  auto const plan              = ast::detail::ast_plan{expr_linearizer, stream, mr};
  auto const num_intermediates = expr_linearizer.intermediate_count();
  auto const shmem_size_per_thread = static_cast<int>(sizeof(std::int64_t) * num_intermediates);

  // Because we are approximating the number of joined elements, our approximation
  // might be incorrect and we might have underestimated the number of joined elements.
  // As such we will need to de-allocate memory and re-allocate memory to ensure
  // that the final output is correct.
  rmm::device_scalar<size_type> write_index(0, stream);
  size_type join_size{0};

  auto left_indices  = std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr);
  auto right_indices = std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr);

  auto current_estimated_size = estimated_size;
  do {
    left_indices->resize(estimated_size, stream);
    right_indices->resize(estimated_size, stream);

    auto output_column = cudf::make_fixed_width_column(expr_linearizer.root_data_type(),
                                                       current_estimated_size,
                                                       mask_state::UNALLOCATED,
                                                       stream,
                                                       mr);
    auto mutable_output_device =
      cudf::mutable_column_device_view::create(output_column->mutable_view(), stream);

    constexpr int block_size{DEFAULT_JOIN_BLOCK_SIZE};
    detail::grid_1d config(left_table->num_rows(), block_size);
    write_index.set_value_zero(stream);

    const auto& join_output_l = flip_join_indices ? right_indices->data() : left_indices->data();
    const auto& join_output_r = flip_join_indices ? left_indices->data() : right_indices->data();
    nested_loop_predicate_join<block_size, DEFAULT_JOIN_CACHE_SIZE>
      <<<config.num_blocks, config.num_threads_per_block, shmem_size_per_thread, stream.value()>>>(
        *left_table,
        *right_table,
        JoinKind,
        join_output_l,
        join_output_r,
        *mutable_output_device,
        write_index.data(),
        plan.dev_plan,
        estimated_size,
        num_intermediates);

    CHECK_CUDA(stream.value());

    join_size              = write_index.value(stream);
    current_estimated_size = estimated_size;
    estimated_size *= 2;
  } while ((current_estimated_size < join_size));

  left_indices->resize(join_size, stream);
  right_indices->resize(join_size, stream);
  return std::make_pair(std::move(left_indices), std::move(right_indices));
}

}  // namespace detail

}  // namespace cudf
