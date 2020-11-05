/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/table/table.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <iostream>

#include "cudf/types.hpp"
#include "hash_join.cuh"
#include "join_common_utils.hpp"
#include "join_kernels.cuh"

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
size_type estimate_nested_loop_join_output_size(table_device_view left,
                                                table_device_view right,
                                                join_kind JoinKind,
                                                null_equality compare_nulls,
                                                cudaStream_t stream)
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
  rmm::device_scalar<size_type> size_estimate(0, stream);

  CHECK_CUDA(stream);

  constexpr int block_size{DEFAULT_JOIN_BLOCK_SIZE};
  int numBlocks{-1};

  CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &numBlocks, compute_nested_loop_join_output_size<block_size>, block_size, 0));

  int dev_id{-1};
  CUDA_TRY(cudaGetDevice(&dev_id));

  int num_sms{-1};
  CUDA_TRY(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id));

  size_estimate.set_value(0, stream);

  row_equality equality{left, right, compare_nulls == null_equality::EQUAL};
  // Determine number of output rows without actually building the output to simply
  // find what the size of the output will be.
  compute_nested_loop_join_output_size<block_size><<<numBlocks * num_sms, block_size, 0, stream>>>(
    left, right, JoinKind, equality, size_estimate.data());
  CHECK_CUDA(stream);

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
std::pair<rmm::device_vector<size_type>, rmm::device_vector<size_type>>
get_base_nested_loop_join_indices(table_view const& left,
                                  table_view const& right,
                                  bool flip_join_indices,
                                  join_kind JoinKind,
                                  null_equality compare_nulls,
                                  cudaStream_t stream)
{
  // The `right` table is always used for the inner loop. We want to use the smaller table
  // for the inner loop. Thus, if `left` is smaller than `right`, swap `left/right`.
  if ((JoinKind == join_kind::INNER_JOIN) && (right.num_rows() > left.num_rows())) {
    return get_base_nested_loop_join_indices(right, left, true, JoinKind, compare_nulls, stream);
  }
  // Trivial left join case - exit early
  if ((JoinKind == join_kind::LEFT_JOIN) && (right.num_rows() == 0)) {
    return get_trivial_left_join_indices(left, stream);
  }

  auto left_table  = table_device_view::create(left, stream);
  auto right_table = table_device_view::create(right, stream);

  size_type estimated_size = estimate_nested_loop_join_output_size(
    *left_table, *right_table, JoinKind, compare_nulls, stream);

  // If the estimated output size is zero, return immediately
  if (estimated_size == 0) {
    return std::make_pair(rmm::device_vector<size_type>{}, rmm::device_vector<size_type>{});
  }

  // Because we are approximating the number of joined elements, our approximation
  // might be incorrect and we might have underestimated the number of joined elements.
  // As such we will need to de-allocate memory and re-allocate memory to ensure
  // that the final output is correct.
  rmm::device_scalar<size_type> write_index(0, stream);
  size_type join_size{0};

  rmm::device_vector<size_type> left_indices;
  rmm::device_vector<size_type> right_indices;
  auto current_estimated_size = estimated_size;
  do {
    left_indices.resize(estimated_size);
    right_indices.resize(estimated_size);

    constexpr int block_size{DEFAULT_JOIN_BLOCK_SIZE};
    detail::grid_1d config(left_table->num_rows(), block_size);
    write_index.set_value(0);

    row_equality equality{*left_table, *right_table, compare_nulls == null_equality::EQUAL};
    const auto& join_output_l =
      flip_join_indices ? right_indices.data().get() : left_indices.data().get();
    const auto& join_output_r =
      flip_join_indices ? left_indices.data().get() : right_indices.data().get();
    nested_loop_join<block_size, DEFAULT_JOIN_CACHE_SIZE>
      <<<config.num_blocks, config.num_threads_per_block, 0, stream>>>(*left_table,
                                                                       *right_table,
                                                                       JoinKind,
                                                                       equality,
                                                                       join_output_l,
                                                                       join_output_r,
                                                                       write_index.data(),
                                                                       estimated_size);

    CHECK_CUDA(stream);

    join_size              = write_index.value();
    current_estimated_size = estimated_size;
    estimated_size *= 2;
  } while ((current_estimated_size < join_size));

  left_indices.resize(join_size);
  right_indices.resize(join_size);
  return std::make_pair(std::move(left_indices), std::move(right_indices));
}

}  // namespace detail

}  // namespace cudf
