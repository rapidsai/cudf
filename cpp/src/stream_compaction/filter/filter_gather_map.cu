/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "filter_gather_map_kernel.hpp"

#include <cudf/ast/detail/expression_parser.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/detail/join/join.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/join/join.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/std/functional>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include <memory>
#include <stdexcept>
#include <utility>

namespace cudf {
namespace detail {

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
filter_gather_map(cudf::table_view const& left,
                  cudf::table_view const& right,
                  cudf::device_span<size_type const> left_indices,
                  cudf::device_span<size_type const> right_indices,
                  ast::expression const& predicate,
                  join_kind join_kind,
                  rmm::cuda_stream_view stream,
                  rmm::device_async_resource_ref mr)
{
  // Validate inputs
  CUDF_EXPECTS(left_indices.size() == right_indices.size(),
               "Left and right index arrays must have the same size",
               std::invalid_argument);

  if (left_indices.empty()) {
    return std::make_pair(std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr),
                          std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr));
  }

  // Check if predicate may evaluate to null
  auto const has_nulls = predicate.may_evaluate_null(left, right, stream);

  // Create expression parser
  auto const parser = ast::detail::expression_parser{
    predicate, left, right, has_nulls, stream, cudf::get_current_device_resource_ref()};

  CUDF_EXPECTS(parser.output_type().id() == type_id::BOOL8,
               "The predicate expression must produce a Boolean output");

  // Check if expression contains complex types
  auto const has_complex_type = parser.has_complex_type();

  // Create device views of tables
  auto left_table  = table_device_view::create(left, stream);
  auto right_table = table_device_view::create(right, stream);

  // Allocate flags array to mark valid indices
  auto flags = rmm::device_uvector<bool>(left_indices.size(), stream);

  // Configure kernel parameters with dynamic shared memory calculation
  int device_id;
  CUDF_CUDA_TRY(cudaGetDevice(&device_id));

  int shmem_limit_per_block;
  CUDF_CUDA_TRY(
    cudaDeviceGetAttribute(&shmem_limit_per_block, cudaDevAttrMaxSharedMemoryPerBlock, device_id));

  auto const block_size =
    parser.shmem_per_thread != 0
      ? std::min(MAX_BLOCK_SIZE, shmem_limit_per_block / parser.shmem_per_thread)
      : MAX_BLOCK_SIZE;

  detail::grid_1d const config(left_indices.size(), block_size);
  auto const shmem_per_block = parser.shmem_per_thread * config.num_threads_per_block;

  // Launch kernel with template dispatch based on nulls and complex types
  if (has_nulls) {
    if (has_complex_type) {
      launch_filter_gather_map_kernel<true, true>(*left_table,
                                                  *right_table,
                                                  left_indices,
                                                  right_indices,
                                                  parser.device_expression_data,
                                                  config,
                                                  shmem_per_block,
                                                  flags.data(),
                                                  stream);
    } else {
      launch_filter_gather_map_kernel<true, false>(*left_table,
                                                   *right_table,
                                                   left_indices,
                                                   right_indices,
                                                   parser.device_expression_data,
                                                   config,
                                                   shmem_per_block,
                                                   flags.data(),
                                                   stream);
    }
  } else {
    if (has_complex_type) {
      launch_filter_gather_map_kernel<false, true>(*left_table,
                                                   *right_table,
                                                   left_indices,
                                                   right_indices,
                                                   parser.device_expression_data,
                                                   config,
                                                   shmem_per_block,
                                                   flags.data(),
                                                   stream);
    } else {
      launch_filter_gather_map_kernel<false, false>(*left_table,
                                                    *right_table,
                                                    left_indices,
                                                    right_indices,
                                                    parser.device_expression_data,
                                                    config,
                                                    shmem_per_block,
                                                    flags.data(),
                                                    stream);
    }
  }

  // Check for kernel launch errors
  CUDF_CHECK_CUDA(stream.value());

  // Handle different join semantics
  if (join_kind == join_kind::INNER_JOIN) {
    // INNER_JOIN: only keep pairs that satisfy the predicate AND have valid indices
    constexpr size_type JoinNoMatchValue = JoinNoMatch;

    // Create a combined predicate that checks both the user predicate and index validity
    auto valid_predicate = [flags_ptr = flags.data(),
                            left_ptr  = left_indices.data(),
                            right_ptr = right_indices.data(),
                            JoinNoMatchValue] __device__(size_type i) -> bool {
      // Check if indices are valid (not null sentinels)
      bool indices_valid = (left_ptr[i] != JoinNoMatchValue) && (right_ptr[i] != JoinNoMatchValue);
      // Only include if both indices are valid AND predicate is true
      return indices_valid && flags_ptr[i];
    };

    // Count valid pairs
    auto const num_valid =
      thrust::count_if(rmm::exec_policy_nosync(stream),
                       thrust::make_counting_iterator<size_type>(0),
                       thrust::make_counting_iterator<size_type>(left_indices.size()),
                       valid_predicate);

    if (num_valid == 0) {
      return std::make_pair(std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr),
                            std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr));
    }

    auto filtered_left_indices =
      std::make_unique<rmm::device_uvector<size_type>>(num_valid, stream, mr);
    auto filtered_right_indices =
      std::make_unique<rmm::device_uvector<size_type>>(num_valid, stream, mr);

    auto input_iter =
      thrust::make_zip_iterator(thrust::make_tuple(left_indices.begin(), right_indices.begin()));
    auto output_iter = thrust::make_zip_iterator(
      thrust::make_tuple(filtered_left_indices->begin(), filtered_right_indices->begin()));

    // Use copy_if with the predicate - the predicate receives the stencil value (index)
    auto counting_iter = thrust::make_counting_iterator<size_type>(0);
    thrust::copy_if(rmm::exec_policy_nosync(stream),
                    input_iter,
                    input_iter + left_indices.size(),
                    counting_iter,
                    output_iter,
                    [valid_predicate] __device__(size_type idx) { return valid_predicate(idx); });

    return std::make_pair(std::move(filtered_left_indices), std::move(filtered_right_indices));
  } else if (join_kind == join_kind::LEFT_JOIN) {
    // LEFT_JOIN: preserve all left rows, nullify right indices for failed predicates
    constexpr size_type JoinNoMatchValue = JoinNoMatch;

    auto filtered_left_indices =
      std::make_unique<rmm::device_uvector<size_type>>(left_indices.size(), stream, mr);
    auto filtered_right_indices =
      std::make_unique<rmm::device_uvector<size_type>>(left_indices.size(), stream, mr);

    // Transform the indices based on predicate results
    auto transform_op =
      [flags_ptr = flags.data(),
       left_ptr  = left_indices.data(),
       right_ptr = right_indices.data(),
       JoinNoMatchValue] __device__(size_type i) -> thrust::tuple<size_type, size_type> {
      auto left_idx  = left_ptr[i];
      auto right_idx = right_ptr[i];

      // If predicate is true, keep the pair as-is
      if (flags_ptr[i]) { return thrust::make_tuple(left_idx, right_idx); }

      // If predicate is false, preserve left index but set right to JoinNoMatch
      return thrust::make_tuple(left_idx, JoinNoMatchValue);
    };

    auto counting_iter = thrust::make_counting_iterator<size_type>(0);
    auto output_iter   = thrust::make_zip_iterator(
      thrust::make_tuple(filtered_left_indices->begin(), filtered_right_indices->begin()));

    thrust::transform(rmm::exec_policy_nosync(stream),
                      counting_iter,
                      counting_iter + left_indices.size(),
                      output_iter,
                      transform_op);

    return std::make_pair(std::move(filtered_left_indices), std::move(filtered_right_indices));
  } else if (join_kind == join_kind::FULL_JOIN) {
    // FULL_JOIN: For matched pairs that fail predicate, we need to generate two pairs:
    // (left_idx, JoinNoMatch) and (JoinNoMatch, right_idx) to preserve both sides
    constexpr size_type JoinNoMatchValue = JoinNoMatch;

    // Count how many additional pairs we need for failed matched pairs
    auto failed_matched_count = thrust::count_if(
      rmm::exec_policy_nosync(stream),
      thrust::make_counting_iterator<size_type>(0),
      thrust::make_counting_iterator<size_type>(left_indices.size()),
      [flags_ptr = flags.data(),
       left_ptr  = left_indices.data(),
       right_ptr = right_indices.data(),
       JoinNoMatchValue] __device__(size_type i) {
        return !flags_ptr[i] && left_ptr[i] != JoinNoMatchValue && right_ptr[i] != JoinNoMatchValue;
      });

    auto total_pairs = left_indices.size() + failed_matched_count;
    auto filtered_left_indices =
      std::make_unique<rmm::device_uvector<size_type>>(total_pairs, stream, mr);
    auto filtered_right_indices =
      std::make_unique<rmm::device_uvector<size_type>>(total_pairs, stream, mr);

    // First pass: handle all original pairs
    auto transform_op =
      [flags_ptr = flags.data(),
       left_ptr  = left_indices.data(),
       right_ptr = right_indices.data(),
       JoinNoMatchValue] __device__(size_type i) -> thrust::tuple<size_type, size_type> {
      auto left_idx  = left_ptr[i];
      auto right_idx = right_ptr[i];

      // If predicate is true, keep the pair as-is
      if (flags_ptr[i]) { return thrust::make_tuple(left_idx, right_idx); }

      // If predicate is false:
      if (left_idx == JoinNoMatchValue) {
        // Unmatched right row - preserve as-is
        return thrust::make_tuple(JoinNoMatchValue, right_idx);
      } else if (right_idx == JoinNoMatchValue) {
        // Unmatched left row - preserve as-is
        return thrust::make_tuple(left_idx, JoinNoMatchValue);
      } else {
        // Matched pair that failed predicate - preserve left side
        return thrust::make_tuple(left_idx, JoinNoMatchValue);
      }
    };

    auto counting_iter = thrust::make_counting_iterator<size_type>(0);
    auto output_iter   = thrust::make_zip_iterator(
      thrust::make_tuple(filtered_left_indices->begin(), filtered_right_indices->begin()));

    thrust::transform(rmm::exec_policy_nosync(stream),
                      counting_iter,
                      counting_iter + left_indices.size(),
                      output_iter,
                      transform_op);

    // Second pass: add the additional (JoinNoMatch, right_idx) pairs for failed matches
    if (failed_matched_count > 0) {
      // Create temporary vectors for the additional pairs
      rmm::device_uvector<size_type> temp_left(left_indices.size(), stream, mr);
      rmm::device_uvector<size_type> temp_right(left_indices.size(), stream, mr);

      // Fill with (JoinNoMatch, right_idx) for all pairs
      thrust::transform(
        rmm::exec_policy_nosync(stream),
        thrust::make_counting_iterator<size_type>(0),
        thrust::make_counting_iterator<size_type>(left_indices.size()),
        thrust::make_zip_iterator(thrust::make_tuple(temp_left.begin(), temp_right.begin())),
        [left_ptr  = left_indices.data(),
         right_ptr = right_indices.data(),
         JoinNoMatchValue] __device__(size_type i) -> thrust::tuple<size_type, size_type> {
          return thrust::make_tuple(JoinNoMatchValue, right_ptr[i]);
        });

      // Copy only the failed matched pairs to the output
      auto additional_iter = thrust::make_zip_iterator(
        thrust::make_tuple(filtered_left_indices->begin() + left_indices.size(),
                           filtered_right_indices->begin() + left_indices.size()));

      auto temp_iter =
        thrust::make_zip_iterator(thrust::make_tuple(temp_left.begin(), temp_right.begin()));

      thrust::copy_if(rmm::exec_policy_nosync(stream),
                      temp_iter,
                      temp_iter + left_indices.size(),
                      thrust::make_counting_iterator<size_type>(0),
                      additional_iter,
                      [flags_ptr = flags.data(),
                       left_ptr  = left_indices.data(),
                       right_ptr = right_indices.data(),
                       JoinNoMatchValue] __device__(size_type i) {
                        return !flags_ptr[i] && left_ptr[i] != JoinNoMatchValue &&
                               right_ptr[i] != JoinNoMatchValue;
                      });
    }

    return std::make_pair(std::move(filtered_left_indices), std::move(filtered_right_indices));
  } else {
    CUDF_FAIL("Unsupported join kind for filter_gather_map");
  }
}

}  // namespace detail

// Public API implementation
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
filter_gather_map(cudf::table_view const& left,
                  cudf::table_view const& right,
                  cudf::device_span<size_type const> left_indices,
                  cudf::device_span<size_type const> right_indices,
                  ast::expression const& predicate,
                  cudf::detail::join_kind join_kind,
                  rmm::cuda_stream_view stream,
                  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::filter_gather_map(
    left, right, left_indices, right_indices, predicate, join_kind, stream, mr);
}

}  // namespace cudf
