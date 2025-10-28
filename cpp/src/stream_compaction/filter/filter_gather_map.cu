/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "filter_gather_map_kernel.hpp"

#include <cudf/ast/detail/expression_parser.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
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
                  rmm::cuda_stream_view stream,
                  rmm::device_async_resource_ref mr)
{
  // Validate inputs
  CUDF_EXPECTS(left_indices.size() == right_indices.size(),
               "Left and right index arrays must have the same size");

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
                                                  flags.data(),
                                                  config,
                                                  shmem_per_block,
                                                  stream);
    } else {
      launch_filter_gather_map_kernel<true, false>(*left_table,
                                                   *right_table,
                                                   left_indices,
                                                   right_indices,
                                                   parser.device_expression_data,
                                                   flags.data(),
                                                   config,
                                                   shmem_per_block,
                                                   stream);
    }
  } else {
    if (has_complex_type) {
      launch_filter_gather_map_kernel<false, true>(*left_table,
                                                   *right_table,
                                                   left_indices,
                                                   right_indices,
                                                   parser.device_expression_data,
                                                   flags.data(),
                                                   config,
                                                   shmem_per_block,
                                                   stream);
    } else {
      launch_filter_gather_map_kernel<false, false>(*left_table,
                                                    *right_table,
                                                    left_indices,
                                                    right_indices,
                                                    parser.device_expression_data,
                                                    flags.data(),
                                                    config,
                                                    shmem_per_block,
                                                    stream);
    }
  }

  // Check for kernel launch errors
  CUDF_CHECK_CUDA(stream.value());

  // Count number of valid pairs
  auto const num_valid =
    thrust::count(rmm::exec_policy_nosync(stream), flags.begin(), flags.end(), true);

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

  thrust::copy_if(rmm::exec_policy_nosync(stream),
                  input_iter,
                  input_iter + left_indices.size(),
                  flags.begin(),
                  output_iter,
                  cuda::std::identity{});

  return std::make_pair(std::move(filtered_left_indices), std::move(filtered_right_indices));
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
                  rmm::cuda_stream_view stream,
                  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::filter_gather_map(left, right, left_indices, right_indices, predicate, stream, mr);
}

}  // namespace cudf
