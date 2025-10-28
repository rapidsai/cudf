/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/ast/detail/expression_evaluator.cuh>
#include <cudf/ast/detail/expression_parser.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/iterator.cuh>
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

namespace {
constexpr int MAX_BLOCK_SIZE = 256;

/**
 * @brief Kernel to evaluate predicate on gather map pairs and mark valid indices
 *
 * @tparam max_block_size The size of the thread block, used to set launch bounds
 * @tparam has_nulls Indicates whether the expression may evaluate to null
 * @tparam has_complex_type Indicates whether the expression may contain complex types
 */
template <cudf::size_type max_block_size, bool has_nulls, bool has_complex_type>
__launch_bounds__(max_block_size) __global__
  void filter_gather_map_kernel(cudf::table_device_view left_table,
                                cudf::table_device_view right_table,
                                cudf::device_span<cudf::size_type const> left_indices,
                                cudf::device_span<cudf::size_type const> right_indices,
                                cudf::ast::detail::expression_device_view device_expression_data,
                                bool* output_flags)
{
  // Shared memory for intermediate storage
  extern __shared__ char raw_intermediate_storage[];
  cudf::ast::detail::IntermediateDataType<has_nulls>* intermediate_storage =
    reinterpret_cast<cudf::ast::detail::IntermediateDataType<has_nulls>*>(raw_intermediate_storage);

  auto thread_intermediate_storage =
    &intermediate_storage[threadIdx.x * device_expression_data.num_intermediates];

  auto const tid    = cudf::detail::grid_1d::global_thread_id();
  auto const stride = cudf::detail::grid_1d::grid_stride();

  // Create evaluator for this thread
  auto evaluator = cudf::ast::detail::expression_evaluator<has_nulls, has_complex_type>(
    left_table, right_table, device_expression_data);

  for (cudf::size_type i = tid; i < left_indices.size(); i += stride) {
    auto const left_row_index  = left_indices[i];
    auto const right_row_index = right_indices[i];

    // Create output destination for the boolean result
    cudf::ast::detail::value_expression_result<bool, has_nulls> result;

    // Evaluate predicate for this pair of rows
    evaluator.evaluate(result, left_row_index, right_row_index, 0, thread_intermediate_storage);

    // Mark this index pair as valid if predicate is true
    output_flags[i] = result.is_valid() && result.value();
  }
}

/**
 * @brief Template dispatch function to launch the appropriate kernel
 */
template <bool has_nulls, bool has_complex_type>
void launch_filter_gather_map_kernel(
  cudf::table_device_view const& left_table,
  cudf::table_device_view const& right_table,
  cudf::device_span<cudf::size_type const> left_indices,
  cudf::device_span<cudf::size_type const> right_indices,
  cudf::ast::detail::expression_device_view device_expression_data,
  bool* output_flags,
  cudf::detail::grid_1d const& config,
  std::size_t shmem_per_block,
  rmm::cuda_stream_view stream)
{
  filter_gather_map_kernel<MAX_BLOCK_SIZE, has_nulls, has_complex_type>
    <<<config.num_blocks, config.num_threads_per_block, shmem_per_block, stream.value()>>>(
      left_table, right_table, left_indices, right_indices, device_expression_data, output_flags);
}

}  // anonymous namespace

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
  auto const parser =
    ast::detail::expression_parser{predicate,
                                   left,
                                   right,
                                   has_nulls,
                                   stream,
                                   cudf::memory_resource::get_current_device_resource()};
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
