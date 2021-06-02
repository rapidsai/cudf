/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <cudf/ast/detail/transform.cuh>
#include <cudf/ast/nodes.hpp>
#include <cudf/ast/operators.hpp>
#include <cudf/ast/transform.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <algorithm>
#include <functional>
#include <iterator>
#include <type_traits>

namespace cudf {
namespace ast {
namespace detail {

/**
 * @brief Kernel for evaluating an expression on a table to produce a new column.
 *
 * This evaluates an expression over a table to produce a new column. Also called an n-ary
 * transform.
 *
 * @tparam block_size
 * @param table The table device view used for evaluation.
 * @param literals Array of literal values used for evaluation.
 * @param output_column The output column where results are stored.
 * @param data_references Array of data references.
 * @param operators Array of operators to perform.
 * @param operator_source_indices Array of source indices for the operators.
 * @param num_operators Number of operators.
 * @param num_intermediates Number of intermediates, used to allocate a portion of shared memory to
 * each thread.
 */
template <cudf::size_type max_block_size, bool has_nulls>
__launch_bounds__(max_block_size) __global__
  void compute_column_kernel(table_device_view const table,
                             dev_ast_plan plan,
                             mutable_column_device_view output_column)
{
  extern __shared__ std::int64_t intermediate_storage[];
  auto thread_intermediate_storage = &intermediate_storage[threadIdx.x * plan.num_intermediates];
  auto const start_idx = static_cast<cudf::size_type>(threadIdx.x + blockIdx.x * blockDim.x);
  auto const stride    = static_cast<cudf::size_type>(blockDim.x * gridDim.x);
  auto evaluator = cudf::ast::detail::expression_evaluator<mutable_column_device_view*, has_nulls>(
    table, plan, thread_intermediate_storage, &output_column);

  for (cudf::size_type row_index = start_idx; row_index < table.num_rows(); row_index += stride) {
    evaluator.evaluate(row_index);
  }
}

std::unique_ptr<column> compute_column(table_view const table,
                                       expression const& expr,
                                       rmm::cuda_stream_view stream,
                                       rmm::mr::device_memory_resource* mr)
{
  auto const plan = ast_plan{expr, table, false, stream, mr};

  // Prepare output column
  auto output_column = cudf::make_fixed_width_column(
    plan.output_type(), table.num_rows(), mask_state::UNALLOCATED, stream, mr);
  auto mutable_output_device =
    cudf::mutable_column_device_view::create(output_column->mutable_view(), stream);

  // Configure kernel parameters
  auto const& dev_plan = plan.dev_plan;
  int device_id;
  CUDA_TRY(cudaGetDevice(&device_id));
  int shmem_limit_per_block;
  CUDA_TRY(
    cudaDeviceGetAttribute(&shmem_limit_per_block, cudaDevAttrMaxSharedMemoryPerBlock, device_id));
  auto constexpr MAX_BLOCK_SIZE = 128;
  auto const block_size =
    dev_plan.shmem_per_thread != 0
      ? std::min(MAX_BLOCK_SIZE, shmem_limit_per_block / dev_plan.shmem_per_thread)
      : MAX_BLOCK_SIZE;
  auto const config          = cudf::detail::grid_1d{table.num_rows(), block_size};
  auto const shmem_per_block = dev_plan.shmem_per_thread * config.num_threads_per_block;

  // Execute the kernel
  auto table_device = table_device_view::create(table, stream);
  cudf::ast::detail::compute_column_kernel<MAX_BLOCK_SIZE, false>
    <<<config.num_blocks, config.num_threads_per_block, shmem_per_block, stream.value()>>>(
      *table_device, dev_plan, *mutable_output_device);
  CHECK_CUDA(stream.value());
  return output_column;
}

}  // namespace detail

std::unique_ptr<column> compute_column(table_view const table,
                                       expression const& expr,
                                       rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::compute_column(table, expr, rmm::cuda_stream_default, mr);
}

}  // namespace ast

}  // namespace cudf
