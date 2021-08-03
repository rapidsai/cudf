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

#include <cudf/ast/detail/expression_evaluator.cuh>
#include <cudf/ast/detail/expression_parser.hpp>
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
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

namespace cudf {
namespace ast {
namespace detail {

/**
 * @brief Kernel for evaluating an expression on a table to produce a new column.
 *
 * This evaluates an expression over a table to produce a new column. Also called an n-ary
 * transform.
 *
 * @tparam max_block_size The size of the thread block, used to set launch
 * bounds and minimize register usage.
 * @tparam has_nulls whether or not the output column may contain nulls.
 *
 * @param table The table device view used for evaluation.
 * @param device_expression_data Container of device data required to evaluate the desired
 * expression.
 * @param output_column The destination for the results of evaluating the expression.
 */
template <cudf::size_type max_block_size, bool has_nulls>
__launch_bounds__(max_block_size) __global__
  void compute_column_kernel(table_device_view const table,
                             ast::detail::expression_device_view device_expression_data,
                             mutable_column_device_view output_column)
{
  // The (required) extern storage of the shared memory array leads to
  // conflicting declarations between different templates. The easiest
  // workaround is to declare an arbitrary (here char) array type then cast it
  // after the fact to the appropriate type.
  extern __shared__ char raw_intermediate_storage[];
  IntermediateDataType<has_nulls>* intermediate_storage =
    reinterpret_cast<IntermediateDataType<has_nulls>*>(raw_intermediate_storage);

  auto thread_intermediate_storage =
    &intermediate_storage[threadIdx.x * device_expression_data.num_intermediates];
  auto const start_idx = static_cast<cudf::size_type>(threadIdx.x + blockIdx.x * blockDim.x);
  auto const stride    = static_cast<cudf::size_type>(blockDim.x * gridDim.x);
  auto evaluator       = cudf::ast::detail::expression_evaluator<has_nulls>(
    table, device_expression_data, thread_intermediate_storage);

  for (cudf::size_type row_index = start_idx; row_index < table.num_rows(); row_index += stride) {
    auto output_dest = mutable_column_expression_result<has_nulls>(output_column);
    evaluator.evaluate(output_dest, row_index);
  }
}

std::unique_ptr<column> compute_column(table_view const table,
                                       expression const& expr,
                                       rmm::cuda_stream_view stream,
                                       rmm::mr::device_memory_resource* mr)
{
  // Prepare output column. Whether or not the output column is nullable is
  // determined by whether any of the columns in the input table are nullable.
  // If none of the input columns actually contain nulls, we can still use the
  // non-nullable version of the expression evaluation code path for
  // performance, so we capture that information as well.
  auto const nullable  = cudf::nullable(table);
  auto const has_nulls = nullable && cudf::has_nulls(table);

  auto const parser = ast::detail::expression_parser{expr, table, has_nulls, stream, mr};

  auto const output_column_mask_state =
    nullable ? (has_nulls ? mask_state::UNINITIALIZED : mask_state::ALL_VALID)
             : mask_state::UNALLOCATED;

  auto output_column = cudf::make_fixed_width_column(
    parser.output_type(), table.num_rows(), output_column_mask_state, stream, mr);
  auto mutable_output_device =
    cudf::mutable_column_device_view::create(output_column->mutable_view(), stream);

  // Configure kernel parameters
  auto const& device_expression_data = parser.device_expression_data;
  int device_id;
  CUDA_TRY(cudaGetDevice(&device_id));
  int shmem_limit_per_block;
  CUDA_TRY(
    cudaDeviceGetAttribute(&shmem_limit_per_block, cudaDevAttrMaxSharedMemoryPerBlock, device_id));
  auto constexpr MAX_BLOCK_SIZE = 128;
  auto const block_size =
    device_expression_data.shmem_per_thread != 0
      ? std::min(MAX_BLOCK_SIZE, shmem_limit_per_block / device_expression_data.shmem_per_thread)
      : MAX_BLOCK_SIZE;
  auto const config = cudf::detail::grid_1d{table.num_rows(), block_size};
  auto const shmem_per_block =
    device_expression_data.shmem_per_thread * config.num_threads_per_block;

  // Execute the kernel
  auto table_device = table_device_view::create(table, stream);
  if (has_nulls) {
    cudf::ast::detail::compute_column_kernel<MAX_BLOCK_SIZE, true>
      <<<config.num_blocks, config.num_threads_per_block, shmem_per_block, stream.value()>>>(
        *table_device, device_expression_data, *mutable_output_device);
  } else {
    cudf::ast::detail::compute_column_kernel<MAX_BLOCK_SIZE, false>
      <<<config.num_blocks, config.num_threads_per_block, shmem_per_block, stream.value()>>>(
        *table_device, device_expression_data, *mutable_output_device);
  }
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
