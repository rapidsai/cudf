/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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
#include <cudf/ast/expressions.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/transform.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

namespace cudf {
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
__launch_bounds__(max_block_size) CUDF_KERNEL
  void compute_column_kernel(table_device_view const table,
                             ast::detail::expression_device_view device_expression_data,
                             mutable_column_device_view output_column)
{
  // The (required) extern storage of the shared memory array leads to
  // conflicting declarations between different templates. The easiest
  // workaround is to declare an arbitrary (here char) array type then cast it
  // after the fact to the appropriate type.
  extern __shared__ char raw_intermediate_storage[];
  ast::detail::IntermediateDataType<has_nulls>* intermediate_storage =
    reinterpret_cast<ast::detail::IntermediateDataType<has_nulls>*>(raw_intermediate_storage);

  auto thread_intermediate_storage =
    &intermediate_storage[threadIdx.x * device_expression_data.num_intermediates];
  auto start_idx    = cudf::detail::grid_1d::global_thread_id();
  auto const stride = cudf::detail::grid_1d::grid_stride();
  auto evaluator =
    cudf::ast::detail::expression_evaluator<has_nulls>(table, device_expression_data);

  for (thread_index_type row_index = start_idx; row_index < table.num_rows(); row_index += stride) {
    auto output_dest = ast::detail::mutable_column_expression_result<has_nulls>(output_column);
    evaluator.evaluate(output_dest, row_index, thread_intermediate_storage);
  }
}

std::unique_ptr<column> compute_column(table_view const& table,
                                       ast::expression const& expr,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  // If evaluating the expression may produce null outputs we create a nullable
  // output column and follow the null-supporting expression evaluation code
  // path.
  auto const has_nulls = expr.may_evaluate_null(table, stream);

  auto const parser = ast::detail::expression_parser{expr, table, has_nulls, stream, mr};

  auto const output_column_mask_state =
    has_nulls ? mask_state::UNINITIALIZED : mask_state::UNALLOCATED;

  auto output_column = cudf::make_fixed_width_column(
    parser.output_type(), table.num_rows(), output_column_mask_state, stream, mr);
  if (table.num_rows() == 0) { return output_column; }
  auto mutable_output_device =
    cudf::mutable_column_device_view::create(output_column->mutable_view(), stream);

  // Configure kernel parameters
  auto const& device_expression_data = parser.device_expression_data;
  int device_id;
  CUDF_CUDA_TRY(cudaGetDevice(&device_id));
  int shmem_limit_per_block;
  CUDF_CUDA_TRY(
    cudaDeviceGetAttribute(&shmem_limit_per_block, cudaDevAttrMaxSharedMemoryPerBlock, device_id));
  auto constexpr MAX_BLOCK_SIZE = 128;
  auto const block_size =
    parser.shmem_per_thread != 0
      ? std::min(MAX_BLOCK_SIZE, shmem_limit_per_block / parser.shmem_per_thread)
      : MAX_BLOCK_SIZE;
  auto const config          = cudf::detail::grid_1d{table.num_rows(), block_size};
  auto const shmem_per_block = parser.shmem_per_thread * config.num_threads_per_block;

  // Execute the kernel
  auto table_device = table_device_view::create(table, stream);
  if (has_nulls) {
    cudf::detail::compute_column_kernel<MAX_BLOCK_SIZE, true>
      <<<config.num_blocks, config.num_threads_per_block, shmem_per_block, stream.value()>>>(
        *table_device, device_expression_data, *mutable_output_device);
  } else {
    cudf::detail::compute_column_kernel<MAX_BLOCK_SIZE, false>
      <<<config.num_blocks, config.num_threads_per_block, shmem_per_block, stream.value()>>>(
        *table_device, device_expression_data, *mutable_output_device);
  }
  CUDF_CHECK_CUDA(stream.value());
  output_column->set_null_count(
    cudf::detail::null_count(mutable_output_device->null_mask(), 0, output_column->size(), stream));
  return output_column;
}

}  // namespace detail

std::unique_ptr<column> compute_column(table_view const& table,
                                       ast::expression const& expr,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::compute_column(table, expr, stream, mr);
}

}  // namespace cudf
