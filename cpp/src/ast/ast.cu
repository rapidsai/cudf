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

#include <thrust/detail/raw_pointer_cast.h>
#include <cudf/ast/ast.cuh>
#include <cudf/ast/linearizer.hpp>
#include <cudf/ast/operators.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include <rmm/device_uvector.hpp>

#include <algorithm>
#include <functional>
#include <iterator>
#include <type_traits>

namespace cudf {

namespace ast {

template <typename Element>
__device__ Element
resolve_input_data_reference(const detail::device_data_reference device_data_reference,
                             const table_device_view table,
                             const std::int64_t* thread_intermediate_storage,
                             cudf::size_type row_index)
{
  switch (device_data_reference.reference_type) {
    case detail::device_data_reference_type::COLUMN: {
      auto column = table.column(device_data_reference.data_index);
      return column.data<Element>()[row_index];
    }
    case detail::device_data_reference_type::LITERAL: {
      // TODO: Fetch and return literal.
      return Element();
    }
    case detail::device_data_reference_type::INTERMEDIATE: {
      return *reinterpret_cast<const Element* const>(
        &thread_intermediate_storage[device_data_reference.data_index]);
    }
    default: {
      // TODO: Error
      return Element();
    }
  }
}

template <typename Element>
__device__ Element* resolve_output_data_reference(
  const detail::device_data_reference device_data_reference,
  const table_device_view table,
  mutable_column_device_view output_column,
  std::int64_t* thread_intermediate_storage,
  cudf::size_type row_index)
{
  switch (device_data_reference.reference_type) {
    case detail::device_data_reference_type::COLUMN: {
      // TODO: Could refactor to support output tables (multiple output columns)
      return &(output_column.element<Element>(row_index));
    }
    case detail::device_data_reference_type::INTERMEDIATE: {
      return reinterpret_cast<Element*>(
        &thread_intermediate_storage[device_data_reference.data_index]);
    }
    default: {
      // TODO: Error
      return nullptr;
    }
  }
}

template <typename OperatorFunctor,
          typename LHS,
          typename RHS,
          typename Out,
          std::enable_if_t<cudf::ast::is_valid_binary_op<OperatorFunctor, LHS, RHS>>*>
CUDA_HOST_DEVICE_CALLABLE decltype(auto) typed_operator_dispatch_functor::operator()(
  const table_device_view table,
  mutable_column_device_view output_column,
  std::int64_t* thread_intermediate_storage,
  cudf::size_type row_index,
  const detail::device_data_reference lhs,
  const detail::device_data_reference rhs,
  const detail::device_data_reference output)
{
  auto const typed_lhs =
    resolve_input_data_reference<LHS>(lhs, table, thread_intermediate_storage, row_index);
  auto const typed_rhs =
    resolve_input_data_reference<RHS>(rhs, table, thread_intermediate_storage, row_index);
  auto typed_output = resolve_output_data_reference<Out>(
    output, table, output_column, thread_intermediate_storage, row_index);
  *typed_output = OperatorFunctor{}(typed_lhs, typed_rhs);
}

template <typename OperatorFunctor,
          typename LHS,
          typename RHS,
          typename Out,
          std::enable_if_t<!cudf::ast::is_valid_binary_op<OperatorFunctor, LHS, RHS>>*>
CUDA_HOST_DEVICE_CALLABLE decltype(auto) typed_operator_dispatch_functor::operator()(
  const table_device_view table,
  mutable_column_device_view output_column,
  std::int64_t* thread_intermediate_storage,
  cudf::size_type row_index,
  const detail::device_data_reference lhs,
  const detail::device_data_reference rhs,
  const detail::device_data_reference output)
{
  // TODO: Need a template to match unsupported types, or prevent the compiler from attempting to
  // compile unsupported types here.
}

__device__ void operate(ast_operator op,
                        const table_device_view table,
                        mutable_column_device_view output_column,
                        std::int64_t* thread_intermediate_storage,
                        cudf::size_type row_index,
                        const detail::device_data_reference lhs,
                        const detail::device_data_reference rhs,
                        const detail::device_data_reference output)
{
  ast_operator_dispatcher(op,
                          lhs.data_type,
                          rhs.data_type,
                          typed_operator_dispatch_functor{},
                          table,
                          output_column,
                          thread_intermediate_storage,
                          row_index,
                          lhs,
                          rhs,
                          output);
}

__device__ void evaluate_row_expression(const table_device_view table,
                                        const detail::device_data_reference* data_references,
                                        // const scalar* literals,
                                        const ast_operator* operators,
                                        const cudf::size_type* operator_source_indices,
                                        cudf::size_type num_operators,
                                        cudf::size_type row_index,
                                        std::int64_t* thread_intermediate_storage,
                                        mutable_column_device_view output_column)
{
  auto operator_source_index = cudf::size_type(0);
  for (cudf::size_type operator_index(0); operator_index < num_operators; operator_index++) {
    // Execute operator
    auto const op = operators[operator_index];
    // TODO: Fix binary operator trait for new dispatch
    // if (is_binary_operator(op)) {
    if (true) {
      auto const lhs_data_ref = data_references[operator_source_indices[operator_source_index]];
      auto const rhs_data_ref = data_references[operator_source_indices[operator_source_index + 1]];
      auto const output_data_ref =
        data_references[operator_source_indices[operator_source_index + 2]];
      operator_source_index += 3;
      operate(op,
              table,
              output_column,
              thread_intermediate_storage,
              row_index,
              lhs_data_ref,
              rhs_data_ref,
              output_data_ref);
    } else {
      // TODO: Support unary/ternary operators
      // Assume operator is unary
      /*
      auto const input_data_ref = data_references[operator_source_indices[operator_source_index]];
      auto const output_data_ref =
        data_references[operator_source_indices[operator_source_index + 1]];
      operator_source_index += 2;
      */
    }
  }
}

template <size_type block_size>
__launch_bounds__(block_size) __global__
  void compute_column_kernel(const table_device_view table,
                             const detail::device_data_reference* data_references,
                             // const scalar* literals,
                             const ast_operator* operators,
                             const cudf::size_type* operator_source_indices,
                             cudf::size_type num_operators,
                             cudf::size_type num_intermediates,
                             mutable_column_device_view output_column)
{
  extern __shared__ std::int64_t intermediate_storage[];
  auto thread_intermediate_storage = &intermediate_storage[threadIdx.x * num_intermediates];
  const cudf::size_type start_idx  = threadIdx.x + blockIdx.x * blockDim.x;
  const cudf::size_type stride     = blockDim.x * gridDim.x;
  auto const num_rows              = table.num_rows();

  for (cudf::size_type row_index = start_idx; row_index < num_rows; row_index += stride) {
    evaluate_row_expression(table,
                            data_references,
                            // literals,
                            operators,
                            operator_source_indices,
                            num_operators,
                            row_index,
                            thread_intermediate_storage,
                            output_column);
  }
}

std::unique_ptr<column> compute_column(table_view const table,
                                       std::reference_wrapper<const expression> expr,
                                       cudaStream_t stream,
                                       rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  // Linearize the AST
  nvtxRangePush("Linearizing...");
  auto expr_linearizer = linearizer(table);
  expr.get().accept(expr_linearizer);
  auto const data_references         = expr_linearizer.get_data_references();
  auto const literals                = expr_linearizer.get_literals();
  auto const operators               = expr_linearizer.get_operators();
  auto const num_operators           = cudf::size_type(operators.size());
  auto const operator_source_indices = expr_linearizer.get_operator_source_indices();
  auto const expr_data_type          = expr_linearizer.get_root_data_type();
  nvtxRangePop();

  // Create device data
  nvtxRangePush("Creating device data...");
  auto device_data_references =
    rmm::device_uvector<detail::device_data_reference>(data_references.size(), stream);
  CUDA_TRY(cudaMemcpyAsync(device_data_references.data(),
                           data_references.data(),
                           sizeof(detail::device_data_reference) * data_references.size(),
                           cudaMemcpyHostToDevice,
                           stream));
  // TODO: Literals
  // auto device_literals = thrust::device_vector<const scalar>();
  auto device_operators = rmm::device_uvector<cudf::ast::ast_operator>(operators.size(), stream);
  CUDA_TRY(cudaMemcpyAsync(device_operators.data(),
                           operators.data(),
                           sizeof(cudf::ast::ast_operator) * operators.size(),
                           cudaMemcpyHostToDevice,
                           stream));
  auto device_operator_source_indices =
    rmm::device_uvector<cudf::size_type>(operator_source_indices.size(), stream);
  CUDA_TRY(cudaMemcpyAsync(device_operator_source_indices.data(),
                           operator_source_indices.data(),
                           sizeof(cudf::size_type) * operator_source_indices.size(),
                           cudaMemcpyHostToDevice,
                           stream));
  // The stream is synced later when the table_device_view is created.
  // CUDA_TRY(cudaStreamSynchronize(stream));
  nvtxRangePop();

  // Output linearizer info
  /*
  std::cout << "LINEARIZER INFO:" << std::endl;
  std::cout << "Number of data references: " << data_references.size() << std::endl;
  std::cout << "Data references: ";
  for (auto const& dr : data_references) {
    switch (dr.reference_type) {
      case detail::device_data_reference_type::COLUMN: std::cout << "C"; break;
      case detail::device_data_reference_type::LITERAL: std::cout << "L"; break;
      case detail::device_data_reference_type::INTERMEDIATE: std::cout << "I";
    }
    std::cout << dr.data_index << ", ";
  }
  std::cout << std::endl;
  std::cout << "Number of operators: " << num_operators << std::endl;
  std::cout << "Number of operator source indices: " << operator_source_indices.size() << std::endl;
  std::cout << "Number of literals: " << literals.size() << std::endl;
  std::cout << "Operator source indices: ";
  for (auto const& v : operator_source_indices) { std::cout << v << ", "; }
  std::cout << std::endl;
  */

  // Create table device view
  nvtxRangePush("Creating table device view...");
  auto table_device         = table_device_view::create(table, stream);
  auto const table_num_rows = table.num_rows();
  nvtxRangePop();

  // Prepare output column
  nvtxRangePush("Preparing output column...");
  auto output_column =
    make_fixed_width_column(expr_data_type, table_num_rows, mask_state::UNALLOCATED, stream, mr);
  auto mutable_output_device =
    cudf::mutable_column_device_view::create(output_column->mutable_view(), stream);
  nvtxRangePop();

  // Configure kernel parameters
  nvtxRangePush("Configuring kernel parameters...");
  auto constexpr block_size = 512;
  cudf::detail::grid_1d config(table_num_rows, block_size);
  auto const num_intermediates = expr_linearizer.get_intermediate_count();
  auto const shmem_size_per_block =
    sizeof(std::int64_t) * num_intermediates * config.num_threads_per_block;
  /*
  std::cout << "Requesting " << config.num_blocks << " blocks, ";
  std::cout << config.num_threads_per_block << " threads/block, ";
  std::cout << shmem_size_per_block << " bytes of shared memory." << std::endl;
  */
  nvtxRangePop();

  // Execute the kernel
  nvtxRangePush("Executing AST kernel...");
  compute_column_kernel<block_size>
    <<<config.num_blocks, config.num_threads_per_block, shmem_size_per_block, stream>>>(
      *table_device,
      thrust::raw_pointer_cast(device_data_references.data()),
      // device_literals,
      thrust::raw_pointer_cast(device_operators.data()),
      thrust::raw_pointer_cast(device_operator_source_indices.data()),
      num_operators,
      num_intermediates,
      *mutable_output_device);
  CHECK_CUDA(stream);
  nvtxRangePop();
  return output_column;
}

}  // namespace ast

}  // namespace cudf