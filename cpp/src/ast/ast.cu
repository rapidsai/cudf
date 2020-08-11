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
#include <cudf/ast/linearizer.cuh>
#include <cudf/ast/operators.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/table/table.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <rmm/device_uvector.hpp>

#include <algorithm>
#include <functional>
#include <iterator>
#include <type_traits>

namespace cudf {

namespace ast {

namespace detail {

template <typename Element>
__device__ Element row_evaluator::resolve_input(
  const detail::device_data_reference device_data_reference, cudf::size_type row_index) const
{
  auto const data_index = device_data_reference.data_index;
  switch (device_data_reference.reference_type) {
    case detail::device_data_reference_type::COLUMN: {
      auto column = this->table.column(data_index);
      return column.data<Element>()[row_index];
    }
    case detail::device_data_reference_type::LITERAL: {
      return this->literals[data_index].value<Element>();
    }
    case detail::device_data_reference_type::INTERMEDIATE: {
      return *reinterpret_cast<const Element*>(&this->thread_intermediate_storage[data_index]);
    }
    default: {
      release_assert(false && "Invalid input device data reference type.");
      return Element();
    }
  }
}

template <typename Element>
__device__ Element* row_evaluator::resolve_output(
  const detail::device_data_reference device_data_reference, cudf::size_type row_index) const
{
  switch (device_data_reference.reference_type) {
    case detail::device_data_reference_type::COLUMN: {
      // TODO: Could refactor to support output tables (multiple output columns)
      return &(this->output_column->element<Element>(row_index));
    }
    case detail::device_data_reference_type::INTERMEDIATE: {
      return reinterpret_cast<Element*>(
        &this->thread_intermediate_storage[device_data_reference.data_index]);
    }
    default: {
      release_assert(false && "Invalid output device data reference type.");
      return nullptr;
    }
  }
}

template <typename OperatorFunctor,
          typename Input,
          typename Out,
          std::enable_if_t<cudf::ast::is_valid_unary_op<OperatorFunctor, Input>>*>
__device__ void row_evaluator::operator()(cudf::size_type row_index,
                                          const detail::device_data_reference input,
                                          const detail::device_data_reference output) const
{
  auto const typed_input = this->resolve_input<Input>(input, row_index);
  auto typed_output      = this->resolve_output<Out>(output, row_index);
  *typed_output          = OperatorFunctor{}(typed_input);
}

template <typename OperatorFunctor,
          typename Input,
          typename Out,
          std::enable_if_t<!cudf::ast::is_valid_unary_op<OperatorFunctor, Input>>*>
__device__ void row_evaluator::operator()(cudf::size_type row_index,
                                          const detail::device_data_reference input,
                                          const detail::device_data_reference output) const
{
  release_assert(false && "Invalid unary dispatch operator for the provided input.");
}

template <typename OperatorFunctor,
          typename LHS,
          typename RHS,
          typename Out,
          std::enable_if_t<cudf::ast::is_valid_binary_op<OperatorFunctor, LHS, RHS>>*>
__device__ void row_evaluator::operator()(cudf::size_type row_index,
                                          const detail::device_data_reference lhs,
                                          const detail::device_data_reference rhs,
                                          const detail::device_data_reference output) const
{
  auto const typed_lhs = this->resolve_input<LHS>(lhs, row_index);
  auto const typed_rhs = this->resolve_input<RHS>(rhs, row_index);
  auto typed_output    = this->resolve_output<Out>(output, row_index);
  *typed_output        = OperatorFunctor{}(typed_lhs, typed_rhs);
}

template <typename OperatorFunctor,
          typename LHS,
          typename RHS,
          typename Out,
          std::enable_if_t<!cudf::ast::is_valid_binary_op<OperatorFunctor, LHS, RHS>>*>
__device__ void row_evaluator::operator()(cudf::size_type row_index,
                                          const detail::device_data_reference lhs,
                                          const detail::device_data_reference rhs,
                                          const detail::device_data_reference output) const
{
  release_assert(false && "Invalid binary dispatch operator for the provided input.");
}

__device__ void evaluate_row_expression(const detail::row_evaluator evaluator,
                                        const detail::device_data_reference* data_references,
                                        const ast_operator* operators,
                                        const cudf::size_type* operator_source_indices,
                                        cudf::size_type num_operators,
                                        cudf::size_type row_index)
{
  auto operator_source_index = cudf::size_type(0);
  for (cudf::size_type operator_index(0); operator_index < num_operators; operator_index++) {
    // Execute operator
    auto const op    = operators[operator_index];
    auto const arity = cudf::ast::ast_operator_arity(op);
    if (arity == 1) {
      // Unary operator
      auto const input  = data_references[operator_source_indices[operator_source_index]];
      auto const output = data_references[operator_source_indices[operator_source_index + 1]];

      unary_operator_dispatcher(op, input.data_type, evaluator, row_index, input, output);
    } else if (arity == 2) {
      // Binary operator
      auto const lhs    = data_references[operator_source_indices[operator_source_index]];
      auto const rhs    = data_references[operator_source_indices[operator_source_index + 1]];
      auto const output = data_references[operator_source_indices[operator_source_index + 2]];
      binary_operator_dispatcher(
        op, lhs.data_type, rhs.data_type, evaluator, row_index, lhs, rhs, output);
    } else {
      release_assert(false && "Invalid operator arity.");
      // Ternary operator
      /*
      auto const condition_data_ref =
        data_references[operator_source_indices[operator_source_index]];
      auto const lhs_data_ref = data_references[operator_source_indices[operator_source_index + 1]];
      auto const rhs_data_ref = data_references[operator_source_indices[operator_source_index + 2]];
      auto const output_data_ref =
        data_references[operator_source_indices[operator_source_index + 3]];
      operate(op,
              table,
              output_column,
              literals,
              thread_intermediate_storage,
              row_index,
              condition_data_ref,
              lhs_data_ref,
              rhs_data_ref,
              output_data_ref);
      */
    }
    operator_source_index += (arity + 1);
  }
}

template <size_type block_size>
__launch_bounds__(block_size) __global__
  void compute_column_kernel(const table_device_view table,
                             const cudf::detail::fixed_width_scalar_device_view_base* literals,
                             mutable_column_device_view output_column,
                             const detail::device_data_reference* data_references,
                             const ast_operator* operators,
                             const cudf::size_type* operator_source_indices,
                             cudf::size_type num_operators,
                             cudf::size_type num_intermediates)
{
  extern __shared__ std::int64_t intermediate_storage[];
  auto thread_intermediate_storage = &intermediate_storage[threadIdx.x * num_intermediates];
  const cudf::size_type start_idx  = threadIdx.x + blockIdx.x * blockDim.x;
  const cudf::size_type stride     = blockDim.x * gridDim.x;
  auto const num_rows              = table.num_rows();
  auto const evaluator =
    cudf::ast::detail::row_evaluator(table, literals, thread_intermediate_storage, &output_column);

  for (cudf::size_type row_index = start_idx; row_index < num_rows; row_index += stride) {
    evaluate_row_expression(
      evaluator, data_references, operators, operator_source_indices, num_operators, row_index);
  }
}

template <typename T>
rmm::device_uvector<T> async_create_device_data(std::vector<T> host_data, cudaStream_t stream)
{
  auto device_data = rmm::device_uvector<T>(host_data.size(), stream);
  CUDA_TRY(cudaMemcpyAsync(device_data.data(),
                           host_data.data(),
                           sizeof(T) * host_data.size(),
                           cudaMemcpyHostToDevice,
                           stream));
  return device_data;
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
  auto const device_data_references = detail::async_create_device_data(data_references, stream);
  auto const device_literals        = detail::async_create_device_data(literals, stream);
  auto const device_operators       = detail::async_create_device_data(operators, stream);
  auto const device_operator_source_indices =
    detail::async_create_device_data(operator_source_indices, stream);
  // The stream is synced later when the table_device_view is created.
  // To reduce overhead, we don't call a stream sync here.
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
  auto output_column = cudf::make_fixed_width_column(
    expr_data_type, table_num_rows, mask_state::UNALLOCATED, stream, mr);
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
  cudf::ast::detail::compute_column_kernel<block_size>
    <<<config.num_blocks, config.num_threads_per_block, shmem_size_per_block, stream>>>(
      *table_device,
      thrust::raw_pointer_cast(device_literals.data()),
      *mutable_output_device,
      thrust::raw_pointer_cast(device_data_references.data()),
      thrust::raw_pointer_cast(device_operators.data()),
      thrust::raw_pointer_cast(device_operator_source_indices.data()),
      num_operators,
      num_intermediates);
  CHECK_CUDA(stream);
  nvtxRangePop();
  return output_column;
}

}  // namespace detail

std::unique_ptr<column> compute_column(table_view const table,
                                       std::reference_wrapper<const expression> expr,
                                       rmm::mr::device_memory_resource* mr)
{
  return detail::compute_column(table, expr, 0, mr);
}

}  // namespace ast

}  // namespace cudf