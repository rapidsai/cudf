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

#include <cudf/ast/linearizer.cuh>
#include <cudf/ast/operators.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/table/table.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

namespace cudf {

namespace ast {

namespace detail {
struct row_evaluator {
 public:
  __device__ row_evaluator(const table_device_view table,
                           const cudf::detail::fixed_width_scalar_device_view_base* literals,
                           std::int64_t* thread_intermediate_storage,
                           mutable_column_device_view* output_column)
    : table(table),
      literals(literals),
      thread_intermediate_storage(thread_intermediate_storage),
      output_column(output_column)
  {
  }

  template <typename Element>
  __device__ Element resolve_input(const detail::device_data_reference device_data_reference,
                                   cudf::size_type row_index) const;

  template <typename Element>
  __device__ Element* resolve_output(const detail::device_data_reference device_data_reference,
                                     cudf::size_type row_index) const;

  template <typename OperatorFunctor,
            typename Input,
            typename Out = simt::std::invoke_result_t<OperatorFunctor, Input>,
            std::enable_if_t<cudf::ast::is_valid_unary_op<OperatorFunctor, Input>>* = nullptr>
  __device__ void operator()(cudf::size_type row_index,
                             const detail::device_data_reference input,
                             const detail::device_data_reference output) const;

  template <typename OperatorFunctor,
            typename Input,
            typename Out                                                             = void,
            std::enable_if_t<!cudf::ast::is_valid_unary_op<OperatorFunctor, Input>>* = nullptr>
  __device__ void operator()(cudf::size_type row_index,
                             const detail::device_data_reference input,
                             const detail::device_data_reference output) const;

  template <typename OperatorFunctor,
            typename LHS,
            typename RHS,
            typename Out = simt::std::invoke_result_t<OperatorFunctor, LHS, RHS>,
            std::enable_if_t<cudf::ast::is_valid_binary_op<OperatorFunctor, LHS, RHS>>* = nullptr>
  __device__ void operator()(cudf::size_type row_index,
                             const detail::device_data_reference lhs,
                             const detail::device_data_reference rhs,
                             const detail::device_data_reference output) const;

  template <typename OperatorFunctor,
            typename LHS,
            typename RHS,
            typename Out                                                                 = void,
            std::enable_if_t<!cudf::ast::is_valid_binary_op<OperatorFunctor, LHS, RHS>>* = nullptr>
  __device__ void operator()(cudf::size_type row_index,
                             const detail::device_data_reference lhs,
                             const detail::device_data_reference rhs,
                             const detail::device_data_reference output) const;

 private:
  const table_device_view table;
  const cudf::detail::fixed_width_scalar_device_view_base* literals;
  std::int64_t* thread_intermediate_storage;
  mutable_column_device_view* output_column;
};

__device__ void evaluate_row_expression(const detail::row_evaluator evaluator,
                                        const detail::device_data_reference* data_references,
                                        const ast_operator* operators,
                                        const cudf::size_type* operator_source_indices,
                                        cudf::size_type num_operators,
                                        cudf::size_type row_index);

template <size_type block_size>
__launch_bounds__(block_size) __global__
  void compute_column_kernel(const table_device_view table,
                             const cudf::detail::fixed_width_scalar_device_view_base* literals,
                             mutable_column_device_view output_column,
                             const detail::device_data_reference* data_references,
                             const ast_operator* operators,
                             const cudf::size_type* operator_source_indices,
                             cudf::size_type num_operators,
                             cudf::size_type num_intermediates);
}  // namespace detail

std::unique_ptr<column> compute_column(
  table_view const table,
  std::reference_wrapper<const expression> expr,
  cudaStream_t stream                 = 0,  // TODO use detail API
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

}  // namespace ast

}  // namespace cudf
