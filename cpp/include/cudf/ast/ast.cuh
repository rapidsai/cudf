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

/**
 * @brief An expression evaluator owned by a single thread operating on rows of a table.
 *
 * This class is designed for n-ary transform evaluation. Currently this class assumes that there's
 * only one relevant "row index" in its methods, which corresponds to a row in a single input table
 * and the same row index in an output column.
 *
 */
struct row_evaluator {
 public:
  /**
   * @brief Construct a row evaluator.
   *
   * @param table The table device view used for evaluation.
   * @param literals Array of literal values used for evaluation.
   * @param thread_intermediate_storage Pointer to this thread's portion of shared memory for
   * storing intermediates.
   * @param output_column The output column where results are stored.
   */
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

  /**
   * @brief Resolves a data reference into a value.
   *
   * @tparam Element Type of element to return.
   * @param device_data_reference Data reference to resolve.
   * @param row_index Row index of data column.
   * @return Element
   */
  template <typename Element>
  __device__ Element resolve_input(const detail::device_data_reference device_data_reference,
                                   cudf::size_type row_index) const;

  /**
   * @brief Resolves a data reference into a pointer to an output.
   *
   * @tparam Element Type of pointer to return.
   * @param device_data_reference Data reference to resolve.
   * @param row_index Row index of data column.
   * @return Element*
   */
  template <typename Element>
  __device__ Element* resolve_output(const detail::device_data_reference device_data_reference,
                                     cudf::size_type row_index) const;

  /**
   * @brief Callable to perform a unary operation.
   *
   * @tparam OperatorFunctor Functor that performs desired operation when `operator()` is called.
   * @tparam Input Type of input value.
   * @tparam Out Type of output value, determined by `std::invoke_result_t<OperatorFunctor, Input>`.
   * @param row_index Row index of data column(s).
   * @param input Input data reference.
   * @param output Output data reference.
   */
  template <typename OperatorFunctor,
            typename Input,
            std::enable_if_t<cudf::ast::is_valid_unary_op<OperatorFunctor, Input>>* = nullptr>
  __device__ void operator()(cudf::size_type row_index,
                             const detail::device_data_reference input,
                             const detail::device_data_reference output) const;

  template <typename OperatorFunctor,
            typename Input,
            std::enable_if_t<!cudf::ast::is_valid_unary_op<OperatorFunctor, Input>>* = nullptr>
  __device__ void operator()(cudf::size_type row_index,
                             const detail::device_data_reference input,
                             const detail::device_data_reference output) const;

  /**
   * @brief Callable to perform a binary operation.
   *
   * @tparam OperatorFunctor Functor that performs desired operation when `operator()` is called.
   * @tparam LHS Type of left input value.
   * @tparam RHS Type of right input value.
   * @tparam Out Type of output value, determined by `std::invoke_result_t<OperatorFunctor, Input>`.
   * @param row_index Row index of data column(s).
   * @param lhs Left input data reference.
   * @param rhs Right input data reference.
   * @param output Output data reference.
   */
  template <typename OperatorFunctor,
            typename LHS,
            typename RHS,
            std::enable_if_t<cudf::ast::is_valid_binary_op<OperatorFunctor, LHS, RHS>>* = nullptr>
  __device__ void operator()(cudf::size_type row_index,
                             const detail::device_data_reference lhs,
                             const detail::device_data_reference rhs,
                             const detail::device_data_reference output) const;

  template <typename OperatorFunctor,
            typename LHS,
            typename RHS,
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

/**
 * @brief Evaluate an expression applied to a row.
 *
 * This function performs an n-ary transform for one row on one thread.
 *
 * @param evaluator The row evaluator used for evaluation.
 * @param data_references Array of data references.
 * @param operators Array of operators to perform.
 * @param operator_source_indices Array of source indices for the operators.
 * @param num_operators Number of operators.
 * @param row_index Row index of data column(s).
 */
__device__ void evaluate_row_expression(const detail::row_evaluator evaluator,
                                        const detail::device_data_reference* data_references,
                                        const ast_operator* operators,
                                        const cudf::size_type* operator_source_indices,
                                        cudf::size_type num_operators,
                                        cudf::size_type row_index);

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

/**
 * @brief Compute a new column by evaluating an expression tree on a table.
 *
 * This evaluates an expression over a table to produce a new column. Also called an n-ary
 * transform.
 *
 * @param table The table used for expression evaluation.
 * @param expr The root of the expression tree.
 * @param stream Stream on which to perform the computation.
 * @param mr Device memory resource.
 * @return std::unique_ptr<column> Output column.
 */
std::unique_ptr<column> compute_column(
  table_view const table,
  expression const& expr,
  cudaStream_t stream                 = 0,  // TODO use detail API
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

}  // namespace detail

/**
 * @brief Compute a new column by evaluating an expression tree on a table.
 *
 * This evaluates an expression over a table to produce a new column. Also called an n-ary
 * transform.
 *
 * @param table The table used for expression evaluation.
 * @param expr The root of the expression tree.
 * @param mr Device memory resource.
 * @return std::unique_ptr<column> Output column.
 */
std::unique_ptr<column> compute_column(
  table_view const table,
  expression const& expr,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

}  // namespace ast

}  // namespace cudf
