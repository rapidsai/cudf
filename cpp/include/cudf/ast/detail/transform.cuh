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

#include <cudf/ast/detail/linearizer.hpp>
#include <cudf/ast/detail/operators.hpp>
#include <cudf/ast/nodes.hpp>
#include <cudf/ast/operators.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/utilities/assert.cuh>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cstring>
#include <numeric>

namespace cudf {

namespace ast {

namespace detail {

// Forward declaration
struct row_evaluator;
struct two_table_evaluator;

struct row_output {
 public:
  __device__ row_output(row_evaluator const& evaluator) : evaluator(evaluator) {}

  /**
   * @brief Resolves an output data reference and assigns result value.
   *
   * Only output columns (COLUMN) and intermediates (INTERMEDIATE) are supported as output reference
   * types. Intermediates must be of fixed width less than or equal to sizeof(std::int64_t). This
   * requirement on intermediates is enforced by the linearizer.
   *
   * @tparam Element Type of result element.
   * @param device_data_reference Data reference to resolve.
   * @param row_index Row index of data column.
   * @param result Value to assign to output.
   */
  template <typename Element, CUDF_ENABLE_IF(is_rep_layout_compatible<Element>())>
  __device__ void resolve_output(detail::device_data_reference device_data_reference,
                                 cudf::size_type row_index,
                                 Element result) const;
  // Definition below after row_evaluator is a complete type

  template <typename Element, CUDF_ENABLE_IF(not is_rep_layout_compatible<Element>())>
  __device__ void resolve_output(detail::device_data_reference device_data_reference,
                                 cudf::size_type row_index,
                                 Element result) const
  {
    cudf_assert(false && "Invalid type in resolve_output.");
  }

 private:
  row_evaluator const& evaluator;
};

template <typename Input>
struct unary_row_output : public row_output {
  __device__ unary_row_output(row_evaluator const& evaluator) : row_output(evaluator) {}

  template <
    ast_operator op,
    std::enable_if_t<detail::is_valid_unary_op<detail::operator_functor<op>, Input>>* = nullptr>
  __device__ void operator()(cudf::size_type row_index,
                             Input input,
                             detail::device_data_reference output) const
  {
    using OperatorFunctor = detail::operator_functor<op>;
    using Out             = cuda::std::invoke_result_t<OperatorFunctor, Input>;
    resolve_output<Out>(output, row_index, OperatorFunctor{}(input));
  }

  template <
    ast_operator op,
    std::enable_if_t<!detail::is_valid_unary_op<detail::operator_functor<op>, Input>>* = nullptr>
  __device__ void operator()(cudf::size_type row_index,
                             Input input,
                             detail::device_data_reference output) const
  {
    cudf_assert(false && "Invalid unary dispatch operator for the provided input.");
  }
};

template <typename LHS, typename RHS>
struct binary_row_output : public row_output {
  __device__ binary_row_output(row_evaluator const& evaluator) : row_output(evaluator) {}

  template <
    ast_operator op,
    std::enable_if_t<detail::is_valid_binary_op<detail::operator_functor<op>, LHS, RHS>>* = nullptr>
  __device__ void operator()(cudf::size_type row_index,
                             LHS lhs,
                             RHS rhs,
                             detail::device_data_reference output) const
  {
    using OperatorFunctor = detail::operator_functor<op>;
    using Out             = cuda::std::invoke_result_t<OperatorFunctor, LHS, RHS>;
    resolve_output<Out>(output, row_index, OperatorFunctor{}(lhs, rhs));
  }

  template <ast_operator op,
            std::enable_if_t<!detail::is_valid_binary_op<detail::operator_functor<op>, LHS, RHS>>* =
              nullptr>
  __device__ void operator()(cudf::size_type row_index,
                             LHS lhs,
                             RHS rhs,
                             detail::device_data_reference output) const
  {
    cudf_assert(false && "Invalid binary dispatch operator for the provided input.");
  }
};

struct two_table_output {
 public:
  __device__ two_table_output(two_table_evaluator const& evaluator) : evaluator(evaluator) {}

  /**
   * @brief Resolves an output data reference and assigns result value.
   *
   * Only output columns (COLUMN) and intermediates (INTERMEDIATE) are supported as output reference
   * types. Intermediates must be of fixed width less than or equal to sizeof(std::int64_t). This
   * requirement on intermediates is enforced by the linearizer.
   *
   * @tparam Element Type of result element.
   * @param device_data_reference Data reference to resolve.
   * @param two_table_index Row index of data column.
   * @param result Value to assign to output.
   */
  template <typename Element, CUDF_ENABLE_IF(is_rep_layout_compatible<Element>())>
  __device__ void resolve_output(detail::device_data_reference device_data_reference,
                                 cudf::size_type row_index,
                                 Element result) const;
  // Definition below after two_table_evaluator is a complete type

  template <typename Element, CUDF_ENABLE_IF(not is_rep_layout_compatible<Element>())>
  __device__ void resolve_output(detail::device_data_reference device_data_reference,
                                 cudf::size_type row_index,
                                 Element result) const
  {
    cudf_assert(false && "Invalid type in resolve_output.");
  }

 private:
  two_table_evaluator const& evaluator;
};

template <typename LHS, typename RHS>
struct binary_two_table_output : public two_table_output {
  __device__ binary_two_table_output(two_table_evaluator const& evaluator)
    : two_table_output(evaluator)
  {
  }

  template <
    ast_operator op,
    std::enable_if_t<detail::is_valid_binary_op<detail::operator_functor<op>, LHS, RHS>>* = nullptr>
  __device__ void operator()(LHS lhs,
                             RHS rhs,
                             detail::device_data_reference output,
                             cudf::size_type output_row_index) const
  {
    using OperatorFunctor = detail::operator_functor<op>;
    using Out             = cuda::std::invoke_result_t<OperatorFunctor, LHS, RHS>;
    resolve_output<Out>(output, output_row_index, OperatorFunctor{}(lhs, rhs));
  }

  template <ast_operator op,
            std::enable_if_t<!detail::is_valid_binary_op<detail::operator_functor<op>, LHS, RHS>>* =
              nullptr>
  __device__ void operator()(LHS lhs,
                             RHS rhs,
                             detail::device_data_reference output,
                             cudf::size_type output_row_index) const
  {
    cudf_assert(false && "Invalid binary dispatch operator for the provided input.");
  }
};

/**
 * @brief An expression evaluator owned by a single thread operating on rows of a table.
 *
 * This class is designed for n-ary transform evaluation. Currently this class assumes that there's
 * only one relevant "row index" in its methods, which corresponds to a row in a single input table
 * and the same row index in an output column.
 */
struct row_evaluator {
  friend struct row_output;
  template <typename Input>
  friend struct unary_row_output;
  template <typename LHS, typename RHS>
  friend struct binary_row_output;

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
  __device__ row_evaluator(
    table_device_view const& table,
    device_span<const cudf::detail::fixed_width_scalar_device_view_base> literals,
    std::int64_t* thread_intermediate_storage,
    mutable_column_device_view* output_column)
    : table(table),
      literals(literals),
      thread_intermediate_storage(thread_intermediate_storage),
      output_column(output_column)
  {
  }

  /**
   * @brief Resolves an input data reference into a value.
   *
   * Only input columns (COLUMN), literal values (LITERAL), and intermediates (INTERMEDIATE) are
   * supported as input data references. Intermediates must be of fixed width less than or equal to
   * sizeof(std::int64_t). This requirement on intermediates is enforced by the linearizer.
   *
   * @tparam Element Type of element to return.
   * @param device_data_reference Data reference to resolve.
   * @param row_index Row index of data column.
   * @return Element
   */
  template <typename Element, CUDF_ENABLE_IF(column_device_view::has_element_accessor<Element>())>
  __device__ Element resolve_input(detail::device_data_reference device_data_reference,
                                   cudf::size_type row_index) const
  {
    auto const data_index = device_data_reference.data_index;
    auto const ref_type   = device_data_reference.reference_type;
    if (ref_type == detail::device_data_reference_type::COLUMN) {
      return table.column(data_index).element<Element>(row_index);
    } else if (ref_type == detail::device_data_reference_type::LITERAL) {
      return literals[data_index].value<Element>();
    } else {  // Assumes ref_type == detail::device_data_reference_type::INTERMEDIATE
      // Using memcpy instead of reinterpret_cast<Element*> for safe type aliasing
      // Using a temporary variable ensures that the compiler knows the result is aligned
      std::int64_t intermediate = thread_intermediate_storage[data_index];
      Element tmp;
      memcpy(&tmp, &intermediate, sizeof(Element));
      return tmp;
    }
  }

  template <typename Element,
            CUDF_ENABLE_IF(not column_device_view::has_element_accessor<Element>())>
  __device__ Element resolve_input(detail::device_data_reference device_data_reference,
                                   cudf::size_type row_index) const
  {
    cudf_assert(false && "Unsupported type in resolve_input.");
    return {};
  }

  /**
   * @brief Callable to perform a unary operation.
   *
   * @tparam OperatorFunctor Functor that performs desired operation when `operator()` is called.
   * @tparam Input Type of input value.
   * @param row_index Row index of data column(s).
   * @param input Input data reference.
   * @param output Output data reference.
   */
  template <typename Input>
  __device__ void operator()(cudf::size_type row_index,
                             detail::device_data_reference input,
                             detail::device_data_reference output,
                             ast_operator op) const
  {
    auto const typed_input = resolve_input<Input>(input, row_index);
    ast_operator_dispatcher(op, unary_row_output<Input>(*this), row_index, typed_input, output);
  }

  /**
   * @brief Callable to perform a binary operation.
   *
   * @tparam OperatorFunctor Functor that performs desired operation when `operator()` is called.
   * @tparam LHS Type of left input value.
   * @tparam RHS Type of right input value.
   * @param row_index Row index of data column(s).
   * @param lhs Left input data reference.
   * @param rhs Right input data reference.
   * @param output Output data reference.
   */
  template <typename LHS, typename RHS>
  __device__ void operator()(cudf::size_type row_index,
                             detail::device_data_reference lhs,
                             detail::device_data_reference rhs,
                             detail::device_data_reference output,
                             ast_operator op) const
  {
    auto const typed_lhs = resolve_input<LHS>(lhs, row_index);
    auto const typed_rhs = resolve_input<RHS>(rhs, row_index);
    ast_operator_dispatcher(
      op, binary_row_output<LHS, RHS>(*this), row_index, typed_lhs, typed_rhs, output);
  }

  template <typename OperatorFunctor,
            typename LHS,
            typename RHS,
            std::enable_if_t<!detail::is_valid_binary_op<OperatorFunctor, LHS, RHS>>* = nullptr>
  __device__ void operator()(cudf::size_type row_index,
                             detail::device_data_reference lhs,
                             detail::device_data_reference rhs,
                             detail::device_data_reference output,
                             ast_operator op) const
  {
    cudf_assert(false && "Invalid binary dispatch operator for the provided input.");
  }

 private:
  table_device_view const& table;
  device_span<const cudf::detail::fixed_width_scalar_device_view_base> literals;
  std::int64_t* thread_intermediate_storage;
  mutable_column_device_view* output_column;
};

/**
 * @brief An expression evaluator owned by a single thread operating on rows of two table.
 *
 * This class is designed for n-ary transform evaluation. It operates on two
 * separate tables, and knows the rows of each one.
 */
struct two_table_evaluator {
  friend struct two_table_output;
  // template <typename Input>
  // friend struct unary_two_table_output;
  template <typename LHS, typename RHS>
  friend struct binary_two_table_output;

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
  __device__ two_table_evaluator(
    table_device_view const& left,
    table_device_view const& right,
    device_span<const cudf::detail::fixed_width_scalar_device_view_base> literals,
    std::int64_t* thread_intermediate_storage,
    mutable_column_device_view* output_column)
    : left(left),
      right(right),
      literals(literals),
      thread_intermediate_storage(thread_intermediate_storage),
      output_column(output_column)
  {
  }

  /**
   * @brief Resolves an input data reference into a value.
   *
   * Only input columns (COLUMN), literal values (LITERAL), and intermediates (INTERMEDIATE) are
   * supported as input data references. Intermediates must be of fixed width less than or equal to
   * sizeof(std::int64_t). This requirement on intermediates is enforced by the linearizer.
   *
   * @tparam Element Type of element to return.
   * @param device_data_reference Data reference to resolve.
   * @param row_index Row index of data column.
   * @return Element
   */
  template <typename Element, CUDF_ENABLE_IF(column_device_view::has_element_accessor<Element>())>
  __device__ Element resolve_input(detail::device_data_reference device_data_reference,
                                   cudf::size_type row_index) const
  {
    auto const data_index = device_data_reference.data_index;
    auto const ref_type   = device_data_reference.reference_type;
    // TODO: Should we error check for table_reference::OUTPUT?
    auto const& table = device_data_reference.table_source == table_reference::LEFT ? left : right;
    if (ref_type == detail::device_data_reference_type::COLUMN) {
      return table.column(data_index).element<Element>(row_index);
    } else if (ref_type == detail::device_data_reference_type::LITERAL) {
      return literals[data_index].value<Element>();
    } else {  // Assumes ref_type == detail::device_data_reference_type::INTERMEDIATE
      // Using memcpy instead of reinterpret_cast<Element*> for safe type aliasing
      // Using a temporary variable ensures that the compiler knows the result is aligned
      std::int64_t intermediate = thread_intermediate_storage[data_index];
      Element tmp;
      memcpy(&tmp, &intermediate, sizeof(Element));
      return tmp;
    }
  }

  template <typename Element,
            CUDF_ENABLE_IF(not column_device_view::has_element_accessor<Element>())>
  __device__ Element resolve_input(detail::device_data_reference device_data_reference,
                                   cudf::size_type row_index) const
  {
    cudf_assert(false && "Unsupported type in resolve_input.");
    return {};
  }

  /**
   * @brief Callable to perform a unary operation.
   *
   * @tparam OperatorFunctor Functor that performs desired operation when `operator()` is called.
   * @tparam Input Type of input value.
   * @param row_index Row index of data column(s).
   * @param input Input data reference.
   * @param output Output data reference.
   */
  // template <typename Input>
  //__device__ void operator()(cudf::size_type row_index,
  //                           detail::device_data_reference input,
  //                           detail::device_data_reference output,
  //                           ast_operator op) const
  //{
  //  auto const typed_input = resolve_input<Input>(input, row_index);
  //  ast_operator_dispatcher(op, unary_row_output<Input>(*this), row_index, typed_input, output);
  //}

  /**
   * @brief Callable to perform a binary operation.
   *
   * @tparam OperatorFunctor Functor that performs desired operation when `operator()` is called.
   * @tparam LHS Type of left input value.
   * @tparam RHS Type of right input value.
   * @param row_index Row index of data column(s).
   * @param lhs Left input data reference.
   * @param rhs Right input data reference.
   * @param output Output data reference.
   */
  template <typename LHS, typename RHS>
  __device__ void operator()(cudf::size_type left_row_index,
                             cudf::size_type right_row_index,
                             detail::device_data_reference lhs,
                             detail::device_data_reference rhs,
                             detail::device_data_reference output,
                             cudf::size_type output_row_index,
                             ast_operator op) const
  {
    auto const typed_lhs = resolve_input<LHS>(lhs, left_row_index);
    auto const typed_rhs = resolve_input<RHS>(rhs, right_row_index);
    ast_operator_dispatcher(
      op, binary_two_table_output<LHS, RHS>(*this), typed_lhs, typed_rhs, output, output_row_index);
  }

  template <typename OperatorFunctor,
            typename LHS,
            typename RHS,
            std::enable_if_t<!detail::is_valid_binary_op<OperatorFunctor, LHS, RHS>>* = nullptr>
  __device__ void operator()(cudf::size_type left_row_index,
                             cudf::size_type right_row_index,
                             detail::device_data_reference lhs,
                             detail::device_data_reference rhs,
                             detail::device_data_reference output,
                             ast_operator op) const
  {
    cudf_assert(false && "Invalid binary dispatch operator for the provided input.");
  }

 private:
  table_device_view const& left;
  table_device_view const& right;
  device_span<const cudf::detail::fixed_width_scalar_device_view_base> literals;
  std::int64_t* thread_intermediate_storage;
  mutable_column_device_view* output_column;
};

template <typename Element, std::enable_if_t<is_rep_layout_compatible<Element>()>*>
__device__ void row_output::resolve_output(detail::device_data_reference device_data_reference,
                                           cudf::size_type row_index,
                                           Element result) const
{
  auto const ref_type = device_data_reference.reference_type;
  if (ref_type == detail::device_data_reference_type::COLUMN) {
    evaluator.output_column->element<Element>(row_index) = result;
  } else {  // Assumes ref_type == detail::device_data_reference_type::INTERMEDIATE
    // Using memcpy instead of reinterpret_cast<Element*> for safe type aliasing.
    // Using a temporary variable ensures that the compiler knows the result is aligned.
    std::int64_t tmp;
    memcpy(&tmp, &result, sizeof(Element));
    evaluator.thread_intermediate_storage[device_data_reference.data_index] = tmp;
  }
}

template <typename Element, std::enable_if_t<is_rep_layout_compatible<Element>()>*>
__device__ void two_table_output::resolve_output(
  detail::device_data_reference device_data_reference,
  cudf::size_type row_index,
  Element result) const
{
  auto const ref_type = device_data_reference.reference_type;
  if (ref_type == detail::device_data_reference_type::COLUMN) {
    evaluator.output_column->element<Element>(row_index) = result;
  } else {  // Assumes ref_type == detail::device_data_reference_type::INTERMEDIATE
    // Using memcpy instead of reinterpret_cast<Element*> for safe type aliasing.
    // Using a temporary variable ensures that the compiler knows the result is aligned.
    std::int64_t tmp;
    memcpy(&tmp, &result, sizeof(Element));
    evaluator.thread_intermediate_storage[device_data_reference.data_index] = tmp;
  }
}

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
__device__ inline void evaluate_row_expression(
  detail::row_evaluator const& evaluator,
  device_span<const detail::device_data_reference> data_references,
  device_span<const ast_operator> operators,
  device_span<const cudf::size_type> operator_source_indices,
  cudf::size_type row_index)
{
  auto operator_source_index = static_cast<cudf::size_type>(0);
  for (cudf::size_type operator_index = 0; operator_index < operators.size(); operator_index++) {
    // Execute operator
    auto const op    = operators[operator_index];
    auto const arity = ast_operator_arity(op);
    if (arity == 1) {
      // Unary operator
      auto const input  = data_references[operator_source_indices[operator_source_index]];
      auto const output = data_references[operator_source_indices[operator_source_index + 1]];
      operator_source_index += arity + 1;
      type_dispatcher(input.data_type, evaluator, row_index, input, output, op);
    } else if (arity == 2) {
      // Binary operator
      auto const lhs    = data_references[operator_source_indices[operator_source_index]];
      auto const rhs    = data_references[operator_source_indices[operator_source_index + 1]];
      auto const output = data_references[operator_source_indices[operator_source_index + 2]];
      operator_source_index += arity + 1;
      type_dispatcher(lhs.data_type,
                      detail::single_dispatch_binary_operator{},
                      evaluator,
                      row_index,
                      lhs,
                      rhs,
                      output,
                      op);
    } else {
      cudf_assert(false && "Invalid operator arity.");
    }
  }
}

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
__device__ inline void evaluate_join_expression(
  detail::two_table_evaluator const& evaluator,
  device_span<const detail::device_data_reference> data_references,
  device_span<const ast_operator> operators,
  device_span<const cudf::size_type> operator_source_indices,
  cudf::size_type left_row_index,
  cudf::size_type right_row_index,
  cudf::size_type output_row_index)
{
  auto operator_source_index = static_cast<cudf::size_type>(0);
  for (cudf::size_type operator_index = 0; operator_index < operators.size(); operator_index++) {
    // Execute operator
    auto const op    = operators[operator_index];
    auto const arity = ast_operator_arity(op);
    // TODO: Enable unary ops. This will require some means of indicating
    // _which_ table the operator is acting on, so the table references need to
    // get added.
    // if (arity == 1) {
    //  // Unary operator
    //  auto const input  = data_references[operator_source_indices[operator_source_index]];
    //  auto const output = data_references[operator_source_indices[operator_source_index + 1]];
    //  operator_source_index += arity + 1;
    //  type_dispatcher(input.data_type, evaluator, left_row_index, right_row_index, input, output,
    //  op);
    //} else if (arity == 2) {
    if (true) {
      // Binary operator
      auto const lhs    = data_references[operator_source_indices[operator_source_index]];
      auto const rhs    = data_references[operator_source_indices[operator_source_index + 1]];
      auto const output = data_references[operator_source_indices[operator_source_index + 2]];
      operator_source_index += arity + 1;
      type_dispatcher(lhs.data_type,
                      detail::single_dispatch_binary_operator{},
                      evaluator,
                      left_row_index,
                      right_row_index,
                      lhs,
                      rhs,
                      output,
                      output_row_index,
                      op);
    } else {
      cudf_assert(false && "Invalid operator arity.");
    }
  }
}

/**
 * @brief The AST plan creates a device buffer of data needed to execute an AST.
 *
 * On construction, an AST plan creates a single "packed" host buffer of all necessary data arrays,
 * and copies that to the device with a single host-device memory copy. Because the plan tends to be
 * small, this is the most efficient approach for low latency.
 *
 */
struct ast_plan {
  ast_plan(linearizer const& expr_linearizer,
           rmm::cuda_stream_view stream,
           rmm::mr::device_memory_resource* mr)
    : _sizes{}, _data_pointers{}
  {
    add_to_plan(expr_linearizer.data_references());
    add_to_plan(expr_linearizer.literals());
    add_to_plan(expr_linearizer.operators());
    add_to_plan(expr_linearizer.operator_source_indices());

    // Create device buffer
    auto const buffer_size = std::accumulate(_sizes.cbegin(), _sizes.cend(), 0);
    auto buffer_offsets    = std::vector<int>(_sizes.size());
    thrust::exclusive_scan(_sizes.cbegin(), _sizes.cend(), buffer_offsets.begin(), 0);

    auto h_data_buffer = std::make_unique<char[]>(buffer_size);
    for (unsigned int i = 0; i < _data_pointers.size(); ++i) {
      std::memcpy(h_data_buffer.get() + buffer_offsets[i], _data_pointers[i], _sizes[i]);
    }

    _device_data_buffer = rmm::device_buffer(h_data_buffer.get(), buffer_size, stream, mr);

    stream.synchronize();

    // Create device pointers to components of plan
    auto device_data_buffer_ptr = static_cast<const char*>(_device_data_buffer.data());
    _device_data_references     = device_span<const detail::device_data_reference>(
      reinterpret_cast<const detail::device_data_reference*>(device_data_buffer_ptr +
                                                             buffer_offsets[0]),
      expr_linearizer.data_references().size());
    _device_literals = device_span<const cudf::detail::fixed_width_scalar_device_view_base>(
      reinterpret_cast<const cudf::detail::fixed_width_scalar_device_view_base*>(
        device_data_buffer_ptr + buffer_offsets[1]),
      expr_linearizer.literals().size());
    _device_operators = device_span<const ast_operator>(
      reinterpret_cast<const ast_operator*>(device_data_buffer_ptr + buffer_offsets[2]),
      expr_linearizer.operators().size());
    _device_operator_source_indices = device_span<const cudf::size_type>(
      reinterpret_cast<const cudf::size_type*>(device_data_buffer_ptr + buffer_offsets[3]),
      expr_linearizer.operator_source_indices().size());
  }

  /**
   * @brief Helper function for adding components (operators, literals, etc) to AST plan
   *
   * @tparam T  The underlying type of the input `std::vector`
   * @param  v  The `std::vector` containing components (operators, literals, etc)
   */
  template <typename T>
  void add_to_plan(std::vector<T> const& v)
  {
    auto const data_size = sizeof(T) * v.size();
    _sizes.push_back(data_size);
    _data_pointers.push_back(v.data());
  }

  std::vector<cudf::size_type> _sizes;
  std::vector<const void*> _data_pointers;

  rmm::device_buffer _device_data_buffer;
  device_span<const detail::device_data_reference> _device_data_references;
  device_span<const cudf::detail::fixed_width_scalar_device_view_base> _device_literals;
  device_span<const ast_operator> _device_operators;
  device_span<const cudf::size_type> _device_operator_source_indices;
};

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
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());
}  // namespace detail

}  // namespace ast

}  // namespace cudf
