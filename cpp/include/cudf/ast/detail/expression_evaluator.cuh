/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

#include <cudf/ast/detail/expression_parser.hpp>
#include <cudf/ast/detail/operators.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/utilities/assert.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {

namespace ast {

namespace detail {

/**
 * @brief A container for capturing the output of an evaluated expression in a scalar.
 *
 * This subclass of `expression_result` functions as an owning container of a
 * (possibly nullable) scalar type that can be written to by the
 * expression_evaluator. The data (and its validity) can then be accessed.
 *
 * @tparam T The underlying data type.
 * @tparam has_nulls Whether or not the result data is nullable.
 */
template <typename T>
struct value_expression_result {
  __device__ inline value_expression_result() {}

  template <typename Element>
  __device__ inline void set_value(cudf::size_type index, Element const& result)
  {
    _obj = result;
  }

  template <typename Element>
  __device__ inline void set_null(cudf::size_type index)
  {
    _obj = thrust::nullopt;
  }

  /**
   * @brief Returns true if the underlying data is valid and false otherwise.
   */
  [[nodiscard]] __device__ inline bool is_valid() const { return _obj.has_value(); }

  /**
   * @brief Returns the underlying data.
   *
   * If the underlying data is not valid, behavior is undefined. Callers should
   * use is_valid to check for validity before accessing the value.
   */
  __device__ inline T value() const { return *_obj; }

  thrust::optional<T> _obj;  ///< The underlying data value, or a nullable version of it.
};

// TODO: The below implementation significantly differs from the default
// implementation above due to the non-owning nature of the container and the
// usage of the index. It would be ideal to unify these further if possible.

/**
 * @brief A container for capturing the output of an evaluated expression in a column.
 *
 * This subclass of `expression_result` functions as a non-owning container
 * that transparently passes calls through to an underlying mutable view to a
 * column. Not all methods are implemented
 *
 * @tparam has_nulls Whether or not the result data is nullable.
 */
struct mutable_column_expression_result {
  __device__ inline mutable_column_expression_result(mutable_column_device_view& obj) : _obj(obj) {}

  template <typename Element>
  __device__ inline void set_value(cudf::size_type index, Element const& result)
  {
    _obj.template element<Element>(index) = result;
    if (_obj.nullable()) { _obj.set_valid(index); }
  }

  template <typename Element>
  __device__ inline void set_null(cudf::size_type index)
  {
    if (_obj.nullable()) { _obj.set_null(index); }
  }

  /**
   * @brief Not implemented for this specialization.
   */
  [[nodiscard]] __device__ inline bool is_valid() const
  {
    // Not implemented since it would require modifying the API in the parent class to accept an
    // index.
    CUDF_UNREACHABLE("This method is not implemented.");
  }

  /**
   * @brief Not implemented for this specialization.
   */
  [[nodiscard]] __device__ inline mutable_column_device_view value() const
  {
    // Not implemented since it would require modifying the API in the parent class to accept an
    // index.
    CUDF_UNREACHABLE("This method is not implemented.");
  }

  mutable_column_device_view& _obj;  ///< The column to which the data is written.
};

/**
 * @brief Dispatch to a binary operator based on a single data type.
 *
 * This functor is a dispatcher for binary operations that assumes that both
 * operands are of the same type. This assumption is encoded in the
 * non-deducible template parameter LHS, the type of the left-hand operand,
 * which is then used as the template parameter for both the left and right
 * operands to the binary operator f.
 */
struct single_dispatch_binary_operator {
  /**
   * @brief Single-type dispatch to a binary operation.
   *
   * @tparam LHS Left input type.
   * @tparam F Type of forwarded binary operator functor.
   * @tparam Ts Parameter pack of forwarded arguments.
   *
   * @param f Binary operator functor.
   * @param args Forwarded arguments to `operator()` of `f`.
   */
  template <typename LHS, typename F, typename... Ts>
  __device__ inline auto operator()(F&& f, Ts&&... args)
  {
    f.template operator()<LHS, LHS>(std::forward<Ts>(args)...);
  }
};

/**
 * @brief The principal object for evaluating AST expressions on device.
 *
 * This class is designed for n-ary transform evaluation. It operates on two
 * tables.
 */
struct expression_evaluator {
 public:
  /**
   * @brief Construct an expression evaluator acting on two tables.
   *
   * @param left View of the left table view used for evaluation.
   * @param right View of the right table view used for evaluation.
   * @param plan The collection of device references representing the expression to evaluate.
   * @param thread_intermediate_storage Pointer to this thread's portion of shared memory for
   * storing intermediates.

   */
  __device__ inline expression_evaluator(table_device_view const& left,
                                         table_device_view const& right,
                                         expression_device_view const& plan,
                                         bool has_nulls)
    : left(left), right(right), plan(plan), has_nulls(has_nulls)
  {
  }

  /**
   * @brief Construct an expression evaluator acting on one table.
   *
   * @param table View of the table view used for evaluation.
   * @param plan The collection of device references representing the expression to evaluate.
   * @param thread_intermediate_storage Pointer to this thread's portion of shared memory for
   * storing intermediates.
   */
  __device__ inline expression_evaluator(table_device_view const& table,
                                         expression_device_view const& plan,
                                         bool has_nulls)
    : expression_evaluator(table, table, plan, has_nulls)
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
   * @tparam has_nulls Whether or not the result data is nullable.
   * @param device_data_reference Data reference to resolve.
   * @param row_index Row index of data column.
   * @return Element The type- and null-resolved data.
   */

  template <typename Element,
            bool has_nulls,
            CUDF_ENABLE_IF(column_device_view::has_element_accessor<Element>())>
  __device__ inline possibly_null_value_t<Element, has_nulls> resolve_input(
    detail::device_data_reference const& input_reference,
    char* thread_intermediate_storage,
    cudf::size_type left_row_index,
    cudf::size_type right_row_index = {}) const
  {
    // TODO: Everywhere in the code assumes that the table reference is either
    // left or right. Should we error-check somewhere to prevent
    // table_reference::OUTPUT from being specified?
    using ReturnType = possibly_null_value_t<Element, has_nulls>;
    if (input_reference.reference_type == detail::device_data_reference_type::COLUMN) {
      // If we have nullable data, return an empty nullable type with no value if the data is null.
      auto const& table = (input_reference.table_source == table_reference::LEFT) ? left : right;
      // Note that the code below assumes that a right index has been passed in
      // any case where input_reference.table_source == table_reference::RIGHT.
      // Otherwise, behavior is undefined.
      auto const row_index =
        (input_reference.table_source == table_reference::LEFT) ? left_row_index : right_row_index;
      if constexpr (has_nulls) {
        return table.column(input_reference.data_index).is_valid(row_index)
                 ? ReturnType(table.column(input_reference.data_index).element<Element>(row_index))
                 : ReturnType();

      } else {
        return ReturnType(table.column(input_reference.data_index).element<Element>(row_index));
      }
    } else if (input_reference.reference_type == detail::device_data_reference_type::LITERAL) {
      if constexpr (has_nulls) {
        return plan.literals[input_reference.data_index].is_valid()
                 ? ReturnType(plan.literals[input_reference.data_index].value<Element>())
                 : ReturnType();

      } else {
        return ReturnType(plan.literals[input_reference.data_index].value<Element>());
      }
    } else {  // Assumes input_reference.reference_type ==
              // detail::device_data_reference_type::INTERMEDIATE
      // Using memcpy instead of reinterpret_cast<Element*> for safe type aliasing
      // Using a temporary variable ensures that the compiler knows the result is aligned
      IntermediateDataType<has_nulls> intermediate =
        thread_intermediate_storage[input_reference.data_index];
      ReturnType tmp;
      memcpy(&tmp, &intermediate, sizeof(ReturnType));
      return tmp;
    }
    // Unreachable return used to silence compiler warnings.
    return {};
  }

  template <typename Element,
            bool has_nulls,
            CUDF_ENABLE_IF(not column_device_view::has_element_accessor<Element>())>
  __device__ inline possibly_null_value_t<Element, has_nulls> resolve_input(
    detail::device_data_reference const& device_data_reference,
    char* thread_intermediate_storage,
    cudf::size_type left_row_index,
    cudf::size_type right_row_index = {}) const
  {
    CUDF_UNREACHABLE("Unsupported type in resolve_input.");
  }

  /**
   * @brief Callable to perform a unary operation.
   *
   * @tparam Input Type of input value.
   * @tparam OutputType The container type that data will be inserted into.
   *
   * @param output_object The container that data will be inserted into.
   * @param input_row_index The row to pull the data from the input table.
   * @param input Input data reference.
   * @param output Output data reference.
   * @param output_row_index The row in the output to insert the result.
   * @param op The operator to act with.
   */
  template <typename Input, typename ExpressionResult>
  __device__ inline void operator()(ExpressionResult& output_object,
                                    cudf::size_type const input_row_index,
                                    detail::device_data_reference const& input,
                                    detail::device_data_reference const& output,
                                    cudf::size_type const output_row_index,
                                    ast_operator const op,
                                    char* thread_intermediate_storage) const
  {
    if (has_nulls) {
      auto const typed_input =
        resolve_input<Input, true>(input, thread_intermediate_storage, input_row_index);
      ast_operator_dispatcher(op,
                              unary_expression_output_handler<Input>{},
                              output_object,
                              output_row_index,
                              typed_input,
                              output,
                              thread_intermediate_storage);

    } else {
      auto const typed_input =
        resolve_input<Input, false>(input, thread_intermediate_storage, input_row_index);
      ast_operator_dispatcher(op,
                              unary_expression_output_handler<Input>{},
                              output_object,
                              output_row_index,
                              typed_input,
                              output,
                              thread_intermediate_storage);
    }
  }

  /**
   * @brief Callable to perform a binary operation.
   *
   * @tparam LHS Type of the left input value.
   * @tparam RHS Type of the right input value.
   * @tparam OutputType The container type that data will be inserted into.
   *
   * @param output_object The container that data will be inserted into.
   * @param left_row_index The row to pull the data from the left table.
   * @param right_row_index The row to pull the data from the right table.
   * @param lhs Left input data reference.
   * @param rhs Right input data reference.
   * @param output Output data reference.
   * @param output_row_index The row in the output to insert the result.
   * @param op The operator to act with.
   */
  template <typename LHS, typename RHS, typename ExpressionResult>
  __device__ inline void operator()(ExpressionResult& output_object,
                                    cudf::size_type const left_row_index,
                                    cudf::size_type const right_row_index,
                                    detail::device_data_reference const& lhs,
                                    detail::device_data_reference const& rhs,
                                    detail::device_data_reference const& output,
                                    cudf::size_type const output_row_index,
                                    ast_operator const op,
                                    char* thread_intermediate_storage) const
  {
    if (has_nulls) {
      auto const typed_lhs =
        resolve_input<LHS, true>(lhs, thread_intermediate_storage, left_row_index, right_row_index);
      auto const typed_rhs =
        resolve_input<RHS, true>(rhs, thread_intermediate_storage, left_row_index, right_row_index);
      ast_operator_dispatcher(op,
                              binary_expression_output_handler<LHS, RHS>{},
                              output_object,
                              output_row_index,
                              typed_lhs,
                              typed_rhs,
                              output,
                              thread_intermediate_storage);
    } else {
      auto const typed_lhs = resolve_input<LHS, false>(
        lhs, thread_intermediate_storage, left_row_index, right_row_index);
      auto const typed_rhs = resolve_input<RHS, false>(
        rhs, thread_intermediate_storage, left_row_index, right_row_index);
      ast_operator_dispatcher(op,
                              binary_expression_output_handler<LHS, RHS>{},
                              output_object,
                              output_row_index,
                              typed_lhs,
                              typed_rhs,
                              output,
                              thread_intermediate_storage);
    }
  }

  /**
   * @brief Evaluate an expression applied to a row.
   *
   * This function performs an n-ary transform for one row on one thread.
   *
   * @tparam OutputType The container type that data will be inserted into.
   *
   * @param output_object The container that data will be inserted into.
   * @param row_index Row index of all input and output data column(s).
   */
  template <typename ExpressionResult>
  __device__ __forceinline__ void evaluate(ExpressionResult& output_object,
                                           cudf::size_type const row_index,
                                           char* thread_intermediate_storage) const
  {
    evaluate(output_object, row_index, row_index, row_index, thread_intermediate_storage);
  }

  /**
   * @brief Evaluate an expression applied to a row.
   *
   * This function performs an n-ary transform for one row on one thread.
   *
   * @tparam OutputType The container type that data will be inserted into.
   *
   * @param output_object The container that data will be inserted into.
   * @param left_row_index The row to pull the data from the left table.
   * @param right_row_index The row to pull the data from the right table.
   * @param output_row_index The row in the output to insert the result.
   */
  template <typename ExpressionResult>
  __device__ __forceinline__ void evaluate(ExpressionResult& output_object,
                                           cudf::size_type const left_row_index,
                                           cudf::size_type const right_row_index,
                                           cudf::size_type const output_row_index,
                                           char* thread_intermediate_storage) const
  {
    cudf::size_type operator_source_index{0};
    for (cudf::size_type operator_index = 0; operator_index < plan.operators.size();
         ++operator_index) {
      // TODO(lamarrr): dispatch on has_nulls
      // Execute operator
      auto const op    = plan.operators[operator_index];
      auto const arity = plan.operator_arities[operator_index];
      if (arity == 1) {
        // Unary operator
        auto const& input =
          plan.data_references[plan.operator_source_indices[operator_source_index++]];
        auto const& output =
          plan.data_references[plan.operator_source_indices[operator_source_index++]];
        auto input_row_index =
          input.table_source == table_reference::LEFT ? left_row_index : right_row_index;
        type_dispatcher(input.data_type,
                        *this,
                        output_object,
                        input_row_index,
                        input,
                        output,
                        output_row_index,
                        op,
                        thread_intermediate_storage);
      } else if (arity == 2) {
        // Binary operator
        auto const& lhs =
          plan.data_references[plan.operator_source_indices[operator_source_index++]];
        auto const& rhs =
          plan.data_references[plan.operator_source_indices[operator_source_index++]];
        auto const& output =
          plan.data_references[plan.operator_source_indices[operator_source_index++]];
        type_dispatcher(lhs.data_type,
                        detail::single_dispatch_binary_operator{},
                        *this,
                        output_object,
                        left_row_index,
                        right_row_index,
                        lhs,
                        rhs,
                        output,
                        output_row_index,
                        op,
                        thread_intermediate_storage);
      } else {
        CUDF_UNREACHABLE("Invalid operator arity.");
      }
    }
  }

 private:
  /**
   * @brief Helper struct for type dispatch on the result of an expression.
   *
   * Evaluating an expression requires multiple levels of type dispatch to
   * determine the input types, the operation type, and the output type. This
   * helper class is a functor that handles the operator dispatch, invokes the
   * operator, and dispatches output writing based on the resulting data type.
   */
  struct expression_output_handler {
   public:
    __device__ inline expression_output_handler() {}

    /**
     * @brief Resolves an output data reference and assigns result value.
     *
     * Only output columns (COLUMN) and intermediates (INTERMEDIATE) are supported as output
     * reference types. Intermediates must be of fixed width less than or equal to
     * sizeof(std::int64_t). This requirement on intermediates is enforced by the linearizer.
     *
     * @tparam Element Type of result element.
     * @tparam OutputType The container type that data will be inserted into.
     *
     * @param output_object The container that data will be inserted into.
     * @param device_data_reference Data reference to resolve.
     * @param row_index Row index of data column.
     * @param result Value to assign to output.
     */
    template <typename Element,
              typename ExpressionResult,
              bool has_nulls,
              CUDF_ENABLE_IF(is_rep_layout_compatible<Element>())>
    __device__ inline void resolve_output(
      ExpressionResult& output_object,
      detail::device_data_reference const& device_data_reference,
      cudf::size_type const row_index,
      char* thread_intermediate_storage,
      possibly_null_value_t<Element, has_nulls> const& result) const
    {
      if (device_data_reference.reference_type == detail::device_data_reference_type::COLUMN) {
        output_object.template set_value<Element>(row_index, result);
      } else {  // Assumes device_data_reference.reference_type ==
                // detail::device_data_reference_type::INTERMEDIATE
        // Using memcpy instead of reinterpret_cast<Element*> for safe type aliasing.
        // Using a temporary variable ensures that the compiler knows the result is aligned.
        IntermediateDataType<has_nulls> tmp;
        memcpy(&tmp, &result, sizeof(possibly_null_value_t<Element, has_nulls>));
        thread_intermediate_storage[device_data_reference.data_index] = tmp;
      }
    }

    template <typename Element,
              typename ExpressionResult,
              bool has_nulls,
              CUDF_ENABLE_IF(!is_rep_layout_compatible<Element>())>
    __device__ inline void resolve_output(
      ExpressionResult& output_object,
      detail::device_data_reference const& device_data_reference,
      cudf::size_type const row_index,
      char* thread_intermediate_storage,
      possibly_null_value_t<Element, has_nulls> const& result) const
    {
      CUDF_UNREACHABLE("Invalid type in resolve_output.");
    }
  };

  /**
   * @brief Subclass of the expression output handler for unary operations.
   *
   * This functor's call operator is specialized to handle unary operations,
   * which only require a single operand.
   */
  template <typename Input>
  struct unary_expression_output_handler : public expression_output_handler {
    __device__ inline unary_expression_output_handler() {}

    /**
     * @brief Callable to perform a unary operation.
     *
     * @tparam op The operation to perform.
     * @tparam OutputType The container type that data will be inserted into.
     *
     * @param output_object The container that data will be inserted into.
     * @param outputrow_index The row in the output object to insert the data.
     * @param input Input to the operation.
     * @param output Output data reference.
     */
    template <ast_operator op,
              typename ExpressionResult,
              std::enable_if_t<
                detail::is_valid_unary_op<detail::operator_functor<op, has_nulls>,
                                          possibly_null_value_t<Input, has_nulls>>>* = nullptr>
    __device__ inline void operator()(ExpressionResult& output_object,
                                      cudf::size_type const output_row_index,
                                      possibly_null_value_t<Input, has_nulls> const& input,
                                      detail::device_data_reference const& output,
                                      char* thread_intermediate_storage) const
    {
      // The output data type is the same whether or not nulls are present, so
      // pull from the non-nullable operator.
      using Out = cuda::std::invoke_result_t<detail::operator_functor<op, false>, Input>;
      this->template resolve_output<Out>(output_object,
                                         output,
                                         output_row_index,
                                         thread_intermediate_storage,
                                         detail::operator_functor<op, has_nulls>{}(input));
    }

    template <ast_operator op,
              typename ExpressionResult,
              std::enable_if_t<
                !detail::is_valid_unary_op<detail::operator_functor<op, has_nulls>,
                                           possibly_null_value_t<Input, has_nulls>>>* = nullptr>
    __device__ inline void operator()(ExpressionResult& output_object,
                                      cudf::size_type const output_row_index,
                                      possibly_null_value_t<Input, has_nulls> const& input,
                                      detail::device_data_reference const& output,
                                      char* thread_intermediate_storage) const
    {
      CUDF_UNREACHABLE("Invalid unary dispatch operator for the provided input.");
    }
  };

  /**
   * @brief Subclass of the expression output handler for binary operations.
   *
   * This functor's call operator is specialized to handle binary operations,
   * which require two operands.
   */
  template <typename LHS, typename RHS>
  struct binary_expression_output_handler : public expression_output_handler {
    __device__ inline binary_expression_output_handler() {}

    /**
     * @brief Callable to perform a binary operation.
     *
     * @tparam op The operation to perform.
     * @tparam OutputType The container type that data will be inserted into.
     *
     * @param output_object The container that data will be inserted into.
     * @param output_row_index The row in the output to insert the result.
     * @param lhs Left input to the operation.
     * @param rhs Right input to the operation.
     * @param output Output data reference.
     */
    template <ast_operator op,
              typename ResultSubclass,
              typename T,
              std::enable_if_t<detail::is_valid_binary_op<detail::operator_functor<op, has_nulls>,
                                                          possibly_null_value_t<LHS, has_nulls>,
                                                          possibly_null_value_t<RHS, has_nulls>>>* =
                nullptr>
    __device__ inline void operator()(expression_result<ResultSubclass, T>& output_object,
                                      cudf::size_type const output_row_index,
                                      possibly_null_value_t<LHS, has_nulls> const& lhs,
                                      possibly_null_value_t<RHS, has_nulls> const& rhs,
                                      detail::device_data_reference const& output,
                                      char* thread_intermediate_storage) const
    {
      // The output data type is the same whether or not nulls are present, so
      // pull from the non-nullable operator.
      using Out = cuda::std::invoke_result_t<detail::operator_functor<op, false>, LHS, RHS>;
      this->template resolve_output<Out>(output_object,
                                         output,
                                         output_row_index,
                                         thread_intermediate_storage,
                                         detail::operator_functor<op, has_nulls>{}(lhs, rhs));
    }

    template <ast_operator op,
              typename ResultSubclass,
              typename T,
              std::enable_if_t<
                !detail::is_valid_binary_op<detail::operator_functor<op, has_nulls>,
                                            possibly_null_value_t<LHS, has_nulls>,
                                            possibly_null_value_t<RHS, has_nulls>>>* = nullptr>
    __device__ inline void operator()(expression_result<ResultSubclass, T>& output_object,
                                      cudf::size_type const output_row_index,
                                      possibly_null_value_t<LHS, has_nulls> const& lhs,
                                      possibly_null_value_t<RHS, has_nulls> const& rhs,
                                      detail::device_data_reference const& output,
                                      char* thread_intermediate_storage) const
    {
      CUDF_UNREACHABLE("Invalid binary dispatch operator for the provided input.");
    }
  };

  table_device_view const& left;   ///< The left table to operate on.
  table_device_view const& right;  ///< The right table to operate on.
  expression_device_view const&
    plan;  ///< The container of device data representing the expression to evaluate.
  bool has_nulls;
};

}  // namespace detail

}  // namespace ast

}  // namespace cudf
