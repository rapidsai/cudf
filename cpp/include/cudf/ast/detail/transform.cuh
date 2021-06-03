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

#include <thrust/optional.h>
#include <thrust/pair.h>

#include <cstring>
#include <numeric>

namespace cudf {

namespace ast {

namespace detail {

/**
 * @brief The AST plan creates a device buffer of data needed to execute an AST.
 *
 * On construction, an AST plan creates a single "packed" host buffer of all necessary data arrays,
 * and copies that to the device with a single host-device memory copy. Because the plan tends to be
 * small, this is the most efficient approach for low latency.
 *
 */
struct dev_ast_plan {
  device_span<const detail::device_data_reference> data_references;
  device_span<const cudf::detail::fixed_width_scalar_device_view_base> literals;
  device_span<const ast_operator> operators;
  device_span<const cudf::size_type> operator_source_indices;
  cudf::size_type num_intermediates;
  int shmem_per_thread;
};

struct ast_plan {
  ast_plan(detail::node const& expr,
           cudf::table_view left,
           cudf::table_view right,
           bool has_nulls,
           rmm::cuda_stream_view stream,
           rmm::mr::device_memory_resource* mr)
    : _linearizer(expr, left, right)
  {
    std::vector<cudf::size_type> _sizes;
    std::vector<const void*> _data_pointers;

    extract_size_and_pointer(_linearizer.data_references(), _sizes, _data_pointers);
    extract_size_and_pointer(_linearizer.literals(), _sizes, _data_pointers);
    extract_size_and_pointer(_linearizer.operators(), _sizes, _data_pointers);
    extract_size_and_pointer(_linearizer.operator_source_indices(), _sizes, _data_pointers);

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
    dev_plan.data_references    = device_span<const detail::device_data_reference>(
      reinterpret_cast<const detail::device_data_reference*>(device_data_buffer_ptr +
                                                             buffer_offsets[0]),
      _linearizer.data_references().size());
    dev_plan.literals = device_span<const cudf::detail::fixed_width_scalar_device_view_base>(
      reinterpret_cast<const cudf::detail::fixed_width_scalar_device_view_base*>(
        device_data_buffer_ptr + buffer_offsets[1]),
      _linearizer.literals().size());
    dev_plan.operators = device_span<const ast_operator>(
      reinterpret_cast<const ast_operator*>(device_data_buffer_ptr + buffer_offsets[2]),
      _linearizer.operators().size());
    dev_plan.operator_source_indices = device_span<const cudf::size_type>(
      reinterpret_cast<const cudf::size_type*>(device_data_buffer_ptr + buffer_offsets[3]),
      _linearizer.operator_source_indices().size());
    dev_plan.num_intermediates = _linearizer.intermediate_count();

    // We cannot pull the required type directly from the evaluator, which
    // would be ideal, because that would introduce a circular dependency
    // between them. The separation between host and device code in this case
    // requires this minor duplication of logic.
    dev_plan.shmem_per_thread =
      static_cast<int>((has_nulls ? sizeof(thrust::optional<std::int64_t>) : sizeof(std::int64_t)) *
                       dev_plan.num_intermediates);
  }

  ast_plan(detail::node const& expr,
           cudf::table_view left,
           bool has_nulls,
           rmm::cuda_stream_view stream,
           rmm::mr::device_memory_resource* mr)
    : ast_plan(expr, left, left, has_nulls, stream, mr)
  {
  }

  cudf::data_type output_type() const { return _linearizer.root_data_type(); }

  dev_ast_plan dev_plan;

 private:
  /**
   * @brief Helper function for adding components (operators, literals, etc) to AST plan
   *
   * @tparam T  The underlying type of the input `std::vector`
   * @param  v  The `std::vector` containing components (operators, literals, etc)
   */
  template <typename T>
  void extract_size_and_pointer(std::vector<T> const& v,
                                std::vector<cudf::size_type>& _sizes,
                                std::vector<const void*>& _data_pointers)
  {
    auto const data_size = sizeof(T) * v.size();
    _sizes.push_back(data_size);
    _data_pointers.push_back(v.data());
  }

  rmm::device_buffer _device_data_buffer;
  linearizer const _linearizer;
};

// Type trait for wrapping nullable types in a thrust::optional. Non-nullable
// types are returned as is.
template <typename T, bool has_nulls>
struct possibly_null_value;

template <typename T>
struct possibly_null_value<T, true> {
  using type = thrust::optional<T>;
};

template <typename T>
struct possibly_null_value<T, false> {
  using type = T;
};

template <typename T, bool has_nulls>
using possibly_null_value_t = typename possibly_null_value<T, has_nulls>::type;

/**
 * @brief An expression evaluator owned by a single thread operating on rows of two table.
 *
 * This class is designed for n-ary transform evaluation. It operates on two
 * separate tables, and knows the rows of each one.
 */
template <typename OutputType, bool has_nulls>
struct expression_evaluator {
 private:
  using IntermediateDataType = possibly_null_value_t<std::int64_t, has_nulls>;

 public:
  /**
   * @brief Construct a row evaluator.
   *
   * @param table The table device view used for evaluation.
   * @param literals Array of literal values used for evaluation.
   * @param thread_intermediate_storage Pointer to this thread's portion of shared memory for
   * storing intermediates.
   */
  __device__ expression_evaluator(table_device_view const& left,
                                  dev_ast_plan const& plan,
                                  IntermediateDataType* thread_intermediate_storage,
                                  table_device_view const& right)
    : left(left), plan(plan), thread_intermediate_storage(thread_intermediate_storage), right(right)
  {
  }

  // Overloaded constructor for single-table case.
  __device__ expression_evaluator(table_device_view const& left,
                                  dev_ast_plan const& plan,
                                  IntermediateDataType* thread_intermediate_storage)
    : left(left), plan(plan), thread_intermediate_storage(thread_intermediate_storage), right(left)
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
  __device__ possibly_null_value_t<Element, has_nulls> resolve_input(
    detail::device_data_reference device_data_reference, cudf::size_type row_index) const
  {
    auto const data_index = device_data_reference.data_index;
    auto const ref_type   = device_data_reference.reference_type;
    // TODO: Everywhere in the code assumes that the tbale reference is either
    // left or right. Should we error-check somewhere to prevent
    // table_reference::OUTPUT from being specified?
    auto const& table = device_data_reference.table_source == table_reference::LEFT ? left : right;
    using ReturnType  = possibly_null_value_t<Element, has_nulls>;
    if (ref_type == detail::device_data_reference_type::COLUMN) {
      // If we have nullable data, return an empty nullable type with no value if the data is null.
      if constexpr (has_nulls) {
        return table.column(data_index).is_valid(row_index)
                 ? ReturnType(table.column(data_index).element<Element>(row_index))
                 : ReturnType();

      } else {
        return ReturnType(table.column(data_index).element<Element>(row_index));
      }
    } else if (ref_type == detail::device_data_reference_type::LITERAL) {
      return ReturnType(plan.literals[data_index].value<Element>());
    } else {  // Assumes ref_type == detail::device_data_reference_type::INTERMEDIATE
      // Using memcpy instead of reinterpret_cast<Element*> for safe type aliasing
      // Using a temporary variable ensures that the compiler knows the result is aligned
      IntermediateDataType intermediate = thread_intermediate_storage[data_index];
      ReturnType tmp;
      memcpy(&tmp, &intermediate, sizeof(ReturnType));
      return tmp;
    }
    // Unreachable return used to silence compiler warnings.
    return {};
  }

  template <typename Element,
            CUDF_ENABLE_IF(not column_device_view::has_element_accessor<Element>())>
  __device__ possibly_null_value_t<Element, has_nulls> resolve_input(
    detail::device_data_reference device_data_reference, cudf::size_type row_index) const
  {
    cudf_assert(false && "Unsupported type in resolve_input.");
    // Unreachable return used to silence compiler warnings.
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
  __device__ void operator()(OutputType output_object,
                             cudf::size_type input_row_index,
                             const detail::device_data_reference input,
                             const detail::device_data_reference output,
                             cudf::size_type output_row_index,
                             const ast_operator op) const
  {
    auto const typed_input = resolve_input<Input>(input, input_row_index);
    ast_operator_dispatcher(op,
                            unary_expression_output<Input>(*this),
                            output_object,
                            output_row_index,
                            typed_input,
                            output);
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
  __device__ void operator()(OutputType output_object,
                             cudf::size_type left_row_index,
                             cudf::size_type right_row_index,
                             const detail::device_data_reference lhs,
                             const detail::device_data_reference rhs,
                             const detail::device_data_reference output,
                             cudf::size_type output_row_index,
                             const ast_operator op) const
  {
    auto const typed_lhs = resolve_input<LHS>(lhs, left_row_index);
    auto const typed_rhs = resolve_input<RHS>(rhs, right_row_index);
    ast_operator_dispatcher(op,
                            binary_expression_output<LHS, RHS>(*this),
                            output_object,
                            output_row_index,
                            typed_lhs,
                            typed_rhs,
                            output);
  }

  template <typename OperatorFunctor,
            typename LHS,
            typename RHS,
            std::enable_if_t<!detail::is_valid_binary_op<OperatorFunctor, LHS, RHS>>* = nullptr>
  __device__ void operator()(OutputType output_object,
                             cudf::size_type left_row_index,
                             cudf::size_type right_row_index,
                             const detail::device_data_reference lhs,
                             const detail::device_data_reference rhs,
                             const detail::device_data_reference output,
                             cudf::size_type output_row_index,
                             const ast_operator op) const
  {
    cudf_assert(false && "Invalid binary dispatch operator for the provided input.");
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
  __device__ void evaluate(OutputType output_object, cudf::size_type row_index)
  {
    evaluate(output_object, row_index, row_index, row_index);
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
  __device__ void evaluate(OutputType output_object,
                           cudf::size_type left_row_index,
                           cudf::size_type right_row_index,
                           cudf::size_type output_row_index)
  {
    auto operator_source_index = static_cast<cudf::size_type>(0);
    for (cudf::size_type operator_index = 0; operator_index < plan.operators.size();
         operator_index++) {
      // Execute operator
      auto const op    = plan.operators[operator_index];
      auto const arity = ast_operator_arity(op);
      if (arity == 1) {
        // Unary operator
        auto const input =
          plan.data_references[plan.operator_source_indices[operator_source_index]];
        auto const output =
          plan.data_references[plan.operator_source_indices[operator_source_index + 1]];
        operator_source_index += arity + 1;
        auto input_row_index =
          input.table_source == table_reference::LEFT ? left_row_index : right_row_index;
        type_dispatcher(input.data_type,
                        *this,
                        output_object,
                        input_row_index,
                        input,
                        output,
                        output_row_index,
                        op);
      } else if (arity == 2) {
        // Binary operator
        auto const lhs = plan.data_references[plan.operator_source_indices[operator_source_index]];
        auto const rhs =
          plan.data_references[plan.operator_source_indices[operator_source_index + 1]];
        auto const output =
          plan.data_references[plan.operator_source_indices[operator_source_index + 2]];
        operator_source_index += arity + 1;
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
                        op);
      } else {
        cudf_assert(false && "Invalid operator arity.");
      }
    }
  }

 private:
  struct expression_output {
   public:
    __device__ expression_output(expression_evaluator<OutputType, has_nulls> const& evaluator)
      : evaluator(evaluator)
    {
    }

    /**
     * @brief Resolves an output data reference and assigns result value.
     *
     * Only output columns (COLUMN) and intermediates (INTERMEDIATE) are supported as output
     * reference types. Intermediates must be of fixed width less than or equal to
     * sizeof(std::int64_t). This requirement on intermediates is enforced by the linearizer.
     *
     * @tparam Element Type of result element.
     * @param device_data_reference Data reference to resolve.
     * @param expression_index Row index of data column.
     * @param result Value to assign to output.
     */
    template <typename Element,
              CUDF_ENABLE_IF(is_rep_layout_compatible<Element>() &&
                             std::is_same<OutputType, mutable_column_device_view*>::value)>
    __device__ void resolve_output(OutputType output_object,
                                   detail::device_data_reference device_data_reference,
                                   cudf::size_type row_index,
                                   possibly_null_value_t<Element, has_nulls> result) const
    {
      auto const ref_type = device_data_reference.reference_type;
      if (ref_type == detail::device_data_reference_type::COLUMN) {
        if constexpr (has_nulls) {
          // TODO: Handle null equality propertly, right now I'm just assuming
          // that null translates to null.
          if (result.has_value()) {
            output_object->template element<Element>(row_index) = *result;
            output_object->set_valid(row_index);
          } else {
            output_object->set_null(row_index);
          }
        } else {
          output_object->template element<Element>(row_index) = result;
        }
      } else {  // Assumes ref_type == detail::device_data_reference_type::INTERMEDIATE
        // Using memcpy instead of reinterpret_cast<Element*> for safe type aliasing.
        // Using a temporary variable ensures that the compiler knows the result is aligned.
        IntermediateDataType tmp;
        memcpy(&tmp, &result, sizeof(possibly_null_value_t<Element, has_nulls>));
        evaluator.thread_intermediate_storage[device_data_reference.data_index] = tmp;
      }
    }

    // TODO: The row index is now completely ignored... not so great.
    // Also now I have a separate override for thrust::optional that's
    // exclusively being used when has_nulls is true, but the override is not
    // specific to that class. Need a better SFINAE, and a better overall design
    // for return values. It would be better if the thrust::optional version only
    // existed when has_nulls=true, but really it would be ideal to not have to
    // go through a lot o fthese hoops in general. Also the use of void * is bad
    // because it makes all the overloads compile but they'll run erroneously.
    // Currently this isn't an issue only because the conditional join code
    // exclusively uses the bool path at runtime.
    template <typename Element,
              CUDF_ENABLE_IF(is_rep_layout_compatible<Element>() &&
                             std::is_same<OutputType, thrust::pair<void*, bool>*>::value)>
    __device__ void resolve_output(OutputType output_object,
                                   detail::device_data_reference device_data_reference,
                                   cudf::size_type row_index,
                                   possibly_null_value_t<Element, has_nulls> result) const
    {
      auto const ref_type = device_data_reference.reference_type;
      // TODO: Rather than creating a separate device data reference for
      // outputting to an arbitrary pointer I templated the
      // expression_evaluator. I think this makes more sense because I think
      // that conceptually the parsing of the expression tree should be
      // independent of the nature of the output, but this should be discussed
      // before finalizing.
      if (ref_type == detail::device_data_reference_type::COLUMN) {
        // TODO: This line assumes that the output optional works whether the
        // input result is either a raw bool (when has_nulls is false) or
        // another optional (when has_nulls is true), which is true but kind of
        // messy here.
        if constexpr (has_nulls) {
          // Super kludgy approach to setting the validity to true for the optional.
          if (result.has_value()) {
            static_cast<Element*>(output_object->first)[row_index] = *result;
            output_object->second                                  = true;
          } else {
            output_object->second = false;
          }
        } else {
          static_cast<Element*>(output_object->first)[row_index] = result;
        }
      } else {  // Assumes ref_type == detail::device_data_reference_type::INTERMEDIATE
        // Using memcpy instead of reinterpret_cast<Element*> for safe type aliasing.
        // Using a temporary variable ensures that the compiler knows the result is aligned.
        IntermediateDataType tmp;
        memcpy(&tmp, &result, sizeof(possibly_null_value_t<Element, has_nulls>));
        evaluator.thread_intermediate_storage[device_data_reference.data_index] = tmp;
      }
    }

    template <typename Element, CUDF_ENABLE_IF(not is_rep_layout_compatible<Element>())>
    __device__ void resolve_output(OutputType output_object,
                                   detail::device_data_reference device_data_reference,
                                   cudf::size_type row_index,
                                   possibly_null_value_t<Element, has_nulls> result) const
    {
      cudf_assert(false && "Invalid type in resolve_output.");
    }

   private:
    expression_evaluator<OutputType, has_nulls> const& evaluator;
  };

  template <typename Input>
  struct unary_expression_output : public expression_output {
    __device__ unary_expression_output(expression_evaluator<OutputType, has_nulls> const& evaluator)
      : expression_output(evaluator)
    {
    }

    template <
      ast_operator op,
      std::enable_if_t<detail::is_valid_unary_op<detail::operator_functor<op>, Input>>* = nullptr>
    __device__ void operator()(OutputType output_object,
                               cudf::size_type output_row_index,
                               possibly_null_value_t<Input, has_nulls> input,
                               detail::device_data_reference output) const
    {
      using OperatorFunctor = detail::operator_functor<op>;
      using Out             = cuda::std::invoke_result_t<OperatorFunctor, Input>;
      if constexpr (has_nulls) {
        auto result = input.has_value()
                        ? possibly_null_value_t<Out, has_nulls>(OperatorFunctor{}(*input))
                        : possibly_null_value_t<Out, has_nulls>();
        this->template resolve_output<Out>(output_object, output, output_row_index, result);
      } else {
        this->template resolve_output<Out>(
          output_object, output, output_row_index, OperatorFunctor{}(input));
      }
    }

    template <
      ast_operator op,
      std::enable_if_t<!detail::is_valid_unary_op<detail::operator_functor<op>, Input>>* = nullptr>
    __device__ void operator()(OutputType output_object,
                               cudf::size_type output_row_index,
                               possibly_null_value_t<Input, has_nulls> input,
                               detail::device_data_reference output) const
    {
      cudf_assert(false && "Invalid unary dispatch operator for the provided input.");
    }
  };

  template <typename LHS, typename RHS>
  struct binary_expression_output : public expression_output {
    __device__ binary_expression_output(
      expression_evaluator<OutputType, has_nulls> const& evaluator)
      : expression_output(evaluator)
    {
    }

    template <ast_operator op,
              std::enable_if_t<
                detail::is_valid_binary_op<detail::operator_functor<op>, LHS, RHS>>* = nullptr>
    __device__ void operator()(OutputType output_object,
                               cudf::size_type output_row_index,
                               possibly_null_value_t<LHS, has_nulls> lhs,
                               possibly_null_value_t<RHS, has_nulls> rhs,
                               detail::device_data_reference output) const
    {
      using OperatorFunctor = detail::operator_functor<op>;
      using Out             = cuda::std::invoke_result_t<OperatorFunctor, LHS, RHS>;
      if constexpr (has_nulls) {
        auto result = (lhs.has_value() && rhs.has_value())
                        ? possibly_null_value_t<Out, has_nulls>(OperatorFunctor{}(*lhs, *rhs))
                        : possibly_null_value_t<Out, has_nulls>();
        this->template resolve_output<Out>(output_object, output, output_row_index, result);
      } else {
        this->template resolve_output<Out>(
          output_object, output, output_row_index, OperatorFunctor{}(lhs, rhs));
      }
    }

    template <ast_operator op,
              std::enable_if_t<
                !detail::is_valid_binary_op<detail::operator_functor<op>, LHS, RHS>>* = nullptr>
    __device__ void operator()(OutputType output_object,
                               cudf::size_type output_row_index,
                               possibly_null_value_t<LHS, has_nulls> lhs,
                               possibly_null_value_t<RHS, has_nulls> rhs,
                               detail::device_data_reference output) const
    {
      cudf_assert(false && "Invalid binary dispatch operator for the provided input.");
    }
  };

  table_device_view const& left;
  table_device_view const& right;
  dev_ast_plan const& plan;
  IntermediateDataType* thread_intermediate_storage;
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
