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

#include <cstring>
#include <numeric>

namespace cudf {

namespace ast {

namespace detail {

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

// Type used for intermediate storage in expression evaluation.
template <bool has_nulls>
using IntermediateDataType = possibly_null_value_t<std::int64_t, has_nulls>;

/**
 * @brief A container for capturing the output of an evaluated expression.
 *
 * This class is designed to be passed by reference as the first argument to
 * expression_evaluator::evaluate. The primary implementation is as an owning
 * container of a (possibly nullable) scalar type that can be written to by the
 * expression_evaluator. The data (and its validity) can then be accessed. The
 * API is designed such that template specializations for specific output types
 * will be able to customize setting behavior if necessary.
 *
 * @tparam has_nulls Whether or not the result data is nullable.
 * @tparam T The underlying data type.
 */
template <bool has_nulls, typename T>
struct expression_result {
  __device__ expression_result() {}

  // TODO: The index is ignored by this function, but is included because it is
  // required by the implementation in the template specialization for column
  // views, see below.
  template <typename Element>
  __device__ void set_value(cudf::size_type index, possibly_null_value_t<Element, has_nulls> result)
  {
    if constexpr (std::is_same_v<Element, T>) {
      _obj = result;
    } else {
      cudf_assert(false && "Output type does not match container type.");
    }
  }

  /**
   * @brief Returns true if the underlying data is valid and false otherwise.
   */
  __device__ bool is_valid() const
  {
    if constexpr (has_nulls) { return _obj.has_value(); }
    return true;
  }

  /**
   * @brief Returns the underlying data.
   *
   * @throws thrust::bad_optional_access if the underlying data is not valid.
   */
  __device__ T value() const
  {
    // Using two separate constexprs silences compiler warnings, whereas an
    // if/else does not. An unconditional return is not ignored by the compiler
    // when has_nulls is true and therefore raises a compiler error.
    if constexpr (has_nulls) { return _obj.value(); }
    if constexpr (!has_nulls) { return _obj; }
  }

  possibly_null_value_t<T, has_nulls>
    _obj;  ///< The underlying data value, or a nullable version of it.
};

// TODO: The below implementation significantly differs from the default
// implementation above due to the non-owning nature of the container and the
// usage of the index. It would be ideal to unify these further if possible.

/**
 * @brief A container for capturing the output of an evaluated expression in a column.
 *
 * This template specialization differs from the primary implementation in that
 * it is non-owning, instead writing output directly to a column view.
 */
template <bool has_nulls>
struct expression_result<has_nulls, mutable_column_device_view> {
  __device__ expression_result(mutable_column_device_view& obj) : _obj(obj) {}

  template <typename Element>
  __device__ void set_value(cudf::size_type index, possibly_null_value_t<Element, has_nulls> result)
  {
    if constexpr (has_nulls) {
      if (result.has_value()) {
        _obj.template element<Element>(index) = *result;
        _obj.set_valid(index);
      } else {
        _obj.set_null(index);
      }
    } else {
      _obj.template element<Element>(index) = result;
    }
  }

  mutable_column_device_view& _obj;  ///< The column to which the data is written.
};

/**
 * @brief A container of all device data required to evaluate an expression on tables.
 *
 * This struct should never be instantiated directly. It is created by the
 * `ast_plan` on construction, and the resulting member is publicly accessible
 * for passing to kernels for constructing `expression_evaluators`.
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

/**
 * @brief Preprocessor for an expression acting on tables to generate data suitable for AST
 * expression evaluation on the GPU.
 *
 * On construction, an AST plan creates a single "packed" host buffer of all
 * data arrays that will be necessary to evaluate an expression on a pair of
 * tables. This data is copied to a single contiguous device buffer, and
 * pointers are generated to the individual components. Because the plan tends
 * to be small, this is the most efficient approach for low latency. All the
 * data required on the GPU can be accessed via the convenient `dev_plan`
 * member struct, which can be used to construct an `expression_evaluator` on
 * the device.
 *
 * Note that the resulting device data cannot be used once this class goes out of scope.
 */
struct ast_plan {
  /**
   * @brief Construct an AST plan for an expression operating on two tables.
   *
   * @param expr The expression for which to construct a plan.
   * @param left The left table on which the expression acts.
   * @param right The right table on which the expression acts.
   * @param has_nulls Boolean indicator of whether or not the data contains nulls.
   * @param stream Stream view on which to allocate resources and queue execution.
   * @param mr Device memory resource used to allocate the returned column's device.
   */
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
    dev_plan.shmem_per_thread  = static_cast<int>(
      (has_nulls ? sizeof(IntermediateDataType<true>) : sizeof(IntermediateDataType<false>)) *
      dev_plan.num_intermediates);
  }

  /**
   * @brief Construct an AST plan for an expression operating on one table.
   *
   * @param expr The expression for which to construct a plan.
   * @param left The left table on which the expression acts.
   * @param has_nulls Boolean indicator of whether or not the data contains nulls.
   * @param stream Stream view on which to allocate resources and queue execution.
   * @param mr Device memory resource used to allocate the returned column's device.
   */
  ast_plan(detail::node const& expr,
           cudf::table_view left,
           bool has_nulls,
           rmm::cuda_stream_view stream,
           rmm::mr::device_memory_resource* mr)
    : ast_plan(expr, left, left, has_nulls, stream, mr)
  {
  }

  cudf::data_type output_type() const { return _linearizer.root_data_type(); }

  dev_ast_plan
    dev_plan;  ///< The collection of data required to evaluate the expression on the device.

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

  rmm::device_buffer
    _device_data_buffer;  ///< The device-side data buffer containing the plan information, which is
                          ///< owned by this class and persists until it is destroyed.
  linearizer const _linearizer;  ///< The linearizer created from the provided expression that is
                                 ///< used to construct device-side operators and references.
};

/**
 * @brief The principal object for evaluating AST expressions on device.
 *
 * This class is designed for n-ary transform evaluation. It operates on two
 * tables.
 */
template <bool has_nulls>
struct expression_evaluator {
 public:
  /**
   * @brief Construct an expression evaluator acting on two tables.
   *
   * @param left View on the left table view used for evaluation.
   * @param plan The collection of device references representing the expression to evaluate.
   * @param thread_intermediate_storage Pointer to this thread's portion of shared memory for
   * storing intermediates.
   * @param left View on the right table view used for evaluation.
   */
  __device__ expression_evaluator(table_device_view const& left,
                                  table_device_view const& right,
                                  dev_ast_plan const& plan,
                                  IntermediateDataType<has_nulls>* thread_intermediate_storage,
                                  null_equality compare_nulls = null_equality::EQUAL)
    : left(left),
      plan(plan),
      thread_intermediate_storage(thread_intermediate_storage),
      right(right),
      compare_nulls(compare_nulls)
  {
  }

  /**
   * @brief Construct an expression evaluator acting on one table.
   *
   * @param left View on the left table view used for evaluation.
   * @param plan The collection of device references representing the expression to evaluate.
   * @param thread_intermediate_storage Pointer to this thread's portion of shared memory for
   * storing intermediates.
   */
  __device__ expression_evaluator(table_device_view const& left,
                                  dev_ast_plan const& plan,
                                  IntermediateDataType<has_nulls>* thread_intermediate_storage,
                                  null_equality compare_nulls = null_equality::EQUAL)
    : left(left),
      right(left),
      plan(plan),
      thread_intermediate_storage(thread_intermediate_storage),
      compare_nulls(compare_nulls)
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
      IntermediateDataType<has_nulls> intermediate = thread_intermediate_storage[data_index];
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
  template <typename Input, typename OutputType>
  __device__ void operator()(OutputType& output_object,
                             const cudf::size_type input_row_index,
                             const detail::device_data_reference input,
                             const detail::device_data_reference output,
                             const cudf::size_type output_row_index,
                             const ast_operator op) const
  {
    auto const typed_input = resolve_input<Input>(input, input_row_index);
    ast_operator_dispatcher(op,
                            unary_expression_output_handler<Input>(*this),
                            output_object,
                            output_row_index,
                            typed_input,
                            output);
  }

  /**
   * @brief Callable to perform a unary operation.
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
  template <typename LHS, typename RHS, typename OutputType>
  __device__ void operator()(OutputType& output_object,
                             const cudf::size_type left_row_index,
                             const cudf::size_type right_row_index,
                             const detail::device_data_reference lhs,
                             const detail::device_data_reference rhs,
                             const detail::device_data_reference output,
                             const cudf::size_type output_row_index,
                             const ast_operator op) const
  {
    auto const typed_lhs = resolve_input<LHS>(lhs, left_row_index);
    auto const typed_rhs = resolve_input<RHS>(rhs, right_row_index);
    ast_operator_dispatcher(op,
                            binary_expression_output_handler<LHS, RHS>(*this),
                            output_object,
                            output_row_index,
                            typed_lhs,
                            typed_rhs,
                            output);
  }

  template <typename OperatorFunctor,
            typename LHS,
            typename RHS,
            typename OutputType,
            std::enable_if_t<!detail::is_valid_binary_op<OperatorFunctor, LHS, RHS>>* = nullptr>
  __device__ void operator()(OutputType& output_object,
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
   * @tparam OutputType The container type that data will be inserted into.
   *
   * @param output_object The container that data will be inserted into.
   * @param row_index Row index of all input and output data column(s).
   */
  template <typename OutputType>
  __device__ void evaluate(OutputType& output_object, cudf::size_type const row_index)
  {
    evaluate(output_object, row_index, row_index, row_index);
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
  template <typename OutputType>
  __device__ void evaluate(OutputType& output_object,
                           cudf::size_type const left_row_index,
                           cudf::size_type const right_row_index,
                           cudf::size_type const output_row_index)
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
    __device__ expression_output_handler(expression_evaluator<has_nulls> const& evaluator)
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
     * @tparam OutputType The container type that data will be inserted into.
     *
     * @param output_object The container that data will be inserted into.
     * @param device_data_reference Data reference to resolve.
     * @param row_index Row index of data column.
     * @param result Value to assign to output.
     */
    template <typename Element,
              typename OutputType,
              CUDF_ENABLE_IF(is_rep_layout_compatible<Element>())>
    __device__ void resolve_output(OutputType& output_object,
                                   const detail::device_data_reference device_data_reference,
                                   const cudf::size_type row_index,
                                   const possibly_null_value_t<Element, has_nulls> result) const
    {
      auto const ref_type = device_data_reference.reference_type;
      if (ref_type == detail::device_data_reference_type::COLUMN) {
        output_object.template set_value<Element>(row_index, result);
      } else {  // Assumes ref_type == detail::device_data_reference_type::INTERMEDIATE
        // Using memcpy instead of reinterpret_cast<Element*> for safe type aliasing.
        // Using a temporary variable ensures that the compiler knows the result is aligned.
        IntermediateDataType<has_nulls> tmp;
        memcpy(&tmp, &result, sizeof(possibly_null_value_t<Element, has_nulls>));
        evaluator.thread_intermediate_storage[device_data_reference.data_index] = tmp;
      }
    }

    template <typename Element,
              typename OutputType,
              CUDF_ENABLE_IF(not is_rep_layout_compatible<Element>())>
    __device__ void resolve_output(OutputType& output_object,
                                   const detail::device_data_reference device_data_reference,
                                   const cudf::size_type row_index,
                                   const possibly_null_value_t<Element, has_nulls> result) const
    {
      cudf_assert(false && "Invalid type in resolve_output.");
    }

   protected:
    expression_evaluator<has_nulls> const& evaluator;
  };

  /**
   * @brief Subclass of the expression output handler for unary operations.
   *
   * This functor's call operator is specialized to handle unary operations,
   * which only require a single operand.
   */
  template <typename Input>
  struct unary_expression_output_handler : public expression_output_handler {
    __device__ unary_expression_output_handler(expression_evaluator<has_nulls> const& evaluator)
      : expression_output_handler(evaluator)
    {
    }

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
    template <
      ast_operator op,
      typename OutputType,
      std::enable_if_t<detail::is_valid_unary_op<detail::operator_functor<op>, Input>>* = nullptr>
    __device__ void operator()(OutputType& output_object,
                               const cudf::size_type output_row_index,
                               const possibly_null_value_t<Input, has_nulls> input,
                               const detail::device_data_reference output) const
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
      typename OutputType,
      std::enable_if_t<!detail::is_valid_unary_op<detail::operator_functor<op>, Input>>* = nullptr>
    __device__ void operator()(OutputType& output_object,
                               const cudf::size_type output_row_index,
                               const possibly_null_value_t<Input, has_nulls> input,
                               const detail::device_data_reference output) const
    {
      cudf_assert(false && "Invalid unary dispatch operator for the provided input.");
    }
  };

  /**
   * @brief Subclass of the expression output handler for binary operations.
   *
   * This functor's call operator is specialized to handle binary operations,
   * which require a two operands.
   */
  template <typename LHS, typename RHS>
  struct binary_expression_output_handler : public expression_output_handler {
    __device__ binary_expression_output_handler(expression_evaluator<has_nulls> const& evaluator)
      : expression_output_handler(evaluator)
    {
    }

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
              typename OutputType,
              std::enable_if_t<
                detail::is_valid_binary_op<detail::operator_functor<op>, LHS, RHS>>* = nullptr>
    __device__ void operator()(OutputType& output_object,
                               const cudf::size_type output_row_index,
                               const possibly_null_value_t<LHS, has_nulls> lhs,
                               const possibly_null_value_t<RHS, has_nulls> rhs,
                               const detail::device_data_reference output) const
    {
      using OperatorFunctor = detail::operator_functor<op>;
      using Out             = cuda::std::invoke_result_t<OperatorFunctor, LHS, RHS>;
      if constexpr (has_nulls) {
        if constexpr (op == ast_operator::EQUAL) {
          // Special handling of the equality operator based on what kind
          // of null handling was requested.
          possibly_null_value_t<Out, has_nulls> result;
          if (!lhs.has_value() && !rhs.has_value()) {
            // Case 1: Both null, so the output is based on compare_nulls.
            result = possibly_null_value_t<Out, has_nulls>(this->evaluator.compare_nulls ==
                                                           null_equality::EQUAL);
          } else if (lhs.has_value() && rhs.has_value()) {
            // Case 2: Neither is null, so the output is given by the operation.
            result = possibly_null_value_t<Out, has_nulls>(OperatorFunctor{}(*lhs, *rhs));
          } else {
            // Case 3: One value is null, while the other is not, so we simply propagate nulls.
            result = possibly_null_value_t<Out, has_nulls>();
          }
          this->template resolve_output<Out>(output_object, output, output_row_index, result);
        } else {
          // Default behavior for all other operators is to propagate nulls.
          auto result = (lhs.has_value() && rhs.has_value())
                          ? possibly_null_value_t<Out, has_nulls>(OperatorFunctor{}(*lhs, *rhs))
                          : possibly_null_value_t<Out, has_nulls>();
          this->template resolve_output<Out>(output_object, output, output_row_index, result);
        }  // if constexpr (op == ast_operator::EQUAL) {
      } else {
        this->template resolve_output<Out>(
          output_object, output, output_row_index, OperatorFunctor{}(lhs, rhs));
      }  // if constexpr (has_nulls) {
    }

    template <ast_operator op,
              typename OutputType,
              std::enable_if_t<
                !detail::is_valid_binary_op<detail::operator_functor<op>, LHS, RHS>>* = nullptr>
    __device__ void operator()(OutputType& output_object,
                               const cudf::size_type output_row_index,
                               const possibly_null_value_t<LHS, has_nulls> lhs,
                               const possibly_null_value_t<RHS, has_nulls> rhs,
                               const detail::device_data_reference output) const
    {
      cudf_assert(false && "Invalid binary dispatch operator for the provided input.");
    }
  };

  table_device_view const& left;   ///< The left table to operate on.
  table_device_view const& right;  ///< The right table to operate on.
  dev_ast_plan const&
    plan;  ///< The container of device data representing the expression to evaluate.
  IntermediateDataType<has_nulls>*
    thread_intermediate_storage;  ///< The shared memory store of intermediates produced during
                                  ///< evaluation.
  null_equality
    compare_nulls;  ///< Whether the equality operators returns true or false for two nulls.
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
