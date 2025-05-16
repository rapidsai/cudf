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
#include <cudf/ast/detail/operators.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda/std/type_traits>

#include <vector>

namespace cudf {
namespace ast {
namespace detail {
namespace {

struct arity_functor {
  template <ast_operator op>
  void operator()(cudf::size_type& result)
  {
    // Arity is not dependent on null handling, so just use the false implementation here.
    result = operator_functor<op, false>::arity;
  }
};

/**
 * @brief Functor to determine the return type of an operator from its input types.
 */
struct return_type_functor {
  /**
   * @brief Callable for binary operators to determine return type.
   *
   * @tparam OperatorFunctor Operator functor to perform.
   * @tparam LHS Left input type.
   * @tparam RHS Right input type.
   * @param result Pointer whose value is assigned to the result data type.
   */
  template <typename OperatorFunctor,
            typename LHS,
            typename RHS,
            std::enable_if_t<is_valid_binary_op<OperatorFunctor, LHS, RHS>>* = nullptr>
  void operator()(cudf::data_type& result)
  {
    using Out = cuda::std::invoke_result_t<OperatorFunctor, LHS, RHS>;
    result    = cudf::data_type{cudf::type_to_id<Out>()};
  }

  template <typename OperatorFunctor,
            typename LHS,
            typename RHS,
            std::enable_if_t<!is_valid_binary_op<OperatorFunctor, LHS, RHS>>* = nullptr>
  void operator()(cudf::data_type& result)
  {
#ifndef __CUDA_ARCH__
    CUDF_FAIL("Invalid binary operation. Return type cannot be determined.");
#else
    CUDF_UNREACHABLE("Invalid binary operation. Return type cannot be determined.");
#endif
    result = cudf::data_type{cudf::type_id::EMPTY};
  }

  /**
   * @brief Callable for unary operators to determine return type.
   *
   * @tparam OperatorFunctor Operator functor to perform.
   * @tparam T Input type.
   * @param result Pointer whose value is assigned to the result data type.
   */
  template <typename OperatorFunctor,
            typename T,
            std::enable_if_t<is_valid_unary_op<OperatorFunctor, T>>* = nullptr>
  void operator()(cudf::data_type& result)
  {
    using Out = cuda::std::invoke_result_t<OperatorFunctor, T>;
    result    = cudf::data_type{cudf::type_to_id<Out>()};
  }

  template <typename OperatorFunctor,
            typename T,
            std::enable_if_t<!is_valid_unary_op<OperatorFunctor, T>>* = nullptr>
  void operator()(cudf::data_type& result)
  {
#ifndef __CUDA_ARCH__
    CUDF_FAIL("Invalid unary operation. Return type cannot be determined.");
#else
    CUDF_UNREACHABLE("Invalid unary operation. Return type cannot be determined.");
#endif
    result = cudf::data_type{cudf::type_id::EMPTY};
  }
};

/**
 * @brief Functor used to single-type-dispatch binary operators.
 *
 * This functor's `operator()` is templated to validate calls to its operators based on the input
 * type, as determined by the `is_valid_binary_op` trait. This function assumes that both inputs are
 * the same type, and dispatches based on the type of the left input.
 *
 * @tparam OperatorFunctor Binary operator functor.
 */
template <typename OperatorFunctor>
struct single_dispatch_binary_operator_types {
  template <typename LHS,
            typename F,
            typename... Ts,
            std::enable_if_t<is_valid_binary_op<OperatorFunctor, LHS, LHS>>* = nullptr>
  inline void operator()(F&& f, Ts&&... args)
  {
    f.template operator()<OperatorFunctor, LHS, LHS>(std::forward<Ts>(args)...);
  }

  template <typename LHS,
            typename F,
            typename... Ts,
            std::enable_if_t<!is_valid_binary_op<OperatorFunctor, LHS, LHS>>* = nullptr>
  inline void operator()(F&& f, Ts&&... args)
  {
#ifndef __CUDA_ARCH__
    CUDF_FAIL("Invalid binary operation.");
#else
    CUDF_UNREACHABLE("Invalid binary operation.");
#endif
  }
};

/**
 * @brief Functor performing a type dispatch for a binary operator.
 *
 * This functor performs single dispatch, which assumes lhs_type == rhs_type. This may not be true
 * for all binary operators but holds for all currently implemented operators.
 */
struct type_dispatch_binary_op {
  /**
   * @brief Performs type dispatch for a binary operator.
   *
   * @tparam op AST operator.
   * @tparam F Type of forwarded functor.
   * @tparam Ts Parameter pack of forwarded arguments.
   * @param lhs_type Type of left input data.
   * @param rhs_type Type of right input data.
   * @param f Forwarded functor to be called.
   * @param args Forwarded arguments to `operator()` of `f`.
   */
  template <ast_operator op, typename F, typename... Ts>
  inline void operator()(cudf::data_type lhs_type, cudf::data_type rhs_type, F&& f, Ts&&... args)
  {
    // Single dispatch (assume lhs_type == rhs_type)
    type_dispatcher(
      lhs_type,
      // Always dispatch to the non-null operator for the purpose of type determination.
      detail::single_dispatch_binary_operator_types<operator_functor<op, false>>{},
      std::forward<F>(f),
      std::forward<Ts>(args)...);
  }
};

/**
 * @brief Dispatches a runtime binary operator to a templated type dispatcher.
 *
 * @tparam F Type of forwarded functor.
 * @tparam Ts Parameter pack of forwarded arguments.
 * @param lhs_type Type of left input data.
 * @param rhs_type Type of right input data.
 * @param f Forwarded functor to be called.
 * @param args Forwarded arguments to `operator()` of `f`.
 */
template <typename F, typename... Ts>
inline constexpr void binary_operator_dispatcher(
  ast_operator op, cudf::data_type lhs_type, cudf::data_type rhs_type, F&& f, Ts&&... args)
{
  ast_operator_dispatcher(op,
                          detail::type_dispatch_binary_op{},
                          lhs_type,
                          rhs_type,
                          std::forward<F>(f),
                          std::forward<Ts>(args)...);
}

/**
 * @brief Functor used to type-dispatch unary operators.
 *
 * This functor's `operator()` is templated to validate calls to its operators based on the input
 * type, as determined by the `is_valid_unary_op` trait.
 *
 * @tparam OperatorFunctor Unary operator functor.
 */
template <typename OperatorFunctor>
struct dispatch_unary_operator_types {
  template <typename InputT,
            typename F,
            typename... Ts,
            std::enable_if_t<is_valid_unary_op<OperatorFunctor, InputT>>* = nullptr>
  inline void operator()(F&& f, Ts&&... args)
  {
    f.template operator()<OperatorFunctor, InputT>(std::forward<Ts>(args)...);
  }

  template <typename InputT,
            typename F,
            typename... Ts,
            std::enable_if_t<!is_valid_unary_op<OperatorFunctor, InputT>>* = nullptr>
  inline void operator()(F&& f, Ts&&... args)
  {
#ifndef __CUDA_ARCH__
    CUDF_FAIL("Invalid unary operation.");
#else
    CUDF_UNREACHABLE("Invalid unary operation.");
#endif
  }
};

/**
 * @brief Functor performing a type dispatch for a unary operator.
 */
struct type_dispatch_unary_op {
  template <ast_operator op, typename F, typename... Ts>
  inline void operator()(cudf::data_type input_type, F&& f, Ts&&... args)
  {
    type_dispatcher(
      input_type,
      // Always dispatch to the non-null operator for the purpose of type determination.
      detail::dispatch_unary_operator_types<operator_functor<op, false>>{},
      std::forward<F>(f),
      std::forward<Ts>(args)...);
  }
};

/**
 * @brief Dispatches a runtime unary operator to a templated type dispatcher.
 *
 * @tparam F Type of forwarded functor.
 * @tparam Ts Parameter pack of forwarded arguments.
 * @param input_type Type of input data.
 * @param f Forwarded functor to be called.
 * @param args Forwarded arguments to `operator()` of `f`.
 */
template <typename F, typename... Ts>
inline constexpr void unary_operator_dispatcher(ast_operator op,
                                                cudf::data_type input_type,
                                                F&& f,
                                                Ts&&... args)
{
  ast_operator_dispatcher(op,
                          detail::type_dispatch_unary_op{},
                          input_type,
                          std::forward<F>(f),
                          std::forward<Ts>(args)...);
}

}  // namespace

cudf::data_type ast_operator_return_type(ast_operator op,
                                         std::vector<cudf::data_type> const& operand_types)
{
  cudf::data_type result{cudf::type_id::EMPTY};
  switch (operand_types.size()) {
    case 1:
      unary_operator_dispatcher(op, operand_types[0], detail::return_type_functor{}, result);
      break;
    case 2:
      binary_operator_dispatcher(
        op, operand_types[0], operand_types[1], detail::return_type_functor{}, result);
      break;
    default: CUDF_FAIL("Unsupported operator return type."); break;
  }
  return result;
}

cudf::size_type ast_operator_arity(ast_operator op)
{
  cudf::size_type result{};
  ast_operator_dispatcher(op, arity_functor{}, result);
  return result;
}

}  // namespace detail

}  // namespace ast

}  // namespace cudf
