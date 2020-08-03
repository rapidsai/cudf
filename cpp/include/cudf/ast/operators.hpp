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

#include <cmath>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <simt/type_traits>
#include <type_traits>
#include <utility>
#include <vector>

namespace cudf {

namespace ast {

enum class ast_operator {
  // Binary operators
  ADD,            ///< operator +
  SUB,            ///< operator -
  MUL,            ///< operator *
  DIV,            ///< operator / using common type of lhs and rhs
  TRUE_DIV,       ///< operator / after promoting type to floating point
  FLOOR_DIV,      ///< operator / after promoting to 64 bit floating point and then
                  ///< flooring the result
  MOD,            ///< operator %
  PYMOD,          ///< operator % but following python's sign rules for negatives
  POW,            ///< lhs ^ rhs
  EQUAL,          ///< operator ==
  NOT_EQUAL,      ///< operator !=
  LESS,           ///< operator <
  GREATER,        ///< operator >
  LESS_EQUAL,     ///< operator <=
  GREATER_EQUAL,  ///< operator >=
  BITWISE_AND,    ///< operator &
  BITWISE_OR,     ///< operator |
  BITWISE_XOR,    ///< operator ^
  LOGICAL_AND,    ///< operator &&
  LOGICAL_OR,     ///< operator ||
  COALESCE,       ///< operator x,y  x is null ? y : x
  // GENERIC_BINARY,        ///< generic binary operator to be generated with input
  //                       ///< ptx code
  SHIFT_LEFT,            ///< operator <<
  SHIFT_RIGHT,           ///< operator >>
  SHIFT_RIGHT_UNSIGNED,  ///< operator >>> (from Java)
                         ///< Logical right shift. Casts to an unsigned value before shifting.
  LOG_BASE,              ///< logarithm to the base
  ATAN2,                 ///< 2-argument arctangent
  PMOD,                  ///< positive modulo operator
                         ///< If remainder is negative, this returns (remainder + divisor) % divisor
                         ///< else, it returns (dividend % divisor)
  NULL_EQUALS,           ///< Returns true when both operands are null; false when one is null; the
                         ///< result of equality when both are non-null
  NULL_MAX,              ///< Returns max of operands when both are non-null; returns the non-null
                         ///< operand when one is null; or invalid when both are null
  NULL_MIN,              ///< Returns min of operands when both are non-null; returns the non-null
                         ///< operand when one is null; or invalid when both are null
  // Unary operators
  SIN,         ///< Trigonometric sine
  COS,         ///< Trigonometric cosine
  TAN,         ///< Trigonometric tangent
  ARCSIN,      ///< Trigonometric sine inverse
  ARCCOS,      ///< Trigonometric cosine inverse
  ARCTAN,      ///< Trigonometric tangent inverse
  SINH,        ///< Hyperbolic sine
  COSH,        ///< Hyperbolic cosine
  TANH,        ///< Hyperbolic tangent
  ARCSINH,     ///< Hyperbolic sine inverse
  ARCCOSH,     ///< Hyperbolic cosine inverse
  ARCTANH,     ///< Hyperbolic tangent inverse
  EXP,         ///< Exponential (base e, Euler number)
  LOG,         ///< Natural Logarithm (base e)
  SQRT,        ///< Square-root (x^0.5)
  CBRT,        ///< Cube-root (x^(1.0/3))
  CEIL,        ///< Smallest integer value not less than arg
  FLOOR,       ///< largest integer value not greater than arg
  ABS,         ///< Absolute value
  RINT,        ///< Rounds the floating-point argument arg to an integer value
  BIT_INVERT,  ///< Bitwise Not (~)
  NOT,         ///< Logical Not (!)
  // Other operators included in BlazingSQL (TODO: may or may not implement)
  IS_NULL,            ///< Unary comparator returning whether the value is null
  COMPONENT_YEAR,     ///< Get year from a timestamp
  COMPONENT_MONTH,    ///< Get month from a timestamp
  COMPONENT_DAY,      ///< Get day from a timestamp
  COMPONENT_WEEKDAY,  ///< Get weekday from a timestamp
  COMPONENT_HOUR,     ///< Get hour from a timestamp
  COMPONENT_MINUTE,   ///< Get minute from a timestamp
  COMPONENT_SECOND,   ///< Get second from a timestamp
  ROUND,              ///< Round a value to a desired precision
  IS_NOT_NULL,        ///< Unary comparator returning whether the value is not null
  COTAN,              ///< Trigonometric cotangent
  CAST,               ///< Type cast operator (TODO: special case)
  CHAR_LENGTH,        ///< String length
  RAND,               ///< Random number (nullary operator)
  NOW,                ///< Current timestamp
  ROW,                ///< Current row of the table
  THREAD_ID,          ///< Could be useful for debugging
  BLOCK_ID            ///< Could be useful for debugging
};

// Traits for valid operator / type combinations
template <typename Op, typename LHS, typename RHS>
constexpr bool is_valid_binary_op = simt::std::is_invocable<Op, LHS, RHS>::value;

template <typename Op, typename T>
constexpr bool is_valid_unary_op = simt::std::is_invocable<Op, T>::value;

// Operator dispatcher
template <typename F, typename... Ts>
CUDA_HOST_DEVICE_CALLABLE constexpr void ast_operator_dispatcher_data_type(ast_operator op,
                                                                           F&& f,
                                                                           Ts&&... args)
{
  switch (op) {
    case ast_operator::ADD:
      f.template operator()<ast_operator::ADD>(std::forward<Ts>(args)...);
      break;
    case ast_operator::SUB:
      f.template operator()<ast_operator::SUB>(std::forward<Ts>(args)...);
      break;
    case ast_operator::MUL:
      f.template operator()<ast_operator::MUL>(std::forward<Ts>(args)...);
      break;
    case ast_operator::DIV:
      f.template operator()<ast_operator::DIV>(std::forward<Ts>(args)...);
      break;
    default:
      // TODO: Error handling?
      break;
  }
}

template <ast_operator op>
struct operator_functor {
};

template <>
struct operator_functor<ast_operator::ADD> {
  template <typename LHS, typename RHS>
  CUDA_HOST_DEVICE_CALLABLE auto operator()(LHS lhs, RHS rhs) -> decltype(lhs + rhs)
  {
    return lhs + rhs;
  }
};

template <>
struct operator_functor<ast_operator::SUB> {
  template <typename LHS, typename RHS>
  CUDA_HOST_DEVICE_CALLABLE auto operator()(LHS lhs, RHS rhs) -> decltype(lhs - rhs)
  {
    return lhs - rhs;
  }
};

/*
template <>
struct operator_functor<ast_operator::MUL> {
  template <typename LHS, typename RHS>
  CUDA_HOST_DEVICE_CALLABLE auto operator()(LHS lhs, RHS rhs) -> decltype(lhs * rhs)
  {
    return lhs * rhs;
  }
};

template <>
struct operator_functor<ast_operator::DIV> {
  template <typename LHS, typename RHS>
  CUDA_HOST_DEVICE_CALLABLE auto operator()(LHS lhs, RHS rhs) -> decltype(lhs / rhs)
  {
    return lhs / rhs;
  }
};

template <>
struct operator_functor<ast_operator::EQUAL> {
  template <typename LHS, typename RHS>
  CUDA_HOST_DEVICE_CALLABLE auto operator()(LHS lhs, RHS rhs) -> decltype(lhs == rhs)
  {
    return lhs == rhs;
  }
};
*/

namespace detail {

template <typename OperatorFunctor>
struct dispatch_operator_functor_types {
  template <typename LHS,
            typename RHS,
            typename F,
            typename... Ts,
            std::enable_if_t<is_valid_binary_op<OperatorFunctor, LHS, RHS>>* = nullptr>
  CUDA_HOST_DEVICE_CALLABLE void operator()(F&& f, Ts&&... args)
  {
    f.template operator()<OperatorFunctor, LHS, RHS>(std::forward<Ts>(args)...);
  }

  template <typename LHS,
            typename RHS,
            typename F,
            typename... Ts,
            std::enable_if_t<!is_valid_binary_op<OperatorFunctor, LHS, RHS>>* = nullptr>
  CUDA_HOST_DEVICE_CALLABLE void operator()(F&& f, Ts&&... args)
  {
#ifndef __CUDA_ARCH__
    CUDF_FAIL("Invalid binary operation.");
#else
    release_assert(false && "Invalid binary operation.");
#endif
  }
};

template <typename OperatorFunctor>
struct dispatch_operator_functor_types_single {
  template <typename LHS,
            typename F,
            typename... Ts,
            std::enable_if_t<is_valid_binary_op<OperatorFunctor, LHS, LHS>>* = nullptr>
  CUDA_HOST_DEVICE_CALLABLE void operator()(F&& f, Ts&&... args)
  {
    f.template operator()<OperatorFunctor, LHS, LHS>(std::forward<Ts>(args)...);
  }

  template <typename LHS,
            typename F,
            typename... Ts,
            std::enable_if_t<!is_valid_binary_op<OperatorFunctor, LHS, LHS>>* = nullptr>
  CUDA_HOST_DEVICE_CALLABLE void operator()(F&& f, Ts&&... args)
  {
#ifndef __CUDA_ARCH__
    CUDF_FAIL("Invalid binary operation.");
#else
    release_assert(false && "Invalid binary operation.");
#endif
  }
};

struct dispatch_op {
  template <ast_operator op, typename F, typename... Ts>
  CUDA_HOST_DEVICE_CALLABLE void operator()(cudf::data_type lhs_t,
                                            cudf::data_type rhs_t,
                                            F&& f,
                                            Ts&&... args)
  {
    // Double dispatch
    double_type_dispatcher(lhs_t,
                           rhs_t,
                           detail::dispatch_operator_functor_types<operator_functor<op>>{},
                           std::forward<F>(f),
                           std::forward<Ts>(args)...);
    // Single dispatch (assume lhs_t == rhs_t)
    /*
    type_dispatcher(lhs_t,
                    detail::dispatch_operator_functor_types_single<operator_functor<op>>{},
                    std::forward<F>(f),
                    std::forward<Ts>(args)...);
    */
  }
};

struct binary_return_type_functor {
  template <typename OperatorFunctor,
            typename LHS,
            typename RHS,
            typename Out = simt::std::invoke_result_t<OperatorFunctor, LHS, RHS>,
            std::enable_if_t<cudf::ast::is_valid_binary_op<OperatorFunctor, LHS, RHS>>* = nullptr>
  CUDA_HOST_DEVICE_CALLABLE void operator()(cudf::data_type* result)
  {
    *result = cudf::data_type(cudf::type_to_id<Out>());
  }

  template <typename OperatorFunctor,
            typename LHS,
            typename RHS,
            typename Out                                                                 = void,
            std::enable_if_t<!cudf::ast::is_valid_binary_op<OperatorFunctor, LHS, RHS>>* = nullptr>
  CUDA_HOST_DEVICE_CALLABLE void operator()(cudf::data_type* result)
  {
    *result = cudf::data_type(cudf::type_id::EMPTY);
  }
};

struct unary_return_type_functor {
  template <typename OperatorFunctor,
            typename T,
            typename Out = simt::std::invoke_result_t<OperatorFunctor, T>,
            std::enable_if_t<cudf::ast::is_valid_unary_op<OperatorFunctor, T>>* = nullptr>
  CUDA_HOST_DEVICE_CALLABLE void operator()(cudf::data_type* result)
  {
    *result = cudf::data_type(cudf::type_to_id<Out>());
  }

  template <typename OperatorFunctor,
            typename T,
            typename Out                                                         = void,
            std::enable_if_t<!cudf::ast::is_valid_unary_op<OperatorFunctor, T>>* = nullptr>
  CUDA_HOST_DEVICE_CALLABLE void operator()(cudf::data_type* result)
  {
    *result = cudf::data_type(cudf::type_id::EMPTY);
  }
};

}  // namespace detail

/**
 * @brief Dispatches two type template parameters to a callable.
 *
 */
#pragma nv_exec_check_disable
template <typename F, typename... Ts>
CUDA_HOST_DEVICE_CALLABLE constexpr void ast_operator_dispatcher(
  ast_operator op, cudf::data_type lhs_t, cudf::data_type rhs_t, F&& f, Ts&&... args)
{
  ast_operator_dispatcher_data_type(
    op, detail::dispatch_op{}, lhs_t, rhs_t, std::forward<F>(f), std::forward<Ts>(args)...);
}

/**
 * @brief Gets the return type of an AST operator.
 *
 */
inline cudf::data_type ast_operator_return_type(ast_operator op,
                                                std::vector<cudf::data_type> const& operand_types)
{
  auto result = cudf::data_type(cudf::type_id::EMPTY);
  switch (operand_types.size()) {
    case 0:
      // TODO: Nullary return type functor
      break;
    case 1:
      // TODO: Unary return type functor
      break;
    case 2:
      ast_operator_dispatcher(op,
                              operand_types.at(0),
                              operand_types.at(1),
                              detail::binary_return_type_functor{},
                              &result);
      break;
    case 3:
      // TODO: Ternary return type functor
      break;
    default: break;
  }
  return result;
}

}  // namespace ast

}  // namespace cudf
