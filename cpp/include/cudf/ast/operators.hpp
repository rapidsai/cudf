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
#include <type_traits>
#include <utility>

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

/*
 * Default all operator traits to false.
 */
template <ast_operator op, typename = void>
struct is_binary_arithmetic_operator_trait_impl : std::false_type {
};

template <ast_operator op, typename = void>
struct is_unary_arithmetic_operator_trait_impl : std::false_type {
};

template <ast_operator op, typename = void>
struct is_binary_comparator_trait_impl : std::false_type {
};

template <ast_operator op, typename = void>
struct is_unary_comparator_trait_impl : std::false_type {
};

template <ast_operator op, typename = void>
struct is_binary_logical_operator_trait_impl : std::false_type {
};

template <ast_operator op, typename = void>
struct is_unary_logical_operator_trait_impl : std::false_type {
};

/*
 * Define templated operator traits.
 */
template <ast_operator op>
CUDA_HOST_DEVICE_CALLABLE constexpr bool is_binary_arithmetic_operator()
{
  return is_binary_arithmetic_operator_trait_impl<op>::value;
}

template <ast_operator op>
CUDA_HOST_DEVICE_CALLABLE constexpr bool is_unary_arithmetic_operator()
{
  return is_unary_arithmetic_operator_trait_impl<op>::value;
}

template <ast_operator op>
CUDA_HOST_DEVICE_CALLABLE constexpr bool is_binary_comparator()
{
  return is_binary_comparator_trait_impl<op>::value;
}

template <ast_operator op>
CUDA_HOST_DEVICE_CALLABLE constexpr bool is_unary_comparator()
{
  return is_unary_comparator_trait_impl<op>::value;
}

template <ast_operator op>
CUDA_HOST_DEVICE_CALLABLE constexpr bool is_binary_logical_operator()
{
  return is_binary_logical_operator_trait_impl<op>::value;
}

template <ast_operator op>
CUDA_HOST_DEVICE_CALLABLE constexpr bool is_unary_logical_operator()
{
  return is_unary_logical_operator_trait_impl<op>::value;
}

// Math operators accept element type(s) and return an element
template <ast_operator op>
CUDA_HOST_DEVICE_CALLABLE constexpr bool is_arithmetic_operator()
{
  return is_binary_arithmetic_operator<op>() || is_unary_arithmetic_operator<op>();
}

// Comparators accept element type(s) and return a boolean
template <ast_operator op>
CUDA_HOST_DEVICE_CALLABLE constexpr bool is_comparator()
{
  return is_binary_comparator<op>() || is_unary_comparator<op>();
}

// Logical accept boolean(s) and return a boolean
template <ast_operator op>
CUDA_HOST_DEVICE_CALLABLE constexpr bool is_logical_operator()
{
  return is_binary_logical_operator<op>() || is_unary_logical_operator<op>();
}

// Binary operators accept two inputs
template <ast_operator op>
CUDA_HOST_DEVICE_CALLABLE constexpr bool is_binary_operator()
{
  return is_binary_arithmetic_operator<op>() || is_binary_comparator<op>() ||
         is_binary_logical_operator<op>();
}

// Unary operators accept one input
template <ast_operator op>
CUDA_HOST_DEVICE_CALLABLE constexpr bool is_unary_operator()
{
  return is_unary_arithmetic_operator<op>() || is_unary_comparator<op>() ||
         is_unary_logical_operator<op>();
}

/*
 * Define traits for each operator.
 */
template <>
struct is_binary_arithmetic_operator_trait_impl<ast_operator::ADD> : std::true_type {
};

template <>
struct is_binary_arithmetic_operator_trait_impl<ast_operator::SUB> : std::true_type {
};

template <>
struct is_binary_arithmetic_operator_trait_impl<ast_operator::MUL> : std::true_type {
};

template <>
struct is_binary_arithmetic_operator_trait_impl<ast_operator::DIV> : std::true_type {
};

template <>
struct is_binary_arithmetic_operator_trait_impl<ast_operator::TRUE_DIV> : std::true_type {
};

template <>
struct is_binary_arithmetic_operator_trait_impl<ast_operator::FLOOR_DIV> : std::true_type {
};

template <>
struct is_binary_arithmetic_operator_trait_impl<ast_operator::MOD> : std::true_type {
};

template <>
struct is_binary_arithmetic_operator_trait_impl<ast_operator::PYMOD> : std::true_type {
};

template <>
struct is_binary_arithmetic_operator_trait_impl<ast_operator::POW> : std::true_type {
};

template <>
struct is_binary_comparator_trait_impl<ast_operator::EQUAL> : std::true_type {
};

template <>
struct is_binary_comparator_trait_impl<ast_operator::NOT_EQUAL> : std::true_type {
};

template <>
struct is_binary_comparator_trait_impl<ast_operator::LESS> : std::true_type {
};

template <>
struct is_binary_comparator_trait_impl<ast_operator::GREATER> : std::true_type {
};

template <>
struct is_binary_comparator_trait_impl<ast_operator::LESS_EQUAL> : std::true_type {
};

template <>
struct is_binary_comparator_trait_impl<ast_operator::GREATER_EQUAL> : std::true_type {
};

template <>
struct is_binary_arithmetic_operator_trait_impl<ast_operator::BITWISE_AND> : std::true_type {
};

template <>
struct is_binary_arithmetic_operator_trait_impl<ast_operator::BITWISE_OR> : std::true_type {
};

template <>
struct is_binary_arithmetic_operator_trait_impl<ast_operator::BITWISE_XOR> : std::true_type {
};

template <>
struct is_binary_logical_operator_trait_impl<ast_operator::LOGICAL_AND> : std::true_type {
};

template <>
struct is_binary_logical_operator_trait_impl<ast_operator::LOGICAL_OR> : std::true_type {
};

template <>
struct is_binary_arithmetic_operator_trait_impl<ast_operator::COALESCE> : std::true_type {
};

template <>
struct is_binary_arithmetic_operator_trait_impl<ast_operator::SHIFT_LEFT> : std::true_type {
};

template <>
struct is_binary_arithmetic_operator_trait_impl<ast_operator::SHIFT_RIGHT> : std::true_type {
};

template <>
struct is_binary_arithmetic_operator_trait_impl<ast_operator::SHIFT_RIGHT_UNSIGNED>
  : std::true_type {
};

template <>
struct is_binary_arithmetic_operator_trait_impl<ast_operator::LOG_BASE> : std::true_type {
};

template <>
struct is_binary_arithmetic_operator_trait_impl<ast_operator::ATAN2> : std::true_type {
};

template <>
struct is_binary_arithmetic_operator_trait_impl<ast_operator::PMOD> : std::true_type {
};

template <>
struct is_binary_comparator_trait_impl<ast_operator::NULL_EQUALS> : std::true_type {
};

template <>
struct is_binary_arithmetic_operator_trait_impl<ast_operator::NULL_MAX> : std::true_type {
};

template <>
struct is_binary_arithmetic_operator_trait_impl<ast_operator::NULL_MIN> : std::true_type {
};

template <>
struct is_unary_arithmetic_operator_trait_impl<ast_operator::SIN> : std::true_type {
};

template <>
struct is_unary_arithmetic_operator_trait_impl<ast_operator::COS> : std::true_type {
};

template <>
struct is_unary_arithmetic_operator_trait_impl<ast_operator::TAN> : std::true_type {
};

template <>
struct is_unary_arithmetic_operator_trait_impl<ast_operator::ARCSIN> : std::true_type {
};

template <>
struct is_unary_arithmetic_operator_trait_impl<ast_operator::ARCCOS> : std::true_type {
};

template <>
struct is_unary_arithmetic_operator_trait_impl<ast_operator::ARCTAN> : std::true_type {
};

template <>
struct is_unary_arithmetic_operator_trait_impl<ast_operator::SINH> : std::true_type {
};

template <>
struct is_unary_arithmetic_operator_trait_impl<ast_operator::COSH> : std::true_type {
};

template <>
struct is_unary_arithmetic_operator_trait_impl<ast_operator::TANH> : std::true_type {
};

template <>
struct is_unary_arithmetic_operator_trait_impl<ast_operator::ARCSINH> : std::true_type {
};

template <>
struct is_unary_arithmetic_operator_trait_impl<ast_operator::ARCCOSH> : std::true_type {
};

template <>
struct is_unary_arithmetic_operator_trait_impl<ast_operator::ARCTANH> : std::true_type {
};

template <>
struct is_unary_arithmetic_operator_trait_impl<ast_operator::EXP> : std::true_type {
};

template <>
struct is_unary_arithmetic_operator_trait_impl<ast_operator::LOG> : std::true_type {
};

template <>
struct is_unary_arithmetic_operator_trait_impl<ast_operator::SQRT> : std::true_type {
};

template <>
struct is_unary_arithmetic_operator_trait_impl<ast_operator::CBRT> : std::true_type {
};

template <>
struct is_unary_arithmetic_operator_trait_impl<ast_operator::CEIL> : std::true_type {
};

template <>
struct is_unary_arithmetic_operator_trait_impl<ast_operator::FLOOR> : std::true_type {
};

template <>
struct is_unary_arithmetic_operator_trait_impl<ast_operator::ABS> : std::true_type {
};

template <>
struct is_unary_arithmetic_operator_trait_impl<ast_operator::RINT> : std::true_type {
};

template <>
struct is_unary_arithmetic_operator_trait_impl<ast_operator::BIT_INVERT> : std::true_type {
};

template <>
struct is_unary_logical_operator_trait_impl<ast_operator::NOT> : std::true_type {
};

// Traits for valid operator / type combinations
template <typename L, typename R, typename Op>
using is_valid_binary_op = decltype(std::declval<Op>()(std::declval<L>(), std::declval<R>()));

template <typename T, typename Op>
using is_valid_unary_op = decltype(std::declval<Op>()(std::declval<T>()));

// Operator dispatcher used for checking traits of operators and returning logical-valued operations
template <typename Functor, typename... Ts>
CUDA_HOST_DEVICE_CALLABLE constexpr decltype(auto) ast_operator_dispatcher_bool(ast_operator op,
                                                                                Functor&& f,
                                                                                Ts&&... args)
{
  switch (op) {
    case ast_operator::ADD:
      return f.template operator()<ast_operator::ADD>(std::forward<Ts>(args)...);
    case ast_operator::SUB:
      return f.template operator()<ast_operator::SUB>(std::forward<Ts>(args)...);
    case ast_operator::MUL:
      return f.template operator()<ast_operator::MUL>(std::forward<Ts>(args)...);
    case ast_operator::DIV:
      return f.template operator()<ast_operator::DIV>(std::forward<Ts>(args)...);
    /*
    case ast_operator::TRUE_DIV:
      return f.template operator()<ast_operator::TRUE_DIV>(std::forward<Ts>(args)...);
    case ast_operator::FLOOR_DIV:
      return f.template operator()<ast_operator::FLOOR_DIV>(std::forward<Ts>(args)...);
    case ast_operator::MOD:
      return f.template operator()<ast_operator::MOD>(std::forward<Ts>(args)...);
    case ast_operator::PYMOD:
      return f.template operator()<ast_operator::PYMOD>(std::forward<Ts>(args)...);
    case ast_operator::POW:
      return f.template operator()<ast_operator::POW>(std::forward<Ts>(args)...);
    case ast_operator::EQUAL:
      return f.template operator()<ast_operator::EQUAL>(std::forward<Ts>(args)...);
    case ast_operator::NOT_EQUAL:
      return f.template operator()<ast_operator::NOT_EQUAL>(std::forward<Ts>(args)...);
    case ast_operator::LESS:
      return f.template operator()<ast_operator::LESS>(std::forward<Ts>(args)...);
    case ast_operator::GREATER:
      return f.template operator()<ast_operator::GREATER>(std::forward<Ts>(args)...);
    case ast_operator::LESS_EQUAL:
      return f.template operator()<ast_operator::LESS_EQUAL>(std::forward<Ts>(args)...);
    case ast_operator::GREATER_EQUAL:
      return f.template operator()<ast_operator::GREATER_EQUAL>(std::forward<Ts>(args)...);
    case ast_operator::BITWISE_AND:
      return f.template operator()<ast_operator::BITWISE_AND>(std::forward<Ts>(args)...);
    case ast_operator::BITWISE_OR:
      return f.template operator()<ast_operator::BITWISE_OR>(std::forward<Ts>(args)...);
    case ast_operator::BITWISE_XOR:
      return f.template operator()<ast_operator::BITWISE_XOR>(std::forward<Ts>(args)...);
    case ast_operator::LOGICAL_AND:
      return f.template operator()<ast_operator::LOGICAL_AND>(std::forward<Ts>(args)...);
    case ast_operator::LOGICAL_OR:
      return f.template operator()<ast_operator::LOGICAL_OR>(std::forward<Ts>(args)...);
    case ast_operator::COALESCE:
      return f.template operator()<ast_operator::COALESCE>(std::forward<Ts>(args)...);
    case ast_operator::SHIFT_LEFT:
      return f.template operator()<ast_operator::SHIFT_LEFT>(std::forward<Ts>(args)...);
    case ast_operator::SHIFT_RIGHT:
      return f.template operator()<ast_operator::SHIFT_RIGHT>(std::forward<Ts>(args)...);
    case ast_operator::SHIFT_RIGHT_UNSIGNED:
      return f.template operator()<ast_operator::SHIFT_RIGHT_UNSIGNED>(std::forward<Ts>(args)...);
    case ast_operator::LOG_BASE:
      return f.template operator()<ast_operator::LOG_BASE>(std::forward<Ts>(args)...);
    case ast_operator::ATAN2:
      return f.template operator()<ast_operator::ATAN2>(std::forward<Ts>(args)...);
    case ast_operator::PMOD:
      return f.template operator()<ast_operator::PMOD>(std::forward<Ts>(args)...);
    case ast_operator::NULL_EQUALS:
      return f.template operator()<ast_operator::NULL_EQUALS>(std::forward<Ts>(args)...);
    case ast_operator::NULL_MAX:
      return f.template operator()<ast_operator::NULL_MAX>(std::forward<Ts>(args)...);
    case ast_operator::NULL_MIN:
      return f.template operator()<ast_operator::NULL_MIN>(std::forward<Ts>(args)...);
    case ast_operator::SIN:
      return f.template operator()<ast_operator::SIN>(std::forward<Ts>(args)...);
    case ast_operator::COS:
      return f.template operator()<ast_operator::COS>(std::forward<Ts>(args)...);
    case ast_operator::TAN:
      return f.template operator()<ast_operator::TAN>(std::forward<Ts>(args)...);
    case ast_operator::ARCSIN:
      return f.template operator()<ast_operator::ARCSIN>(std::forward<Ts>(args)...);
    case ast_operator::ARCCOS:
      return f.template operator()<ast_operator::ARCCOS>(std::forward<Ts>(args)...);
    case ast_operator::ARCTAN:
      return f.template operator()<ast_operator::ARCTAN>(std::forward<Ts>(args)...);
    case ast_operator::SINH:
      return f.template operator()<ast_operator::SINH>(std::forward<Ts>(args)...);
    case ast_operator::COSH:
      return f.template operator()<ast_operator::COSH>(std::forward<Ts>(args)...);
    case ast_operator::TANH:
      return f.template operator()<ast_operator::TANH>(std::forward<Ts>(args)...);
    case ast_operator::ARCSINH:
      return f.template operator()<ast_operator::ARCSINH>(std::forward<Ts>(args)...);
    case ast_operator::ARCCOSH:
      return f.template operator()<ast_operator::ARCCOSH>(std::forward<Ts>(args)...);
    case ast_operator::ARCTANH:
      return f.template operator()<ast_operator::ARCTANH>(std::forward<Ts>(args)...);
    case ast_operator::EXP:
      return f.template operator()<ast_operator::EXP>(std::forward<Ts>(args)...);
    case ast_operator::LOG:
      return f.template operator()<ast_operator::LOG>(std::forward<Ts>(args)...);
    case ast_operator::SQRT:
      return f.template operator()<ast_operator::SQRT>(std::forward<Ts>(args)...);
    case ast_operator::CBRT:
      return f.template operator()<ast_operator::CBRT>(std::forward<Ts>(args)...);
    case ast_operator::CEIL:
      return f.template operator()<ast_operator::CEIL>(std::forward<Ts>(args)...);
    case ast_operator::FLOOR:
      return f.template operator()<ast_operator::FLOOR>(std::forward<Ts>(args)...);
    case ast_operator::ABS:
      return f.template operator()<ast_operator::ABS>(std::forward<Ts>(args)...);
    case ast_operator::RINT:
      return f.template operator()<ast_operator::RINT>(std::forward<Ts>(args)...);
    case ast_operator::BIT_INVERT:
      return f.template operator()<ast_operator::BIT_INVERT>(std::forward<Ts>(args)...);
    case ast_operator::NOT:
      return f.template operator()<ast_operator::NOT>(std::forward<Ts>(args)...);
    */
    default: return false;  // TODO: Error handling?
  }
}

// Operator dispatcher used for returning numeric-valued operations
template <typename Functor, typename T0, typename... Ts>
CUDA_HOST_DEVICE_CALLABLE constexpr decltype(auto) ast_operator_dispatcher_typed(ast_operator op,
                                                                                 Functor&& f,
                                                                                 T0 arg0,
                                                                                 Ts&&... args)
{
  // We capture the first argument's type in T0 so we can construct a "default" value of the correct
  // (matching) type.
  switch (op) {
    case ast_operator::ADD:
      return f.template operator()<ast_operator::ADD>(std::forward<T0>(arg0),
                                                      std::forward<Ts>(args)...);
    case ast_operator::SUB:
      return f.template operator()<ast_operator::SUB>(std::forward<T0>(arg0),
                                                      std::forward<Ts>(args)...);
    case ast_operator::MUL:
      return f.template operator()<ast_operator::MUL>(std::forward<T0>(arg0),
                                                      std::forward<Ts>(args)...);
    case ast_operator::DIV:
      return f.template operator()<ast_operator::DIV>(std::forward<T0>(arg0),
                                                      std::forward<Ts>(args)...);
    /*
    case ast_operator::TRUE_DIV:
      return f.template operator()<ast_operator::TRUE_DIV>(std::forward<T0>(arg0),
                                                           std::forward<Ts>(args)...);
    case ast_operator::FLOOR_DIV:
      return f.template operator()<ast_operator::FLOOR_DIV>(std::forward<T0>(arg0),
                                                            std::forward<Ts>(args)...);
    case ast_operator::MOD:
      return f.template operator()<ast_operator::MOD>(std::forward<T0>(arg0),
                                                      std::forward<Ts>(args)...);
    case ast_operator::PYMOD:
      return f.template operator()<ast_operator::PYMOD>(std::forward<T0>(arg0),
                                                        std::forward<Ts>(args)...);
    case ast_operator::POW:
      return f.template operator()<ast_operator::POW>(std::forward<T0>(arg0),
                                                      std::forward<Ts>(args)...);
    case ast_operator::EQUAL:
      return f.template operator()<ast_operator::EQUAL>(std::forward<T0>(arg0),
                                                        std::forward<Ts>(args)...);
    case ast_operator::NOT_EQUAL:
      return f.template operator()<ast_operator::NOT_EQUAL>(std::forward<T0>(arg0),
                                                            std::forward<Ts>(args)...);
    case ast_operator::LESS:
      return f.template operator()<ast_operator::LESS>(std::forward<T0>(arg0),
                                                       std::forward<Ts>(args)...);
    case ast_operator::GREATER:
      return f.template operator()<ast_operator::GREATER>(std::forward<T0>(arg0),
                                                          std::forward<Ts>(args)...);
    case ast_operator::LESS_EQUAL:
      return f.template operator()<ast_operator::LESS_EQUAL>(std::forward<T0>(arg0),
                                                             std::forward<Ts>(args)...);
    case ast_operator::GREATER_EQUAL:
      return f.template operator()<ast_operator::GREATER_EQUAL>(std::forward<T0>(arg0),
                                                                std::forward<Ts>(args)...);
    case ast_operator::BITWISE_AND:
      return f.template operator()<ast_operator::BITWISE_AND>(std::forward<T0>(arg0),
                                                              std::forward<Ts>(args)...);
    case ast_operator::BITWISE_OR:
      return f.template operator()<ast_operator::BITWISE_OR>(std::forward<T0>(arg0),
                                                             std::forward<Ts>(args)...);
    case ast_operator::BITWISE_XOR:
      return f.template operator()<ast_operator::BITWISE_XOR>(std::forward<T0>(arg0),
                                                              std::forward<Ts>(args)...);
    case ast_operator::LOGICAL_AND:
      return f.template operator()<ast_operator::LOGICAL_AND>(std::forward<T0>(arg0),
                                                              std::forward<Ts>(args)...);
    case ast_operator::LOGICAL_OR:
      return f.template operator()<ast_operator::LOGICAL_OR>(std::forward<T0>(arg0),
                                                             std::forward<Ts>(args)...);
    case ast_operator::COALESCE:
      return f.template operator()<ast_operator::COALESCE>(std::forward<T0>(arg0),
                                                           std::forward<Ts>(args)...);
    case ast_operator::SHIFT_LEFT:
      return f.template operator()<ast_operator::SHIFT_LEFT>(std::forward<T0>(arg0),
                                                             std::forward<Ts>(args)...);
    case ast_operator::SHIFT_RIGHT:
      return f.template operator()<ast_operator::SHIFT_RIGHT>(std::forward<T0>(arg0),
                                                              std::forward<Ts>(args)...);
    case ast_operator::SHIFT_RIGHT_UNSIGNED:
      return f.template operator()<ast_operator::SHIFT_RIGHT_UNSIGNED>(std::forward<T0>(arg0),
                                                                       std::forward<Ts>(args)...);
    case ast_operator::LOG_BASE:
      return f.template operator()<ast_operator::LOG_BASE>(std::forward<T0>(arg0),
                                                           std::forward<Ts>(args)...);
    case ast_operator::ATAN2:
      return f.template operator()<ast_operator::ATAN2>(std::forward<T0>(arg0),
                                                        std::forward<Ts>(args)...);
    case ast_operator::PMOD:
      return f.template operator()<ast_operator::PMOD>(std::forward<T0>(arg0),
                                                       std::forward<Ts>(args)...);
    case ast_operator::NULL_EQUALS:
      return f.template operator()<ast_operator::NULL_EQUALS>(std::forward<T0>(arg0),
                                                              std::forward<Ts>(args)...);
    case ast_operator::NULL_MAX:
      return f.template operator()<ast_operator::NULL_MAX>(std::forward<T0>(arg0),
                                                           std::forward<Ts>(args)...);
    case ast_operator::NULL_MIN:
      return f.template operator()<ast_operator::NULL_MIN>(std::forward<T0>(arg0),
                                                           std::forward<Ts>(args)...);
    case ast_operator::SIN:
      return f.template operator()<ast_operator::SIN>(std::forward<T0>(arg0),
                                                      std::forward<Ts>(args)...);
    case ast_operator::COS:
      return f.template operator()<ast_operator::COS>(std::forward<T0>(arg0),
                                                      std::forward<Ts>(args)...);
    case ast_operator::TAN:
      return f.template operator()<ast_operator::TAN>(std::forward<T0>(arg0),
                                                      std::forward<Ts>(args)...);
    case ast_operator::ARCSIN:
      return f.template operator()<ast_operator::ARCSIN>(std::forward<T0>(arg0),
                                                         std::forward<Ts>(args)...);
    case ast_operator::ARCCOS:
      return f.template operator()<ast_operator::ARCCOS>(std::forward<T0>(arg0),
                                                         std::forward<Ts>(args)...);
    case ast_operator::ARCTAN:
      return f.template operator()<ast_operator::ARCTAN>(std::forward<T0>(arg0),
                                                         std::forward<Ts>(args)...);
    case ast_operator::SINH:
      return f.template operator()<ast_operator::SINH>(std::forward<T0>(arg0),
                                                       std::forward<Ts>(args)...);
    case ast_operator::COSH:
      return f.template operator()<ast_operator::COSH>(std::forward<T0>(arg0),
                                                       std::forward<Ts>(args)...);
    case ast_operator::TANH:
      return f.template operator()<ast_operator::TANH>(std::forward<T0>(arg0),
                                                       std::forward<Ts>(args)...);
    case ast_operator::ARCSINH:
      return f.template operator()<ast_operator::ARCSINH>(std::forward<T0>(arg0),
                                                          std::forward<Ts>(args)...);
    case ast_operator::ARCCOSH:
      return f.template operator()<ast_operator::ARCCOSH>(std::forward<T0>(arg0),
                                                          std::forward<Ts>(args)...);
    case ast_operator::ARCTANH:
      return f.template operator()<ast_operator::ARCTANH>(std::forward<T0>(arg0),
                                                          std::forward<Ts>(args)...);
    case ast_operator::EXP:
      return f.template operator()<ast_operator::EXP>(std::forward<T0>(arg0),
                                                      std::forward<Ts>(args)...);
    case ast_operator::LOG:
      return f.template operator()<ast_operator::LOG>(std::forward<T0>(arg0),
                                                      std::forward<Ts>(args)...);
    case ast_operator::SQRT:
      return f.template operator()<ast_operator::SQRT>(std::forward<T0>(arg0),
                                                       std::forward<Ts>(args)...);
    case ast_operator::CBRT:
      return f.template operator()<ast_operator::CBRT>(std::forward<T0>(arg0),
                                                       std::forward<Ts>(args)...);
    case ast_operator::CEIL:
      return f.template operator()<ast_operator::CEIL>(std::forward<T0>(arg0),
                                                       std::forward<Ts>(args)...);
    case ast_operator::FLOOR:
      return f.template operator()<ast_operator::FLOOR>(std::forward<T0>(arg0),
                                                        std::forward<Ts>(args)...);
    case ast_operator::ABS:
      return f.template operator()<ast_operator::ABS>(std::forward<T0>(arg0),
                                                      std::forward<Ts>(args)...);
    case ast_operator::RINT:
      return f.template operator()<ast_operator::RINT>(std::forward<T0>(arg0),
                                                       std::forward<Ts>(args)...);
    case ast_operator::BIT_INVERT:
      return f.template operator()<ast_operator::BIT_INVERT>(std::forward<T0>(arg0),
                                                             std::forward<Ts>(args)...);
    case ast_operator::NOT:
      return f.template operator()<ast_operator::NOT>(std::forward<T0>(arg0),
                                                      std::forward<Ts>(args)...);
    */
    default: return T0(0);  // TODO: Error handling?
  }
}

/*
 * Define operator-dispatched traits.
 */
struct is_binary_arithmetic_operator_impl {
  template <ast_operator op>
  CUDA_HOST_DEVICE_CALLABLE bool operator()()
  {
    return is_binary_arithmetic_operator<op>();
  }
};
CUDA_HOST_DEVICE_CALLABLE bool is_binary_arithmetic_operator(ast_operator op)
{
  return ast_operator_dispatcher_bool(op, is_binary_arithmetic_operator_impl{});
}

struct is_unary_arithmetic_operator_impl {
  template <ast_operator op>
  CUDA_HOST_DEVICE_CALLABLE bool operator()()
  {
    return is_unary_arithmetic_operator<op>();
  }
};
CUDA_HOST_DEVICE_CALLABLE bool is_unary_arithmetic_operator(ast_operator op)
{
  return ast_operator_dispatcher_bool(op, is_unary_arithmetic_operator_impl{});
}

struct is_binary_comparator_impl {
  template <ast_operator op>
  CUDA_HOST_DEVICE_CALLABLE bool operator()()
  {
    return is_binary_comparator<op>();
  }
};
CUDA_HOST_DEVICE_CALLABLE bool is_binary_comparator(ast_operator op)
{
  return ast_operator_dispatcher_bool(op, is_binary_comparator_impl{});
}

struct is_unary_comparator_impl {
  template <ast_operator op>
  CUDA_HOST_DEVICE_CALLABLE bool operator()()
  {
    return is_unary_comparator<op>();
  }
};
CUDA_HOST_DEVICE_CALLABLE bool is_unary_comparator(ast_operator op)
{
  return ast_operator_dispatcher_bool(op, is_unary_comparator_impl{});
}

struct is_binary_logical_operator_impl {
  template <ast_operator op>
  CUDA_HOST_DEVICE_CALLABLE bool operator()()
  {
    return is_binary_logical_operator<op>();
  }
};
CUDA_HOST_DEVICE_CALLABLE bool is_binary_logical_operator(ast_operator op)
{
  return ast_operator_dispatcher_bool(op, is_binary_logical_operator_impl{});
}

struct is_unary_logical_operator_impl {
  template <ast_operator op>
  CUDA_HOST_DEVICE_CALLABLE bool operator()()
  {
    return is_unary_logical_operator<op>();
  }
};
CUDA_HOST_DEVICE_CALLABLE bool is_unary_logical_operator(ast_operator op)
{
  return ast_operator_dispatcher_bool(op, is_unary_logical_operator_impl{});
}

// Arithmetic operators accept element type(s) and return an element
CUDA_HOST_DEVICE_CALLABLE bool is_arithmetic_operator(ast_operator op)
{
  return is_binary_arithmetic_operator(op) || is_unary_arithmetic_operator(op);
}

// Comparators accept element type(s) and return a boolean
CUDA_HOST_DEVICE_CALLABLE bool is_comparator(ast_operator op)
{
  return is_binary_comparator(op) || is_unary_comparator(op);
}

// Logical accept boolean(s) and return a boolean
CUDA_HOST_DEVICE_CALLABLE bool is_logical_operator(ast_operator op)
{
  return is_binary_logical_operator(op) || is_unary_logical_operator(op);
}

// Binary operators accept two inputs
CUDA_HOST_DEVICE_CALLABLE bool is_binary_operator(ast_operator op)
{
  return is_binary_arithmetic_operator(op) || is_binary_comparator(op) ||
         is_binary_logical_operator(op);
}

// Unary operators accept one input
CUDA_HOST_DEVICE_CALLABLE bool is_unary_operator(ast_operator op)
{
  return is_unary_arithmetic_operator(op) || is_unary_comparator(op) ||
         is_unary_logical_operator(op);
}

template <ast_operator op>
struct binop {
  // TODO: This default might need to be removed - just making things compile for now
  template <typename T>
  __device__ T operator()(T const& lhs, T const& rhs)
  {
    return 0;
  }
};

template <>
struct binop<ast_operator::ADD> {
  template <typename T>
  __device__ T operator()(T const& lhs, T const& rhs)
  {
    return lhs + rhs;
  }
};

template <>
struct binop<ast_operator::SUB> {
  template <typename T>
  __device__ T operator()(T const& lhs, T const& rhs)
  {
    return lhs - rhs;
  }
};

template <>
struct binop<ast_operator::MUL> {
  template <typename T>
  __device__ T operator()(T const& lhs, T const& rhs)
  {
    return lhs * rhs;
  }
};

template <>
struct binop<ast_operator::DIV> {
  template <typename T>
  __device__ T operator()(T const& lhs, T const& rhs)
  {
    return lhs / rhs;
  }
};

template <>
struct binop<ast_operator::TRUE_DIV> {
  template <typename T>
  __device__ T operator()(T const& lhs, T const& rhs)
  {
    return lhs / rhs;
  }
};

template <>
struct binop<ast_operator::FLOOR_DIV> {
  template <typename T>
  __device__ T operator()(T const& lhs, T const& rhs)
  {
    return floor(lhs / rhs);
  }
};

template <>
struct binop<ast_operator::MOD> {
  // TODO: May need more templating here to use fmod / fmodf if T is float/double
  template <typename T>
  __device__ T operator()(T const& lhs, T const& rhs)
  {
    return lhs % rhs;
  }
};

template <>
struct binop<ast_operator::PYMOD> {
  // TODO: May need more templating here to use fmod / fmodf if T is float/double
  template <typename T>
  __device__ T operator()(T const& lhs, T const& rhs)
  {
    return ((lhs % rhs) + rhs) % rhs;
  }
};

template <>
struct binop<ast_operator::POW> {
  template <typename T>
  __device__ T operator()(T const& lhs, T const& rhs)
  {
    return pow(lhs, rhs);
  }
};

template <>
struct binop<ast_operator::BITWISE_AND> {
  template <typename T>
  __device__ T operator()(T const& lhs, T const& rhs)
  {
    return lhs & rhs;
  }
};

template <>
struct binop<ast_operator::BITWISE_OR> {
  template <typename T>
  __device__ T operator()(T const& lhs, T const& rhs)
  {
    return lhs | rhs;
  }
};

template <>
struct binop<ast_operator::BITWISE_XOR> {
  template <typename T>
  __device__ T operator()(T const& lhs, T const& rhs)
  {
    return lhs ^ rhs;
  }
};

template <>
struct binop<ast_operator::COALESCE> {
  // TODO: Not yet implemented correctly. Needs to know nullity of lhs, rhs.
  template <typename T>
  __device__ T operator()(T const& lhs, T const& rhs)
  {
    return lhs;
  }
};

template <>
struct binop<ast_operator::SHIFT_LEFT> {
  template <typename T>
  __device__ T operator()(T const& lhs, T const& rhs)
  {
    return lhs << rhs;
  }
};

template <>
struct binop<ast_operator::SHIFT_RIGHT> {
  template <typename T>
  __device__ T operator()(T const& lhs, T const& rhs)
  {
    return lhs >> rhs;
  }
};

template <>
struct binop<ast_operator::SHIFT_RIGHT_UNSIGNED> {
  template <typename T>
  __device__ T operator()(T const& lhs, T const& rhs)
  {
    return static_cast<std::make_unsigned_t<T>>(lhs) >> rhs;
  }
};

template <>
struct binop<ast_operator::LOG_BASE> {
  template <typename T>
  __device__ T operator()(T const& lhs, T const& rhs)
  {
    return std::log(lhs) / std::log(rhs);
  }
};

template <>
struct binop<ast_operator::ATAN2> {
  template <typename T>
  __device__ T operator()(T const& lhs, T const& rhs)
  {
    return std::atan2(rhs, lhs);
  }
};

template <>
struct binop<ast_operator::PMOD> {
  // TODO: May need more templating here to use fmod / fmodf if T is float/double
  template <typename T>
  __device__ T operator()(T const& lhs, T const& rhs)
  {
    T rem = lhs % rhs;
    if (rem < 0) rem = (rem + rhs) % rhs;
    return rem;
  }
};

template <>
struct binop<ast_operator::NULL_MAX> {
  // TODO: Not yet implemented correctly. Needs to know nullity of lhs, rhs.
  template <typename T>
  __device__ T operator()(T const& lhs, T const& rhs)
  {
    return lhs > rhs ? lhs : rhs;
  }
};

template <>
struct binop<ast_operator::NULL_MIN> {
  // TODO: Not yet implemented correctly. Needs to know nullity of lhs, rhs.
  template <typename T>
  __device__ T operator()(T const& lhs, T const& rhs)
  {
    return lhs < rhs ? lhs : rhs;
  }
};

template <typename F, typename... Ts>
__device__ decltype(auto) binop_dispatcher(ast_operator op, F&& f, Ts&&... args)
{
  switch (op) {
    case ast_operator::ADD:
      return f.template operator()<ast_operator::ADD>(std::forward<Ts>(args)...);
    case ast_operator::SUB:
      return f.template operator()<ast_operator::SUB>(std::forward<Ts>(args)...);
    case ast_operator::MUL:
      return f.template operator()<ast_operator::MUL>(std::forward<Ts>(args)...);
    case ast_operator::DIV:
      return f.template operator()<ast_operator::DIV>(std::forward<Ts>(args)...);
    /*
    case ast_operator::TRUE_DIV:
      return f.template operator()<ast_operator::TRUE_DIV>(std::forward<Ts>(args)...);
    case ast_operator::FLOOR_DIV:
      return f.template operator()<ast_operator::FLOOR_DIV>(std::forward<Ts>(args)...);
    case ast_operator::MOD:
      return f.template operator()<ast_operator::MOD>(std::forward<Ts>(args)...);
    case ast_operator::PYMOD:
      return f.template operator()<ast_operator::PYMOD>(std::forward<Ts>(args)...);
    case ast_operator::POW:
      return f.template operator()<ast_operator::POW>(std::forward<Ts>(args)...);
    case ast_operator::BITWISE_AND:
      return f.template operator()<ast_operator::BITWISE_AND>(std::forward<Ts>(args)...);
    case ast_operator::BITWISE_OR:
      return f.template operator()<ast_operator::BITWISE_OR>(std::forward<Ts>(args)...);
    case ast_operator::BITWISE_XOR:
      return f.template operator()<ast_operator::BITWISE_XOR>(std::forward<Ts>(args)...);
    case ast_operator::COALESCE:
      return f.template operator()<ast_operator::COALESCE>(std::forward<Ts>(args)...);
    case ast_operator::SHIFT_LEFT:
      return f.template operator()<ast_operator::SHIFT_LEFT>(std::forward<Ts>(args)...);
    case ast_operator::SHIFT_RIGHT:
      return f.template operator()<ast_operator::SHIFT_RIGHT>(std::forward<Ts>(args)...);
    case ast_operator::SHIFT_RIGHT_UNSIGNED:
      return f.template operator()<ast_operator::SHIFT_RIGHT_UNSIGNED>(std::forward<Ts>(args)...);
    case ast_operator::LOG_BASE:
      return f.template operator()<ast_operator::LOG_BASE>(std::forward<Ts>(args)...);
    case ast_operator::ATAN2:
      return f.template operator()<ast_operator::ATAN2>(std::forward<Ts>(args)...);
    case ast_operator::PMOD:
      return f.template operator()<ast_operator::PMOD>(std::forward<Ts>(args)...);
    case ast_operator::NULL_MAX:
      return f.template operator()<ast_operator::NULL_MAX>(std::forward<Ts>(args)...);
    case ast_operator::NULL_MIN:
      return f.template operator()<ast_operator::NULL_MIN>(std::forward<Ts>(args)...);
    */
    default: return 0;  // TODO: Error handling
  }
}

template <typename T>
struct do_binop {
  template <ast_operator OP>
  __device__ T operator()(T const& lhs, T const& rhs)
  {
    return binop<OP>{}(lhs, rhs);
  }
};

}  // namespace ast

}  // namespace cudf
