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

#include <type_traits>

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
  // Other ones?
  IS_NULL,  ///< Unary comparator returning whether the value is null
};

template <ast_operator op, typename = void>
struct is_binary_arithmetic_operator_impl : std::false_type {
};

template <ast_operator op, typename = void>
struct is_unary_arithmetic_operator_impl : std::false_type {
};

template <ast_operator op, typename = void>
struct is_binary_comparator_impl : std::false_type {
};

template <ast_operator op, typename = void>
struct is_unary_comparator_impl : std::false_type {
};

template <ast_operator op, typename = void>
struct is_binary_logical_operator_impl : std::false_type {
};

template <ast_operator op, typename = void>
struct is_unary_logical_operator_impl : std::false_type {
};

template <ast_operator op>
constexpr inline bool is_binary_arithmetic_operator()
{
  return is_binary_arithmetic_operator_impl<op>::value;
}

template <ast_operator op>
constexpr inline bool is_unary_arithmetic_operator()
{
  return is_unary_arithmetic_operator_impl<op>::value;
}

template <ast_operator op>
constexpr inline bool is_binary_comparator()
{
  return is_binary_comparator_impl<op>::value;
}

template <ast_operator op>
constexpr inline bool is_unary_comparator()
{
  return is_unary_comparator_impl<op>::value;
}

template <ast_operator op>
constexpr inline bool is_binary_logical_operator()
{
  return is_binary_logical_operator_impl<op>::value;
}

template <ast_operator op>
constexpr inline bool is_unary_logical_operator()
{
  return is_unary_logical_operator_impl<op>::value;
}

// Math operators accept element type(s) and return an element
template <ast_operator op>
constexpr inline bool is_arithmetic_operator()
{
  return is_binary_arithmetic_operator<op>() || is_unary_arithmetic_operator<op>();
}

// Comparators accept element type(s) and return a boolean
template <ast_operator op>
constexpr inline bool is_comparator()
{
  return is_binary_comparator<op>() || is_unary_comparator<op>();
}

// Logical accept boolean(s) and return a boolean
template <ast_operator op>
constexpr inline bool is_logical_operator()
{
  return is_binary_logical_operator<op>() || is_unary_logical_operator<op>();
}

// Binary operators accept two inputs
template <ast_operator op>
constexpr inline bool is_binary_operator()
{
  return is_binary_arithmetic_operator<op>() || is_binary_comparator<op>() ||
         is_binary_logical_operator<op>();
}

// Unary operators accept one input
template <ast_operator op>
constexpr inline bool is_unary_operator()
{
  return is_unary_arithmetic_operator<op>() || is_unary_comparator<op>() ||
         is_unary_logical_operator<op>();
}

template <>
struct is_binary_arithmetic_operator_impl<ast_operator::ADD> : std::true_type {
};

template <>
struct is_binary_arithmetic_operator_impl<ast_operator::SUB> : std::true_type {
};

template <>
struct is_binary_arithmetic_operator_impl<ast_operator::MUL> : std::true_type {
};

template <>
struct is_binary_arithmetic_operator_impl<ast_operator::DIV> : std::true_type {
};

template <>
struct is_binary_arithmetic_operator_impl<ast_operator::TRUE_DIV> : std::true_type {
};

template <>
struct is_binary_arithmetic_operator_impl<ast_operator::FLOOR_DIV> : std::true_type {
};

template <>
struct is_binary_arithmetic_operator_impl<ast_operator::MOD> : std::true_type {
};

template <>
struct is_binary_arithmetic_operator_impl<ast_operator::PYMOD> : std::true_type {
};

template <>
struct is_binary_arithmetic_operator_impl<ast_operator::POW> : std::true_type {
};

template <>
struct is_binary_arithmetic_operator_impl<ast_operator::BITWISE_AND> : std::true_type {
};

template <>
struct is_binary_arithmetic_operator_impl<ast_operator::BITWISE_OR> : std::true_type {
};

template <>
struct is_binary_arithmetic_operator_impl<ast_operator::BITWISE_XOR> : std::true_type {
};

template <>
struct is_binary_arithmetic_operator_impl<ast_operator::COALESCE> : std::true_type {
};

template <>
struct is_binary_arithmetic_operator_impl<ast_operator::SHIFT_LEFT> : std::true_type {
};

template <>
struct is_binary_arithmetic_operator_impl<ast_operator::SHIFT_RIGHT> : std::true_type {
};

template <>
struct is_binary_arithmetic_operator_impl<ast_operator::SHIFT_RIGHT_UNSIGNED> : std::true_type {
};

template <>
struct is_binary_arithmetic_operator_impl<ast_operator::LOG_BASE> : std::true_type {
};

template <>
struct is_binary_arithmetic_operator_impl<ast_operator::ATAN2> : std::true_type {
};

template <>
struct is_binary_arithmetic_operator_impl<ast_operator::PMOD> : std::true_type {
};

template <>
struct is_binary_arithmetic_operator_impl<ast_operator::NULL_MAX> : std::true_type {
};

template <>
struct is_binary_arithmetic_operator_impl<ast_operator::NULL_MIN> : std::true_type {
};

template <>
struct is_binary_comparator_impl<ast_operator::EQUAL> : std::true_type {
};

template <>
struct is_binary_comparator_impl<ast_operator::NOT_EQUAL> : std::true_type {
};

template <>
struct is_binary_comparator_impl<ast_operator::LESS> : std::true_type {
};

template <>
struct is_binary_comparator_impl<ast_operator::GREATER> : std::true_type {
};

template <>
struct is_binary_comparator_impl<ast_operator::LESS_EQUAL> : std::true_type {
};

template <>
struct is_binary_comparator_impl<ast_operator::GREATER_EQUAL> : std::true_type {
};

template <>
struct is_binary_comparator_impl<ast_operator::NULL_EQUALS> : std::true_type {
};

template <>
struct is_binary_logical_operator_impl<ast_operator::LOGICAL_AND> : std::true_type {
};

template <>
struct is_binary_logical_operator_impl<ast_operator::LOGICAL_OR> : std::true_type {
};

template <>
struct is_unary_arithmetic_operator_impl<ast_operator::SIN> : std::true_type {
};

template <>
struct is_unary_arithmetic_operator_impl<ast_operator::COS> : std::true_type {
};

template <>
struct is_unary_arithmetic_operator_impl<ast_operator::TAN> : std::true_type {
};

template <>
struct is_unary_arithmetic_operator_impl<ast_operator::ARCSIN> : std::true_type {
};

template <>
struct is_unary_arithmetic_operator_impl<ast_operator::ARCCOS> : std::true_type {
};

template <>
struct is_unary_arithmetic_operator_impl<ast_operator::ARCTAN> : std::true_type {
};

template <>
struct is_unary_arithmetic_operator_impl<ast_operator::SINH> : std::true_type {
};

template <>
struct is_unary_arithmetic_operator_impl<ast_operator::COSH> : std::true_type {
};

template <>
struct is_unary_arithmetic_operator_impl<ast_operator::TANH> : std::true_type {
};

template <>
struct is_unary_arithmetic_operator_impl<ast_operator::ARCSINH> : std::true_type {
};

template <>
struct is_unary_arithmetic_operator_impl<ast_operator::ARCCOSH> : std::true_type {
};

template <>
struct is_unary_arithmetic_operator_impl<ast_operator::ARCTANH> : std::true_type {
};

template <>
struct is_unary_arithmetic_operator_impl<ast_operator::EXP> : std::true_type {
};

template <>
struct is_unary_arithmetic_operator_impl<ast_operator::LOG> : std::true_type {
};

template <>
struct is_unary_arithmetic_operator_impl<ast_operator::SQRT> : std::true_type {
};

template <>
struct is_unary_arithmetic_operator_impl<ast_operator::CBRT> : std::true_type {
};

template <>
struct is_unary_arithmetic_operator_impl<ast_operator::CEIL> : std::true_type {
};

template <>
struct is_unary_arithmetic_operator_impl<ast_operator::FLOOR> : std::true_type {
};

template <>
struct is_unary_arithmetic_operator_impl<ast_operator::ABS> : std::true_type {
};

template <>
struct is_unary_arithmetic_operator_impl<ast_operator::RINT> : std::true_type {
};

template <>
struct is_unary_arithmetic_operator_impl<ast_operator::BIT_INVERT> : std::true_type {
};

template <>
struct is_unary_logical_operator_impl<ast_operator::NOT> : std::true_type {
};

/*

template <ast_operator op>
struct binop {
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

template <typename F, typename... Ts>
__device__ decltype(auto) binop_dispatcher(ast_operator op, F&& f, Ts&&... args)
{
  switch (op) {
    case ast_operator::ADD:
      return f.template operator()<binary_operator::ADD>(std::forward<Ts>(args)...);
    case ast_operator::SUB:
      return f.template operator()<binary_operator::SUB>(std::forward<Ts>(args)...);
    default: return 0;  // TODO: Error handling
  }
}

template <typename T>
struct do_binop {
  template <binary_operator OP>
  __device__ T operator()(T const& lhs, T const& rhs)
  {
    return binop<OP>{}(lhs, rhs);
  }
};

template <comparator>
struct compareop {
};

template <>
struct compareop<comparator::LESS> {
  template <typename T>
  __device__ bool operator()(T const& lhs, T const& rhs)
  {
    return lhs < rhs;
  }
};

template <>
struct compareop<comparator::GREATER> {
  template <typename T>
  __device__ bool operator()(T const& lhs, T const& rhs)
  {
    return lhs > rhs;
  }
};

template <typename F, typename... Ts>
__device__ decltype(auto) compareop_dispatcher(comparator op, F&& f, Ts&&... args)
{
  switch (op) {
    case comparator::LESS:
      return f.template operator()<comparator::LESS>(std::forward<Ts>(args)...);
    case comparator::GREATER:
      return f.template operator()<comparator::GREATER>(std::forward<Ts>(args)...);
    default: return false;  // TODO: Error handling
  }
}

template <typename T>
struct do_compareop {
  template <comparator OP>
  __device__ bool operator()(T const& lhs, T const& rhs)
  {
    return compareop<OP>{}(lhs, rhs);
  }
};

*/

}  // namespace ast

}  // namespace cudf
