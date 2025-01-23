/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

#include <cudf/ast/expressions.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda/std/optional>
#include <cuda/std/type_traits>

#include <cmath>
#include <type_traits>
#include <utility>
#include <vector>

namespace CUDF_EXPORT cudf {

namespace ast {

namespace detail {

// Type trait for wrapping nullable types in a cuda::std::optional. Non-nullable
// types are returned as is.
template <typename T, bool has_nulls>
struct possibly_null_value;

template <typename T>
struct possibly_null_value<T, true> {
  using type = cuda::std::optional<T>;
};

template <typename T>
struct possibly_null_value<T, false> {
  using type = T;
};

template <typename T, bool has_nulls>
using possibly_null_value_t = typename possibly_null_value<T, has_nulls>::type;

// Traits for valid operator / type combinations
template <typename Op, typename LHS, typename RHS>
constexpr bool is_valid_binary_op = cuda::std::is_invocable_v<Op, LHS, RHS>;

template <typename Op, typename T>
constexpr bool is_valid_unary_op = cuda::std::is_invocable_v<Op, T>;

/**
 * @brief Operator dispatcher
 *
 * @tparam F Type of forwarded functor.
 * @tparam Ts Parameter pack of forwarded arguments.
 * @param f Forwarded functor to be called.
 * @param args Forwarded arguments to `operator()` of `f`.
 */
template <typename F, typename... Ts>
CUDF_HOST_DEVICE inline constexpr decltype(auto) ast_operator_dispatcher(ast_operator op,
                                                                         F&& f,
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
    case ast_operator::NULL_EQUAL:
      return f.template operator()<ast_operator::NULL_EQUAL>(std::forward<Ts>(args)...);
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
    case ast_operator::NULL_LOGICAL_AND:
      return f.template operator()<ast_operator::NULL_LOGICAL_AND>(std::forward<Ts>(args)...);
    case ast_operator::LOGICAL_OR:
      return f.template operator()<ast_operator::LOGICAL_OR>(std::forward<Ts>(args)...);
    case ast_operator::NULL_LOGICAL_OR:
      return f.template operator()<ast_operator::NULL_LOGICAL_OR>(std::forward<Ts>(args)...);
    case ast_operator::IDENTITY:
      return f.template operator()<ast_operator::IDENTITY>(std::forward<Ts>(args)...);
    case ast_operator::IS_NULL:
      return f.template operator()<ast_operator::IS_NULL>(std::forward<Ts>(args)...);
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
    case ast_operator::CAST_TO_INT64:
      return f.template operator()<ast_operator::CAST_TO_INT64>(std::forward<Ts>(args)...);
    case ast_operator::CAST_TO_UINT64:
      return f.template operator()<ast_operator::CAST_TO_UINT64>(std::forward<Ts>(args)...);
    case ast_operator::CAST_TO_FLOAT64:
      return f.template operator()<ast_operator::CAST_TO_FLOAT64>(std::forward<Ts>(args)...);
    default: {
#ifndef __CUDA_ARCH__
      CUDF_FAIL("Invalid operator.");
#else
      CUDF_UNREACHABLE("Invalid operator.");
#endif
    }
  }
}

/**
 * @brief Operator functor.
 *
 * This functor is templated on an `ast_operator`, with each template specialization defining a
 * callable `operator()` that executes the operation. The functor specialization also has a member
 * `arity` defining the number of operands that are accepted by the call to `operator()`. The
 * `operator()` is templated on the types of its inputs (e.g. `typename LHS` and `typename RHS` for
 * a binary operator). Trailing return types are defined as `decltype(result)` where `result` is
 * the returned value. The trailing return types allow SFINAE to only consider template
 * instantiations for valid combinations of types. This, in turn, allows the operator functors to be
 * used with traits like `is_valid_binary_op` that rely on `std::is_invocable` and related features.
 *
 * @tparam op AST operator.
 */
template <ast_operator op, bool has_nulls>
struct operator_functor {};

template <>
struct operator_functor<ast_operator::ADD, false> {
  static constexpr auto arity{2};

  template <typename LHS, typename RHS>
  __device__ inline auto operator()(LHS lhs, RHS rhs) -> decltype(lhs + rhs)
  {
    return lhs + rhs;
  }
};

template <>
struct operator_functor<ast_operator::SUB, false> {
  static constexpr auto arity{2};

  template <typename LHS, typename RHS>
  __device__ inline auto operator()(LHS lhs, RHS rhs) -> decltype(lhs - rhs)
  {
    return lhs - rhs;
  }
};

template <>
struct operator_functor<ast_operator::MUL, false> {
  static constexpr auto arity{2};

  template <typename LHS, typename RHS>
  __device__ inline auto operator()(LHS lhs, RHS rhs) -> decltype(lhs * rhs)
  {
    return lhs * rhs;
  }
};

template <>
struct operator_functor<ast_operator::DIV, false> {
  static constexpr auto arity{2};

  template <typename LHS, typename RHS>
  __device__ inline auto operator()(LHS lhs, RHS rhs) -> decltype(lhs / rhs)
  {
    return lhs / rhs;
  }
};

template <>
struct operator_functor<ast_operator::TRUE_DIV, false> {
  static constexpr auto arity{2};

  template <typename LHS, typename RHS>
  __device__ inline auto operator()(LHS lhs, RHS rhs)
    -> decltype(static_cast<double>(lhs) / static_cast<double>(rhs))
  {
    return static_cast<double>(lhs) / static_cast<double>(rhs);
  }
};

template <>
struct operator_functor<ast_operator::FLOOR_DIV, false> {
  static constexpr auto arity{2};

  template <typename LHS, typename RHS>
  __device__ inline auto operator()(LHS lhs, RHS rhs)
    -> decltype(floor(static_cast<double>(lhs) / static_cast<double>(rhs)))
  {
    return floor(static_cast<double>(lhs) / static_cast<double>(rhs));
  }
};

template <>
struct operator_functor<ast_operator::MOD, false> {
  static constexpr auto arity{2};

  template <typename LHS,
            typename RHS,
            typename CommonType                               = std::common_type_t<LHS, RHS>,
            std::enable_if_t<std::is_integral_v<CommonType>>* = nullptr>
  __device__ inline auto operator()(LHS lhs, RHS rhs)
    -> decltype(static_cast<CommonType>(lhs) % static_cast<CommonType>(rhs))
  {
    return static_cast<CommonType>(lhs) % static_cast<CommonType>(rhs);
  }

  template <typename LHS,
            typename RHS,
            typename CommonType                                  = std::common_type_t<LHS, RHS>,
            std::enable_if_t<std::is_same_v<CommonType, float>>* = nullptr>
  __device__ inline auto operator()(LHS lhs, RHS rhs)
    -> decltype(fmodf(static_cast<CommonType>(lhs), static_cast<CommonType>(rhs)))
  {
    return fmodf(static_cast<CommonType>(lhs), static_cast<CommonType>(rhs));
  }

  template <typename LHS,
            typename RHS,
            typename CommonType                                   = std::common_type_t<LHS, RHS>,
            std::enable_if_t<std::is_same_v<CommonType, double>>* = nullptr>
  __device__ inline auto operator()(LHS lhs, RHS rhs)
    -> decltype(fmod(static_cast<CommonType>(lhs), static_cast<CommonType>(rhs)))
  {
    return fmod(static_cast<CommonType>(lhs), static_cast<CommonType>(rhs));
  }
};

template <>
struct operator_functor<ast_operator::PYMOD, false> {
  static constexpr auto arity{2};

  template <typename LHS,
            typename RHS,
            typename CommonType                               = std::common_type_t<LHS, RHS>,
            std::enable_if_t<std::is_integral_v<CommonType>>* = nullptr>
  __device__ inline auto operator()(LHS lhs, RHS rhs)
    -> decltype(((static_cast<CommonType>(lhs) % static_cast<CommonType>(rhs)) +
                 static_cast<CommonType>(rhs)) %
                static_cast<CommonType>(rhs))
  {
    return ((static_cast<CommonType>(lhs) % static_cast<CommonType>(rhs)) +
            static_cast<CommonType>(rhs)) %
           static_cast<CommonType>(rhs);
  }

  template <typename LHS,
            typename RHS,
            typename CommonType                                  = std::common_type_t<LHS, RHS>,
            std::enable_if_t<std::is_same_v<CommonType, float>>* = nullptr>
  __device__ inline auto operator()(LHS lhs, RHS rhs)
    -> decltype(fmodf(fmodf(static_cast<CommonType>(lhs), static_cast<CommonType>(rhs)) +
                        static_cast<CommonType>(rhs),
                      static_cast<CommonType>(rhs)))
  {
    return fmodf(fmodf(static_cast<CommonType>(lhs), static_cast<CommonType>(rhs)) +
                   static_cast<CommonType>(rhs),
                 static_cast<CommonType>(rhs));
  }

  template <typename LHS,
            typename RHS,
            typename CommonType                                   = std::common_type_t<LHS, RHS>,
            std::enable_if_t<std::is_same_v<CommonType, double>>* = nullptr>
  __device__ inline auto operator()(LHS lhs, RHS rhs)
    -> decltype(fmod(fmod(static_cast<CommonType>(lhs), static_cast<CommonType>(rhs)) +
                       static_cast<CommonType>(rhs),
                     static_cast<CommonType>(rhs)))
  {
    return fmod(fmod(static_cast<CommonType>(lhs), static_cast<CommonType>(rhs)) +
                  static_cast<CommonType>(rhs),
                static_cast<CommonType>(rhs));
  }
};

template <>
struct operator_functor<ast_operator::POW, false> {
  static constexpr auto arity{2};

  template <typename LHS, typename RHS>
  __device__ inline auto operator()(LHS lhs, RHS rhs) -> decltype(std::pow(lhs, rhs))
  {
    return std::pow(lhs, rhs);
  }
};

template <>
struct operator_functor<ast_operator::EQUAL, false> {
  static constexpr auto arity{2};

  template <typename LHS, typename RHS>
  __device__ inline auto operator()(LHS lhs, RHS rhs) -> decltype(lhs == rhs)
  {
    return lhs == rhs;
  }
};

// Alias NULL_EQUAL = EQUAL in the non-nullable case.
template <>
struct operator_functor<ast_operator::NULL_EQUAL, false>
  : public operator_functor<ast_operator::EQUAL, false> {};

template <>
struct operator_functor<ast_operator::NOT_EQUAL, false> {
  static constexpr auto arity{2};

  template <typename LHS, typename RHS>
  __device__ inline auto operator()(LHS lhs, RHS rhs) -> decltype(lhs != rhs)
  {
    return lhs != rhs;
  }
};

template <>
struct operator_functor<ast_operator::LESS, false> {
  static constexpr auto arity{2};

  template <typename LHS, typename RHS>
  __device__ inline auto operator()(LHS lhs, RHS rhs) -> decltype(lhs < rhs)
  {
    return lhs < rhs;
  }
};

template <>
struct operator_functor<ast_operator::GREATER, false> {
  static constexpr auto arity{2};

  template <typename LHS, typename RHS>
  __device__ inline auto operator()(LHS lhs, RHS rhs) -> decltype(lhs > rhs)
  {
    return lhs > rhs;
  }
};

template <>
struct operator_functor<ast_operator::LESS_EQUAL, false> {
  static constexpr auto arity{2};

  template <typename LHS, typename RHS>
  __device__ inline auto operator()(LHS lhs, RHS rhs) -> decltype(lhs <= rhs)
  {
    return lhs <= rhs;
  }
};

template <>
struct operator_functor<ast_operator::GREATER_EQUAL, false> {
  static constexpr auto arity{2};

  template <typename LHS, typename RHS>
  __device__ inline auto operator()(LHS lhs, RHS rhs) -> decltype(lhs >= rhs)
  {
    return lhs >= rhs;
  }
};

template <>
struct operator_functor<ast_operator::BITWISE_AND, false> {
  static constexpr auto arity{2};

  template <typename LHS, typename RHS>
  __device__ inline auto operator()(LHS lhs, RHS rhs) -> decltype(lhs & rhs)
  {
    return lhs & rhs;
  }
};

template <>
struct operator_functor<ast_operator::BITWISE_OR, false> {
  static constexpr auto arity{2};

  template <typename LHS, typename RHS>
  __device__ inline auto operator()(LHS lhs, RHS rhs) -> decltype(lhs | rhs)
  {
    return lhs | rhs;
  }
};

template <>
struct operator_functor<ast_operator::BITWISE_XOR, false> {
  static constexpr auto arity{2};

  template <typename LHS, typename RHS>
  __device__ inline auto operator()(LHS lhs, RHS rhs) -> decltype(lhs ^ rhs)
  {
    return lhs ^ rhs;
  }
};

template <>
struct operator_functor<ast_operator::LOGICAL_AND, false> {
  static constexpr auto arity{2};

  template <typename LHS, typename RHS>
  __device__ inline auto operator()(LHS lhs, RHS rhs) -> decltype(lhs && rhs)
  {
    return lhs && rhs;
  }
};

// Alias NULL_LOGICAL_AND = LOGICAL_AND in the non-nullable case.
template <>
struct operator_functor<ast_operator::NULL_LOGICAL_AND, false>
  : public operator_functor<ast_operator::LOGICAL_AND, false> {};

template <>
struct operator_functor<ast_operator::LOGICAL_OR, false> {
  static constexpr auto arity{2};

  template <typename LHS, typename RHS>
  __device__ inline auto operator()(LHS lhs, RHS rhs) -> decltype(lhs || rhs)
  {
    return lhs || rhs;
  }
};

// Alias NULL_LOGICAL_OR = LOGICAL_OR in the non-nullable case.
template <>
struct operator_functor<ast_operator::NULL_LOGICAL_OR, false>
  : public operator_functor<ast_operator::LOGICAL_OR, false> {};

template <>
struct operator_functor<ast_operator::IDENTITY, false> {
  static constexpr auto arity{1};

  template <typename InputT>
  __device__ inline auto operator()(InputT input) -> decltype(input)
  {
    return input;
  }
};

template <>
struct operator_functor<ast_operator::IS_NULL, false> {
  static constexpr auto arity{1};

  template <typename InputT>
  __device__ inline auto operator()(InputT input) -> bool
  {
    return false;
  }
};

template <>
struct operator_functor<ast_operator::SIN, false> {
  static constexpr auto arity{1};

  template <typename InputT, std::enable_if_t<std::is_floating_point_v<InputT>>* = nullptr>
  __device__ inline auto operator()(InputT input) -> decltype(std::sin(input))
  {
    return std::sin(input);
  }
};

template <>
struct operator_functor<ast_operator::COS, false> {
  static constexpr auto arity{1};

  template <typename InputT, std::enable_if_t<std::is_floating_point_v<InputT>>* = nullptr>
  __device__ inline auto operator()(InputT input) -> decltype(std::cos(input))
  {
    return std::cos(input);
  }
};

template <>
struct operator_functor<ast_operator::TAN, false> {
  static constexpr auto arity{1};

  template <typename InputT, std::enable_if_t<std::is_floating_point_v<InputT>>* = nullptr>
  __device__ inline auto operator()(InputT input) -> decltype(std::tan(input))
  {
    return std::tan(input);
  }
};

template <>
struct operator_functor<ast_operator::ARCSIN, false> {
  static constexpr auto arity{1};

  template <typename InputT, std::enable_if_t<std::is_floating_point_v<InputT>>* = nullptr>
  __device__ inline auto operator()(InputT input) -> decltype(std::asin(input))
  {
    return std::asin(input);
  }
};

template <>
struct operator_functor<ast_operator::ARCCOS, false> {
  static constexpr auto arity{1};

  template <typename InputT, std::enable_if_t<std::is_floating_point_v<InputT>>* = nullptr>
  __device__ inline auto operator()(InputT input) -> decltype(std::acos(input))
  {
    return std::acos(input);
  }
};

template <>
struct operator_functor<ast_operator::ARCTAN, false> {
  static constexpr auto arity{1};

  template <typename InputT, std::enable_if_t<std::is_floating_point_v<InputT>>* = nullptr>
  __device__ inline auto operator()(InputT input) -> decltype(std::atan(input))
  {
    return std::atan(input);
  }
};

template <>
struct operator_functor<ast_operator::SINH, false> {
  static constexpr auto arity{1};

  template <typename InputT, std::enable_if_t<std::is_floating_point_v<InputT>>* = nullptr>
  __device__ inline auto operator()(InputT input) -> decltype(std::sinh(input))
  {
    return std::sinh(input);
  }
};

template <>
struct operator_functor<ast_operator::COSH, false> {
  static constexpr auto arity{1};

  template <typename InputT, std::enable_if_t<std::is_floating_point_v<InputT>>* = nullptr>
  __device__ inline auto operator()(InputT input) -> decltype(std::cosh(input))
  {
    return std::cosh(input);
  }
};

template <>
struct operator_functor<ast_operator::TANH, false> {
  static constexpr auto arity{1};

  template <typename InputT, std::enable_if_t<std::is_floating_point_v<InputT>>* = nullptr>
  __device__ inline auto operator()(InputT input) -> decltype(std::tanh(input))
  {
    return std::tanh(input);
  }
};

template <>
struct operator_functor<ast_operator::ARCSINH, false> {
  static constexpr auto arity{1};

  template <typename InputT, std::enable_if_t<std::is_floating_point_v<InputT>>* = nullptr>
  __device__ inline auto operator()(InputT input) -> decltype(std::asinh(input))
  {
    return std::asinh(input);
  }
};

template <>
struct operator_functor<ast_operator::ARCCOSH, false> {
  static constexpr auto arity{1};

  template <typename InputT, std::enable_if_t<std::is_floating_point_v<InputT>>* = nullptr>
  __device__ inline auto operator()(InputT input) -> decltype(std::acosh(input))
  {
    return std::acosh(input);
  }
};

template <>
struct operator_functor<ast_operator::ARCTANH, false> {
  static constexpr auto arity{1};

  template <typename InputT, std::enable_if_t<std::is_floating_point_v<InputT>>* = nullptr>
  __device__ inline auto operator()(InputT input) -> decltype(std::atanh(input))
  {
    return std::atanh(input);
  }
};

template <>
struct operator_functor<ast_operator::EXP, false> {
  static constexpr auto arity{1};

  template <typename InputT>
  __device__ inline auto operator()(InputT input) -> decltype(std::exp(input))
  {
    return std::exp(input);
  }
};

template <>
struct operator_functor<ast_operator::LOG, false> {
  static constexpr auto arity{1};

  template <typename InputT>
  __device__ inline auto operator()(InputT input) -> decltype(std::log(input))
  {
    return std::log(input);
  }
};

template <>
struct operator_functor<ast_operator::SQRT, false> {
  static constexpr auto arity{1};

  template <typename InputT>
  __device__ inline auto operator()(InputT input) -> decltype(std::sqrt(input))
  {
    return std::sqrt(input);
  }
};

template <>
struct operator_functor<ast_operator::CBRT, false> {
  static constexpr auto arity{1};

  template <typename InputT>
  __device__ inline auto operator()(InputT input) -> decltype(std::cbrt(input))
  {
    return std::cbrt(input);
  }
};

template <>
struct operator_functor<ast_operator::CEIL, false> {
  static constexpr auto arity{1};

  template <typename InputT>
  __device__ inline auto operator()(InputT input) -> decltype(std::ceil(input))
  {
    return std::ceil(input);
  }
};

template <>
struct operator_functor<ast_operator::FLOOR, false> {
  static constexpr auto arity{1};

  template <typename InputT>
  __device__ inline auto operator()(InputT input) -> decltype(std::floor(input))
  {
    return std::floor(input);
  }
};

template <>
struct operator_functor<ast_operator::ABS, false> {
  static constexpr auto arity{1};

  // Only accept signed or unsigned types (both require is_arithmetic<T> to be true)
  template <typename InputT, std::enable_if_t<std::is_signed_v<InputT>>* = nullptr>
  __device__ inline auto operator()(InputT input) -> decltype(std::abs(input))
  {
    return std::abs(input);
  }

  template <typename InputT, std::enable_if_t<std::is_unsigned_v<InputT>>* = nullptr>
  __device__ inline auto operator()(InputT input) -> decltype(input)
  {
    return input;
  }
};

template <>
struct operator_functor<ast_operator::RINT, false> {
  static constexpr auto arity{1};

  template <typename InputT>
  __device__ inline auto operator()(InputT input) -> decltype(std::rint(input))
  {
    return std::rint(input);
  }
};

template <>
struct operator_functor<ast_operator::BIT_INVERT, false> {
  static constexpr auto arity{1};

  template <typename InputT>
  __device__ inline auto operator()(InputT input) -> decltype(~input)
  {
    return ~input;
  }
};

template <>
struct operator_functor<ast_operator::NOT, false> {
  static constexpr auto arity{1};

  template <typename InputT>
  __device__ inline auto operator()(InputT input) -> decltype(!input)
  {
    return !input;
  }
};

template <typename To>
struct cast {
  static constexpr auto arity{1};
  template <typename From, typename std::enable_if_t<is_fixed_point<From>()>* = nullptr>
  __device__ inline auto operator()(From f) -> To
  {
    if constexpr (cuda::std::is_floating_point_v<To>) {
      return convert_fixed_to_floating<To>(f);
    } else {
      return static_cast<To>(f);
    }
  }

  template <typename From, typename cuda::std::enable_if_t<!is_fixed_point<From>()>* = nullptr>
  __device__ inline auto operator()(From f) -> decltype(static_cast<To>(f))
  {
    return static_cast<To>(f);
  }
};

template <>
struct operator_functor<ast_operator::CAST_TO_INT64, false> : cast<int64_t> {};
template <>
struct operator_functor<ast_operator::CAST_TO_UINT64, false> : cast<uint64_t> {};
template <>
struct operator_functor<ast_operator::CAST_TO_FLOAT64, false> : cast<double> {};

/*
 * The default specialization of nullable operators is to fall back to the non-nullable
 * implementation
 */
template <ast_operator op>
struct operator_functor<op, true> {
  using NonNullOperator       = operator_functor<op, false>;
  static constexpr auto arity = NonNullOperator::arity;

  template <typename LHS,
            typename RHS,
            std::size_t arity_placeholder             = arity,
            std::enable_if_t<arity_placeholder == 2>* = nullptr>
  __device__ inline auto operator()(LHS const lhs, RHS const rhs)
    -> possibly_null_value_t<decltype(NonNullOperator{}(*lhs, *rhs)), true>
  {
    using Out = possibly_null_value_t<decltype(NonNullOperator{}(*lhs, *rhs)), true>;
    return (lhs.has_value() && rhs.has_value()) ? Out{NonNullOperator{}(*lhs, *rhs)} : Out{};
  }

  template <typename Input,
            std::size_t arity_placeholder             = arity,
            std::enable_if_t<arity_placeholder == 1>* = nullptr>
  __device__ inline auto operator()(Input const input)
    -> possibly_null_value_t<decltype(NonNullOperator{}(*input)), true>
  {
    using Out = possibly_null_value_t<decltype(NonNullOperator{}(*input)), true>;
    return input.has_value() ? Out{NonNullOperator{}(*input)} : Out{};
  }
};

// IS_NULL(null) is true, IS_NULL(valid) is false
template <>
struct operator_functor<ast_operator::IS_NULL, true> {
  using NonNullOperator       = operator_functor<ast_operator::IS_NULL, false>;
  static constexpr auto arity = NonNullOperator::arity;

  template <typename LHS>
  __device__ inline auto operator()(LHS const lhs) -> decltype(!lhs.has_value())
  {
    return !lhs.has_value();
  }
};

// NULL_EQUAL(null, null) is true, NULL_EQUAL(null, valid) is false, and NULL_EQUAL(valid, valid) ==
// EQUAL(valid, valid)
template <>
struct operator_functor<ast_operator::NULL_EQUAL, true> {
  using NonNullOperator       = operator_functor<ast_operator::NULL_EQUAL, false>;
  static constexpr auto arity = NonNullOperator::arity;

  template <typename LHS, typename RHS>
  __device__ inline auto operator()(LHS const lhs, RHS const rhs)
    -> possibly_null_value_t<decltype(NonNullOperator{}(*lhs, *rhs)), true>
  {
    // Case 1: Neither is null, so the output is given by the operation.
    if (lhs.has_value() && rhs.has_value()) { return {NonNullOperator{}(*lhs, *rhs)}; }
    // Case 2: Two nulls compare equal.
    if (!lhs.has_value() && !rhs.has_value()) { return {true}; }
    // Case 3: One value is null, while the other is not, so we return false.
    return {false};
  }
};

///< NULL_LOGICAL_AND(null, null) is null, NULL_LOGICAL_AND(null, true) is null,
///< NULL_LOGICAL_AND(null, false) is false, and NULL_LOGICAL_AND(valid, valid) ==
///< LOGICAL_AND(valid, valid)
template <>
struct operator_functor<ast_operator::NULL_LOGICAL_AND, true> {
  using NonNullOperator       = operator_functor<ast_operator::NULL_LOGICAL_AND, false>;
  static constexpr auto arity = NonNullOperator::arity;

  template <typename LHS, typename RHS>
  __device__ inline auto operator()(LHS const lhs, RHS const rhs)
    -> possibly_null_value_t<decltype(NonNullOperator{}(*lhs, *rhs)), true>
  {
    // Case 1: Neither is null, so the output is given by the operation.
    if (lhs.has_value() && rhs.has_value()) { return {NonNullOperator{}(*lhs, *rhs)}; }
    // Case 2: Two nulls return null.
    if (!lhs.has_value() && !rhs.has_value()) { return {}; }
    // Case 3: One value is null, while the other is not. If it's true we return null, otherwise we
    // return false.
    auto const& valid_element = lhs.has_value() ? lhs : rhs;
    if (*valid_element) { return {}; }
    return {false};
  }
};

///< NULL_LOGICAL_OR(null, null) is null, NULL_LOGICAL_OR(null, true) is true, NULL_LOGICAL_OR(null,
///< false) is null, and NULL_LOGICAL_OR(valid, valid) == LOGICAL_OR(valid, valid)
template <>
struct operator_functor<ast_operator::NULL_LOGICAL_OR, true> {
  using NonNullOperator       = operator_functor<ast_operator::NULL_LOGICAL_OR, false>;
  static constexpr auto arity = NonNullOperator::arity;

  template <typename LHS, typename RHS>
  __device__ inline auto operator()(LHS const lhs, RHS const rhs)
    -> possibly_null_value_t<decltype(NonNullOperator{}(*lhs, *rhs)), true>
  {
    // Case 1: Neither is null, so the output is given by the operation.
    if (lhs.has_value() && rhs.has_value()) { return {NonNullOperator{}(*lhs, *rhs)}; }
    // Case 2: Two nulls return null.
    if (!lhs.has_value() && !rhs.has_value()) { return {}; }
    // Case 3: One value is null, while the other is not. If it's true we return true, otherwise we
    // return null.
    auto const& valid_element = lhs.has_value() ? lhs : rhs;
    if (*valid_element) { return {true}; }
    return {};
  }
};

/**
 * @brief Gets the return type of an AST operator.
 *
 * @param op Operator used to evaluate return type.
 * @param operand_types Vector of input types to the operator.
 * @return cudf::data_type Return type of the operator.
 */
cudf::data_type ast_operator_return_type(ast_operator op,
                                         std::vector<cudf::data_type> const& operand_types);

/**
 * @brief Gets the arity (number of operands) of an AST operator.
 *
 * @param op Operator used to determine arity.
 * @return Arity of the operator.
 */
cudf::size_type ast_operator_arity(ast_operator op);

}  // namespace detail

}  // namespace ast

}  // namespace CUDF_EXPORT cudf
