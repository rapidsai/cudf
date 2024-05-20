/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda/std/optional>
#include <cuda/std/type_traits>

#include <cmath>
#include <type_traits>
#include <utility>
#include <vector>

namespace cudf {

namespace ast {

namespace detail {

// Type trait for wrapping nullable types in a thrust::optional. Non-nullable
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
CUDF_HOST_DEVICE inline constexpr void ast_operator_dispatcher(ast_operator op, F&& f, Ts&&... args)
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
    case ast_operator::TRUE_DIV:
      f.template operator()<ast_operator::TRUE_DIV>(std::forward<Ts>(args)...);
      break;
    case ast_operator::FLOOR_DIV:
      f.template operator()<ast_operator::FLOOR_DIV>(std::forward<Ts>(args)...);
      break;
    case ast_operator::MOD:
      f.template operator()<ast_operator::MOD>(std::forward<Ts>(args)...);
      break;
    case ast_operator::PYMOD:
      f.template operator()<ast_operator::PYMOD>(std::forward<Ts>(args)...);
      break;
    case ast_operator::POW:
      f.template operator()<ast_operator::POW>(std::forward<Ts>(args)...);
      break;
    case ast_operator::EQUAL:
      f.template operator()<ast_operator::EQUAL>(std::forward<Ts>(args)...);
      break;
    case ast_operator::NULL_EQUAL:
      f.template operator()<ast_operator::NULL_EQUAL>(std::forward<Ts>(args)...);
      break;
    case ast_operator::NOT_EQUAL:
      f.template operator()<ast_operator::NOT_EQUAL>(std::forward<Ts>(args)...);
      break;
    case ast_operator::LESS:
      f.template operator()<ast_operator::LESS>(std::forward<Ts>(args)...);
      break;
    case ast_operator::GREATER:
      f.template operator()<ast_operator::GREATER>(std::forward<Ts>(args)...);
      break;
    case ast_operator::LESS_EQUAL:
      f.template operator()<ast_operator::LESS_EQUAL>(std::forward<Ts>(args)...);
      break;
    case ast_operator::GREATER_EQUAL:
      f.template operator()<ast_operator::GREATER_EQUAL>(std::forward<Ts>(args)...);
      break;
    case ast_operator::BITWISE_AND:
      f.template operator()<ast_operator::BITWISE_AND>(std::forward<Ts>(args)...);
      break;
    case ast_operator::BITWISE_OR:
      f.template operator()<ast_operator::BITWISE_OR>(std::forward<Ts>(args)...);
      break;
    case ast_operator::BITWISE_XOR:
      f.template operator()<ast_operator::BITWISE_XOR>(std::forward<Ts>(args)...);
      break;
    case ast_operator::LOGICAL_AND:
      f.template operator()<ast_operator::LOGICAL_AND>(std::forward<Ts>(args)...);
      break;
    case ast_operator::NULL_LOGICAL_AND:
      f.template operator()<ast_operator::NULL_LOGICAL_AND>(std::forward<Ts>(args)...);
      break;
    case ast_operator::LOGICAL_OR:
      f.template operator()<ast_operator::LOGICAL_OR>(std::forward<Ts>(args)...);
      break;
    case ast_operator::NULL_LOGICAL_OR:
      f.template operator()<ast_operator::NULL_LOGICAL_OR>(std::forward<Ts>(args)...);
      break;
    case ast_operator::IDENTITY:
      f.template operator()<ast_operator::IDENTITY>(std::forward<Ts>(args)...);
      break;
    case ast_operator::IS_NULL:
      f.template operator()<ast_operator::IS_NULL>(std::forward<Ts>(args)...);
      break;
    case ast_operator::SIN:
      f.template operator()<ast_operator::SIN>(std::forward<Ts>(args)...);
      break;
    case ast_operator::COS:
      f.template operator()<ast_operator::COS>(std::forward<Ts>(args)...);
      break;
    case ast_operator::TAN:
      f.template operator()<ast_operator::TAN>(std::forward<Ts>(args)...);
      break;
    case ast_operator::ARCSIN:
      f.template operator()<ast_operator::ARCSIN>(std::forward<Ts>(args)...);
      break;
    case ast_operator::ARCCOS:
      f.template operator()<ast_operator::ARCCOS>(std::forward<Ts>(args)...);
      break;
    case ast_operator::ARCTAN:
      f.template operator()<ast_operator::ARCTAN>(std::forward<Ts>(args)...);
      break;
    case ast_operator::SINH:
      f.template operator()<ast_operator::SINH>(std::forward<Ts>(args)...);
      break;
    case ast_operator::COSH:
      f.template operator()<ast_operator::COSH>(std::forward<Ts>(args)...);
      break;
    case ast_operator::TANH:
      f.template operator()<ast_operator::TANH>(std::forward<Ts>(args)...);
      break;
    case ast_operator::ARCSINH:
      f.template operator()<ast_operator::ARCSINH>(std::forward<Ts>(args)...);
      break;
    case ast_operator::ARCCOSH:
      f.template operator()<ast_operator::ARCCOSH>(std::forward<Ts>(args)...);
      break;
    case ast_operator::ARCTANH:
      f.template operator()<ast_operator::ARCTANH>(std::forward<Ts>(args)...);
      break;
    case ast_operator::EXP:
      f.template operator()<ast_operator::EXP>(std::forward<Ts>(args)...);
      break;
    case ast_operator::LOG:
      f.template operator()<ast_operator::LOG>(std::forward<Ts>(args)...);
      break;
    case ast_operator::SQRT:
      f.template operator()<ast_operator::SQRT>(std::forward<Ts>(args)...);
      break;
    case ast_operator::CBRT:
      f.template operator()<ast_operator::CBRT>(std::forward<Ts>(args)...);
      break;
    case ast_operator::CEIL:
      f.template operator()<ast_operator::CEIL>(std::forward<Ts>(args)...);
      break;
    case ast_operator::FLOOR:
      f.template operator()<ast_operator::FLOOR>(std::forward<Ts>(args)...);
      break;
    case ast_operator::ABS:
      f.template operator()<ast_operator::ABS>(std::forward<Ts>(args)...);
      break;
    case ast_operator::RINT:
      f.template operator()<ast_operator::RINT>(std::forward<Ts>(args)...);
      break;
    case ast_operator::BIT_INVERT:
      f.template operator()<ast_operator::BIT_INVERT>(std::forward<Ts>(args)...);
      break;
    case ast_operator::NOT:
      f.template operator()<ast_operator::NOT>(std::forward<Ts>(args)...);
      break;
    case ast_operator::CAST_TO_INT64:
      f.template operator()<ast_operator::CAST_TO_INT64>(std::forward<Ts>(args)...);
      break;
    case ast_operator::CAST_TO_UINT64:
      f.template operator()<ast_operator::CAST_TO_UINT64>(std::forward<Ts>(args)...);
      break;
    case ast_operator::CAST_TO_FLOAT64:
      f.template operator()<ast_operator::CAST_TO_FLOAT64>(std::forward<Ts>(args)...);
      break;
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
  template <typename From>
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
  CUDF_HOST_DEVICE inline void operator()(F&& f, Ts&&... args)
  {
    f.template operator()<OperatorFunctor, LHS, LHS>(std::forward<Ts>(args)...);
  }

  template <typename LHS,
            typename F,
            typename... Ts,
            std::enable_if_t<!is_valid_binary_op<OperatorFunctor, LHS, LHS>>* = nullptr>
  CUDF_HOST_DEVICE inline void operator()(F&& f, Ts&&... args)
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
  CUDF_HOST_DEVICE inline void operator()(cudf::data_type lhs_type,
                                          cudf::data_type rhs_type,
                                          F&& f,
                                          Ts&&... args)
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
CUDF_HOST_DEVICE inline constexpr void binary_operator_dispatcher(
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
  CUDF_HOST_DEVICE inline void operator()(F&& f, Ts&&... args)
  {
    f.template operator()<OperatorFunctor, InputT>(std::forward<Ts>(args)...);
  }

  template <typename InputT,
            typename F,
            typename... Ts,
            std::enable_if_t<!is_valid_unary_op<OperatorFunctor, InputT>>* = nullptr>
  CUDF_HOST_DEVICE inline void operator()(F&& f, Ts&&... args)
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
  CUDF_HOST_DEVICE inline void operator()(cudf::data_type input_type, F&& f, Ts&&... args)
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
CUDF_HOST_DEVICE inline constexpr void unary_operator_dispatcher(ast_operator op,
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
   * @param result Reference whose value is assigned to the result data type.
   */
  template <typename OperatorFunctor,
            typename LHS,
            typename RHS,
            std::enable_if_t<is_valid_binary_op<OperatorFunctor, LHS, RHS>>* = nullptr>
  CUDF_HOST_DEVICE inline void operator()(cudf::data_type& result)
  {
    using Out = cuda::std::invoke_result_t<OperatorFunctor, LHS, RHS>;
    result    = cudf::data_type(cudf::type_to_id<Out>());
  }

  template <typename OperatorFunctor,
            typename LHS,
            typename RHS,
            std::enable_if_t<!is_valid_binary_op<OperatorFunctor, LHS, RHS>>* = nullptr>
  CUDF_HOST_DEVICE inline void operator()(cudf::data_type& result)
  {
#ifndef __CUDA_ARCH__
    CUDF_FAIL("Invalid binary operation. Return type cannot be determined.");
#else
    CUDF_UNREACHABLE("Invalid binary operation. Return type cannot be determined.");
#endif
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
  CUDF_HOST_DEVICE inline void operator()(cudf::data_type& result)
  {
    using Out = cuda::std::invoke_result_t<OperatorFunctor, T>;
    result    = cudf::data_type(cudf::type_to_id<Out>());
  }

  template <typename OperatorFunctor,
            typename T,
            std::enable_if_t<!is_valid_unary_op<OperatorFunctor, T>>* = nullptr>
  CUDF_HOST_DEVICE inline void operator()(cudf::data_type& result)
  {
#ifndef __CUDA_ARCH__
    CUDF_FAIL("Invalid unary operation. Return type cannot be determined.");
#else
    CUDF_UNREACHABLE("Invalid unary operation. Return type cannot be determined.");
#endif
  }
};

/**
 * @brief Gets the return type of an AST operator.
 *
 * @param op Operator used to evaluate return type.
 * @param operand_types Vector of input types to the operator.
 * @return cudf::data_type Return type of the operator.
 */
inline cudf::data_type ast_operator_return_type(ast_operator op,
                                                std::vector<cudf::data_type> const& operand_types)
{
  auto result = cudf::data_type(cudf::type_id::EMPTY);
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

/**
 * @brief Functor to determine the arity (number of operands) of an operator.
 */
struct arity_functor {
  template <ast_operator op>
  CUDF_HOST_DEVICE inline void operator()(cudf::size_type& result)
  {
    // Arity is not dependent on null handling, so just use the false implementation here.
    result = operator_functor<op, false>::arity;
  }
};

/**
 * @brief Gets the arity (number of operands) of an AST operator.
 *
 * @param op Operator used to determine arity.
 * @return Arity of the operator.
 */
CUDF_HOST_DEVICE inline cudf::size_type ast_operator_arity(ast_operator op)
{
  auto result = cudf::size_type(0);
  ast_operator_dispatcher(op, detail::arity_functor{}, result);
  return result;
}

}  // namespace detail

}  // namespace ast

}  // namespace cudf
