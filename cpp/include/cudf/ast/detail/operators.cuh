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

#include <cuda/std/cmath>
#include <cuda/std/optional>
#include <cuda/std/type_traits>

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
      return f.template operator()<ast_operator::ADD>(cuda::std::forward<Ts>(args)...);
    case ast_operator::SUB:
      return f.template operator()<ast_operator::SUB>(cuda::std::forward<Ts>(args)...);
    case ast_operator::MUL:
      return f.template operator()<ast_operator::MUL>(cuda::std::forward<Ts>(args)...);
    case ast_operator::DIV:
      return f.template operator()<ast_operator::DIV>(cuda::std::forward<Ts>(args)...);
    case ast_operator::TRUE_DIV:
      return f.template operator()<ast_operator::TRUE_DIV>(cuda::std::forward<Ts>(args)...);
    case ast_operator::FLOOR_DIV:
      return f.template operator()<ast_operator::FLOOR_DIV>(cuda::std::forward<Ts>(args)...);
    case ast_operator::MOD:
      return f.template operator()<ast_operator::MOD>(cuda::std::forward<Ts>(args)...);
    case ast_operator::PYMOD:
      return f.template operator()<ast_operator::PYMOD>(cuda::std::forward<Ts>(args)...);
    case ast_operator::POW:
      return f.template operator()<ast_operator::POW>(cuda::std::forward<Ts>(args)...);
    case ast_operator::EQUAL:
      return f.template operator()<ast_operator::EQUAL>(cuda::std::forward<Ts>(args)...);
    case ast_operator::NULL_EQUAL:
      return f.template operator()<ast_operator::NULL_EQUAL>(cuda::std::forward<Ts>(args)...);
    case ast_operator::NOT_EQUAL:
      return f.template operator()<ast_operator::NOT_EQUAL>(cuda::std::forward<Ts>(args)...);
    case ast_operator::LESS:
      return f.template operator()<ast_operator::LESS>(cuda::std::forward<Ts>(args)...);
    case ast_operator::GREATER:
      return f.template operator()<ast_operator::GREATER>(cuda::std::forward<Ts>(args)...);
    case ast_operator::LESS_EQUAL:
      return f.template operator()<ast_operator::LESS_EQUAL>(cuda::std::forward<Ts>(args)...);
    case ast_operator::GREATER_EQUAL:
      return f.template operator()<ast_operator::GREATER_EQUAL>(cuda::std::forward<Ts>(args)...);
    case ast_operator::BITWISE_AND:
      return f.template operator()<ast_operator::BITWISE_AND>(cuda::std::forward<Ts>(args)...);
    case ast_operator::BITWISE_OR:
      return f.template operator()<ast_operator::BITWISE_OR>(cuda::std::forward<Ts>(args)...);
    case ast_operator::BITWISE_XOR:
      return f.template operator()<ast_operator::BITWISE_XOR>(cuda::std::forward<Ts>(args)...);
    case ast_operator::LOGICAL_AND:
      return f.template operator()<ast_operator::LOGICAL_AND>(cuda::std::forward<Ts>(args)...);
    case ast_operator::NULL_LOGICAL_AND:
      return f.template operator()<ast_operator::NULL_LOGICAL_AND>(cuda::std::forward<Ts>(args)...);
    case ast_operator::LOGICAL_OR:
      return f.template operator()<ast_operator::LOGICAL_OR>(cuda::std::forward<Ts>(args)...);
    case ast_operator::NULL_LOGICAL_OR:
      return f.template operator()<ast_operator::NULL_LOGICAL_OR>(cuda::std::forward<Ts>(args)...);
    case ast_operator::IDENTITY:
      return f.template operator()<ast_operator::IDENTITY>(cuda::std::forward<Ts>(args)...);
    case ast_operator::IS_NULL:
      return f.template operator()<ast_operator::IS_NULL>(cuda::std::forward<Ts>(args)...);
    case ast_operator::SIN:
      return f.template operator()<ast_operator::SIN>(cuda::std::forward<Ts>(args)...);
    case ast_operator::COS:
      return f.template operator()<ast_operator::COS>(cuda::std::forward<Ts>(args)...);
    case ast_operator::TAN:
      return f.template operator()<ast_operator::TAN>(cuda::std::forward<Ts>(args)...);
    case ast_operator::ARCSIN:
      return f.template operator()<ast_operator::ARCSIN>(cuda::std::forward<Ts>(args)...);
    case ast_operator::ARCCOS:
      return f.template operator()<ast_operator::ARCCOS>(cuda::std::forward<Ts>(args)...);
    case ast_operator::ARCTAN:
      return f.template operator()<ast_operator::ARCTAN>(cuda::std::forward<Ts>(args)...);
    case ast_operator::SINH:
      return f.template operator()<ast_operator::SINH>(cuda::std::forward<Ts>(args)...);
    case ast_operator::COSH:
      return f.template operator()<ast_operator::COSH>(cuda::std::forward<Ts>(args)...);
    case ast_operator::TANH:
      return f.template operator()<ast_operator::TANH>(cuda::std::forward<Ts>(args)...);
    case ast_operator::ARCSINH:
      return f.template operator()<ast_operator::ARCSINH>(cuda::std::forward<Ts>(args)...);
    case ast_operator::ARCCOSH:
      return f.template operator()<ast_operator::ARCCOSH>(cuda::std::forward<Ts>(args)...);
    case ast_operator::ARCTANH:
      return f.template operator()<ast_operator::ARCTANH>(cuda::std::forward<Ts>(args)...);
    case ast_operator::EXP:
      return f.template operator()<ast_operator::EXP>(cuda::std::forward<Ts>(args)...);
    case ast_operator::LOG:
      return f.template operator()<ast_operator::LOG>(cuda::std::forward<Ts>(args)...);
    case ast_operator::SQRT:
      return f.template operator()<ast_operator::SQRT>(cuda::std::forward<Ts>(args)...);
    case ast_operator::CBRT:
      return f.template operator()<ast_operator::CBRT>(cuda::std::forward<Ts>(args)...);
    case ast_operator::CEIL:
      return f.template operator()<ast_operator::CEIL>(cuda::std::forward<Ts>(args)...);
    case ast_operator::FLOOR:
      return f.template operator()<ast_operator::FLOOR>(cuda::std::forward<Ts>(args)...);
    case ast_operator::ABS:
      return f.template operator()<ast_operator::ABS>(cuda::std::forward<Ts>(args)...);
    case ast_operator::RINT:
      return f.template operator()<ast_operator::RINT>(cuda::std::forward<Ts>(args)...);
    case ast_operator::BIT_INVERT:
      return f.template operator()<ast_operator::BIT_INVERT>(cuda::std::forward<Ts>(args)...);
    case ast_operator::NOT:
      return f.template operator()<ast_operator::NOT>(cuda::std::forward<Ts>(args)...);
    case ast_operator::CAST_TO_INT64:
      return f.template operator()<ast_operator::CAST_TO_INT64>(cuda::std::forward<Ts>(args)...);
    case ast_operator::CAST_TO_UINT64:
      return f.template operator()<ast_operator::CAST_TO_UINT64>(cuda::std::forward<Ts>(args)...);
    case ast_operator::CAST_TO_FLOAT64:
      return f.template operator()<ast_operator::CAST_TO_FLOAT64>(cuda::std::forward<Ts>(args)...);
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

  template <typename LHS, typename RHS, typename CommonType = cuda::std::common_type_t<LHS, RHS>>
  __device__ inline auto operator()(LHS lhs, RHS rhs)
    -> decltype(static_cast<CommonType>(lhs) % static_cast<CommonType>(rhs))
    requires(cuda::std::is_integral_v<CommonType>)
  {
    return static_cast<CommonType>(lhs) % static_cast<CommonType>(rhs);
  }

  template <typename LHS, typename RHS, typename CommonType = cuda::std::common_type_t<LHS, RHS>>
  __device__ inline auto operator()(LHS lhs, RHS rhs)
    -> decltype(fmodf(static_cast<CommonType>(lhs), static_cast<CommonType>(rhs)))
    requires(cuda::std::is_same_v<CommonType, float>)
  {
    return fmodf(static_cast<CommonType>(lhs), static_cast<CommonType>(rhs));
  }

  template <typename LHS, typename RHS, typename CommonType = cuda::std::common_type_t<LHS, RHS>>
  __device__ inline auto operator()(LHS lhs, RHS rhs)
    -> decltype(fmod(static_cast<CommonType>(lhs), static_cast<CommonType>(rhs)))
    requires(cuda::std::is_same_v<CommonType, double>)
  {
    return fmod(static_cast<CommonType>(lhs), static_cast<CommonType>(rhs));
  }
};

template <>
struct operator_functor<ast_operator::PYMOD, false> {
  static constexpr auto arity{2};

  template <typename LHS, typename RHS, typename CommonType = cuda::std::common_type_t<LHS, RHS>>
  __device__ inline auto operator()(LHS lhs, RHS rhs)
    -> decltype(((static_cast<CommonType>(lhs) % static_cast<CommonType>(rhs)) +
                 static_cast<CommonType>(rhs)) %
                static_cast<CommonType>(rhs))
    requires(cuda::std::is_integral_v<CommonType>)
  {
    return ((static_cast<CommonType>(lhs) % static_cast<CommonType>(rhs)) +
            static_cast<CommonType>(rhs)) %
           static_cast<CommonType>(rhs);
  }

  template <typename LHS, typename RHS, typename CommonType = cuda::std::common_type_t<LHS, RHS>>
  __device__ inline auto operator()(LHS lhs, RHS rhs)
    -> decltype(fmodf(fmodf(static_cast<CommonType>(lhs), static_cast<CommonType>(rhs)) +
                        static_cast<CommonType>(rhs),
                      static_cast<CommonType>(rhs)))
    requires(cuda::std::is_same_v<CommonType, float>)
  {
    return fmodf(fmodf(static_cast<CommonType>(lhs), static_cast<CommonType>(rhs)) +
                   static_cast<CommonType>(rhs),
                 static_cast<CommonType>(rhs));
  }

  template <typename LHS, typename RHS, typename CommonType = cuda::std::common_type_t<LHS, RHS>>
  __device__ inline auto operator()(LHS lhs, RHS rhs)
    -> decltype(fmod(fmod(static_cast<CommonType>(lhs), static_cast<CommonType>(rhs)) +
                       static_cast<CommonType>(rhs),
                     static_cast<CommonType>(rhs)))
    requires(cuda::std::is_same_v<CommonType, double>)
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
  __device__ inline auto operator()(LHS lhs, RHS rhs) -> decltype(cuda::std::pow(lhs, rhs))
  {
    return cuda::std::pow(lhs, rhs);
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

  template <typename InputT>
  __device__ inline auto operator()(InputT input) -> decltype(cuda::std::sin(input))
    requires(cuda::std::is_floating_point_v<InputT>)
  {
    return cuda::std::sin(input);
  }
};

template <>
struct operator_functor<ast_operator::COS, false> {
  static constexpr auto arity{1};

  template <typename InputT>
  __device__ inline auto operator()(InputT input) -> decltype(cuda::std::cos(input))
    requires(cuda::std::is_floating_point_v<InputT>)
  {
    return cuda::std::cos(input);
  }
};

template <>
struct operator_functor<ast_operator::TAN, false> {
  static constexpr auto arity{1};

  template <typename InputT>
  __device__ inline auto operator()(InputT input) -> decltype(cuda::std::tan(input))
    requires(cuda::std::is_floating_point_v<InputT>)
  {
    return cuda::std::tan(input);
  }
};

template <>
struct operator_functor<ast_operator::ARCSIN, false> {
  static constexpr auto arity{1};

  template <typename InputT>
  __device__ inline auto operator()(InputT input) -> decltype(cuda::std::asin(input))
    requires(cuda::std::is_floating_point_v<InputT>)
  {
    return cuda::std::asin(input);
  }
};

template <>
struct operator_functor<ast_operator::ARCCOS, false> {
  static constexpr auto arity{1};

  template <typename InputT>
  __device__ inline auto operator()(InputT input) -> decltype(cuda::std::acos(input))
    requires(cuda::std::is_floating_point_v<InputT>)
  {
    return cuda::std::acos(input);
  }
};

template <>
struct operator_functor<ast_operator::ARCTAN, false> {
  static constexpr auto arity{1};

  template <typename InputT>
  __device__ inline auto operator()(InputT input) -> decltype(cuda::std::atan(input))
    requires(cuda::std::is_floating_point_v<InputT>)
  {
    return cuda::std::atan(input);
  }
};

template <>
struct operator_functor<ast_operator::SINH, false> {
  static constexpr auto arity{1};

  template <typename InputT>
  __device__ inline auto operator()(InputT input) -> decltype(cuda::std::sinh(input))
    requires(cuda::std::is_floating_point_v<InputT>)
  {
    return cuda::std::sinh(input);
  }
};

template <>
struct operator_functor<ast_operator::COSH, false> {
  static constexpr auto arity{1};

  template <typename InputT>
  __device__ inline auto operator()(InputT input) -> decltype(cuda::std::cosh(input))
    requires(cuda::std::is_floating_point_v<InputT>)
  {
    return cuda::std::cosh(input);
  }
};

template <>
struct operator_functor<ast_operator::TANH, false> {
  static constexpr auto arity{1};

  template <typename InputT>
  __device__ inline auto operator()(InputT input) -> decltype(cuda::std::tanh(input))
    requires(cuda::std::is_floating_point_v<InputT>)
  {
    return cuda::std::tanh(input);
  }
};

template <>
struct operator_functor<ast_operator::ARCSINH, false> {
  static constexpr auto arity{1};

  template <typename InputT>
  __device__ inline auto operator()(InputT input) -> decltype(cuda::std::asinh(input))
    requires(cuda::std::is_floating_point_v<InputT>)
  {
    return cuda::std::asinh(input);
  }
};

template <>
struct operator_functor<ast_operator::ARCCOSH, false> {
  static constexpr auto arity{1};

  template <typename InputT>
  __device__ inline auto operator()(InputT input) -> decltype(cuda::std::acosh(input))
    requires(cuda::std::is_floating_point_v<InputT>)
  {
    return cuda::std::acosh(input);
  }
};

template <>
struct operator_functor<ast_operator::ARCTANH, false> {
  static constexpr auto arity{1};

  template <typename InputT>
  __device__ inline auto operator()(InputT input) -> decltype(cuda::std::atanh(input))
    requires(cuda::std::is_floating_point_v<InputT>)
  {
    return cuda::std::atanh(input);
  }
};

template <>
struct operator_functor<ast_operator::EXP, false> {
  static constexpr auto arity{1};

  template <typename InputT>
  __device__ inline auto operator()(InputT input) -> decltype(cuda::std::exp(input))
  {
    return cuda::std::exp(input);
  }
};

template <>
struct operator_functor<ast_operator::LOG, false> {
  static constexpr auto arity{1};

  template <typename InputT>
  __device__ inline auto operator()(InputT input) -> decltype(cuda::std::log(input))
  {
    return cuda::std::log(input);
  }
};

template <>
struct operator_functor<ast_operator::SQRT, false> {
  static constexpr auto arity{1};

  template <typename InputT>
  __device__ inline auto operator()(InputT input) -> decltype(cuda::std::sqrt(input))
  {
    return cuda::std::sqrt(input);
  }
};

template <>
struct operator_functor<ast_operator::CBRT, false> {
  static constexpr auto arity{1};

  template <typename InputT>
  __device__ inline auto operator()(InputT input) -> decltype(cuda::std::cbrt(input))
  {
    return cuda::std::cbrt(input);
  }
};

template <>
struct operator_functor<ast_operator::CEIL, false> {
  static constexpr auto arity{1};

  template <typename InputT>
  __device__ inline auto operator()(InputT input) -> decltype(cuda::std::ceil(input))
  {
    return cuda::std::ceil(input);
  }
};

template <>
struct operator_functor<ast_operator::FLOOR, false> {
  static constexpr auto arity{1};

  template <typename InputT>
  __device__ inline auto operator()(InputT input) -> decltype(cuda::std::floor(input))
  {
    return cuda::std::floor(input);
  }
};

template <>
struct operator_functor<ast_operator::ABS, false> {
  static constexpr auto arity{1};

  // Only accept signed or unsigned types (both require is_arithmetic<T> to be true)
  template <typename InputT>
  __device__ inline auto operator()(InputT input) -> decltype(cuda::std::abs(input))
    requires(cuda::std::is_signed_v<InputT>)
  {
    return cuda::std::abs(input);
  }

  template <typename InputT>
  __device__ inline auto operator()(InputT input) -> decltype(input)
    requires(cuda::std::is_unsigned_v<InputT>)
  {
    return input;
  }
};

template <>
struct operator_functor<ast_operator::RINT, false> {
  static constexpr auto arity{1};

  template <typename InputT>
  __device__ inline auto operator()(InputT input) -> decltype(cuda::std::rint(input))
  {
    return cuda::std::rint(input);
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
  __device__ inline auto operator()(From f) -> To
    requires(is_fixed_point<From>())
  {
    if constexpr (cuda::std::is_floating_point_v<To>) {
      return convert_fixed_to_floating<To>(f);
    } else {
      return static_cast<To>(f);
    }
  }

  template <typename From>
  __device__ inline auto operator()(From f) -> decltype(static_cast<To>(f))
    requires(!is_fixed_point<From>())
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

  template <typename LHS, typename RHS, std::size_t arity_placeholder = arity>
  __device__ inline auto operator()(LHS const lhs, RHS const rhs)
    -> possibly_null_value_t<decltype(NonNullOperator{}(*lhs, *rhs)), true>
    requires(arity_placeholder == 2)
  {
    using Out = possibly_null_value_t<decltype(NonNullOperator{}(*lhs, *rhs)), true>;
    return (lhs.has_value() && rhs.has_value()) ? Out{NonNullOperator{}(*lhs, *rhs)} : Out{};
  }

  template <typename Input, std::size_t arity_placeholder = arity>
  __device__ inline auto operator()(Input const input)
    -> possibly_null_value_t<decltype(NonNullOperator{}(*input)), true>
    requires(arity_placeholder == 1)
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
  __device__ inline auto operator()(LHS const lhs) -> bool
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
