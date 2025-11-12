/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/ast/ast_operator.hpp>
#include <cudf/ast/detail/possibly_null.cuh>
#include <cudf/fixed_point/conv.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/traits.hpp>

#include <cuda/std/cmath>

namespace CUDF_EXPORT cudf {
namespace ast::detail {

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
  __device__ inline auto operator()(LHS lhs, RHS rhs) const noexcept -> decltype(lhs + rhs)
  {
    return lhs + rhs;
  }
};

template <>
struct operator_functor<ast_operator::SUB, false> {
  static constexpr auto arity{2};

  template <typename LHS, typename RHS>
  __device__ inline auto operator()(LHS lhs, RHS rhs) const noexcept -> decltype(lhs - rhs)
  {
    return lhs - rhs;
  }
};

template <>
struct operator_functor<ast_operator::MUL, false> {
  static constexpr auto arity{2};

  template <typename LHS, typename RHS>
  __device__ inline auto operator()(LHS lhs, RHS rhs) const noexcept -> decltype(lhs * rhs)
  {
    return lhs * rhs;
  }
};

template <>
struct operator_functor<ast_operator::DIV, false> {
  static constexpr auto arity{2};

  template <typename LHS, typename RHS>
  __device__ inline auto operator()(LHS lhs, RHS rhs) const noexcept -> decltype(lhs / rhs)
  {
    return lhs / rhs;
  }
};

template <>
struct operator_functor<ast_operator::TRUE_DIV, false> {
  static constexpr auto arity{2};

  template <typename LHS, typename RHS>
  __device__ inline auto operator()(LHS lhs, RHS rhs) const noexcept
    -> decltype(static_cast<double>(lhs) / static_cast<double>(rhs))
  {
    return static_cast<double>(lhs) / static_cast<double>(rhs);
  }
};

template <>
struct operator_functor<ast_operator::FLOOR_DIV, false> {
  static constexpr auto arity{2};

  template <typename LHS, typename RHS>
  __device__ inline auto operator()(LHS lhs, RHS rhs) const noexcept
    -> decltype(floor(static_cast<double>(lhs) / static_cast<double>(rhs)))
  {
    return floor(static_cast<double>(lhs) / static_cast<double>(rhs));
  }
};

template <>
struct operator_functor<ast_operator::MOD, false> {
  static constexpr auto arity{2};

  template <typename LHS, typename RHS, typename CommonType = cuda::std::common_type_t<LHS, RHS>>
  __device__ inline auto operator()(LHS lhs, RHS rhs) const noexcept
    -> decltype(static_cast<CommonType>(lhs) % static_cast<CommonType>(rhs))
    requires(cuda::std::is_integral_v<CommonType>)
  {
    return static_cast<CommonType>(lhs) % static_cast<CommonType>(rhs);
  }

  template <typename LHS, typename RHS, typename CommonType = cuda::std::common_type_t<LHS, RHS>>
  __device__ inline auto operator()(LHS lhs, RHS rhs) const noexcept
    -> decltype(fmodf(static_cast<CommonType>(lhs), static_cast<CommonType>(rhs)))
    requires(cuda::std::is_same_v<CommonType, float>)
  {
    return fmodf(static_cast<CommonType>(lhs), static_cast<CommonType>(rhs));
  }

  template <typename LHS, typename RHS, typename CommonType = cuda::std::common_type_t<LHS, RHS>>
  __device__ inline auto operator()(LHS lhs, RHS rhs) const noexcept
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
  __device__ inline auto operator()(LHS lhs, RHS rhs) const noexcept
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
  __device__ inline auto operator()(LHS lhs, RHS rhs) const noexcept
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
  __device__ inline auto operator()(LHS lhs, RHS rhs) const noexcept
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
  __device__ inline auto operator()(LHS lhs, RHS rhs) const noexcept
    -> decltype(cuda::std::pow(lhs, rhs))
  {
    return cuda::std::pow(lhs, rhs);
  }
};

template <>
struct operator_functor<ast_operator::EQUAL, false> {
  static constexpr auto arity{2};

  template <typename LHS, typename RHS>
  __device__ inline auto operator()(LHS lhs, RHS rhs) const noexcept -> decltype(lhs == rhs)
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
  __device__ inline auto operator()(LHS lhs, RHS rhs) const noexcept -> decltype(lhs != rhs)
  {
    return lhs != rhs;
  }
};

template <>
struct operator_functor<ast_operator::LESS, false> {
  static constexpr auto arity{2};

  template <typename LHS, typename RHS>
  __device__ inline auto operator()(LHS lhs, RHS rhs) const noexcept -> decltype(lhs < rhs)
  {
    return lhs < rhs;
  }
};

template <>
struct operator_functor<ast_operator::GREATER, false> {
  static constexpr auto arity{2};

  template <typename LHS, typename RHS>
  __device__ inline auto operator()(LHS lhs, RHS rhs) const noexcept -> decltype(lhs > rhs)
  {
    return lhs > rhs;
  }
};

template <>
struct operator_functor<ast_operator::LESS_EQUAL, false> {
  static constexpr auto arity{2};

  template <typename LHS, typename RHS>
  __device__ inline auto operator()(LHS lhs, RHS rhs) const noexcept -> decltype(lhs <= rhs)
  {
    return lhs <= rhs;
  }
};

template <>
struct operator_functor<ast_operator::GREATER_EQUAL, false> {
  static constexpr auto arity{2};

  template <typename LHS, typename RHS>
  __device__ inline auto operator()(LHS lhs, RHS rhs) const noexcept -> decltype(lhs >= rhs)
  {
    return lhs >= rhs;
  }
};

template <>
struct operator_functor<ast_operator::BITWISE_AND, false> {
  static constexpr auto arity{2};

  template <typename LHS, typename RHS>
  __device__ inline auto operator()(LHS lhs, RHS rhs) const noexcept -> decltype(lhs & rhs)
  {
    return lhs & rhs;
  }
};

template <>
struct operator_functor<ast_operator::BITWISE_OR, false> {
  static constexpr auto arity{2};

  template <typename LHS, typename RHS>
  __device__ inline auto operator()(LHS lhs, RHS rhs) const noexcept -> decltype(lhs | rhs)
  {
    return lhs | rhs;
  }
};

template <>
struct operator_functor<ast_operator::BITWISE_XOR, false> {
  static constexpr auto arity{2};

  template <typename LHS, typename RHS>
  __device__ inline auto operator()(LHS lhs, RHS rhs) const noexcept -> decltype(lhs ^ rhs)
  {
    return lhs ^ rhs;
  }
};

template <>
struct operator_functor<ast_operator::LOGICAL_AND, false> {
  static constexpr auto arity{2};

  template <typename LHS, typename RHS>
  __device__ inline auto operator()(LHS lhs, RHS rhs) const noexcept -> decltype(lhs && rhs)
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
  __device__ inline auto operator()(LHS lhs, RHS rhs) const noexcept -> decltype(lhs || rhs)
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
  __device__ inline auto operator()(InputT input) const noexcept -> decltype(input)
  {
    return input;
  }
};

template <>
struct operator_functor<ast_operator::IS_NULL, false> {
  static constexpr auto arity{1};

  template <typename InputT>
  __device__ inline auto operator()(InputT input) const noexcept -> bool
  {
    return false;
  }
};

template <>
struct operator_functor<ast_operator::SIN, false> {
  static constexpr auto arity{1};

  template <typename InputT>
  __device__ inline auto operator()(InputT input) const noexcept -> decltype(cuda::std::sin(input))
    requires(cuda::std::is_floating_point_v<InputT>)
  {
    return cuda::std::sin(input);
  }
};

template <>
struct operator_functor<ast_operator::COS, false> {
  static constexpr auto arity{1};

  template <typename InputT>
  __device__ inline auto operator()(InputT input) const noexcept -> decltype(cuda::std::cos(input))
    requires(cuda::std::is_floating_point_v<InputT>)
  {
    return cuda::std::cos(input);
  }
};

template <>
struct operator_functor<ast_operator::TAN, false> {
  static constexpr auto arity{1};

  template <typename InputT>
  __device__ inline auto operator()(InputT input) const noexcept -> decltype(cuda::std::tan(input))
    requires(cuda::std::is_floating_point_v<InputT>)
  {
    return cuda::std::tan(input);
  }
};

template <>
struct operator_functor<ast_operator::ARCSIN, false> {
  static constexpr auto arity{1};

  template <typename InputT>
  __device__ inline auto operator()(InputT input) const noexcept -> decltype(cuda::std::asin(input))
    requires(cuda::std::is_floating_point_v<InputT>)
  {
    return cuda::std::asin(input);
  }
};

template <>
struct operator_functor<ast_operator::ARCCOS, false> {
  static constexpr auto arity{1};

  template <typename InputT>
  __device__ inline auto operator()(InputT input) const noexcept -> decltype(cuda::std::acos(input))
    requires(cuda::std::is_floating_point_v<InputT>)
  {
    return cuda::std::acos(input);
  }
};

template <>
struct operator_functor<ast_operator::ARCTAN, false> {
  static constexpr auto arity{1};

  template <typename InputT>
  __device__ inline auto operator()(InputT input) const noexcept -> decltype(cuda::std::atan(input))
    requires(cuda::std::is_floating_point_v<InputT>)
  {
    return cuda::std::atan(input);
  }
};

template <>
struct operator_functor<ast_operator::SINH, false> {
  static constexpr auto arity{1};

  template <typename InputT>
  __device__ inline auto operator()(InputT input) const noexcept -> decltype(cuda::std::sinh(input))
    requires(cuda::std::is_floating_point_v<InputT>)
  {
    return cuda::std::sinh(input);
  }
};

template <>
struct operator_functor<ast_operator::COSH, false> {
  static constexpr auto arity{1};

  template <typename InputT>
  __device__ inline auto operator()(InputT input) const noexcept -> decltype(cuda::std::cosh(input))
    requires(cuda::std::is_floating_point_v<InputT>)
  {
    return cuda::std::cosh(input);
  }
};

template <>
struct operator_functor<ast_operator::TANH, false> {
  static constexpr auto arity{1};

  template <typename InputT>
  __device__ inline auto operator()(InputT input) const noexcept -> decltype(cuda::std::tanh(input))
    requires(cuda::std::is_floating_point_v<InputT>)
  {
    return cuda::std::tanh(input);
  }
};

template <>
struct operator_functor<ast_operator::ARCSINH, false> {
  static constexpr auto arity{1};

  template <typename InputT>
  __device__ inline auto operator()(InputT input) const noexcept
    -> decltype(cuda::std::asinh(input))
    requires(cuda::std::is_floating_point_v<InputT>)
  {
    return cuda::std::asinh(input);
  }
};

template <>
struct operator_functor<ast_operator::ARCCOSH, false> {
  static constexpr auto arity{1};

  template <typename InputT>
  __device__ inline auto operator()(InputT input) const noexcept
    -> decltype(cuda::std::acosh(input))
    requires(cuda::std::is_floating_point_v<InputT>)
  {
    return cuda::std::acosh(input);
  }
};

template <>
struct operator_functor<ast_operator::ARCTANH, false> {
  static constexpr auto arity{1};

  template <typename InputT>
  __device__ inline auto operator()(InputT input) const noexcept
    -> decltype(cuda::std::atanh(input))
    requires(cuda::std::is_floating_point_v<InputT>)
  {
    return cuda::std::atanh(input);
  }
};

template <>
struct operator_functor<ast_operator::EXP, false> {
  static constexpr auto arity{1};

  template <typename InputT>
  __device__ inline auto operator()(InputT input) const noexcept -> decltype(cuda::std::exp(input))
  {
    return cuda::std::exp(input);
  }
};

template <>
struct operator_functor<ast_operator::LOG, false> {
  static constexpr auto arity{1};

  template <typename InputT>
  __device__ inline auto operator()(InputT input) const noexcept -> decltype(cuda::std::log(input))
  {
    return cuda::std::log(input);
  }
};

template <>
struct operator_functor<ast_operator::SQRT, false> {
  static constexpr auto arity{1};

  template <typename InputT>
  __device__ inline auto operator()(InputT input) const noexcept -> decltype(cuda::std::sqrt(input))
  {
    return cuda::std::sqrt(input);
  }
};

template <>
struct operator_functor<ast_operator::CBRT, false> {
  static constexpr auto arity{1};

  template <typename InputT>
  __device__ inline auto operator()(InputT input) const noexcept -> decltype(cuda::std::cbrt(input))
  {
    return cuda::std::cbrt(input);
  }
};

template <>
struct operator_functor<ast_operator::CEIL, false> {
  static constexpr auto arity{1};

  template <typename InputT>
  __device__ inline auto operator()(InputT input) const noexcept -> decltype(cuda::std::ceil(input))
  {
    return cuda::std::ceil(input);
  }
};

template <>
struct operator_functor<ast_operator::FLOOR, false> {
  static constexpr auto arity{1};

  template <typename InputT>
  __device__ inline auto operator()(InputT input) const noexcept
    -> decltype(cuda::std::floor(input))
  {
    return cuda::std::floor(input);
  }
};

template <>
struct operator_functor<ast_operator::ABS, false> {
  static constexpr auto arity{1};

  // Only accept signed or unsigned types (both require is_arithmetic<T> to be true)
  template <typename InputT>
  __device__ inline auto operator()(InputT input) const noexcept -> decltype(cuda::std::abs(input))
    requires(cuda::std::is_signed_v<InputT>)
  {
    return cuda::std::abs(input);
  }

  template <typename InputT>
  __device__ inline auto operator()(InputT input) const noexcept -> decltype(input)
    requires(cuda::std::is_unsigned_v<InputT>)
  {
    return input;
  }
};

template <>
struct operator_functor<ast_operator::RINT, false> {
  static constexpr auto arity{1};

  template <typename InputT>
  __device__ inline auto operator()(InputT input) const noexcept -> decltype(cuda::std::rint(input))
  {
    return cuda::std::rint(input);
  }
};

template <>
struct operator_functor<ast_operator::BIT_INVERT, false> {
  static constexpr auto arity{1};

  template <typename InputT>
  __device__ inline auto operator()(InputT input) const noexcept -> decltype(~input)
  {
    return ~input;
  }
};

template <>
struct operator_functor<ast_operator::NOT, false> {
  static constexpr auto arity{1};

  template <typename InputT>
  __device__ inline auto operator()(InputT input) const noexcept -> decltype(!input)
  {
    return !input;
  }
};

template <typename To>
struct cast {
  static constexpr auto arity{1};
  template <typename From>
  __device__ inline auto operator()(From f) const noexcept -> To
    requires(is_fixed_point<From>())
  {
    if constexpr (cuda::std::is_floating_point_v<To>) {
      return convert_fixed_to_floating<To>(f);
    } else {
      return static_cast<To>(f);
    }
  }

  template <typename From>
  __device__ inline auto operator()(From f) const noexcept -> decltype(static_cast<To>(f))
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
  __device__ inline auto operator()(LHS const lhs, RHS const rhs) const noexcept
    -> possibly_null_value_t<decltype(NonNullOperator{}(*lhs, *rhs)), true>
    requires(arity_placeholder == 2)
  {
    using Out = possibly_null_value_t<decltype(NonNullOperator{}(*lhs, *rhs)), true>;
    return (lhs.has_value() && rhs.has_value()) ? Out{NonNullOperator{}(*lhs, *rhs)} : Out{};
  }

  template <typename Input, std::size_t arity_placeholder = arity>
  __device__ inline auto operator()(Input const input) const noexcept
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
  __device__ inline auto operator()(LHS const lhs) const noexcept -> bool
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
  __device__ inline auto operator()(LHS const lhs, RHS const rhs) const noexcept
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
  __device__ inline auto operator()(LHS const lhs, RHS const rhs) const noexcept
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
  __device__ inline auto operator()(LHS const lhs, RHS const rhs) const noexcept
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

}  // namespace ast::detail
}  // namespace CUDF_EXPORT cudf
