/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include "operation.cuh"

#include <cudf/binaryop.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <optional>

namespace cudf::binops::compiled {

namespace {
/**
 * @brief Functor that returns optional common type of 2 or 3 given types.
 *
 */
struct common_type_functor {
  template <typename TypeLhs, typename TypeRhs>
  struct nested_common_type_functor {
    template <typename TypeOut>
    std::optional<data_type> operator()()
    {
      // If common_type exists
      if constexpr (cudf::has_common_type_v<TypeOut, TypeLhs, TypeRhs>) {
        using TypeCommon = typename std::common_type<TypeOut, TypeLhs, TypeRhs>::type;
        return data_type{type_to_id<TypeCommon>()};
      } else if constexpr (cudf::has_common_type_v<TypeLhs, TypeRhs>) {
        using TypeCommon = typename std::common_type<TypeLhs, TypeRhs>::type;
        // Eg. d=t-t
        return data_type{type_to_id<TypeCommon>()};
      }
      return {};
    }
  };
  template <typename TypeLhs, typename TypeRhs>
  std::optional<data_type> operator()(data_type out)
  {
    return type_dispatcher(out, nested_common_type_functor<TypeLhs, TypeRhs>{});
  }
};

/**
 * @brief Functor that return true if BinaryOperator supports given input and output types.
 *
 * @tparam BinaryOperator binary operator functor
 */
template <typename BinaryOperator>
struct is_binary_operation_supported {
  // For types where Out type is fixed. (eg. comparison types)
  template <typename TypeLhs, typename TypeRhs>
  inline constexpr bool operator()()
  {
    if constexpr (column_device_view::has_element_accessor<TypeLhs>() and
                  column_device_view::has_element_accessor<TypeRhs>()) {
      if constexpr (has_common_type_v<TypeLhs, TypeRhs>) {
        using common_t = std::common_type_t<TypeLhs, TypeRhs>;
        return std::is_invocable_v<BinaryOperator, common_t, common_t>;
      } else {
        return std::is_invocable_v<BinaryOperator, TypeLhs, TypeRhs>;
      }
    } else {
      return false;
    }
  }

  template <typename TypeOut, typename TypeLhs, typename TypeRhs>
  inline constexpr bool operator()()
  {
    if constexpr (column_device_view::has_element_accessor<TypeLhs>() and
                  column_device_view::has_element_accessor<TypeRhs>() and
                  (mutable_column_device_view::has_element_accessor<TypeOut>() or
                   is_fixed_point<TypeOut>())) {
      if constexpr (has_common_type_v<TypeLhs, TypeRhs>) {
        using common_t = std::common_type_t<TypeLhs, TypeRhs>;
        if constexpr (std::is_invocable_v<BinaryOperator, common_t, common_t>) {
          using ReturnType = std::invoke_result_t<BinaryOperator, common_t, common_t>;
          return std::is_constructible_v<TypeOut, ReturnType> or
                 (is_fixed_point<ReturnType>() and is_fixed_point<TypeOut>());
        }
      } else {
        if constexpr (std::is_invocable_v<BinaryOperator, TypeLhs, TypeRhs>) {
          using ReturnType = std::invoke_result_t<BinaryOperator, TypeLhs, TypeRhs>;
          return std::is_constructible_v<TypeOut, ReturnType>;
        }
      }
    }
    return false;
  }
};

struct is_supported_operation_functor {
  template <typename TypeLhs, typename TypeRhs>
  struct nested_support_functor {
    template <typename BinaryOperator, typename TypeOut>
    inline constexpr bool call()
    {
      return is_binary_operation_supported<BinaryOperator>{}
        .template operator()<TypeOut, TypeLhs, TypeRhs>();
    }
    template <typename TypeOut>
    inline constexpr bool operator()(binary_operator op)
    {
      switch (op) {
        // clang-format off
        case binary_operator::ADD:                  return call<ops::Add, TypeOut>();
        case binary_operator::SUB:                  return call<ops::Sub, TypeOut>();
        case binary_operator::MUL:                  return call<ops::Mul, TypeOut>();
        case binary_operator::DIV:                  return call<ops::Div, TypeOut>();
        case binary_operator::TRUE_DIV:             return call<ops::TrueDiv, TypeOut>();
        case binary_operator::FLOOR_DIV:            return call<ops::FloorDiv, TypeOut>();
        case binary_operator::MOD:                  return call<ops::Mod, TypeOut>();
        case binary_operator::PYMOD:                return call<ops::PyMod, TypeOut>();
        case binary_operator::POW:                  return call<ops::Pow, TypeOut>();
        case binary_operator::BITWISE_AND:          return call<ops::BitwiseAnd, TypeOut>();
        case binary_operator::BITWISE_OR:           return call<ops::BitwiseOr, TypeOut>();
        case binary_operator::BITWISE_XOR:          return call<ops::BitwiseXor, TypeOut>();
        case binary_operator::SHIFT_LEFT:           return call<ops::ShiftLeft, TypeOut>();
        case binary_operator::SHIFT_RIGHT:          return call<ops::ShiftRight, TypeOut>();
        case binary_operator::SHIFT_RIGHT_UNSIGNED: return call<ops::ShiftRightUnsigned, TypeOut>();
        case binary_operator::LOG_BASE:             return call<ops::LogBase, TypeOut>();
        case binary_operator::ATAN2:                return call<ops::ATan2, TypeOut>();
        case binary_operator::PMOD:                 return call<ops::PMod, TypeOut>();
        case binary_operator::NULL_MAX:             return call<ops::NullMax, TypeOut>();
        case binary_operator::NULL_MIN:             return call<ops::NullMin, TypeOut>();
        /*
        case binary_operator::GENERIC_BINARY:       // defined in jit only.
        */
        default:                                    return false;
          // clang-format on
      }
    }
  };

  template <typename BinaryOperator, typename TypeLhs, typename TypeRhs>
  inline constexpr bool bool_op(data_type out)
  {
    return out.id() == type_id::BOOL8 and
           is_binary_operation_supported<BinaryOperator>{}.template operator()<TypeLhs, TypeRhs>();
  }
  template <typename TypeLhs, typename TypeRhs>
  inline constexpr bool operator()(data_type out, binary_operator op)
  {
    switch (op) {
      // output type should be bool type.
      case binary_operator::LOGICAL_AND: return bool_op<ops::LogicalAnd, TypeLhs, TypeRhs>(out);
      case binary_operator::LOGICAL_OR: return bool_op<ops::LogicalOr, TypeLhs, TypeRhs>(out);
      case binary_operator::EQUAL: return bool_op<ops::Equal, TypeLhs, TypeRhs>(out);
      case binary_operator::NOT_EQUAL: return bool_op<ops::NotEqual, TypeLhs, TypeRhs>(out);
      case binary_operator::LESS: return bool_op<ops::Less, TypeLhs, TypeRhs>(out);
      case binary_operator::GREATER: return bool_op<ops::Greater, TypeLhs, TypeRhs>(out);
      case binary_operator::LESS_EQUAL: return bool_op<ops::LessEqual, TypeLhs, TypeRhs>(out);
      case binary_operator::GREATER_EQUAL: return bool_op<ops::GreaterEqual, TypeLhs, TypeRhs>(out);
      case binary_operator::NULL_EQUALS: return bool_op<ops::NullEquals, TypeLhs, TypeRhs>(out);
      case binary_operator::NULL_LOGICAL_AND:
        return bool_op<ops::NullLogicalAnd, TypeLhs, TypeRhs>(out);
      case binary_operator::NULL_LOGICAL_OR:
        return bool_op<ops::NullLogicalOr, TypeLhs, TypeRhs>(out);
      default: return type_dispatcher(out, nested_support_functor<TypeLhs, TypeRhs>{}, op);
    }
    return false;
  }
};

}  // namespace

std::optional<data_type> get_common_type(data_type out, data_type lhs, data_type rhs)
{
  return double_type_dispatcher(lhs, rhs, common_type_functor{}, out);
}

bool is_supported_operation(data_type out, data_type lhs, data_type rhs, binary_operator op)
{
  return double_type_dispatcher(lhs, rhs, is_supported_operation_functor{}, out, op);
}
}  // namespace cudf::binops::compiled
