/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
#include "traits.hpp"

#include <cudf/binaryop.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

namespace cudf::binops::compiled {

namespace {
// common_type
struct common_type_functor {
  template <typename TypeLhs, typename TypeRhs>
  struct nested_common_type_functor {
    template <typename TypeOut>
    data_type operator()()
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
      return data_type{type_id::EMPTY};
    }
  };
  template <typename TypeLhs, typename TypeRhs>
  data_type operator()(data_type out)
  {
    return type_dispatcher(out, nested_common_type_functor<TypeLhs, TypeRhs>{});
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
        case binary_operator::EQUAL:                return call<ops::Equal, TypeOut>();
        case binary_operator::NOT_EQUAL:            return call<ops::NotEqual, TypeOut>();
        case binary_operator::LESS:                 return call<ops::Less, TypeOut>();
        case binary_operator::GREATER:              return call<ops::Greater, TypeOut>();
        case binary_operator::LESS_EQUAL:           return call<ops::LessEqual, TypeOut>();
        case binary_operator::GREATER_EQUAL:        return call<ops::GreaterEqual, TypeOut>();
        case binary_operator::BITWISE_AND:          return call<ops::BitwiseAnd, TypeOut>();
        case binary_operator::BITWISE_OR:           return call<ops::BitwiseOr, TypeOut>();
        case binary_operator::BITWISE_XOR:          return call<ops::BitwiseXor, TypeOut>();
        case binary_operator::LOGICAL_AND:          return call<ops::LogicalAnd, TypeOut>();
        case binary_operator::LOGICAL_OR:           return call<ops::LogicalOr, TypeOut>();
        //case binary_operator::GENERIC_BINARY:       return call<ops::UserDefinedOp, TypeOut>();
        case binary_operator::SHIFT_LEFT:           return call<ops::ShiftLeft, TypeOut>();
        case binary_operator::SHIFT_RIGHT:          return call<ops::ShiftRight, TypeOut>();
        case binary_operator::SHIFT_RIGHT_UNSIGNED: return call<ops::ShiftRightUnsigned, TypeOut>();
        case binary_operator::LOG_BASE:             return call<ops::LogBase, TypeOut>();
        case binary_operator::ATAN2:                return call<ops::ATan2, TypeOut>();
        case binary_operator::PMOD:                 return call<ops::PMod, TypeOut>();
        /*
        case binary_operator::NULL_EQUALS:          return call<ops::NullEquals, TypeOut>();
        case binary_operator::NULL_MAX:             return call<ops::NullMax, TypeOut>();
        case binary_operator::NULL_MIN:             return call<ops::NullMin, TypeOut>();
        */
        default:                                    return false;
          // clang-format on
      }
    }
  };
  template <typename TypeLhs, typename TypeRhs>
  inline constexpr bool operator()(data_type out, binary_operator op)
  {
    return type_dispatcher(out, nested_support_functor<TypeLhs, TypeRhs>{}, op);
  }
};

}  // namespace

data_type get_common_type(data_type out, data_type lhs, data_type rhs)
{
  return double_type_dispatcher(lhs, rhs, common_type_functor{}, out);
}

bool is_supported_operation(data_type out, data_type lhs, data_type rhs, binary_operator op)
{
  return double_type_dispatcher(lhs, rhs, is_supported_operation_functor{}, out, op);
}
}  // namespace cudf::binops::compiled
