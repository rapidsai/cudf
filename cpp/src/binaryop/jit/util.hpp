/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cudf/binaryop.hpp>

#include <string>

namespace cudf {
namespace binops {
namespace jit {

/**
 * @brief Orientation of lhs and rhs in operator
 */
enum class OperatorType {
  Direct,  ///< Orientation of operands is op(lhs, rhs)
  Reverse  ///< Orientation of operands is op(rhs, lhs)
};

/**
 * @brief Get the Operator Name
 *
 * @param op The binary operator as enum of type `cudf::binary_op`
 * @param type @see OperatorType
 * @return std::string The name of the operator as string
 */
std::string inline get_operator_name(binary_op op, OperatorType type)
{
  std::string const operator_name = [op] {
    // clang-format off
    switch (op) {
      case binary_op::ADD:                  return "Add";
      case binary_op::SUB:                  return "Sub";
      case binary_op::MUL:                  return "Mul";
      case binary_op::DIV:                  return "Div";
      case binary_op::TRUE_DIV:             return "TrueDiv";
      case binary_op::FLOOR_DIV:            return "FloorDiv";
      case binary_op::MOD:                  return "Mod";
      case binary_op::PYMOD:                return "PyMod";
      case binary_op::POW:                  return "Pow";
      case binary_op::EQUAL:                return "Equal";
      case binary_op::NOT_EQUAL:            return "NotEqual";
      case binary_op::LESS:                 return "Less";
      case binary_op::GREATER:              return "Greater";
      case binary_op::LESS_EQUAL:           return "LessEqual";
      case binary_op::GREATER_EQUAL:        return "GreaterEqual";
      case binary_op::BITWISE_AND:          return "BitwiseAnd";
      case binary_op::BITWISE_OR:           return "BitwiseOr";
      case binary_op::BITWISE_XOR:          return "BitwiseXor";
      case binary_op::LOGICAL_AND:          return "LogicalAnd";
      case binary_op::LOGICAL_OR:           return "LogicalOr";
      case binary_op::GENERIC_BINARY:       return "UserDefinedOp";
      case binary_op::SHIFT_LEFT:           return "ShiftLeft";
      case binary_op::SHIFT_RIGHT:          return "ShiftRight";
      case binary_op::SHIFT_RIGHT_UNSIGNED: return "ShiftRightUnsigned";
      case binary_op::LOG_BASE:             return "LogBase";
      case binary_op::ATAN2:                return "ATan2";
      case binary_op::PMOD:                 return "PMod";
      case binary_op::NULL_EQUALS:          return "NullEquals";
      case binary_op::NULL_MAX:             return "NullMax";
      case binary_op::NULL_MIN:             return "NullMin";
      default:                              return "None";
    }
    // clang-format on
  }();
  return type == OperatorType::Direct ? operator_name : 'R' + operator_name;
}

}  // namespace jit
}  // namespace binops
}  // namespace cudf
