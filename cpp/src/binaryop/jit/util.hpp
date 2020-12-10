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
 * @param op The binary operator as enum of type cudf::binary_operator
 * @param type @see OperatorType
 * @return std::string The name of the operator as string
 */
std::string inline get_operator_name(binary_operator op, OperatorType type)
{
  std::string const operator_name = [op] {
    // clang-format off
    switch (op) {
      case binary_operator::ADD:                  return "Add";
      case binary_operator::SUB:                  return "Sub";
      case binary_operator::MUL:                  return "Mul";
      case binary_operator::DIV:                  return "Div";
      case binary_operator::TRUE_DIV:             return "TrueDiv";
      case binary_operator::FLOOR_DIV:            return "FloorDiv";
      case binary_operator::MOD:                  return "Mod";
      case binary_operator::PYMOD:                return "PyMod";
      case binary_operator::POW:                  return "Pow";
      case binary_operator::EQUAL:                return "Equal";
      case binary_operator::NOT_EQUAL:            return "NotEqual";
      case binary_operator::LESS:                 return "Less";
      case binary_operator::GREATER:              return "Greater";
      case binary_operator::LESS_EQUAL:           return "LessEqual";
      case binary_operator::GREATER_EQUAL:        return "GreaterEqual";
      case binary_operator::BITWISE_AND:          return "BitwiseAnd";
      case binary_operator::BITWISE_OR:           return "BitwiseOr";
      case binary_operator::BITWISE_XOR:          return "BitwiseXor";
      case binary_operator::LOGICAL_AND:          return "LogicalAnd";
      case binary_operator::LOGICAL_OR:           return "LogicalOr";
      case binary_operator::GENERIC_BINARY:       return "UserDefinedOp";
      case binary_operator::SHIFT_LEFT:           return "ShiftLeft";
      case binary_operator::SHIFT_RIGHT:          return "ShiftRight";
      case binary_operator::SHIFT_RIGHT_UNSIGNED: return "ShiftRightUnsigned";
      case binary_operator::LOG_BASE:             return "LogBase";
      case binary_operator::ATAN2:                return "ATan2";
      case binary_operator::PMOD:                 return "PMod";
      case binary_operator::NULL_EQUALS:          return "NullEquals";
      case binary_operator::NULL_MAX:             return "NullMax";
      case binary_operator::NULL_MIN:             return "NullMin";
      default:                              return "None";
    }
    // clang-format on
  }();
  return type == OperatorType::Direct ? operator_name : 'R' + operator_name;
}

}  // namespace jit
}  // namespace binops
}  // namespace cudf
