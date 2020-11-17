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
  std::string operator_name;
  switch (op) {
    case binary_op::ADD: operator_name = "Add"; break;
    case binary_op::SUB: operator_name = "Sub"; break;
    case binary_op::MUL: operator_name = "Mul"; break;
    case binary_op::DIV: operator_name = "Div"; break;
    case binary_op::TRUE_DIV: operator_name = "TrueDiv"; break;
    case binary_op::FLOOR_DIV: operator_name = "FloorDiv"; break;
    case binary_op::MOD: operator_name = "Mod"; break;
    case binary_op::PYMOD: operator_name = "PyMod"; break;
    case binary_op::POW: operator_name = "Pow"; break;
    case binary_op::EQUAL: operator_name = "Equal"; break;
    case binary_op::NOT_EQUAL: operator_name = "NotEqual"; break;
    case binary_op::LESS: operator_name = "Less"; break;
    case binary_op::GREATER: operator_name = "Greater"; break;
    case binary_op::LESS_EQUAL: operator_name = "LessEqual"; break;
    case binary_op::GREATER_EQUAL: operator_name = "GreaterEqual"; break;
    case binary_op::BITWISE_AND: operator_name = "BitwiseAnd"; break;
    case binary_op::BITWISE_OR: operator_name = "BitwiseOr"; break;
    case binary_op::BITWISE_XOR: operator_name = "BitwiseXor"; break;
    case binary_op::LOGICAL_AND: operator_name = "LogicalAnd"; break;
    case binary_op::LOGICAL_OR: operator_name = "LogicalOr"; break;
    case binary_op::GENERIC_BINARY: operator_name = "UserDefinedOp"; break;
    case binary_op::SHIFT_LEFT: operator_name = "ShiftLeft"; break;
    case binary_op::SHIFT_RIGHT: operator_name = "ShiftRight"; break;
    case binary_op::SHIFT_RIGHT_UNSIGNED: operator_name = "ShiftRightUnsigned"; break;
    case binary_op::LOG_BASE: operator_name = "LogBase"; break;
    case binary_op::ATAN2: operator_name = "ATan2"; break;
    case binary_op::PMOD: operator_name = "PMod"; break;
    case binary_op::NULL_EQUALS: operator_name = "NullEquals"; break;
    case binary_op::NULL_MAX: operator_name = "NullMax"; break;
    case binary_op::NULL_MIN: operator_name = "NullMin"; break;
    default: operator_name = "None"; break;
  }
  if (type == OperatorType::Direct) {
    return operator_name;
  } else {
    return 'R' + operator_name;
  }
}

}  // namespace jit
}  // namespace binops
}  // namespace cudf
