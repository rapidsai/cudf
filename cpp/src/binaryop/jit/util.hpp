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
namespace experimental {
namespace binops {
namespace jit {

  /**
   * @brief Orientation of lhs and rhs in operator
   */
  enum class OperatorType {
    Direct,   ///< Orientation of operands is op(lhs, rhs)
    Reverse   ///< Orientation of operands is op(rhs, lhs)
  };

  /**
   * @brief Get the Operator Name
   * 
   * @param op The binary operator as enum of type cudf::binary_operator
   * @param type @see OperatorType
   * @return std::string The name of the operator as string
   */
  std::string get_operator_name(binary_operator op, OperatorType type) {
    std::string operator_name;
    switch (op) {
      case ADD:
        operator_name = "Add"; break;
      case SUB:
        operator_name = "Sub"; break;
      case MUL:
        operator_name = "Mul"; break;
      case DIV:
        operator_name = "Div"; break;
      case TRUE_DIV:
        operator_name = "TrueDiv"; break;
      case FLOOR_DIV:
        operator_name = "FloorDiv"; break;
      case MOD:
        operator_name = "Mod"; break;
      case PYMOD:
        operator_name = "PyMod"; break;
      case POW:
        operator_name = "Pow"; break;
      case EQUAL:
        operator_name = "Equal"; break;
      case NOT_EQUAL:
        operator_name = "NotEqual"; break;
      case LESS:
        operator_name = "Less"; break;
      case GREATER:
        operator_name = "Greater"; break;
      case LESS_EQUAL:
        operator_name = "LessEqual"; break;
      case GREATER_EQUAL:
        operator_name = "GreaterEqual"; break;
      case BITWISE_AND:
        operator_name = "BitwiseAnd"; break;
      case BITWISE_OR:
        operator_name = "BitwiseOr"; break;
      case BITWISE_XOR:
        operator_name = "BitwiseXor"; break;
      case LOGICAL_AND:
        operator_name = "LogicalAnd"; break;
      case LOGICAL_OR:
        operator_name = "LogicalOr"; break;
      case GENERIC_BINARY:
        operator_name = "UserDefinedOp"; break;
      default:
        operator_name = "None"; break;
    }
    if (type == OperatorType::Direct) {
      return operator_name;
    } else {
      return 'R' + operator_name;
    }
  }

} // namespace jit
} // namespace binops
} // namespace experimental
} // namespace cudf
