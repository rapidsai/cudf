/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Copyright 2018-2019 BlazingDB, Inc.
 *     Copyright 2018 Christian Noboa Mardini <christian@blazingdb.com>
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

#include "type.h"
#include <cudf/utilities/legacy/type_dispatcher.hpp>

namespace cudf {
namespace binops {
namespace jit {

    /**---------------------------------------------------------------------------*
     * @brief Get the Operator Name
     * 
     * @param ope (enum) The binary operator as enum of type gdf_binary_operator
     * @return std::string The name of the operator as string
     *---------------------------------------------------------------------------**/
    std::string getOperatorName(gdf_binary_operator ope) {
        switch (ope) {
            case GDF_ADD:
                return "Add";
            case GDF_SUB:
                return "Sub";
            case GDF_MUL:
                return "Mul";
            case GDF_DIV:
                return "Div";
            case GDF_TRUE_DIV:
                return "TrueDiv";
            case GDF_FLOOR_DIV:
                return "FloorDiv";
            case GDF_MOD:
                return "Mod";
            case GDF_PYMOD:
                return "PyMod";
            case GDF_POW:
                return "Pow";
            case GDF_EQUAL:
                return "Equal";
            case GDF_NOT_EQUAL:
                return "NotEqual";
            case GDF_LESS:
                return "Less";
            case GDF_GREATER:
                return "Greater";
            case GDF_LESS_EQUAL:
                return "LessEqual";
            case GDF_GREATER_EQUAL:
                return "GreaterEqual";
            case GDF_BITWISE_AND:
                return "BitwiseAnd";
            case GDF_BITWISE_OR:
                return "BitwiseOr";
            case GDF_BITWISE_XOR:
                return "BitwiseXor";
            case GDF_LOGICAL_AND:
                return "LogicalAnd";
            case GDF_LOGICAL_OR:
                return "LogicalOr";
            case GDF_GENERIC_BINARY:
                return "UserDefinedOp";
            default:
                return "None";
        }
    }

} // namespace jit
} // namespace binops
} // namespace cudf
