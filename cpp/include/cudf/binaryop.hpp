/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#ifndef BINARYOP_HPP
#define BINARYOP_HPP

#include "cudf.h"

/**
 * @brief Types of binary operations that can be performed on data.
 */
typedef enum {
  GDF_ADD,            ///< operator +
  GDF_SUB,            ///< operator -
  GDF_MUL,            ///< operator *
  GDF_DIV,            ///< operator / using common type of lhs and rhs
  GDF_TRUE_DIV,       ///< operator / after promoting type to floating point
  GDF_FLOOR_DIV,      ///< operator / after promoting to float and then flooring the result
  GDF_MOD,            ///< operator %
  GDF_PYMOD,          ///< operator % but following python's sign rules for negatives
  GDF_POW,            ///< lhs ^ rhs
  GDF_EQUAL,          ///< operator ==
  GDF_NOT_EQUAL,      ///< operator !=
  GDF_LESS,           ///< operator <
  GDF_GREATER,        ///< operator >
  GDF_LESS_EQUAL,     ///< operator <=
  GDF_GREATER_EQUAL,  ///< operator >=
  GDF_BITWISE_AND,    ///< operator &
  GDF_BITWISE_OR,     ///< operator |
  GDF_BITWISE_XOR,    ///< operator ^
  GDF_LOGICAL_AND,    ///< operator &&
  GDF_LOGICAL_OR,     ///< operator ||
  GDF_COALESCE,       ///< operator x,y  x is null ? y : x
  GDF_GENERIC_OP,     ///< generic binary operator to be generated with input ptx code
  GDF_INVALID_BINARY  ///< invalid operation
} gdf_binary_operator;

namespace cudf
{

/**
 * @brief Performs a binary operation between a gdf_scalar and a gdf_column.
 *
 * The desired output type must be specified in out->dtype.
 *
 * If the valid field in the gdf_column output is not nullptr, then it will be
 * filled with the bitwise AND of the valid mask of rhs gdf_column and is_valid
 * bool of lhs gdf_scalar
 *
 * @param out (gdf_column) Output of the operation.
 * @param lhs (gdf_scalar) First operand of the operation.
 * @param rhs (gdf_column) Second operand of the operation.
 * @param ope (enum) The binary operator to use
 */
void binary_operation(gdf_column*           out,
                      gdf_scalar*           lhs,
                      gdf_column*           rhs,
                      gdf_binary_operator   ope);

/**
 * @brief Performs a binary operation between a gdf_column and a gdf_scalar.
 *
 * The desired output type must be specified in out->dtype.
 *
 * If the valid field in the gdf_column output is not nullptr, then it will be
 * filled with the bitwise AND of the valid mask of lhs gdf_column and is_valid
 * bool of rhs gdf_scalar
 *
 * @param out (gdf_column) Output of the operation.
 * @param lhs (gdf_column) First operand of the operation.
 * @param rhs (gdf_scalar) Second operand of the operation.
 * @param ope (enum) The binary operator to use
 */
void binary_operation(gdf_column*           out,
                      gdf_column*           lhs,
                      gdf_scalar*           rhs,
                      gdf_binary_operator   ope);

/**
 * @brief Performs a binary operation between two gdf_columns.
 *
 * The desired output type must be specified in out->dtype.
 *
 * If the valid field in the gdf_column output is not nullptr, then it will be
 * filled with the bitwise AND of the valid masks of lhs and rhs gdf_columns
 *
 * @param out (gdf_column) Output of the operation.
 * @param lhs (gdf_column) First operand of the operation.
 * @param rhs (gdf_column) Second operand of the operation.
 * @param ope (enum) The binary operator to use
 */
void binary_operation(gdf_column*           out,
                      gdf_column*           lhs,
                      gdf_column*           rhs,
                      gdf_binary_operator   ope);


/**
 * @brief Performs a binary operation between two gdf_columns using a user-defined PTX function.
 *
 * Accepts a user-defined PTX function to apply between the `lhs` and `rhs`.
 *
 * The desired output type must be specified in out->dtype.
 *
 * If the valid field in the gdf_column output is not nullptr, then it will be
 * filled with the bitwise AND of the valid masks of lhs and rhs gdf_columns
 *
 * @param out (gdf_column) Output of the operation.
 * @param lhs (gdf_column) First operand of the operation.
 * @param rhs (gdf_column) Second operand of the operation.
 * @param ptx String containing the PTX of a binary function to apply between `lhs` and `rhs`
 */
void binary_operation(gdf_column*           out,
                      gdf_column*           lhs,
                      gdf_column*           rhs,
                      const std::string&    ptx);

} // namespace cudf


#endif // BINARYOP_HPP


