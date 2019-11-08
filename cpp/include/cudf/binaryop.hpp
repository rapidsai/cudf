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

#pragma once

#include <cudf/cudf.h>
#include <cudf/column/column.hpp>
#include <cudf/scalar/scalar.hpp>

#include <memory>

namespace cudf {
namespace experimental {

/**
 * @brief Types of binary operations that can be performed on data.
 */
enum class binary_operator{
  ADD,        ///< operator +
  SUB,        ///< operator -
  MUL,        ///< operator *
  DIV,        ///< operator / using common type of lhs and rhs
  TRUE_DIV,   ///< operator / after promoting type to floating point
  FLOOR_DIV,  ///< operator / after promoting to float and then flooring the
              ///< result
  MOD,        ///< operator %
  PYMOD,      ///< operator % but following python's sign rules for negatives
  POW,        ///< lhs ^ rhs
  EQUAL,      ///< operator ==
  NOT_EQUAL,       ///< operator !=
  LESS,            ///< operator <
  GREATER,         ///< operator >
  LESS_EQUAL,      ///< operator <=
  GREATER_EQUAL,   ///< operator >=
  BITWISE_AND,     ///< operator &
  BITWISE_OR,      ///< operator |
  BITWISE_XOR,     ///< operator ^
  LOGICAL_AND,     ///< operator &&
  LOGICAL_OR,      ///< operator ||
  COALESCE,        ///< operator x,y  x is null ? y : x
  GENERIC_BINARY,  ///< generic binary operator to be generated with input
                   ///< ptx code
  INVALID_BINARY   ///< invalid operation
};

// TODO: All these docs
// TODO: Stream and memory resource
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
 * @param op (enum) The binary operator to use
 */
std::unique_ptr<column> binary_operation(
  scalar const& lhs,
  column_view const& rhs,
  binary_operator op,
  data_type output_type,
  rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource());

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
 * @param op (enum) The binary operator to use
 */
std::unique_ptr<column> binary_operation(
  column_view const& lhs,
  scalar const& rhs,
  binary_operator op,
  data_type output_type,
  rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource());

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
 * @param op (enum) The binary operator to use
 */
std::unique_ptr<column> binary_operation(
  column_view const& lhs,
  column_view const& rhs,
  binary_operator op,
  data_type output_type,
  rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource());

/**
 * @brief Performs a binary operation between two gdf_columns using a
 * user-defined PTX function.
 *
 * Accepts a user-defined PTX function to apply between the `lhs` and `rhs`.
 *
 * The desired output type must be specified in output_type. It is assumed that
 * this output type is compatable with the output type in the PTX code.
 *
 * The output column will be allocated and it is the user's reponsibility to
 * free the device memory
 *
 * If the valid field in the gdf_column output is not nullptr, then it will be
 * filled with the bitwise AND of the valid masks of lhs and rhs gdf_columns
 *
 * @return A gdf_column as the output of the operation.
 * @param lhs (gdf_column) First operand of the operation.
 * @param rhs (gdf_column) Second operand of the operation.
 * @param ptx String containing the PTX of a binary function to apply between
 * `lhs` and `rhs`
 * @param output_type The desired output type
 */
std::unique_ptr<column> binary_operation(
  column_view const& lhs,
  column_view const& rhs,
  std::string const& ptx,
  data_type output_type,
  rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource());

} // namespace experimental
}  // namespace cudf
