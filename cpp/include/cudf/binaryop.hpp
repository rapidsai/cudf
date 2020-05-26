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

#include <cudf/column/column.hpp>
#include <cudf/scalar/scalar.hpp>

#include <memory>

namespace cudf {

/**
 * @addtogroup transformation_binaryops
 * @{
 */

/**
 * @brief Types of binary operations that can be performed on data.
 */
enum class binary_operator : int32_t {
  ADD,                   ///< operator +
  SUB,                   ///< operator -
  MUL,                   ///< operator *
  DIV,                   ///< operator / using common type of lhs and rhs
  TRUE_DIV,              ///< operator / after promoting type to floating point
  FLOOR_DIV,             ///< operator / after promoting to 64 bit floating point and then
                         ///< flooring the result
  MOD,                   ///< operator %
  PYMOD,                 ///< operator % but following python's sign rules for negatives
  POW,                   ///< lhs ^ rhs
  EQUAL,                 ///< operator ==
  NOT_EQUAL,             ///< operator !=
  LESS,                  ///< operator <
  GREATER,               ///< operator >
  LESS_EQUAL,            ///< operator <=
  GREATER_EQUAL,         ///< operator >=
  BITWISE_AND,           ///< operator &
  BITWISE_OR,            ///< operator |
  BITWISE_XOR,           ///< operator ^
  LOGICAL_AND,           ///< operator &&
  LOGICAL_OR,            ///< operator ||
  COALESCE,              ///< operator x,y  x is null ? y : x
  GENERIC_BINARY,        ///< generic binary operator to be generated with input
                         ///< ptx code
  SHIFT_LEFT,            ///< operator <<
  SHIFT_RIGHT,           ///< operator >>
  SHIFT_RIGHT_UNSIGNED,  ///< operator >>> (from Java)
                         ///< Logical right shift. Casts to an unsigned value before shifting.
  LOG_BASE,              ///< logarithm to the base
  ATAN2,                 ///< 2-argument arctangent
  PMOD,                  ///< positive modulo operator
                         ///< If remainder is negative, this returns (remainder + divisor) % divisor
                         ///< else, it returns (dividend % divisor)
  NULL_EQUALS,           ///< Returns true when both operands are null; false when one is null; the
                         ///< result of equality when both are non-null
  NULL_MAX,              ///< Returns max of operands when both are non-null; returns the non-null
                         ///< operand when one is null; or invalid when both are null
  NULL_MIN,              ///< Returns min of operands when both are non-null; returns the non-null
                         ///< operand when one is null; or invalid when both are null
  INVALID_BINARY         ///< invalid operation
};
/**
 * @brief Performs a binary operation between a scalar and a column.
 *
 * The output contains the result of `op(lhs, rhs[i])` for all `0 <= i < rhs.size()`
 * The scalar is the left operand and the column elements are the right operand.
 * This distinction is significant in case of non-commutative binary operations
 *
 * Regardless of the operator, the validity of the output value is the logical
 * AND of the validity of the two operands
 *
 * @param lhs         The left operand scalar
 * @param rhs         The right operand column
 * @param output_type The desired data type of the output column
 * @param mr          Device memory resource used to allocate the returned column's device memory
 * @return            Output column of `output_type` type containing the result of
 *                    the binary operation
 * @throw cudf::logic_error if @p output_type dtype isn't fixed-width
 */
std::unique_ptr<column> binary_operation(
  scalar const& lhs,
  column_view const& rhs,
  binary_operator op,
  data_type output_type,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Performs a binary operation between a column and a scalar.
 *
 * The output contains the result of `op(lhs[i], rhs)` for all `0 <= i < lhs.size()`
 * The column elements are the left operand and the scalar is the right operand.
 * This distinction is significant in case of non-commutative binary operations
 *
 * Regardless of the operator, the validity of the output value is the logical
 * AND of the validity of the two operands
 *
 * @param lhs         The left operand column
 * @param rhs         The right operand scalar
 * @param output_type The desired data type of the output column
 * @param mr          Device memory resource used to allocate the returned column's device memory
 * @return            Output column of `output_type` type containing the result of
 *                    the binary operation
 * @throw cudf::logic_error if @p output_type dtype isn't fixed-width
 */
std::unique_ptr<column> binary_operation(
  column_view const& lhs,
  scalar const& rhs,
  binary_operator op,
  data_type output_type,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Performs a binary operation between two columns.
 *
 * The output contains the result of `op(lhs[i], rhs[i])` for all `0 <= i < lhs.size()`
 *
 * Regardless of the operator, the validity of the output value is the logical
 * AND of the validity of the two operands
 *
 * @param lhs         The left operand column
 * @param rhs         The right operand column
 * @param output_type The desired data type of the output column
 * @param mr          Device memory resource used to allocate the returned column's device memory
 * @return            Output column of `output_type` type containing the result of
 *                    the binary operation
 * @throw cudf::logic_error if @p lhs and @p rhs are different sizes
 * @throw cudf::logic_error if @p output_type dtype isn't fixed-width
 */
std::unique_ptr<column> binary_operation(
  column_view const& lhs,
  column_view const& rhs,
  binary_operator op,
  data_type output_type,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Performs a binary operation between two columns using a
 * user-defined PTX function.
 *
 * The output contains the result of `op(lhs[i], rhs[i])` for all `0 <= i < lhs.size()`
 *
 * Regardless of the operator, the validity of the output value is the logical
 * AND of the validity of the two operands
 *
 * @param lhs         The left operand column
 * @param rhs         The right operand column
 * @param ptx         String containing the PTX of a binary function
 * @param output_type The desired data type of the output column. It is assumed
 *                    that output_type is compatible with the output data type
 *                    of the function in the PTX code
 * @param mr          Device memory resource used to allocate the returned column's device memory
 * @return            Output column of `output_type` type containing the result of
 *                    the binary operation
 * @throw cudf::logic_error if @p lhs and @p rhs are different sizes
 * @throw cudf::logic_error if @p lhs and @p rhs dtypes aren't numeric
 * @throw cudf::logic_error if @p output_type dtype isn't numeric
 */
std::unique_ptr<column> binary_operation(
  column_view const& lhs,
  column_view const& rhs,
  std::string const& ptx,
  data_type output_type,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/** @} */  // end of group
}  // namespace cudf
