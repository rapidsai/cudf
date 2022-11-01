/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <cstdint>

namespace cudf {
namespace ast {

// Forward declaration.
namespace detail {
class expression_parser;
}

/**
 * @brief A generic expression that can be evaluated to return a value.
 *
 * This class is a part of a "visitor" pattern with the `expression_parser` class.
 * Expressions inheriting from this class can accept parsers as visitors.
 */
struct expression {
  /**
   * @brief Accepts a visitor class.
   *
   * @param visitor The `expression_parser` parsing this expression tree
   * @return Index of device data reference for this instance
   */
  virtual cudf::size_type accept(detail::expression_parser& visitor) const = 0;

  /**
   * @brief Returns true if the expression may evaluate to null.
   *
   * @param left The left operand of the expression (The same is used as right operand)
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @return `true` if the expression may evaluate to null, otherwise false
   */
  [[nodiscard]] bool may_evaluate_null(table_view const& left, rmm::cuda_stream_view stream) const
  {
    return may_evaluate_null(left, left, stream);
  }

  /**
   * @brief Returns true if the expression may evaluate to null.
   *
   * @param left The left operand of the expression
   * @param right The right operand of the expression
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @return `true` if the expression may evaluate to null, otherwise false
   */
  [[nodiscard]] virtual bool may_evaluate_null(table_view const& left,
                                               table_view const& right,
                                               rmm::cuda_stream_view stream) const = 0;

  virtual ~expression() {}
};

/**
 * @brief Enum of supported operators.
 */
enum class ast_operator : int32_t {
  // Binary operators
  ADD,         ///< operator +
  SUB,         ///< operator -
  MUL,         ///< operator *
  DIV,         ///< operator / using common type of lhs and rhs
  TRUE_DIV,    ///< operator / after promoting type to floating point
  FLOOR_DIV,   ///< operator / after promoting to 64 bit floating point and then
               ///< flooring the result
  MOD,         ///< operator %
  PYMOD,       ///< operator % using Python's sign rules for negatives
  POW,         ///< lhs ^ rhs
  EQUAL,       ///< operator ==
  NULL_EQUAL,  ///< operator == with Spark rules: NULL_EQUAL(null, null) is true, NULL_EQUAL(null,
               ///< valid) is false, and
               ///< NULL_EQUAL(valid, valid) == EQUAL(valid, valid)
  NOT_EQUAL,   ///< operator !=
  LESS,        ///< operator <
  GREATER,     ///< operator >
  LESS_EQUAL,  ///< operator <=
  GREATER_EQUAL,     ///< operator >=
  BITWISE_AND,       ///< operator &
  BITWISE_OR,        ///< operator |
  BITWISE_XOR,       ///< operator ^
  LOGICAL_AND,       ///< operator &&
  NULL_LOGICAL_AND,  ///< operator && with Spark rules: NULL_LOGICAL_AND(null, null) is null,
                     ///< NULL_LOGICAL_AND(null, true) is
                     ///< null, NULL_LOGICAL_AND(null, false) is false, and NULL_LOGICAL_AND(valid,
                     ///< valid) == LOGICAL_AND(valid, valid)
  LOGICAL_OR,        ///< operator ||
  NULL_LOGICAL_OR,   ///< operator || with Spark rules: NULL_LOGICAL_OR(null, null) is null,
                     ///< NULL_LOGICAL_OR(null, true) is true,
                     ///< NULL_LOGICAL_OR(null, false) is null, and NULL_LOGICAL_OR(valid, valid) ==
                     ///< LOGICAL_OR(valid, valid)
  // Unary operators
  IDENTITY,        ///< Identity function
  SIN,             ///< Trigonometric sine
  COS,             ///< Trigonometric cosine
  TAN,             ///< Trigonometric tangent
  ARCSIN,          ///< Trigonometric sine inverse
  ARCCOS,          ///< Trigonometric cosine inverse
  ARCTAN,          ///< Trigonometric tangent inverse
  SINH,            ///< Hyperbolic sine
  COSH,            ///< Hyperbolic cosine
  TANH,            ///< Hyperbolic tangent
  ARCSINH,         ///< Hyperbolic sine inverse
  ARCCOSH,         ///< Hyperbolic cosine inverse
  ARCTANH,         ///< Hyperbolic tangent inverse
  EXP,             ///< Exponential (base e, Euler number)
  LOG,             ///< Natural Logarithm (base e)
  SQRT,            ///< Square-root (x^0.5)
  CBRT,            ///< Cube-root (x^(1.0/3))
  CEIL,            ///< Smallest integer value not less than arg
  FLOOR,           ///< largest integer value not greater than arg
  ABS,             ///< Absolute value
  RINT,            ///< Rounds the floating-point argument arg to an integer value
  BIT_INVERT,      ///< Bitwise Not (~)
  NOT,             ///< Logical Not (!)
  CAST_TO_INT64,   ///< Cast value to int64_t
  CAST_TO_UINT64,  ///< Cast value to uint64_t
  CAST_TO_FLOAT64  ///< Cast value to double
};

/**
 * @brief Enum of table references.
 *
 * This determines which table to use in cases with two tables (e.g. joins).
 */
enum class table_reference {
  LEFT,   ///< Column index in the left table
  RIGHT,  ///< Column index in the right table
  OUTPUT  ///< Column index in the output table
};

/**
 * @brief A literal value used in an abstract syntax tree.
 */
class literal : public expression {
 public:
  /**
   * @brief Construct a new literal object.
   *
   * @tparam T Numeric scalar template type
   * @param value A numeric scalar value
   */
  template <typename T>
  literal(cudf::numeric_scalar<T>& value)
    : scalar(value), value(cudf::get_scalar_device_view(value))
  {
  }

  /**
   * @brief Construct a new literal object.
   *
   * @tparam T Timestamp scalar template type
   * @param value A timestamp scalar value
   */
  template <typename T>
  literal(cudf::timestamp_scalar<T>& value)
    : scalar(value), value(cudf::get_scalar_device_view(value))
  {
  }

  /**
   * @brief Construct a new literal object.
   *
   * @tparam T Duration scalar template type
   * @param value A duration scalar value
   */
  template <typename T>
  literal(cudf::duration_scalar<T>& value)
    : scalar(value), value(cudf::get_scalar_device_view(value))
  {
  }

  /**
   * @brief Get the data type.
   *
   * @return The data type of the literal
   */
  [[nodiscard]] cudf::data_type get_data_type() const { return get_value().type(); }

  /**
   * @brief Get the value object.
   *
   * @return The device scalar object
   */
  [[nodiscard]] cudf::detail::fixed_width_scalar_device_view_base get_value() const
  {
    return value;
  }

  /**
   * @brief Accepts a visitor class.
   *
   * @param visitor The `expression_parser` parsing this expression tree
   * @return Index of device data reference for this instance
   */
  cudf::size_type accept(detail::expression_parser& visitor) const override;

  [[nodiscard]] bool may_evaluate_null(table_view const& left,
                                       table_view const& right,
                                       rmm::cuda_stream_view stream) const override
  {
    return !is_valid(stream);
  }

  /**
   * @brief Check if the underlying scalar is valid.
   *
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @return true if the underlying scalar is valid
   */
  [[nodiscard]] bool is_valid(rmm::cuda_stream_view stream) const
  {
    return scalar.is_valid(stream);
  }

 private:
  cudf::scalar const& scalar;
  cudf::detail::fixed_width_scalar_device_view_base const value;
};

/**
 * @brief A expression referring to data from a column in a table.
 */
class column_reference : public expression {
 public:
  /**
   * @brief Construct a new column reference object
   *
   * @param column_index Index of this column in the table (provided when the expression is
   * evaluated).
   * @param table_source Which table to use in cases with two tables (e.g. joins)
   */
  column_reference(cudf::size_type column_index,
                   table_reference table_source = table_reference::LEFT)
    : column_index(column_index), table_source(table_source)
  {
  }

  /**
   * @brief Get the column index.
   *
   * @return The column index of the column reference
   */
  [[nodiscard]] cudf::size_type get_column_index() const { return column_index; }

  /**
   * @brief Get the table source.
   *
   * @return table_reference The reference to the table containing this column
   */
  [[nodiscard]] table_reference get_table_source() const { return table_source; }

  /**
   * @brief Get the data type.
   *
   * @param table Table used to determine types
   * @return The data type of the column
   */
  [[nodiscard]] cudf::data_type get_data_type(table_view const& table) const
  {
    return table.column(get_column_index()).type();
  }

  /**
   * @brief Get the data type.
   *
   * @param left_table Left table used to determine types
   * @param right_table Right table used to determine types
   * @return The data type of the column
   */
  [[nodiscard]] cudf::data_type get_data_type(table_view const& left_table,
                                              table_view const& right_table) const
  {
    auto const table = [&] {
      if (get_table_source() == table_reference::LEFT) {
        return left_table;
      } else if (get_table_source() == table_reference::RIGHT) {
        return right_table;
      } else {
        CUDF_FAIL("Column reference data type cannot be determined from unknown table.");
      }
    }();
    return table.column(get_column_index()).type();
  }

  /**
   * @brief Accepts a visitor class.
   *
   * @param visitor The `expression_parser` parsing this expression tree
   * @return Index of device data reference for this instance
   */
  cudf::size_type accept(detail::expression_parser& visitor) const override;

  [[nodiscard]] bool may_evaluate_null(table_view const& left,
                                       table_view const& right,
                                       rmm::cuda_stream_view stream) const override
  {
    return (table_source == table_reference::LEFT ? left : right).column(column_index).has_nulls();
  }

 private:
  cudf::size_type column_index;
  table_reference table_source;
};

/**
 * @brief An operation expression holds an operator and zero or more operands.
 */
class operation : public expression {
 public:
  /**
   * @brief Construct a new unary operation object.
   *
   * @param op Operator
   * @param input Input expression (operand)
   */
  operation(ast_operator op, expression const& input);

  /**
   * @brief Construct a new binary operation object.
   *
   * @param op Operator
   * @param left Left input expression (left operand)
   * @param right Right input expression (right operand)
   */
  operation(ast_operator op, expression const& left, expression const& right);

  // operation only stores references to expressions, so it does not accept r-value
  // references: the calling code must own the expressions.
  operation(ast_operator op, expression&& input)                         = delete;
  operation(ast_operator op, expression&& left, expression&& right)      = delete;
  operation(ast_operator op, expression&& left, expression const& right) = delete;
  operation(ast_operator op, expression const& left, expression&& right) = delete;

  /**
   * @brief Get the operator.
   *
   * @return The operator
   */
  [[nodiscard]] ast_operator get_operator() const { return op; }

  /**
   * @brief Get the operands.
   *
   * @return Vector of operands
   */
  std::vector<std::reference_wrapper<expression const>> get_operands() const { return operands; }

  /**
   * @brief Accepts a visitor class.
   *
   * @param visitor The `expression_parser` parsing this expression tree
   * @return Index of device data reference for this instance
   */
  cudf::size_type accept(detail::expression_parser& visitor) const override;

  [[nodiscard]] bool may_evaluate_null(table_view const& left,
                                       table_view const& right,
                                       rmm::cuda_stream_view stream) const override
  {
    return std::any_of(operands.cbegin(),
                       operands.cend(),
                       [&left, &right, &stream](std::reference_wrapper<expression const> subexpr) {
                         return subexpr.get().may_evaluate_null(left, right, stream);
                       });
  };

 private:
  ast_operator const op;
  std::vector<std::reference_wrapper<expression const>> const operands;
};

}  // namespace ast

}  // namespace cudf
