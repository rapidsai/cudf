/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cudf/ast/detail/linearizer.hpp>
#include <cudf/ast/detail/operators.hpp>
#include <cudf/ast/operators.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

namespace cudf {

namespace ast {

/**
 * @brief Enum of table references.
 *
 * This determines which table to use in cases with two tables (e.g. joins).
 *
 */
enum class table_reference {
  LEFT,   // Column index in the left table
  RIGHT,  // Column index in the right table
  OUTPUT  // Column index in the output table
};

// Forward declaration
class literal;
class column_reference;
class expression;

/**
 * @brief A literal value used in an abstract syntax tree.
 *
 */
class literal : public detail::node {
  friend class detail::linearizer;

 public:
  /**
   * @brief Construct a new literal object.
   *
   * @tparam T Numeric scalar template type.
   * @param value A numeric scalar value.
   */
  template <typename T>
  literal(cudf::numeric_scalar<T>& value) : value(cudf::get_scalar_device_view(value))
  {
  }

  /**
   * @brief Construct a new literal object.
   *
   * @tparam T Timestamp scalar template type.
   * @param value A timestamp scalar value.
   */
  template <typename T>
  literal(cudf::timestamp_scalar<T>& value) : value(cudf::get_scalar_device_view(value))
  {
  }

  /**
   * @brief Construct a new literal object.
   *
   * @tparam T Duration scalar template type.
   * @param value A duration scalar value.
   */
  template <typename T>
  literal(cudf::duration_scalar<T>& value) : value(cudf::get_scalar_device_view(value))
  {
  }

  /**
   * @brief Get the data type.
   *
   * @return cudf::data_type
   */
  cudf::data_type get_data_type() const { return get_value().type(); }

 private:
  /**
   * @brief Get the value object.
   *
   * @return cudf::detail::fixed_width_scalar_device_view_base
   */
  cudf::detail::fixed_width_scalar_device_view_base get_value() const { return value; }

  /**
   * @brief Accepts a visitor class.
   *
   * @param visitor Visitor.
   * @return cudf::size_type Index of device data reference for this instance.
   */
  cudf::size_type accept(detail::linearizer& visitor) const override;

  const cudf::detail::fixed_width_scalar_device_view_base value;
};

/**
 * @brief A node referring to data from a column in a table.
 *
 */
class column_reference : public detail::node {
  friend class detail::linearizer;

 public:
  /**
   * @brief Construct a new column reference object
   *
   * @param column_index Index of this column in the table (provided when the node is
   * evaluated).
   * @param table_source Which table to use in cases with two tables (e.g. joins).
   */
  column_reference(cudf::size_type column_index,
                   table_reference table_source = table_reference::LEFT)
    : column_index(column_index), table_source(table_source)
  {
  }

  /**
   * @brief Get the column index.
   *
   * @return cudf::size_type
   */
  cudf::size_type get_column_index() const { return column_index; }

  /**
   * @brief Get the table source.
   *
   * @return table_reference
   */
  table_reference get_table_source() const { return table_source; }

  /**
   * @brief Get the data type.
   *
   * @param table Table used to determine types.
   * @return cudf::data_type
   */
  cudf::data_type get_data_type(const table_view& table) const
  {
    return table.column(get_column_index()).type();
  }

  /**
   * @brief Get the data type.
   *
   * @param left_table Left table used to determine types.
   * @param right_table Right table used to determine types.
   * @return cudf::data_type
   */
  cudf::data_type get_data_type(const table_view& left_table, const table_view& right_table) const
  {
    const auto table = [&] {
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

 private:
  /**
   * @brief Accepts a visitor class.
   *
   * @param visitor Visitor.
   * @return cudf::size_type Index of device data reference for this instance.
   */
  cudf::size_type accept(detail::linearizer& visitor) const override;

  cudf::size_type column_index;
  table_reference table_source;
};

/**
 * @brief An expression node holds an operator and zero or more operands.
 *
 */
class expression : public detail::node {
  friend class detail::linearizer;

 public:
  /**
   * @brief Construct a new unary expression object.
   *
   * @param op Operator
   * @param input Input node (operand)
   */
  expression(ast_operator op, node const& input) : op(op), operands({input})
  {
    if (cudf::ast::detail::ast_operator_arity(op) != 1) {
      CUDF_FAIL("The provided operator is not a unary operator.");
    }
  }

  /**
   * @brief `expression` doesn't accept r-value references for expression nodes
   */
  expression(ast_operator op, node&& input) = delete;

  /**
   * @brief Construct a new binary expression object.
   *
   * @param op Operator
   * @param left Left input node (left operand)
   * @param right Right input node (right operand)
   */
  expression(ast_operator op, node const& left, node const& right) : op(op), operands({left, right})
  {
    if (cudf::ast::detail::ast_operator_arity(op) != 2) {
      CUDF_FAIL("The provided operator is not a binary operator.");
    }
  }

  /**
   * @brief `expression` doesn't accept r-value references for expression nodes
   */
  expression(ast_operator op, node&& left, node&& right) = delete;

  /**
   * @brief `expression` doesn't accept r-value references for expression nodes
   */
  expression(ast_operator op, node&& left, node const& right) = delete;

  /**
   * @brief `expression` doesn't accept r-value references for expression nodes
   */
  expression(ast_operator op, node const& left, node&& right) = delete;

  /**
   * @brief Get the operator.
   *
   * @return ast_operator
   */
  ast_operator get_operator() const { return op; }

  /**
   * @brief Get the operands.
   *
   * @return std::vector<std::reference_wrapper<const node>>
   */
  std::vector<std::reference_wrapper<const node>> get_operands() const { return operands; }

 private:
  /**
   * @brief Accepts a visitor class.
   *
   * @param visitor Visitor.
   * @return cudf::size_type Index of device data reference for this instance.
   */
  cudf::size_type accept(detail::linearizer& visitor) const override;

  const ast_operator op;
  const std::vector<std::reference_wrapper<const node>> operands;
};

}  // namespace ast

}  // namespace cudf
