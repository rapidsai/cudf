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

#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include "operators.hpp"

namespace cudf {

namespace ast {

enum class table_reference {
  LEFT,   // Column index in the left table
  RIGHT,  // Column index in the right table
  OUTPUT  // Column index in the output table
};

namespace detail {

enum class device_data_reference_type {
  COLUMN,       // A value in a table column
  LITERAL,      // A literal value
  INTERMEDIATE  // An internal temporary value
};

struct device_data_reference {
  cudf::ast::detail::device_data_reference_type reference_type;  // Source of data
  cudf::data_type data_type;                                     // Type of data
  cudf::size_type
    data_index;  // The column index of a table, index of a literal, or index of an intermediate
  cudf::ast::table_reference table_reference = cudf::ast::table_reference::LEFT;

  inline bool operator==(const device_data_reference& rhs) const
  {
    return data_index == rhs.data_index && reference_type == rhs.reference_type &&
           table_reference == rhs.table_reference;
  }
};

class intermediate_counter {
 public:
  intermediate_counter() : used_values(), max_used(0) {}
  cudf::size_type take();
  void give(cudf::size_type value);
  cudf::size_type get_max_used() const { return this->max_used; }

 private:
  cudf::size_type find_first_missing(cudf::size_type start, cudf::size_type end) const;
  std::vector<cudf::size_type> used_values;
  cudf::size_type max_used;
};

}  // namespace detail

enum class data_reference {
  COLUMN,  // A value from a column in a table
  LITERAL  // A literal value
};

struct expression_source {
  data_reference source;  // Source of data
  cudf::size_type
    data_index;  // The column index of a table, index of a literal, or index of an intermediate
  table_reference table_source = table_reference::LEFT;  // Left or right table (if applicable)
};

// Forward declaration
class abstract_visitor;
class literal;
class column_reference;
class operator_expression;

// Visitor pattern. The visitor tracks a node index as it traverses the tree.
class expression {
 public:
  virtual cudf::size_type accept(abstract_visitor& visitor) const = 0;
};

class abstract_visitor {
 public:
  virtual cudf::size_type visit(literal const& expr)             = 0;
  virtual cudf::size_type visit(column_reference const& expr)    = 0;
  virtual cudf::size_type visit(operator_expression const& expr) = 0;
};

class literal : public expression {
 public:
  literal(std::reference_wrapper<const cudf::scalar> value) : value(std::cref(value)) {}
  std::reference_wrapper<const cudf::scalar> get_value() const { return this->value; }
  cudf::size_type accept(abstract_visitor& visitor) const override { return visitor.visit(*this); }
  cudf::data_type get_data_type() const { return this->get_value().get().type(); }

 private:
  std::reference_wrapper<const cudf::scalar> value;
};

class column_reference : public expression {
 public:
  column_reference(cudf::size_type column_index,
                   table_reference table_source = table_reference::LEFT)
    : column_index(column_index), table_source(table_source)
  {
  }
  cudf::size_type get_column_index() const { return this->column_index; }
  table_reference get_table_source() const { return this->table_source; }
  cudf::data_type get_data_type(const table_view& table) const
  {
    return table.column(this->get_column_index()).type();
  }
  cudf::data_type get_data_type(const table_view& left_table, const table_view& right_table) const
  {
    const auto table = [&] {
      if (this->get_table_source() == table_reference::LEFT) {
        return left_table;
      } else if (this->get_table_source() == table_reference::RIGHT) {
        return right_table;
      } else {
        CUDF_FAIL("Column reference data type cannot be determined from unknown table.");
      }
    }();
    return table.column(this->get_column_index()).type();
  }

  cudf::size_type accept(abstract_visitor& visitor) const override { return visitor.visit(*this); }

 private:
  cudf::size_type column_index;
  table_reference table_source;
};

class operator_expression : public expression {
 public:
  ast_operator get_operator() const { return this->op; }
  virtual std::vector<std::reference_wrapper<const expression>> get_operands() const = 0;

 protected:
  operator_expression(ast_operator op) : op(op) {}
  const ast_operator op;
};

class binary_expression : public operator_expression {
 public:
  binary_expression(ast_operator op,
                    std::reference_wrapper<const expression> left,
                    std::reference_wrapper<const expression> right)
    : operator_expression(op), left(left), right(right)
  {
  }
  std::reference_wrapper<const expression> get_left() const { return this->left; }
  std::reference_wrapper<const expression> get_right() const { return this->right; }
  std::vector<std::reference_wrapper<const expression>> get_operands() const override
  {
    return std::vector<std::reference_wrapper<const expression>>{this->get_left(),
                                                                 this->get_right()};
  }
  cudf::size_type accept(abstract_visitor& visitor) const override { return visitor.visit(*this); }

 private:
  std::reference_wrapper<const expression> left;
  std::reference_wrapper<const expression> right;
};

class linearizer : public abstract_visitor {
 public:
  linearizer(cudf::table_view table) : table(table), node_index(0), intermediate_counter() {}

  cudf::size_type visit(literal const& expr) override;
  cudf::size_type visit(column_reference const& expr) override;
  cudf::size_type visit(operator_expression const& expr) override;
  cudf::data_type get_root_data_type() const;
  cudf::size_type get_intermediate_count() const
  {
    return this->intermediate_counter.get_max_used();
  }
  std::vector<detail::device_data_reference> get_data_references() const
  {
    return this->data_references;
  }
  std::vector<ast_operator> get_operators() const { return this->operators; }
  std::vector<cudf::size_type> get_operator_source_indices() const
  {
    return this->operator_source_indices;
  }
  std::vector<std::reference_wrapper<const scalar>> get_literals() const { return this->literals; }

 private:
  std::vector<cudf::size_type> visit_operands(
    std::vector<std::reference_wrapper<const expression>> operands);
  cudf::size_type add_data_reference(detail::device_data_reference data_ref);

  // State information about the "linearized" GPU execution plan
  cudf::table_view table;
  cudf::size_type node_index;
  detail::intermediate_counter intermediate_counter;
  std::vector<detail::device_data_reference> data_references;
  std::vector<ast_operator> operators;
  std::vector<cudf::size_type> operator_source_indices;
  std::vector<std::reference_wrapper<const scalar>> literals;
};

}  // namespace ast

}  // namespace cudf
