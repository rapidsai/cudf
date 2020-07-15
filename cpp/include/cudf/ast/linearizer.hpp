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

#include <thrust/detail/raw_pointer_cast.h>
#include <algorithm>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include <functional>
#include <iterator>
#include <rmm/device_uvector.hpp>
#include <type_traits>
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
  device_data_reference_type reference_type;  // Source of data
  cudf::data_type data_type;                  // Type of data
  cudf::size_type
    data_index;  // The column index of a table, index of a literal, or index of an intermediate
  table_reference table_reference = table_reference::LEFT;

  inline bool operator==(const device_data_reference& rhs) const
  {
    return data_index == rhs.data_index && reference_type == rhs.reference_type &&
           table_reference == rhs.table_reference;
  }
};

class intermediate_counter {
 public:
  intermediate_counter() : used_values(), max_used(0) {}
  cudf::size_type take()
  {
    auto first_missing =
      this->used_values.size() ? this->find_first_missing(0, this->used_values.size() - 1) : 0;
    this->used_values.insert(this->used_values.cbegin() + first_missing, first_missing);
    this->max_used = std::max(max_used, first_missing + 1);
    return first_missing;
  }
  void give(cudf::size_type value)
  {
    auto found = std::find(this->used_values.cbegin(), this->used_values.cend(), value);
    if (found != this->used_values.cend()) { this->used_values.erase(found); }
  }
  cudf::size_type get_max_used() const { return this->max_used; }

 private:
  cudf::size_type find_first_missing(cudf::size_type start, cudf::size_type end) const
  {
    // Given a sorted container, find the smallest value not already in the container
    if (start > end) return end + 1;

    // If the value at an index is not equal to its index, it must be the missing value
    if (start != used_values.at(start)) return start;

    auto mid = (start + end) / 2;

    // Use binary search and check the left half or right half
    // We can assume the missing value must be in the right half
    if (used_values.at(mid) == mid) return this->find_first_missing(mid + 1, end);
    // The missing value must be in the left half
    return this->find_first_missing(start, mid);
  }

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
  linearizer(cudf::table_view table) : table(table), intermediate_counter() {}

  cudf::size_type visit(literal const& expr) override
  {
    // Resolve node type
    auto data_type = expr.get_data_type();
    // TODO: Use scalar device view (?)
    // Push literal
    auto literal_index = cudf::size_type(this->literals.size());
    this->literals.push_back(expr.get_value());
    // Push data reference
    auto source = detail::device_data_reference{
      detail::device_data_reference_type::LITERAL, data_type, literal_index};
    return this->add_data_reference(source);
  }

  cudf::size_type visit(column_reference const& expr) override
  {
    // Resolve node type
    auto data_type = expr.get_data_type(this->table);
    // Push data reference
    auto source = detail::device_data_reference{detail::device_data_reference_type::COLUMN,
                                                data_type,
                                                expr.get_column_index(),
                                                expr.get_table_source()};
    return this->add_data_reference(source);
  }

  cudf::size_type visit(operator_expression const& expr) override
  {
    const auto op                       = expr.get_operator();
    auto operand_data_reference_indices = this->visit_operands(expr.get_operands());
    // Resolve operand types
    auto operand_types = std::vector<cudf::data_type>();
    std::transform(operand_data_reference_indices.cbegin(),
                   operand_data_reference_indices.cend(),
                   std::back_inserter(operand_types),
                   [this](auto const& data_reference_index) -> cudf::data_type {
                     return this->get_data_references()[data_reference_index].data_type;
                   });
    // Validate types of operand data references match
    if (std::adjacent_find(operand_types.cbegin(), operand_types.cend(), std::not_equal_to<>()) !=
        operand_types.cend()) {
      CUDF_FAIL("An AST operator expression was provided non-matching operand types.");
    }
    // Give back intermediate storage locations that are consumed by this operation
    std::for_each(
      operand_data_reference_indices.cbegin(),
      operand_data_reference_indices.cend(),
      [this](auto const& data_reference_index) {
        auto operand_source = this->get_data_references()[data_reference_index];
        if (operand_source.reference_type == detail::device_data_reference_type::INTERMEDIATE) {
          auto intermediate_index = operand_source.data_index;
          this->intermediate_counter.give(intermediate_index);
        }
      });
    // Resolve node type
    auto data_type = operand_types.at(0);
    /* TODO: Need to fix. Can't support comparators yet.
    auto data_type = [&] {
      const ast_operator oper = op;
      if (cudf::ast::is_arithmetic_operator<oper>()) {
        return operand_types.at(0);
      } else {
        return cudf::data_type(cudf::type_id::EMPTY);
      }
    }();
    */
    // Push operator
    this->operators.push_back(op);
    // Push data reference
    auto source = detail::device_data_reference{detail::device_data_reference_type::INTERMEDIATE,
                                                data_type,
                                                this->intermediate_counter.take()};
    auto index  = this->add_data_reference(source);
    // Insert source indices from all operands (sources) and operator (destination)
    this->operator_source_indices.insert(this->operator_source_indices.end(),
                                         operand_data_reference_indices.cbegin(),
                                         operand_data_reference_indices.cend());
    this->operator_source_indices.push_back(index);
    return index;
  }

  cudf::data_type get_root_data_type() const
  {
    if (this->get_data_references().size() == 0) {
      return cudf::data_type(cudf::type_id::EMPTY);
    } else {
      return this->get_data_references().back().data_type;
    }
  }
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
    std::vector<std::reference_wrapper<const expression>> operands)
  {
    auto operand_data_reference_indices = std::vector<cudf::size_type>();
    for (auto& operand : operands) {
      auto operand_data_reference_index = operand.get().accept(*this);
      operand_data_reference_indices.push_back(operand_data_reference_index);
    }
    return operand_data_reference_indices;
  }
  cudf::size_type add_data_reference(detail::device_data_reference data_ref)
  {
    // If an equivalent data reference already exists, return its index. Otherwise add this data
    // reference and return the new index.
    auto it = std::find(this->data_references.cbegin(), this->data_references.cend(), data_ref);
    if (it != this->data_references.cend()) {
      return std::distance(this->data_references.cbegin(), it);
    } else {
      this->data_references.push_back(data_ref);
      return this->data_references.size() - 1;
    }
  }
  // State information about the "linearized" GPU execution plan
  cudf::table_view table;
  detail::intermediate_counter intermediate_counter;
  std::vector<detail::device_data_reference> data_references;
  std::vector<ast_operator> operators;
  std::vector<cudf::size_type> operator_source_indices;
  std::vector<std::reference_wrapper<const scalar>> literals;
};

}  // namespace ast

}  // namespace cudf
