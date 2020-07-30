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
#include <algorithm>
#include <cudf/ast/linearizer.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <functional>
#include <iterator>
#include "cudf/ast/operators.hpp"

namespace cudf {

namespace ast {

namespace detail {

cudf::size_type intermediate_counter::take()
{
  auto const first_missing =
    this->used_values.size() ? this->find_first_missing(0, this->used_values.size() - 1) : 0;
  this->used_values.insert(this->used_values.cbegin() + first_missing, first_missing);
  this->max_used = std::max(max_used, first_missing + 1);
  return first_missing;
}

void intermediate_counter::give(cudf::size_type value)
{
  auto const found = std::find(this->used_values.cbegin(), this->used_values.cend(), value);
  if (found != this->used_values.cend()) { this->used_values.erase(found); }
}

cudf::size_type intermediate_counter::find_first_missing(cudf::size_type start,
                                                         cudf::size_type end) const
{
  // Given a sorted container, find the smallest value not already in the container
  if (start > end) return end + 1;

  // If the value at an index is not equal to its index, it must be the missing value
  if (start != used_values.at(start)) return start;

  auto const mid = (start + end) / 2;

  // Use binary search and check the left half or right half
  // We can assume the missing value must be in the right half
  if (used_values.at(mid) == mid) return this->find_first_missing(mid + 1, end);
  // The missing value must be in the left half
  return this->find_first_missing(start, mid);
}

}  // namespace detail

cudf::size_type linearizer::visit(literal const& expr)
{
  // Track the node index
  auto const node_index = this->node_count++;
  // Resolve node type
  auto const data_type = expr.get_data_type();
  // TODO: Use scalar device view (?)
  // Push literal
  auto const literal_index = cudf::size_type(this->literals.size());
  this->literals.push_back(expr.get_value());
  // Push data reference
  auto const source = detail::device_data_reference{
    detail::device_data_reference_type::LITERAL, data_type, literal_index};
  return this->add_data_reference(source);
}

cudf::size_type linearizer::visit(column_reference const& expr)
{
  // Track the node index
  auto const node_index = this->node_count++;
  // Resolve node type
  auto const data_type = expr.get_data_type(this->table);
  // Push data reference
  auto const source = detail::device_data_reference{detail::device_data_reference_type::COLUMN,
                                                    data_type,
                                                    expr.get_column_index(),
                                                    expr.get_table_source()};
  return this->add_data_reference(source);
}

cudf::size_type linearizer::visit(operator_expression const& expr)
{
  // Track the node index
  auto const node_index = this->node_count++;
  // Visit children (operands) of this node
  auto const operand_data_reference_indices = this->visit_operands(expr.get_operands());
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
      auto const operand_source = this->get_data_references()[data_reference_index];
      if (operand_source.reference_type == detail::device_data_reference_type::INTERMEDIATE) {
        auto const intermediate_index = operand_source.data_index;
        this->intermediate_counter.give(intermediate_index);
      }
    });
  // Resolve node type
  auto const op        = expr.get_operator();
  auto const data_type = [&] {
    if (cudf::ast::is_arithmetic_operator(op)) {
      return operand_types.at(0);
    } else if (cudf::ast::is_comparator(op)) {
      return cudf::data_type(cudf::type_id::BOOL8);
    } else if (cudf::ast::is_logical_operator(op)) {
      return cudf::data_type(cudf::type_id::BOOL8);
    }
    CUDF_FAIL("An invalid AST operator was provided.");
    return cudf::data_type(cudf::type_id::EMPTY);
  }();
  // Push operator
  this->operators.push_back(op);
  // Push data reference
  auto const source = [&]() {
    if (node_index == 0) {
      // This node is the root. Output should be directed to the output column.
      // TODO: Could refactor to support output tables (multiple output columns)
      printf("USING ROOT NODE\n");
      return detail::device_data_reference{
        detail::device_data_reference_type::COLUMN, data_type, 0, table_reference::OUTPUT};
    } else {
      // This node is not the root. Output is an intermediate value.
      return detail::device_data_reference{detail::device_data_reference_type::INTERMEDIATE,
                                           data_type,
                                           this->intermediate_counter.take()};
    }
  }();
  auto const index = this->add_data_reference(source);
  // Insert source indices from all operands (sources) and this operator (destination)
  this->operator_source_indices.insert(this->operator_source_indices.end(),
                                       operand_data_reference_indices.cbegin(),
                                       operand_data_reference_indices.cend());
  this->operator_source_indices.push_back(index);
  return index;
}

cudf::data_type linearizer::get_root_data_type() const
{
  if (this->get_data_references().size() == 0) {
    return cudf::data_type(cudf::type_id::EMPTY);
  } else {
    return this->get_data_references().back().data_type;
  }
}

std::vector<cudf::size_type> linearizer::visit_operands(
  std::vector<std::reference_wrapper<const expression>> operands)
{
  auto operand_data_reference_indices = std::vector<cudf::size_type>();
  for (auto const& operand : operands) {
    auto const operand_data_reference_index = operand.get().accept(*this);
    operand_data_reference_indices.push_back(operand_data_reference_index);
  }
  return operand_data_reference_indices;
}

cudf::size_type linearizer::add_data_reference(detail::device_data_reference data_ref)
{
  // If an equivalent data reference already exists, return its index. Otherwise add this data
  // reference and return the new index.
  auto const it = std::find(this->data_references.cbegin(), this->data_references.cend(), data_ref);
  if (it != this->data_references.cend()) {
    return std::distance(this->data_references.cbegin(), it);
  } else {
    this->data_references.push_back(data_ref);
    return this->data_references.size() - 1;
  }
}

}  // namespace ast

}  // namespace cudf
