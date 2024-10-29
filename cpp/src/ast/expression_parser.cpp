/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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
#include <cudf/ast/detail/expression_parser.hpp>
#include <cudf/ast/detail/operators.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <thrust/iterator/transform_iterator.h>

#include <algorithm>
#include <functional>
#include <iterator>

namespace cudf {

namespace ast {

namespace detail {

device_data_reference::device_data_reference(device_data_reference_type reference_type,
                                             cudf::data_type data_type,
                                             cudf::size_type data_index,
                                             table_reference table_source)
  : reference_type(reference_type),
    data_type(data_type),
    data_index(data_index),
    table_source(table_source)
{
}

device_data_reference::device_data_reference(device_data_reference_type reference_type,
                                             cudf::data_type data_type,
                                             cudf::size_type data_index)
  : reference_type(reference_type),
    data_type(data_type),
    data_index(data_index),
    table_source(table_reference::LEFT)
{
}

cudf::size_type expression_parser::intermediate_counter::take()
{
  auto const first_missing = find_first_missing();
  used_values.insert(used_values.cbegin() + first_missing, first_missing);
  max_used = std::max(max_used, first_missing + 1);
  return first_missing;
}

void expression_parser::intermediate_counter::give(cudf::size_type value)
{
  // TODO: add comment
  auto const lower_bound = std::lower_bound(used_values.cbegin(), used_values.cend(), value);
  if ((lower_bound != used_values.cend()) && (*lower_bound == value))
    used_values.erase(lower_bound);
}

cudf::size_type expression_parser::intermediate_counter::find_first_missing() const
{
  if (used_values.empty() || (used_values.front() != 0)) { return 0; }
  // Search for the first non-contiguous pair of elements.
  auto diff_not_one = [](auto a, auto b) { return a != b - 1; };
  auto it           = std::adjacent_find(used_values.cbegin(), used_values.cend(), diff_not_one);
  return it != used_values.cend()
           ? *it + 1              // A missing value was found and is returned.
           : used_values.size();  // No missing elements. Return the next element in the sequence.
}

cudf::size_type expression_parser::visit(literal const& expr)
{
  if (_expression_count == 0) {
    // Handle the trivial case of a literal as the entire expression.
    return visit(operation(ast_operator::IDENTITY, expr));
  } else {
    _expression_count++;                                           // Increment the expression index
    auto const data_type     = expr.get_data_type();               // Resolve expression type
    auto device_view         = expr.get_value();                   // Construct a scalar device view
    auto const literal_index = cudf::size_type(_literals.size());  // Push literal
    _literals.push_back(device_view);
    auto const source = detail::device_data_reference(detail::device_data_reference_type::LITERAL,
                                                      data_type,
                                                      literal_index);  // Push data reference
    return add_data_reference(source);
  }
}

cudf::size_type expression_parser::visit(column_reference const& expr)
{
  if (_expression_count == 0) {
    // Handle the trivial case of a column reference as the entire expression.
    return visit(operation(ast_operator::IDENTITY, expr));
  } else {
    // Increment the expression index
    _expression_count++;
    // Resolve expression type
    cudf::data_type data_type;
    if (expr.get_table_source() == table_reference::LEFT) {
      data_type = expr.get_data_type(_left);
    } else {
      if (_right.has_value()) {
        data_type = expr.get_data_type(*_right);
      } else {
        CUDF_FAIL(
          "Your expression contains a reference to the RIGHT table even though it will only be "
          "evaluated on a single table (by convention, the LEFT table).");
      }
    }
    // Push data reference
    auto const source = detail::device_data_reference(detail::device_data_reference_type::COLUMN,
                                                      data_type,
                                                      expr.get_column_index(),
                                                      expr.get_table_source());
    return add_data_reference(source);
  }
}

cudf::size_type expression_parser::visit(operation const& expr)
{
  // Increment the expression index
  auto const expression_index = _expression_count++;
  // Visit children (operands) of this expression
  auto const operand_data_ref_indices = visit_operands(expr.get_operands());
  // Resolve operand types
  auto data_ref = [this](auto const& index) { return _data_references[index].data_type; };
  auto begin    = thrust::make_transform_iterator(operand_data_ref_indices.cbegin(), data_ref);
  auto end      = begin + operand_data_ref_indices.size();
  auto const operand_types = std::vector<cudf::data_type>(begin, end);

  // Validate types of operand data references match
  if (std::adjacent_find(operand_types.cbegin(), operand_types.cend(), std::not_equal_to<>()) !=
      operand_types.cend()) {
    CUDF_FAIL("An AST expression was provided non-matching operand types.");
  }

  // Give back intermediate storage locations that are consumed by this operation
  std::for_each(
    operand_data_ref_indices.cbegin(),
    operand_data_ref_indices.cend(),
    [this](auto const& data_reference_index) {
      auto const operand_source = _data_references[data_reference_index];
      if (operand_source.reference_type == detail::device_data_reference_type::INTERMEDIATE) {
        auto const intermediate_index = operand_source.data_index;
        _intermediate_counter.give(intermediate_index);
      }
    });
  // Resolve expression type
  auto const op        = expr.get_operator();
  auto const data_type = cudf::ast::detail::ast_operator_return_type(op, operand_types);
  _operators.push_back(op);
  // Push data reference
  auto const output = [&]() {
    if (expression_index == 0) {
      // This expression is the root. Output should be directed to the output column.
      return detail::device_data_reference(
        detail::device_data_reference_type::COLUMN, data_type, 0, table_reference::OUTPUT);
    } else {
      // This expression is not the root. Output is an intermediate value.
      // Ensure that the output type is fixed width and fits in the intermediate storage.
      if (!cudf::is_fixed_width(data_type)) {
        CUDF_FAIL(
          "The output data type is not a fixed-width type but must be stored in an intermediate.");
      } else if (cudf::size_of(data_type) > (_has_nulls ? sizeof(IntermediateDataType<true>)
                                                        : sizeof(IntermediateDataType<false>))) {
        CUDF_FAIL("The output data type is too large to be stored in an intermediate.");
      }
      return detail::device_data_reference(
        detail::device_data_reference_type::INTERMEDIATE, data_type, _intermediate_counter.take());
    }
  }();
  auto const index = add_data_reference(output);
  // Insert source indices from all operands (sources) and this operator (destination)
  _operator_source_indices.insert(_operator_source_indices.end(),
                                  operand_data_ref_indices.cbegin(),
                                  operand_data_ref_indices.cend());
  _operator_source_indices.push_back(index);
  return index;
}

// TODO: Eliminate column name references from expression_parser because
// 2 code paths diverge in supporting column name references:
// 1. column name references are specific to cuIO
// 2. column name references are not supported in the libcudf table operations such as join,
// transform.
cudf::size_type expression_parser::visit(column_name_reference const& expr)
{
  CUDF_FAIL("Column name references are not supported in the AST expression parser.");
}

cudf::data_type expression_parser::output_type() const
{
  return _data_references.empty() ? cudf::data_type(cudf::type_id::EMPTY)
                                  : _data_references.back().data_type;
}

std::vector<cudf::size_type> expression_parser::visit_operands(
  cudf::host_span<std::reference_wrapper<expression const> const> operands)
{
  auto operand_data_reference_indices = std::vector<cudf::size_type>();
  for (auto const& operand : operands) {
    auto const operand_data_reference_index = operand.get().accept(*this);
    operand_data_reference_indices.push_back(operand_data_reference_index);
  }
  return operand_data_reference_indices;
}

cudf::size_type expression_parser::add_data_reference(detail::device_data_reference data_ref)
{
  // If an equivalent data reference already exists, return its index. Otherwise add this data
  // reference and return the new index.
  auto const it = std::find(_data_references.cbegin(), _data_references.cend(), data_ref);
  if (it != _data_references.cend()) {
    return std::distance(_data_references.cbegin(), it);
  } else {
    _data_references.push_back(data_ref);
    return _data_references.size() - 1;
  }
}

}  // namespace detail

}  // namespace ast

}  // namespace cudf
