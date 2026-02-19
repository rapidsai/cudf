/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "expression_transform_helpers.hpp"

#include <cudf/ast/detail/expression_transformer.hpp>
#include <cudf/ast/detail/operators.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/transform.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/bit.hpp>

#include <thrust/iterator/counting_iterator.h>

namespace cudf::io::parquet::detail {

named_to_reference_converter::named_to_reference_converter(
  std::optional<std::reference_wrapper<ast::expression const>> expr, table_metadata const& metadata)
{
  if (!expr.has_value()) { return; }
  // create map for column name.
  std::transform(metadata.schema_info.cbegin(),
                 metadata.schema_info.cend(),
                 thrust::counting_iterator<size_t>(0),
                 std::inserter(_column_name_to_index, _column_name_to_index.end()),
                 [](auto const& sch, auto index) { return std::make_pair(sch.name, index); });

  expr.value().get().accept(*this);
}

std::reference_wrapper<ast::expression const> named_to_reference_converter::visit(
  ast::literal const& expr)
{
  _converted_expr = std::reference_wrapper<ast::expression const>(expr);
  return expr;
}

std::reference_wrapper<ast::expression const> named_to_reference_converter::visit(
  ast::column_reference const& expr)
{
  _converted_expr = std::reference_wrapper<ast::expression const>(expr);
  return expr;
}

std::reference_wrapper<ast::expression const> named_to_reference_converter::visit(
  ast::column_name_reference const& expr)
{
  // check if column name is in metadata
  auto col_index_it = _column_name_to_index.find(expr.get_column_name());
  if (col_index_it == _column_name_to_index.end()) {
    CUDF_FAIL("Column name not found in metadata");
  }
  auto col_index = col_index_it->second;
  _col_ref.emplace_back(col_index);
  _converted_expr = std::reference_wrapper<ast::expression const>(_col_ref.back());
  return std::reference_wrapper<ast::expression const>(_col_ref.back());
}

std::reference_wrapper<ast::expression const> named_to_reference_converter::visit(
  ast::operation const& expr)
{
  auto const operands       = expr.get_operands();
  auto op                   = expr.get_operator();
  auto new_operands         = visit_operands(operands);
  auto const operator_arity = cudf::ast::detail::ast_operator_arity(op);
  if (operator_arity == 2) {
    _operators.emplace_back(op, new_operands.front(), new_operands.back());
  } else if (operator_arity == 1) {
    _operators.emplace_back(op, new_operands.front());
  }
  _converted_expr = std::reference_wrapper<ast::expression const>(_operators.back());
  return std::reference_wrapper<ast::expression const>(_operators.back());
}

std::vector<std::reference_wrapper<ast::expression const>>
named_to_reference_converter::visit_operands(
  cudf::host_span<std::reference_wrapper<ast::expression const> const> operands)
{
  std::vector<std::reference_wrapper<ast::expression const>> transformed_operands;
  for (auto const& operand : operands) {
    auto const new_operand = operand.get().accept(*this);
    transformed_operands.push_back(new_operand);
  }
  return transformed_operands;
}

names_from_expression::names_from_expression(
  std::optional<std::reference_wrapper<ast::expression const>> expr,
  std::vector<std::string> const& skip_names,
  cudf::io::parquet_reader_options const& options,
  std::vector<SchemaElement> const& schema_tree)
  : _skip_names(skip_names.cbegin(), skip_names.cend())
{
  if (!expr.has_value()) { return; }

  _column_indices_to_names = map_column_indices_to_names(options, schema_tree);

  expr.value().get().accept(*this);
}

std::reference_wrapper<ast::expression const> names_from_expression::visit(ast::literal const& expr)
{
  return expr;
}

std::reference_wrapper<ast::expression const> names_from_expression::visit(
  ast::column_reference const& expr)
{
  // Map the column index to its name
  auto const col_name = _column_indices_to_names[expr.get_column_index()];
  // If the column name is not in the skip_names, add it to the set
  if (_skip_names.count(col_name) == 0) { _column_names.insert(col_name); }
  return expr;
}

std::reference_wrapper<ast::expression const> names_from_expression::visit(
  ast::column_name_reference const& expr)
{
  // collect column names
  auto col_name = expr.get_column_name();
  if (_skip_names.count(col_name) == 0) { _column_names.insert(col_name); }
  return expr;
}

std::reference_wrapper<ast::expression const> names_from_expression::visit(
  ast::operation const& expr)
{
  visit_operands(expr.get_operands());
  return expr;
}

std::vector<std::string> names_from_expression::to_vector() &&
{
  return {std::make_move_iterator(_column_names.begin()),
          std::make_move_iterator(_column_names.end())};
}

void names_from_expression::visit_operands(
  cudf::host_span<std::reference_wrapper<ast::expression const> const> operands)
{
  for (auto const& operand : operands) {
    operand.get().accept(*this);
  }
}

[[nodiscard]] std::unordered_map<cudf::size_type, std::string> map_column_indices_to_names(
  cudf::io::parquet_reader_options const& options, std::vector<SchemaElement> const& schema_tree)
{
  std::unordered_map<cudf::size_type, std::string> column_indices_to_names;

  auto const& selected_columns        = options.get_column_names();
  auto const& selected_column_indices = options.get_column_indices();

  CUDF_EXPECTS(
    not(selected_columns.has_value() and selected_column_indices.has_value()),
    "Parquet reader encountered column selection by both names and indices simultaneously");

  // Map counting indices to the selected column by names
  if (selected_columns.has_value()) {
    std::transform(selected_columns->begin(),
                   selected_columns->end(),
                   thrust::counting_iterator<cudf::size_type>(0),
                   std::inserter(column_indices_to_names, column_indices_to_names.end()),
                   [](auto const& col_name, auto const col_index) {
                     return std::make_pair(col_index, col_name);
                   });
  } else {
    // Map selected top-level column indices to their names from the schema tree
    auto const& root = schema_tree.front();

    if (selected_column_indices.has_value()) {
      std::transform(selected_column_indices->begin(),
                     selected_column_indices->end(),
                     thrust::counting_iterator<cudf::size_type>(0),
                     std::inserter(column_indices_to_names, column_indices_to_names.end()),
                     [&](auto selected_col_idx, auto const mapped_col_idx) {
                       auto const schema_idx = root.children_idx[selected_col_idx];
                       return std::make_pair(mapped_col_idx, schema_tree[schema_idx].name);
                     });
    } else {
      // Map all top-level column indices to their names from the schema tree
      std::for_each(thrust::counting_iterator<int32_t>(0),
                    thrust::counting_iterator<int32_t>(root.children_idx.size()),
                    [&](auto col_idx) {
                      auto const schema_idx = root.children_idx[col_idx];
                      column_indices_to_names.insert({col_idx, schema_tree[schema_idx].name});
                    });
    }
  }

  return column_indices_to_names;
}

[[nodiscard]] std::vector<std::string> get_column_names_in_expression(
  std::optional<std::reference_wrapper<ast::expression const>> expr,
  std::vector<std::string> const& skip_names,
  cudf::io::parquet_reader_options const& options,
  std::vector<SchemaElement> const& schema_tree)
{
  return names_from_expression(expr, skip_names, options, schema_tree).to_vector();
}

std::optional<std::vector<std::vector<size_type>>> collect_filtered_row_group_indices(
  cudf::table_view table,
  std::reference_wrapper<ast::expression const> ast_expr,
  host_span<std::vector<size_type> const> input_row_group_indices,
  rmm::cuda_stream_view stream)
{
  // Filter the input table using AST expression
  auto predicate_col = cudf::detail::compute_column(
    table, ast_expr.get(), stream, cudf::get_current_device_resource_ref());
  auto predicate = predicate_col->view();
  CUDF_EXPECTS(predicate.type().id() == cudf::type_id::BOOL8,
               "Filter expression must return a boolean column");

  auto host_bitmask = [&] {
    std::size_t const num_bitmasks = num_bitmask_words(predicate.size());
    if (predicate.nullable()) {
      return cudf::detail::make_pinned_vector(
        cudf::device_span<bitmask_type const>{predicate.null_mask(), num_bitmasks}, stream);
    } else {
      auto bitmask = cudf::detail::make_pinned_vector<bitmask_type>(num_bitmasks, stream);
      std::fill(bitmask.begin(), bitmask.end(), ~bitmask_type{0});
      return bitmask;
    }
  }();

  auto validity_it = cudf::detail::make_counting_transform_iterator(
    0, [bitmask = host_bitmask.data()](auto bit_index) { return bit_is_set(bitmask, bit_index); });

  // Return only filtered row groups based on predicate
  auto is_row_group_required = cudf::detail::make_pinned_vector(
    cudf::device_span<uint8_t const>{predicate.data<uint8_t>(),
                                     static_cast<size_t>(predicate.size())},
    stream);

  // Return if all are required, or all are nulls.
  if (predicate.null_count() == predicate.size() or std::all_of(is_row_group_required.cbegin(),
                                                                is_row_group_required.cend(),
                                                                [](auto i) { return bool(i); })) {
    return std::nullopt;
  }

  // Collect indices of the filtered row groups
  size_type is_required_idx = 0;
  std::vector<std::vector<size_type>> filtered_row_group_indices;
  for (auto const& input_row_group_index : input_row_group_indices) {
    std::vector<size_type> filtered_row_groups;
    for (auto const rg_idx : input_row_group_index) {
      if ((!validity_it[is_required_idx]) || is_row_group_required[is_required_idx]) {
        filtered_row_groups.push_back(rg_idx);
      }
      ++is_required_idx;
    }
    filtered_row_group_indices.push_back(std::move(filtered_row_groups));
  }

  return {filtered_row_group_indices};
}

namespace {

/**
 * @brief Inverts the non-commutative binary operator
 *
 * @param op Operator to invert
 * @return Inverted operator
 */
ast::ast_operator invert_non_commutative_operators(ast::ast_operator op)
{
  switch (op) {
    case ast::ast_operator::LESS: return ast::ast_operator::GREATER;
    case ast::ast_operator::GREATER: return ast::ast_operator::LESS;
    case ast::ast_operator::LESS_EQUAL: return ast::ast_operator::GREATER_EQUAL;
    case ast::ast_operator::GREATER_EQUAL: return ast::ast_operator::LESS_EQUAL;
    default: return op;
  }
}

}  // namespace

std::tuple<ast::column_reference const*, ast::literal const*, ast::ast_operator, int>
extract_operands_and_operator(ast::operation const& expr)
{
  auto const operands       = expr.get_operands();
  auto const input_operator = expr.get_operator();
  auto const operator_arity = cudf::ast::detail::ast_operator_arity(input_operator);
  CUDF_EXPECTS(operator_arity == 1 or operator_arity == 2,
               "Only unary and binary operations are supported");

  auto* col_ref = dynamic_cast<ast::column_reference const*>(&operands[0].get());

  // Unary operation
  if (operator_arity == 1) { return {col_ref, nullptr, input_operator, operator_arity}; }

  // Binary operation
  if (col_ref != nullptr) {
    auto* literal = dynamic_cast<ast::literal const*>(&operands[1].get());
    return {col_ref, literal, input_operator, operator_arity};
  } else {
    auto const inverted_op = invert_non_commutative_operators(input_operator);
    auto* col_ref          = dynamic_cast<ast::column_reference const*>(&operands[1].get());
    auto* literal          = dynamic_cast<ast::literal const*>(&operands[0].get());
    return {col_ref, literal, inverted_op, operator_arity};
  }
}

}  // namespace cudf::io::parquet::detail
