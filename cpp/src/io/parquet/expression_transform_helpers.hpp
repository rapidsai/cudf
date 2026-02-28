/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/ast/detail/expression_transformer.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <list>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cudf::io::parquet::detail {

/**
 * @brief Classification of an AST expression operand
 */
enum class operand_kind : uint8_t { COLUMN_REF = 0, LITERAL = 1, EXPRESSION = 2 };

/**
 * @brief Extracted unary operand from an AST operation
 */
struct unary_operand {
  operand_kind operand_type;
  ast::column_reference const* col_ref;  ///< Non-null only when the operand is COLUMN_REF
};

/**
 * @brief Extracted binary operator and operands from an AST operation
 *
 * For `lit op col` expressions, the input non-commutative operator is inverted and the
 * operands are normalized to `col op lit` form.
 */
struct binary_operands {
  ast::ast_operator op;  ///< Input or inverted operator to normalize the `lit op col` expressions
  operand_kind lhs_type;
  operand_kind rhs_type;
  ast::column_reference const* col_ref;  ///< Non-null when exactly one operand is COLUMN_REF
  ast::literal const* literal;           ///< Non-null when exactly one operand is LITERAL
};

/**
 * @brief Extracts the unary operand from a unary operation
 */
[[nodiscard]] unary_operand extract_unary_operand(ast::operation const& expr);

/**
 * @brief Decomposes a binary operation into classified parts.
 *
 * When the expression is of the form `lit op col`, the operator is inverted and the result
 * is normalized so that col_ref and literal are set as if the form were `col op lit`.
 */
[[nodiscard]] binary_operands extract_binary_operands(ast::operation const& expr);

/**
 * @brief Collects column names from the expression ignoring the `skip_names`
 */
class names_from_expression : public ast::detail::expression_transformer {
 public:
  names_from_expression() = default;

  names_from_expression(std::optional<std::reference_wrapper<ast::expression const>> expr,
                        std::vector<std::string> const& skip_names,
                        cudf::io::parquet_reader_options const& options,
                        std::vector<SchemaElement> const& schema_tree);

  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::literal const& )
   */
  std::reference_wrapper<ast::expression const> visit(ast::literal const& expr) override;

  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::column_reference const& )
   */
  std::reference_wrapper<ast::expression const> visit(ast::column_reference const& expr) override;

  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::column_name_reference const& )
   */
  std::reference_wrapper<ast::expression const> visit(
    ast::column_name_reference const& expr) override;

  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::operation const& )
   */
  std::reference_wrapper<ast::expression const> visit(ast::operation const& expr) override;

  /**
   * @brief Returns the column names in AST.
   *
   * @return AST operation expression
   */
  [[nodiscard]] std::vector<std::string> to_vector() &&;

 private:
  void visit_operands(
    cudf::host_span<std::reference_wrapper<ast::expression const> const> operands);

  std::unordered_map<cudf::size_type, std::string> _column_indices_to_names;
  std::unordered_set<std::string> _column_names;
  std::unordered_set<std::string> _skip_names;
};

/**
 * @brief Converts named columns to index reference columns
 */
class named_to_reference_converter : public ast::detail::expression_transformer {
 public:
  named_to_reference_converter() = default;

  named_to_reference_converter(std::optional<std::reference_wrapper<ast::expression const>> expr,
                               table_metadata const& metadata);

  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::literal const& )
   */
  std::reference_wrapper<ast::expression const> visit(ast::literal const& expr) override;
  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::column_reference const& )
   */
  std::reference_wrapper<ast::expression const> visit(ast::column_reference const& expr) override;
  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::column_name_reference const& )
   */
  std::reference_wrapper<ast::expression const> visit(
    ast::column_name_reference const& expr) override;
  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::operation const& )
   */
  std::reference_wrapper<ast::expression const> visit(ast::operation const& expr) override;

  /**
   * @brief Returns the converted AST expression
   *
   * @return AST operation expression
   */
  [[nodiscard]] std::optional<std::reference_wrapper<ast::expression const>> get_converted_expr()
    const
  {
    return _converted_expr;
  }

 protected:
  std::vector<std::reference_wrapper<ast::expression const>> visit_operands(
    cudf::host_span<std::reference_wrapper<ast::expression const> const> operands);

  std::unordered_map<std::string, size_type> _column_name_to_index;
  std::optional<std::reference_wrapper<ast::expression const>> _converted_expr;
  // Using std::list or std::deque to avoid reference invalidation
  std::list<ast::column_reference> _col_ref;
  std::list<ast::operation> _operators;
};

/**
 * @brief Collects lists of equality predicate literals in the AST expression, one list per input
 * table column. This is used in row group filtering based on bloom filters.
 */
class equality_literals_collector : public ast::detail::expression_transformer {
 public:
  equality_literals_collector() = default;

  equality_literals_collector(ast::expression const& expr, cudf::size_type num_input_columns);

  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::literal const& )
   */
  std::reference_wrapper<ast::expression const> visit(ast::literal const& expr) override;

  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::column_reference const& )
   */
  std::reference_wrapper<ast::expression const> visit(ast::column_reference const& expr) override;

  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::column_name_reference const& )
   */
  std::reference_wrapper<ast::expression const> visit(
    ast::column_name_reference const& expr) override;

  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::operation const& )
   */
  std::reference_wrapper<ast::expression const> visit(ast::operation const& expr) override;

  /**
   * @brief Vectors of equality literals in the AST expression, one per input table column
   *
   * @return Vectors of equality literals, one per input table column
   */
  [[nodiscard]] std::vector<std::vector<ast::literal*>> get_literals() &&;

 protected:
  std::vector<std::reference_wrapper<ast::expression const>> visit_operands(
    cudf::host_span<std::reference_wrapper<ast::expression const> const> operands);

  size_type _num_input_columns;
  std::vector<std::vector<ast::literal*>> _literals;
};

/**
 * @brief Maps indices of (all or selected) columns to their names
 *
 * @param options Parquet reader options
 * @param schema_tree Parquet schema tree
 *
 * @return Map of column indices to their names
 */
[[nodiscard]] std::unordered_map<cudf::size_type, std::string> map_column_indices_to_names(
  cudf::io::parquet_reader_options const& options, std::vector<SchemaElement> const& schema_tree);

/**
 * @brief Get the column names in expression object
 *
 * @param expr The optional expression object to get the column names from
 * @param skip_names The names of column names to skip in returned column names
 * @return The column names present in expression object except the skip_names
 */
[[nodiscard]] std::vector<std::string> get_column_names_in_expression(
  std::optional<std::reference_wrapper<ast::expression const>> expr,
  std::vector<std::string> const& skip_names,
  cudf::io::parquet_reader_options const& options,
  std::vector<SchemaElement> const& schema_tree);

/**
 * @brief Filter table using the provided (StatsAST or BloomfilterAST) expression and
 * collect filtered row group indices
 *
 * @param table Table of stats or bloom filter membership columns
 * @param ast_expr StatsAST or BloomfilterAST expression to filter with
 * @param input_row_group_indices Lists of input row groups to read, one per source
 * @param stream CUDA stream used for device memory operations and kernel launches
 *
 * @return Collected filtered row group indices, one vector per source, if any. A std::nullopt if
 * all row groups are required or if the computed predicate is all nulls
 */
[[nodiscard]] std::optional<std::vector<std::vector<size_type>>> collect_filtered_row_group_indices(
  cudf::table_view ast_table,
  std::reference_wrapper<ast::expression const> ast_expr,
  host_span<std::vector<size_type> const> input_row_group_indices,
  rmm::cuda_stream_view stream);

}  // namespace cudf::io::parquet::detail
