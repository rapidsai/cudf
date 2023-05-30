/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
#include "reader_impl_helpers.hpp"

#include <cudf/ast/detail/expression_parser.hpp>  // possibly_null_value defined here, but used in operators.hpp
#include <cudf/ast/detail/expression_transformer.hpp>
#include <cudf/ast/detail/operators.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/mr/device/per_device_resource.hpp>

#include <algorithm>
#include <optional>

namespace cudf::io::detail::parquet {

/**
 * @brief Converts statistics in column chunks to 2 device columns - min, max values.
 *
 */
struct stats_caster {
  size_type total_row_groups;
  std::vector<metadata> const& per_file_metadata;
  host_span<std::vector<size_type> const> row_group_indices;
  rmm::cuda_stream_view stream;

  template <typename T>
  static T convert(uint8_t const* stats_val, Type const type)
  {
    switch (type) {
      case BOOLEAN: return static_cast<T>(*reinterpret_cast<bool const*>(stats_val));
      case INT32: return static_cast<T>(*reinterpret_cast<int32_t const*>(stats_val));
      case INT64: return static_cast<T>(*reinterpret_cast<int64_t const*>(stats_val));
      case INT96:  // Deprecated
        return static_cast<T>(
          static_cast<__int128_t>(reinterpret_cast<int64_t const*>(stats_val)[0]) << 32 |
          reinterpret_cast<int32_t const*>(stats_val)[2]);
      case FLOAT: return static_cast<T>(*reinterpret_cast<float const*>(stats_val));
      case DOUBLE: return static_cast<T>(*reinterpret_cast<double const*>(stats_val));
      // TODO string support
      case BYTE_ARRAY:
      case FIXED_LEN_BYTE_ARRAY:
      default: return static_cast<T>(0);
    }
  }

  template <typename T>
  std::pair<std::unique_ptr<column>, std::unique_ptr<column>> operator()(size_t col_idx) const
  {
    if constexpr (!cudf::is_numeric<T>())
      CUDF_FAIL("Only numeric types are supported");
    else {
      struct host_column {
        thrust::host_vector<T> val;
        std::vector<bitmask_type> null_mask;
        cudf::size_type null_count = 0;
        host_column(size_type total_row_groups)
          : val(total_row_groups),
            null_mask(cudf::bitmask_allocation_size_bytes(total_row_groups) / sizeof(bitmask_type),
                      ~bitmask_type{0})
        {
        }
        void set_index(size_type index, std::vector<uint8_t> const& binary_value, Type const type)
        {
          val[index] = binary_value.empty() ? T{0} : convert<T>(binary_value.data(), type);
          if (binary_value.empty()) {
            clear_bit_unsafe(null_mask.data(), index);
            null_count++;
          }
        }
        auto to_device(rmm::cuda_stream_view stream)
        {
          return std::make_unique<column>(
            cudf::detail::make_device_uvector_async(
              val, stream, rmm::mr::get_current_device_resource()),
            rmm::device_buffer{
              null_mask.data(), cudf::bitmask_allocation_size_bytes(val.size()), stream},
            null_count);
        }
      };
      host_column min(total_row_groups);
      host_column max(total_row_groups);

      int stats_idx = 0;
      std::cout << "col_idx[" << col_idx << "]:min, max\n";
      // TODO first 2 for-loops as iterator.
      for (size_t src_idx = 0; src_idx < row_group_indices.size(); ++src_idx) {
        for (auto const rg_idx : row_group_indices[src_idx]) {
          auto const& row_group = per_file_metadata[src_idx].row_groups[rg_idx];
          auto const& colchunk  = row_group.columns[col_idx];
          // To support deprecated min, max fields.
          auto const& min_value = colchunk.meta_data.statistics_blob.min_value.size() > 0
                                    ? colchunk.meta_data.statistics_blob.min_value
                                    : colchunk.meta_data.statistics_blob.min;
          auto const& max_value = colchunk.meta_data.statistics_blob.max_value.size() > 0
                                    ? colchunk.meta_data.statistics_blob.max_value
                                    : colchunk.meta_data.statistics_blob.max;
          // translate binary data to Type then to <T>
          min.set_index(stats_idx, min_value, colchunk.meta_data.type);
          max.set_index(stats_idx, max_value, colchunk.meta_data.type);
          std::cout << min.val[stats_idx] << "," << max.val[stats_idx] << "\n";
          stats_idx++;
        }
      };
      return {min.to_device(stream), max.to_device(stream)};
    }
  }
};

/**
 * @brief Converts AST expression to StatsAST for comparing with column statistics
 * This is used in row group filtering based on predicate.
 * statistics min value of a column is referenced by column_index*2
 * statistics min value of a column is referenced by column_index*2+1
 *
 */
class stats_expression_converter : public ast::detail::expression_transformer {
 public:
  stats_expression_converter(ast::expression const& expr, size_type const& num_columns)
    : _num_columns{num_columns}
  {
    expr.accept(*this);
  }

  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::literal const& )
   */
  std::reference_wrapper<ast::expression const> visit(ast::literal const& expr) override
  {
    _stats_expr = std::reference_wrapper<ast::expression const>(expr);
    return expr;
  }

  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::column_reference const& )
   */
  std::reference_wrapper<ast::expression const> visit(ast::column_reference const& expr) override
  {
    CUDF_EXPECTS(expr.get_table_source() == ast::table_reference::LEFT,
                 "Statistics AST supports only left table");
    CUDF_EXPECTS(expr.get_column_index() < _num_columns,
                 "Column index cannot be more than number of columns in the table");
    _stats_expr = std::reference_wrapper<ast::expression const>(expr);
    return expr;
  }

  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::operation const& )
   */
  std::reference_wrapper<ast::expression const> visit(ast::operation const& expr) override
  {
    auto const operands = expr.get_operands();
    auto op             = expr.get_operator();

    if (auto* v = dynamic_cast<ast::column_reference const*>(&operands[0].get())) {
      // First operand should be column reference, second should be literal.
      CUDF_EXPECTS(cudf::ast::detail::ast_operator_arity(op) == 2,
                   "Only binary operations are supported on column reference");
      CUDF_EXPECTS(dynamic_cast<ast::literal const*>(&operands[1].get()) != nullptr,
                   "Second operand of binary operation with column reference must be a literal");
      auto const col_index = v->get_column_index();
      switch (op) {
        /* transform to stats conditions. op(col, literal)
        col1 == val --> vmin <= val && vmax >= val
        col1 != val --> vmin > val or vmax < val
        col1 >  val --> vmax > val
        col1 <  val --> vmin < val
        col1 >= val --> vmax >= val
        col1 <= val --> vmin <= val
        */
        case ast::ast_operator::EQUAL: {
          _col_ref.push_back(ast::column_reference(col_index * 2));
          auto const& vmin = _col_ref.back();
          _col_ref.push_back(ast::column_reference(col_index * 2 + 1));
          auto const& vmax = _col_ref.back();
          _operators.push_back(
            ast::operation(ast::ast_operator::LESS_EQUAL, vmin, operands[1].get()));
          _operators.push_back(
            ast::operation(ast::ast_operator::GREATER_EQUAL, vmax, operands[1].get()));
          _operators.push_back(ast::operation(
            ast::ast_operator::LOGICAL_AND, *(_operators.end() - 2), _operators.back()));
          break;
        }
        case ast::ast_operator::NOT_EQUAL: {
          _col_ref.push_back(ast::column_reference(col_index * 2));
          auto const& vmin = _col_ref.back();
          _col_ref.push_back(ast::column_reference(col_index * 2 + 1));
          auto const& vmax = _col_ref.back();
          _operators.push_back(ast::operation(ast::ast_operator::GREATER, vmin, operands[1].get()));
          _operators.push_back(ast::operation(ast::ast_operator::LESS, vmax, operands[1].get()));
          _operators.push_back(ast::operation(
            ast::ast_operator::LOGICAL_OR, *(_operators.end() - 2), _operators.back()));
          break;
        }
        // can these be combined in any way?
        case ast::ast_operator::LESS: [[fallthrough]];
        case ast::ast_operator::LESS_EQUAL: {
          _col_ref.push_back(ast::column_reference(col_index * 2));
          auto const& vmin = _col_ref.back();
          _operators.push_back(ast::operation(op, vmin, operands[1].get()));
          break;
        }
        case ast::ast_operator::GREATER: [[fallthrough]];
        case ast::ast_operator::GREATER_EQUAL: {
          _col_ref.push_back(ast::column_reference(col_index * 2 + 1));
          auto const& vmax = _col_ref.back();
          _operators.push_back(ast::operation(op, vmax, operands[1].get()));
          break;
        }
        default: CUDF_FAIL("Unsupported operation in Statistics AST");
      };
    } else {
      auto new_operands = visit_operands(operands);
      if (cudf::ast::detail::ast_operator_arity(op) == 2)
        _operators.push_back(ast::operation(op, new_operands.front(), new_operands.back()));
      else if (cudf::ast::detail::ast_operator_arity(op) == 1)
        _operators.push_back(ast::operation(op, new_operands.front()));
    }
    _stats_expr = std::reference_wrapper<ast::expression const>(_operators.back());
    return std::reference_wrapper<ast::expression const>(_operators.back());
  }

  /**
   * @brief Returns the AST to apply on Column chunk statistics.
   *
   * @return AST operation expression
   */
  [[nodiscard]] std::reference_wrapper<ast::expression const> get_stats_expr() const
  {
    // if(!_stats_expr.has_value()) _stats_expr = ast::operation(ast_operator::IDENTITY, expr)
    return _stats_expr.value().get();
  }

 private:
  std::vector<std::reference_wrapper<ast::expression const>> visit_operands(
    std::vector<std::reference_wrapper<ast::expression const>> operands)
  {
    std::vector<std::reference_wrapper<ast::expression const>> transformed_operands;
    for (auto const& operand : operands) {
      auto const new_operand = operand.get().accept(*this);
      transformed_operands.push_back(new_operand);
    }
    return transformed_operands;
  }
  std::optional<std::reference_wrapper<ast::expression const>> _stats_expr;
  size_type _num_columns;
  std::vector<ast::literal> _literals;
  std::vector<ast::column_reference> _col_ref;
  std::vector<ast::operation> _operators;
};

std::optional<std::vector<std::vector<size_type>>> aggregate_reader_metadata::filter_row_groups(
  host_span<std::vector<size_type> const> row_group_indices,
  host_span<data_type const> output_dtypes,
  std::reference_wrapper<ast::expression const> filter) const
{
  // Create row group indices.
  std::vector<std::vector<size_type>> filtered_row_group_indices;
  std::vector<std::vector<size_type>> all_row_group_indices;
  host_span<std::vector<size_type> const> input_row_group_indices;
  if (row_group_indices.empty()) {
    std::transform(per_file_metadata.begin(),
                   per_file_metadata.end(),
                   std::back_inserter(all_row_group_indices),
                   [](auto const& file_meta) {
                     std::vector<size_type> rg_idx(file_meta.row_groups.size());
                     std::iota(rg_idx.begin(), rg_idx.end(), 0);
                     return rg_idx;
                   });
    input_row_group_indices = host_span<std::vector<size_type> const>(all_row_group_indices);
  } else {
    input_row_group_indices = row_group_indices;
  }
  auto const total_row_groups = std::accumulate(input_row_group_indices.begin(),
                                                input_row_group_indices.end(),
                                                0,
                                                [](size_type sum, auto const& per_file_row_groups) {
                                                  return sum + per_file_row_groups.size();
                                                });

  // Converts Column chunk statistics to a table
  // where min(col[i]) = columns[i*2], max(col[i])=columns[i*2+1]
  // For each column, it contains #sources * #column_chunks_per_src rows.
  std::vector<std::unique_ptr<column>> columns;
  stats_caster stats_col{
    total_row_groups, per_file_metadata, input_row_group_indices, cudf::get_default_stream()};
  for (size_t col_idx = 0; col_idx < output_dtypes.size(); col_idx++) {
    auto const& dtype = output_dtypes[col_idx];
    // Only numeric supported for now.
    if (!cudf::is_numeric(dtype)) {
      std::cout << "non_numeric[" << col_idx << "]=" << static_cast<int>(dtype.id()) << "\n";
      columns.push_back(
        cudf::make_numeric_column(data_type{cudf::type_id::BOOL8}, total_row_groups));
      columns.push_back(
        cudf::make_numeric_column(data_type{cudf::type_id::BOOL8}, total_row_groups));
      continue;
    }
    auto [min_col, max_col] = cudf::type_dispatcher(dtype, stats_col, col_idx);
    columns.push_back(std::move(min_col));
    columns.push_back(std::move(max_col));
  }
  auto stats_table = cudf::table(std::move(columns));

  // Converts AST to StatsAST with reference to min, max columns in above `stats_table`.
  stats_expression_converter stats_expr{filter, static_cast<size_type>(output_dtypes.size())};
  auto stats_ast = stats_expr.get_stats_expr();
  auto predicate =
    cudf::compute_column(stats_table, stats_ast.get(), rmm::mr::get_current_device_resource());

  auto is_row_group_required = cudf::detail::make_std_vector_sync(
    device_span<uint8_t const>(predicate->view().data<uint8_t>(), predicate->size()),
    cudf::get_default_stream());
  // Return only filtered row groups based on predicate
  if (!std::all_of(is_row_group_required.begin(), is_row_group_required.end(), [](auto i) {
        return bool(i);
      })) {
    size_type is_required_idx = 0;
    for (size_t src_idx = 0; src_idx < input_row_group_indices.size(); ++src_idx) {
      std::vector<size_type> filtered_row_groups;
      for (auto const rg_idx : input_row_group_indices[src_idx]) {
        if (is_row_group_required[is_required_idx++]) {
          filtered_row_groups.push_back(rg_idx);
          std::cout << "Read [src_idx, rg_idx]=[" << src_idx << "," << rg_idx << "]\n";
        }
      }
      filtered_row_group_indices.push_back(std::move(filtered_row_groups));
    }
    return {std::move(filtered_row_group_indices)};
  }
  return std::nullopt;
  // TODO will no of col chunks for all row groups be same?
}

}  // namespace cudf::io::detail::parquet
