/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.
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

#include <cudf/ast/detail/expression_transformer.hpp>
#include <cudf/ast/detail/operators.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/transform.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/mr/device/aligned_resource_adaptor.hpp>

#include <thrust/iterator/counting_iterator.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include <optional>
#include <unordered_set>

namespace cudf::io::parquet::detail {

namespace {
/**
 * @brief Converts statistics in column chunks to 2 device columns - min, max values.
 *
 */
struct stats_caster {
  size_type total_row_groups;
  std::vector<metadata> const& per_file_metadata;
  host_span<std::vector<size_type> const> row_group_indices;

  template <typename ToType, typename FromType>
  static ToType targetType(FromType const value)
  {
    if constexpr (cudf::is_timestamp<ToType>()) {
      return static_cast<ToType>(
        typename ToType::duration{static_cast<typename ToType::rep>(value)});
    } else if constexpr (std::is_same_v<ToType, string_view>) {
      return ToType{nullptr, 0};
    } else {
      return static_cast<ToType>(value);
    }
  }

  // uses storage type as T
  template <typename T, CUDF_ENABLE_IF(cudf::is_dictionary<T>() or cudf::is_nested<T>())>
  static T convert(uint8_t const* stats_val, size_t stats_size, Type const type)
  {
    CUDF_FAIL("unsupported type for stats casting");
  }

  template <typename T, CUDF_ENABLE_IF(cudf::is_boolean<T>())>
  static T convert(uint8_t const* stats_val, size_t stats_size, Type const type)
  {
    CUDF_EXPECTS(type == BOOLEAN, "Invalid type and stats combination");
    return targetType<T>(*reinterpret_cast<bool const*>(stats_val));
  }

  // integral but not boolean, and fixed_point, and chrono.
  template <typename T,
            CUDF_ENABLE_IF((cudf::is_integral<T>() and !cudf::is_boolean<T>()) or
                           cudf::is_fixed_point<T>() or cudf::is_chrono<T>())>
  static T convert(uint8_t const* stats_val, size_t stats_size, Type const type)
  {
    switch (type) {
      case INT32: return targetType<T>(*reinterpret_cast<int32_t const*>(stats_val));
      case INT64: return targetType<T>(*reinterpret_cast<int64_t const*>(stats_val));
      case INT96:  // Deprecated in parquet specification
        return targetType<T>(static_cast<__int128_t>(reinterpret_cast<int64_t const*>(stats_val)[0])
                               << 32 |
                             reinterpret_cast<int32_t const*>(stats_val)[2]);
      case BYTE_ARRAY: [[fallthrough]];
      case FIXED_LEN_BYTE_ARRAY:
        if (stats_size == sizeof(T)) {
          // if type size == length of stats_val. then typecast and return.
          if constexpr (cudf::is_chrono<T>()) {
            return targetType<T>(*reinterpret_cast<typename T::rep const*>(stats_val));
          } else {
            return targetType<T>(*reinterpret_cast<T const*>(stats_val));
          }
        }
        // unsupported type
      default: CUDF_FAIL("Invalid type and stats combination");
    }
  }

  template <typename T, CUDF_ENABLE_IF(cudf::is_floating_point<T>())>
  static T convert(uint8_t const* stats_val, size_t stats_size, Type const type)
  {
    switch (type) {
      case FLOAT: return targetType<T>(*reinterpret_cast<float const*>(stats_val));
      case DOUBLE: return targetType<T>(*reinterpret_cast<double const*>(stats_val));
      default: CUDF_FAIL("Invalid type and stats combination");
    }
  }

  template <typename T, CUDF_ENABLE_IF(std::is_same_v<T, string_view>)>
  static T convert(uint8_t const* stats_val, size_t stats_size, Type const type)
  {
    switch (type) {
      case BYTE_ARRAY: [[fallthrough]];
      case FIXED_LEN_BYTE_ARRAY:
        return string_view(reinterpret_cast<char const*>(stats_val), stats_size);
      default: CUDF_FAIL("Invalid type and stats combination");
    }
  }

  // Creates device columns from column statistics (min, max)
  template <typename T>
  std::pair<std::unique_ptr<column>, std::unique_ptr<column>> operator()(
    int schema_idx,
    cudf::data_type dtype,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const
  {
    // List, Struct, Dictionary types are not supported
    if constexpr (cudf::is_compound<T>() && !std::is_same_v<T, string_view>) {
      CUDF_FAIL("Compound types do not have statistics");
    } else {
      // Local struct to hold host columns
      struct host_column {
        // using thrust::host_vector because std::vector<bool> uses bitmap instead of byte per bool.
        cudf::detail::host_vector<T> val;
        std::vector<bitmask_type> null_mask;
        cudf::size_type null_count = 0;
        host_column(size_type total_row_groups, rmm::cuda_stream_view stream)
          : val{cudf::detail::make_host_vector<T>(total_row_groups, stream)},
            null_mask(
              cudf::util::div_rounding_up_safe<size_type>(
                cudf::bitmask_allocation_size_bytes(total_row_groups), sizeof(bitmask_type)),
              ~bitmask_type{0})
        {
        }

        void set_index(size_type index,
                       std::optional<std::vector<uint8_t>> const& binary_value,
                       Type const type)
        {
          if (binary_value.has_value()) {
            val[index] = convert<T>(binary_value.value().data(), binary_value.value().size(), type);
          }
          if (not binary_value.has_value()) {
            clear_bit_unsafe(null_mask.data(), index);
            null_count++;
          }
        }

        static auto make_strings_children(host_span<string_view> host_strings,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr)
        {
          auto const total_char_count = std::accumulate(
            host_strings.begin(), host_strings.end(), 0, [](auto sum, auto const& str) {
              return sum + str.size_bytes();
            });
          auto chars = cudf::detail::make_empty_host_vector<char>(total_char_count, stream);
          auto offsets =
            cudf::detail::make_empty_host_vector<cudf::size_type>(host_strings.size() + 1, stream);
          offsets.push_back(0);
          for (auto const& str : host_strings) {
            auto tmp =
              str.empty() ? std::string_view{} : std::string_view(str.data(), str.size_bytes());
            chars.insert(chars.end(), std::cbegin(tmp), std::cend(tmp));
            offsets.push_back(offsets.back() + tmp.length());
          }
          auto d_chars   = cudf::detail::make_device_uvector_async(chars, stream, mr);
          auto d_offsets = cudf::detail::make_device_uvector_sync(offsets, stream, mr);
          return std::tuple{std::move(d_chars), std::move(d_offsets)};
        }

        auto to_device(cudf::data_type dtype,
                       rmm::cuda_stream_view stream,
                       rmm::device_async_resource_ref mr)
        {
          if constexpr (std::is_same_v<T, string_view>) {
            auto [d_chars, d_offsets] = make_strings_children(val, stream, mr);
            return cudf::make_strings_column(
              val.size(),
              std::make_unique<column>(std::move(d_offsets), rmm::device_buffer{}, 0),
              d_chars.release(),
              null_count,
              rmm::device_buffer{
                null_mask.data(), cudf::bitmask_allocation_size_bytes(val.size()), stream, mr});
          }
          return std::make_unique<column>(
            dtype,
            val.size(),
            cudf::detail::make_device_uvector_async(val, stream, mr).release(),
            rmm::device_buffer{
              null_mask.data(), cudf::bitmask_allocation_size_bytes(val.size()), stream, mr},
            null_count);
        }
      };  // local struct host_column
      host_column min(total_row_groups, stream);
      host_column max(total_row_groups, stream);
      size_type stats_idx = 0;
      for (size_t src_idx = 0; src_idx < row_group_indices.size(); ++src_idx) {
        for (auto const rg_idx : row_group_indices[src_idx]) {
          auto const& row_group = per_file_metadata[src_idx].row_groups[rg_idx];
          auto col              = std::find_if(
            row_group.columns.begin(),
            row_group.columns.end(),
            [schema_idx](ColumnChunk const& col) { return col.schema_idx == schema_idx; });
          if (col != std::end(row_group.columns)) {
            auto const& colchunk = *col;
            // To support deprecated min, max fields.
            auto const& min_value = colchunk.meta_data.statistics.min_value.has_value()
                                      ? colchunk.meta_data.statistics.min_value
                                      : colchunk.meta_data.statistics.min;
            auto const& max_value = colchunk.meta_data.statistics.max_value.has_value()
                                      ? colchunk.meta_data.statistics.max_value
                                      : colchunk.meta_data.statistics.max;
            // translate binary data to Type then to <T>
            min.set_index(stats_idx, min_value, colchunk.meta_data.type);
            max.set_index(stats_idx, max_value, colchunk.meta_data.type);
          } else {
            // Marking it null, if column present in row group
            min.set_index(stats_idx, std::nullopt, {});
            max.set_index(stats_idx, std::nullopt, {});
          }
          stats_idx++;
        }
      };
      return {min.to_device(dtype, stream, mr), max.to_device(dtype, stream, mr)};
    }
  }
};

/**
 * @brief Converts AST expression to StatsAST for comparing with column statistics
 * This is used in row group filtering based on predicate.
 * statistics min value of a column is referenced by column_index*2
 * statistics max value of a column is referenced by column_index*2+1
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
    return expr;
  }

  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::column_name_reference const& )
   */
  std::reference_wrapper<ast::expression const> visit(
    ast::column_name_reference const& expr) override
  {
    CUDF_FAIL("Column name reference is not supported in statistics AST");
  }

  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::operation const& )
   */
  std::reference_wrapper<ast::expression const> visit(ast::operation const& expr) override
  {
    using cudf::ast::ast_operator;
    auto const operands = expr.get_operands();
    auto const op       = expr.get_operator();

    if (auto* v = dynamic_cast<ast::column_reference const*>(&operands[0].get())) {
      // First operand should be column reference, second should be literal.
      CUDF_EXPECTS(cudf::ast::detail::ast_operator_arity(op) == 2,
                   "Only binary operations are supported on column reference");
      CUDF_EXPECTS(dynamic_cast<ast::literal const*>(&operands[1].get()) != nullptr,
                   "Second operand of binary operation with column reference must be a literal");
      v->accept(*this);
      // Push literal into the ast::tree
      auto const& literal =
        _stats_expr.push(*dynamic_cast<ast::literal const*>(&operands[1].get()));
      auto const col_index = v->get_column_index();
      switch (op) {
        /* transform to stats conditions. op(col, literal)
        col1 == val --> vmin <= val && vmax >= val
        col1 != val --> !(vmin == val && vmax == val)
        col1 >  val --> vmax > val
        col1 <  val --> vmin < val
        col1 >= val --> vmax >= val
        col1 <= val --> vmin <= val
        */
        case ast_operator::EQUAL: {
          auto const& vmin = _stats_expr.push(ast::column_reference{col_index * 2});
          auto const& vmax = _stats_expr.push(ast::column_reference{col_index * 2 + 1});
          _stats_expr.push(ast::operation{
            ast::ast_operator::LOGICAL_AND,
            _stats_expr.push(ast::operation{ast_operator::GREATER_EQUAL, vmax, literal}),
            _stats_expr.push(ast::operation{ast_operator::LESS_EQUAL, vmin, literal})});
          break;
        }
        case ast_operator::NOT_EQUAL: {
          auto const& vmin = _stats_expr.push(ast::column_reference{col_index * 2});
          auto const& vmax = _stats_expr.push(ast::column_reference{col_index * 2 + 1});
          _stats_expr.push(ast::operation{
            ast_operator::LOGICAL_OR,
            _stats_expr.push(ast::operation{ast_operator::NOT_EQUAL, vmin, vmax}),
            _stats_expr.push(ast::operation{ast_operator::NOT_EQUAL, vmax, literal})});
          break;
        }
        case ast_operator::LESS: [[fallthrough]];
        case ast_operator::LESS_EQUAL: {
          auto const& vmin = _stats_expr.push(ast::column_reference{col_index * 2});
          _stats_expr.push(ast::operation{op, vmin, literal});
          break;
        }
        case ast_operator::GREATER: [[fallthrough]];
        case ast_operator::GREATER_EQUAL: {
          auto const& vmax = _stats_expr.push(ast::column_reference{col_index * 2 + 1});
          _stats_expr.push(ast::operation{op, vmax, literal});
          break;
        }
        default: CUDF_FAIL("Unsupported operation in Statistics AST");
      };
    } else {
      auto new_operands = visit_operands(operands);
      if (cudf::ast::detail::ast_operator_arity(op) == 2) {
        _stats_expr.push(ast::operation{op, new_operands.front(), new_operands.back()});
      } else if (cudf::ast::detail::ast_operator_arity(op) == 1) {
        _stats_expr.push(ast::operation{op, new_operands.front()});
      }
    }
    return _stats_expr.back();
  }

  /**
   * @brief Returns the AST to apply on Column chunk statistics.
   *
   * @return AST operation expression
   */
  [[nodiscard]] std::reference_wrapper<ast::expression const> get_stats_expr() const
  {
    return _stats_expr.back();
  }

 private:
  std::vector<std::reference_wrapper<ast::expression const>> visit_operands(
    cudf::host_span<std::reference_wrapper<ast::expression const> const> operands)
  {
    std::vector<std::reference_wrapper<ast::expression const>> transformed_operands;
    for (auto const& operand : operands) {
      auto const new_operand = operand.get().accept(*this);
      transformed_operands.push_back(new_operand);
    }
    return transformed_operands;
  }
  ast::tree _stats_expr;
  size_type _num_columns;
};
}  // namespace

std::optional<std::vector<std::vector<size_type>>> aggregate_reader_metadata::apply_stats_filters(
  host_span<std::vector<size_type> const> input_row_group_indices,
  size_type total_row_groups,
  host_span<data_type const> output_dtypes,
  host_span<int const> output_column_schemas,
  std::reference_wrapper<ast::expression const> filter,
  rmm::cuda_stream_view stream) const
{
  auto mr = cudf::get_current_device_resource_ref();

  // Converts Column chunk statistics to a table
  // where min(col[i]) = columns[i*2], max(col[i])=columns[i*2+1]
  // For each column, it contains #sources * #column_chunks_per_src rows.
  std::vector<std::unique_ptr<column>> columns;
  stats_caster const stats_col{
    static_cast<size_type>(total_row_groups), per_file_metadata, input_row_group_indices};
  for (size_t col_idx = 0; col_idx < output_dtypes.size(); col_idx++) {
    auto const schema_idx = output_column_schemas[col_idx];
    auto const& dtype     = output_dtypes[col_idx];
    // Only comparable types except fixed point are supported.
    if (cudf::is_compound(dtype) && dtype.id() != cudf::type_id::STRING) {
      // placeholder only for unsupported types.
      columns.push_back(cudf::make_numeric_column(
        data_type{cudf::type_id::BOOL8}, total_row_groups, rmm::device_buffer{}, 0, stream, mr));
      columns.push_back(cudf::make_numeric_column(
        data_type{cudf::type_id::BOOL8}, total_row_groups, rmm::device_buffer{}, 0, stream, mr));
      continue;
    }
    auto [min_col, max_col] =
      cudf::type_dispatcher<dispatch_storage_type>(dtype, stats_col, schema_idx, dtype, stream, mr);
    columns.push_back(std::move(min_col));
    columns.push_back(std::move(max_col));
  }
  auto stats_table = cudf::table(std::move(columns));

  // Converts AST to StatsAST with reference to min, max columns in above `stats_table`.
  stats_expression_converter const stats_expr{filter.get(),
                                              static_cast<size_type>(output_dtypes.size())};

  // Filter stats table with StatsAST expression and collect filtered row group indices
  return collect_filtered_row_group_indices(
    stats_table, stats_expr.get_stats_expr(), input_row_group_indices, stream);
}

std::pair<std::optional<std::vector<std::vector<size_type>>>, surviving_row_group_metrics>
aggregate_reader_metadata::filter_row_groups(
  host_span<std::unique_ptr<datasource> const> sources,
  host_span<std::vector<size_type> const> input_row_group_indices,
  size_type total_row_groups,
  host_span<data_type const> output_dtypes,
  host_span<int const> output_column_schemas,
  std::reference_wrapper<ast::expression const> filter,
  rmm::cuda_stream_view stream) const
{
  // Apply stats filtering on input row groups
  auto const stats_filtered_row_groups = apply_stats_filters(input_row_group_indices,
                                                             total_row_groups,
                                                             output_dtypes,
                                                             output_column_schemas,
                                                             filter,
                                                             stream);

  // Number of surviving row groups after applying stats filter
  auto const num_stats_filtered_row_groups =
    stats_filtered_row_groups.has_value()
      ? std::accumulate(stats_filtered_row_groups.value().cbegin(),
                        stats_filtered_row_groups.value().cend(),
                        size_type{0},
                        [](auto& sum, auto const& per_file_row_groups) {
                          return sum + per_file_row_groups.size();
                        })
      : total_row_groups;

  // Span of row groups to apply bloom filtering on.
  auto const bloom_filter_input_row_groups =
    stats_filtered_row_groups.has_value()
      ? host_span<std::vector<size_type> const>(stats_filtered_row_groups.value())
      : input_row_group_indices;

  // Collect equality literals for each input table column for bloom filtering
  auto const equality_literals =
    equality_literals_collector{filter.get(), static_cast<cudf::size_type>(output_dtypes.size())}
      .get_literals();

  // Collect schema indices of columns with equality predicate(s)
  std::vector<cudf::size_type> equality_col_schemas;
  thrust::copy_if(thrust::host,
                  output_column_schemas.begin(),
                  output_column_schemas.end(),
                  equality_literals.begin(),
                  std::back_inserter(equality_col_schemas),
                  [](auto& eq_literals) { return not eq_literals.empty(); });

  // Return early if no column with equality predicate(s)
  if (equality_col_schemas.empty()) {
    return {stats_filtered_row_groups,
            {std::make_optional(num_stats_filtered_row_groups), std::nullopt}};
  }

  // Aligned resource adaptor to allocate bloom filter buffers with
  auto aligned_mr = rmm::mr::aligned_resource_adaptor(cudf::get_current_device_resource(),
                                                      get_bloom_filter_alignment());

  // Read a vector of bloom filter bitset device buffers for all columns with equality
  // predicate(s) across all row groups
  auto bloom_filter_data = read_bloom_filters(sources,
                                              bloom_filter_input_row_groups,
                                              equality_col_schemas,
                                              num_stats_filtered_row_groups,
                                              stream,
                                              aligned_mr);

  // No bloom filter buffers, return early
  if (bloom_filter_data.empty()) {
    return {stats_filtered_row_groups,
            {std::make_optional(num_stats_filtered_row_groups), std::nullopt}};
  }

  // Apply bloom filtering on the output row groups from stats filter
  auto const bloom_filtered_row_groups = apply_bloom_filters(bloom_filter_data,
                                                             bloom_filter_input_row_groups,
                                                             equality_literals,
                                                             num_stats_filtered_row_groups,
                                                             output_dtypes,
                                                             equality_col_schemas,
                                                             filter,
                                                             stream);

  // Number of surviving row groups after applying bloom filter
  auto const num_bloom_filtered_row_groups =
    bloom_filtered_row_groups.has_value()
      ? std::accumulate(bloom_filtered_row_groups.value().cbegin(),
                        bloom_filtered_row_groups.value().cend(),
                        size_type{0},
                        [](auto& sum, auto const& per_file_row_groups) {
                          return sum + per_file_row_groups.size();
                        })
      : num_stats_filtered_row_groups;

  // Return bloom filtered row group indices iff collected
  return {
    bloom_filtered_row_groups.has_value() ? bloom_filtered_row_groups : stats_filtered_row_groups,
    {std::make_optional(num_stats_filtered_row_groups),
     std::make_optional(num_bloom_filtered_row_groups)}};
}

// convert column named expression to column index reference expression
named_to_reference_converter::named_to_reference_converter(
  std::optional<std::reference_wrapper<ast::expression const>> expr, table_metadata const& metadata)
{
  if (!expr.has_value()) return;
  // create map for column name.
  std::transform(metadata.schema_info.cbegin(),
                 metadata.schema_info.cend(),
                 thrust::counting_iterator<size_t>(0),
                 std::inserter(column_name_to_index, column_name_to_index.end()),
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
  auto col_index_it = column_name_to_index.find(expr.get_column_name());
  if (col_index_it == column_name_to_index.end()) {
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
  auto const operands = expr.get_operands();
  auto op             = expr.get_operator();
  auto new_operands   = visit_operands(operands);
  if (cudf::ast::detail::ast_operator_arity(op) == 2) {
    _operators.emplace_back(op, new_operands.front(), new_operands.back());
  } else if (cudf::ast::detail::ast_operator_arity(op) == 1) {
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

/**
 * @brief Converts named columns to index reference columns
 *
 */
class names_from_expression : public ast::detail::expression_transformer {
 public:
  names_from_expression(std::optional<std::reference_wrapper<ast::expression const>> expr,
                        std::vector<std::string> const& skip_names)
    : _skip_names(skip_names.cbegin(), skip_names.cend())
  {
    if (!expr.has_value()) return;
    expr.value().get().accept(*this);
  }

  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::literal const& )
   */
  std::reference_wrapper<ast::expression const> visit(ast::literal const& expr) override
  {
    return expr;
  }
  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::column_reference const& )
   */
  std::reference_wrapper<ast::expression const> visit(ast::column_reference const& expr) override
  {
    return expr;
  }
  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::column_name_reference const& )
   */
  std::reference_wrapper<ast::expression const> visit(
    ast::column_name_reference const& expr) override
  {
    // collect column names
    auto col_name = expr.get_column_name();
    if (_skip_names.count(col_name) == 0) { _column_names.insert(col_name); }
    return expr;
  }
  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::operation const& )
   */
  std::reference_wrapper<ast::expression const> visit(ast::operation const& expr) override
  {
    visit_operands(expr.get_operands());
    return expr;
  }

  /**
   * @brief Returns the column names in AST.
   *
   * @return AST operation expression
   */
  [[nodiscard]] std::vector<std::string> to_vector() &&
  {
    return {std::make_move_iterator(_column_names.begin()),
            std::make_move_iterator(_column_names.end())};
  }

 private:
  void visit_operands(cudf::host_span<std::reference_wrapper<ast::expression const> const> operands)
  {
    for (auto const& operand : operands) {
      operand.get().accept(*this);
    }
  }

  std::unordered_set<std::string> _column_names;
  std::unordered_set<std::string> _skip_names;
};

[[nodiscard]] std::vector<std::string> get_column_names_in_expression(
  std::optional<std::reference_wrapper<ast::expression const>> expr,
  std::vector<std::string> const& skip_names)
{
  return names_from_expression(expr, skip_names).to_vector();
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

  auto const host_bitmask = [&] {
    auto const num_bitmasks = num_bitmask_words(predicate.size());
    if (predicate.nullable()) {
      return cudf::detail::make_host_vector_sync(
        device_span<bitmask_type const>(predicate.null_mask(), num_bitmasks), stream);
    } else {
      auto bitmask = cudf::detail::make_host_vector<bitmask_type>(num_bitmasks, stream);
      std::fill(bitmask.begin(), bitmask.end(), ~bitmask_type{0});
      return bitmask;
    }
  }();

  auto validity_it = cudf::detail::make_counting_transform_iterator(
    0, [bitmask = host_bitmask.data()](auto bit_index) { return bit_is_set(bitmask, bit_index); });

  // Return only filtered row groups based on predicate
  auto const is_row_group_required = cudf::detail::make_host_vector_sync(
    device_span<uint8_t const>(predicate.data<uint8_t>(), predicate.size()), stream);

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

}  // namespace cudf::io::parquet::detail
