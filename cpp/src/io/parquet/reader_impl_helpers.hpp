/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include "parquet_gpu.hpp"

#include <cudf/ast/detail/expression_transformer.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/types.hpp>

#include <list>
#include <tuple>
#include <vector>

namespace cudf::io::parquet::detail {

/**
 * @brief page location and size info
 */
struct page_info {
  // page location info from the offset index
  PageLocation location;
  // number of rows in the page, calculated from offset index
  int64_t num_rows;
  // number of valid values in page, calculated from definition level histogram if present
  std::optional<int64_t> num_valid;
  // number of null values in page, calculated from definition level histogram if present
  std::optional<int64_t> num_nulls;
  // number of bytes of variable-length data from the offset index (byte_array columns only)
  std::optional<int64_t> var_bytes_size;
};

/**
 * @brief column chunk metadata
 */
struct column_chunk_info {
  // offset in file of the dictionary (if present)
  std::optional<int64_t> dictionary_offset;
  // size of dictionary (if present)
  std::optional<int32_t> dictionary_size;
  std::vector<page_info> pages;

  /**
   * @brief Determine if this column chunk has a dictionary page.
   *
   * @return `true` if this column chunk has a dictionary page.
   */
  [[nodiscard]] constexpr bool has_dictionary() const
  {
    return dictionary_offset.has_value() && dictionary_size.has_value();
  }
};

/**
 * @brief The row_group_info class
 */
struct row_group_info {
  size_type index;  // row group index within a file. aggregate_reader_metadata::get_row_group() is
                    // called with index and source_index
  size_t start_row;
  size_type source_index;  // file index.

  // Optional metadata pulled from the column and offset indexes, if present.
  std::optional<std::vector<column_chunk_info>> column_chunks;

  row_group_info() = default;

  row_group_info(size_type index, size_t start_row, size_type source_index)
    : index{index}, start_row{start_row}, source_index{source_index}
  {
  }

  /**
   * @brief Indicates the presence of page-level indexes.
   */
  [[nodiscard]] bool has_page_index() const { return column_chunks.has_value(); }
};

/**
 * @brief Function that translates Parquet datatype to cuDF type enum
 */
[[nodiscard]] type_id to_type_id(SchemaElement const& schema,
                                 bool strings_to_categorical,
                                 type_id timestamp_type_id);

/**
 * @brief Converts cuDF type enum to column logical type
 */
[[nodiscard]] inline data_type to_data_type(type_id t_id, SchemaElement const& schema)
{
  return t_id == type_id::DECIMAL32 || t_id == type_id::DECIMAL64 || t_id == type_id::DECIMAL128
           ? data_type{t_id, numeric::scale_type{-schema.decimal_scale}}
           : data_type{t_id};
}

/**
 * @brief Class for parsing dataset metadata
 */
struct metadata : public FileMetaData {
  explicit metadata(datasource* source);
  void sanitize_schema();
};

struct arrow_schema_data_types {
  std::vector<arrow_schema_data_types> children;
  data_type type{type_id::EMPTY};
};

class aggregate_reader_metadata {
  std::vector<metadata> per_file_metadata;
  std::vector<std::unordered_map<std::string, std::string>> keyval_maps;

  int64_t num_rows;
  size_type num_row_groups;

  /**
   * @brief Create a metadata object from each element in the source vector
   */
  static std::vector<metadata> metadatas_from_sources(
    host_span<std::unique_ptr<datasource> const> sources);

  /**
   * @brief Collect the keyvalue maps from each per-file metadata object into a vector of maps.
   */
  [[nodiscard]] std::vector<std::unordered_map<std::string, std::string>> collect_keyval_metadata()
    const;

  /**
   * @brief Decodes and constructs the arrow schema from the "ARROW:schema" IPC message
   * in key value metadata section of Parquet file footer
   */
  [[nodiscard]] arrow_schema_data_types collect_arrow_schema() const;

  /**
   * @brief Co-walks the collected arrow and Parquet schema, updates
   * dtypes and destroys the no longer needed arrow schema object(s).
   */
  void apply_arrow_schema();

  /**
   * @brief Decode an arrow:IPC message and returns an optional string_view of
   * its metadata header
   */
  [[nodiscard]] std::optional<std::string_view> decode_ipc_message(
    std::string_view const serialized_message) const;

  /**
   * @brief Sums up the number of rows of each source
   */
  [[nodiscard]] int64_t calc_num_rows() const;

  /**
   * @brief Sums up the number of row groups of each source
   */
  [[nodiscard]] size_type calc_num_row_groups() const;

  /**
   * @brief Calculate column index info for the given `row_group_info`
   *
   * @param rg_info Struct used to summarize metadata for a single row group
   * @param chunk_start_row Global index of first row in the row group
   */
  void column_info_for_row_group(row_group_info& rg_info, size_type chunk_start_row) const;

 public:
  aggregate_reader_metadata(host_span<std::unique_ptr<datasource> const> sources,
                            bool use_arrow_schema);

  [[nodiscard]] RowGroup const& get_row_group(size_type row_group_index, size_type src_idx) const;

  [[nodiscard]] ColumnChunkMetaData const& get_column_metadata(size_type row_group_index,
                                                               size_type src_idx,
                                                               int schema_idx) const;

  /**
   * @brief Extracts high-level metadata for all row groups
   *
   * @return List of maps containing metadata information for each row group
   */
  [[nodiscard]] std::vector<std::unordered_map<std::string, int64_t>> get_rowgroup_metadata() const;

  [[nodiscard]] auto get_num_rows() const { return num_rows; }

  [[nodiscard]] auto get_num_row_groups() const { return num_row_groups; }

  [[nodiscard]] auto const& get_schema(int schema_idx) const
  {
    return per_file_metadata[0].schema[schema_idx];
  }

  [[nodiscard]] auto const& get_key_value_metadata() const& { return keyval_maps; }
  [[nodiscard]] auto&& get_key_value_metadata() && { return std::move(keyval_maps); }

  /**
   * @brief Gets the concrete nesting depth of output cudf columns
   *
   * @param schema_index Schema index of the input column
   *
   * @return comma-separated index column names in quotes
   */
  [[nodiscard]] inline int get_output_nesting_depth(int schema_index) const
  {
    auto& pfm = per_file_metadata[0];
    int depth = 0;

    // walk upwards, skipping repeated fields
    while (schema_index > 0) {
      auto const& elm = pfm.schema[schema_index];
      if (!elm.is_stub()) { depth++; }
      // schema of one-level encoding list doesn't contain nesting information, so we need to
      // manually add an extra nesting level
      if (elm.is_one_level_list(pfm.schema[elm.parent_idx])) { depth++; }
      schema_index = elm.parent_idx;
    }
    return depth;
  }

  /**
   * @brief Extracts the pandas "index_columns" section
   *
   * PANDAS adds its own metadata to the key_value section when writing out the
   * dataframe to a file to aid in exact reconstruction. The JSON-formatted
   * metadata contains the index column(s) and PANDA-specific datatypes.
   *
   * @return comma-separated index column names in quotes
   */
  [[nodiscard]] std::string get_pandas_index() const;

  /**
   * @brief Extracts the column name(s) used for the row indexes in a dataframe
   *
   * @param names List of column names to load, where index column name(s) will be added
   */
  [[nodiscard]] std::vector<std::string> get_pandas_index_names() const;

  /**
   * @brief Filters the row groups based on predicate filter
   *
   * @param row_group_indices Lists of row groups to read, one per source
   * @param output_dtypes Datatypes of of output columns
   * @param output_column_schemas schema indices of output columns
   * @param filter AST expression to filter row groups based on Column chunk statistics
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @return Filtered row group indices, if any is filtered.
   */
  [[nodiscard]] std::optional<std::vector<std::vector<size_type>>> filter_row_groups(
    host_span<std::vector<size_type> const> row_group_indices,
    host_span<data_type const> output_dtypes,
    host_span<int const> output_column_schemas,
    std::reference_wrapper<ast::expression const> filter,
    rmm::cuda_stream_view stream) const;

  /**
   * @brief Filters and reduces down to a selection of row groups
   *
   * The input `row_start` and `row_count` parameters will be recomputed and output as the valid
   * values based on the input row group list.
   *
   * @param row_group_indices Lists of row groups to read, one per source
   * @param row_start Starting row of the selection
   * @param row_count Total number of rows selected
   * @param output_dtypes Datatypes of of output columns
   * @param output_column_schemas schema indices of output columns
   * @param filter Optional AST expression to filter row groups based on Column chunk statistics
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @return A tuple of corrected row_start, row_count and list of row group indexes and its
   *         starting row
   */
  [[nodiscard]] std::tuple<int64_t, size_type, std::vector<row_group_info>> select_row_groups(
    host_span<std::vector<size_type> const> row_group_indices,
    int64_t row_start,
    std::optional<size_type> const& row_count,
    host_span<data_type const> output_dtypes,
    host_span<int const> output_column_schemas,
    std::optional<std::reference_wrapper<ast::expression const>> filter,
    rmm::cuda_stream_view stream) const;

  /**
   * @brief Filters and reduces down to a selection of columns
   *
   * @param use_names List of paths of column names to select; `nullopt` if user did not select
   * columns to read
   * @param filter_columns_names List of paths of column names that are present only in filter
   * @param include_index Whether to always include the PANDAS index column(s)
   * @param strings_to_categorical Type conversion parameter
   * @param timestamp_type_id Type conversion parameter
   *
   * @return input column information, output column information, list of output column schema
   * indices
   */
  [[nodiscard]] std::tuple<std::vector<input_column_info>,
                           std::vector<cudf::io::detail::inline_column_buffer>,
                           std::vector<size_type>>
  select_columns(std::optional<std::vector<std::string>> const& use_names,
                 std::optional<std::vector<std::string>> const& filter_columns_names,
                 bool include_index,
                 bool strings_to_categorical,
                 type_id timestamp_type_id) const;
};

/**
 * @brief Converts named columns to index reference columns
 *
 */
class named_to_reference_converter : public ast::detail::expression_transformer {
 public:
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
   * @brief Returns the AST to apply on Column chunk statistics.
   *
   * @return AST operation expression
   */
  [[nodiscard]] std::optional<std::reference_wrapper<ast::expression const>> get_converted_expr()
    const
  {
    return _stats_expr;
  }

 private:
  std::vector<std::reference_wrapper<ast::expression const>> visit_operands(
    std::vector<std::reference_wrapper<ast::expression const>> operands);

  std::unordered_map<std::string, size_type> column_name_to_index;
  std::optional<std::reference_wrapper<ast::expression const>> _stats_expr;
  // Using std::list or std::deque to avoid reference invalidation
  std::list<ast::column_reference> _col_ref;
  std::list<ast::operation> _operators;
};

/**
 * @brief Get the column names in expression object
 *
 * @param expr The optional expression object to get the column names from
 * @param skip_names The names of column names to skip in returned column names
 * @return The column names present in expression object except the skip_names
 */
[[nodiscard]] std::vector<std::string> get_column_names_in_expression(
  std::optional<std::reference_wrapper<ast::expression const>> expr,
  std::vector<std::string> const& skip_names);

}  // namespace cudf::io::parquet::detail
