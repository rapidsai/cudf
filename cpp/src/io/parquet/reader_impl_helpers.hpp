/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
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
  metadata() = default;
  explicit metadata(datasource* source);
  void sanitize_schema();
};

/**
 * @brief Class to extract data types from arrow schema tree
 */
struct arrow_schema_data_types {
  std::vector<arrow_schema_data_types> children;
  data_type type{type_id::EMPTY};
};

/**
 * @brief Struct to store the number of row groups surviving each predicate pushdown filter.
 */
struct surviving_row_group_metrics {
  std::optional<size_type> after_stats_filter;  // number of surviving row groups after stats filter
  std::optional<size_type> after_bloom_filter;  // number of surviving row groups after bloom filter
};

class aggregate_reader_metadata {
 protected:
  std::vector<metadata> per_file_metadata;
  std::vector<std::unordered_map<std::string, std::string>> keyval_maps;
  std::vector<std::unordered_map<int32_t, int32_t>> schema_idx_maps;

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
   * @brief Initialize the vector of schema_idx maps.
   *
   * Initializes a vector of hash maps that will store the one-to-one mappings between the
   * schema_idx'es of the selected columns in the zeroth per_file_metadata (source) and each
   * kth per_file_metadata (destination) for k in range: [1, per_file_metadata.size()-1].
   *
   * @param has_cols_from_mismatched_srcs True if we are reading select cols from mismatched
   * parquet schemas.
   */
  [[nodiscard]] std::vector<std::unordered_map<int32_t, int32_t>> init_schema_idx_maps(
    bool has_cols_from_mismatched_srcs) const;

  /**
   * @brief Decodes and constructs the arrow schema from the ARROW_SCHEMA_KEY IPC message
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
  void column_info_for_row_group(row_group_info& rg_info, size_t chunk_start_row) const;

  /**
   * @brief Returns the required alignment for bloom filter buffers
   */
  [[nodiscard]] size_t get_bloom_filter_alignment() const;

  /**
   * @brief Reads bloom filter bitsets for the specified columns from the given lists of row
   * groups.
   *
   * @param sources Dataset sources
   * @param row_group_indices Lists of row groups to read bloom filters from, one per source
   * @param[out] bloom_filter_data List of bloom filter data device buffers
   * @param column_schemas Schema indices of columns whose bloom filters will be read
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param aligned_mr Aligned device memory resource to allocate bloom filter buffers
   *
   * @return A flattened list of bloom filter bitset device buffers for each predicate column across
   * row group
   */
  [[nodiscard]] std::vector<rmm::device_buffer> read_bloom_filters(
    host_span<std::unique_ptr<datasource> const> sources,
    host_span<std::vector<size_type> const> row_group_indices,
    host_span<int const> column_schemas,
    size_type num_row_groups,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref aligned_mr) const;

  /**
   * @brief Collects Parquet types for the columns with the specified schema indices
   *
   * @param row_group_indices Lists of row groups, once per source
   * @param column_schemas Schema indices of columns whose types will be collected
   *
   * @return A list of parquet types for the columns matching the provided schema indices
   */
  [[nodiscard]] std::vector<Type> get_parquet_types(
    host_span<std::vector<size_type> const> row_group_indices,
    host_span<int const> column_schemas) const;

  /**
   * @brief Filters the row groups using row bounds (`skip_rows` and `num_rows`)
   *
   * @param rows_to_skip Number of rows to skip
   * @param rows_to_read Number of rows to read
   *
   * @return A tuple of number of rows to skip from the first surviving row group's row offset,
   * a vector of surviving row group indices, and two vectors of effective (trimmed) row counts and
   * offsets across surviving row group indices respectively
   */
  [[nodiscard]] std::tuple<int64_t,
                           std::vector<std::vector<size_type>>,
                           std::vector<std::vector<size_t>>,
                           std::vector<std::vector<size_t>>>
  apply_row_bounds_filter(cudf::host_span<std::vector<size_type> const> input_row_group_indices,
                          int64_t rows_to_skip,
                          int64_t rows_to_read) const;

  /**
   * @brief Filters the row groups using stats filter
   *
   * @param input_row_group_indices Lists of input row groups, one per source
   * @param total_row_groups Total number of row groups in `input_row_group_indices`
   * @param output_dtypes Datatypes of output columns
   * @param output_column_schemas schema indices of output columns
   * @param filter AST expression to filter row groups based on bloom filter membership
   * @param stream CUDA stream used for device memory operations and kernel launches
   *
   * @return Surviving row group indices if any of them are filtered.
   */
  [[nodiscard]] std::optional<std::vector<std::vector<size_type>>> apply_stats_filters(
    host_span<std::vector<size_type> const> input_row_group_indices,
    size_type total_row_groups,
    host_span<data_type const> output_dtypes,
    host_span<int const> output_column_schemas,
    std::reference_wrapper<ast::expression const> filter,
    rmm::cuda_stream_view stream) const;

  /**
   * @brief Filters the row groups using bloom filters
   *
   * @param bloom_filter_data Bloom filter data device buffers for each input row group
   * @param input_row_group_indices Lists of input row groups, one per source
   * @param literals Lists of equality literals, one per each input row group
   * @param total_row_groups Total number of row groups in `input_row_group_indices`
   * @param output_dtypes Datatypes of output columns
   * @param bloom_filter_col_schemas Schema indices of bloom filter columns only
   * @param filter AST expression to filter row groups based on bloom filter membership
   * @param stream CUDA stream used for device memory operations and kernel launches
   *
   * @return Surviving row group indices if any of them are filtered.
   */
  [[nodiscard]] std::optional<std::vector<std::vector<size_type>>> apply_bloom_filters(
    cudf::host_span<rmm::device_buffer> bloom_filter_data,
    host_span<std::vector<size_type> const> input_row_group_indices,
    host_span<std::vector<ast::literal*> const> literals,
    size_type total_row_groups,
    host_span<data_type const> output_dtypes,
    host_span<int const> bloom_filter_col_schemas,
    std::reference_wrapper<ast::expression const> filter,
    rmm::cuda_stream_view stream) const;

 public:
  aggregate_reader_metadata(host_span<std::unique_ptr<datasource> const> sources,
                            bool use_arrow_schema,
                            bool has_cols_from_mismatched_srcs);

  [[nodiscard]] RowGroup const& get_row_group(size_type row_group_index, size_type src_idx) const;

  /**
   * @brief Extracts the schema_idx'th column chunk metadata from row_group_index'th row group of
   * the src_idx'th file.
   *
   * Extracts the schema_idx'th column chunk metadata from the specified row group index of the
   * src_idx'th file. Note that the schema_idx is actually the index in the zeroth file which may
   * not be the same in all files, in which case, the schema_idx is mapped to the corresponding
   * index in the src_idx'th file and returned. A range_error error is thrown if schema_idx
   * doesn't exist or isn't mapped to the src_idx file.
   *
   * @param row_group_index The row group index in the file to extract column chunk metadata from.
   * @param src_idx The per_file_metadata index to extract extract column chunk metadata from.
   * @param schema_idx The schema_idx of the column chunk to be extracted
   *
   * @return The requested column chunk metadata or a range_error error if the schema index isn't
   * valid.
   */
  [[nodiscard]] ColumnChunkMetaData const& get_column_metadata(size_type row_group_index,
                                                               size_type src_idx,
                                                               int schema_idx) const;

  /**
   * @brief Extracts high-level metadata for all row groups
   *
   * @return List of maps containing metadata information for each row group
   */
  [[nodiscard]] std::vector<std::unordered_map<std::string, int64_t>> get_rowgroup_metadata() const;

  /**
   * @brief Maps leaf column names to vectors of `total_uncompressed_size` fields from their all
   *        column chunks
   *
   * @return Map of leaf column names to vectors of `total_uncompressed_size` fields from all their
   *         column chunks
   */
  [[nodiscard]] std::unordered_map<std::string, std::vector<int64_t>> get_column_chunk_metadata()
    const;

  /**
   * @brief Get total number of rows across all files
   *
   * @return Total number of rows across all files
   */
  [[nodiscard]] auto get_num_rows() const { return num_rows; }

  /**
   * @brief Get total number of row groups across all files
   *
   * @return Total number of row groups across all files
   */
  [[nodiscard]] auto get_num_row_groups() const { return num_row_groups; }

  /**
   * @brief Get the number of row groups per file
   *
   * @return Number of row groups per file
   */
  [[nodiscard]] std::vector<size_type> get_num_row_groups_per_file() const;

  /**
   * @brief Checks if a schema index from 0th source is mapped to the specified file index
   *
   * @param schema_idx The index of the SchemaElement in the zeroth file.
   * @param pfm_idx The index of the file (per_file_metadata) to check mappings for.
   *
   * @return True if schema index is mapped
   */
  [[nodiscard]] bool is_schema_index_mapped(int schema_idx, int pfm_idx) const;

  /**
   * @brief Maps schema index from 0th source file to the specified file index
   *
   * @param schema_idx The index of the SchemaElement in the zeroth file.
   * @param pfm_idx The index of the file (per_file_metadata) to map the schema_idx to.
   *
   * @return Mapped schema index
   */
  [[nodiscard]] int map_schema_index(int schema_idx, int pfm_idx) const;

  /**
   * @brief Extracts the schema_idx'th SchemaElement from the pfm_idx'th file
   *
   * @param schema_idx The index of the SchemaElement to be extracted.
   * @param pfm_idx The index of the per_file_metadata to extract SchemaElement from, default = 0 if
   * not specified.
   *
   * @return The requested SchemaElement or an error if invalid schema_idx or pfm_idx.
   */
  [[nodiscard]] auto const& get_schema(int schema_idx, int pfm_idx = 0) const
  {
    CUDF_EXPECTS(
      schema_idx >= 0 and pfm_idx >= 0 and std::cmp_less(pfm_idx, per_file_metadata.size()),
      "Parquet reader encountered an invalid schema_idx or pfm_idx",
      std::out_of_range);
    return per_file_metadata[pfm_idx].schema[schema_idx];
  }

  [[nodiscard]] auto const& get_key_value_metadata() const& { return keyval_maps; }
  [[nodiscard]] auto&& get_key_value_metadata() && { return std::move(keyval_maps); }

  /**
   * @brief Gets the concrete nesting depth of output cudf columns.
   *
   * Gets the nesting depth of the output cudf column for the given schema.
   * The nesting depth must be equal for the given schema_index across all sources.
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
   * @brief Filters the row groups using stats and bloom filters based on predicate filter
   *
   * @param sources Lists of input datasources
   * @param input_row_group_indices Lists of input row groups, one per source
   * @param total_row_groups Total number of row groups in `input_row_group_indices`
   * @param output_dtypes Datatypes of output columns
   * @param output_column_schemas schema indices of output columns
   * @param filter AST expression to filter row groups based on Column chunk statistics
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @return A pair of a list of filtered row group indices if any are filtered, and a struct
   *         containing the number of row groups surviving each predicate pushdown filter
   */
  [[nodiscard]] std::pair<std::optional<std::vector<std::vector<size_type>>>,
                          surviving_row_group_metrics>
  filter_row_groups(host_span<std::unique_ptr<datasource> const> sources,
                    host_span<std::vector<size_type> const> input_row_group_indices,
                    size_type total_row_groups,
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
   * @param sources Lists of input datasources
   * @param row_group_indices Lists of row groups to read, one per source
   * @param row_start Starting row of the selection
   * @param row_count Total number of rows selected
   * @param output_dtypes Datatypes of of output columns
   * @param output_column_schemas schema indices of output columns
   * @param filter Optional AST expression to filter row groups based on Column chunk statistics
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @return A tuple of corrected row_start, row_count, list of row group indexes and its
   *         starting row, list of number of rows per source, number of input row groups, and a
   *         struct containing the number of row groups surviving each predicate pushdown filter
   */
  [[nodiscard]] std::tuple<int64_t,
                           size_t,
                           std::vector<row_group_info>,
                           std::vector<size_t>,
                           size_type,
                           surviving_row_group_metrics>
  select_row_groups(host_span<std::unique_ptr<datasource> const> sources,
                    host_span<std::vector<size_type> const> row_group_indices,
                    int64_t row_start,
                    std::optional<int64_t> const& row_count,
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
   * @return input column information, output column buffers, list of output column schema
   * indices
   */
  [[nodiscard]] std::tuple<std::vector<input_column_info>,
                           std::vector<cudf::io::detail::inline_column_buffer>,
                           std::vector<size_type>>
  select_columns(std::optional<std::vector<std::string>> const& use_names,
                 std::optional<std::vector<std::string>> const& filter_columns_names,
                 bool include_index,
                 bool strings_to_categorical,
                 type_id timestamp_type_id);
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
   * @brief Returns the converted AST expression
   *
   * @return AST operation expression
   */
  [[nodiscard]] std::optional<std::reference_wrapper<ast::expression const>> get_converted_expr()
    const
  {
    return _converted_expr;
  }

 private:
  std::vector<std::reference_wrapper<ast::expression const>> visit_operands(
    cudf::host_span<std::reference_wrapper<ast::expression const> const> operands);

  std::unordered_map<std::string, size_type> column_name_to_index;
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
 * @brief Get the column names in expression object
 *
 * @param expr The optional expression object to get the column names from
 * @param skip_names The names of column names to skip in returned column names
 * @return The column names present in expression object except the skip_names
 */
[[nodiscard]] std::vector<std::string> get_column_names_in_expression(
  std::optional<std::reference_wrapper<ast::expression const>> expr,
  std::vector<std::string> const& skip_names);

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
