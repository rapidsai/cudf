/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include "io/parquet/parquet_gpu.hpp"
#include "io/parquet/reader_impl_helpers.hpp"

#include <cudf/io/detail/utils.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <memory>
#include <optional>
#include <vector>

namespace cudf::experimental::io::parquet::detail {

using metadata_base                  = cudf::io::parquet::detail::metadata;
using aggregate_reader_metadata_base = cudf::io::parquet::detail::aggregate_reader_metadata;
using row_group_info                 = cudf::io::parquet::detail::row_group_info;
using input_column_info              = cudf::io::parquet::detail::input_column_info;
using inline_column_buffer           = cudf::io::detail::inline_column_buffer;
using equality_literals_collector    = cudf::io::parquet::detail::equality_literals_collector;

/**
 * @brief Class for parsing dataset metadata
 */
struct metadata : private metadata_base {
  explicit metadata(cudf::host_span<uint8_t const> footer_bytes);
  metadata_base get_file_metadata() && { return std::move(*this); }
};

class aggregate_reader_metadata : public aggregate_reader_metadata_base {
 private:
  /**
   * @brief Materializes column chunk dictionary pages into `cuco::static_set`s
   *
   * @param dictionary_page_data Dictionary page data device buffers for each input row group
   * @param input_row_group_indices Lists of input row groups, one per source
   * @param total_row_groups Total number of row groups in `input_row_group_indices`
   * @param output_dtypes Datatypes of output columns
   * @param dictionary_col_schemas schema indices of dictionary columns only
   * @param stream CUDA stream used for device memory operations and kernel launches
   *
   * @return A flattened list of `cuco::static_set_ref` device buffers for each filter column
   * across row groups
   */
  [[nodiscard]] std::vector<rmm::device_buffer> materialize_dictionaries(
    cudf::host_span<rmm::device_buffer> dictionary_page_data,
    host_span<std::vector<size_type> const> input_row_group_indices,
    host_span<data_type const> output_dtypes,
    host_span<int const> dictionary_col_schemas,
    rmm::cuda_stream_view stream) const;

  /**
   * @brief Filters the row groups using dictionary pages
   *
   * @param dictionaries `cuco::static_set_ref` device buffers for column chunk dictionary
   * @param input_row_group_indices Lists of input row groups, one per source
   * @param literals Lists of literals, one per input column
   * @param operators Lists of operators, one per input column
   * @param total_row_groups Total number of row groups in `input_row_group_indices`
   * @param output_dtypes Datatypes of output columns
   * @param dictionary_col_schemas schema indices of dictionary columns only
   * @param filter AST expression to filter row groups based on bloom filter membership
   * @param stream CUDA stream used for device memory operations and kernel launches
   *
   * @return A pair of filtered row group indices if any is filtered.
   */
  [[nodiscard]] std::optional<std::vector<std::vector<size_type>>> apply_dictionary_filter(
    cudf::host_span<rmm::device_buffer> dictionaries,
    host_span<std::vector<size_type> const> input_row_group_indices,
    host_span<std::vector<ast::literal*> const> literals,
    host_span<std::vector<ast::ast_operator> const> operators,
    size_type total_row_groups,
    host_span<data_type const> output_dtypes,
    host_span<int const> dictionary_col_schemas,
    std::reference_wrapper<ast::expression const> filter,
    rmm::cuda_stream_view stream) const;

 public:
  /**
   * @brief Constructor for aggregate_reader_metadata
   *
   * @param footer_bytes Host span of Parquet file footer buffer bytes
   * @param use_arrow_schema Whether to use Arrow schema
   * @param has_cols_from_mismatched_srcs Whether to have columns from mismatched sources
   */
  aggregate_reader_metadata(cudf::host_span<uint8_t const> footer_bytes,
                            bool use_arrow_schema,
                            bool has_cols_from_mismatched_srcs);

  /**
   * @brief Fetch the byte range of the `PageIndex` in the Parquet file
   */
  [[nodiscard]] cudf::io::text::byte_range_info get_page_index_bytes() const;

  /**
   * @brief Setup the PageIndex
   *
   * @param page_index_bytes Host span of Parquet `PageIndex` buffer bytes
   */
  void setup_page_index(cudf::host_span<uint8_t const> page_index_bytes);

  /**
   * @brief Filters and reduces down to the selection of filter columns
   *
   * @param filter_columns_names List of paths of column names that are present only in filter
   * @param include_index Whether to always include the PANDAS index column(s)
   * @param strings_to_categorical Type conversion parameter
   * @param timestamp_type_id Type conversion parameter
   *
   * @return input column information, output column information, list of output column schema
   * indices
   */
  [[nodiscard]] std::
    tuple<std::vector<input_column_info>, std::vector<inline_column_buffer>, std::vector<size_type>>
    select_filter_columns(std::optional<std::vector<std::string>> const& filter_columns_names,
                          bool include_index,
                          bool strings_to_categorical,
                          type_id timestamp_type_id);

  /**
   * @brief Filters and reduces down to the selection of payload columns
   *
   * @param column_names List of paths of column names that are present only in payload and filter
   * @param filter_columns_names List of paths of column names that are present only in filter
   * @param include_index Whether to always include the PANDAS index column(s)
   * @param strings_to_categorical Type conversion parameter
   * @param timestamp_type_id Type conversion parameter
   *
   * @return input column information, output column information, list of output column schema
   * indices
   */
  [[nodiscard]] std::
    tuple<std::vector<input_column_info>, std::vector<inline_column_buffer>, std::vector<size_type>>
    select_payload_columns(std::optional<std::vector<std::string>> const& column_names,
                           std::optional<std::vector<std::string>> const& filter_columns_names,
                           bool include_index,
                           bool strings_to_categorical,
                           type_id timestamp_type_id);

  /**
   * @brief Filters and reduces down to a selection of row groups
   *
   * The input `row_start` and `row_count` parameters will be recomputed and output as the valid
   * values based on the input row group list.
   *
   * @param row_group_indices Lists of row groups to read, one per source
   * @param row_start Starting row of the selection
   * @param row_count Total number of rows selected
   *
   * @return A tuple of corrected row_start, row_count, list of row group indexes and its
   *         starting row, list of number of rows per source, number of input row groups, and a
   *         struct containing the number of row groups surviving each predicate pushdown filter
   */
  [[nodiscard]] std::tuple<int64_t, size_type, std::vector<row_group_info>> select_row_groups(
    host_span<std::vector<size_type> const> row_group_indices,
    int64_t row_start,
    std::optional<size_type> const& row_count);

  /**
   * @brief Filter the row groups with statistics based on predicate filter
   *
   * @param row_group_indices Input row groups indices
   * @param output_dtypes Datatypes of output columns
   * @param output_column_schemas schema indices of output columns
   * @param filter Optional AST expression to filter row groups based on Column chunk statistics
   * @param stream CUDA stream used for device memory operations and kernel launches
   *
   * @return Filtered row group indices, if any are filtered
   */
  [[nodiscard]] std::vector<std::vector<size_type>> filter_row_groups_with_stats(
    host_span<std::vector<size_type> const> row_group_indices,
    host_span<data_type const> output_dtypes,
    host_span<int const> output_column_schemas,
    std::optional<std::reference_wrapper<ast::expression const>> filter,
    rmm::cuda_stream_view stream) const;

  /**
   * @brief Get the bloom filter byte ranges, one per input column chunk
   *
   * @param row_group_indices Input row groups indices
   * @param output_dtypes Datatypes of output columns
   * @param output_column_schemas schema indices of output columns
   * @param filter Optional AST expression to filter row groups based on bloom filters
   *
   * @return Byte ranges of bloom filters, one per input column chunk
   */
  [[nodiscard]] std::vector<cudf::io::text::byte_range_info> get_bloom_filter_bytes(
    cudf::host_span<std::vector<size_type> const> row_group_indices,
    host_span<data_type const> output_dtypes,
    host_span<int const> output_column_schemas,
    std::optional<std::reference_wrapper<ast::expression const>> filter);

  /**
   * @brief Get the dictionary page byte ranges, one per input column chunk
   *
   * @param row_group_indices Input row groups indices
   * @param output_dtypes Datatypes of output columns
   * @param output_column_schemas schema indices of output columns
   * @param filter Optional AST expression to filter row groups based on dictionary pages
   *
   * @return Byte ranges of dictionary pages, one per input column chunk
   */
  [[nodiscard]] std::vector<cudf::io::text::byte_range_info> get_dictionary_page_bytes(
    cudf::host_span<std::vector<size_type> const> row_group_indices,
    host_span<data_type const> output_dtypes,
    host_span<int const> output_column_schemas,
    std::optional<std::reference_wrapper<ast::expression const>> filter);

  /**
   * @brief Filter the row groups using dictionaries based on predicate filter
   *
   * @param dictionary_page_data Device buffers of dictionary pages, one per input column chunk
   * @param row_group_indices Input row groups indices
   * @param output_dtypes Datatypes of output columns
   * @param output_column_schemas schema indices of output columns
   * @param filter Optional AST expression to filter row groups based on dictionary pages
   * @param stream CUDA stream used for device memory operations and kernel launches
   *
   * @return Filtered row group indices, if any are filtered
   */
  [[nodiscard]] std::vector<std::vector<size_type>> filter_row_groups_with_dictionary_pages(
    std::vector<rmm::device_buffer>& dictionary_page_data,
    host_span<std::vector<size_type> const> row_group_indices,
    host_span<data_type const> output_dtypes,
    host_span<int const> output_column_schemas,
    std::optional<std::reference_wrapper<ast::expression const>> filter,
    rmm::cuda_stream_view stream) const;

  /**
   * @brief Filter the row groups using bloom filters based on predicate filter
   *
   * @param bloom_filter_data Device buffers of bloom filters, one per input column chunk
   * @param row_group_indices Input row groups indices
   * @param output_dtypes Datatypes of output columns
   * @param output_column_schemas schema indices of output columns
   * @param filter Optional AST expression to filter row groups based on bloom filters
   * @param stream CUDA stream used for device memory operations and kernel launches
   *
   * @return Filtered row group indices, if any are filtered
   */
  [[nodiscard]] std::vector<std::vector<size_type>> filter_row_groups_with_bloom_filters(
    std::vector<rmm::device_buffer>& bloom_filter_data,
    host_span<std::vector<size_type> const> row_group_indices,
    host_span<data_type const> output_dtypes,
    host_span<int const> output_column_schemas,
    std::optional<std::reference_wrapper<ast::expression const>> filter,
    rmm::cuda_stream_view stream) const;

  /**
   * @brief Filter data pages using statistics page-level statistics based on predicate filter
   *
   * @param row_group_indices Input row groups indices
   * @param output_dtypes Datatypes of output columns
   * @param output_column_schemas schema indices of output columns
   * @param filter Optional AST expression to filter data pages based on `PageIndex` statistics
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned column's device memory
   *
   * @return A boolean column representing a mask of rows surviving the predicate filter at
   *         page-level
   */
  [[nodiscard]] std::unique_ptr<cudf::column> filter_data_pages_with_stats(
    cudf::host_span<std::vector<size_type> const> row_group_indices,
    host_span<data_type const> output_dtypes,
    host_span<int const> output_column_schemas,
    std::optional<std::reference_wrapper<ast::expression const>> filter,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const;

  /**
   * @brief Computes which data pages need decoding to construct input columns based on the row mask
   *
   * Compute a vector of boolean vectors indicating which data pages need to be decoded to
   * construct each input column based on the row mask, one vector per column
   *
   * @param row_mask Boolean column indicating which rows need to be read after page-pruning
   * @param row_group_indices Input row groups indices
   * @param output_dtypes Datatypes of output columns
   * @param output_column_schemas schema indices of output columns
   * @param stream CUDA stream used for device memory operations and kernel launches
   *
   * @return A vector of boolean vectors indicating which data pages need to be decoded to produce
   *         the output table based on the input row mask, one per input column
   */
  [[nodiscard]] std::vector<std::vector<bool>> compute_data_page_mask(
    cudf::column_view row_mask,
    cudf::host_span<std::vector<size_type> const> row_group_indices,
    host_span<data_type const> output_dtypes,
    host_span<int const> output_column_schemas,
    rmm::cuda_stream_view stream) const;
};

/**
 * @brief Collects lists of equal and not-equal predicate literals in the AST expression, one list
 * per input table column. This is used in row group filtering based on dictionary pages.
 */
class dictionary_literals_and_operators_collector : public equality_literals_collector {
 public:
  dictionary_literals_and_operators_collector();

  dictionary_literals_and_operators_collector(ast::expression const& expr,
                                              cudf::size_type num_input_columns);

  using equality_literals_collector::visit;

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
   * @brief Returns the vectors of dictionary page filter literals in the AST expression, one per
   * input table column
   */
  [[nodiscard]] std::vector<std::vector<ast::literal*>> get_literals() = delete;

  /**
   * @brief Returns a pair of two vectors containing dictionary filter literals and operators
   * in the AST expression respectively, one per input table column
   */
  [[nodiscard]] std::pair<std::vector<std::vector<ast::literal*>>,
                          std::vector<std::vector<ast::ast_operator>>>
  get_literals_and_operators() &&;

 private:
  std::vector<std::vector<ast::ast_operator>> _operators;
};

}  // namespace cudf::experimental::io::parquet::detail
