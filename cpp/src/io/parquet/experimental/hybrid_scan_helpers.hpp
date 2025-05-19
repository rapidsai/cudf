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

namespace cudf::io::parquet::experimental::detail {

using aggregate_reader_metadata_base = parquet::detail::aggregate_reader_metadata;
using metadata_base                  = parquet::detail::metadata;

using io::detail::inline_column_buffer;
using parquet::detail::equality_literals_collector;
using parquet::detail::input_column_info;
using parquet::detail::row_group_info;

/**
 * @brief Class for parsing dataset metadata
 */
struct metadata : private metadata_base {
  explicit metadata(cudf::host_span<uint8_t const> footer_bytes);
  metadata_base get_file_metadata() && { return std::move(*this); }
};

class aggregate_reader_metadata : public aggregate_reader_metadata_base {
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
   * @brief Fetch the byte range of the page index in the Parquet file
   */
  [[nodiscard]] text::byte_range_info page_index_byte_range() const;

  /**
   * @brief Get the Parquet file metadata
   */
  [[nodiscard]] FileMetaData parquet_metadata() const;

  /**
   * @brief Setup and populate the page index structs in `FileMetaData`
   *
   * @param page_index_bytes Host span of Parquet page index buffer bytes
   */
  void setup_page_index(cudf::host_span<uint8_t const> page_index_bytes);

  /**
   * @brief Get the total number of top-level rows in the row groups
   *
   * @param row_group_indices Input row groups indices
   * @return Total number of top-level rows in the row groups
   */
  [[nodiscard]] size_type total_rows_in_row_groups(
    cudf::host_span<std::vector<size_type> const> row_group_indices) const;

  /**
   * @brief Filters and reduces down to the selection of payload columns
   *
   * @param payload_column_names List of paths of select payload column names, if any
   * @param filter_columns_names List of paths of column names present only in filter, if any
   * @param include_index Whether to always include the PANDAS index column(s)
   * @param strings_to_categorical Type conversion parameter
   * @param timestamp_type_id Type conversion parameter
   *
   * @return input column information, output column buffers, list of output column schema
   * indices
   */
  [[nodiscard]] std::
    tuple<std::vector<input_column_info>, std::vector<inline_column_buffer>, std::vector<size_type>>
    select_payload_columns(std::optional<std::vector<std::string>> const& payload_column_names,
                           std::optional<std::vector<std::string>> const& filter_column_names,
                           bool include_index,
                           bool strings_to_categorical,
                           type_id timestamp_type_id);

  /**
   * @brief Filter the row groups with statistics based on predicate filter
   *
   * @param row_group_indices Input row groups indices
   * @param output_dtypes Datatypes of output columns
   * @param output_column_schemas schema indices of output columns
   * @param filter AST expression to filter row groups based on Column chunk statistics
   * @param stream CUDA stream used for device memory operations and kernel launches
   *
   * @return Filtered row group indices, if any are filtered
   */
  [[nodiscard]] std::vector<std::vector<size_type>> filter_row_groups_with_stats(
    cudf::host_span<std::vector<size_type> const> row_group_indices,
    cudf::host_span<data_type const> output_dtypes,
    cudf::host_span<cudf::size_type const> output_column_schemas,
    std::reference_wrapper<ast::expression const> filter,
    rmm::cuda_stream_view stream) const;

  /**
   * @brief Get the bloom filter byte ranges, one per column chunk with equality predicate
   *
   * @param row_group_indices Input row groups indices
   * @param output_dtypes Datatypes of output columns
   * @param output_column_schemas schema indices of output columns
   * @param filter AST expression to filter row groups based on bloom filters
   *
   * @return Byte ranges of bloom filters, one per column chunk with equality predicate
   */
  [[nodiscard]] std::vector<cudf::io::text::byte_range_info> get_bloom_filter_bytes(
    cudf::host_span<std::vector<size_type> const> row_group_indices,
    cudf::host_span<data_type const> output_dtypes,
    cudf::host_span<cudf::size_type const> output_column_schemas,
    std::reference_wrapper<ast::expression const> filter);

  /**
   * @brief Get the dictionary page byte ranges, one per column chunk with (in)equality predicate
   *
   * @param row_group_indices Input row groups indices
   * @param output_dtypes Datatypes of output columns
   * @param output_column_schemas schema indices of output columns
   * @param filter AST expression to filter row groups based on dictionary pages
   *
   * @return Byte ranges of dictionary pages, one input column chunk with (in)equality predicate
   */
  [[nodiscard]] std::vector<cudf::io::text::byte_range_info> get_dictionary_page_bytes(
    cudf::host_span<std::vector<size_type> const> row_group_indices,
    cudf::host_span<data_type const> output_dtypes,
    cudf::host_span<cudf::size_type const> output_column_schemas,
    std::reference_wrapper<ast::expression const> filter);

  /**
   * @brief Filter the row groups using dictionaries based on predicate filter
   *
   * @param chunks Host device span of column chunk descriptors, one per column chunk with
   *               dictionary page and (in)equality predicate
   * @param pages Host device span of decoded page headers, one per column chunk with dictionary
   *              page and (in)equality predicate
   * @param row_group_indices Input row groups indices
   * @param literals Lists of literals, one per input column
   * @param output_dtypes Datatypes of output columns
   * @param dictionary_col_schemas schema indices of dictionary columns only
   * @param filter AST expression to filter row groups based on dictionary pages
   * @param stream CUDA stream used for device memory operations and kernel launches
   *
   * @return Filtered row group indices, if any are filtered
   */
  [[nodiscard]] std::vector<std::vector<cudf::size_type>> filter_row_groups_with_dictionary_pages(
    cudf::detail::hostdevice_span<parquet::detail::ColumnChunkDesc const> chunks,
    cudf::detail::hostdevice_span<parquet::detail::PageInfo const> pages,
    cudf::host_span<std::vector<cudf::size_type> const> row_group_indices,
    cudf::host_span<std::vector<ast::literal*> const> literals,
    cudf::host_span<data_type const> output_dtypes,
    cudf::host_span<cudf::size_type const> output_column_schemas,
    std::reference_wrapper<ast::expression const> filter,
    rmm::cuda_stream_view stream) const;

  /**
   * @brief Filter the row groups using bloom filters based on predicate filter
   *
   * @param bloom_filter_data Device buffers of bloom filters, one per input column chunk
   * @param row_group_indices Input row groups indices
   * @param output_dtypes Datatypes of output columns
   * @param output_column_schemas schema indices of output columns
   * @param filter AST expression to filter row groups based on bloom filters
   * @param stream CUDA stream used for device memory operations and kernel launches
   *
   * @return Filtered row group indices, if any are filtered
   */
  [[nodiscard]] std::vector<std::vector<size_type>> filter_row_groups_with_bloom_filters(
    cudf::host_span<rmm::device_buffer> bloom_filter_data,
    host_span<std::vector<size_type> const> row_group_indices,
    host_span<data_type const> output_dtypes,
    host_span<cudf::size_type const> output_column_schemas,
    std::reference_wrapper<ast::expression const> filter,
    rmm::cuda_stream_view stream) const;
};

/**
 * @brief Collects lists of equal and not-equal predicate literals in the AST expression, one list
 * per input table column. This is used in row group filtering based on dictionary pages.
 */
class dictionary_literals_collector : public equality_literals_collector {
 public:
  dictionary_literals_collector() = default;

  dictionary_literals_collector(ast::expression const& expr, cudf::size_type num_input_columns);

  using equality_literals_collector::visit;

  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::operation const& )
   */
  std::reference_wrapper<ast::expression const> visit(ast::operation const& expr) override;
};

}  // namespace cudf::io::parquet::experimental::detail
