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
using equality_literals_collector    = parquet::detail::equality_literals_collector;
using inline_column_buffer           = io::detail::inline_column_buffer;
using input_column_info              = parquet::detail::input_column_info;
using metadata_base                  = parquet::detail::metadata;
using row_group_info                 = parquet::detail::row_group_info;

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
   * @brief Get the Parquet file metadata
   */
  [[nodiscard]] FileMetaData const& get_parquet_metadata() const;

  /**
   * @brief Setup the PageIndex
   *
   * @param page_index_bytes Host span of Parquet `PageIndex` buffer bytes
   */
  void setup_page_index(cudf::host_span<uint8_t const> page_index_bytes);

  /**
   * @brief Filters and reduces down to the selection of payload columns
   *
   * @param payload_column_names List of paths of select payload column names, if any
   * @param filter_columns_names List of paths of column names present only in filter, if any
   * @param include_index Whether to always include the PANDAS index column(s)
   * @param strings_to_categorical Type conversion parameter
   * @param timestamp_type_id Type conversion parameter
   *
   * @return input column information, output column information, list of output column schema
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
};

}  // namespace cudf::io::parquet::experimental::detail
