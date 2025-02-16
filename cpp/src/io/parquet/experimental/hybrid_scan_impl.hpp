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

/**
 * @file reader_impl.hpp
 * @brief cuDF-IO Parquet reader class implementation header
 */

#pragma once

#include "hybrid_scan_helpers.hpp"
#include "io/parquet/parquet_gpu.hpp"
// #include "io/parquet/reader_impl_chunking.hpp"

#include <cudf/io/detail/experimental/hybrid_scan.hpp>
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

/**
 * @brief Implementation for Parquet reader
 */
class impl {
 public:
  /**
   * @brief Constructor from an array of dataset sources with reader options.
   *
   * By using this constructor, each call to `read()` or `read_chunk()` will perform reading the
   * entire given file.
   *
   * @param sources Dataset sources
   * @param options Settings for controlling reading behavior
   */
  explicit impl(cudf::host_span<uint8_t const> footer_bytes,
                cudf::host_span<uint8_t const> page_index_bytes,
                cudf::io::parquet_reader_options const& options);

  [[nodiscard]] std::vector<size_type> get_valid_row_groups(
    cudf::io::parquet_reader_options const& options) const;

  [[nodiscard]] std::vector<std::vector<size_type>> filter_row_groups_with_stats(
    cudf::host_span<std::vector<size_type> const> row_group_indices,
    cudf::io::parquet_reader_options const& options,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const;

  [[nodiscard]] std::pair<std::vector<cudf::io::text::byte_range_info>,
                          std::vector<cudf::io::text::byte_range_info>>
  get_secondary_filters(cudf::host_span<std::vector<size_type> const> row_group_indices,
                        cudf::io::parquet_reader_options const& options) const;

  [[nodiscard]] std::vector<std::vector<size_type>> filter_row_groups_with_dictionary_pages(
    std::vector<rmm::device_buffer>& dictionary_page_data,
    cudf::host_span<std::vector<size_type> const> row_group_indices,
    cudf::io::parquet_reader_options const& options,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const;

  [[nodiscard]] std::vector<std::vector<size_type>> filter_row_groups_with_bloom_filters(
    std::vector<rmm::device_buffer>& bloom_filter_data,
    cudf::host_span<std::vector<size_type> const> row_group_indices,
    cudf::io::parquet_reader_options const& options,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const;

 private:
  using table_metadata = cudf::io::table_metadata;

  /**
   * @brief Populate the output table metadata from the parquet file metadata.
   *
   * @param out_metadata The output table metadata to add to
   */
  void populate_metadata(table_metadata& out_metadata) const;

 private:
  using named_to_reference_converter = cudf::io::parquet::detail::named_to_reference_converter;
  using input_column_info            = cudf::io::parquet::detail::input_column_info;
  using inline_column_buffer         = cudf::io::detail::inline_column_buffer;

  std::unique_ptr<aggregate_reader_metadata> _metadata;

  // name to reference converter to extract AST output filter
  // named_to_reference_converter _expr_conv{std::nullopt, cudf::io::table_metadata{}};

  // input columns to be processed
  std::vector<input_column_info> _input_columns;
  // Buffers for generating output columns
  std::vector<inline_column_buffer> _output_buffers;
  // Buffers copied from `_output_buffers` after construction for reuse
  std::vector<inline_column_buffer> _output_buffers_template;
  // _output_buffers associated schema indices
  std::vector<int> _output_column_schemas;

  // _output_buffers associated metadata
  std::unique_ptr<table_metadata> _output_metadata;

  // number of extra filter columns
  std::size_t _num_filter_only_columns{0};

  cudf::io::parquet::detail::file_intermediate_data _file_itm_data;
};

}  // namespace cudf::experimental::io::parquet::detail
