/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include "aggregate_orc_metadata.hpp"
#include "reader_impl_chunking.hpp"
#include "reader_impl_helpers.hpp"

#include <io/utilities/column_buffer.hpp>

#include <cudf/io/datasource.hpp>
#include <cudf/io/detail/orc.hpp>
#include <cudf/io/orc.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>
#include <optional>
#include <vector>

namespace cudf::io::orc::detail {

/**
 * @brief Implementation for ORC reader.
 */
class reader::impl {
 public:
  /**
   * @brief Constructor from a dataset source with reader options.
   *
   * By using this constructor, each call to `read()` or `read_chunk()` will perform reading the
   * entire given file.
   *
   * @param sources Dataset sources
   * @param options Settings for controlling reading behavior
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource to use for device memory allocation
   */
  explicit impl(std::vector<std::unique_ptr<datasource>>&& sources,
                orc_reader_options const& options,
                rmm::cuda_stream_view stream,
                rmm::mr::device_memory_resource* mr);

  /**
   * @brief Constructor from a chunk read limit and an array of dataset sources with reader options.
   *
   * By using this constructor, the reader will support iterative (chunked) reading through
   * `has_next() ` and `read_chunk()`. For example:
   * ```
   *  do {
   *    auto const chunk = reader.read_chunk();
   *    // Process chunk
   *  } while (reader.has_next());
   *
   * ```
   *
   * Reading the whole given file at once through `read()` function is still supported if
   * `chunk_read_limit == 0` (i.e., no reading limit).
   * In such case, `read_chunk()` will also return rows of the entire file.
   *
   * @param chunk_read_limit Limit on total number of bytes to be returned per read,
   *        or `0` if there is no limit
   * @param sources Dataset sources
   * @param options Settings for controlling reading behavior
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource to use for device memory allocation
   */
  explicit impl(std::size_t chunk_read_limit,
                std::vector<std::unique_ptr<datasource>>&& sources,
                orc_reader_options const& options,
                rmm::cuda_stream_view stream,
                rmm::mr::device_memory_resource* mr);

  /**
   * @copydoc cudf::io::chunked_orc_reader::has_next
   */
  bool has_next();

  /**
   * @copydoc cudf::io::chunked_orc_reader::read_chunk
   */
  table_with_metadata read_chunk();

  /**
   * @brief Read an entire set or a subset of data and returns a set of columns
   *
   * @param skip_rows Number of rows to skip from the start
   * @param num_rows_opt Optional number of rows to read, or `std::nullopt` to read all rows
   * @param stripes Indices of individual stripes to load if non-empty
   * @return The set of columns along with metadata
   */
  table_with_metadata read(uint64_t skip_rows,
                           std::optional<size_type> const& num_rows_opt,
                           std::vector<std::vector<size_type>> const& stripes);

 private:
  /**
   * @brief Perform all the necessary data preprocessing before creating an output table.
   *
   * @param skip_rows Number of rows to skip from the start
   * @param num_rows_opt Optional number of rows to read, or `std::nullopt` to read all rows
   * @param stripes Indices of individual stripes to load if non-empty
   */
  void prepare_data(uint64_t skip_rows,
                    std::optional<size_type> const& num_rows_opt,
                    std::vector<std::vector<size_type>> const& stripes);

  /**
   * @brief Compute the ranges (begin and end rows) to read each chunk.
   */
  void compute_chunk_ranges();

  /**
   * @brief Create the output table metadata from file metadata.
   *
   * @return Columns' metadata to output with the table read from file
   */
  table_metadata make_output_metadata();

  /**
   * @brief Read a chunk of data from the input source and return an output table with metadata.
   *
   * This function is called internally and expects all preprocessing steps have already been done.
   *
   * @return The output table along with columns' metadata
   */
  table_with_metadata read_chunk_internal();

  rmm::cuda_stream_view const _stream;
  rmm::mr::device_memory_resource* const _mr;

  // Reader configs
  data_type const _timestamp_type;  // Override output timestamp resolution
  bool const _use_index;            // Enable or disable attempt to use row index for parsing
  bool const _use_np_dtypes;        // Enable or disable the conversion to numpy-compatible dtypes
  std::vector<std::string> const _decimal128_columns;  // Control decimals conversion

  // Intermediate data for internal processing.
  std::vector<std::unique_ptr<datasource>> const _sources;  // Unused but owns data for `_metadata`
  aggregate_orc_metadata _metadata;
  column_hierarchy const _selected_columns;  // Construct from `_metadata` thus declare after it
  reader_column_meta _col_meta;              // Track of orc mapping and child details
  file_intermediate_data _file_itm_data;
  chunk_read_info _chunk_read_info;  // Data for chunked reading.
  std::unique_ptr<table_metadata> _output_metadata;
  std::vector<std::vector<cudf::io::detail::column_buffer>> _out_buffers;
};

}  // namespace cudf::io::orc::detail
