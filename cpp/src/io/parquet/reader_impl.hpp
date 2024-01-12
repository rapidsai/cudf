/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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

#include "parquet_gpu.hpp"
#include "reader_impl_chunking.hpp"
#include "reader_impl_helpers.hpp"

#include <cudf/io/datasource.hpp>
#include <cudf/io/detail/parquet.hpp>
#include <cudf/io/parquet.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <memory>
#include <optional>
#include <vector>

namespace cudf::io::parquet::detail {

/**
 * @brief Implementation for Parquet reader
 */
class reader::impl {
 public:
  /**
   * @brief Constructor from an array of dataset sources with reader options.
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
                parquet_reader_options const& options,
                rmm::cuda_stream_view stream,
                rmm::mr::device_memory_resource* mr);

  /**
   * @brief Read an entire set or a subset of data and returns a set of columns
   *
   * @param skip_rows Number of rows to skip from the start
   * @param num_rows Number of rows to read
   * @param uses_custom_row_bounds Whether or not num_rows and skip_rows represents user-specific
   *        bounds
   * @param row_group_indices Lists of row groups to read, one per source
   * @param filter Optional AST expression to filter output rows
   *
   * @return The set of columns along with metadata
   */
  table_with_metadata read(int64_t skip_rows,
                           std::optional<size_type> const& num_rows,
                           bool uses_custom_row_bounds,
                           host_span<std::vector<size_type> const> row_group_indices,
                           std::optional<std::reference_wrapper<ast::expression const>> filter);

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
   * `chunk_read_limit == 0` (i.e., no reading limit) and `pass_read_limit == 0` (no temporary
   * memory limit) In such case, `read_chunk()` will also return rows of the entire file.
   *
   * @param chunk_read_limit Limit on total number of bytes to be returned per read,
   *        or `0` if there is no limit
   * @param pass_read_limit Limit on memory usage for the purposes of decompression and processing
   * of input, or `0` if there is no limit.
   * @param sources Dataset sources
   * @param options Settings for controlling reading behavior
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource to use for device memory allocation
   */
  explicit impl(std::size_t chunk_read_limit,
                std::size_t pass_read_limit,
                std::vector<std::unique_ptr<datasource>>&& sources,
                parquet_reader_options const& options,
                rmm::cuda_stream_view stream,
                rmm::mr::device_memory_resource* mr);

  /**
   * @copydoc cudf::io::chunked_parquet_reader::has_next
   */
  bool has_next();

  /**
   * @copydoc cudf::io::chunked_parquet_reader::read_chunk
   */
  table_with_metadata read_chunk();

 private:
  /**
   * @brief Perform the necessary data preprocessing for parsing file later on.
   *
   * @param skip_rows Number of rows to skip from the start
   * @param num_rows Number of rows to read, or `std::nullopt` to read all rows
   * @param uses_custom_row_bounds Whether or not num_rows and skip_rows represents user-specific
   *        bounds
   * @param row_group_indices Lists of row groups to read (one per source), or empty if read all
   * @param filter Optional AST expression to filter row groups based on column chunk statistics
   */
  void prepare_data(int64_t skip_rows,
                    std::optional<size_type> const& num_rows,
                    bool uses_custom_row_bounds,
                    host_span<std::vector<size_type> const> row_group_indices,
                    std::optional<std::reference_wrapper<ast::expression const>> filter);

  /**
   * @brief Create chunk information and start file reads
   *
   * @return pair of boolean indicating if compressed chunks were found and a vector of futures for
   * read completion
   */
  std::pair<bool, std::vector<std::future<void>>> read_and_decompress_column_chunks();

  /**
   * @brief Load and decompress the input file(s) into memory.
   */
  void load_and_decompress_data();

  /**
   * @brief Perform some preprocessing for page data and also compute the split locations
   * {skip_rows, num_rows} for chunked reading.
   *
   * There are several pieces of information we can't compute directly from row counts in
   * the parquet headers when dealing with nested schemas:
   * - The total sizes of all output columns at all nesting levels
   * - The starting output buffer offset for each page, for each nesting level
   *
   * For flat schemas, these values are computed during header decoding (see gpuDecodePageHeaders).
   *
   * @param uses_custom_row_bounds Whether or not num_rows and skip_rows represents user-specific
   *        bounds
   * @param chunk_read_limit Limit on total number of bytes to be returned per read,
   *        or `0` if there is no limit
   */
  void preprocess_pages(bool uses_custom_row_bounds, size_t chunk_read_limit);

  /**
   * @brief Allocate nesting information storage for all pages and set pointers to it.
   *
   * One large contiguous buffer of PageNestingInfo structs is allocated and
   * distributed among the PageInfo structs.
   *
   * Note that this gets called even in the flat schema case so that we have a
   * consistent place to store common information such as value counts, etc.
   */
  void allocate_nesting_info();

  /**
   * @brief Allocate space for use when decoding definition/repetition levels.
   *
   * One large contiguous buffer of data allocated and
   * distributed among the PageInfo structs.
   */
  void allocate_level_decode_space();

  /**
   * @brief Populate the output table metadata from the parquet file metadata.
   *
   * @param out_metadata The output table metadata to add to
   */
  void populate_metadata(table_metadata& out_metadata);

  /**
   * @brief Read a chunk of data and return an output table.
   *
   * This function is called internally and expects all preprocessing steps have already been done.
   *
   * @param uses_custom_row_bounds Whether or not num_rows and skip_rows represents user-specific
   *        bounds
   * @param filter Optional AST expression to filter output rows
   * @return The output table along with columns' metadata
   */
  table_with_metadata read_chunk_internal(
    bool uses_custom_row_bounds,
    std::optional<std::reference_wrapper<ast::expression const>> filter);

  /**
   * @brief Finalize the output table by adding empty columns for the non-selected columns in
   * schema.
   *
   * @param out_metadata The output table metadata
   * @param out_columns The columns for building the output table
   * @param filter Optional AST expression to filter output rows
   * @return The output table along with columns' metadata
   */
  table_with_metadata finalize_output(
    table_metadata& out_metadata,
    std::vector<std::unique_ptr<column>>& out_columns,
    std::optional<std::reference_wrapper<ast::expression const>> filter);

  /**
   * @brief Allocate data buffers for the output columns.
   *
   * @param skip_rows Crop all rows below skip_rows
   * @param num_rows Maximum number of rows to read
   * @param uses_custom_row_bounds Whether or not num_rows and skip_rows represents user-specific
   *        bounds
   */
  void allocate_columns(size_t skip_rows, size_t num_rows, bool uses_custom_row_bounds);

  /**
   * @brief Calculate per-page offsets for string data
   *
   * @return Vector of total string data sizes for each column
   */
  std::vector<size_t> calculate_page_string_offsets();

  /**
   * @brief Converts the page data and outputs to columns.
   *
   * @param skip_rows Minimum number of rows from start
   * @param num_rows Number of rows to output
   */
  void decode_page_data(size_t skip_rows, size_t num_rows);

  /**
   * @brief Creates file-wide parquet chunk information.
   *
   * Creates information about all chunks in the file, storing it in
   * the file-wide _file_itm_data structure.
   */
  void create_global_chunk_info();

  /**
   * @brief Computes all of the passes we will perform over the file.
   */
  void compute_input_passes();

  /**
   * @brief Close out the existing pass (if any) and prepare for the next pass.
   */
  void setup_next_pass();

  /**
   * @brief Given a set of pages that have had their sizes computed by nesting level and
   * a limit on total read size, generate a set of {skip_rows, num_rows} pairs representing
   * a set of reads that will generate output columns of total size <= `chunk_read_limit` bytes.
   */
  void compute_splits_for_pass();

 private:
  rmm::cuda_stream_view _stream;
  rmm::mr::device_memory_resource* _mr = nullptr;

  std::vector<std::unique_ptr<datasource>> _sources;
  std::unique_ptr<aggregate_reader_metadata> _metadata;

  // input columns to be processed
  std::vector<input_column_info> _input_columns;

  // Buffers for generating output columns
  std::vector<cudf::io::detail::inline_column_buffer> _output_buffers;

  // Buffers copied from `_output_buffers` after construction for reuse
  std::vector<cudf::io::detail::inline_column_buffer> _output_buffers_template;

  // _output_buffers associated schema indices
  std::vector<int> _output_column_schemas;

  // _output_buffers associated metadata
  std::unique_ptr<table_metadata> _output_metadata;

  bool _strings_to_categorical = false;
  std::optional<std::vector<reader_column_schema>> _reader_column_schema;
  data_type _timestamp_type{type_id::EMPTY};

  // chunked reading happens in 2 parts:
  //
  // At the top level, the entire file is divided up into "passes" omn which we try and limit the
  // total amount of temporary memory (compressed data, decompressed data) in use
  // via _input_pass_read_limit.
  //
  // Within a pass, we produce one or more chunks of output, whose maximum total
  // byte size is controlled by _output_chunk_read_limit.

  file_intermediate_data _file_itm_data;
  bool _file_preprocessed{false};

  std::unique_ptr<pass_intermediate_data> _pass_itm_data;
  bool _pass_preprocessed{false};

  std::size_t _output_chunk_read_limit{0};  // output chunk size limit in bytes
  std::size_t _input_pass_read_limit{0};    // input pass memory usage limit in bytes

  std::size_t _current_input_pass{0};  // current input pass index
  std::size_t _chunk_count{0};         // how many output chunks we have produced
};

}  // namespace cudf::io::parquet::detail
