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
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

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
                rmm::device_async_resource_ref mr);

  /**
   * @brief Read an entire set or a subset of data and returns a set of columns
   *
   * @return The set of columns along with metadata
   */
  table_with_metadata read();

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
   * // Alternatively
   *
   *  while (reader.has_next()) {
   *    auto const chunk = reader.read_chunk();
   *    // Process chunk
   *  }
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
                rmm::device_async_resource_ref mr);

  /**
   * @copydoc cudf::io::chunked_parquet_reader::has_next
   */
  bool has_next();

  /**
   * @copydoc cudf::io::chunked_parquet_reader::read_chunk
   */
  table_with_metadata read_chunk();

  // top level functions involved with ratcheting through the passes, subpasses
  // and output chunks of the read process
 private:
  /**
   * @brief The enum indicating whether the data sources are read all at once or chunk by chunk.
   */
  enum class read_mode { READ_ALL, CHUNKED_READ };

  /**
   * @brief Perform the necessary data preprocessing for parsing file later on.
   *
   * @param read_mode Value indicating if the data sources are read all at once or chunk by chunk
   */
  void prepare_data(read_mode mode);

  /**
   * @brief Preprocess step for the entire file.
   *
   * Only ever called once. This function reads in rowgroup and associated chunk
   * information and computes the schedule of top level passes (see `pass_intermediate_data`).
   *
   * @param read_mode Value indicating if the data sources are read all at once or chunk by chunk
   */
  void preprocess_file(read_mode mode);

  /**
   * @brief Ratchet the pass/subpass/chunk process forward.
   *
   * @param read_mode Value indicating if the data sources are read all at once or chunk by chunk
   */
  void handle_chunking(read_mode mode);

  /**
   * @brief Setup step for the next input read pass.
   *
   * A 'pass' is defined as a subset of row groups read out of the globally
   * requested set of all row groups.
   *
   * @param read_mode Value indicating if the data sources are read all at once or chunk by chunk
   */
  void setup_next_pass(read_mode mode);

  /**
   * @brief Setup step for the next decompression subpass.
   *
   * A 'subpass' is defined as a subset of pages within a pass that are
   * decompressed and decoded as a batch. Subpasses may be further subdivided
   * into output chunks.
   *
   * @param read_mode Value indicating if the data sources are read all at once or chunk by chunk
   *
   */
  void setup_next_subpass(read_mode mode);

  /**
   * @brief Read a chunk of data and return an output table.
   *
   * This function is called internally and expects all preprocessing steps have already been done.
   *
   * @param read_mode Value indicating if the data sources are read all at once or chunk by chunk
   * @return The output table along with columns' metadata
   */
  table_with_metadata read_chunk_internal(read_mode mode);

  // utility functions
 private:
  /**
   * @brief Read the set of column chunks to be processed for this pass.
   *
   * Does not decompress the chunk data.
   *
   * @return pair of boolean indicating if compressed chunks were found and a vector of futures for
   * read completion
   */
  std::pair<bool, std::vector<std::future<void>>> read_column_chunks();

  /**
   * @brief Read compressed data and page information for the current pass.
   */
  void read_compressed_data();

  /**
   * @brief Build string dictionary indices for a pass.
   *
   */
  void build_string_dict_indices();

  /**
   * @brief For list columns, generate estimated row counts for pages in the current pass.
   *
   * The row counts in the pages that come out of the file only reflect the number of values in
   * all of the rows in the page, not the number of rows themselves. In order to do subpass reading
   * more accurately, we would like to have a more accurate guess of the real number of rows per
   * page.
   */
  void generate_list_column_row_count_estimates();

  /**
   * @brief Perform some preprocessing for subpass page data and also compute the split locations
   * {skip_rows, num_rows} for chunked reading.
   *
   * There are several pieces of information we can't compute directly from row counts in
   * the parquet headers when dealing with nested schemas:
   * - The total sizes of all output columns at all nesting levels
   * - The starting output buffer offset for each page, for each nesting level
   *
   * For flat schemas, these values are computed during header decoding (see gpuDecodePageHeaders).
   *
   * @param read_mode Value indicating if the data sources are read all at once or chunk by chunk
   * @param chunk_read_limit Limit on total number of bytes to be returned per read,
   *        or `0` if there is no limit
   */
  void preprocess_subpass_pages(read_mode mode, size_t chunk_read_limit);

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
   * @brief Finalize the output table by adding empty columns for the non-selected columns in
   * schema.
   *
   * @param out_metadata The output table metadata
   * @param out_columns The columns for building the output table
   * @return The output table along with columns' metadata
   */
  table_with_metadata finalize_output(table_metadata& out_metadata,
                                      std::vector<std::unique_ptr<column>>& out_columns);

  /**
   * @brief Allocate data buffers for the output columns.
   *
   * @param read_mode Value indicating if the data sources are read all at once or chunk by chunk
   * @param skip_rows Crop all rows below skip_rows
   * @param num_rows Maximum number of rows to read
   */
  void allocate_columns(read_mode mode, size_t skip_rows, size_t num_rows);

  /**
   * @brief Calculate per-page offsets for string data
   *
   * @return Vector of total string data sizes for each column
   */
  std::vector<size_t> calculate_page_string_offsets();

  /**
   * @brief Converts the page data and outputs to columns.
   *
   * @param read_mode Value indicating if the data sources are read all at once or chunk by chunk
   * @param skip_rows Minimum number of rows from start
   * @param num_rows Number of rows to output
   */
  void decode_page_data(read_mode mode, size_t skip_rows, size_t num_rows);

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
   * @brief Given a set of pages that have had their sizes computed by nesting level and
   * a limit on total read size, generate a set of {skip_rows, num_rows} pairs representing
   * a set of reads that will generate output columns of total size <= `chunk_read_limit` bytes.
   */
  void compute_output_chunks_for_subpass();

  [[nodiscard]] bool has_more_work() const
  {
    return _file_itm_data.num_passes() > 0 &&
           _file_itm_data._current_input_pass < _file_itm_data.num_passes();
  }

 private:
  /**
   * @brief Check if the user has specified custom row bounds
   *
   * @param read_mode Value indicating if the data sources are read all at once or chunk by chunk
   * @return True if the user has specified custom row bounds
   */
  [[nodiscard]] bool uses_custom_row_bounds(read_mode mode) const
  {
    // TODO: `read_mode` is hardcoded to `true` when `read_mode::CHUNKED_READ` to enforce
    // `ComputePageSizes()` computation for all remaining chunks.
    return (mode == read_mode::READ_ALL)
             ? (_options.num_rows.has_value() or _options.skip_rows != 0)
             : true;
  }

  [[nodiscard]] bool is_first_output_chunk() const
  {
    return _file_itm_data._output_chunk_count == 0;
  }

  rmm::cuda_stream_view _stream;
  rmm::device_async_resource_ref _mr{rmm::mr::get_current_device_resource()};

  // Reader configs.
  struct {
    // timestamp_type
    data_type timestamp_type{type_id::EMPTY};
    // User specified reading rows/stripes selection.
    int64_t const skip_rows;
    std::optional<int64_t> num_rows;
    std::vector<std::vector<size_type>> row_group_indices;
  } const _options;

  // name to reference converter to extract AST output filter
  named_to_reference_converter _expr_conv{std::nullopt, table_metadata{}};

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

  // number of extra filter columns
  std::size_t _num_filter_only_columns{0};

  bool _strings_to_categorical = false;

  // are there usable page indexes available
  bool _has_page_index = false;

  std::optional<std::vector<reader_column_schema>> _reader_column_schema;

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

  std::size_t _output_chunk_read_limit{0};  // output chunk size limit in bytes
  std::size_t _input_pass_read_limit{0};    // input pass memory usage limit in bytes
};

}  // namespace cudf::io::parquet::detail
