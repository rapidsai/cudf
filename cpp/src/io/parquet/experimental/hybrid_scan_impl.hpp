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
#include "io/parquet/reader_impl_chunking.hpp"

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
    rmm::cuda_stream_view stream);

  [[nodiscard]] std::pair<std::vector<cudf::io::text::byte_range_info>,
                          std::vector<cudf::io::text::byte_range_info>>
  get_secondary_filters(cudf::host_span<std::vector<size_type> const> row_group_indices,
                        cudf::io::parquet_reader_options const& options);

  [[nodiscard]] std::vector<std::vector<size_type>> filter_row_groups_with_dictionary_pages(
    std::vector<rmm::device_buffer>& dictionary_page_data,
    cudf::host_span<std::vector<size_type> const> row_group_indices,
    cudf::io::parquet_reader_options const& options,
    rmm::cuda_stream_view stream);

  [[nodiscard]] std::vector<std::vector<size_type>> filter_row_groups_with_bloom_filters(
    std::vector<rmm::device_buffer>& bloom_filter_data,
    cudf::host_span<std::vector<size_type> const> row_group_indices,
    cudf::io::parquet_reader_options const& options,
    rmm::cuda_stream_view stream);

  [[nodiscard]] std::pair<std::unique_ptr<cudf::column>, std::vector<std::vector<bool>>>
  filter_data_pages_with_stats(cudf::host_span<std::vector<size_type> const> row_group_indices,
                               cudf::io::parquet_reader_options const& options,
                               rmm::cuda_stream_view stream,
                               rmm::device_async_resource_ref mr);

  [[nodiscard]] std::pair<std::vector<cudf::io::text::byte_range_info>,
                          std::vector<cudf::size_type>>
  get_filter_column_chunk_byte_ranges(
    cudf::host_span<std::vector<size_type> const> row_group_indices,
    cudf::io::parquet_reader_options const& options);

  [[nodiscard]] cudf::io::table_with_metadata materialize_filter_columns(
    cudf::host_span<std::vector<bool> const> data_page_validity,
    cudf::host_span<std::vector<size_type> const> row_group_indices,
    std::vector<rmm::device_buffer> column_chunk_buffers,
    cudf::mutable_column_view predicate,
    cudf::io::parquet_reader_options const& options,
    rmm::cuda_stream_view stream);

  static void update_predicate(cudf::column_view in_predicate,
                               cudf::mutable_column_view out_predicate,
                               rmm::cuda_stream_view stream);

  [[nodiscard]] std::pair<std::vector<cudf::io::text::byte_range_info>,
                          std::vector<cudf::size_type>>
  get_payload_column_chunk_byte_ranges(
    cudf::host_span<std::vector<size_type> const> row_group_indices,
    cudf::io::parquet_reader_options const& options);

  [[nodiscard]] cudf::io::table_with_metadata materialize_payload_columns(
    cudf::host_span<std::vector<size_type> const> row_group_indices,
    std::vector<rmm::device_buffer> column_chunk_buffers,
    cudf::mutable_column_view predicate,
    cudf::io::parquet_reader_options const& options,
    rmm::cuda_stream_view stream);

 private:
  using table_metadata = cudf::io::table_metadata;
  /**
   * @brief The enum indicating whether we are reading the predicate columns or the payload columns.
   */
  enum class read_mode { FILTER_COLUMNS, PAYLOAD_COLUMNS };

  /**
   * @brief Initialize the necessary options related internal variables for use later on.
   */
  void initialize_options(cudf::host_span<std::vector<size_type> const> row_group_indices,
                          cudf::io::parquet_reader_options const& options,
                          rmm::cuda_stream_view stream);

  void set_page_validity(cudf::host_span<std::vector<bool> const> data_page_validity);

  /**
   * @brief Select the columns to be read.
   *
   * @param read_mode Read mode
   * @param options Reader options
   */
  void select_columns(read_mode read_mode, cudf::io::parquet_reader_options const& options);

  /**
   * @brief Get the byte ranges for the input column chunks.
   *
   * @param row_group_indices The row groups to read
   * @return A pair of vectors containing the byte ranges and the source indices
   */
  [[nodiscard]] std::pair<std::vector<cudf::io::text::byte_range_info>,
                          std::vector<cudf::size_type>>
  get_input_column_chunk_byte_ranges(
    cudf::host_span<std::vector<size_type> const> row_group_indices) const;

  /**
   * @brief Invalidate output buffer bitmasks at row indices spanned by pruned pages
   *
   * @param page_validity A boolean host vector indicating if a subpass page is valid
   */
  void update_output_nullmasks_for_pruned_pages(cudf::host_span<bool const> page_validity);

  /**
   * @brief Perform the necessary data preprocessing for parsing file later on.
   *
   * @param read_mode Value indicating if the data sources are read all at once or chunk by chunk
   */
  void prepare_data(cudf::host_span<std::vector<size_type> const> row_group_indices,
                    std::vector<rmm::device_buffer> column_chunk_buffers,
                    cudf::io::parquet_reader_options const& options);

  /**
   * @brief Preprocess step for the entire file.
   *
   * Only ever called once. This function reads in rowgroup and associated chunk
   * information and computes the schedule of top level passes (see `pass_intermediate_data`).
   *
   * @param read_mode Value indicating if the data sources are read all at once or chunk by chunk
   */
  void prepare_row_groups(cudf::host_span<std::vector<size_type> const> row_group_indices,
                          cudf::io::parquet_reader_options const& options);

  /**
   * @brief Ratchet the pass/subpass/chunk process forward.
   *
   * @param read_mode Value indicating if the data sources are read all at once or chunk by chunk
   */
  void handle_chunking(std::vector<rmm::device_buffer> column_chunk_buffers,
                       cudf::io::parquet_reader_options const& options);

  /**
   * @brief Setup step for the next input read pass.
   *
   * A 'pass' is defined as a subset of row groups read out of the globally
   * requested set of all row groups.
   *
   * @param read_mode Value indicating if the data sources are read all at once or chunk by chunk
   */
  void setup_next_pass(std::vector<rmm::device_buffer> column_chunk_buffers,
                       cudf::io::parquet_reader_options const& options);

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
  void setup_next_subpass(cudf::io::parquet_reader_options const& options);

  /**
   * @brief Populate the output table metadata from the parquet file metadata.
   *
   * @param out_metadata The output table metadata to add to
   */
  void populate_metadata(table_metadata& out_metadata) const;

  /**
   * @brief Read the set of column chunks to be processed for this pass.
   *
   * Does not decompress the chunk data.
   *
   * @return boolean indicating if compressed chunks were found
   */
  bool read_column_chunks();

  /**
   * @brief Read compressed column chunks data and page information for the current pass.
   */
  void read_compressed_data(std::vector<rmm::device_buffer> column_chunk_buffers);

  /**
   * @brief Build string dictionary indices for a pass.
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
  void preprocess_subpass_pages(size_t chunk_read_limit);

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
   * @brief Reset the internal state of the reader.
   */
  void reset_internal_state();

  /**
   * @brief Finalize the output table by adding empty columns for the non-selected columns in
   * schema.
   *
   * @param out_metadata The output table metadata
   * @param out_columns The columns for building the output table
   * @return The output table along with columns' metadata
   */
  cudf::io::table_with_metadata finalize_output(read_mode read_mode,
                                                table_metadata& out_metadata,
                                                std::vector<std::unique_ptr<column>>& out_columns,
                                                cudf::mutable_column_view out_predicate);

  /**
   * @brief Allocate data buffers for the output columns.
   *
   * @param skip_rows Crop all rows below skip_rows
   * @param num_rows Maximum number of rows to read
   */
  void allocate_columns(size_t skip_rows, size_t num_rows);

  /**
   * @brief Calculate per-page offsets for string data
   *
   * @return Vector of total string data sizes for each column
   */
  cudf::detail::host_vector<size_t> calculate_page_string_offsets();

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
  void create_global_chunk_info(cudf::io::parquet_reader_options const& options);

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

  /**
   * @brief Read a chunk of data and return an output table.
   *
   * This function is called internally and expects all preprocessing steps have already been done.
   *
   * @param read_mode Value indicating if the data sources are read all at once or chunk by chunk
   * @return The output table along with columns' metadata
   */
  cudf::io::table_with_metadata read_chunk_internal(read_mode read_mode,
                                                    cudf::mutable_column_view out_predicate = {});

 private:
  /**
   * @brief Check if the user has specified custom row bounds
   *
   * @param read_mode Value indicating if the data sources are read all at once or chunk by chunk
   * @return True if the user has specified custom row bounds
   */
  [[nodiscard]] constexpr bool uses_custom_row_bounds() const { return false; }

  /**
   * @brief Check if this is the first output chunk
   *
   * @return True if this is the first output chunk
   */
  [[nodiscard]] bool is_first_output_chunk() const
  {
    return _file_itm_data._output_chunk_count == 0;
  }

 private:
  using named_to_reference_converter = cudf::io::parquet::detail::named_to_reference_converter;
  using input_column_info            = cudf::io::parquet::detail::input_column_info;
  using inline_column_buffer         = cudf::io::detail::inline_column_buffer;
  using reader_column_schema         = cudf::io::reader_column_schema;
  using file_intermediate_data       = cudf::io::parquet::detail::file_intermediate_data;
  using pass_intermediate_data       = cudf::io::parquet::detail::pass_intermediate_data;

  rmm::cuda_stream_view _stream;
  rmm::device_async_resource_ref _mr{cudf::get_current_device_resource_ref()};

  std::unique_ptr<aggregate_reader_metadata> _metadata;

  // name to reference converter to extract AST output filter
  named_to_reference_converter _expr_conv{std::nullopt, cudf::io::table_metadata{}};

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
  std::optional<std::vector<std::string>> _filter_columns_names;

  bool _strings_to_categorical = false;

  // are there usable page indexes available
  bool _has_page_index = false;

  size_type _num_sources{1};

  // timestamp_type
  cudf::data_type _timestamp_type{type_id::EMPTY};

  std::optional<std::vector<reader_column_schema>> _reader_column_schema;

  std::vector<bool> _page_validity;

  file_intermediate_data _file_itm_data;
  bool _file_preprocessed{false};
  bool _uses_custom_row_bounds{false};

  bool _is_filter_columns_selected{false};
  bool _is_payload_columns_selected{false};

  std::unique_ptr<pass_intermediate_data> _pass_itm_data;
};

}  // namespace cudf::experimental::io::parquet::detail
