/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/io/types.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <memory>
#include <utility>
#include <vector>

namespace CUDF_EXPORT cudf {
namespace io::parquet::experimental::detail {
/**
 * @brief Internal experimental Parquet reader optimized for highly selective filters, called a
 *        Hybrid Scan operation.
 */
class hybrid_scan_reader_impl;
}  // namespace io::parquet::experimental::detail
}  // namespace CUDF_EXPORT cudf

//! Using `byte_range_info` from cudf::io::text
using cudf::io::text::byte_range_info;

namespace CUDF_EXPORT cudf {
namespace io::parquet::experimental {
/**
 * @addtogroup io_readers
 * @{
 * @file
 */

/**
 * @brief Whether to compute and use a page mask using the row mask to skip decompression and
 * decoding of the masked pages
 */
enum class use_data_page_mask : bool {
  YES = true,  ///< Compute and use a data page mask
  NO  = false  ///< Do not compute or use a data page mask
};

/**
 * @brief The experimental parquet reader class to optimally read parquet files subject to
 *        highly selective filters, called a Hybrid Scan operation
 *
 * This class is designed to best exploit reductive optimization techniques to speed up reading
 * Parquet files subject to highly selective filters. The parquet file contents are read in two
 * passes. In the first pass, only the `filter` columns (i.e. columns that appear in the filter
 * expression) are read allowing pruning of row groups and filter column data pages using the filter
 * expression. In the second pass, only the `payload` columns (i.e. columns that do not appear in
 * the filter expression) are optimally read by applying the surviving row mask from the first pass
 * to prune payload column data pages.
 *
 * The following code snippets demonstrate how to use the experimental parquet reader.
 *
 * Start with an instance of the experimental reader with a span of parquet file footer
 * bytes and parquet reader options.
 * @code{.cpp}
 * // Example filter expression `A < 100`
 * auto filter_expression = cudf::ast::operation(cudf::ast::ast_operator::LESS,
 *                            column_name_reference{"A"}, literal{100});
 *
 * using namespace cudf::io;
 *
 * // Parquet reader options with empty source info
 * auto options = parquet_reader_options::builder(source_info(nullptr, 0))
 *                  .filter(filter_expression);
 *
 *  // Fetch parquet file footer bytes from the file
 *  cudf::host_span<uint8_t const> footer_bytes = fetch_parquet_footer_bytes();
 *
 * // Create the reader
 *  auto reader =
 *    std::make_unique<parquet::experimental::hybrid_scan_reader>(footer_bytes, options);
 * @endcode
 *
 * Metadata handling (OPTIONAL): Get a materialized parquet file footer metadata struct
 * (`FileMetaData`) from the reader to get insights into the parquet data as needed. Optionally,
 * set up the page index to materialize page level stats used for data page pruning.
 * @code{.cpp}
 * // Get Parquet file metadata from the reader
 * auto metadata = reader->parquet_metadata();
 *
 * // Example metadata use: Calculate the number of rows in the file
 * auto nrows = std::accumulate(metadata.row_groups.begin(),
 *                              metadata.row_groups.end(),
 *                              size_type{0},
 *                              [](auto sum, auto const& rg) {
 *                                return sum + rg.num_rows;
 *                              });
 *
 * // Get the page index byte range from the reader
 * auto page_index_byte_range = reader->page_index_byte_range();
 *
 * // Fetch the page index bytes from the parquet file
 * cudf::host_span<uint8_t const> page_index_bytes = fetch_parquet_bytes(page_index_byte_range);
 *
 * // Set up the page index
 * reader->setup_page_index(page_index_bytes);
 *
 * // A new `FileMetaData` struct with populated page index structs may be obtained
 * // using `parquet_metadata()` at this point. Page index may be set up at any time.
 * auto metadata_with_page_index = reader->parquet_metadata();
 * @endcode
 *
 * Row group pruning (OPTIONAL): Start with either a list of custom or all row group indices in the
 * parquet file and optionally filter it using a byte range and/or the filter expression using
 * column chunk statistics, dictionaries and bloom filters. Byte ranges for column chunk dictionary
 * pages and bloom filters within parquet file may be obtained via `secondary_filters_byte_ranges()`
 * function. The byte ranges may be read into device buffers and their device spans may be passed
 * to the row group filtration functions.
 * @code{.cpp}
 * // Start with a list of all parquet row group indices from the file footer
 * auto all_row_group_indices = reader->all_row_groups(options);
 *
 * // Span to track the indices of row groups currently at hand
 * auto current_row_group_indices = cudf::host_span<size_type>(all_row_group_indices);
 *
 * // Optional: Prune row group indices to the ones that start within the byte range
 * auto byte_range_filtered_row_group_indices = reader->filter_row_groups_with_byte_range(
 *   current_row_group_indices, options);
 *
 * // Update current row group indices to byte range filtered row group indices
 * current_row_group_indices = byte_range_filtered_row_group_indices;
 *
 * // Optional: Prune row group indices subject to filter expression using row group statistics
 * auto stats_filtered_row_group_indices =
 *   reader->filter_row_groups_with_stats(current_row_group_indices, options, stream);
 *
 * // Update current row group indices to now track the stats-filtered row group indices
 * current_row_group_indices = stats_filtered_row_group_indices;
 *
 * // Get byte ranges of bloom filters and dictionaries for the current row groups
 * auto [bloom_filter_byte_ranges, dict_page_byte_ranges] =
 *   reader->secondary_filters_byte_ranges(current_row_group_indices, options);
 *
 * // Optional: Prune row groups if we have valid dictionary pages
 * auto dictionary_page_filtered_row_group_indices = std::vector<size_type>{};
 *
 * if (dict_page_byte_ranges.size()) {
 *   // Fetch dictionary page byte ranges into device buffers and create spans
 *   std::vector<rmm::device_buffer> dictionary_page_buffers =
 *     fetch_device_buffers(dict_page_byte_ranges, stream, mr);
 *   auto dictionary_page_data = make_device_spans<uint8_t>(dictionary_page_buffers);
 *
 *   // Prune row groups using dictionaries
 *   dictionary_page_filtered_row_group_indices = reader->filter_row_groups_with_dictionary_pages(
 *     dictionary_page_data, current_row_group_indices, options, stream);
 *
 *   // Update current row group indices to dictionary page filtered row group indices
 *   current_row_group_indices = dictionary_page_filtered_row_group_indices;
 * }
 *
 * // Optional: Prune row groups if we have valid bloom filters
 * auto bloom_filtered_row_group_indices = std::vector<size_type>{};
 *
 * if (bloom_filter_byte_ranges.size()) {
 *   // Fetch bloom filter byte ranges into device buffers and create spans
 *   std::vector<rmm::device_buffer> bloom_filter_buffers =
 *     fetch_device_buffers(bloom_filter_byte_ranges, stream, mr);
 *   auto bloom_filter_data = make_device_spans<uint8_t>(bloom_filter_buffers);
 *
 *   // Prune row groups using bloom filters
 *   bloom_filtered_row_group_indices = reader->filter_row_groups_with_bloom_filters(
 *     bloom_filter_data, current_row_group_indices, options, stream);
 *
 *   // Update current row group indices to bloom filtered row group indices
 *   current_row_group_indices = bloom_filtered_row_group_indices;
 * }
 * @endcode
 *
 * Build an initial row mask: Once the row groups are filtered, the next step is to build an initial
 * BOOL8 row mask column indicating which rows in the current span of row groups survive in the
 * final table. This row mask column may contain all `true` values built using the
 * `build_all_true_row_mask()` function or it may contain a `true` value for only the rows that
 * survive the page-level statistics from the page index subject to the same filter as row groups
 * (needs page index to be set up using the `setup_page_index()` function). The size of this row
 * mask column must be equal to the total number of rows in the current span of row groups.
 * @code{.cpp}
 * // If not already done, get the page index byte range
 * auto page_index_byte_range = reader->page_index_byte_range();
 *
 * // If not already done, fetch the page index bytes from the parquet file
 * cudf::host_span<uint8_t const> page_index_bytes = fetch_parquet_bytes(page_index_byte_range);
 *
 * // If not already done, Set up the page index now
 * reader->setup_page_index(page_index_bytes);
 *
 * // Build a row mask column containing all `true` values
 * auto const num_rows = reader->total_rows_in_row_groups(current_row_group_indices);
 * auto row_mask = cudf::make_numeric_column(
 *     cudf::data_type{cudf::type_id::BOOL8}, num_rows, rmm::device_buffer{}, 0, stream, mr);
 *
 * // Alternatively, build a row mask column indicating only the rows that survive the page-level
 * statistics in the page index
 * row_mask = reader->build_row_mask_with_page_index_stats(current_row_group_indices, options,
 *                                                         stream, mr);
 * @endcode
 *
 * Materialize filter columns: Once we are done with pruning row groups and constructing the row
 * mask, the next step is to materialize filter columns into a table (first reader pass). This is
 * done using the `materialize_filter_columns()` function. This function requires a span of
 * device spans of column chunk data for the current list of row groups, and a mutable view
 * of the current row mask. The function optionally builds a mask for the current data pages using
 * the input row mask to skip decompression and decoding of the pruned pages based on the
 * `mask_data_pages` argument. The filter columns are then read into a table and filtered based on
 * the filter expression and the row mask is updated to only indicate the rows that survive in the
 * read table. The final table is returned. The byte ranges for the required column chunk data may
 * be obtained using the `filter_column_chunks_byte_ranges()` function and read into device buffers
 * with corresponding device spans.
 * @code{.cpp}
 * // Get byte ranges of column chunk byte ranges from the reader
 * auto const filter_column_chunk_byte_ranges =
 *   reader->filter_column_chunks_byte_ranges(current_row_group_indices, options);
 *
 * // Fetch column chunk data into device buffers and create spans
 * std::vector<rmm::device_buffer> filter_column_chunk_buffers =
 *   fetch_device_buffers(filter_column_chunk_byte_ranges, stream, mr);
 * auto filter_column_chunk_data = make_device_spans<uint8_t>(filter_column_chunk_buffers);
 *
 * // Materialize the table with only the filter columns
 * auto [filter_table, filter_metadata] =
 *   reader->materialize_filter_columns(current_row_group_indices,
 *                                      filter_column_chunk_data,
 *                                      row_mask->mutable_view(),
 *                                      use_data_page_mask::YES/NO,
 *                                      options,
 *                                      stream);
 * @endcode
 *
 * Materialize payload columns: Once the filter columns are materialized, the final step is to
 * materialize the payload columns into another table (second reader pass). This is done using the
 * `materialize_payload_columns()` function which is identical to the `materialize_filter_columns()`
 * in terms of functionality except that it accepts an immutable view of the row mask and uses it to
 * filter the read output table before returning it. The byte ranges for the required column chunk
 * data may be obtained using the `payload_column_chunks_byte_ranges()` function and read into
 * device buffers with corresponding device spans.
 * @code{.cpp}
 * // Get column chunk byte ranges from the reader
 * auto const payload_column_chunk_byte_ranges =
 *   reader->payload_column_chunks_byte_ranges(current_row_group_indices, options);
 *
 * // Fetch column chunk data into device buffers and create spans
 * std::vector<rmm::device_buffer> payload_column_chunk_buffers =
 *   fetch_device_buffers(payload_column_chunk_byte_ranges, stream, mr);
 * auto payload_column_chunk_data = make_device_spans<uint8_t>(payload_column_chunk_buffers);
 *
 * // Materialize the table with only the payload columns
 * auto [payload_table, payload_metadata] =
 *   reader->materialize_payload_columns(current_row_group_indices,
 *                                       payload_column_chunk_data,
 *                                       row_mask->view(),
 *                                       use_data_page_mask::YES/NO,
 *                                       options,
 *                                       stream);
 * @endcode
 *
 * Once both reader passes are complete, the filter and payload column tables may be trivially
 * combined by releasing the columns from both tables and moving them into a new cudf table.
 *
 * @note The performance advantage of this reader is most prominent when the filter expression
 * is highly selective, i.e. when the data in filter columns are at least partially ordered and the
 * number of rows that survive the filter is small compared to the total number of rows in the
 * parquet file. Otherwise, the performance is identical to the `cudf::io::read_parquet()` function.
 */
class hybrid_scan_reader {
 public:
  /**
   * @brief Constructor for the experimental parquet reader class to optimally read Parquet files
   * subject to highly selective filters
   *
   * @param footer_bytes Host span of parquet file footer bytes
   * @param options Parquet reader options
   */
  explicit hybrid_scan_reader(cudf::host_span<uint8_t const> footer_bytes,
                              parquet_reader_options const& options);

  /**
   * @brief Constructor for the experimental parquet reader class to optimally read Parquet files
   * subject to highly selective filters
   *
   * @param parquet_metadata Pre-populated Parquet file metadata
   * @param options Parquet reader options
   */
  explicit hybrid_scan_reader(FileMetaData const& parquet_metadata,
                              parquet_reader_options const& options);

  /**
   * @brief Destructor for the experimental parquet reader class
   */
  ~hybrid_scan_reader();

  /**
   * @brief Get the Parquet file footer metadata
   *
   * Returns the materialized Parquet file footer metadata struct. The footer will contain the
   * materialized page index if called after `setup_page_index()`.
   *
   * @return Parquet file footer metadata
   */
  [[nodiscard]] FileMetaData parquet_metadata() const;

  /**
   * @brief Get the byte range of the page index in the Parquet file
   *
   * @return Byte range of the page index
   */
  [[nodiscard]] byte_range_info page_index_byte_range() const;

  /**
   * @brief Setup the page index within the Parquet file metadata (`FileMetaData`)
   *
   * Materialize the `ColumnIndex` and `OffsetIndex` structs (collectively called the page index)
   * within the Parquet file metadata struct (returned by `parquet_metadata()`). The statistics
   * contained in page index can be used to prune data pages before decoding.
   *
   * @param page_index_bytes Host span of Parquet page index buffer bytes
   */
  void setup_page_index(cudf::host_span<uint8_t const> page_index_bytes) const;

  /**
   * @brief Get all available row groups from the parquet file
   *
   * @param options Parquet reader options
   * @return Vector of row group indices
   */
  [[nodiscard]] std::vector<size_type> all_row_groups(parquet_reader_options const& options) const;

  /**
   * @brief Get the total number of top-level rows in the row groups
   *
   * @param row_group_indices Input row groups indices
   * @return Total number of top-level rows in the row groups
   */
  [[nodiscard]] size_type total_rows_in_row_groups(
    cudf::host_span<size_type const> row_group_indices) const;

  /**
   * @brief Filter the row groups using the specified byte range specified by [`bytes_to_skip`,
   * `bytes_to_skip + bytes_to_read`)
   *
   * Filters the row groups such that only the row groups that start within the byte range are
   * selected. Note that the last selected row group may end beyond the byte range.
   *
   * @param row_group_indices Input row groups indices
   * @param options Parquet reader options
   * @return Filtered row group indices
   */
  [[nodiscard]] std::vector<size_type> filter_row_groups_with_byte_range(
    cudf::host_span<size_type const> row_group_indices,
    parquet_reader_options const& options) const;

  /**
   * @brief Filter the input row groups using column chunk statistics
   *
   * @param row_group_indices Input row groups indices
   * @param options Parquet reader options
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @return Filtered row group indices
   */
  [[nodiscard]] std::vector<size_type> filter_row_groups_with_stats(
    cudf::host_span<size_type const> row_group_indices,
    parquet_reader_options const& options,
    rmm::cuda_stream_view stream) const;

  /**
   * @brief Get byte ranges of bloom filters and dictionary pages (secondary filters) for row group
   *        pruning
   *
   * @note Device buffers for bloom filter byte ranges must be allocated using a 32 byte
   *       aligned memory resource
   *
   * @param row_group_indices Input row groups indices
   * @param options Parquet reader options
   * @return Pair of vectors of byte ranges of column chunk with bloom filters and dictionary
   *         pages subject to filter predicate
   */
  [[nodiscard]] std::pair<std::vector<byte_range_info>, std::vector<byte_range_info>>
  secondary_filters_byte_ranges(cudf::host_span<size_type const> row_group_indices,
                                parquet_reader_options const& options) const;

  /**
   * @brief Filter the row groups using column chunk dictionary pages
   *
   * @param dictionary_page_data Device spans of dictionary page data of column chunks with an
   *                             (in)equality predicate
   * @param row_group_indices Input row groups indices
   * @param options Parquet reader options
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @return Filtered row group indices
   */
  [[nodiscard]] std::vector<size_type> filter_row_groups_with_dictionary_pages(
    cudf::host_span<cudf::device_span<uint8_t const> const> dictionary_page_data,
    cudf::host_span<size_type const> row_group_indices,
    parquet_reader_options const& options,
    rmm::cuda_stream_view stream) const;

  /**
   * @brief Filter the row groups using column chunk bloom filters
   *
   * @note The `bloom_filter_data` device spans must point to 32-byte aligned addresses
   *
   * @param bloom_filter_data Device spans of bloom filter data of column chunks with an equality
   *                          predicate
   * @param row_group_indices Input row groups indices
   * @param options Parquet reader options
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @return Filtered row group indices
   */
  [[nodiscard]] std::vector<size_type> filter_row_groups_with_bloom_filters(
    cudf::host_span<cudf::device_span<uint8_t const> const> bloom_filter_data,
    cudf::host_span<size_type const> row_group_indices,
    parquet_reader_options const& options,
    rmm::cuda_stream_view stream) const;

  /**
   * @brief Builds a boolean (survival) column of size equal to the total number of rows in the row
   * groups containing all `true` values
   *
   * @param row_group_indices Input row groups indices
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned column's device memory
   * @return An all-true boolean (survival) column of size equal to the total number of rows in the
   * row groups
   */
  [[nodiscard]] std::unique_ptr<cudf::column> build_all_true_row_mask(
    cudf::host_span<size_type const> row_group_indices,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const;

  /**
   * @brief Builds a boolean column indicating surviving rows using page-level statistics in the
   * page index
   *
   * @param row_group_indices Input row groups indices
   * @param options Parquet reader options
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned column's device memory
   * @return A boolean column indicating which filter column rows survive the statistics in the page
   * index
   */
  [[nodiscard]] std::unique_ptr<cudf::column> build_row_mask_with_page_index_stats(
    cudf::host_span<size_type const> row_group_indices,
    parquet_reader_options const& options,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const;

  /**
   * @brief Get byte ranges of column chunks of filter columns
   *
   * @param row_group_indices Input row groups indices
   * @param options Parquet reader options
   * @return Vector of byte ranges to column chunks of filter columns
   */
  [[nodiscard]] std::vector<byte_range_info> filter_column_chunks_byte_ranges(
    cudf::host_span<size_type const> row_group_indices,
    parquet_reader_options const& options) const;

  /**
   * @brief Materializes filter columns and updates the input row mask to only the rows
   *        that exist in the output table
   *
   * @param row_group_indices Input row groups indices
   * @param column_chunk_data Device spans of column chunk data of filter columns
   * @param[in,out] row_mask Mutable boolean column indicating surviving rows from page pruning
   * @param mask_data_pages Whether to build and use a data page mask using the row mask
   * @param options Parquet reader options
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @return Table of materialized filter columns and metadata
   */
  [[nodiscard]] table_with_metadata materialize_filter_columns(
    cudf::host_span<size_type const> row_group_indices,
    cudf::host_span<cudf::device_span<uint8_t const> const> column_chunk_data,
    cudf::mutable_column_view& row_mask,
    use_data_page_mask mask_data_pages,
    parquet_reader_options const& options,
    rmm::cuda_stream_view stream) const;

  /**
   * @brief Get byte ranges of column chunks of payload columns
   *
   * @param row_group_indices Input row groups indices
   * @param options Parquet reader options
   * @return Vector of byte ranges to column chunks of payload columns
   */
  [[nodiscard]] std::vector<byte_range_info> payload_column_chunks_byte_ranges(
    cudf::host_span<size_type const> row_group_indices,
    parquet_reader_options const& options) const;

  /**
   * @brief Materialize payload columns and applies the row mask to the output table
   *
   * @param row_group_indices Input row groups indices
   * @param column_chunk_data Device spans of column chunk data of payload columns
   * @param row_mask Boolean column indicating which rows need to be read. All rows read if empty
   * @param mask_data_pages Whether to build and use a data page mask using the row mask
   * @param options Parquet reader options
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @return Table of materialized payload columns and metadata
   */
  [[nodiscard]] table_with_metadata materialize_payload_columns(
    cudf::host_span<size_type const> row_group_indices,
    cudf::host_span<cudf::device_span<uint8_t const> const> column_chunk_data,
    cudf::column_view const& row_mask,
    use_data_page_mask mask_data_pages,
    parquet_reader_options const& options,
    rmm::cuda_stream_view stream) const;

  /**
   * @brief Get byte ranges of column chunks of all (or selected) columns
   *
   * @param row_group_indices Input row groups indices
   * @param options Parquet reader options
   * @return Vector of byte ranges to column chunks of all (or selected) columns
   */
  [[nodiscard]] std::vector<byte_range_info> all_column_chunks_byte_ranges(
    cudf::host_span<size_type const> row_group_indices,
    parquet_reader_options const& options) const;

  /**
   * @brief Materializes all (or selected) columns and returns the final output table
   *
   * @param row_group_indices Input row groups indices
   * @param column_chunk_data Device spans of column chunk data of all columns
   * @param options Parquet reader options
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @return Table of all materialized columns and metadata
   */
  [[nodiscard]] table_with_metadata materialize_all_columns(
    cudf::host_span<size_type const> row_group_indices,
    cudf::host_span<cudf::device_span<uint8_t const> const> column_chunk_data,
    parquet_reader_options const& options,
    rmm::cuda_stream_view stream) const;
  /**
   * @brief Setup chunking information for filter columns and preprocess the input data pages
   *
   * @param chunk_read_limit Limit on total number of bytes to be returned per table chunk. `0` if
   * there is no limit
   * @param pass_read_limit Limit on the memory used for reading and decompressing data. `0` if
   * there is no limit
   * @param row_group_indices Input row groups indices
   * @param row_mask Boolean column indicating which rows need to be read. All rows read if empty
   * @param mask_data_pages Whether to build and use a data page mask using the row mask
   * @param column_chunk_data Device spans of column chunk data of filter columns
   * @param options Parquet reader options
   * @param stream CUDA stream used for device memory operations and kernel launches
   */
  void setup_chunking_for_filter_columns(
    std::size_t chunk_read_limit,
    std::size_t pass_read_limit,
    cudf::host_span<size_type const> row_group_indices,
    cudf::column_view const& row_mask,
    use_data_page_mask mask_data_pages,
    cudf::host_span<cudf::device_span<uint8_t const> const> column_chunk_data,
    parquet_reader_options const& options,
    rmm::cuda_stream_view stream) const;

  /**
   * @brief Materializes a chunk of filter columns and updates the corresponding range of input row
   * mask to only the rows that exist in the output table
   *
   * @param[in,out] row_mask Mutable boolean column indicating surviving rows from page pruning
   * @param stream CUDA stream used for device memory operations and kernel launches
   *
   * @return Table chunk of materialized filter columns and metadata
   */
  [[nodiscard]] table_with_metadata materialize_filter_columns_chunk(
    cudf::mutable_column_view& row_mask, rmm::cuda_stream_view stream) const;

  /**
   * @brief Setup chunking information for payload columns and preprocess the input data pages
   *
   * @param chunk_read_limit Limit on total number of bytes to be returned per table chunk. `0` if
   * there is no limit
   * @param pass_read_limit Limit on the memory used for reading and decompressing data. `0` if
   * there is no limit
   * @param row_group_indices Input row groups indices
   * @param row_mask Boolean column indicating which rows need to be read. All rows read if empty
   * @param mask_data_pages Whether to build and use a data page mask using the row mask
   * @param column_chunk_data Device spans of column chunk data of payload columns
   * @param options Parquet reader options
   * @param stream CUDA stream used for device memory operations and kernel launches
   */
  void setup_chunking_for_payload_columns(
    std::size_t chunk_read_limit,
    std::size_t pass_read_limit,
    cudf::host_span<size_type const> row_group_indices,
    cudf::column_view const& row_mask,
    use_data_page_mask mask_data_pages,
    cudf::host_span<cudf::device_span<uint8_t const> const> column_chunk_data,
    parquet_reader_options const& options,
    rmm::cuda_stream_view stream) const;

  /**
   * @brief Materializes a chunk of payload columns and applies the corresponding range of input row
   * mask to the output table chunk
   *
   * @param row_mask Boolean column indicating which rows need to be read. All rows read if empty
   * @param stream CUDA stream used for device memory operations and kernel launches
   *
   * @return Table chunk of materialized filter columns and metadata
   */
  [[nodiscard]] table_with_metadata materialize_payload_columns_chunk(
    cudf::column_view const& row_mask, rmm::cuda_stream_view stream) const;

  /**
   * @brief Check if there is any parquet data left to read for the current setup
   *
   * @return Boolean indicating if there is any data left to read
   */
  [[nodiscard]] bool has_next_table_chunk() const;

 private:
  std::unique_ptr<detail::hybrid_scan_reader_impl> _impl;
};

/** @} */  // end of group

}  // namespace io::parquet::experimental
}  // namespace CUDF_EXPORT cudf
