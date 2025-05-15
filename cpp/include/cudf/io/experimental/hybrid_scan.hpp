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

#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/io/types.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>

#include <thrust/host_vector.h>

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
 * parquet file and optionally filter it subject to filter expression using column chunk statistics,
 * dictionaries and bloom filters. Byte ranges for column chunk dictionary pages and bloom filters
 * within parquet file may be obtained via `secondary_filters_byte_ranges()` function. The byte
 * ranges may be read into a corresponding vector of device buffers and passed to the corresponding
 * row group filtration function.
 * @code{.cpp}
 * // Start with a list of all parquet row group indices from the file footer
 * auto all_row_group_indices = reader->all_row_groups(options);
 *
 * // Span to track the indices of row groups currently at hand
 * auto current_row_group_indices = cudf::host_span<size_type>(all_row_group_indices);
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
 *   // Fetch dictionary page byte ranges into device buffers
 *   std::vector<rmm::device_buffer> dictionary_page_data =
 *     fetch_device_buffers(dict_page_byte_ranges);
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
 *   // Fetch bloom filter byte ranges into device buffers
 *   std::vector<rmm::device_buffer> bloom_filter_data =
 *     fetch_device_buffers(bloom_filter_byte_ranges);
 *
 *  // Prune row groups using bloom filters
 *   bloom_filtered_row_group_indices = reader->filter_row_groups_with_bloom_filters(
 *     bloom_filter_data, current_row_group_indices, options, stream);
 *
 *   // Update current row group indices to bloom filtered row group indices
 *   current_row_group_indices = bloom_filtered_row_group_indices;
 * }
 * @endcode
 *
 * Filter column page pruning (OPTIONAL): Once the row groups are filtered, the next step is to
 * optionally prune the data pages within the current span of row groups subject to the same filter
 * expression using page statistics contained in the page index of the parquet file. To get started,
 * first set up the page index using the `setup_page_index()` function if not previously done and
 * then filter the data pages using the `filter_data_pages_with_stats()` function. This function
 * returns a row mask. i.e. BOOL8 column indicating which rows may survive in the materialized table
 * of filter columns (first reader pass), and a data page mask. i.e. a vector of boolean host
 * vectors indicating which data pages for each filter column need to be processed to materialize
 * the table filter columns (first reader pass).
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
 * // Optional: Prune filter column data pages using statistics in page index
 * auto [row_mask, data_page_mask] =
 *   reader->filter_data_pages_with_stats(current_row_group_indices, options, stream, mr);
 * @endcode
 *
 * Materialize filter columns: Once we are finished with pruning row groups and filter column data
 * pages, the next step is to materialize filter columns into a table (first reader pass). This is
 * done using the `materialize_filter_columns()` function. This function requires a vector of device
 * buffers containing column chunk data for the current list of row groups, and the data page and
 * row masks obtained from the page pruning step. The function returns a table of materialized
 * filter columns and also updates the row mask column to only the valid rows that satisfy the
 * filter expression. If no row group pruning is needed, pass a span of all row group indices from
 * `all_row_groups()` function as the current list of row groups. Similarly, if no page pruning is
 * desired, pass an empty span as data page mask and a mutable view of a BOOL8 column of size equal
 * to total number of rows in the current row groups list (computed by `total_rows_in_row_groups()`)
 * containing all `true` values as row mask. Further, the byte ranges for the required column chunk
 * data may be obtained using the `filter_column_chunks_byte_ranges()` function and read into a
 * corresponding vector of vectors of device buffers.
 * @code{.cpp}
 * // Get byte ranges of column chunk byte ranges from the reader
 * auto const filter_column_chunk_byte_ranges =
 *   reader->filter_column_chunks_byte_ranges(current_row_group_indices, options);
 *
 * // Fetch column chunk device buffers from the input buffer
 * auto filter_column_chunk_buffers =
 *   fetch_device_buffers(filter_column_chunk_byte_ranges);
 *
 * // Materialize the table with only the filter columns
 * auto [filter_table, filter_metadata] =
 *   reader->materialize_filter_columns(data_page_mask,
 *                                      current_row_group_indices,
 *                                      std::move(filter_column_chunk_buffers),
 *                                      row_mask->mutable_view(),
 *                                      options,
 *                                      stream);
 * @endcode
 *
 * Materialize payload columns: Once the filter columns are materialized, the final step is to
 * materialize the payload columns into another table (second reader pass). This is done using the
 * `materialize_payload_columns()` function. This function requires a vector of device buffers
 * containing column chunk data for the current list of row groups, and the updated row mask from
 * the `materialize_filter_columns()`. The function uses the row mask - may be a BOOL8 column of
 * size equal to total number of rows in the current row groups list containing all `true` values if
 * no pruning is desired - to internally prune payload column data pages and mask the materialized
 * payload columns to the desired rows. Similar to the first reader pass, the byte ranges for the
 * required column chunk data may be obtained using the `payload_column_chunks_byte_ranges()`
 * function and read into a corresponding vector of vectors of device buffers.
 * @code{.cpp}
 * // Get column chunk byte ranges from the reader
 * auto const payload_column_chunk_byte_ranges =
 *   reader->payload_column_chunks_byte_ranges(current_row_group_indices, options);
 *
 * // Fetch column chunk device buffers from the input buffer
 * auto payload_column_chunk_buffers =
 *   fetch_device_buffers(payload_column_chunk_byte_ranges);
 *
 * // Materialize the table with only the payload columns
 * auto [payload_table, payload_metadata] =
 *   reader->materialize_payload_columns(current_row_group_indices,
 *                                       std::move(payload_column_chunk_buffers),
 *                                       row_mask->view(),
 *                                       options,
 *                                       stream);
 * @endcode
 *
 * Once both reader passes are complete, the filter and payload column tables may be trivially
 * combined by releasing the columns from both tables and moving them into a new cudf table.
 *
 * @note The performance advantage of this reader is most pronounced when the filter expression
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
   * @param dictionary_page_data Device buffers containing dictionary page data of column chunks
   *                             with (in)equality predicate
   * @param row_group_indices Input row groups indices
   * @param options Parquet reader options
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @return Filtered row group indices
   */
  [[nodiscard]] std::vector<size_type> filter_row_groups_with_dictionary_pages(
    cudf::host_span<rmm::device_buffer> dictionary_page_data,
    cudf::host_span<size_type const> row_group_indices,
    parquet_reader_options const& options,
    rmm::cuda_stream_view stream) const;

  /**
   * @brief Filter the row groups using column chunk bloom filters
   *
   * @note The `bloom_filter_data` device buffers must be allocated using a 32
   *       byte aligned memory resource
   *
   * @param bloom_filter_data Device buffers containing bloom filter data of column chunks with
   *                          an equality predicate
   * @param row_group_indices Input row groups indices
   * @param options Parquet reader options
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @return Filtered row group indices
   */
  [[nodiscard]] std::vector<size_type> filter_row_groups_with_bloom_filters(
    cudf::host_span<rmm::device_buffer> bloom_filter_data,
    cudf::host_span<size_type const> row_group_indices,
    parquet_reader_options const& options,
    rmm::cuda_stream_view stream) const;

  /**
   * @brief Filter data pages of filter columns using page statistics from page index metadata
   *
   * @param row_group_indices Input row groups indices
   * @param options Parquet reader options
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned column's device memory
   * @return A pair of boolean column indicating rows corresponding to data pages after
   *         page-pruning, and a list of boolean vectors indicating which data pages are not pruned,
   *         one per filter column.
   */
  [[nodiscard]] std::pair<std::unique_ptr<cudf::column>, std::vector<thrust::host_vector<bool>>>
  filter_data_pages_with_stats(cudf::host_span<size_type const> row_group_indices,
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
   * @param page_mask Boolean vectors indicating which data pages are not pruned, one per filter
   *                  column. All data pages considered not pruned if empty
   * @param row_group_indices Input row groups indices
   * @param column_chunk_buffers Device buffers containing column chunk data of filter columns
   * @param[in,out] row_mask Mutable boolean column indicating surviving rows from page pruning
   * @param options Parquet reader options
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @return Table of materialized filter columns and metadata
   */
  [[nodiscard]] table_with_metadata materialize_filter_columns(
    cudf::host_span<thrust::host_vector<bool> const> page_mask,
    cudf::host_span<size_type const> row_group_indices,
    std::vector<rmm::device_buffer> column_chunk_buffers,
    cudf::mutable_column_view row_mask,
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
   * @param column_chunk_buffers Device buffers containing column chunk data of payload columns
   * @param row_mask Boolean column indicating which rows need to be read. All rows read if empty
   * @param options Parquet reader options
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @return Table of materialized payload columns and metadata
   */
  [[nodiscard]] table_with_metadata materialize_payload_columns(
    cudf::host_span<size_type const> row_group_indices,
    std::vector<rmm::device_buffer> column_chunk_buffers,
    cudf::column_view row_mask,
    parquet_reader_options const& options,
    rmm::cuda_stream_view stream) const;

 private:
  std::unique_ptr<detail::hybrid_scan_reader_impl> _impl;
};

/** @} */  // end of group

}  // namespace io::parquet::experimental
}  // namespace CUDF_EXPORT cudf
