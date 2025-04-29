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
 * @file hybrid_scan.hpp
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
 * Parquet files subject to highly selective filters. This class reads file contents in two passes,
 * where the first pass optimally reads the `filter` columns (i.e. columns that appear in the filter
 * expression) and the second pass optimally reads the `payload` columns (i.e. columns that do not
 * appear in the filter expression).
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
                              cudf::io::parquet_reader_options const& options);

  /**
   * @brief Destructor for the experimental parquet reader class
   */
  ~hybrid_scan_reader();

  /**
   * @brief Get the Parquet file footer metadata
   *
   * @return Parquet file footer metadata
   */
  [[nodiscard]] FileMetaData parquet_metadata() const;

  /**
   * @brief Get the byte range of the `PageIndex` in the Parquet file
   *
   * @return Byte range of the `PageIndex`
   */
  [[nodiscard]] text::byte_range_info get_page_index_bytes() const;

  /**
   * @brief Setup the PageIndex
   *
   * @param page_index_bytes Host span of Parquet `PageIndex` buffer bytes
   */
  void setup_page_index(cudf::host_span<uint8_t const> page_index_bytes);

  /**
   * @brief Get all available row groups from the parquet file
   *
   * @param options Parquet reader options
   * @return Vector of row group indices
   */
  [[nodiscard]] std::vector<size_type> all_row_groups(
    cudf::io::parquet_reader_options const& options) const;

  /**
   * @brief Get the total number of rows in the row groups
   *
   * @param row_group_indices Input row groups indices
   * @return Total number of rows in the row groups
   */
  [[nodiscard]] size_type num_rows_in_row_groups(
    cudf::host_span<size_type const> row_group_indices) const;

  /**
   * @brief Filter the row groups with statistics
   *
   * @param row_group_indices Input row groups indices
   * @param options Parquet reader options
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @return Filtered row group indices
   */
  [[nodiscard]] std::vector<cudf::size_type> filter_row_groups_with_stats(
    cudf::host_span<size_type const> row_group_indices,
    cudf::io::parquet_reader_options const& options,
    rmm::cuda_stream_view stream) const;

  /**
   * @brief Fetches byte ranges of bloom filters and dictionary pages (secondary filters) for
   * further row group pruning
   *
   * @param row_group_indices Input row groups indices
   * @param options Parquet reader options
   * @return Pair of vectors of byte ranges to per-column-chunk bloom filters and dictionary pages
   */
  [[nodiscard]] std::pair<std::vector<text::byte_range_info>, std::vector<text::byte_range_info>>
  secondary_filters_byte_ranges(cudf::host_span<size_type const> row_group_indices,
                                cudf::io::parquet_reader_options const& options) const;

  /**
   * @brief Filter the row groups with dictionary pages
   *
   * @param dictionary_page_data Device buffers containing per-column-chunk dictionary page data
   * @param row_group_indices Input row groups indices
   * @param options Parquet reader options
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @return Filtered row group indices
   */
  [[nodiscard]] std::vector<cudf::size_type> filter_row_groups_with_dictionary_pages(
    cudf::host_span<rmm::device_buffer> dictionary_page_data,
    cudf::host_span<size_type const> row_group_indices,
    cudf::io::parquet_reader_options const& options,
    rmm::cuda_stream_view stream) const;

  /**
   * @brief Filter the row groups with bloom filters
   *
   * @param bloom_filter_data Device buffers containing per-column-chunk bloom filter data
   * @param row_group_indices Input row groups indices
   * @param options Parquet reader options
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @return Filtered row group indices
   */
  [[nodiscard]] std::vector<cudf::size_type> filter_row_groups_with_bloom_filters(
    cudf::host_span<rmm::device_buffer> bloom_filter_data,
    cudf::host_span<size_type const> row_group_indices,
    cudf::io::parquet_reader_options const& options,
    rmm::cuda_stream_view stream) const;

  /**
   * @brief Filter data pages of filter columns using statistics containing in page index metadata
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
                               cudf::io::parquet_reader_options const& options,
                               rmm::cuda_stream_view stream,
                               rmm::device_async_resource_ref mr) const;

  /**
   * @brief Fetches byte ranges of column chunks of filter columns
   *
   * @param row_group_indices Input row groups indices
   * @param options Parquet reader options
   * @return Vector of byte ranges to column chunks of filter columns
   */
  [[nodiscard]] std::vector<text::byte_range_info> filter_column_chunks_byte_ranges(
    cudf::host_span<size_type const> row_group_indices,
    cudf::io::parquet_reader_options const& options) const;

  /**
   * @brief Materializes filter columns and updates the input row mask to only the rows
   *        that exist in the output table
   *
   * @param page_mask Boolean vectors indicating surviving data pages, one per filter column
   * @param row_group_indices Input row groups indices
   * @param column_chunk_buffers Device buffers containing column chunk data of filter columns
   * @param[in,out] row_mask Mutable boolean column indicating surviving rows so far
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
   * @brief Fetches byte ranges of column chunks of payload columns
   *
   * @param row_group_indices Input row groups indices
   * @param options Parquet reader options
   * @return Vector of byte ranges to column chunks of payload columns
   */
  [[nodiscard]] std::vector<text::byte_range_info> payload_column_chunks_byte_ranges(
    cudf::host_span<size_type const> row_group_indices,
    parquet_reader_options const& options) const;

  /**
   * @brief Materializes payload columns and applies the row mask to the output table
   *
   * @param row_group_indices Input row groups indices
   * @param column_chunk_buffers Device buffers containing column chunk data of payload columns
   * @param row_mask Boolean column indicating which rows need to be read
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

}  // namespace io::parquet::experimental
}  // namespace CUDF_EXPORT cudf
