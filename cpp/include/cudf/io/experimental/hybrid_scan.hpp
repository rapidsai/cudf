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
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/io/types.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>

#include <memory>
#include <utility>
#include <vector>

namespace CUDF_EXPORT cudf {
namespace experimental::io {

namespace parquet {

namespace detail {

/**
 * @brief Internal experimental Parquet reader optimized for highly selective filters (Hybrid Scan
 * operation).
 */
class impl;

}  // namespace detail

/**
 * @brief The experimental parquet reader class to optimally read parquet files subject to
 *        highly selective filters (Hybrid Scan operation)
 *
 * This class is designed to best exploit reductive optimization techniques to speed up reading
 * Parquet files subject to highly selective filters (Hybrid scan operation). This class reads file
 * contents in two passes, where the first pass optimally reads the `filter` columns (i.e. columns
 * that appear in the filter expression) and the second pass optimally reads the `payload` columns
 * (i.e. columns that do not appear in the filter expression)
 */
class hybrid_scan_reader {
 public:
  hybrid_scan_reader();

  /**
   * @brief Constructor for the experimental parquet reader class to optimally read Parquet files
   * subject to highly selective filters
   *
   * @param footer_bytes Host span of parquet file footer bytes
   * @param options Parquet reader options
   */
  hybrid_scan_reader(cudf::host_span<uint8_t const> footer_bytes,
                     cudf::io::parquet_reader_options const& options);

  /**
   * @brief Destructor for the experimental parquet reader class
   */
  ~hybrid_scan_reader();

  /**
   * @brief Get the byte range of the `PageIndex` in the Parquet file
   *
   * @return Byte range of the `PageIndex`
   */
  [[nodiscard]] cudf::io::text::byte_range_info get_page_index_bytes() const;

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
  [[nodiscard]] std::vector<size_type> get_all_row_groups(
    cudf::io::parquet_reader_options const& options) const;

  /**
   * @brief Filter the row groups with statistics
   *
   * @param row_group_indices Input row groups indices
   * @param options Parquet reader options
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @return Filtered row group indices
   */
  [[nodiscard]] std::vector<std::vector<size_type>> filter_row_groups_with_stats(
    cudf::host_span<std::vector<size_type> const> row_group_indices,
    cudf::io::parquet_reader_options const& options,
    rmm::cuda_stream_view stream) const;

  /**
   * @brief Fetches byte ranges of secondary filters for further row group pruning
   *
   * Fetches a pair of vectors of byte ranges corresponding to available per-column-chunk dictionary
   * pages and bloom filters (called secondary filters) for further row group pruning
   *
   * @param row_group_indices Input row groups indices
   * @param options Parquet reader options
   * @return Pair of vectors of byte ranges to per-column-chunk dictionary pages and bloom filters
   */
  [[nodiscard]] std::pair<std::vector<cudf::io::text::byte_range_info>,
                          std::vector<cudf::io::text::byte_range_info>>
  get_secondary_filters(cudf::host_span<std::vector<size_type> const> row_group_indices,
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
  [[nodiscard]] std::vector<std::vector<size_type>> filter_row_groups_with_dictionary_pages(
    std::vector<rmm::device_buffer>& dictionary_page_data,
    cudf::host_span<std::vector<size_type> const> row_group_indices,
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
  [[nodiscard]] std::vector<std::vector<size_type>> filter_row_groups_with_bloom_filters(
    std::vector<rmm::device_buffer>& bloom_filter_data,
    cudf::host_span<std::vector<size_type> const> row_group_indices,
    cudf::io::parquet_reader_options const& options,
    rmm::cuda_stream_view stream) const;

  /**
   * @brief Filter data pages of filter columns using statistics containing in `PageIndex` metadata
   *
   * @param row_group_indices Input row groups indices
   * @param options Parquet reader options
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned column's device memory
   * @return A pair of boolean column indicating rows that survive the filter predicate at
   *         page-level, and a list of boolean vectors indicating the corresponding surviving data
   *         pages, one per filter column.
   */
  [[nodiscard]] std::pair<std::unique_ptr<cudf::column>, std::vector<std::vector<bool>>>
  filter_data_pages_with_stats(cudf::host_span<std::vector<size_type> const> row_group_indices,
                               cudf::io::parquet_reader_options const& options,
                               rmm::cuda_stream_view stream,
                               rmm::device_async_resource_ref mr) const;

  /**
   * @brief Fetches byte ranges of column chunks of filter columns
   *
   * @param row_group_indices Input row groups indices
   * @param options Parquet reader options
   * @return Pair of vectors of byte ranges to column chunks of filter columns and their
   *         corresponding input source file indices
   */
  [[nodiscard]] std::pair<std::vector<cudf::io::text::byte_range_info>,
                          std::vector<cudf::size_type>>
  get_filter_column_chunk_byte_ranges(
    cudf::host_span<std::vector<size_type> const> row_group_indices,
    cudf::io::parquet_reader_options const& options) const;

  /**
   * @brief Materializes filter columns, and updates the input row validity mask to only the rows
   *        that survive the row selection predicate at row level
   *
   * @param valid_data_pages Boolean vectors indicating surviving data pages, one per filter column
   * @param row_group_indices Input row groups indices
   * @param column_chunk_buffers Device buffers containing column chunk data of filter columns
   * @param[in,out] row_mask Mutable boolean column indicating rows that survive page-pruning
   * @param options Parquet reader options
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @return Table of materialized filter columns and metadata
   */
  [[nodiscard]] cudf::io::table_with_metadata materialize_filter_columns(
    cudf::host_span<std::vector<bool> const> valid_data_pages,
    cudf::host_span<std::vector<size_type> const> row_group_indices,
    std::vector<rmm::device_buffer> column_chunk_buffers,
    cudf::mutable_column_view row_mask,
    cudf::io::parquet_reader_options const& options,
    rmm::cuda_stream_view stream) const;

  /**
   * @brief Fetches byte ranges of column chunks of payload columns
   *
   * @param row_group_indices Input row groups indices
   * @param options Parquet reader options
   * @return Pair of vectors of byte ranges to column chunks of payload columns and their
   *         corresponding input source file indices
   */
  [[nodiscard]] std::pair<std::vector<cudf::io::text::byte_range_info>,
                          std::vector<cudf::size_type>>
  get_payload_column_chunk_byte_ranges(
    cudf::host_span<std::vector<size_type> const> row_group_indices,
    cudf::io::parquet_reader_options const& options) const;

  /**
   * @brief Materializes payload columns
   *
   * @param row_group_indices Input row groups indices
   * @param column_chunk_buffers Device buffers containing column chunk data of payload columns
   * @param row_mask Boolean column indicating which rows need to be read
   * @param options Parquet reader options
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @return Table of materialized payload columns and metadata
   */
  [[nodiscard]] cudf::io::table_with_metadata materialize_payload_columns(
    cudf::host_span<std::vector<size_type> const> row_group_indices,
    std::vector<rmm::device_buffer> column_chunk_buffers,
    cudf::column_view row_mask,
    cudf::io::parquet_reader_options const& options,
    rmm::cuda_stream_view stream) const;

 private:
  std::unique_ptr<detail::impl> _impl;
};

}  // namespace parquet

/**
 * @brief Factory function to create a hybrid scan reader
 *
 * @param footer_bytes Host span of parquet file footer bytes
 * @param options Parquet reader options
 * @return A unique pointer to the hybrid scan reader
 */
[[nodiscard]] std::unique_ptr<parquet::hybrid_scan_reader> make_hybrid_scan_reader(
  cudf::host_span<uint8_t const> footer_bytes, cudf::io::parquet_reader_options const& options);

/**
 * @brief Get the byte range of the `PageIndex` in the Parquet file
 *
 * @param reader Hybrid scan reader
 * @return Byte range of the `PageIndex`
 */
[[nodiscard]] cudf::io::text::byte_range_info get_page_index_bytes(
  std::unique_ptr<parquet::hybrid_scan_reader> const& reader);

/**
 * @brief Setup the PageIndex
 *
 * @param reader Hybrid scan reader
 * @param page_index_bytes Host span of Parquet `PageIndex` buffer bytes
 */
void setup_page_index(std::unique_ptr<parquet::hybrid_scan_reader> const& reader,
                      cudf::host_span<uint8_t const> page_index_bytes);

/**
 * @brief Get all row groups from the parquet file
 *
 * @param reader Hybrid scan reader
 * @param options Parquet reader options
 * @return Vector of row group indices
 */
[[nodiscard]] std::vector<size_type> get_all_row_groups(
  std::unique_ptr<parquet::hybrid_scan_reader> const& reader,
  cudf::io::parquet_reader_options const& options);

/**
 * @brief Filter the row groups with statistics based on predicate filter
 *
 * @param reader Hybrid scan reader
 * @param row_group_indices Input row groups indices
 * @param options Parquet reader options
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return Filtered row group indices
 */
[[nodiscard]] std::vector<size_type> filter_row_groups_with_stats(
  std::unique_ptr<parquet::hybrid_scan_reader> const& reader,
  cudf::host_span<size_type const> row_group_indices,
  cudf::io::parquet_reader_options const& options,
  rmm::cuda_stream_view stream);

/**
 * @brief Fetches byte ranges of secondary filters for further row group pruning
 *
 * @param reader Hybrid scan reader
 * @param row_group_indices Input row groups indices
 * @param options Parquet reader options
 * @return Pair of vectors of byte ranges to per-column-chunk dictionary pages and bloom filters
 */
[[nodiscard]] std::pair<std::vector<cudf::io::text::byte_range_info>,
                        std::vector<cudf::io::text::byte_range_info>>
get_secondary_filters(std::unique_ptr<parquet::hybrid_scan_reader> const& reader,
                      cudf::host_span<size_type const> row_group_indices,
                      cudf::io::parquet_reader_options const& options);

/**
 * @brief Filter the row groups with dictionary pages based on predicate filter
 *
 * @param reader Hybrid scan reader
 * @param dictionary_page_data Device buffers containing per-column-chunk dictionary page data
 * @param row_group_indices Input row groups indices
 * @param options Parquet reader options
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return Filtered row group indices
 */
[[nodiscard]] std::vector<size_type> filter_row_groups_with_dictionary_pages(
  std::unique_ptr<parquet::hybrid_scan_reader> const& reader,
  std::vector<rmm::device_buffer>& dictionary_page_data,
  cudf::host_span<size_type const> row_group_indices,
  cudf::io::parquet_reader_options const& options,
  rmm::cuda_stream_view stream);

/**
 * @brief Filter the row groups with bloom filters based on predicate filter
 *
 * @param reader Hybrid scan reader
 * @param bloom_filter_data Device buffers containing per-column-chunk bloom filter data
 * @param row_group_indices Input row groups indices
 * @param options Parquet reader options
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return Filtered row group indices
 */
[[nodiscard]] std::vector<size_type> filter_row_groups_with_bloom_filters(
  std::unique_ptr<parquet::hybrid_scan_reader> const& reader,
  std::vector<rmm::device_buffer>& bloom_filter_data,
  cudf::host_span<size_type const> row_group_indices,
  cudf::io::parquet_reader_options const& options,
  rmm::cuda_stream_view stream);

/**
 * @brief Filter data pages of filter columns using statistics containing in `PageIndex` metadata
 *        based on predicate filter
 *
 * @param reader Hybrid scan reader
 * @param row_group_indices Input row groups indices
 * @param options Parquet reader options
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return A pair of boolean column indicating rows that survive the row selection predicate at
 *         page-level, and a list of boolean vectors indicating the corresponding surviving data
 *         pages, one per filter column.
 */
[[nodiscard]] std::pair<std::unique_ptr<cudf::column>, std::vector<std::vector<bool>>>
filter_data_pages_with_stats(std::unique_ptr<parquet::hybrid_scan_reader> const& reader,
                             cudf::host_span<size_type const> row_group_indices,
                             cudf::io::parquet_reader_options const& options,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr);

/**
 * @brief Fetches byte ranges of column chunks of filter columns
 *
 * @param reader Hybrid scan reader
 * @param row_group_indices Input row groups indices
 * @param options Parquet reader options
 * @return Vector of byte ranges to column chunks of filter columns
 */
[[nodiscard]] std::vector<cudf::io::text::byte_range_info> get_filter_column_chunk_byte_ranges(
  std::unique_ptr<parquet::hybrid_scan_reader> const& reader,
  cudf::host_span<size_type const> row_group_indices,
  cudf::io::parquet_reader_options const& options);

/**
 * @brief Materializes filter columns and updates the input row validity mask to only the rows
 *        that survive the row selection predicate at row level
 *
 * @param reader Hybrid scan reader
 * @param filtered_data_pages Boolean vectors indicating surviving data pages, one per filter column
 * @param row_group_indices Input row groups indices
 * @param column_chunk_buffers Device buffers containing column chunk data of filter columns
 * @param[in,out] row_mask Mutable boolean column indicating rows that survive page-pruning
 * @param options Parquet reader options
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return Table of materialized filter columns and metadata
 */
[[nodiscard]] cudf::io::table_with_metadata materialize_filter_columns(
  std::unique_ptr<parquet::hybrid_scan_reader> const& reader,
  cudf::host_span<std::vector<bool> const> filtered_data_pages,
  cudf::host_span<size_type const> row_group_indices,
  std::vector<rmm::device_buffer> column_chunk_buffers,
  cudf::mutable_column_view row_mask,
  cudf::io::parquet_reader_options const& options,
  rmm::cuda_stream_view stream);

/**
 * @brief Fetches byte ranges of column chunks of payload columns
 *
 * @param reader Hybrid scan reader
 * @param row_group_indices Input row groups indices
 * @param options Parquet reader options
 * @return Vector of byte ranges to column chunks of payload columns
 */
[[nodiscard]] std::vector<cudf::io::text::byte_range_info> get_payload_column_chunk_byte_ranges(
  std::unique_ptr<parquet::hybrid_scan_reader> const& reader,
  cudf::host_span<size_type const> row_group_indices,
  cudf::io::parquet_reader_options const& options);

/**
 * @brief Materializes payload columns
 *
 * @param reader Hybrid scan reader
 * @param row_group_indices Input row groups indices
 * @param column_chunk_buffers Device buffers containing column chunk data of payload columns
 * @param row_mask Boolean column indicating which rows need to be read
 * @param options Parquet reader options
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return Table of materialized payload columns and metadata
 */
[[nodiscard]] cudf::io::table_with_metadata materialize_payload_columns(
  std::unique_ptr<parquet::hybrid_scan_reader> const& reader,
  cudf::host_span<size_type const> row_group_indices,
  std::vector<rmm::device_buffer> column_chunk_buffers,
  cudf::column_view row_mask,
  cudf::io::parquet_reader_options const& options,
  rmm::cuda_stream_view stream);

}  // namespace experimental::io
}  // namespace CUDF_EXPORT cudf
