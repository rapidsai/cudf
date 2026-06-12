/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/io/types.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <memory>
#include <vector>

namespace cudf::io::parquet::experimental::detail {
/**
 * @brief Internal experimental Parquet reader optimized for highly selective filters, called a
 *        Hybrid Scan operation.
 */
class hybrid_scan_reader_impl;
}  // namespace cudf::io::parquet::experimental::detail

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
 * @brief Multi-file variant of the experimental Hybrid Scan Parquet reader
 *
 * Vectorizes `hybrid_scan_reader` APIs to support multiple Parquet sources. Inputs and outputs are
 * indexed by source order except for the row mask which is a single BOOL8 column spanning all rows
 * from all sources concatenated in source order, then row-group order within a source.
 *
 * @note Detailed usage documentation will be added once all APIs are in place. This reader will
 * eventually move to `hybrid_scan.hpp` and the existing single-file reader (`hybrid_scan_reader`)
 * will become its subclass. Only keeping this separate here for now to reduce noise.
 */
class hybrid_scan_multifile {
 public:
  /**
   * @brief Constructor for the multi-file experimental Parquet reader
   *
   * @param footer_bytes Host span of Parquet file footer byte spans, one per source
   * @param options Parquet reader options
   */
  explicit hybrid_scan_multifile(cudf::host_span<cudf::host_span<uint8_t const> const> footer_bytes,
                                 parquet_reader_options const& options);

  /**
   * @brief Constructor for the multi-file experimental Parquet reader
   *
   * @param parquet_metadata Host span of pre-populated Parquet file metadata, one per source
   * @param options Parquet reader options
   */
  explicit hybrid_scan_multifile(cudf::host_span<FileMetaData const> parquet_metadata,
                                 parquet_reader_options const& options);

  /**
   * @brief Destructor for the multi-file experimental Parquet reader
   */
  ~hybrid_scan_multifile();

  /**
   * @brief Get parquet metadatas for all sources
   *
   * @return Vector of parquet metadata, one per source
   */
  [[nodiscard]] std::vector<FileMetaData> parquet_metadatas() const;

  /**
   * @brief Get byte ranges of the page index for all sources
   *
   * @return Vector of page index byte ranges, one per source
   */
  [[nodiscard]] std::vector<byte_range_info> page_index_byte_ranges() const;

  /**
   * @brief Setup the per-source page index within each Parquet file metadata
   *
   * @param page_index_bytes Host span of Parquet page index buffer bytes, one per source
   */
  void setup_page_indexes(
    cudf::host_span<cudf::host_span<uint8_t const> const> page_index_bytes) const;

  /**
   * @brief Get all available per-source row group indices from the parquet files
   *
   * @param options Parquet reader options
   * @return Vector of row group indices, one inner vector per source
   */
  [[nodiscard]] std::vector<std::vector<size_type>> all_row_groups(
    parquet_reader_options const& options) const;

  /**
   * @brief Get the total number of top-level rows in the per-source row groups
   *
   * @param row_group_indices Input per-source row group indices (one inner vector per source)
   * @return Total number of top-level rows across all sources
   */
  [[nodiscard]] size_type total_rows_in_row_groups(
    cudf::host_span<std::vector<size_type> const> row_group_indices) const;

  /**
   * @brief Resets the current column selection
   *
   * Resets the current column selection state forcing column re-selection in subsequent filter,
   * byte range, setup chunking and materialization APIs. This is useful if the filter expression
   * has been cascaded (and-ed) to include new columns.
   */
  void reset_column_selection() const;

  /**
   * @brief Filter the row groups using the byte range specified by [`bytes_to_skip`,
   * `bytes_to_skip + bytes_to_read`)
   *
   * Filters the row groups such that only the row groups that start within the byte range are
   * selected. Note that the last selected row group may end beyond the byte range.
   *
   * @param row_group_indices Input row group indices, one per source
   * @param options Parquet reader options
   * @return Filtered per-source row group indices (one inner vector per source)
   */
  [[nodiscard]] std::vector<std::vector<size_type>> filter_row_groups_with_byte_range(
    cudf::host_span<std::vector<size_type> const> row_group_indices,
    parquet_reader_options const& options) const;

  /**
   * @brief Filter the input row groups using column chunk statistics
   *
   * @param row_group_indices Input row group indices, one per source
   * @param options Parquet reader options
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @return Filtered row group indices, one per source
   */
  [[nodiscard]] std::vector<std::vector<size_type>> filter_row_groups_with_stats(
    cudf::host_span<std::vector<size_type> const> row_group_indices,
    parquet_reader_options const& options,
    rmm::cuda_stream_view stream) const;

  /**
   * @brief Get byte ranges of bloom filters and dictionary pages (secondary filters) for row group
   *        pruning
   *
   * @note Device buffers for bloom filter byte ranges must be allocated using a 32 byte
   *       aligned memory resource
   *
   * @param row_group_indices Input row group indices, one per source
   * @param options Parquet reader options
   * @return Pair of vectors of byte ranges of column chunk with bloom filters and dictionary
   *         pages subject to filter predicate
   */
  [[nodiscard]] std::pair<std::vector<byte_range_info>, std::vector<byte_range_info>>
  secondary_filters_byte_ranges(cudf::host_span<std::vector<size_type> const> row_group_indices,
                                parquet_reader_options const& options) const;

  /**
   * @brief Get byte ranges of column chunk dictionary pages for row group pruning
   *
   * @param row_group_indices Input row group indices, one per source
   * @param options Parquet reader options
   * @return Vector of byte ranges of column chunks with dictionary pages subject to the filter
   *         predicate, ordered source-major then row-group then dictionary column
   */
  [[nodiscard]] std::vector<byte_range_info> dictionary_pages_byte_ranges(
    cudf::host_span<std::vector<size_type> const> row_group_indices,
    parquet_reader_options const& options) const;

  /**
   * @brief Filter the row groups using column chunk dictionary pages
   *
   * @param dictionary_page_data Device spans of dictionary page data of column chunks with an
   *                             (in)equality predicate, ordered to match the dictionary page byte
   *                             ranges returned by `dictionary_pages_byte_ranges`
   * @param row_group_indices Input row group indices, one per source
   * @param options Parquet reader options
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @return Filtered per-source row group indices (one inner vector per source)
   */
  [[nodiscard]] std::vector<std::vector<size_type>> filter_row_groups_with_dictionary_pages(
    cudf::host_span<cudf::device_span<uint8_t const> const> dictionary_page_data,
    cudf::host_span<std::vector<size_type> const> row_group_indices,
    parquet_reader_options const& options,
    rmm::cuda_stream_view stream) const;

 private:
  std::unique_ptr<detail::hybrid_scan_reader_impl> _impl;
};

/** @} */  // end of group

}  // namespace io::parquet::experimental
}  // namespace CUDF_EXPORT cudf
