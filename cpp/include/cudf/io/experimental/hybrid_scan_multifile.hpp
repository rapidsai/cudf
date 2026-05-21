/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
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
   * @brief Get the per-source Parquet file footer metadata
   *
   * @return Vector of file metadata, one per source
   */
  [[nodiscard]] std::vector<FileMetaData> parquet_metadata() const;

  /**
   * @brief Get the per-source byte range of the page index in each Parquet file
   *
   * The returned vector always has one entry per source. A source's entry is a
   * default-constructed `byte_range_info{}` if that source has no row groups, no columns, or no
   * page index offsets. A `CUDF_LOG_WARN` is emitted once if some sources have a page index and
   * others do not.
   *
   * @return Vector of page index byte ranges, one per source
   */
  [[nodiscard]] std::vector<byte_range_info> page_index_byte_range() const;

  /**
   * @brief Setup the per-source page index within each Parquet file metadata
   *
   * Materializes `ColumnIndex` and `OffsetIndex` (page index) inside each source's
   * `FileMetaData`. The input span size must equal the number of sources. A per-source empty
   * span is skipped with a one-time warning. Sources whose corresponding span is non-empty must
   * have row groups and valid page index offsets.
   *
   * @param page_index_bytes Host span of Parquet page index buffer bytes, one per source
   */
  void setup_page_index(
    cudf::host_span<cudf::host_span<uint8_t const> const> page_index_bytes) const;

  /**
   * @brief Get all available per-source row group indices from the parquet files
   *
   * If `options.get_row_groups()` is non-empty, its size must equal the number of sources and it
   * is returned as-is. Otherwise builds `[0 .. per_source_num_row_groups[i])` for each source.
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

 private:
  std::unique_ptr<detail::hybrid_scan_reader_impl> _impl;
};

/** @} */  // end of group

}  // namespace io::parquet::experimental
}  // namespace CUDF_EXPORT cudf
