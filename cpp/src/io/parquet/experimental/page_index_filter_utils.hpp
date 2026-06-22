/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "io/parquet/reader_impl_helpers.hpp"

#include <cudf/detail/utilities/host_vector.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

namespace cudf::io::parquet::experimental::detail {

using metadata_base = parquet::detail::metadata;

/**
 * @brief Find the offset of the column chunk with the given schema index in the row group
 *
 * @param row_group Row group
 * @param schema_idx Schema index
 * @return Offset of the column chunk iterator
 */
[[nodiscard]] size_type find_colchunk_iter_offset(RowGroup const& row_group, size_type schema_idx);

/**
 * @brief Compute if the page index is present in all parquet data sources for all columns
 *
 * @param file_metadatas Span of parquet footer metadata
 * @param row_group_indices Span of input row group indices
 * @return Boolean indicating if the page index is present in all parquet data sources for all
 * columns
 */
[[nodiscard]] bool compute_has_page_index(
  cudf::host_span<metadata_base const> file_metadatas,
  cudf::host_span<std::vector<size_type> const> row_group_indices);

/**
 * @brief Compute page row counts and page row offsets and column chunk page (count) offsets for a
 * given column schema index
 *
 * @param per_file_metadata Span of parquet footer metadata
 * @param row_group_indices Span of input row group indices
 * @param schema_idx Column's schema index
 * @param stream CUDA stream
 * @return Tuple of page row counts, page row offsets, and column chunk page (count) offsets
 */
[[nodiscard]] std::tuple<cudf::detail::host_vector<size_type>,
                         cudf::detail::host_vector<size_type>,
                         cudf::detail::host_vector<size_type>>
compute_page_row_counts_and_offsets(cudf::host_span<metadata_base const> per_file_metadata,
                                    cudf::host_span<std::vector<size_type> const> row_group_indices,
                                    size_type schema_idx,
                                    rmm::cuda_stream_view stream);

/**
 * @brief Computes page row offsets and the size (number of rows) of the largest page for a given
 * column schema index
 *
 * @param per_file_metadata Span of parquet footer metadata
 * @param row_group_indices Span of input row group indices
 * @param schema_idx Column's schema index
 * @return A pair of page row offsets and the size of the largest page in this
 * column
 */
[[nodiscard]] std::pair<std::vector<size_type>, size_type> compute_page_row_offsets(
  cudf::host_span<metadata_base const> per_file_metadata,
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  size_type schema_idx);

/**
 * @brief Computes a device vector where each row contains the index of the page it belongs to
 *
 * @param page_row_counts Span of page row counts
 * @param page_row_offsets Span of page row offsets
 * @param total_rows Total number of rows
 * @param stream CUDA stream
 * @param mr Device memory resource for the output device vector
 * @return Device vector where each row contains the index of the page it belongs to
 */
[[nodiscard]] rmm::device_uvector<size_type> compute_page_indices_async(
  cudf::host_span<cudf::size_type const> page_row_counts,
  cudf::host_span<cudf::size_type const> page_row_offsets,
  cudf::size_type total_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

/**
 * @brief Computes the offsets of the Fenwick tree levels (level 1 and higher) until the tree level
 * block size becomes larger than the maximum page (search range) size
 *
 * @param level0_size Size of the zeroth tree level (the row mask)
 * @param max_page_size Maximum page (search range) size
 * @return Fenwick tree level offsets
 */
[[nodiscard]] std::vector<size_type> compute_fenwick_tree_level_offsets(
  cudf::size_type level0_size, cudf::size_type max_page_size);

}  // namespace cudf::io::parquet::experimental::detail
