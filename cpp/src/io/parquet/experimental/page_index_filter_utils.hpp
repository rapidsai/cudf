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

#include "io/parquet/reader_impl_helpers.hpp"

#include <cudf/detail/utilities/host_vector.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

namespace cudf::io::parquet::experimental::detail {

using metadata_base = parquet::detail::metadata;

/**
 * @brief Compute if the page index is present in all parquet data sources for all columns
 *
 * @param file_metadatas Span of parquet footer metadata
 * @param row_group_indices Span of input row group indices
 * @param column_schema_indices Span of input column schema indices
 * @return Boolean indicating if the page index is present in all parquet data sources for all
 * columns
 */
[[nodiscard]] bool compute_has_page_index(
  cudf::host_span<metadata_base const> file_metadatas,
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  cudf::host_span<size_type const> column_schema_indices);

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
 * @brief Compute page row offsets for a given column schema index
 *
 * @param per_file_metadata Span of parquet footer metadata
 * @param row_group_indices Span of input row group indices
 * @param schema_idx Column's schema index
 * @param row_mask_offset Offset of the row mask
 * @return Tuple of page row offsets, number of pages, and the size of the largest page in this
 * column
 */
[[nodiscard]] std::tuple<std::vector<size_type>, size_type, size_type> compute_page_row_offsets(
  cudf::host_span<metadata_base const> per_file_metadata,
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  size_type schema_idx,
  cudf::size_type row_mask_offset = 0);

/**
 * @brief Make a device vector where each row contains the index of the page it belongs to
 *
 * @param page_row_counts Span of page row counts
 * @param page_row_offsets Span of page row offsets
 * @param total_rows Total number of rows
 * @param stream CUDA stream
 * @return Device vector where each row contains the index of the page it belongs to
 */
[[nodiscard]] rmm::device_uvector<size_type> make_page_indices_async(
  cudf::host_span<cudf::size_type const> page_row_counts,
  cudf::host_span<cudf::size_type const> page_row_offsets,
  cudf::size_type total_rows,
  rmm::cuda_stream_view stream);

/**
 * @brief Compute the levels of the row mask
 *
 * @param num_rows Number of rows in the row mask
 * @param max_page_size Maximum page size
 * @return Pair of level offsets and total levels size
 */
[[nodiscard]] std::pair<std::vector<size_type>, size_type> compute_row_mask_levels(
  cudf::size_type num_rows, cudf::size_type max_page_size);

}  // namespace cudf::io::parquet::experimental::detail
