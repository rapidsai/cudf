/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hybrid_scan_impl.hpp"

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/io/experimental/hybrid_scan_multifile.hpp>

namespace cudf::io::parquet::experimental {

hybrid_scan_multifile::hybrid_scan_multifile(
  cudf::host_span<cudf::host_span<uint8_t const> const> footer_bytes,
  parquet_reader_options const& options)
  : _impl{std::make_unique<detail::hybrid_scan_reader_impl>(footer_bytes, options)}
{
}

hybrid_scan_multifile::hybrid_scan_multifile(cudf::host_span<FileMetaData const> parquet_metadata,
                                             parquet_reader_options const& options)
  : _impl{std::make_unique<detail::hybrid_scan_reader_impl>(parquet_metadata, options)}
{
}

hybrid_scan_multifile::~hybrid_scan_multifile() = default;

std::vector<FileMetaData> hybrid_scan_multifile::parquet_metadatas() const
{
  return _impl->parquet_metadatas();
}

std::vector<text::byte_range_info> hybrid_scan_multifile::page_index_byte_ranges() const
{
  return _impl->page_index_byte_ranges();
}

void hybrid_scan_multifile::setup_page_indexes(
  cudf::host_span<cudf::host_span<uint8_t const> const> page_index_bytes) const
{
  CUDF_FUNC_RANGE();
  _impl->setup_page_indexes(page_index_bytes);
}

std::vector<std::vector<size_type>> hybrid_scan_multifile::all_row_groups(
  parquet_reader_options const& options) const
{
  return _impl->all_row_groups(options);
}

size_type hybrid_scan_multifile::total_rows_in_row_groups(
  cudf::host_span<std::vector<size_type> const> row_group_indices) const
{
  if (row_group_indices.empty()) { return 0; }
  return _impl->total_rows_in_row_groups(row_group_indices);
}

void hybrid_scan_multifile::reset_column_selection() const { _impl->reset_column_selection(); }

std::vector<std::vector<size_type>> hybrid_scan_multifile::filter_row_groups_with_byte_range(
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  parquet_reader_options const& options) const
{
  CUDF_FUNC_RANGE();
  return _impl->filter_row_groups_with_byte_range(row_group_indices, options);
}

std::vector<std::vector<size_type>> hybrid_scan_multifile::filter_row_groups_with_stats(
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  parquet_reader_options const& options,
  rmm::cuda_stream_view stream) const
{
  CUDF_FUNC_RANGE();
  return _impl->filter_row_groups_with_stats(row_group_indices, options, stream);
}

std::pair<std::vector<text::byte_range_info>, std::vector<text::byte_range_info>>
hybrid_scan_multifile::secondary_filters_byte_ranges(
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  parquet_reader_options const& options) const
{
  CUDF_FUNC_RANGE();
  return _impl->secondary_filters_byte_ranges(row_group_indices, options);
}

}  // namespace cudf::io::parquet::experimental
