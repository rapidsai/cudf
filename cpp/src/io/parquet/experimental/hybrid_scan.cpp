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

#include "cudf/utilities/error.hpp"
#include "hybrid_scan_impl.hpp"

#include <cudf/io/experimental/hybrid_scan.hpp>

namespace cudf::experimental::io::parquet {

hybrid_scan_reader::hybrid_scan_reader() = default;

hybrid_scan_reader::hybrid_scan_reader(cudf::host_span<uint8_t const> footer_bytes,
                                       cudf::host_span<uint8_t const> page_index_bytes,
                                       cudf::io::parquet_reader_options const& options)
  : _impl{std::make_unique<detail::impl>(footer_bytes, page_index_bytes, options)}
{
}

hybrid_scan_reader::~hybrid_scan_reader() = default;

std::vector<std::vector<size_type>> hybrid_scan_reader::filter_row_groups_with_stats(
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  cudf::io::parquet_reader_options const& options,
  rmm::cuda_stream_view stream) const
{
  // Prefer user provided row group indices, else use the ones in options.
  auto const input_row_group_indices =
    row_group_indices.size() ? row_group_indices : options.get_row_groups();

  return _impl->filter_row_groups_with_stats(input_row_group_indices, options, stream);
}

std::vector<size_type> hybrid_scan_reader::get_valid_row_groups(
  cudf::io::parquet_reader_options const& options) const
{
  CUDF_EXPECTS(options.get_row_groups().size() == 0 or options.get_row_groups().size() == 1, "");
  if (options.get_row_groups().size()) { return options.get_row_groups()[0]; }

  return _impl->get_valid_row_groups(options);
}

std::pair<std::vector<cudf::io::text::byte_range_info>,
          std::vector<cudf::io::text::byte_range_info>>
hybrid_scan_reader::get_secondary_filters(
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  cudf::io::parquet_reader_options const& options) const
{
  return _impl->get_secondary_filters(row_group_indices, options);
}

std::vector<std::vector<size_type>> hybrid_scan_reader::filter_row_groups_with_dictionary_pages(
  std::vector<rmm::device_buffer>& dictionary_page_data,
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  cudf::io::parquet_reader_options const& options,
  rmm::cuda_stream_view stream) const
{
  CUDF_EXPECTS(row_group_indices.size() == 1 and row_group_indices[0].size(), "");
  CUDF_EXPECTS(row_group_indices[0].size() == dictionary_page_data.size(), "");
  return _impl->filter_row_groups_with_dictionary_pages(
    dictionary_page_data, row_group_indices, options, stream);
}

std::vector<std::vector<size_type>> hybrid_scan_reader::filter_row_groups_with_bloom_filters(
  std::vector<rmm::device_buffer>& bloom_filter_data,
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  cudf::io::parquet_reader_options const& options,
  rmm::cuda_stream_view stream) const
{
  CUDF_EXPECTS(row_group_indices.size() == 1 and row_group_indices[0].size(), "");
  CUDF_EXPECTS(row_group_indices[0].size() == bloom_filter_data.size(), "");
  return _impl->filter_row_groups_with_bloom_filters(
    bloom_filter_data, row_group_indices, options, stream);
}

std::vector<std::vector<bool>> hybrid_scan_reader::filter_data_pages_with_stats(
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  cudf::io::parquet_reader_options const& options,
  rmm::cuda_stream_view stream) const
{
  CUDF_EXPECTS(row_group_indices.size() == 1 and row_group_indices[0].size(), "");
  return _impl->filter_data_pages_with_stats(row_group_indices, options, stream);
}

[[nodiscard]] std::pair<std::vector<cudf::io::text::byte_range_info>, std::vector<cudf::size_type>>
hybrid_scan_reader::get_column_chunk_byte_ranges(
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  cudf::io::parquet_reader_options const& options) const
{
  CUDF_EXPECTS(row_group_indices.size() == 1 and row_group_indices[0].size(), "");
  return _impl->get_column_chunk_byte_ranges(row_group_indices, options);
}

cudf::io::table_with_metadata hybrid_scan_reader::materialize_filter_columns(
  cudf::host_span<std::vector<bool> const> filtered_data_pages,
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  std::vector<rmm::device_buffer> column_chunk_buffers,
  cudf::io::parquet_reader_options const& options,
  rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(row_group_indices.size() == 1 and row_group_indices[0].size(), "");
  // TODO: Implement this
  return _impl->materialize_filter_columns(
    filtered_data_pages, row_group_indices, std::move(column_chunk_buffers), options, stream);
}

}  // namespace cudf::experimental::io::parquet
