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

#include <thrust/host_vector.h>

namespace cudf::io::parquet::experimental {

hybrid_scan_reader::hybrid_scan_reader(cudf::host_span<uint8_t const> footer_bytes,
                                       parquet_reader_options const& options)
  : _impl{std::make_unique<detail::impl>(footer_bytes, options)}
{
}

hybrid_scan_reader::~hybrid_scan_reader() = default;

[[nodiscard]] cudf::io::text::byte_range_info hybrid_scan_reader::get_page_index_bytes() const
{
  return _impl->get_page_index_bytes();
}

[[nodiscard]] FileMetaData hybrid_scan_reader::parquet_metadata() const
{
  return _impl->parquet_metadata();
}

void hybrid_scan_reader::setup_page_index(cudf::host_span<uint8_t const> page_index_bytes)
{
  return _impl->setup_page_index(page_index_bytes);
}

std::vector<cudf::size_type> hybrid_scan_reader::all_row_groups(
  parquet_reader_options const& options) const
{
  CUDF_EXPECTS(options.get_row_groups().size() <= 1,
               "Encountered invalid size of row group indices in parquet reader options");

  // If row groups are specified in parquet reader options, return them as is
  if (options.get_row_groups().size() == 1) { return options.get_row_groups().front(); }

  return _impl->all_row_groups(options);
}

std::vector<cudf::size_type> hybrid_scan_reader::filter_row_groups_with_stats(
  cudf::host_span<size_type const> row_group_indices,
  parquet_reader_options const& options,
  rmm::cuda_stream_view stream) const
{
  auto const input_row_group_indices =
    std::vector<std::vector<size_type>>{{row_group_indices.begin(), row_group_indices.end()}};
  return _impl->filter_row_groups_with_stats(input_row_group_indices, options, stream).front();
}

std::pair<std::vector<cudf::io::text::byte_range_info>,
          std::vector<cudf::io::text::byte_range_info>>
hybrid_scan_reader::secondary_filters_byte_ranges(
  cudf::host_span<size_type const> row_group_indices, parquet_reader_options const& options) const
{
  auto const input_row_group_indices =
    std::vector<std::vector<size_type>>{{row_group_indices.begin(), row_group_indices.end()}};
  return _impl->secondary_filters_byte_ranges(input_row_group_indices, options);
}

std::vector<cudf::size_type> hybrid_scan_reader::filter_row_groups_with_dictionary_pages(
  cudf::host_span<rmm::device_buffer> dictionary_page_data,
  cudf::host_span<size_type const> row_group_indices,
  parquet_reader_options const& options,
  rmm::cuda_stream_view stream) const
{
  CUDF_EXPECTS(row_group_indices.size() == dictionary_page_data.size(),
               "Mismatch in size of input row group indices and dictionary page device buffers");
  auto const input_row_group_indices =
    std::vector<std::vector<size_type>>{{row_group_indices.begin(), row_group_indices.end()}};
  return _impl
    ->filter_row_groups_with_dictionary_pages(
      dictionary_page_data, input_row_group_indices, options, stream)
    .front();
}

std::vector<cudf::size_type> hybrid_scan_reader::filter_row_groups_with_bloom_filters(
  cudf::host_span<rmm::device_buffer> bloom_filter_data,
  cudf::host_span<size_type const> row_group_indices,
  parquet_reader_options const& options,
  rmm::cuda_stream_view stream) const
{
  CUDF_EXPECTS(row_group_indices.size() == bloom_filter_data.size(),
               "Mismatch in size of input row group indices and bloom filter device buffers");
  auto const input_row_group_indices =
    std::vector<std::vector<size_type>>{{row_group_indices.begin(), row_group_indices.end()}};
  return _impl
    ->filter_row_groups_with_bloom_filters(
      bloom_filter_data, input_row_group_indices, options, stream)
    .front();
}

std::pair<std::unique_ptr<cudf::column>, std::vector<thrust::host_vector<bool>>>
hybrid_scan_reader::filter_data_pages_with_stats(cudf::host_span<size_type const> row_group_indices,
                                                 parquet_reader_options const& options,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::device_async_resource_ref mr) const
{
  auto const input_row_group_indices =
    std::vector<std::vector<size_type>>{{row_group_indices.begin(), row_group_indices.end()}};
  return _impl->filter_data_pages_with_stats(input_row_group_indices, options, stream, mr);
}

[[nodiscard]] std::vector<cudf::io::text::byte_range_info>
hybrid_scan_reader::filter_column_chunks_byte_ranges(
  cudf::host_span<size_type const> row_group_indices, parquet_reader_options const& options) const
{
  auto const input_row_group_indices =
    std::vector<std::vector<size_type>>{{row_group_indices.begin(), row_group_indices.end()}};
  return _impl->filter_column_chunks_byte_ranges(input_row_group_indices, options).first;
}

table_with_metadata hybrid_scan_reader::materialize_filter_columns(
  cudf::host_span<thrust::host_vector<bool> const> data_page_mask,
  cudf::host_span<size_type const> row_group_indices,
  std::vector<rmm::device_buffer> column_chunk_buffers,
  cudf::mutable_column_view row_mask,
  parquet_reader_options const& options,
  rmm::cuda_stream_view stream) const
{
  auto const input_row_group_indices =
    std::vector<std::vector<size_type>>{{row_group_indices.begin(), row_group_indices.end()}};
  return _impl->materialize_filter_columns(data_page_mask,
                                           input_row_group_indices,
                                           std::move(column_chunk_buffers),
                                           row_mask,
                                           options,
                                           stream);
}

[[nodiscard]] std::vector<cudf::io::text::byte_range_info>
hybrid_scan_reader::payload_column_chunks_byte_ranges(
  cudf::host_span<size_type const> row_group_indices, parquet_reader_options const& options) const
{
  auto const input_row_group_indices =
    std::vector<std::vector<size_type>>{{row_group_indices.begin(), row_group_indices.end()}};
  return _impl->payload_column_chunks_byte_ranges(input_row_group_indices, options).first;
}

table_with_metadata hybrid_scan_reader::materialize_payload_columns(
  cudf::host_span<size_type const> row_group_indices,
  std::vector<rmm::device_buffer> column_chunk_buffers,
  cudf::column_view row_mask,
  parquet_reader_options const& options,
  rmm::cuda_stream_view stream) const
{
  auto const input_row_group_indices =
    std::vector<std::vector<size_type>>{{row_group_indices.begin(), row_group_indices.end()}};

  return _impl->materialize_payload_columns(
    input_row_group_indices, std::move(column_chunk_buffers), row_mask, options, stream);
}

}  // namespace cudf::io::parquet::experimental
