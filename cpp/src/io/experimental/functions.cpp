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

#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/utilities/error.hpp>

namespace cudf::experimental::io {

using hybrid_scan_reader = parquet::hybrid_scan_reader;

// API # 1
std::unique_ptr<hybrid_scan_reader> make_hybrid_scan_reader(
  cudf::host_span<uint8_t const> footer_bytes, cudf::io::parquet_reader_options const& options)
{
  return std::make_unique<hybrid_scan_reader>(footer_bytes, options);
}

// API # 2
[[nodiscard]] cudf::io::parquet::FileMetaData const& get_parquet_metadata(
  std::unique_ptr<hybrid_scan_reader> const& reader)
{
  return reader->get_parquet_metadata();
}

// API # 3
[[nodiscard]] cudf::io::text::byte_range_info get_page_index_bytes(
  std::unique_ptr<hybrid_scan_reader> const& reader)
{
  return reader->get_page_index_bytes();
}

// API # 4
void setup_page_index(std::unique_ptr<hybrid_scan_reader> const& reader,
                      cudf::host_span<uint8_t const> page_index_bytes)
{
  return reader->setup_page_index(page_index_bytes);
}

// API # 5
std::vector<size_type> get_all_row_groups(std::unique_ptr<hybrid_scan_reader> const& reader,
                                          cudf::io::parquet_reader_options const& options)
{
  return reader->get_all_row_groups(options);
}

// API # 6
std::vector<size_type> filter_row_groups_with_stats(
  std::unique_ptr<hybrid_scan_reader> const& reader,
  cudf::host_span<size_type const> row_group_indices,
  cudf::io::parquet_reader_options const& options,
  rmm::cuda_stream_view stream)
{
  auto const input_row_group_indices =
    std::vector<std::vector<size_type>>{{row_group_indices.begin(), row_group_indices.end()}};
  return reader->filter_row_groups_with_stats(input_row_group_indices, options, stream)[0];
}

// API # 7
[[nodiscard]] std::pair<std::vector<cudf::io::text::byte_range_info>,
                        std::vector<cudf::io::text::byte_range_info>>
get_secondary_filters(std::unique_ptr<hybrid_scan_reader> const& reader,
                      cudf::host_span<size_type const> row_group_indices,
                      cudf::io::parquet_reader_options const& options)
{
  auto const input_row_group_indices =
    std::vector<std::vector<size_type>>{{row_group_indices.begin(), row_group_indices.end()}};
  return reader->get_secondary_filters(input_row_group_indices, options);
}

// API # 8
std::vector<size_type> filter_row_groups_with_dictionary_pages(
  std::unique_ptr<hybrid_scan_reader> const& reader,
  std::vector<rmm::device_buffer>& dictionary_page_data,
  cudf::host_span<size_type const> row_group_indices,
  cudf::io::parquet_reader_options const& options,
  rmm::cuda_stream_view stream)
{
  auto const input_row_group_indices =
    std::vector<std::vector<size_type>>{{row_group_indices.begin(), row_group_indices.end()}};
  return reader->filter_row_groups_with_dictionary_pages(
    dictionary_page_data, input_row_group_indices, options, stream)[0];
}

// API # 9
std::vector<size_type> filter_row_groups_with_bloom_filters(
  std::unique_ptr<hybrid_scan_reader> const& reader,
  std::vector<rmm::device_buffer>& bloom_filter_data,
  cudf::host_span<size_type const> row_group_indices,
  cudf::io::parquet_reader_options const& options,
  rmm::cuda_stream_view stream)
{
  auto const input_row_group_indices =
    std::vector<std::vector<size_type>>{{row_group_indices.begin(), row_group_indices.end()}};
  return reader->filter_row_groups_with_bloom_filters(
    bloom_filter_data, input_row_group_indices, options, stream)[0];
}

// API # 10
std::pair<std::unique_ptr<cudf::column>, std::vector<std::vector<bool>>>
filter_data_pages_with_stats(std::unique_ptr<parquet::hybrid_scan_reader> const& reader,
                             cudf::host_span<size_type const> row_group_indices,
                             cudf::io::parquet_reader_options const& options,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr)
{
  auto const input_row_group_indices =
    std::vector<std::vector<size_type>>{{row_group_indices.begin(), row_group_indices.end()}};
  return reader->filter_data_pages_with_stats(input_row_group_indices, options, stream, mr);
}

// API # 11
std::vector<cudf::io::text::byte_range_info> get_filter_column_chunk_byte_ranges(
  std::unique_ptr<parquet::hybrid_scan_reader> const& reader,
  cudf::host_span<size_type const> row_group_indices,
  cudf::io::parquet_reader_options const& options)
{
  auto const input_row_group_indices =
    std::vector<std::vector<size_type>>{{row_group_indices.begin(), row_group_indices.end()}};
  return reader->get_filter_column_chunk_byte_ranges(input_row_group_indices, options).first;
}

// API # 12
cudf::io::table_with_metadata materialize_filter_columns(
  std::unique_ptr<parquet::hybrid_scan_reader> const& reader,
  cudf::host_span<std::vector<bool> const> filtered_data_pages,
  cudf::host_span<size_type const> row_group_indices,
  std::vector<rmm::device_buffer> column_chunk_buffers,
  cudf::mutable_column_view predicate,
  cudf::io::parquet_reader_options const& options,
  rmm::cuda_stream_view stream)
{
  auto const input_row_group_indices =
    std::vector<std::vector<size_type>>{{row_group_indices.begin(), row_group_indices.end()}};
  return reader->materialize_filter_columns(filtered_data_pages,
                                            input_row_group_indices,
                                            std::move(column_chunk_buffers),
                                            predicate,
                                            options,
                                            stream);
}

// API # 13
std::vector<cudf::io::text::byte_range_info> get_payload_column_chunk_byte_ranges(
  std::unique_ptr<parquet::hybrid_scan_reader> const& reader,
  cudf::host_span<size_type const> row_group_indices,
  cudf::io::parquet_reader_options const& options)
{
  auto const input_row_group_indices =
    std::vector<std::vector<size_type>>{{row_group_indices.begin(), row_group_indices.end()}};
  return reader->get_payload_column_chunk_byte_ranges(input_row_group_indices, options).first;
}

// API # 14
cudf::io::table_with_metadata materialize_payload_columns(
  std::unique_ptr<parquet::hybrid_scan_reader> const& reader,
  cudf::host_span<size_type const> row_group_indices,
  std::vector<rmm::device_buffer> column_chunk_buffers,
  cudf::column_view predicate,
  cudf::io::parquet_reader_options const& options,
  rmm::cuda_stream_view stream)
{
  auto const input_row_group_indices =
    std::vector<std::vector<size_type>>{{row_group_indices.begin(), row_group_indices.end()}};
  return reader->materialize_payload_columns(
    input_row_group_indices, std::move(column_chunk_buffers), predicate, options, stream);
}

}  // namespace cudf::experimental::io
