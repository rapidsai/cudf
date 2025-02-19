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
#include "io/parquet/experimental/hybrid_scan_impl.hpp"

#include <cudf/io/experimental/hybrid_scan.hpp>

namespace cudf::experimental::io::parquet {

// API # 1
std::unique_ptr<hybrid_scan_reader> make_hybrid_scan_reader(
  cudf::host_span<uint8_t const> footer_bytes,
  cudf::host_span<uint8_t const> page_index_bytes,
  cudf::io::parquet_reader_options const& options)
{
  return std::make_unique<hybrid_scan_reader>(footer_bytes, page_index_bytes, options);
}

// API # 2
std::vector<size_type> get_valid_row_groups(std::unique_ptr<hybrid_scan_reader> reader,
                                            cudf::io::parquet_reader_options const& options)
{
  return reader->get_valid_row_groups(options);
}

// API # 3
std::vector<size_type> filter_row_groups_with_stats(
  std::unique_ptr<hybrid_scan_reader> reader,
  cudf::host_span<size_type const> row_group_indices,
  cudf::io::parquet_reader_options const& options,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const input_row_group_indices =
    std::vector<std::vector<size_type>>{{row_group_indices.begin(), row_group_indices.end()}};
  return reader->filter_row_groups_with_stats(input_row_group_indices, options, stream, mr)[0];
}

// API # 4
[[nodiscard]] std::pair<std::vector<cudf::io::text::byte_range_info>,
                        std::vector<cudf::io::text::byte_range_info>>
get_secondary_filters(std::unique_ptr<hybrid_scan_reader> reader,
                      cudf::host_span<size_type const> row_group_indices,
                      cudf::io::parquet_reader_options const& options)
{
  auto const input_row_group_indices =
    std::vector<std::vector<size_type>>{{row_group_indices.begin(), row_group_indices.end()}};
  return reader->get_secondary_filters(input_row_group_indices, options);
}

// API # 5
std::vector<size_type> filter_row_groups_with_dictionary_pages(
  std::unique_ptr<hybrid_scan_reader> reader,
  std::vector<rmm::device_buffer>& dictionary_page_data,
  cudf::host_span<size_type const> row_group_indices,
  cudf::io::parquet_reader_options const& options,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const input_row_group_indices =
    std::vector<std::vector<size_type>>{{row_group_indices.begin(), row_group_indices.end()}};
  return reader->filter_row_groups_with_dictionary_pages(
    dictionary_page_data, input_row_group_indices, options, stream, mr)[0];
}

// API # 6
std::vector<size_type> filter_row_groups_with_bloom_filters(
  std::unique_ptr<hybrid_scan_reader> reader,
  std::vector<rmm::device_buffer>& bloom_filter_data,
  cudf::host_span<size_type const> row_group_indices,
  cudf::io::parquet_reader_options const& options,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const input_row_group_indices =
    std::vector<std::vector<size_type>>{{row_group_indices.begin(), row_group_indices.end()}};
  return reader->filter_row_groups_with_bloom_filters(
    bloom_filter_data, input_row_group_indices, options, stream, mr)[0];
}

// API # 7
std::unique_ptr<cudf::column> filter_data_pages_with_stats(
  std::unique_ptr<parquet::hybrid_scan_reader> reader,
  cudf::host_span<size_type const> row_group_indices,
  cudf::io::parquet_reader_options const& options,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const input_row_group_indices =
    std::vector<std::vector<size_type>>{{row_group_indices.begin(), row_group_indices.end()}};
  return reader->filter_data_pages_with_stats(input_row_group_indices, options, stream, mr);
}

// API # 8
std::vector<std::vector<cudf::io::text::byte_range_info>> get_filter_columns_data_pages(
  std::unique_ptr<parquet::hybrid_scan_reader> reader,
  cudf::column_view input_rows,
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  cudf::io::parquet_reader_options const& options,
  rmm::cuda_stream_view stream)
{
  {
    auto const input_row_group_indices =
      std::vector<std::vector<size_type>>{{row_group_indices.begin(), row_group_indices.end()}};
    return reader->get_filter_columns_data_pages(
      input_rows, input_row_group_indices, options, stream);
  }
}

}  // namespace cudf::experimental::io::parquet