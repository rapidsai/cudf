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

#include "hybrid_scan_impl.hpp"

#include "cudf/io/text/byte_range_info.hpp"
#include "hybrid_scan_helpers.hpp"

#include <cudf/detail/stream_compaction.hpp>
#include <cudf/detail/transform.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <thrust/iterator/counting_iterator.h>

#include <bitset>
#include <iterator>
#include <limits>
#include <numeric>

namespace cudf::experimental::io::parquet::detail {

using byte_range_info       = cudf::io::text::byte_range_info;
using ColumnChunkDesc       = cudf::io::parquet::detail::ColumnChunkDesc;
using decode_kernel_mask    = cudf::io::parquet::detail::decode_kernel_mask;
using FileMetaData          = cudf::io::parquet::FileMetaData;
using LogicalType           = cudf::io::parquet::LogicalType;
using PageInfo              = cudf::io::parquet::detail::PageInfo;
using PageNestingDecodeInfo = cudf::io::parquet::detail::PageNestingDecodeInfo;
using Type                  = cudf::io::parquet::Type;

impl::impl(cudf::host_span<uint8_t const> footer_bytes,
           cudf::io::parquet_reader_options const& options)
{
  // Open and parse the source dataset metadata
  _metadata = std::make_unique<aggregate_reader_metadata>(
    footer_bytes,
    options.is_enabled_use_arrow_schema(),
    options.get_columns().has_value() and options.is_enabled_allow_mismatched_pq_schemas());
}

FileMetaData const& impl::get_parquet_metadata() const { return _metadata->get_parquet_metadata(); }

cudf::io::text::byte_range_info impl::get_page_index_bytes() const
{
  return _metadata->get_page_index_bytes();
}

void impl::setup_page_index(cudf::host_span<uint8_t const> page_index_bytes) const
{
  _metadata->setup_page_index(page_index_bytes);
}

std::vector<size_type> impl::get_all_row_groups(
  cudf::io::parquet_reader_options const& options) const
{
  auto const num_row_groups = _metadata->get_num_row_groups();
  auto row_groups_indices   = std::vector<size_type>(num_row_groups);
  std::iota(row_groups_indices.begin(), row_groups_indices.end(), size_type{0});
  return row_groups_indices;
}

std::vector<std::vector<size_type>> impl::filter_row_groups_with_stats(
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  cudf::io::parquet_reader_options const& options,
  rmm::cuda_stream_view stream)
{
  return {};
}

std::pair<std::vector<byte_range_info>, std::vector<byte_range_info>> impl::get_secondary_filters(
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  cudf::io::parquet_reader_options const& options)
{
  return {};
}

std::vector<std::vector<size_type>> impl::filter_row_groups_with_dictionary_pages(
  std::vector<rmm::device_buffer>& dictionary_page_data,
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  cudf::io::parquet_reader_options const& options,
  rmm::cuda_stream_view stream)
{
  return {};
}

std::vector<std::vector<size_type>> impl::filter_row_groups_with_bloom_filters(
  std::vector<rmm::device_buffer>& bloom_filter_data,
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  cudf::io::parquet_reader_options const& options,
  rmm::cuda_stream_view stream)
{
  return {};
}

std::pair<std::unique_ptr<cudf::column>, std::vector<std::vector<bool>>>
impl::filter_data_pages_with_stats(cudf::host_span<std::vector<size_type> const> row_group_indices,
                                   cudf::io::parquet_reader_options const& options,
                                   rmm::cuda_stream_view stream,
                                   rmm::device_async_resource_ref mr)
{
  return {};
}

std::pair<std::vector<byte_range_info>, std::vector<cudf::size_type>>
impl::get_input_column_chunk_byte_ranges(
  cudf::host_span<std::vector<size_type> const> row_group_indices) const
{
  return {};
}

std::pair<std::vector<byte_range_info>, std::vector<cudf::size_type>>
impl::get_filter_column_chunk_byte_ranges(
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  cudf::io::parquet_reader_options const& options)
{
  return {};
}

std::pair<std::vector<byte_range_info>, std::vector<cudf::size_type>>
impl::get_payload_column_chunk_byte_ranges(
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  cudf::io::parquet_reader_options const& options)
{
  return {};
}

cudf::io::table_with_metadata impl::materialize_filter_columns(
  cudf::host_span<std::vector<bool> const> data_page_mask,
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  std::vector<rmm::device_buffer> column_chunk_buffers,
  cudf::mutable_column_view row_mask,
  cudf::io::parquet_reader_options const& options,
  rmm::cuda_stream_view stream)
{
  return {};
}

cudf::io::table_with_metadata impl::materialize_payload_columns(
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  std::vector<rmm::device_buffer> column_chunk_buffers,
  cudf::column_view row_mask,
  cudf::io::parquet_reader_options const& options,
  rmm::cuda_stream_view stream)
{
  return {};
}

}  // namespace cudf::experimental::io::parquet::detail
