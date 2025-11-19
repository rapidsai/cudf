
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hybrid_scan_common.hpp"

#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/mr/aligned_resource_adaptor.hpp>

#include <format>
#include <string>

cudf::host_span<uint8_t const> fetch_footer_bytes(cudf::host_span<uint8_t const> buffer)
{
  using namespace cudf::io::parquet;

  constexpr auto header_len = sizeof(file_header_s);
  constexpr auto ender_len  = sizeof(file_ender_s);
  size_t const len          = buffer.size();

  auto const header_buffer = cudf::host_span<uint8_t const>(buffer.data(), header_len);
  auto const header        = reinterpret_cast<file_header_s const*>(header_buffer.data());
  auto const ender_buffer =
    cudf::host_span<uint8_t const>(buffer.data() + len - ender_len, ender_len);
  auto const ender = reinterpret_cast<file_ender_s const*>(ender_buffer.data());
  CUDF_EXPECTS(len > header_len + ender_len, "Incorrect data source");
  constexpr uint32_t parquet_magic = (('P' << 0) | ('A' << 8) | ('R' << 16) | ('1' << 24));
  CUDF_EXPECTS(header->magic == parquet_magic && ender->magic == parquet_magic,
               "Corrupted header or footer");
  CUDF_EXPECTS(ender->footer_len != 0 && ender->footer_len <= (len - header_len - ender_len),
               "Incorrect footer length");

  return cudf::host_span<uint8_t const>(buffer.data() + len - ender->footer_len - ender_len,
                                        ender->footer_len);
}

cudf::host_span<uint8_t const> fetch_page_index_bytes(
  cudf::host_span<uint8_t const> buffer, cudf::io::text::byte_range_info const page_index_bytes)
{
  return cudf::host_span<uint8_t const>(
    reinterpret_cast<uint8_t const*>(buffer.data()) + page_index_bytes.offset(),
    page_index_bytes.size());
}

std::vector<rmm::device_buffer> fetch_byte_ranges(
  cudf::host_span<uint8_t const> host_buffer,
  cudf::host_span<cudf::io::text::byte_range_info const> byte_ranges,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  std::vector<rmm::device_buffer> buffers{};
  buffers.reserve(byte_ranges.size());

  std::transform(
    byte_ranges.begin(),
    byte_ranges.end(),
    std::back_inserter(buffers),
    [&](auto const& byte_range) {
      auto const chunk_offset = host_buffer.data() + byte_range.offset();
      auto const chunk_size   = byte_range.size();
      auto buffer             = rmm::device_buffer(chunk_size, stream, mr);
      CUDF_CUDA_TRY(cudaMemcpyAsync(
        buffer.data(), chunk_offset, chunk_size, cudaMemcpyHostToDevice, stream.value()));
      return buffer;
    });

  stream.synchronize_no_throw();
  return buffers;
}

cudf::test::strings_column_wrapper constant_strings(cudf::size_type value)
{
  CUDF_EXPECTS(value >= 0 && value <= 9999, "String value must be between 0000 and 9999");

  auto elements = thrust::make_transform_iterator(thrust::make_constant_iterator(value),
                                                  [](auto i) { return std::format("{:04d}", i); });
  return cudf::test::strings_column_wrapper(elements, elements + num_ordered_rows);
}

std::unique_ptr<cudf::table> concatenate_tables(std::vector<std::unique_ptr<cudf::table>> tables,
                                                rmm::cuda_stream_view stream)
{
  if (tables.size() == 1) { return std::move(tables[0]); }

  std::vector<cudf::table_view> table_views;
  table_views.reserve(tables.size());
  std::transform(
    tables.begin(), tables.end(), std::back_inserter(table_views), [&](auto const& tbl) {
      return tbl->view();
    });
  // Construct the final table
  return cudf::concatenate(table_views, stream);
}

/**
 * @brief Apply parquet filters to the file buffer
 *
 * @param file_buffer_span Input file buffer span
 * @param options Reader options
 * @param stream CUDA stream
 * @param mr Device memory resource
 *
 * @return A tuple of the reader, filtered row group indices, and row mask and data page mask from
 * data page pruning
 */
auto apply_parquet_filters(cudf::host_span<uint8_t const> file_buffer_span,
                           cudf::io::parquet_reader_options const& options,
                           rmm::cuda_stream_view stream,
                           rmm::device_async_resource_ref mr)
{
  // Fetch footer and page index bytes from the buffer.
  auto const footer_buffer = fetch_footer_bytes(file_buffer_span);

  // Create hybrid scan reader with footer bytes
  auto reader =
    std::make_unique<cudf::io::parquet::experimental::hybrid_scan_reader>(footer_buffer, options);

  // Get page index byte range from the reader
  auto const page_index_byte_range = reader->page_index_byte_range();

  // Fetch page index bytes from the input buffer
  auto const page_index_buffer = fetch_page_index_bytes(file_buffer_span, page_index_byte_range);

  // Setup page index
  reader->setup_page_index(page_index_buffer);

  // Get all row groups from the reader
  auto input_row_group_indices = reader->all_row_groups(options);

  // Span to track current row group indices
  auto current_row_group_indices = cudf::host_span<cudf::size_type>(input_row_group_indices);

  // Filter row groups with stats
  auto stats_filtered_row_group_indices =
    reader->filter_row_groups_with_stats(current_row_group_indices, options, stream);

  // Update current row group indices
  current_row_group_indices = stats_filtered_row_group_indices;

  // Get bloom filter and dictionary page byte ranges from the reader
  auto [bloom_filter_byte_ranges, dict_page_byte_ranges] =
    reader->secondary_filters_byte_ranges(current_row_group_indices, options);

  // If we have dictionary page byte ranges, filter row groups with dictionary pages
  std::vector<cudf::size_type> dictionary_page_filtered_row_group_indices;
  dictionary_page_filtered_row_group_indices.reserve(current_row_group_indices.size());
  if (dict_page_byte_ranges.size()) {
    // Fetch dictionary page buffers from the input file buffer
    std::vector<rmm::device_buffer> dictionary_page_buffers =
      fetch_byte_ranges(file_buffer_span, dict_page_byte_ranges, stream, mr);

    // Filter row groups with dictionary pages
    dictionary_page_filtered_row_group_indices = reader->filter_row_groups_with_dictionary_pages(
      dictionary_page_buffers, current_row_group_indices, options, stream);

    // Update current row group indices
    current_row_group_indices = dictionary_page_filtered_row_group_indices;
  }

  // If we have bloom filter byte ranges, filter row groups with bloom filters
  std::vector<cudf::size_type> bloom_filtered_row_group_indices;
  bloom_filtered_row_group_indices.reserve(current_row_group_indices.size());
  if (bloom_filter_byte_ranges.size()) {
    // Fetch 32 byte aligned bloom filter data buffers from the input file buffer
    auto aligned_mr = rmm::mr::aligned_resource_adaptor<rmm::device_async_resource_ref>(
      cudf::get_current_device_resource_ref(), bloom_filter_alignment);

    std::vector<rmm::device_buffer> bloom_filter_data =
      fetch_byte_ranges(file_buffer_span, bloom_filter_byte_ranges, stream, aligned_mr);

    // Filter row groups with bloom filters
    bloom_filtered_row_group_indices = reader->filter_row_groups_with_bloom_filters(
      bloom_filter_data, current_row_group_indices, options, stream);

    // Update current row group indices
    current_row_group_indices = bloom_filtered_row_group_indices;
  }

  // Build row mask using page index stats
  auto row_mask =
    reader->build_row_mask_with_page_index_stats(current_row_group_indices, options, stream, mr);

  std::vector<cudf::size_type> final_row_group_indices(current_row_group_indices.begin(),
                                                       current_row_group_indices.end());

  return std::tuple{std::move(reader), std::move(final_row_group_indices), std::move(row_mask)};
}

std::tuple<std::unique_ptr<cudf::table>,
           std::unique_ptr<cudf::table>,
           cudf::io::table_metadata,
           cudf::io::table_metadata,
           std::unique_ptr<cudf::column>>
hybrid_scan(std::vector<char>& buffer,
            cudf::ast::operation const& filter_expression,
            cudf::size_type num_filter_columns,
            std::optional<std::vector<std::string>> const& payload_column_names,
            rmm::cuda_stream_view stream,
            rmm::device_async_resource_ref mr,
            rmm::mr::aligned_resource_adaptor<rmm::mr::device_memory_resource>& aligned_mr)
{
  // Create reader options with empty source info
  cudf::io::parquet_reader_options options =
    cudf::io::parquet_reader_options::builder().filter(filter_expression);

  // Set payload column names if provided
  if (payload_column_names.has_value()) { options.set_columns(payload_column_names.value()); }

  // Input file buffer span
  auto const file_buffer_span =
    cudf::host_span<uint8_t const>(reinterpret_cast<uint8_t const*>(buffer.data()), buffer.size());

  auto [reader, filtered_row_group_indices, row_mask] =
    apply_parquet_filters(file_buffer_span, options, stream, mr);

  auto current_row_group_indices = cudf::host_span<cudf::size_type>(filtered_row_group_indices);

  // Get column chunk byte ranges from the reader
  auto const filter_column_chunk_byte_ranges =
    reader->filter_column_chunks_byte_ranges(current_row_group_indices, options);

  // Fetch column chunk device buffers from the input buffer
  auto filter_column_chunk_buffers =
    fetch_byte_ranges(file_buffer_span, filter_column_chunk_byte_ranges, stream, mr);

  // Materialize the table with only the filter columns
  auto row_mask_mutable_view = row_mask->mutable_view();
  auto [filter_table, filter_metadata] =
    reader->materialize_filter_columns(current_row_group_indices,
                                       std::move(filter_column_chunk_buffers),
                                       row_mask_mutable_view,
                                       cudf::io::parquet::experimental::use_data_page_mask::YES,
                                       options,
                                       stream);

  // Get column chunk byte ranges from the reader
  auto const payload_column_chunk_byte_ranges =
    reader->payload_column_chunks_byte_ranges(current_row_group_indices, options);

  // Fetch column chunk device buffers from the input buffer
  auto payload_column_chunk_buffers =
    fetch_byte_ranges(file_buffer_span, payload_column_chunk_byte_ranges, stream, mr);

  // Materialize the table with only the payload columns
  auto [payload_table, payload_metadata] =
    reader->materialize_payload_columns(current_row_group_indices,
                                        std::move(payload_column_chunk_buffers),
                                        row_mask->view(),
                                        cudf::io::parquet::experimental::use_data_page_mask::YES,
                                        options,
                                        stream);

  return std::tuple{std::move(filter_table),
                    std::move(payload_table),
                    std::move(filter_metadata),
                    std::move(payload_metadata),
                    std::move(row_mask)};
}

std::tuple<std::unique_ptr<cudf::table>,
           std::unique_ptr<cudf::table>,
           cudf::io::table_metadata,
           cudf::io::table_metadata,
           std::unique_ptr<cudf::column>>
chunked_hybrid_scan(std::vector<char> const& buffer,
                    cudf::ast::operation const& filter_expression,
                    cudf::size_type num_filter_columns,
                    std::optional<std::vector<std::string>> const& payload_column_names,
                    rmm::cuda_stream_view stream,
                    rmm::device_async_resource_ref mr,
                    rmm::mr::aligned_resource_adaptor<rmm::mr::device_memory_resource>& aligned_mr)
{
  // Create reader options with empty source info
  cudf::io::parquet_reader_options options =
    cudf::io::parquet_reader_options::builder().filter(filter_expression);

  // Set payload column names if provided
  if (payload_column_names.has_value()) { options.set_columns(payload_column_names.value()); }

  // Input file buffer span
  auto const file_buffer_span =
    cudf::host_span<uint8_t const>(reinterpret_cast<uint8_t const*>(buffer.data()), buffer.size());

  auto [reader, filtered_row_group_indices, row_mask] =
    apply_parquet_filters(file_buffer_span, options, stream, mr);

  auto current_row_group_indices = cudf::host_span<cudf::size_type>(filtered_row_group_indices);

  // Helper to split the materialization of filter columns into chunks
  auto tables          = std::vector<std::unique_ptr<cudf::table>>{};
  auto filter_metadata = cudf::io::table_metadata{};
  auto const materialize_filter_columns =
    [&](cudf::host_span<cudf::size_type const> row_group_indices) {
      // Get column chunk byte ranges from the reader and fetch device buffers
      auto const filter_column_chunk_byte_ranges =
        reader->filter_column_chunks_byte_ranges(row_group_indices, options);
      auto filter_column_chunk_buffers =
        fetch_byte_ranges(file_buffer_span, filter_column_chunk_byte_ranges, stream, mr);

      // Setup chunking for filter columns and materialize the columns
      reader->setup_chunking_for_filter_columns(
        1024,
        10240,
        row_group_indices,
        row_mask->view(),
        cudf::io::parquet::experimental::use_data_page_mask::YES,
        std::move(filter_column_chunk_buffers),
        options,
        stream);

      auto row_mask_mutable_view = row_mask->mutable_view();
      while (reader->has_next_table_chunk()) {
        auto chunk = reader->materialize_filter_columns_chunk(row_mask_mutable_view, stream);
        tables.push_back(std::move(chunk.tbl));
        filter_metadata = std::move(chunk.metadata);
      }
    };

  if (current_row_group_indices.size() > 1) {
    auto const row_group_split = current_row_group_indices.size() / 2;
    materialize_filter_columns(
      cudf::host_span<cudf::size_type const>{current_row_group_indices.begin(), row_group_split});
    materialize_filter_columns(
      cudf::host_span<cudf::size_type const>{current_row_group_indices.begin() + row_group_split,
                                             current_row_group_indices.size() - row_group_split});
  } else {
    materialize_filter_columns(current_row_group_indices);
  }

  auto filter_table = concatenate_tables(std::move(tables), stream);

  // Helper to split the materialization of payload columns into chunks
  tables.clear();
  auto payload_metadata = cudf::io::table_metadata{};
  auto const materialize_payload_columns =
    [&](cudf::host_span<cudf::size_type const> row_group_indices) {
      // Get column chunk byte ranges from the reader and fetch device buffers
      auto const payload_column_chunk_byte_ranges =
        reader->payload_column_chunks_byte_ranges(row_group_indices, options);
      auto payload_column_chunk_buffers =
        fetch_byte_ranges(file_buffer_span, payload_column_chunk_byte_ranges, stream, mr);

      // Setup chunking for payload columns and materialize the table
      reader->setup_chunking_for_payload_columns(
        1024,
        10240,
        row_group_indices,
        row_mask->view(),
        cudf::io::parquet::experimental::use_data_page_mask::YES,
        std::move(payload_column_chunk_buffers),
        options,
        stream);

      while (reader->has_next_table_chunk()) {
        auto chunk = reader->materialize_payload_columns_chunk(row_mask->view(), stream);
        tables.push_back(std::move(chunk.tbl));
        payload_metadata = std::move(chunk.metadata);
      }
    };

  if (current_row_group_indices.size() > 1) {
    auto const row_group_split = current_row_group_indices.size() / 2;
    materialize_payload_columns(
      cudf::host_span<cudf::size_type const>{current_row_group_indices.begin(), row_group_split});
    materialize_payload_columns(
      cudf::host_span<cudf::size_type const>{current_row_group_indices.begin() + row_group_split,
                                             current_row_group_indices.size() - row_group_split});
  } else {
    materialize_payload_columns(current_row_group_indices);
  }

  auto payload_table = concatenate_tables(std::move(tables), stream);

  // Return the filter table and metadata, payload table and metadata, and the final row mask
  return std::tuple{std::move(filter_table),
                    std::move(payload_table),
                    std::move(filter_metadata),
                    std::move(payload_metadata),
                    std::move(row_mask)};
}
