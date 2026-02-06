/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hybrid_scan_composer.hpp"

#include <cudf/concatenate.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/mr/aligned_resource_adaptor.hpp>

#include <vector>

using cudf::io::parquet::experimental::hybrid_scan_reader;

namespace {

/**
 * @brief Converts a datasource buffer to a host span
 *
 * @param buffer Datasource buffer
 *
 * @return Host span of the datasource buffer
 */
cudf::host_span<uint8_t const> make_host_span(cudf::io::datasource::buffer const& buffer)
{
  return cudf::host_span<uint8_t const>{static_cast<uint8_t const*>(buffer.data()), buffer.size()};
}

/**
 * @brief Sets up the a hybrid scan reader instance
 *
 * @param datasource Data source
 * @param options Parquet reader options
 * @param verbose Whether to print verbose output
 *
 * @return Hybrid scan reader
 */

std::unique_ptr<hybrid_scan_reader> setup_reader(cudf::io::datasource& datasource,
                                                 cudf::io::parquet_reader_options const& options)
{
  // Fetch footer bytes and setup reader
  auto const footer_buffer = cudf::io::parquet::fetch_footer_to_host(datasource);
  return std::make_unique<hybrid_scan_reader>(make_host_span(*footer_buffer), options);
}

/**
 * @brief Sets up the page index for the hybrid scan reader
 *
 * @param datasource Data source
 * @param reader Hybrid scan reader
 */
void setup_page_index(cudf::io::datasource& datasource, hybrid_scan_reader const& reader)
{
  auto const page_index_byte_range = reader.page_index_byte_range();
  if (not page_index_byte_range.is_empty()) {
    auto const page_index_buffer =
      cudf::io::parquet::fetch_page_index_to_host(datasource, page_index_byte_range);
    reader.setup_page_index(make_host_span(*page_index_buffer));
  }
}

/*
 * @brief Concatenate a vector of tables and return the resultant table
 *
 * @param tables Vector of tables to concatenate
 * @param stream CUDA stream to use
 *
 * @return Unique pointer to the resultant concatenated table.
 */
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

}  // namespace

auto apply_hybrid_scan_filters(cudf::io::datasource& datasource,
                               hybrid_scan_reader const& reader,
                               cudf::io::parquet_reader_options const& options,
                               rmm::cuda_stream_view stream,
                               rmm::device_async_resource_ref mr)
{
  // Get all row groups from the reader
  auto input_row_group_indices = reader.all_row_groups(options);

  // Span to track current row group indices
  auto current_row_group_indices = cudf::host_span<cudf::size_type>(input_row_group_indices);

  // Filter row groups with stats
  auto stats_filtered_row_group_indices =
    reader.filter_row_groups_with_stats(current_row_group_indices, options, stream);

  // Update current row group indices
  current_row_group_indices = stats_filtered_row_group_indices;

  // Get bloom filter and dictionary page byte ranges from the reader
  auto [bloom_filter_byte_ranges, dict_page_byte_ranges] =
    reader.secondary_filters_byte_ranges(current_row_group_indices, options);

  // If we have dictionary page byte ranges, filter row groups with dictionary pages
  std::vector<cudf::size_type> dictionary_page_filtered_row_group_indices;
  dictionary_page_filtered_row_group_indices.reserve(current_row_group_indices.size());
  if (dict_page_byte_ranges.size()) {
    // Fetch dictionary page buffers from the input file buffer
    auto [dict_page_buffers, dict_page_data, dict_read_tasks] =
      cudf::io::parquet::fetch_byte_ranges_to_device_async(
        datasource, dict_page_byte_ranges, stream, mr);
    dict_read_tasks.get();

    // Filter row groups with dictionary pages
    dictionary_page_filtered_row_group_indices = reader.filter_row_groups_with_dictionary_pages(
      dict_page_data, current_row_group_indices, options, stream);

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

    auto [bloom_filter_buffers, bloom_filter_data, bloom_read_tasks] =
      cudf::io::parquet::fetch_byte_ranges_to_device_async(
        datasource, bloom_filter_byte_ranges, stream, aligned_mr);
    bloom_read_tasks.get();

    // Filter row groups with bloom filters
    bloom_filtered_row_group_indices = reader.filter_row_groups_with_bloom_filters(
      bloom_filter_data, current_row_group_indices, options, stream);

    // Update current row group indices
    current_row_group_indices = bloom_filtered_row_group_indices;
  }

  // Build row mask using page index stats or all true if no filter is provided
  auto row_mask = [&]() {
    if (options.get_filter().has_value()) {
      return reader.build_row_mask_with_page_index_stats(
        current_row_group_indices, options, stream, mr);
    } else {
      return reader.build_all_true_row_mask(current_row_group_indices, stream, mr);
    }
  }();

  std::vector<cudf::size_type> final_row_group_indices(current_row_group_indices.begin(),
                                                       current_row_group_indices.end());

  return std::tuple{std::move(final_row_group_indices), std::move(row_mask)};
}

std::tuple<std::unique_ptr<cudf::table>,
           std::unique_ptr<cudf::table>,
           cudf::io::table_metadata,
           cudf::io::table_metadata,
           std::unique_ptr<cudf::column>>
hybrid_scan(cudf::io::datasource& datasource,
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
  if (payload_column_names.has_value()) { options.set_column_names(payload_column_names.value()); }

  auto const reader = setup_reader(datasource, options);
  auto reader_ref   = std::ref(*reader);

  setup_page_index(datasource, reader_ref);

  auto [filtered_row_group_indices, row_mask] =
    apply_hybrid_scan_filters(datasource, reader_ref, options, stream, mr);

  auto current_row_group_indices = cudf::host_span<cudf::size_type>(filtered_row_group_indices);

  // Get column chunk byte ranges from the reader
  auto const filter_column_chunk_byte_ranges =
    reader->filter_column_chunks_byte_ranges(current_row_group_indices, options);

  // Fetch column chunk device buffers and spans from the input buffer
  auto [filter_col_buffers, filter_col_data, filter_col_tasks] =
    cudf::io::parquet::fetch_byte_ranges_to_device_async(
      datasource, filter_column_chunk_byte_ranges, stream, mr);
  filter_col_tasks.get();

  // Materialize the table with only the filter columns
  auto row_mask_mutable_view = row_mask->mutable_view();
  auto [filter_table, filter_metadata] =
    reader->materialize_filter_columns(current_row_group_indices,
                                       filter_col_data,
                                       row_mask_mutable_view,
                                       cudf::io::parquet::experimental::use_data_page_mask::YES,
                                       options,
                                       stream,
                                       mr);

  // Get column chunk byte ranges from the reader
  auto const payload_column_chunk_byte_ranges =
    reader->payload_column_chunks_byte_ranges(current_row_group_indices, options);

  // Fetch column chunk device buffers and spans from the input buffer
  auto [payload_col_buffers, payload_col_data, payload_col_tasks] =
    cudf::io::parquet::fetch_byte_ranges_to_device_async(
      datasource, payload_column_chunk_byte_ranges, stream, mr);
  payload_col_tasks.get();

  // Materialize the table with only the payload columns
  auto [payload_table, payload_metadata] =
    reader->materialize_payload_columns(current_row_group_indices,
                                        payload_col_data,
                                        row_mask->view(),
                                        cudf::io::parquet::experimental::use_data_page_mask::YES,
                                        options,
                                        stream,
                                        mr);

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
chunked_hybrid_scan(cudf::io::datasource& datasource,
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
  if (payload_column_names.has_value()) { options.set_column_names(payload_column_names.value()); }

  auto const reader = setup_reader(datasource, options);
  auto reader_ref   = std::ref(*reader);

  setup_page_index(datasource, reader_ref);

  auto [filtered_row_group_indices, row_mask] =
    apply_hybrid_scan_filters(datasource, reader_ref, options, stream, mr);

  auto current_row_group_indices = cudf::host_span<cudf::size_type>(filtered_row_group_indices);

  // Helper to split the materialization of filter columns into chunks
  auto tables          = std::vector<std::unique_ptr<cudf::table>>{};
  auto filter_metadata = cudf::io::table_metadata{};
  auto const materialize_filter_columns =
    [&](cudf::host_span<cudf::size_type const> row_group_indices) {
      // Get column chunk byte ranges from the reader and fetch device buffers
      auto const filter_column_chunk_byte_ranges =
        reader->filter_column_chunks_byte_ranges(row_group_indices, options);
      auto [filter_col_buffers, filter_col_data, filter_col_tasks] =
        cudf::io::parquet::fetch_byte_ranges_to_device_async(
          datasource, filter_column_chunk_byte_ranges, stream, mr);
      filter_col_tasks.get();

      // Setup chunking for filter columns and materialize the columns
      reader->setup_chunking_for_filter_columns(
        1024,
        10240,
        row_group_indices,
        row_mask->view(),
        cudf::io::parquet::experimental::use_data_page_mask::YES,
        filter_col_data,
        options,
        stream,
        mr);

      auto row_mask_mutable_view = row_mask->mutable_view();
      while (reader->has_next_table_chunk()) {
        auto chunk = reader->materialize_filter_columns_chunk(row_mask_mutable_view);
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
      auto [payload_col_buffers, payload_col_data, payload_col_tasks] =
        cudf::io::parquet::fetch_byte_ranges_to_device_async(
          datasource, payload_column_chunk_byte_ranges, stream, mr);
      payload_col_tasks.get();

      // Setup chunking for payload columns and materialize the table
      reader->setup_chunking_for_payload_columns(
        1024,
        10240,
        row_group_indices,
        row_mask->view(),
        cudf::io::parquet::experimental::use_data_page_mask::YES,
        payload_col_data,
        options,
        stream,
        mr);

      while (reader->has_next_table_chunk()) {
        auto chunk = reader->materialize_payload_columns_chunk(row_mask->view());
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

cudf::io::table_with_metadata hybrid_scan_single_step(
  cudf::io::datasource& datasource,
  std::optional<cudf::ast::operation> filter_expression,
  std::optional<std::vector<std::string>> const& column_names,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  // Create reader options with empty source info
  cudf::io::parquet_reader_options options = cudf::io::parquet_reader_options::builder().build();

  if (column_names.has_value()) { options.set_column_names(column_names.value()); }
  if (filter_expression.has_value()) { options.set_filter(filter_expression.value()); }

  auto const reader = setup_reader(datasource, options);
  auto reader_ref   = std::ref(*reader);

  setup_page_index(datasource, reader_ref);

  auto [filtered_row_group_indices, _ /* row_mask */] =
    apply_hybrid_scan_filters(datasource, reader_ref, options, stream, mr);

  auto current_row_group_indices = cudf::host_span<cudf::size_type>(filtered_row_group_indices);

  // Get all column chunk byte ranges from the reader
  auto const all_column_chunk_byte_ranges =
    reader->all_column_chunks_byte_ranges(current_row_group_indices, options);

  // Fetch column chunk device buffers and spans from the input buffer
  auto [all_col_buffers, all_col_data, all_col_tasks] =
    cudf::io::parquet::fetch_byte_ranges_to_device_async(
      datasource, all_column_chunk_byte_ranges, stream, mr);
  all_col_tasks.get();

  // Materialize the table with all columns
  return reader->materialize_all_columns(
    current_row_group_indices, all_col_data, options, stream, mr);
}
