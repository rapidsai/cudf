/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hybrid_scan_commons.hpp"

#include "io_utils.hpp"
#include "timer.hpp"

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/io/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/mr/aligned_resource_adaptor.hpp>

#include <unordered_set>

/**
 * @file hybrid_scan_commons.cpp
 * @brief Definitions for common hybrid scan related functions for hybrid_scan examples
 */

namespace {

/**
 * @brief Combine columns from filter and payload tables into a single table
 *
 * @param filter_table Filter table
 * @param payload_table Payload table
 * @param verbose Whether to print verbose output
 *
 * @return Combined table
 */
std::unique_ptr<cudf::table> combine_tables(std::unique_ptr<cudf::table> filter_table,
                                            std::unique_ptr<cudf::table> payload_table,
                                            bool verbose)
{
  if (verbose) { std::cout << "READER: Combine tables...\n"; }

  timer timer;
  auto filter_columns  = filter_table->release();
  auto payload_columns = payload_table->release();

  auto all_columns = std::vector<std::unique_ptr<cudf::column>>{};
  all_columns.reserve(filter_columns.size() + payload_columns.size());
  std::move(filter_columns.begin(), filter_columns.end(), std::back_inserter(all_columns));
  std::move(payload_columns.begin(), payload_columns.end(), std::back_inserter(all_columns));
  auto table = std::make_unique<cudf::table>(std::move(all_columns));

  if (verbose) { timer.print_elapsed_millis(); }

  return std::move(table);
}

}  // namespace

namespace detail {

std::unique_ptr<hybrid_scan_reader> setup_reader(cudf::io::datasource& datasource,
                                                 cudf::io::parquet_reader_options const& options,
                                                 bool use_page_index,
                                                 bool verbose)
{
  if (verbose) { std::cout << "READER: Setup, metadata and page index...\n"; }

  timer timer;
  // Fetch footer bytes and setup reader
  auto const footer_buffer = fetch_footer_bytes(datasource);
  auto reader              = std::make_unique<cudf::io::parquet::experimental::hybrid_scan_reader>(
    make_host_span(*footer_buffer), options);

  if (use_page_index) {
    auto const page_index_byte_range = reader->page_index_byte_range();
    if (page_index_byte_range.is_empty()) {
      throw std::runtime_error("Page index is not present in the input parquet file");
    }
    auto const page_index_buffer = fetch_page_index_bytes(datasource, page_index_byte_range);
    reader->setup_page_index(make_host_span(*page_index_buffer));
  }
  if (verbose) { timer.print_elapsed_millis(); }

  return std::move(reader);
}

std::vector<cudf::size_type> apply_row_group_filters(
  cudf::io::datasource& datasource,
  cudf::io::parquet::experimental::hybrid_scan_reader const& reader,
  std::unordered_set<hybrid_scan_filter_type> const& filters,
  cudf::host_span<cudf::size_type> input_row_group_indices,
  cudf::io::parquet_reader_options const& options,
  bool verbose,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  // Span to track current row group indices
  auto current_row_group_indices = cudf::host_span<cudf::size_type>(input_row_group_indices);

  if (verbose) {
    std::cout << "Input row group indices size: " << current_row_group_indices.size() << "\n";
  }

  timer timer;

  // Filter row groups with stats
  auto stats_filtered_row_group_indices = std::vector<cudf::size_type>{};
  if (filters.contains(hybrid_scan_filter_type::ROW_GROUPS_WITH_STATS)) {
    if (verbose) { std::cout << "READER: Filter row groups with stats...\n"; }

    timer.reset();
    stats_filtered_row_group_indices =
      reader.filter_row_groups_with_stats(current_row_group_indices, options, stream);

    current_row_group_indices = stats_filtered_row_group_indices;
    if (verbose) {
      std::cout << "Current row group indices size: " << current_row_group_indices.size() << "\n";
      timer.print_elapsed_millis();
    }
  }

  // Get bloom filter and dictionary page byte ranges from the reader
  std::vector<cudf::io::text::byte_range_info> bloom_filter_byte_ranges;
  std::vector<cudf::io::text::byte_range_info> dict_page_byte_ranges;
  if (filters.contains(hybrid_scan_filter_type::ROW_GROUPS_WITH_DICT_PAGES) or
      filters.contains(hybrid_scan_filter_type::ROW_GROUPS_WITH_BLOOM_FILTERS)) {
    if (verbose) { std::cout << "READER: Get bloom filter and dictionary page byte ranges...\n"; }
    timer.reset();
    std::tie(bloom_filter_byte_ranges, dict_page_byte_ranges) =
      reader.secondary_filters_byte_ranges(current_row_group_indices, options);
    if (verbose) { timer.print_elapsed_millis(); }
  }

  // Filter row groups with dictionary pages
  std::vector<cudf::size_type> dictionary_page_filtered_row_group_indices;
  dictionary_page_filtered_row_group_indices.reserve(current_row_group_indices.size());
  if (filters.contains(hybrid_scan_filter_type::ROW_GROUPS_WITH_DICT_PAGES) and
      dict_page_byte_ranges.size()) {
    if (verbose) { std::cout << "READER: Filter row groups with dictionary pages...\n"; }
    timer.reset();
    // Fetch dictionary page buffers and corresponding device spans from the input file buffer
    auto [dictionary_page_buffers, dictionary_page_data, dict_read_tasks] =
      fetch_byte_ranges(datasource, dict_page_byte_ranges, stream, mr);
    dict_read_tasks.get();

    dictionary_page_filtered_row_group_indices = reader.filter_row_groups_with_dictionary_pages(
      dictionary_page_data, current_row_group_indices, options, stream);

    current_row_group_indices = dictionary_page_filtered_row_group_indices;
    if (verbose) {
      std::cout << "Current row group indices size: " << current_row_group_indices.size() << "\n";
      timer.print_elapsed_millis();
    }
  } else if (verbose) {
    std::cout << "SKIP: Row group filtering with dictionary pages...\n\n";
  }

  // Filter row groups with bloom filters
  std::vector<cudf::size_type> bloom_filtered_row_group_indices;
  bloom_filtered_row_group_indices.reserve(current_row_group_indices.size());
  if (filters.contains(hybrid_scan_filter_type::ROW_GROUPS_WITH_BLOOM_FILTERS) and
      bloom_filter_byte_ranges.size()) {
    // Fetch 32-byte aligned bloom filter data buffers from the input file buffer
    auto constexpr bloom_filter_alignment = rmm::CUDA_ALLOCATION_ALIGNMENT;
    auto aligned_mr = rmm::mr::aligned_resource_adaptor<rmm::mr::device_memory_resource>(
      mr, bloom_filter_alignment);
    if (verbose) { std::cout << "READER: Filter row groups with bloom filters...\n"; }
    timer.reset();
    auto [bloom_filter_buffers, bloom_filter_data, bloom_read_tasks] =
      fetch_byte_ranges(datasource, bloom_filter_byte_ranges, stream, aligned_mr);
    bloom_read_tasks.get();

    bloom_filtered_row_group_indices = reader.filter_row_groups_with_bloom_filters(
      bloom_filter_data, current_row_group_indices, options, stream);

    current_row_group_indices = bloom_filtered_row_group_indices;
    if (verbose) {
      std::cout << "Current row group indices size: " << current_row_group_indices.size() << "\n";
      timer.print_elapsed_millis();
    }
  } else if (verbose) {
    std::cout << "SKIP: Row group filtering with bloom filters...\n\n";
  }

  if (verbose) {
    std::cout << "Filtered row group indices size: " << current_row_group_indices.size() << "\n";
    timer.print_elapsed_millis();
  }

  return std::vector<cudf::size_type>(current_row_group_indices.begin(),
                                      current_row_group_indices.end());
}

std::unique_ptr<cudf::table> single_step_materialize(
  cudf::io::datasource& datasource,
  cudf::io::parquet::experimental::hybrid_scan_reader const& reader,
  cudf::host_span<cudf::size_type> current_row_group_indices,
  cudf::io::parquet_reader_options const& options,
  bool verbose,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  if (verbose) { std::cout << "READER: Single step materialize...\n"; }

  timer timer;

  auto const all_column_chunk_byte_ranges =
    reader.all_column_chunks_byte_ranges(current_row_group_indices, options);
  auto [all_column_chunk_buffers, all_column_chunk_data, all_column_chunk_read_tasks] =
    fetch_byte_ranges(datasource, all_column_chunk_byte_ranges, stream, mr);
  all_column_chunk_read_tasks.get();

  auto read_table =
    reader
      .materialize_all_columns(current_row_group_indices, all_column_chunk_data, options, stream)
      .tbl;

  if (verbose) { timer.print_elapsed_millis(); }

  return std::move(read_table);
}

std::unique_ptr<cudf::table> two_step_materialize(
  cudf::io::datasource& datasource,
  cudf::io::parquet::experimental::hybrid_scan_reader const& reader,
  std::unordered_set<hybrid_scan_filter_type> const& filters,
  cudf::host_span<cudf::size_type> current_row_group_indices,
  cudf::io::parquet_reader_options const& options,
  bool verbose,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  // Check whether to prune filter column data pages
  using cudf::io::parquet::experimental::use_data_page_mask;
  auto const prune_filter_data_pages =
    filters.contains(hybrid_scan_filter_type::FILTER_COLUMN_PAGES_WITH_PAGE_INDEX);

  if (verbose) {
    if (prune_filter_data_pages) {
      std::cout << "READER: Filter data pages of filter columns with page index stats...\n";
    } else {
      std::cout << "SKIP: Filter column data page filtering with page index stats...\n\n";
    }
  }

  timer timer;

  // Build initial row mask
  auto row_mask = [&]() {
    if (prune_filter_data_pages) {
      return reader.build_row_mask_with_page_index_stats(
        current_row_group_indices, options, stream, mr);
    } else {
      return reader.build_all_true_row_mask(current_row_group_indices, stream, mr);
    }
  }();
  if (verbose) {
    timer.print_elapsed_millis();
    std::cout << "READER: Materialize filter columns...\n";
  }

  timer.reset();

  // Get column chunk byte ranges from the reader
  auto const filter_column_chunk_byte_ranges =
    reader.filter_column_chunks_byte_ranges(current_row_group_indices, options);
  auto [filter_column_chunk_buffers, filter_column_chunk_data, filter_col_read_tasks] =
    fetch_byte_ranges(datasource, filter_column_chunk_byte_ranges, stream, mr);
  filter_col_read_tasks.get();

  auto row_mask_mutable_view = row_mask->mutable_view();
  auto filter_table =
    reader
      .materialize_filter_columns(
        current_row_group_indices,
        filter_column_chunk_data,
        row_mask_mutable_view,
        prune_filter_data_pages ? use_data_page_mask::YES : use_data_page_mask::NO,
        options,
        stream)
      .tbl;
  if (verbose) { timer.print_elapsed_millis(); }

  // Check whether to prune payload column data pages
  auto const prune_payload_data_pages =
    filters.contains(hybrid_scan_filter_type::PAYLOAD_COLUMN_PAGES_WITH_ROW_MASK);

  if (verbose) {
    std::cout << "READER: Materialize payload columns...\n";
    if (prune_payload_data_pages) {
      std::cout << "READER: Filter data pages of payload columns with row mask...\n";
    } else {
      std::cout << "SKIP: Payload column data page filtering with row mask...\n\n";
    }
  }

  timer.reset();

  // Get column chunk byte ranges from the reader
  auto const payload_column_chunk_byte_ranges =
    reader.payload_column_chunks_byte_ranges(current_row_group_indices, options);
  auto [payload_column_chunk_buffers, payload_column_chunk_data, payload_col_read_tasks] =
    fetch_byte_ranges(datasource, payload_column_chunk_byte_ranges, stream, mr);
  payload_col_read_tasks.get();

  auto payload_table =
    reader
      .materialize_payload_columns(
        current_row_group_indices,
        payload_column_chunk_data,
        row_mask->view(),
        prune_payload_data_pages ? use_data_page_mask::YES : use_data_page_mask::NO,
        options,
        stream)
      .tbl;

  if (verbose) { timer.print_elapsed_millis(); }

  return combine_tables(std::move(filter_table), std::move(payload_table), verbose);
}

}  // namespace detail