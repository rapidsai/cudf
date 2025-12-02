/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common_utils.hpp"

#include "timer.hpp"

#include <cudf/ast/expressions.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/join/filtered_join.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/aligned.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/aligned_resource_adaptor.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/owning_wrapper.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <filesystem>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

/**
 * @file common_utils.cpp
 * @brief Definitions for utilities for `hybrid_scan_io` example
 */

std::shared_ptr<rmm::mr::device_memory_resource> create_memory_resource(bool is_pool_used)
{
  if (is_pool_used) {
    return rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(
      std::make_shared<rmm::mr::cuda_memory_resource>(), rmm::percent_of_free_device_memory(80));
  }
  return std::make_shared<rmm::mr::cuda_async_memory_resource>();
}

std::vector<io_source> extract_input_sources(std::string const& paths,
                                             int32_t input_multiplier,
                                             int32_t thread_count,
                                             io_source_type io_source_type,
                                             rmm::cuda_stream_view stream)
{
  // Get the delimited paths to directory and/or files.
  std::vector<std::string> const delimited_paths = [&]() {
    std::vector<std::string> paths_list;
    std::stringstream strstream{paths};
    std::string path;
    // Extract the delimited paths.
    while (std::getline(strstream, path, char{','})) {
      paths_list.push_back(path);
    }
    return paths_list;
  }();

  // List of parquet files
  std::vector<std::string> parquet_files;
  std::for_each(delimited_paths.cbegin(), delimited_paths.cend(), [&](auto const& path_string) {
    auto const path = std::filesystem::path{path_string};
    // If this is a parquet file, add it.
    if (std::filesystem::is_regular_file(path)) {
      parquet_files.push_back(path_string);
    }
    // If this is a directory, add all files in the directory.
    else if (std::filesystem::is_directory(path)) {
      for (auto const& file : std::filesystem::directory_iterator(path)) {
        if (std::filesystem::is_regular_file(file.path())) {
          parquet_files.push_back(file.path().string());
        } else {
          std::cout << "Skipping sub-directory: " << file.path().string() << "\n";
        }
      }
    } else {
      throw std::runtime_error("Encountered an invalid input path\n");
    }
  });

  // Current size of list of parquet files
  auto const initial_size = parquet_files.size();
  if (initial_size == 0) { return {}; }

  // Reserve space
  parquet_files.reserve(std::max<size_t>(thread_count, input_multiplier * parquet_files.size()));

  // Append the input files by input_multiplier times
  std::for_each(thrust::make_counting_iterator(1),
                thrust::make_counting_iterator(input_multiplier),
                [&](auto i) {
                  parquet_files.insert(parquet_files.end(),
                                       parquet_files.begin(),
                                       parquet_files.begin() + initial_size);
                });

  // Cycle append parquet files from the existing ones if less than the thread_count
  std::cout << "Warning: Number of input sources < thread count. Cycling from\n"
               "and appending to current input sources such that the number of\n"
               "input source == thread count\n";
  for (size_t idx = 0; thread_count > static_cast<int>(parquet_files.size()); idx++) {
    parquet_files.emplace_back(parquet_files[idx % initial_size]);
  }

  // Vector of io sources
  std::vector<io_source> input_sources;
  input_sources.reserve(parquet_files.size());
  // Transform input files to the specified io sources
  std::transform(
    parquet_files.begin(),
    parquet_files.end(),
    std::back_inserter(input_sources),
    [&](auto const& file_name) { return io_source{file_name, io_source_type, stream}; });
  stream.synchronize();
  return input_sources;
}

cudf::ast::operation create_filter_expression(std::string const& column_name,
                                              std::string const& literal_value)
{
  auto const column_reference = cudf::ast::column_name_reference(column_name);
  auto scalar                 = cudf::string_scalar(literal_value);
  auto literal                = cudf::ast::literal(scalar);
  return cudf::ast::operation(cudf::ast::ast_operator::EQUAL, column_reference, literal);
}

void check_tables_equal(cudf::table_view const& lhs_table,
                        cudf::table_view const& rhs_table,
                        rmm::cuda_stream_view stream)
{
  try {
    // Left anti-join the original and transcoded tables identical tables should not throw an
    // exception and return an empty indices vector
    cudf::filtered_join join_obj(
      lhs_table, cudf::null_equality::EQUAL, cudf::set_as_build_table::RIGHT, stream);
    auto const indices = join_obj.anti_join(rhs_table, stream);
    // No exception thrown, check indices
    auto const tables_equal = indices->size() == 0;
    if (tables_equal) {
      std::cout << "Tables identical: " << std::boolalpha << tables_equal << "\n\n";
    } else {
      // Helper to write parquet data for inspection
      auto const write_parquet =
        [](cudf::table_view table, std::string filepath, rmm::cuda_stream_view stream) {
          auto sink_info = cudf::io::sink_info(filepath);
          auto opts      = cudf::io::parquet_writer_options::builder(sink_info, table).build();
          cudf::io::write_parquet(opts, stream);
        };
      write_parquet(lhs_table, "lhs_table.parquet", stream);
      write_parquet(rhs_table, "rhs_table.parquet", stream);
      throw std::logic_error("Tables identical: false\n\n");
    }
  } catch (std::exception& e) {
    std::cout << e.what() << std::endl;
  }
}

namespace {

/**
 * @brief Fetches a host span of Parquet footer bytes from the input buffer span
 *
 * @param buffer Input buffer span
 * @return A host span of the footer bytes
 */
cudf::host_span<uint8_t const> fetch_footer_bytes(cudf::host_span<uint8_t const> buffer)
{
  CUDF_FUNC_RANGE();

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

/**
 * @brief Fetches a host span of Parquet PageIndexbytes from the input buffer span
 *
 * @param buffer Input buffer span
 * @param page_index_bytes Byte range of `PageIndex` to fetch
 * @return A host span of the PageIndex bytes
 */
cudf::host_span<uint8_t const> fetch_page_index_bytes(
  cudf::host_span<uint8_t const> buffer, cudf::io::text::byte_range_info const page_index_bytes)
{
  return cudf::host_span<uint8_t const>(
    reinterpret_cast<uint8_t const*>(buffer.data()) + page_index_bytes.offset(),
    page_index_bytes.size());
}

/**
 * @brief Fetches a list of byte ranges from a host buffer into a vector of device buffers
 *
 * @param host_buffer Host buffer span
 * @param byte_ranges Byte ranges to fetch
 * @param stream CUDA stream
 * @param mr Device memory resource to create device buffers with
 *
 * @return Vector of device buffers
 */
std::vector<rmm::device_buffer> fetch_byte_ranges(
  cudf::host_span<uint8_t const> host_buffer,
  cudf::host_span<cudf::io::text::byte_range_info const> byte_ranges,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  std::vector<rmm::device_buffer> buffers(byte_ranges.size());

  std::for_each(thrust::counting_iterator<size_t>(0),
                thrust::counting_iterator(byte_ranges.size()),
                [&](auto const idx) {
                  auto const chunk_offset = host_buffer.data() + byte_ranges[idx].offset();
                  auto const chunk_size   = byte_ranges[idx].size();
                  auto buffer             = rmm::device_buffer(chunk_size, stream, mr);
                  CUDF_CUDA_TRY(cudaMemcpyAsync(
                    buffer.data(), chunk_offset, chunk_size, cudaMemcpyDefault, stream.value()));
                  buffers[idx] = std::move(buffer);
                });

  stream.synchronize_no_throw();
  return buffers;
}

/**
 * @brief Combine columns from filter and payload tables into a single table
 *
 * @param filter_table Filter table
 * @param payload_table Payload table
 * @return Combined table
 */
std::unique_ptr<cudf::table> combine_tables(std::unique_ptr<cudf::table> filter_table,
                                            std::unique_ptr<cudf::table> payload_table)
{
  auto filter_columns  = filter_table->release();
  auto payload_columns = payload_table->release();

  auto all_columns = std::vector<std::unique_ptr<cudf::column>>{};
  all_columns.reserve(filter_columns.size() + payload_columns.size());
  std::move(filter_columns.begin(), filter_columns.end(), std::back_inserter(all_columns));
  std::move(payload_columns.begin(), payload_columns.end(), std::back_inserter(all_columns));
  auto table = std::make_unique<cudf::table>(std::move(all_columns));

  return table;
}

}  // namespace

/**
 * @brief Read parquet file with the next-gen parquet reader
 *
 * @param io_source io source to read
 * @param filter_expression Filter expression
 * @param filters Set of parquet filters to apply
 * @param stream CUDA stream for hybrid scan reader
 * @param mr Device memory resource
 *
 * @return Tuple of filter table, payload table, filter metadata, payload metadata, and the final
 *         row validity column
 */
template <bool print_progress>
std::unique_ptr<cudf::table> hybrid_scan(io_source const& io_source,
                                         cudf::ast::operation const& filter_expression,
                                         std::unordered_set<parquet_filter_type> const& filters,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  auto options = cudf::io::parquet_reader_options::builder().filter(filter_expression).build();

  // Input file buffer span
  auto const file_buffer_span = io_source.get_host_buffer_span();

  if constexpr (print_progress) { std::cout << "\nREADER: Setup, metadata and page index...\n"; }
  timer timer;

  // Fetch footer bytes and setup reader
  auto const footer_buffer = fetch_footer_bytes(file_buffer_span);
  auto const reader =
    std::make_unique<cudf::io::parquet::experimental::hybrid_scan_reader>(footer_buffer, options);

  // Get page index byte range from the reader
  auto const page_index_byte_range = reader->page_index_byte_range();
  auto const page_index_buffer = fetch_page_index_bytes(file_buffer_span, page_index_byte_range);
  reader->setup_page_index(page_index_buffer);

  // Get all row groups from the reader
  auto input_row_group_indices   = reader->all_row_groups(options);
  auto current_row_group_indices = cudf::host_span<cudf::size_type>(input_row_group_indices);
  if constexpr (print_progress) {
    std::cout << "Current row group indices size: " << current_row_group_indices.size() << "\n";
    timer.print_elapsed_millis();
  }

  // Filter row groups with stats
  auto stats_filtered_row_group_indices = std::vector<cudf::size_type>{};
  if (filters.contains(parquet_filter_type::ROW_GROUPS_WITH_STATS)) {
    if constexpr (print_progress) {
      std::cout << "READER: Filter row groups with stats...\n";
      timer.reset();
    }
    stats_filtered_row_group_indices =
      reader->filter_row_groups_with_stats(current_row_group_indices, options, stream);

    // Update current row group indices
    current_row_group_indices = stats_filtered_row_group_indices;
    if constexpr (print_progress) {
      std::cout << "Current row group indices size: " << current_row_group_indices.size() << "\n";
      timer.print_elapsed_millis();
    }
  }

  std::vector<cudf::io::text::byte_range_info> bloom_filter_byte_ranges;
  std::vector<cudf::io::text::byte_range_info> dict_page_byte_ranges;

  // Get bloom filter and dictionary page byte ranges from the reader
  if (filters.contains(parquet_filter_type::ROW_GROUPS_WITH_DICT_PAGES) or
      filters.contains(parquet_filter_type::ROW_GROUPS_WITH_BLOOM_FILTERS)) {
    if constexpr (print_progress) {
      std::cout << "READER: Get bloom filter and dictionary page byte ranges...\n";
      timer.reset();
    }
    std::tie(bloom_filter_byte_ranges, dict_page_byte_ranges) =
      reader->secondary_filters_byte_ranges(current_row_group_indices, options);
    if constexpr (print_progress) { timer.print_elapsed_millis(); }
  }

  // Filter row groups with dictionary pages
  std::vector<cudf::size_type> dictionary_page_filtered_row_group_indices;
  dictionary_page_filtered_row_group_indices.reserve(current_row_group_indices.size());
  if (filters.contains(parquet_filter_type::ROW_GROUPS_WITH_DICT_PAGES) and
      dict_page_byte_ranges.size()) {
    if constexpr (print_progress) {
      std::cout << "READER: Filter row groups with dictionary pages...\n";
      timer.reset();
    }
    // Fetch dictionary page buffers from the input file buffer
    std::vector<rmm::device_buffer> dictionary_page_buffers =
      fetch_byte_ranges(file_buffer_span, dict_page_byte_ranges, stream, mr);
    dictionary_page_filtered_row_group_indices = reader->filter_row_groups_with_dictionary_pages(
      dictionary_page_buffers, current_row_group_indices, options, stream);

    // Update current row group indices
    current_row_group_indices = dictionary_page_filtered_row_group_indices;
    if constexpr (print_progress) {
      std::cout << "Current row group indices size: " << current_row_group_indices.size() << "\n";
      timer.print_elapsed_millis();
    }
  } else {
    if constexpr (print_progress) {
      std::cout << "SKIP: Row group filtering with dictionary pages...\n\n";
    }
  }

  // Filter row groups with bloom filters
  std::vector<cudf::size_type> bloom_filtered_row_group_indices;
  bloom_filtered_row_group_indices.reserve(current_row_group_indices.size());
  if (filters.contains(parquet_filter_type::ROW_GROUPS_WITH_BLOOM_FILTERS) and
      bloom_filter_byte_ranges.size()) {
    // Fetch 32 byte aligned bloom filter data buffers from the input file buffer
    auto constexpr bloom_filter_alignment = rmm::CUDA_ALLOCATION_ALIGNMENT;
    auto aligned_mr = rmm::mr::aligned_resource_adaptor<rmm::mr::device_memory_resource>(
      mr, bloom_filter_alignment);
    if constexpr (print_progress) {
      std::cout << "READER: Filter row groups with bloom filters...\n";
      timer.reset();
    }
    std::vector<rmm::device_buffer> bloom_filter_data =
      fetch_byte_ranges(file_buffer_span, bloom_filter_byte_ranges, stream, aligned_mr);
    // Filter row groups with bloom filters
    bloom_filtered_row_group_indices = reader->filter_row_groups_with_bloom_filters(
      bloom_filter_data, current_row_group_indices, options, stream);

    // Update current row group indices
    current_row_group_indices = bloom_filtered_row_group_indices;
    if constexpr (print_progress) {
      std::cout << "Current row group indices size: " << current_row_group_indices.size() << "\n";
      timer.print_elapsed_millis();
    }
  } else {
    if constexpr (print_progress) {
      std::cout << "SKIP: Row group filtering with bloom filters...\n\n";
    }
  }

  // Check whether to prune filter column data pages
  using cudf::io::parquet::experimental::use_data_page_mask;
  auto const prune_filter_data_pages =
    filters.contains(parquet_filter_type::FILTER_COLUMN_PAGES_WITH_PAGE_INDEX);

  auto row_mask = std::unique_ptr<cudf::column>{};
  if (prune_filter_data_pages) {
    if constexpr (print_progress) {
      std::cout << "READER: Filter data pages of filter columns with page index stats...\n";
      timer.reset();
    }
    // Filter data pages with page index stats
    row_mask =
      reader->build_row_mask_with_page_index_stats(current_row_group_indices, options, stream, mr);
    if constexpr (print_progress) { timer.print_elapsed_millis(); }
  } else {
    if constexpr (print_progress) {
      std::cout << "SKIP: Filter column data page filtering with page index stats...\n\n";
    }
    auto num_rows = reader->total_rows_in_row_groups(current_row_group_indices);
    row_mask      = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::BOOL8}, num_rows, rmm::device_buffer{}, 0, stream, mr);
  }

  if constexpr (print_progress) {
    std::cout << "READER: Materialize filter columns...\n";
    timer.reset();
  }
  // Get column chunk byte ranges from the reader
  auto const filter_column_chunk_byte_ranges =
    reader->filter_column_chunks_byte_ranges(current_row_group_indices, options);
  auto filter_column_chunk_buffers =
    fetch_byte_ranges(file_buffer_span, filter_column_chunk_byte_ranges, stream, mr);

  // Materialize the table with only the filter columns
  auto row_mask_mutable_view = row_mask->mutable_view();
  auto filter_table =
    reader
      ->materialize_filter_columns(
        current_row_group_indices,
        std::move(filter_column_chunk_buffers),
        row_mask_mutable_view,
        prune_filter_data_pages ? use_data_page_mask::YES : use_data_page_mask::NO,
        options,
        stream)
      .tbl;
  if constexpr (print_progress) { timer.print_elapsed_millis(); }

  // Check whether to prune payload column data pages
  auto const prune_payload_data_pages =
    filters.contains(parquet_filter_type::PAYLOAD_COLUMN_PAGES_WITH_ROW_MASK);

  if constexpr (print_progress) {
    if (prune_payload_data_pages) {
      std::cout << "READER: Filter data pages of payload columns with row mask...\n";
    } else {
      std::cout << "SKIP: Payload column data page filtering with row mask...\n\n";
    }

    std::cout << "READER: Materialize payload columns...\n";
    timer.reset();
  }
  // Get column chunk byte ranges from the reader
  auto const payload_column_chunk_byte_ranges =
    reader->payload_column_chunks_byte_ranges(current_row_group_indices, options);
  auto payload_column_chunk_buffers =
    fetch_byte_ranges(file_buffer_span, payload_column_chunk_byte_ranges, stream, mr);

  // Materialize the table with only the payload columns
  auto payload_table =
    reader
      ->materialize_payload_columns(
        current_row_group_indices,
        std::move(payload_column_chunk_buffers),
        row_mask->view(),
        prune_payload_data_pages ? use_data_page_mask::YES : use_data_page_mask::NO,
        options,
        stream)
      .tbl;
  if constexpr (print_progress) { timer.print_elapsed_millis(); }

  return combine_tables(std::move(filter_table), std::move(payload_table));
}

// Explicit template instantiations
template std::unique_ptr<cudf::table> hybrid_scan<true>(
  io_source const& io_source,
  cudf::ast::operation const& filter_expression,
  std::unordered_set<parquet_filter_type> const& filters,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

template std::unique_ptr<cudf::table> hybrid_scan<false>(
  io_source const& io_source,
  cudf::ast::operation const& filter_expression,
  std::unordered_set<parquet_filter_type> const& filters,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);
