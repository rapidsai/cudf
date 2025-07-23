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

#include "../utilities/timer.hpp"
#include "common_utils.hpp"
#include "io_source.hpp"

#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/mr/device/aligned_resource_adaptor.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/statistics_resource_adaptor.hpp>

#include <thrust/host_vector.h>

#include <filesystem>
#include <functional>
#include <string>

/**
 * @file hybrid_scan.cpp
 *
 * @brief This example demonstrates the use of libcudf next-gen parquet reader to optimally read
 * a parquet file subject to a highly selective string-type point lookup filter. The same file is
 * also read using the libcudf legacy parquet reader and the read times are compared.
 */

namespace {

/**
 * @brief Fetches a host span of Parquet footer bytes from the input buffer span
 *
 * @param buffer Input buffer span
 * @return A host span of the footer bytes
 */
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

/**
 * @brief Read parquet file with the next-gen parquet reader
 *
 * @param buffer Buffer containing the parquet file
 * @param filter_expression Filter expression
 * @param num_filter_columns Number of filter columns
 * @param payload_column_names List of paths of select payload column names, if any
 * @param stream CUDA stream for hybrid scan reader
 * @param mr Device memory resource
 *
 * @return Tuple of filter table, payload table, filter metadata, payload metadata, and the final
 *         row validity column
 */

/**
 * @brief Read parquet input using the next-gen parquet reader from io source
 *
 * @param io_source io source to read
 * @return cudf::io::table_with_metadata
 */
auto read_parquet_with_next_gen_reader(io_source const& io_source,
                                       cudf::ast::operation const& filter_expression,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  auto options = cudf::io::parquet_reader_options::builder().filter(filter_expression).build();

  // Input file buffer span
  auto const file_buffer_span = io_source.get_buffer_span();

  std::cout << "Fetch footer bytes...\n";
  cudf::examples::timer timer;
  // Fetch footer and page index bytes from the buffer
  auto const footer_buffer = fetch_footer_bytes(file_buffer_span);
  timer.print_elapsed_millis();

  std::cout << "Create hybrid scan reader with footer bytes...\n";
  timer.reset();
  // Create hybrid scan reader with footer bytes
  auto const reader =
    std::make_unique<cudf::io::parquet::experimental::hybrid_scan_reader>(footer_buffer, options);
  timer.print_elapsed_millis();

  std::cout << "Get page index byte range from the reader...\n";
  timer.reset();
  // Get page index byte range from the reader
  auto const page_index_byte_range = reader->page_index_byte_range();
  timer.print_elapsed_millis();

  std::cout << "Fetch page index bytes...\n";
  timer.reset();
  // Fetch page index bytes from the input buffer
  auto const page_index_buffer = fetch_page_index_bytes(file_buffer_span, page_index_byte_range);
  timer.print_elapsed_millis();

  std::cout << "Setup page index...\n";
  timer.reset();
  // Setup page index
  reader->setup_page_index(page_index_buffer);
  timer.print_elapsed_millis();

  // Get all row groups from the reader
  auto input_row_group_indices = reader->all_row_groups(options);

  // Span to track current row group indices
  auto current_row_group_indices = cudf::host_span<cudf::size_type>(input_row_group_indices);

  std::cout << "Filter row groups with stats...\n";
  timer.reset();
  // Filter row groups with stats
  auto stats_filtered_row_group_indices =
    reader->filter_row_groups_with_stats(current_row_group_indices, options, stream);
  timer.print_elapsed_millis();

  // Update current row group indices
  current_row_group_indices = stats_filtered_row_group_indices;

  std::cout << "Get bloom filter and dictionary page byte ranges from the reader...\n";
  timer.reset();
  // Get bloom filter and dictionary page byte ranges from the reader
  auto [bloom_filter_byte_ranges, dict_page_byte_ranges] =
    reader->secondary_filters_byte_ranges(current_row_group_indices, options);
  timer.print_elapsed_millis();

  // If we have dictionary page byte ranges, filter row groups with dictionary pages
  std::vector<cudf::size_type> dictionary_page_filtered_row_group_indices;
  dictionary_page_filtered_row_group_indices.reserve(current_row_group_indices.size());
  if (dict_page_byte_ranges.size()) {
    std::cout << "Fetch dictionary page buffers...\n";
    timer.reset();
    // Fetch dictionary page buffers from the input file buffer
    std::vector<rmm::device_buffer> dictionary_page_buffers =
      fetch_byte_ranges(file_buffer_span, dict_page_byte_ranges, stream, mr);
    timer.print_elapsed_millis();

    std::cout << "Filter row groups with dictionary pages...\n";
    timer.reset();
    dictionary_page_filtered_row_group_indices = reader->filter_row_groups_with_dictionary_pages(
      dictionary_page_buffers, current_row_group_indices, options, stream);
    timer.print_elapsed_millis();

    // Update current row group indices
    current_row_group_indices = dictionary_page_filtered_row_group_indices;
  } else {
    std::cout << "No dictionary page byte ranges found...\n";
  }

  // If we have bloom filter byte ranges, filter row groups with bloom filters
  std::vector<cudf::size_type> bloom_filtered_row_group_indices;
  bloom_filtered_row_group_indices.reserve(current_row_group_indices.size());
  if (bloom_filter_byte_ranges.size()) {
    // Fetch 32 byte aligned bloom filter data buffers from the input file buffer
    auto constexpr bloom_filter_alignment = 32;
    auto aligned_mr = rmm::mr::aligned_resource_adaptor<rmm::mr::device_memory_resource>(
      mr, bloom_filter_alignment);

    std::cout << "Fetch bloom filter data buffers...\n";
    timer.reset();
    std::vector<rmm::device_buffer> bloom_filter_data =
      fetch_byte_ranges(file_buffer_span, bloom_filter_byte_ranges, stream, aligned_mr);
    timer.print_elapsed_millis();

    std::cout << "Filter row groups with bloom filters...\n";
    timer.reset();
    // Filter row groups with bloom filters
    bloom_filtered_row_group_indices = reader->filter_row_groups_with_bloom_filters(
      bloom_filter_data, current_row_group_indices, options, stream);
    timer.print_elapsed_millis();

    // Update current row group indices
    current_row_group_indices = bloom_filtered_row_group_indices;
  } else {
    std::cout << "No bloom filter byte ranges found...\n";
  }

  std::cout << "Filter data pages with page index stats...\n";
  timer.reset();
  // Filter data pages with page index stats
  auto [row_mask, data_page_mask] =
    reader->filter_data_pages_with_stats(current_row_group_indices, options, stream, mr);
  timer.print_elapsed_millis();

  std::cout << "Get filter column chunk byte ranges from the reader...\n";
  timer.reset();
  // Get column chunk byte ranges from the reader
  auto const filter_column_chunk_byte_ranges =
    reader->filter_column_chunks_byte_ranges(current_row_group_indices, options);
  timer.print_elapsed_millis();

  std::cout << "Fetch filter column chunk device buffers...\n";
  timer.reset();
  // Fetch column chunk device buffers from the input buffer
  auto filter_column_chunk_buffers =
    fetch_byte_ranges(file_buffer_span, filter_column_chunk_byte_ranges, stream, mr);
  timer.print_elapsed_millis();

  std::cout << "Materialize filter columns...\n";
  timer.reset();
  // Materialize the table with only the filter columns
  auto filter_table = reader
                        ->materialize_filter_columns(data_page_mask,
                                                     current_row_group_indices,
                                                     std::move(filter_column_chunk_buffers),
                                                     row_mask->mutable_view(),
                                                     options,
                                                     stream)
                        .tbl;
  timer.print_elapsed_millis();

  std::cout << "Get payload column chunk byte ranges from the reader...\n";
  timer.reset();
  // Get column chunk byte ranges from the reader
  auto const payload_column_chunk_byte_ranges =
    reader->payload_column_chunks_byte_ranges(current_row_group_indices, options);
  timer.print_elapsed_millis();

  std::cout << "Fetch payload column chunk device buffers...\n";
  timer.reset();
  // Fetch column chunk device buffers from the input buffer
  auto payload_column_chunk_buffers =
    fetch_byte_ranges(file_buffer_span, payload_column_chunk_byte_ranges, stream, mr);
  timer.print_elapsed_millis();

  std::cout << "Materialize payload columns...\n";
  timer.reset();
  // Materialize the table with only the payload columns
  auto payload_table = reader
                         ->materialize_payload_columns(current_row_group_indices,
                                                       std::move(payload_column_chunk_buffers),
                                                       row_mask->view(),
                                                       options,
                                                       stream)
                         .tbl;
  timer.print_elapsed_millis();

  std::cout << "Combine filter and payload tables...\n";
  timer.reset();
  auto table = combine_tables(std::move(filter_table), std::move(payload_table));
  timer.print_elapsed_millis();

  return std::make_tuple(std::move(table), std::move(row_mask));
}

/**
 * @brief Read parquet input using the legacy parquet reader from io source
 *
 * @param io_source io source to read
 * @return cudf::io::table_with_metadata
 */
cudf::io::table_with_metadata read_parquet_with_legacy_reader(
  io_source const& io_source,
  cudf::ast::operation const& filter_expression,
  rmm::cuda_stream_view stream)
{
  auto source_info = io_source.get_source_info();
  auto options =
    cudf::io::parquet_reader_options::builder(source_info).filter(filter_expression).build();
  return cudf::io::read_parquet(options);
}

/**
 * @brief Write parquet output to file
 *
 * @param input table to write
 * @param metadata metadata of input table read by parquet reader
 * @param filepath path to output parquet file
 */
void write_parquet(cudf::table_view input, cudf::io::table_metadata metadata, std::string filepath)
{
  // Write the data for inspection
  auto sink_info      = cudf::io::sink_info(filepath);
  auto table_metadata = cudf::io::table_input_metadata{metadata};
  auto options        = cudf::io::parquet_writer_options::builder(sink_info, input)
                   .metadata(table_metadata)
                   .compression(cudf::io::compression_type::AUTO)
                   .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN)
                   .build();

  // write parquet data
  cudf::io::write_parquet(options);
}

/**
 * @brief Function to print example usage and argument information.
 */
void inline print_usage()
{
  std::cout << std::endl
            << "Usage: hybrid_scan <input parquet file> <column name> <literal> <output parquet "
               "file> <io source type>\n"
            << "Note: Both the column name and literal must be of `string` type\n"
            << "Available IO source types: FILEPATH, HOST_BUFFER, PINNED_BUFFER (Default), "
               "DEVICE_BUFFER\n\n"
            << "Example: hybrid_scan example.parquet col_a 100 output.parquet PINNED_BUFFER\n\n";
}

}  // namespace

/**
 * @brief Main for hybrid scan example
 *
 * Command line parameters:
 * 1. parquet input file name/path (default: "example.parquet")
 * 2. parquet output file name/path (default: "output.parquet")
 * 3. column name for filter expression (default: "col_a")
 * 4. literal for filter expression (default: "100")
 * 5. io source type (default: "PINNED_BUFFER")
 *
 * Example invocation from directory `cudf/cpp/examples/hybrid_scan`:
 * ./build/hybrid_scan example.parquet output.parquet col_a 100 output.parquet PINNED_BUFFER
 *
 */
int main(int argc, char const** argv)
{
  auto input_filepath  = std::string{"example.parquet"};
  auto output_filepath = std::string{"output.parquet"};
  auto column_name     = std::string{"col_a"};
  auto literal         = std::string{"100"};
  auto io_source_type  = io_source_type::PINNED_BUFFER;

  switch (argc) {
    case 6: io_source_type = get_io_source_type(argv[5]); [[fallthrough]];
    case 5: literal = argv[4]; [[fallthrough]];
    case 4: column_name = argv[3]; [[fallthrough]];
    case 3: output_filepath = argv[2]; [[fallthrough]];
    case 2:  // Check if instead of input_paths, the first argument is `-h` or `--help`
      if (auto arg = std::string{argv[1]}; arg != "-h" and arg != "--help") {
        input_filepath = std::move(arg);
        break;
      }
      [[fallthrough]];
    default: print_usage(); throw std::invalid_argument("Invalid arguments");
  }

  // Initialize mr, default stream and stream pool
  auto constexpr is_pool_used = true;
  auto stream                 = cudf::get_default_stream();
  auto resource               = create_memory_resource(is_pool_used);
  auto stats_mr =
    rmm::mr::statistics_resource_adaptor<rmm::mr::device_memory_resource>(resource.get());
  rmm::mr::set_current_device_resource(&stats_mr);

  auto const data_source = [&]() -> io_source {
    if (std::filesystem::is_regular_file(input_filepath)) {
      return io_source{input_filepath, io_source_type, stream};
    } else {
      throw std::runtime_error("Input file does not exist");
    }
  }();

  // Create filter expression
  auto const filter_expression = create_filter_expression(column_name, literal);

  // Read the parquet file written with encoding and compression
  std::cout << "Reading " << input_filepath << " with next-gen parquet reader...\n";

  cudf::examples::timer timer;
  auto [table_next_gen_reader, row_mask] =
    read_parquet_with_next_gen_reader(data_source, filter_expression, stream, stats_mr);
  timer.print_elapsed_millis();

  // Read the parquet file written with encoding and compression
  std::cout << "Reading " << input_filepath << " with next-gen parquet reader...\n";
  // Reset the timer
  timer.reset();
  auto [table_legacy_reader, metadata] =
    read_parquet_with_legacy_reader(data_source, filter_expression, stream);
  timer.print_elapsed_millis();

  // Check for validity
  check_tables_equal(table_next_gen_reader->view(), table_legacy_reader->view());

  // Write the output parquet file
  std::cout << "Writing " << output_filepath << " with next-gen parquet reader...\n";
  timer.reset();
  write_parquet(table_next_gen_reader->view(), metadata, output_filepath);
  timer.print_elapsed_millis();

  return 0;
}
