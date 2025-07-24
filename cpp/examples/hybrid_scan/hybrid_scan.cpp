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

#include "../utilities/table_utils.hpp"
#include "../utilities/timer.hpp"
#include "common_utils.hpp"
#include "io_source.hpp"

#include <cudf/column/column_factories.hpp>
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
#include <string>
#include <unordered_set>

/**
 * @file hybrid_scan.cpp
 *
 * @brief This example demonstrates the use of libcudf next-gen parquet reader to optimally read
 * a parquet file subject to a highly selective string-type point lookup filter. The same file is
 * also read using the libcudf legacy parquet reader and the read times are compared.
 */

namespace {

/**
 * @brief Enum to represent the available parquet filters
 */
enum class parquet_filter_type : uint8_t {
  ROW_GROUPS_WITH_STATS         = 0,
  ROW_GROUPS_WITH_DICT_PAGES    = 1,
  ROW_GROUPS_WITH_BLOOM_FILTERS = 2,
  DATA_PAGES_WITH_PAGE_INDEX    = 3
};

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
auto hybrid_scan(io_source const& io_source,
                 cudf::ast::operation const& filter_expression,
                 std::unordered_set<parquet_filter_type> const& filters,
                 rmm::cuda_stream_view stream,
                 rmm::device_async_resource_ref mr)
{
  auto options = cudf::io::parquet_reader_options::builder().filter(filter_expression).build();

  // Input file buffer span
  auto const file_buffer_span = io_source.get_buffer_span();

  std::cout << "\nREADER: Setup, metadata and page index...\n";
  cudf::examples::timer timer;

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
  std::cout << "Current row group indices size: " << current_row_group_indices.size() << "\n";

  timer.print_elapsed_millis();

  // Filter row groups with stats
  auto stats_filtered_row_group_indices = std::vector<cudf::size_type>{};
  if (filters.contains(parquet_filter_type::ROW_GROUPS_WITH_STATS)) {
    std::cout << "READER: Filter row groups with stats...\n";
    timer.reset();
    stats_filtered_row_group_indices =
      reader->filter_row_groups_with_stats(current_row_group_indices, options, stream);

    // Update current row group indices
    current_row_group_indices = stats_filtered_row_group_indices;
    std::cout << "Current row group indices size: " << current_row_group_indices.size() << "\n";
    timer.print_elapsed_millis();
  }

  std::vector<cudf::io::text::byte_range_info> bloom_filter_byte_ranges;
  std::vector<cudf::io::text::byte_range_info> dict_page_byte_ranges;

  // Get bloom filter and dictionary page byte ranges from the reader
  if (filters.contains(parquet_filter_type::ROW_GROUPS_WITH_DICT_PAGES) or
      filters.contains(parquet_filter_type::ROW_GROUPS_WITH_BLOOM_FILTERS)) {
    std::cout << "READER: Get bloom filter and dictionary page byte ranges...\n";
    timer.reset();
    std::tie(bloom_filter_byte_ranges, dict_page_byte_ranges) =
      reader->secondary_filters_byte_ranges(current_row_group_indices, options);
    timer.print_elapsed_millis();
  }

  // Filter row groups with dictionary pages
  std::vector<cudf::size_type> dictionary_page_filtered_row_group_indices;
  dictionary_page_filtered_row_group_indices.reserve(current_row_group_indices.size());
  if (filters.contains(parquet_filter_type::ROW_GROUPS_WITH_DICT_PAGES) and
      dict_page_byte_ranges.size()) {
    std::cout << "READER: Filter row groups with dictionary pages...\n";
    timer.reset();
    // Fetch dictionary page buffers from the input file buffer
    std::vector<rmm::device_buffer> dictionary_page_buffers =
      fetch_byte_ranges(file_buffer_span, dict_page_byte_ranges, stream, mr);
    dictionary_page_filtered_row_group_indices = reader->filter_row_groups_with_dictionary_pages(
      dictionary_page_buffers, current_row_group_indices, options, stream);

    // Update current row group indices
    current_row_group_indices = dictionary_page_filtered_row_group_indices;
    std::cout << "Current row group indices size: " << current_row_group_indices.size() << "\n";
    timer.print_elapsed_millis();
  } else {
    std::cout << "SKIP: Row group filtering with dictionary pages...\n\n";
  }

  // Filter row groups with bloom filters
  std::vector<cudf::size_type> bloom_filtered_row_group_indices;
  bloom_filtered_row_group_indices.reserve(current_row_group_indices.size());
  if (filters.contains(parquet_filter_type::ROW_GROUPS_WITH_BLOOM_FILTERS) and
      bloom_filter_byte_ranges.size()) {
    // Fetch 32 byte aligned bloom filter data buffers from the input file buffer
    auto constexpr bloom_filter_alignment = 32;
    auto aligned_mr = rmm::mr::aligned_resource_adaptor<rmm::mr::device_memory_resource>(
      mr, bloom_filter_alignment);
    std::cout << "READER: Filter row groups with bloom filters...\n";
    timer.reset();
    std::vector<rmm::device_buffer> bloom_filter_data =
      fetch_byte_ranges(file_buffer_span, bloom_filter_byte_ranges, stream, aligned_mr);
    // Filter row groups with bloom filters
    bloom_filtered_row_group_indices = reader->filter_row_groups_with_bloom_filters(
      bloom_filter_data, current_row_group_indices, options, stream);

    // Update current row group indices
    current_row_group_indices = bloom_filtered_row_group_indices;
    std::cout << "Current row group indices size: " << current_row_group_indices.size() << "\n";
    timer.print_elapsed_millis();
  } else {
    std::cout << "SKIP: Row group filtering with bloom filters...\n\n";
  }

  auto row_mask       = std::unique_ptr<cudf::column>{};
  auto data_page_mask = std::vector<std::vector<bool>>{};

  if (filters.contains(parquet_filter_type::DATA_PAGES_WITH_PAGE_INDEX)) {
    std::cout << "READER: Filter data pages with page index stats...\n";
    timer.reset();
    // Filter data pages with page index stats
    std::tie(row_mask, data_page_mask) =
      reader->filter_data_pages_with_stats(current_row_group_indices, options, stream, mr);
    timer.print_elapsed_millis();
  } else {
    std::cout << "SKIP: Row group filtering with data pages...\n\n";
    auto num_rows = reader->total_rows_in_row_groups(current_row_group_indices);
    row_mask      = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::BOOL8}, num_rows, rmm::device_buffer{}, 0, stream, mr);
  }

  std::cout << "READER: Materialize filter columns...\n";
  timer.reset();
  // Get column chunk byte ranges from the reader
  auto const filter_column_chunk_byte_ranges =
    reader->filter_column_chunks_byte_ranges(current_row_group_indices, options);
  auto filter_column_chunk_buffers =
    fetch_byte_ranges(file_buffer_span, filter_column_chunk_byte_ranges, stream, mr);

  // Materialize the table with only the filter columns
  auto filter_table = reader
                        ->materialize_filter_columns({},
                                                     current_row_group_indices,
                                                     std::move(filter_column_chunk_buffers),
                                                     row_mask->mutable_view(),
                                                     options,
                                                     stream)
                        .tbl;
  timer.print_elapsed_millis();

  std::cout << "READER: Materialize payload columns...\n";
  timer.reset();
  // Get column chunk byte ranges from the reader
  auto const payload_column_chunk_byte_ranges =
    reader->payload_column_chunks_byte_ranges(current_row_group_indices, options);
  auto payload_column_chunk_buffers =
    fetch_byte_ranges(file_buffer_span, payload_column_chunk_byte_ranges, stream, mr);

  // Materialize the table with only the payload columns
  auto payload_table = reader
                         ->materialize_payload_columns(current_row_group_indices,
                                                       std::move(payload_column_chunk_buffers),
                                                       row_mask->view(),
                                                       options,
                                                       stream)
                         .tbl;
  timer.print_elapsed_millis();

  return std::make_tuple(combine_tables(std::move(filter_table), std::move(payload_table)),
                         std::move(row_mask));
}

/**
 * @brief Read parquet input using the legacy parquet reader from io source
 *
 * @param io_source io source to read
 * @return cudf::io::table_with_metadata
 */
cudf::io::table_with_metadata read_parquet(io_source const& io_source,
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
void write_parquet(cudf::table_view input, std::string filepath)
{
  // Write the data for inspection
  auto sink_info = cudf::io::sink_info(filepath);
  auto options   = cudf::io::parquet_writer_options::builder(sink_info, input)
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
            << "Usage: hybrid_scan <input parquet file> <column name> <literal> <io source type> "
               "<output parquet file>\n"
            << "Note: Both the column name and literal must be of `string` type\n"
            << "Available IO source types: FILEPATH, HOST_BUFFER, PINNED_BUFFER (Default) \n\n"
            << "Example: hybrid_scan example.parquet col_a 100 output.parquet PINNED_BUFFER\n\n";
}

}  // namespace

/**
 * @brief Main for hybrid scan example
 *
 * Command line parameters:
 * 1. parquet input file name/path (default: "example.parquet")
 * 2. column name for filter expression (default: "col_a")
 * 3. literal for filter expression (default: "100")
 * 4. io source type (default: "PINNED_BUFFER")
 * 5. parquet output file name/path (default: "output.parquet")
 *
 * Example invocation from directory `cudf/cpp/examples/hybrid_scan`:
 * ./build/hybrid_scan example.parquet col_a 100 output.parquet PINNED_BUFFER output.parquet
 *
 */
int main(int argc, char const** argv)
{
  auto input_filepath  = std::string{"example.parquet"};
  auto output_filepath = std::string{"output.parquet"};
  auto column_name     = std::string{"col_a"};
  auto literal_value   = std::string{"100"};
  auto io_source_type  = io_source_type::PINNED_BUFFER;

  switch (argc) {
    case 6: output_filepath = argv[5]; [[fallthrough]];
    case 5: io_source_type = get_io_source_type(argv[4]); [[fallthrough]];
    case 4: literal_value = argv[3]; [[fallthrough]];
    case 3: column_name = argv[2]; [[fallthrough]];
    case 2:  // Check if instead of input_paths, the first argument is `-h` or `--help`
      if (auto arg = std::string{argv[1]}; arg != "-h" and arg != "--help") {
        input_filepath = std::move(arg);
        break;
      }
      [[fallthrough]];
    default: print_usage(); throw std::invalid_argument("Invalid arguments");
  }

  // Check if input file exists
  if (not std::filesystem::is_regular_file(input_filepath)) {
    throw std::runtime_error("Input file does not exist");
  }

  // Initialize mr, default stream and stream pool
  auto constexpr is_pool_used = true;
  auto stream                 = cudf::get_default_stream();
  auto resource               = create_memory_resource(is_pool_used);
  auto stats_mr =
    rmm::mr::statistics_resource_adaptor<rmm::mr::device_memory_resource>(resource.get());
  rmm::mr::set_current_device_resource(&stats_mr);

  // Create filter expression
  auto const column_reference = cudf::ast::column_name_reference(column_name);
  auto scalar                 = cudf::string_scalar(literal_value);
  auto literal                = cudf::ast::literal(scalar);
  auto filter_expression =
    cudf::ast::operation(cudf::ast::ast_operator::EQUAL, column_reference, literal);

  // Create io source
  auto const data_source = io_source{input_filepath, io_source_type, stream};

  // Read with legacy reader without timing
  {
    std::cout << "\nReading " << input_filepath << "...\n";
    std::cout << "Note: Not timing this initial parquet read as it may include\n"
                 "times for nvcomp, cufile loading and RMM growth.\n\n";
    auto [table_legacy_reader, metadata] = read_parquet(data_source, filter_expression, stream);
  }

  // Insert which filters to apply
  std::unordered_set<parquet_filter_type> filters;
  {
    filters.insert(parquet_filter_type::ROW_GROUPS_WITH_STATS);
    filters.insert(parquet_filter_type::ROW_GROUPS_WITH_DICT_PAGES);
    filters.insert(parquet_filter_type::ROW_GROUPS_WITH_BLOOM_FILTERS);
    filters.insert(parquet_filter_type::DATA_PAGES_WITH_PAGE_INDEX);
  }

  cudf::examples::timer timer;
  std::cout << "Reading " << input_filepath << " with next-gen parquet reader...\n";
  timer.reset();
  auto [table_next_gen_reader, row_mask] =
    hybrid_scan(data_source, filter_expression, filters, stream, stats_mr);
  timer.print_elapsed_millis();

  std::cout << "Reading " << input_filepath << " with legacy parquet reader...\n";
  timer.reset();
  auto [table_legacy_reader, metadata] = read_parquet(data_source, filter_expression, stream);
  timer.print_elapsed_millis();

  std::cout << "Writing " << output_filepath << "...\n";
  write_parquet(table_next_gen_reader->view(), "next_gen_" + output_filepath);
  write_parquet(table_legacy_reader->view(), "legacy_" + output_filepath);

  // Check for validity
  // FIXME:For lists, this will fail on column types mismatch. The data would still be intact.
  cudf::examples::check_tables_equal(table_next_gen_reader->view(), table_legacy_reader->view());

  return 0;
}
