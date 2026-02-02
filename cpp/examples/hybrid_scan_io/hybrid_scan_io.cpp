/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common_utils.hpp"
#include "io_source.hpp"
#include "timer.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/aligned.hpp>
#include <rmm/mr/aligned_resource_adaptor.hpp>
#include <rmm/mr/device_memory_resource.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <thrust/host_vector.h>

#include <filesystem>
#include <string>
#include <unordered_set>

/**
 * @file hybrid_scan.cpp
 *
 * @brief This example demonstrates the use of libcudf next-gen parquet reader to optimally read
 * a parquet file subject to a highly selective string-type point lookup (col_name ==
 * literal) filter. The same file is also read using the libcudf legacy parquet reader and the read
 * times are compared.
 */

namespace {

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
 * @brief Enum to represent the available parquet filters
 */
enum class parquet_filter_type : uint8_t {
  ROW_GROUPS_WITH_STATS               = 0,
  ROW_GROUPS_WITH_DICT_PAGES          = 1,
  ROW_GROUPS_WITH_BLOOM_FILTERS       = 2,
  FILTER_COLUMN_PAGES_WITH_PAGE_INDEX = 3,
  PAYLOAD_COLUMN_PAGES_WITH_ROW_MASK  = 4,
};

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
auto hybrid_scan(io_source const& io_source,
                 cudf::ast::operation const& filter_expression,
                 std::unordered_set<parquet_filter_type> const& filters,
                 rmm::cuda_stream_view stream,
                 rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  auto options = cudf::io::parquet_reader_options::builder().filter(filter_expression).build();

  // Input file buffer span
  auto const file_buffer_span = io_source.get_host_buffer_span();

  std::cout << "\nREADER: Setup, metadata and page index...\n";
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
    // Fetch dictionary page buffers and corresponding device spans from the input file buffer
    auto dictionary_page_buffers =
      fetch_byte_ranges(file_buffer_span, dict_page_byte_ranges, stream, mr);
    auto dictionary_page_data = make_device_spans<uint8_t>(dictionary_page_buffers);
    dictionary_page_filtered_row_group_indices = reader->filter_row_groups_with_dictionary_pages(
      dictionary_page_data, current_row_group_indices, options, stream);

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
    auto constexpr bloom_filter_alignment = rmm::CUDA_ALLOCATION_ALIGNMENT;
    auto aligned_mr = rmm::mr::aligned_resource_adaptor<rmm::mr::device_memory_resource>(
      mr, bloom_filter_alignment);
    std::cout << "READER: Filter row groups with bloom filters...\n";
    timer.reset();
    auto bloom_filter_buffers =
      fetch_byte_ranges(file_buffer_span, bloom_filter_byte_ranges, stream, aligned_mr);
    auto bloom_filter_data = make_device_spans<uint8_t>(bloom_filter_buffers);
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

  // Check whether to prune filter column data pages
  using cudf::io::parquet::experimental::use_data_page_mask;
  auto const prune_filter_data_pages =
    filters.contains(parquet_filter_type::FILTER_COLUMN_PAGES_WITH_PAGE_INDEX);

  auto row_mask = std::unique_ptr<cudf::column>{};
  if (prune_filter_data_pages) {
    std::cout << "READER: Filter data pages of filter columns with page index stats...\n";
    timer.reset();
    // Filter data pages with page index stats
    row_mask =
      reader->build_row_mask_with_page_index_stats(current_row_group_indices, options, stream, mr);
    timer.print_elapsed_millis();
  } else {
    std::cout << "SKIP: Filter column data page filtering with page index stats...\n\n";
    timer.reset();
    row_mask = reader->build_all_true_row_mask(current_row_group_indices, stream, mr);
    timer.print_elapsed_millis();
  }

  std::cout << "READER: Materialize filter columns...\n";
  timer.reset();
  // Get column chunk byte ranges from the reader
  auto const filter_column_chunk_byte_ranges =
    reader->filter_column_chunks_byte_ranges(current_row_group_indices, options);
  auto filter_column_chunk_buffers =
    fetch_byte_ranges(file_buffer_span, filter_column_chunk_byte_ranges, stream, mr);
  auto filter_column_chunk_data = make_device_spans<uint8_t>(filter_column_chunk_buffers);

  // Materialize the table with only the filter columns
  auto row_mask_mutable_view = row_mask->mutable_view();
  auto filter_table =
    reader
      ->materialize_filter_columns(
        current_row_group_indices,
        filter_column_chunk_data,
        row_mask_mutable_view,
        prune_filter_data_pages ? use_data_page_mask::YES : use_data_page_mask::NO,
        options,
        stream)
      .tbl;
  timer.print_elapsed_millis();

  // Check whether to prune payload column data pages
  auto const prune_payload_data_pages =
    filters.contains(parquet_filter_type::PAYLOAD_COLUMN_PAGES_WITH_ROW_MASK);

  if (prune_payload_data_pages) {
    std::cout << "READER: Filter data pages of payload columns with row mask...\n";
  } else {
    std::cout << "SKIP: Payload column data page filtering with row mask...\n\n";
  }

  std::cout << "READER: Materialize payload columns...\n";
  timer.reset();
  // Get column chunk byte ranges from the reader
  auto const payload_column_chunk_byte_ranges =
    reader->payload_column_chunks_byte_ranges(current_row_group_indices, options);
  auto payload_column_chunk_buffers =
    fetch_byte_ranges(file_buffer_span, payload_column_chunk_byte_ranges, stream, mr);
  auto payload_column_chunk_data = make_device_spans<uint8_t>(payload_column_chunk_buffers);

  // Materialize the table with only the payload columns
  auto payload_table =
    reader
      ->materialize_payload_columns(
        current_row_group_indices,
        payload_column_chunk_data,
        row_mask->view(),
        prune_payload_data_pages ? use_data_page_mask::YES : use_data_page_mask::NO,
        options,
        stream)
      .tbl;
  timer.print_elapsed_millis();

  return std::make_tuple(combine_tables(std::move(filter_table), std::move(payload_table)),
                         std::move(row_mask));
}

/**
 * @brief Function to print example usage and argument information.
 */
void inline print_usage()
{
  std::cout
    << std::endl
    << "Usage: hybrid_scan <input parquet file> <column name> <literal> <io source type>\n\n"
    << "Available IO source types: HOST_BUFFER, PINNED_BUFFER (Default) \n\n"
    << "Note: Both the column name and literal must be of `string` type. The constructed filter "
       "expression\n      will be of the form <column name> == <literal>\n\n"
    << "Example usage: hybrid_scan example.parquet string_col 0000001 PINNED_BUFFER \n\n";
}

}  // namespace

/**
 * @brief Main for hybrid scan example
 *
 * Command line parameters:
 * 1. parquet input file name/path (default: "example.parquet")
 * 2. column name for filter expression (default: "string_col")
 * 3. literal for filter expression (default: "0000001")
 * 4. io source type (default: "PINNED_BUFFER")
 *
 * The filter expression will be of the form col_name == literal (default: string_col == 0000001)
 *
 * Example invocation from directory `cudf/cpp/examples/hybrid_scan`:
 * ./build/hybrid_scan example.parquet string_col 0000001 PINNED_BUFFER
 *
 */
int main(int argc, char const** argv)
{
  auto input_filepath = std::string{"example.parquet"};
  auto column_name    = std::string{"string_col"};
  auto literal_value  = std::string{"0000001"};
  auto io_source_type = io_source_type::PINNED_BUFFER;

  switch (argc) {
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
  auto constexpr is_pool_used = false;
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

  // Read with the main reader without timing
  {
    std::cout << "\nReading " << input_filepath << "...\n";
    std::cout << "Note: Not timing this initial parquet read as it may include\n"
                 "times for nvcomp, cufile loading and RMM growth.\n\n";
    std::ignore = read_parquet(data_source, filter_expression, stream);
  }

  // Insert which filters to apply
  std::unordered_set<parquet_filter_type> filters;
  {
    filters.insert(parquet_filter_type::ROW_GROUPS_WITH_STATS);
    filters.insert(parquet_filter_type::ROW_GROUPS_WITH_DICT_PAGES);
    filters.insert(parquet_filter_type::ROW_GROUPS_WITH_BLOOM_FILTERS);
    // Deliberately disabled as it has a high cost to benefit ratio
    // filters.insert(parquet_filter_type::FILTER_COLUMN_PAGES_WITH_PAGE_INDEX);
    filters.insert(parquet_filter_type::PAYLOAD_COLUMN_PAGES_WITH_ROW_MASK);
  }

  timer timer;
  std::cout << "Reading " << input_filepath << " with next-gen parquet reader...\n";
  timer.reset();
  auto [table_next_gen_reader, row_mask] =
    hybrid_scan(data_source, filter_expression, filters, stream, stats_mr);
  timer.print_elapsed_millis();

  std::cout << "Reading " << input_filepath << " with main parquet reader...\n";
  timer.reset();
  auto [table_main_reader, metadata] = read_parquet(data_source, filter_expression, stream);
  timer.print_elapsed_millis();

  // Check for validity
  check_tables_equal(table_next_gen_reader->view(), table_main_reader->view(), stream);

  return 0;
}
