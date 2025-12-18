/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common_utils.hpp"
#include "io_source.hpp"
#include "timer.hpp"

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table_view.hpp>

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
  auto const table_next_gen_reader =
    hybrid_scan<true>(data_source, filter_expression, filters, stream, stats_mr);
  timer.print_elapsed_millis();

  std::cout << "Reading " << input_filepath << " with main parquet reader...\n";
  timer.reset();
  auto const [table_main_reader, metadata] = read_parquet(data_source, filter_expression, stream);
  timer.print_elapsed_millis();

  // Check for validity
  check_tables_equal(table_next_gen_reader->view(), table_main_reader->view(), stream);

  return 0;
}
