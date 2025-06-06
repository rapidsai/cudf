/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table_view.hpp>

#include <string>

/**
 * @file parquet_io_metadata_caching.cpp
 * @brief Demonstrates usage of the libcudf APIs to read parquet file format
 * with and without metadata caching.
 */

/**
 * @brief Read parquet input from file
 *
 * @param filepath path to input parquet file
 * @return cudf::io::table_with_metadata
 */
cudf::io::table_with_metadata read_parquet(std::string filepath)
{
  auto source_info = cudf::io::source_info(filepath);
  auto builder     = cudf::io::parquet_reader_options::builder(source_info);
  auto options     = builder.build();
  return cudf::io::read_parquet(options);
}

/**
 * @brief Read parquet input from file iterarting all rowgroups
 *
 * @param filepath path to input parquet file
 * @param num_rowgroups the number of rowgroups
 */
void read_parquet_RGs(std::string filepath, cudf::size_type num_rowgroups)
{
  for (cudf::size_type rg_id = 0; rg_id < num_rowgroups; rg_id++) {
    auto source_info = cudf::io::source_info(filepath);
    auto builder     = cudf::io::parquet_reader_options::builder(source_info);
    auto options     = builder.build();
    options.set_row_groups({{rg_id}});
    cudf::io::read_parquet(options);
  }
}

/**
 * @brief Read parquet input from file with metadata-caching
 *
 * @param filepath path to input parquet file
 * @param aggregate_reader_metadata_ptr the cached metadata pointer
 * @return cudf::io::table_with_metadata
 */
cudf::io::table_with_metadata read_parquet_with_metadata_caching(
  std::string filepath, std::uintptr_t aggregate_reader_metadata_ptr)
{
  auto source_info = cudf::io::source_info(filepath);
  auto builder     = cudf::io::parquet_reader_options::builder(source_info);
  auto options     = builder.build();
  options.set_aggregate_reader_metadata(
    aggregate_reader_metadata_ptr);  // here to enable metadata caching
  return cudf::io::read_parquet(options);
}

/**
 * @brief Read parquet input from file iterarting all rowgroups
 *
 * @param filepath path to input parquet file
 * @param aggregate_reader_metadata_ptr the cached metadata pointer
 * @param num_rowgroups the number of rowgroups
 */
void read_parquet_RGs_with_metadata_caching(std::string filepath,
                                            std::uintptr_t aggregate_reader_metadata_ptr,
                                            cudf::size_type num_rowgroups)
{
  for (cudf::size_type rg_id = 0; rg_id < num_rowgroups; rg_id++) {
    auto source_info = cudf::io::source_info(filepath);
    auto builder     = cudf::io::parquet_reader_options::builder(source_info);
    auto options     = builder.build();
    options.set_row_groups({{rg_id}});
    options.set_aggregate_reader_metadata(
      aggregate_reader_metadata_ptr);  // here to enable metadata caching
    cudf::io::read_parquet(options);
  }
}

/**
 * @brief Function to print example usage and argument information.
 */
void print_usage() { std::cout << "\nUsage: parquet_io_metadata_caching <input parquet file>\n"; }

/**
 * @brief Main for nested_types examples
 *
 * Command line parameters:
 * 1. parquet input file name/path (default: "example.parquet")
 *
 */
int main(int argc, char const** argv)
{
  std::string input_filepath = "example.parquet";

  switch (argc) {
    case 2:  // Check if instead of input_paths, the first argument is `-h` or `--help`
      if (auto arg = std::string{argv[1]}; arg != "-h" and arg != "--help") {
        input_filepath = std::move(arg);
        break;
      }
      [[fallthrough]];
    default: print_usage(); throw std::runtime_error("");
  }

  // Create and use a memory pool
  bool is_pool_used = true;
  auto resource     = create_memory_resource(is_pool_used);
  cudf::set_current_device_resource(resource.get());

  // Prepare the metadata outside the timer scope
  auto source_info                   = cudf::io::source_info(input_filepath);
  auto metadata                      = cudf::io::read_parquet_metadata(source_info);
  auto aggregate_reader_metadata_ptr = metadata.get_aggregate_reader_metadata_ptr();

  // Bluk Read: read the file in one read call
  {
    // Read input parquet file
    // We do not want to time the initial read time as it may include
    // time for nvcomp, cufile loading and RMM growth
    std::cout << "\nReading " << input_filepath << " without metadata-caching...\n";
    std::cout << "Note: Not timing the initial parquet read as it may include\n"
                 "times for nvcomp, cufile loading and RMM growth.\n\n";

    cudf::examples::timer timer;
    auto [first_read_input, first_read_metadata] = read_parquet(input_filepath);
    timer.print_elapsed_millis();

    // Read the parquet file written with encoding and compression
    std::cout << "Reading " << input_filepath << " with metadata-caching...\n";

    // Reset the timer
    timer.reset();
    auto [second_read_input, second_read_metadata] =
      read_parquet_with_metadata_caching(input_filepath, aggregate_reader_metadata_ptr);
    timer.print_elapsed_millis();

    // Check for validity
    check_tables_equal(first_read_input->view(), second_read_input->view());
  }

  // Iterating all row-groups one-by-one
  {
    const auto num_rowgroups = metadata.num_rowgroups();
    std::cout << "Number of Parquet row-groups of the inputfile: " << num_rowgroups << std::endl;

    std::cout << "Iterating all rowgroups " << input_filepath << " without metadata-caching...\n";
    cudf::examples::timer timer;
    read_parquet_RGs(input_filepath, num_rowgroups);
    timer.print_elapsed_millis();

    std::cout << "Iterating all rowgroups " << input_filepath << " with metadata-caching...\n";
    timer.reset();
    read_parquet_RGs_with_metadata_caching(
      input_filepath, aggregate_reader_metadata_ptr, num_rowgroups);
    timer.print_elapsed_millis();
  }
  return 0;
}
