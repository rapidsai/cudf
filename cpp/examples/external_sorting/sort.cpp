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

/**
 * @file sort.cpp
 * @brief External sorting example that reads parquet files from a directory and sorts using sample sort.
 *
 * This example demonstrates:
 * 1. Reading parquet files from a specified directory without concatenating (to avoid memory limits)
 * 2. Sorting each individual table by the first column
 * 3. Sampling splitters from each sorted table for external sorting
 * 4. Implementing a sample sort algorithm suitable for large datasets
 *
 * Usage: ./sort [input_dir] [num_files]
 *
 * Example: ./sort /path/to/data 4
 * This reads 4 files (data_0.parquet to data_3.parquet) from /path/to/data, 
 * sorts each individually, and demonstrates external sorting with sample sort.
 */

#include "../utilities/timer.hpp"
#include "cusort.hpp"
#include "parquet_io.hpp"

#include <cudf/copying.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

/**
 * @brief Check if all expected data files exist in the input directory
 */
bool input_files_exist(std::string const& input_dir, int num_files)
{
  for (int i = 0; i < num_files; ++i) {
    std::string filepath = input_dir + "/data_" + std::to_string(i) + ".parquet";
    if (!std::filesystem::exists(filepath)) {
      return false;
    }
  }
  return true;
}

/**
 * @brief Print usage information
 */
void print_usage()
{
  std::cout << "\nUsage: sort [input_dir] [num_files]\n\n"
            << "Arguments:\n"
            << "  input_dir       : Directory containing parquet files to read (default: ./sort_data)\n"
            << "  num_files       : Number of parquet files to read (default: 4)\n\n"
            << "The program expects parquet files named 'data_0.parquet', 'data_1.parquet', etc.\n"
            << "in the input directory.\n\n"
            << "This implements external sorting using sample sort algorithm:\n"
            << "  1. Reads each parquet file individually (avoids memory limits)\n"
            << "  2. Sorts each table by the first column\n"
            << "  3. Samples splitters from each sorted table for partitioning\n\n"
            << "Example: ./sort /path/to/data 8\n"
            << "This reads files data_0.parquet through data_7.parquet from /path/to/data,\n"
            << "and demonstrates external sorting with sample sort algorithm.\n\n";
}

/**
 * @brief Main function
 */
int main(int argc, char** argv)
{
  // Default parameters
  std::string input_dir = "./sort_data";
  int num_files = 4;

  // Parse command line arguments
  if (argc >= 2) {
    if (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
      print_usage();
      return 0;
    }
    input_dir = std::string{argv[1]};
  }
  if (argc >= 3) num_files = std::stoi(argv[2]);

  // Validate parameters
  if (num_files <= 0) {
    std::cerr << "Error: Number of files must be a positive integer" << std::endl;
    print_usage();
    return 1;
  }

  // Check if input directory exists
  if (!std::filesystem::exists(input_dir) || !std::filesystem::is_directory(input_dir)) {
    std::cerr << "Error: Input directory '" << input_dir << "' does not exist or is not a directory" << std::endl;
    return 1;
  }

  // Check if all required input files exist
  if (!input_files_exist(input_dir, num_files)) {
    std::cerr << "Error: Not all required parquet files exist in " << input_dir << std::endl;
    std::cerr << "Expected files: ";
    for (int i = 0; i < num_files; ++i) {
      std::cerr << "data_" << i << ".parquet";
      if (i < num_files - 1) std::cerr << ", ";
    }
    std::cerr << std::endl;
    return 1;
  }

  // TODO: add number of keys to be sorted as a parameter
  std::cout << "External Sorting Example (Sample Sort Algorithm)" << std::endl;
  std::cout << "================================================" << std::endl;
  std::cout << "Input directory: " << input_dir << std::endl;
  std::cout << "Number of files: " << num_files << std::endl << std::endl;

  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  
  cudf::examples::timer perf_timer;

  std::vector<std::unique_ptr<cudf::column>> splitters;
  for (int i = 0; i < num_files; i++) {
    auto const filepath = input_dir + "/data_" + std::to_string(i) + ".parquet"; 
    auto table = cudf::examples::read_parquet_file(filepath, stream, mr);
    splitters.push_back(cudf::examples::sample_splitters(table->view(), num_files, stream, mr));
  }

  perf_timer.print_elapsed_millis();
  perf_timer.reset();

  return 0;
}
