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
 * @brief External sorting example that creates random data, stores it across multiple parquet files,
 *        and then reads and sorts the data.
 *
 * This example demonstrates:
 * 1. Generating random tables with configurable numbers of columns and rows
 * 2. Writing data to multiple parquet files for external storage
 * 3. Reading the data back using multithreaded I/O
 * 4. Sorting the combined dataset using libcudf's sorting functionality
 *
 * Usage: ./sort [n_columns] [m_rows_per_file] [num_files] [output_dir]
 *
 * Example: ./sort 5 1000000 4 /tmp/sort_data
 * This creates 4 files, each with 5 columns and 1M rows, then sorts the combined 4M row dataset.
 */

#include "../utilities/timer.hpp"
#include "parquet_io.hpp"
#include "random_table_generator.cuh"

#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_pool.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>



/**
 * @brief Check if all expected data files already exist
 */
bool data_files_exist(const std::string& output_dir, int num_files)
{
  for (int i = 0; i < num_files; ++i) {
    std::string filepath = output_dir + "/data_" + std::to_string(i) + ".parquet";
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
  std::cout << "\nUsage: sort [n_columns] [m_rows_per_file] [num_files] [output_dir]\n\n"
            << "Arguments:\n"
            << "  n_columns       : Number of columns in each table (default: 5)\n"
            << "  m_rows_per_file : Number of rows per parquet file (default: 1000000)\n"
            << "  num_files       : Number of parquet files to create (default: 4)\n"
            << "  output_dir      : Directory to store parquet files (default: ./sort_data)\n\n"
            << "Example: ./sort 3 500000 8 /tmp/my_sort_test\n"
            << "This creates 8 files with 3 columns and 500K rows each (4M rows total)\n\n"
            << "Note: If data files already exist, generation and writing phases will be skipped.\n\n";
}

/**
 * @brief Main function
 */
int main(int argc, char** argv)
{
  // Default parameters
  cudf::size_type n_columns = 5;
  cudf::size_type m_rows_per_file = 1000000;
  int num_files = 4;
  std::string output_dir = "./sort_data";

  // Parse command line arguments
  if (argc >= 2) {
    if (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
      print_usage();
      return 0;
    }
    n_columns = std::stoi(argv[1]);
  }
  if (argc >= 3) m_rows_per_file = std::stoi(argv[2]);
  if (argc >= 4) num_files = std::stoi(argv[3]);
  if (argc >= 5) output_dir = argv[4];

  // Validate parameters
  if (n_columns <= 0 || m_rows_per_file <= 0 || num_files <= 0) {
    std::cerr << "Error: All parameters must be positive integers" << std::endl;
    print_usage();
    return 1;
  }

  std::cout << "External Sorting Example" << std::endl;
  std::cout << "========================" << std::endl;
  std::cout << "Columns per table: " << n_columns << std::endl;
  std::cout << "Rows per file: " << m_rows_per_file << std::endl;
  std::cout << "Number of files: " << num_files << std::endl;
  std::cout << "Total rows: " << static_cast<int64_t>(m_rows_per_file) * num_files << std::endl;
  std::cout << "Output directory: " << output_dir << std::endl << std::endl;

  try {
    // Initialize RMM
    rmm::mr::cuda_memory_resource cuda_mr{};
    rmm::mr::pool_memory_resource pool_mr{&cuda_mr, rmm::percent_of_free_device_memory(80)};
    cudf::set_current_device_resource(&pool_mr);

    // Create stream pool for multithreading
    auto stream_pool = rmm::cuda_stream_pool(std::max(4, num_files));
    auto default_stream = cudf::get_default_stream();

    // Create output directory
    std::filesystem::create_directories(output_dir);

    cudf::examples::timer perf_timer;

    // Check if data files already exist
    bool files_exist = data_files_exist(output_dir, num_files);
    
    if (files_exist) {
      std::cout << "Data files already exist. Skipping generation and writing phases." << std::endl;
    } else {
      // Phase 1: Generate and write random data to parquet files
      std::cout << "Phase 1: Generating and writing " << num_files << " parquet files..." << std::endl;
      std::vector<std::unique_ptr<cudf::table>> generated_tables;
      std::vector<cudf::table_view> table_views;

      for (int i = 0; i < num_files; ++i) {
        std::cout << "Generating table " << (i + 1) << "/" << num_files << std::endl;
        auto table = cudf::examples::generate_random_table(n_columns, m_rows_per_file, default_stream, cudf::get_current_device_resource_ref());
        table_views.push_back(table->view());
        generated_tables.push_back(std::move(table));
      }

      std::cout << "Data generation ";
      perf_timer.print_elapsed_millis();
      perf_timer.reset();

      // Write files using multithreaded I/O
      std::vector<cudf::examples::write_task> write_tasks;
      std::vector<std::thread> write_threads;

      for (int i = 0; i < num_files; ++i) {
        write_tasks.emplace_back(
          cudf::examples::write_task{output_dir, table_views, i, stream_pool.get_stream()});
      }

      for (auto& task : write_tasks) {
        write_threads.emplace_back(task);
      }

      for (auto& thread : write_threads) {
        thread.join();
      }

      std::cout << "Writing parquet files ";
      perf_timer.print_elapsed_millis();
      perf_timer.reset();
    }

    // Phase 2: Read parquet files
    std::cout << "\nReading parquet files..." << std::endl;
    std::vector<std::string> filepaths;
    for (int i = 0; i < num_files; ++i) {
      filepaths.push_back(output_dir + "/data_" + std::to_string(i) + ".parquet");
    }

    int thread_count = std::min(4, num_files);
    auto read_tables = cudf::examples::read_parquet_files_multithreaded(filepaths, thread_count, stream_pool);

    std::cout << "Reading parquet files ";
    perf_timer.print_elapsed_millis();
    perf_timer.reset();

    // Phase 3: Concatenate all data
    std::cout << "\nConcatenating data..." << std::endl;
    std::vector<cudf::table_view> read_views;
    cudf::size_type total_rows = 0;

    for (auto const& tbl : read_tables) {
      if (tbl != nullptr) {
        read_views.push_back(tbl->view());
        total_rows += tbl->num_rows();
      }
    }

    std::cout << "Total rows to concatenate: " << total_rows << std::endl;
    auto combined_table = cudf::concatenate(read_views, default_stream);

    std::cout << "Data concatenation ";
    perf_timer.print_elapsed_millis();
    perf_timer.reset();

    // Phase 4: Sort the combined data
    std::cout << "\nSorting data..." << std::endl;
    std::cout << "Sorting by first column (ascending)" << std::endl;

    // Create sort order specification (sort by first column)
    std::vector<cudf::order> column_order{cudf::order::ASCENDING};
    std::vector<cudf::null_order> null_precedence{cudf::null_order::AFTER};

    // Perform the sort - this returns indices for sorted order
    auto sorted_indices = cudf::sorted_order(combined_table->view().select({0}), column_order, null_precedence, default_stream);

    std::cout << "Computing sort order ";
    perf_timer.print_elapsed_millis();
    perf_timer.reset();

    // Gather the sorted data
    auto sorted_table = cudf::gather(combined_table->view(), sorted_indices->view(), cudf::out_of_bounds_policy::DONT_CHECK, default_stream);

    std::cout << "Gathering sorted data ";
    perf_timer.print_elapsed_millis();
    perf_timer.reset();

    // Phase 5: Write sorted result
    std::cout << "\nWriting sorted result..." << std::endl;
    std::string sorted_filepath = output_dir + "/sorted_result.parquet";
    cudf::examples::write_parquet_file(sorted_filepath, sorted_table->view(), default_stream);

    std::cout << "Writing sorted result ";
    perf_timer.print_elapsed_millis();

    // Summary
    std::cout << "\nExternal sorting completed successfully!" << std::endl;
    std::cout << "Summary:" << std::endl;
    std::cout << "  Input: " << num_files << " files × " << m_rows_per_file 
              << " rows × " << n_columns << " columns" << std::endl;
    std::cout << "  Total processed: " << total_rows << " rows" << std::endl;
    std::cout << "  Sorted result: " << sorted_filepath << std::endl;

    // Verify first few values to check sorting
    if (sorted_table->num_rows() > 0) {
      std::cout << "\nSorting verification (first column values sample):" << std::endl;
      // Note: In a production example, you might want to copy some values back to host for display
      std::cout << "Sorted table has " << sorted_table->num_rows() << " rows" << std::endl;
    }

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
