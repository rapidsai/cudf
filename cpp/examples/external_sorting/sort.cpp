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

#include <cudf/ast/expressions.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <filesystem>
#include <iostream>
#include <memory>
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

struct scalar_accessor {  
 private:
  template <typename ResultType>
  static constexpr bool is_supported() {
    return cudf::is_numeric<ResultType>() && !std::is_same_v<ResultType, bool>;
  }

 public:
  template <typename ResultType>  
  cudf::ast::literal operator()(std::vector<std::byte> &splitters, int pos, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
  requires(is_supported<ResultType>())
  {
    auto casted_splitters = reinterpret_cast<ResultType*>(splitters.data());
    auto scalar_val = casted_splitters[pos];
    using ScalarType = cudf::scalar_type_t<ResultType>;  
    auto literal_bound = cudf::numeric_scalar<ResultType>(scalar_val, true, stream, mr);
    return cudf::ast::literal(literal_bound);
  }  
  template <typename ResultType>  
  cudf::ast::literal operator()(std::vector<std::byte> &splitters, int pos, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
  requires(not is_supported<ResultType>())
  {
    CUDF_FAIL("AST literal not implemented for nested types");
  }
};

struct to_host {  
 private:
  template <typename T>
  static constexpr bool is_supported() {
    return cudf::is_numeric<T>() && !std::is_same_v<T, bool>;
  }

 public:
  template <typename T>  
  size_t operator()(cudf::column_view c, std::vector<std::byte> &host_data, rmm::cuda_stream_view stream)
  requires(is_supported<T>())
  {
    auto col_span  = cudf::device_span<T const>(c.data<T>(), c.size());
    auto host_data_ = cudf::detail::make_std_vector(col_span, stream);
    std::memcpy(host_data.data(), host_data_.data(), host_data_.size() * sizeof(T));
    return sizeof(T);
  }  
  template <typename T>  
  size_t operator()(cudf::column_view c, std::vector<std::byte> &host_data, rmm::cuda_stream_view stream)
  requires(not is_supported<T>())
  {
    CUDF_FAIL("AST literal not implemented for nested types");
  }
};

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

  std::vector<std::unique_ptr<cudf::column>> per_table_splitters;
  for (int i = 0; i < num_files; i++) {
    auto const filepath = input_dir + "/data_" + std::to_string(i) + ".parquet"; 
    auto table = cudf::examples::read_parquet_file(filepath, stream, mr);
    per_table_splitters.push_back(cudf::examples::sample_splitters(table->view(), num_files, stream, mr));
  }
  std::vector<cudf::column_view> per_table_splitters_view;
  for(auto const &c : per_table_splitters) {
    per_table_splitters_view.push_back(c->view());
  }
  auto concatenated_splitters = cudf::concatenate(per_table_splitters_view, stream, mr);

  auto num_splitters = num_files - 1;
  auto splitters = cudf::examples::sample_splitters(cudf::table_view({concatenated_splitters->view()}), num_splitters, stream, mr);
  auto sort_col_type = splitters->type();
  std::vector<std::byte> h_splitters(num_splitters * 8);
  auto sort_col_type_size = cudf::type_dispatcher(sort_col_type, to_host{}, splitters->view(), h_splitters, stream);

  auto col_ref = cudf::ast::column_reference(0);
  std::vector<std::vector<std::unique_ptr<cudf::table>>> table_splits(num_splitters + 1);
  for (int i = 0; i < num_files; i++) {
    auto const filepath = input_dir + "/data_" + std::to_string(i) + ".parquet"; 
    auto table = cudf::examples::read_parquet_file(filepath, stream, mr);
    auto upper_bound = cudf::type_dispatcher(sort_col_type, scalar_accessor{}, h_splitters, 0, stream, mr);
    auto less_expr = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref, upper_bound);
    auto boolean_mask = cudf::compute_column(table->view(), less_expr);
    table_splits[0].push_back(cudf::apply_boolean_mask(table->view(), boolean_mask->view()));
    
    for(int j = 1; j < num_splitters; j++) {
      auto lower_bound = cudf::type_dispatcher(sort_col_type, scalar_accessor{}, h_splitters, j - 1, stream, mr);
      auto upper_bound = cudf::type_dispatcher(sort_col_type, scalar_accessor{}, h_splitters, j, stream, mr);
      auto greater_expr = cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, col_ref, lower_bound);
      auto less_expr = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref, upper_bound);
      auto filter_expr = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, less_expr, greater_expr);
      auto boolean_mask = cudf::compute_column(table->view(), filter_expr);
      table_splits[j].push_back(cudf::apply_boolean_mask(table->view(), boolean_mask->view()));
    }

    auto lower_bound = cudf::type_dispatcher(sort_col_type, scalar_accessor{}, h_splitters, num_splitters, stream, mr);
    auto greater_expr = cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, col_ref, lower_bound);
    boolean_mask = cudf::compute_column(table->view(), greater_expr);
    table_splits[num_splitters].push_back(cudf::apply_boolean_mask(table->view(), boolean_mask->view()));
  }

  for(int i = 0; i <= num_splitters; i++) {
    std::vector<cudf::table_view> views;
    for(auto const &t : table_splits[i]) {
      views.push_back(t->view());
    }
    auto partition = cudf::concatenate(views, stream, mr);
    auto sorted_indices = cudf::sorted_order(partition->view().select({0}), std::vector<cudf::order>{cudf::order::ASCENDING}, std::vector<cudf::null_order>{cudf::null_order::AFTER}, stream, mr);
  }

  perf_timer.print_elapsed_millis();
  perf_timer.reset();

  return 0;
}
