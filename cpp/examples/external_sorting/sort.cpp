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
 * @brief External sorting example that reads parquet files from a directory and sorts using sample
 * sort.
 *
 * This example demonstrates:
 * 1. Reading parquet files from a specified directory without concatenating (to avoid memory
 * limits)
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

#include "cusort.hpp"
#include "parquet_io.hpp"
#include "random_table_generator.hpp"
#include "timer.hpp"

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
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

/**
 * @brief Check if all expected data files exist in the input directory
 */
bool input_files_exist(std::string const& input_dir, int num_files)
{
  for (int i = 0; i < num_files; ++i) {
    std::string filepath = input_dir + "/data_" + std::to_string(i) + ".parquet";
    if (!std::filesystem::exists(filepath)) { return false; }
  }
  return true;
}

/**
 * @brief Print usage information
 */
void print_usage()
{
  std::cout
    << "\nUsage: sort [input_dir] [num_files]\n\n"
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

struct make_ast_literal {
 public:
  template <typename InputType>
  std::unique_ptr<cudf::ast::literal> operator()(cudf::scalar &scalar)
    requires(cudf::is_numeric<InputType>())
  {
    auto& typed_scalar = static_cast<cudf::numeric_scalar<InputType>&>(scalar);  
    return std::make_unique<cudf::ast::literal>(typed_scalar);
  }

  template <typename InputType>  
  std::unique_ptr<cudf::ast::literal> operator()(cudf::scalar& scalar)  
    requires(cudf::is_timestamp<InputType>()) {  
    auto& typed_scalar = static_cast<cudf::timestamp_scalar<InputType>&>(scalar);  
    return std::make_unique<cudf::ast::literal>(typed_scalar);  
  }  
    
  template <typename InputType>  
  std::unique_ptr<cudf::ast::literal> operator()(cudf::scalar& scalar)  
    requires(cudf::is_duration<InputType>()) {  
    auto& typed_scalar = static_cast<cudf::duration_scalar<InputType>&>(scalar);  
    return std::make_unique<cudf::ast::literal>(typed_scalar);  
  }

  template <typename InputType>
  std::unique_ptr<cudf::ast::literal> operator()(cudf::scalar &scalar)
    requires(not cudf::is_numeric<InputType>() and not cudf::is_timestamp<InputType>() and not cudf::is_duration<InputType>())
  {
    CUDF_FAIL("AST literal not implemented for non-numeric types");
  }
};

bool sanity_checks(int num_files, std::string const directory)
{
  // Validate parameters
  if (num_files <= 0) {
    std::cerr << "Error: Number of files must be a positive integer" << std::endl;
    print_usage();
    return false;
  }

  // Check if input directory exists
  if (!std::filesystem::exists(directory) || !std::filesystem::is_directory(directory)) {
    std::cerr << "Error: Input directory '" << directory << "' does not exist or is not a directory"
              << std::endl;
    return false;
  }

  return true;
}

/**
 * @brief Main function
 */
int main(int argc, char** argv)
{
  // Default parameters
  std::string input_dir    = "./sort_data";
  int num_files            = 4;
  cudf::size_type num_cols = 13;
  cudf::size_type num_rows = 5000000;
  timer watch;

  // Parse command line arguments
  if (argc >= 2) {
    if (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
      print_usage();
      return 0;
    }
    input_dir = std::string{argv[1]};
  }
  if (argc >= 3) num_files = std::stoi(argv[2]);

  if (!sanity_checks(num_files, input_dir)) { return 1; }

  // TODO: add number of keys to be sorted as a parameter
  std::cout << "External Sorting Example (Sample Sort Algorithm)" << std::endl;
  std::cout << "================================================" << std::endl;
  std::cout << "Input directory: " << input_dir << std::endl;
  std::cout << "Number of files: " << num_files << std::endl << std::endl;

  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();

  if (!input_files_exist(input_dir, num_files)) {
    cudf::examples::write_random_table(input_dir, num_files, num_rows, num_cols, stream, mr);
  }

  std::vector<std::unique_ptr<cudf::column>> per_table_splitters;
  cudf::data_type sort_col_type;
  for (int i = 0; i < num_files; i++) {
    auto const filepath = input_dir + "/data_" + std::to_string(i) + ".parquet";
    auto table          = cudf::examples::read_parquet_file(filepath, stream, mr);
    sort_col_type = table->view().column(0).type();
    per_table_splitters.push_back(
      cudf::examples::sample_splitters(table->view(), num_files, stream, mr));
  }
  watch.print_elapsed_millis();

  auto sort_col_element_size = cudf::size_of(sort_col_type);
  std::vector<cudf::column_view> per_table_splitters_view;
  for (auto const& c : per_table_splitters) {
    per_table_splitters_view.push_back(c->view());
  }
  auto concatenated_splitters = cudf::concatenate(per_table_splitters_view, stream, mr);

  auto num_splitters = num_files - 1;
  auto splitters     = cudf::examples::sample_splitters(
    cudf::table_view({concatenated_splitters->view()}), num_splitters, stream, mr);
  
  watch.reset();
  auto col_ref = cudf::ast::column_reference(0);
  std::vector<std::vector<std::unique_ptr<cudf::table>>> table_splits(num_splitters + 1);
  std::unique_ptr<cudf::scalar> ub_scalar_literal, lb_scalar_literal;
  std::unique_ptr<cudf::ast::literal> upper_bound, lower_bound;
  for (int i = 0; i < num_files; i++) {
    auto const filepath = input_dir + "/data_" + std::to_string(i) + ".parquet";
    auto table          = cudf::examples::read_parquet_file(filepath, stream, mr);
    ub_scalar_literal       = cudf::get_element(splitters->view(), 0, stream, mr);
    upper_bound = cudf::type_dispatcher(sort_col_type, make_ast_literal{}, *ub_scalar_literal);
    auto less_expr    = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref, *upper_bound);
    auto boolean_mask = cudf::compute_column(table->view(), less_expr, stream, mr);
    table_splits[0].push_back(cudf::apply_boolean_mask(table->view(), boolean_mask->view(), stream, mr));

    for (int j = 1; j < num_splitters; j++) {
      lb_scalar_literal       = std::move(ub_scalar_literal);
      lower_bound = cudf::type_dispatcher(sort_col_type, make_ast_literal{}, *lb_scalar_literal);
      ub_scalar_literal = cudf::get_element(splitters->view(), j, stream, mr);
      upper_bound = cudf::type_dispatcher(sort_col_type, make_ast_literal{}, *ub_scalar_literal);
      auto greater_expr =
        cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, col_ref, *lower_bound);
      less_expr = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref, *upper_bound);
      auto filter_expr =
        cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, less_expr, greater_expr);
      auto boolean_mask = cudf::compute_column(table->view(), filter_expr, stream, mr);
      table_splits[j].push_back(cudf::apply_boolean_mask(table->view(), boolean_mask->view(), stream, mr));
    }

    lb_scalar_literal       = std::move(ub_scalar_literal);
    lower_bound = cudf::type_dispatcher(sort_col_type, make_ast_literal{}, *lb_scalar_literal);
    auto greater_expr =
      cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, col_ref, *lower_bound);
    boolean_mask = cudf::compute_column(table->view(), greater_expr, stream, mr);
    table_splits[num_splitters].push_back(
      cudf::apply_boolean_mask(table->view(), boolean_mask->view(), stream, mr));
  }
  watch.print_elapsed_millis();

  watch.reset();
  for (int i = 0; i <= num_splitters; i++) {
    std::vector<cudf::table_view> views;
    for (auto const& t : table_splits[i]) {
      views.push_back(t->view());
    }
    auto partition      = cudf::concatenate(views, stream, mr);
    auto sorted_indices = cudf::sorted_order(partition->view().select({0}),
                                             std::vector<cudf::order>{cudf::order::ASCENDING},
                                             std::vector<cudf::null_order>{cudf::null_order::AFTER},
                                             stream,
                                             mr);
  }
  watch.print_elapsed_millis();

  return 0;
}
