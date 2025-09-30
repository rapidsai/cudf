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

#include "parquet_inspect_utils.hpp"

#include <cudf/io/parquet_schema.hpp>
#include <cudf/io/types.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <filesystem>
#include <iostream>
#include <string>

/**
 * @file parquet_inspect.cpp
 * @brief Inspects a parquet file and writes two parquet files containing row group and page
 * metadata respectively.
 */

namespace {

/**
 * @brief Function to print example usage and argument information.
 */
void print_usage()
{
  std::cout << "\nUsage: parquet_inspect <input parquet file> <output path>\n\n";
}

}  // namespace

/**
 * @brief Main for parquet_inspect examples
 *
 * Command line parameters:
 * 1. parquet input file name (default: "example.parquet")
 * 2. parquet output path (default: "$pwd")
 *
 * Example invocation from directory `cudf/cpp/examples/parquet_inspect`:
 * ./build/parquet_inspect example.parquet ./
 */
int main(int argc, char const** argv)
{
  std::string input_filepath                          = "example.parquet";
  std::string output_path                             = std::filesystem::current_path().string();
  std::optional<cudf::io::statistics_freq> page_stats = std::nullopt;

  switch (argc) {
    case 3: output_path = argv[2]; [[fallthrough]];
    case 2:  // Check if instead of input_paths, the first argument is `-h` or `--help`
    {
      auto const arg = std::string{argv[1]};
      if (arg == "-h" or arg == "--help") {
        print_usage();
        return 0;
      } else if (arg != "-h" and arg != "--help") {
        input_filepath = std::filesystem::absolute(arg).string();
        break;
      }
      [[fallthrough]];
    }
    default: print_usage(); throw std::runtime_error("Invalid arguments");
  }

  CUDF_EXPECTS(
    std::filesystem::exists(input_filepath) and std::filesystem::is_regular_file(input_filepath),
    "Input file '" + input_filepath + "' does not exist or is not a regular file.",
    std::invalid_argument);

  auto const filename = std::filesystem::path(input_filepath).stem().string();

  auto const stream = cudf::get_default_stream();
  auto const mr     = create_memory_resource(true);
  cudf::set_current_device_resource(mr.get());

  // Read parquet footer metadata
  auto [metadata, has_page_index] = read_parquet_file_metadata(input_filepath);

  // Write row group metadata
  auto output_filepath = output_path + "/" + filename + ".rowgroups.parquet";
  write_rowgroup_metadata(metadata, output_filepath, stream);

  // Write page metadata
  if (has_page_index) {
    auto output_filepath = output_path + "/" + filename + ".pages.parquet";
    write_page_metadata(metadata, output_filepath, stream);
  }

  stream.synchronize();

  return 0;
}
