/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <cudf/io/orc.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table_view.hpp>

#include <string>

/**
 * @file orc_io.cpp
 * @brief Demonstrates usage of the libcudf APIs to read and write
 * orc file format with different encodings and compression types
 *
 * The following encoding and compression ztypes are demonstrated:
 * Encoding Types: DEFAULT, DICTIONARY, PLAIN, DELTA_BINARY_PACKED,
 *                 DELTA_LENGTH_BYTE_ARRAY, DELTA_BYTE_ARRAY
 *
 * Compression Types: NONE, AUTO, SNAPPY, LZ4, ZSTD
 *
 */

/**
 * @brief Read orc input from file
 *
 * @param filepath path to input orc file
 * @return cudf::io::table_with_metadata
 */
cudf::io::table_with_metadata read_orc(std::string filepath)
{
  auto source_info = cudf::io::source_info(filepath);
  auto builder     = cudf::io::orc_reader_options::builder(source_info).columns({"b"});
  auto options     = builder.build();
  return cudf::io::read_orc(options);
}

/**
 * @brief Function to print example usage and argument information.
 */
void print_usage()
{
  std::cout << "\nUsage: orc <input orc file> <output orc file> <encoding type>\n"
               "                  <compression type> <write page stats: yes/no>\n\n"
               "Available encoding types: DEFAULT, DICTIONARY, PLAIN, DELTA_BINARY_PACKED,\n"
               "                 DELTA_LENGTH_BYTE_ARRAY, DELTA_BYTE_ARRAY\n\n"
               "Available compression types: NONE, AUTO, SNAPPY, LZ4, ZSTD\n\n";
}

/**
 * @brief Main for nested_types examples
 *
 * Command line parameters:
 * 1. orc input file name/path (default: "example.orc")
 * 2. orc output file name/path (default: "output.orc")
 * 3. encoding type for columns (default: "DELTA_BINARY_PACKED")
 * 4. compression type (default: "ZSTD")
 * 5. optional: use page size stats metadata (default: "NO")
 *
 * Example invocation from directory `cudf/cpp/examples/orc_io`:
 * ./build/orc_io example.orc output.orc DELTA_BINARY_PACKED ZSTD
 *
 */
int main(int argc, char const** argv)
{
  std::string input_filepath                          = "example.orc";
  std::string output_filepath                         = "output.orc";
  cudf::io::column_encoding encoding                  = get_encoding_type("DELTA_BINARY_PACKED");
  cudf::io::compression_type compression              = get_compression_type("ZSTD");
  std::optional<cudf::io::statistics_freq> page_stats = std::nullopt;

  switch (argc) {
    case 6:
      page_stats = get_boolean(argv[5])
                     ? std::make_optional(cudf::io::statistics_freq::STATISTICS_COLUMN)
                     : std::nullopt;
      [[fallthrough]];
    case 5: compression = get_compression_type(argv[4]); [[fallthrough]];
    case 4: encoding = get_encoding_type(argv[3]); [[fallthrough]];
    case 3: output_filepath = argv[2]; [[fallthrough]];
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

  // Read input orc file
  cudf::examples::timer timer;
  auto [input, metadata] = read_orc(input_filepath);
  timer.print_elapsed_millis();

  return 0;
}
