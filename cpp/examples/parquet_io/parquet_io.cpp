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

#include "parquet_io.hpp"

/**
 * @file parquet_io.cpp
 * @brief Demonstrates usage of the libcudf APIs to read and write
 * parquet file format with different encodings and compression types
 *
 * The following encoding and compression ztypes are demonstrated:
 * Encoding Types: DEFAULT, DICTIONARY, PLAIN, DELTA_BINARY_PACKED,
 *                 DELTA_LENGTH_BYTE_ARRAY, DELTA_BYTE_ARRAY
 *
 * Compression Types: NONE, AUTO, SNAPPY, BZIP2, BROTLI, ZIP, XZ,
 *                    ZLIB, LZ4, LZO, ZSTD
 *
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
 * @brief Write parquet output to file
 *
 * @param input table to write
 * @param metadata metadata of input table read by parquet reader
 * @param filepath path to output parquet file
 */
void write_parquet(cudf::table_view input,
                   cudf::io::table_metadata metadata,
                   std::string filepath,
                   cudf::io::column_encoding encoding,
                   cudf::io::compression_type compression)
{
  // write the data for inspection
  auto sink_info      = cudf::io::sink_info(filepath);
  auto builder        = cudf::io::parquet_writer_options::builder(sink_info, input);
  auto table_metadata = cudf::io::table_input_metadata{metadata};

  std::for_each(table_metadata.column_metadata.begin(),
                table_metadata.column_metadata.end(),
                [=](auto& col_meta) { col_meta.set_encoding(encoding); });

  builder.metadata(table_metadata);
  auto options = builder.build();
  options.set_compression(compression);
  cudf::io::write_parquet(options);
}

/**
 * @brief Main for nested_types examples
 *
 * Command line parameters:
 * 1. parquet input file name/path (default: "example.parquet")
 * 2. parquet output file name/path (default: "output.parquet")
 * 3. encoding type for columns (default: "DELTA_BINARY_PACKED")
 * 4. compression type (default: "ZSTD")
 *
 * Example invocation from directory `cudf/cpp/examples/parquet_io`:
 * ./build/parquet_io example.parquet output.parquet DELTA_BINARY_PACKED ZSTD
 *
 */
int main(int argc, char const** argv)
{
  std::string input_filepath;
  std::string output_filepath;
  cudf::io::column_encoding encoding;
  cudf::io::compression_type compression;

  // minimal argument parsing
  if (argc != 5 && argc != 1) {
    std::cout << "Either provide all command-line arguments, or none to use defaults" << std::endl;
    return 1;
  }
  if (argc == 1) {
    input_filepath  = "example.parquet";
    output_filepath = "output.parquet";
    encoding        = get_encoding_type("DELTA_BINARY_PACKED");
    compression     = get_compression_type("ZSTD");
  } else {
    input_filepath  = argv[1];
    output_filepath = argv[2];
    encoding        = get_encoding_type(argv[3]);
    compression     = get_compression_type(argv[4]);
  };

  // create and use a memory pool
  bool pool     = true;
  auto resource = create_memory_resource(pool);
  rmm::mr::set_current_device_resource(resource.get());

  // timer is automatically started here
  Timer timer;
  // read input parquet file
  std::cout << "Reading " << input_filepath << "..." << std::endl;
  auto [input, metadata] = read_parquet(input_filepath);
  timer.stop();
  timer.print_elapsed_millis();

  // manually restart the timer now
  timer.start();
  // write parquet file with the specified encoding and compression
  std::cout << "Writing " << output_filepath << " with encoding and compression..." << std::endl;
  write_parquet(input->view(), metadata, output_filepath, encoding, compression);
  timer.stop();
  timer.print_elapsed_millis();

  // restart the timer
  timer.start();
  // read the parquet file written with encoding and compression
  std::cout << "Reading " << output_filepath << "..." << std::endl;
  auto [transcoded_input, transcoded_metadata] = read_parquet(output_filepath);
  timer.stop();
  timer.print_elapsed_millis();

  // check for validity
  bool valid = true;
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> indices;
  try {
    // left anti-join the original and transcoded tables
    // identical tables should not throw an excpetion and
    // retun an empty indices vector
    indices = cudf::left_anti_join(
      input->view(), transcoded_input->view(), cudf::null_equality::EQUAL, resource.get());
  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl << std::endl;
    std::cout << "Transcoding valid: false" << std::endl;
    valid = false;
  }

  // no exception thrown, check ofr indices->size
  if (valid) {
    bool valid = indices->size() == 0;
    std::cout << "Transcoding valid: " << std::boolalpha << valid << std::endl;
  }

  return 0;
}
