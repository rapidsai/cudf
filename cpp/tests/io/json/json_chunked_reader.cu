/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include "json_utils.cuh"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/table_utilities.hpp>

#include <rmm/resource_ref.hpp>

#include <fstream>
#include <string>
#include <vector>

#include <filesystem>

/**
 * @brief Base test fixture for JSON reader tests
 */
struct JsonReaderTest : public cudf::test::BaseFixture {};

cudf::test::TempDirTestEnvironment* const temp_env =
  static_cast<cudf::test::TempDirTestEnvironment*>(
    ::testing::AddGlobalTestEnvironment(new cudf::test::TempDirTestEnvironment));

TEST_F(JsonReaderTest, ByteRange_SingleSource)
{
  std::string const json_string = R"(
    { "a": { "y" : 6}, "b" : [1, 2, 3], "c": 11 }
    { "a": { "y" : 6}, "b" : [4, 5   ], "c": 12 }
    { "a": { "y" : 6}, "b" : [6      ], "c": 13 }
    { "a": { "y" : 6}, "b" : [7      ], "c": 14 })";

  // Initialize parsing options (reading json lines)
  cudf::io::json_reader_options json_lines_options =
    cudf::io::json_reader_options::builder(
      cudf::io::source_info{json_string.c_str(), json_string.size()})
      .compression(cudf::io::compression_type::NONE)
      .lines(true);

  // Read full test data via existing, nested JSON lines reader
  cudf::io::table_with_metadata current_reader_table = cudf::io::read_json(json_lines_options);

  auto datasources = cudf::io::datasource::create(json_lines_options.get_source().host_buffers());

  // Test for different chunk sizes
  for (auto chunk_size : {7, 10, 15, 20, 40, 50, 100, 200, 500}) {
    auto const tables = split_byte_range_reading(datasources,
                                                 json_lines_options,
                                                 chunk_size,
                                                 cudf::get_default_stream(),
                                                 rmm::mr::get_current_device_resource());

    auto table_views = std::vector<cudf::table_view>(tables.size());
    std::transform(tables.begin(), tables.end(), table_views.begin(), [](auto& table) {
      return table.tbl->view();
    });
    auto result = cudf::concatenate(table_views);

    // Verify that the data read via chunked reader matches the data read via nested JSON reader
    // cannot use EQUAL due to concatenate removing null mask
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(current_reader_table.tbl->view(), result->view());
  }
}

TEST_F(JsonReaderTest, ReadCompleteFiles)
{
  std::string const json_string = R"(
    { "a": { "y" : 6}, "b" : [1, 2, 3], "c": 11 }
    { "a": { "y" : 6}, "b" : [4, 5   ], "c": 12 }
    { "a": { "y" : 6}, "b" : [6      ], "c": 13 }
    { "a": { "y" : 6}, "b" : [7      ], "c": 14 })";
  auto filename                 = temp_env->get_temp_dir() + "ParseInRangeIntegers.json";
  {
    std::ofstream outfile(filename, std::ofstream::out);
    outfile << json_string;
  }

  constexpr int num_sources = 5;
  std::vector<std::string> filepaths(num_sources, filename);

  cudf::io::json_reader_options in_options =
    cudf::io::json_reader_options::builder(cudf::io::source_info{filepaths})
      .lines(true)
      .recovery_mode(cudf::io::json_recovery_mode_t::FAIL);

  cudf::io::table_with_metadata result = cudf::io::read_json(in_options);

  std::vector<cudf::io::table_with_metadata> part_tables;
  for (auto filepath : filepaths) {
    cudf::io::json_reader_options part_in_options =
      cudf::io::json_reader_options::builder(cudf::io::source_info{filepath})
        .lines(true)
        .recovery_mode(cudf::io::json_recovery_mode_t::FAIL);

    part_tables.push_back(cudf::io::read_json(part_in_options));
  }

  auto part_table_views = std::vector<cudf::table_view>(part_tables.size());
  std::transform(part_tables.begin(), part_tables.end(), part_table_views.begin(), [](auto& table) {
    return table.tbl->view();
  });

  auto expected_result = cudf::concatenate(part_table_views);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(result.tbl->view(), expected_result->view());
}

TEST_F(JsonReaderTest, ByteRange_MultiSource)
{
  std::string const json_string = R"(
    { "a": { "y" : 6}, "b" : [1, 2, 3], "c": 11 }
    { "a": { "y" : 6}, "b" : [4, 5   ], "c": 12 }
    { "a": { "y" : 6}, "b" : [6      ], "c": 13 }
    { "a": { "y" : 6}, "b" : [7      ], "c": 14 })";
  auto filename                 = temp_env->get_temp_dir() + "ParseInRangeIntegers.json";
  {
    std::ofstream outfile(filename, std::ofstream::out);
    outfile << json_string;
  }

  constexpr int num_sources = 5;
  std::vector<std::string> filepaths(num_sources, filename);

  // Initialize parsing options (reading json lines)
  cudf::io::json_reader_options json_lines_options =
    cudf::io::json_reader_options::builder(cudf::io::source_info{filepaths})
      .lines(true)
      .compression(cudf::io::compression_type::NONE)
      .recovery_mode(cudf::io::json_recovery_mode_t::FAIL);

  // Read full test data via existing, nested JSON lines reader
  cudf::io::table_with_metadata current_reader_table = cudf::io::read_json(json_lines_options);

  auto file_paths = json_lines_options.get_source().filepaths();
  std::vector<std::unique_ptr<cudf::io::datasource>> datasources;
  for (auto& fp : file_paths) {
    datasources.emplace_back(cudf::io::datasource::create(fp));
  }

  // Test for different chunk sizes
  for (auto chunk_size : {7, 10, 15, 20, 40, 50, 100, 200, 500, 1000, 2000}) {
    auto const tables = split_byte_range_reading(datasources,
                                                 json_lines_options,
                                                 chunk_size,
                                                 cudf::get_default_stream(),
                                                 rmm::mr::get_current_device_resource());

    auto table_views = std::vector<cudf::table_view>(tables.size());
    std::transform(tables.begin(), tables.end(), table_views.begin(), [](auto& table) {
      return table.tbl->view();
    });
    auto result = cudf::concatenate(table_views);

    // Verify that the data read via chunked reader matches the data read via nested JSON reader
    // cannot use EQUAL due to concatenate removing null mask
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(current_reader_table.tbl->view(), result->view());
  }
}

TEST_F(JsonReaderTest, ByteRangeWithRealloc_MultiSource)
{
  std::string long_string = "haha";
  std::size_t log_repetitions = 10;
  long_string.reserve(long_string.size() * log_repetitions);
  for (std::size_t i = 0; i < log_repetitions; i++) {
    long_string += long_string;
  }

  auto json_string = [&long_string]() {
    std::string json_string = R"(
      { "a": { "y" : 6}, "b" : [1, 2, 3], "c": 11 }
      { "a": { "y" : 6}, "b" : [4, 5   ], "c": 12 }
      { "a": { "y" : 6}, "b" : [6      ], "c": 13 }
      { "a": { "y" : 6}, "b" : [7      ], "c": 14 })";
    std::string replace_chars = "c";
    std::size_t pos = json_string.find(replace_chars); 
    while (pos != std::string::npos) { 
        // Replace the substring with the specified string 
        json_string.replace(pos, replace_chars.size(), long_string); 
  
        // Find the next occurrence of the substring 
        pos = json_string.find(replace_chars, 
                         pos + long_string.size()); 
    }
    return json_string;
  }();

  auto filename                 = temp_env->get_temp_dir() + "ParseInRangeIntegers.json";
  {
    std::ofstream outfile(filename, std::ofstream::out);
    outfile << json_string;
  }

  constexpr int num_sources = 5;
  std::vector<std::string> filepaths(num_sources, filename);

  // Initialize parsing options (reading json lines)
  cudf::io::json_reader_options json_lines_options =
    cudf::io::json_reader_options::builder(cudf::io::source_info{filepaths})
      .lines(true)
      .compression(cudf::io::compression_type::NONE)
      .recovery_mode(cudf::io::json_recovery_mode_t::FAIL);

  // Read full test data via existing, nested JSON lines reader
  cudf::io::table_with_metadata current_reader_table = cudf::io::read_json(json_lines_options);

  auto file_paths = json_lines_options.get_source().filepaths();
  std::vector<std::unique_ptr<cudf::io::datasource>> datasources;
  for (auto& fp : file_paths) {
    datasources.emplace_back(cudf::io::datasource::create(fp));
  }

  // Test for different chunk sizes
  std::vector<int> chunk_sizes{7, 10, 15, 20, 40, 50, 100, 200, 500, 1000, 2000};
  for (auto chunk_size : chunk_sizes) {
    auto const tables = split_byte_range_reading(datasources,
                                                 json_lines_options,
                                                 chunk_size,
                                                 cudf::get_default_stream(),
                                                 rmm::mr::get_current_device_resource());

    auto table_views = std::vector<cudf::table_view>(tables.size());
    std::transform(tables.begin(), tables.end(), table_views.begin(), [](auto& table) {
      return table.tbl->view();
    });
    auto result = cudf::concatenate(table_views);

    // Verify that the data read via chunked reader matches the data read via nested JSON reader
    // cannot use EQUAL due to concatenate removing null mask
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(current_reader_table.tbl->view(), result->view());
    break;
  }
}

TEST_F(JsonReaderTest, ReadFromFiles) 
{
  namespace fs = std::filesystem;
  std::string dirpath = "/home/coder/datasets/adattagupta/redpajama_debug_2/";
  std::set<fs::path> filepaths;
  for(auto &entry : fs::directory_iterator(dirpath))
    filepaths.insert(entry.path());
  
  int numfiles = 27;
  std::vector<std::string> filenames;
  std::vector<uintmax_t> filesizes;
  for(auto &filepath : filepaths) {
    if(!numfiles) break;
    filenames.push_back(filepath.string());
    filesizes.push_back(fs::file_size(filepath));
    numfiles--;
  }

  /*
  filenames.erase(filenames.begin());
  filesizes.erase(filesizes.begin());
  */

  /*
  std::cout << "intmax = " << std::numeric_limits<int>::max() << std::endl;
  std::cout << "files being parsed: \n";
  uintmax_t prefsum = 0;
  for(size_t i = 0; i < filenames.size(); i++) {
    prefsum += filesizes[i];
    std::cout << filenames[i] << " " << filesizes[i] << " " << prefsum << std::endl;
  }
  */

  cudf::io::json_reader_options json_lines_options =
    cudf::io::json_reader_options::builder(cudf::io::source_info{filenames})
      .lines(true)
      .compression(cudf::io::compression_type::NONE)
      .recovery_mode(cudf::io::json_recovery_mode_t::FAIL);

  // Read full test data via existing, nested JSON lines reader
  cudf::io::table_with_metadata current_reader_table = cudf::io::read_json(json_lines_options);
}

TEST_F(JsonReaderTest, LolFromFiles) 
{
  namespace fs = std::filesystem;
  std::string filename = "/home/coder/cudf/datasets/lol.json";

  cudf::io::json_reader_options json_lines_options =
    cudf::io::json_reader_options::builder(cudf::io::source_info{filename})
      .lines(true)
      .compression(cudf::io::compression_type::NONE)
      .recovery_mode(cudf::io::json_recovery_mode_t::FAIL);

  // Read full test data via existing, nested JSON lines reader
  cudf::io::table_with_metadata current_reader_table = cudf::io::read_json(json_lines_options);
}
