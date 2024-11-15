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

#include "io/comp/comp.hpp"
#include "json_utils.cuh"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf/utilities/memory_resource.hpp>

#include <fstream>
#include <string>
#include <vector>

/**
 * @brief Base test fixture for JSON reader tests
 */
struct JsonReaderTest : public cudf::test::BaseFixture,
                        public testing::WithParamInterface<cudf::io::compression_type> {};

// Parametrize qualifying JSON tests for multiple compression types
INSTANTIATE_TEST_SUITE_P(JsonReaderTest,
                         JsonReaderTest,
                         ::testing::Values(cudf::io::compression_type::GZIP,
                                           cudf::io::compression_type::SNAPPY,
                                           cudf::io::compression_type::NONE));

cudf::test::TempDirTestEnvironment* const temp_env =
  static_cast<cudf::test::TempDirTestEnvironment*>(
    ::testing::AddGlobalTestEnvironment(new cudf::test::TempDirTestEnvironment));

TEST_P(JsonReaderTest, ByteRange_SingleSource)
{
  cudf::io::compression_type const comptype = GetParam();

  std::string const json_string = R"(
    { "a": { "y" : 6}, "b" : [1, 2, 3], "c": 11 }
    { "a": { "y" : 6}, "b" : [4, 5   ], "c": 12 }
    { "a": { "y" : 6}, "b" : [6      ], "c": 13 }
    { "a": { "y" : 6}, "b" : [7      ], "c": 14 })";

  std::vector<std::uint8_t> cdata;
  if (comptype != cudf::io::compression_type::NONE) {
    cdata = cudf::io::detail::compress(
      comptype,
      cudf::host_span<uint8_t const>(reinterpret_cast<uint8_t const*>(json_string.data()),
                                     json_string.size()),
      cudf::get_default_stream());
  } else
    cdata = std::vector<uint8_t>(
      reinterpret_cast<uint8_t const*>(json_string.data()),
      reinterpret_cast<uint8_t const*>(json_string.data()) + json_string.size());

  // Initialize parsing options (reading json lines)
  cudf::io::json_reader_options json_lines_options =
    cudf::io::json_reader_options::builder(
      cudf::io::source_info{json_string.c_str(), json_string.size()})
      .compression(cudf::io::compression_type::NONE)
      .lines(true);
  cudf::io::json_reader_options cjson_lines_options =
    cudf::io::json_reader_options::builder(
      cudf::io::source_info{cudf::host_span<uint8_t>(cdata.data(), cdata.size())})
      .compression(comptype)
      .lines(true);

  // Read full test data via existing, nested JSON lines reader
  cudf::io::table_with_metadata current_reader_table = cudf::io::read_json(cjson_lines_options);

  auto datasources  = cudf::io::datasource::create(json_lines_options.get_source().host_buffers());
  auto cdatasources = cudf::io::datasource::create(cjson_lines_options.get_source().host_buffers());

  // Test for different chunk sizes
  for (auto chunk_size : {7, 10, 15, 20, 40, 50, 100, 200, 500}) {
    auto const tables = split_byte_range_reading(datasources,
                                                 cdatasources,
                                                 json_lines_options,
                                                 cjson_lines_options,
                                                 chunk_size,
                                                 cudf::get_default_stream(),
                                                 cudf::get_current_device_resource_ref());

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

TEST_P(JsonReaderTest, ReadCompleteFiles)
{
  cudf::io::compression_type const comptype = GetParam();

  std::string const json_string = R"(
    { "a": { "y" : 6}, "b" : [1, 2, 3], "c": 11 }
    { "a": { "y" : 6}, "b" : [4, 5   ], "c": 12 }
    { "a": { "y" : 6}, "b" : [6      ], "c": 13 }
    { "a": { "y" : 6}, "b" : [7      ], "c": 14 })";

  std::vector<std::uint8_t> cdata;
  if (comptype != cudf::io::compression_type::NONE) {
    cdata = cudf::io::detail::compress(
      comptype,
      cudf::host_span<uint8_t const>(reinterpret_cast<uint8_t const*>(json_string.data()),
                                     json_string.size()),
      cudf::get_default_stream());
  } else
    cdata = std::vector<uint8_t>(
      reinterpret_cast<uint8_t const*>(json_string.data()),
      reinterpret_cast<uint8_t const*>(json_string.data()) + json_string.size());

  auto cfilename = temp_env->get_temp_dir() + "cParseInRangeIntegers.json";
  {
    std::ofstream outfile(cfilename, std::ofstream::out);
    std::copy(cdata.begin(), cdata.end(), std::ostreambuf_iterator<char>(outfile));
  }

  constexpr int num_sources = 5;
  std::vector<std::string> cfilepaths(num_sources, cfilename);

  cudf::io::json_reader_options cin_options =
    cudf::io::json_reader_options::builder(cudf::io::source_info{cfilepaths})
      .lines(true)
      .compression(comptype)
      .recovery_mode(cudf::io::json_recovery_mode_t::FAIL);

  cudf::io::table_with_metadata result = cudf::io::read_json(cin_options);

  std::vector<cudf::io::table_with_metadata> part_tables;
  for (auto cfilepath : cfilepaths) {
    cudf::io::json_reader_options part_cin_options =
      cudf::io::json_reader_options::builder(cudf::io::source_info{cfilepath})
        .lines(true)
        .compression(comptype)
        .recovery_mode(cudf::io::json_recovery_mode_t::FAIL);

    part_tables.push_back(cudf::io::read_json(part_cin_options));
  }

  auto part_table_views = std::vector<cudf::table_view>(part_tables.size());
  std::transform(part_tables.begin(), part_tables.end(), part_table_views.begin(), [](auto& table) {
    return table.tbl->view();
  });

  auto expected_result = cudf::concatenate(part_table_views);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(result.tbl->view(), expected_result->view());
}

TEST_P(JsonReaderTest, ByteRange_MultiSource)
{
  cudf::io::compression_type const comptype = GetParam();

  std::string const json_string = R"(
    { "a": { "y" : 6}, "b" : [1, 2, 3], "c": 11 }
    { "a": { "y" : 6}, "b" : [4, 5   ], "c": 12 }
    { "a": { "y" : 6}, "b" : [6      ], "c": 13 }
    { "a": { "y" : 6}, "b" : [7      ], "c": 14 })";

  std::vector<std::uint8_t> cdata;
  if (comptype != cudf::io::compression_type::NONE) {
    cdata = cudf::io::detail::compress(
      comptype,
      cudf::host_span<uint8_t const>(reinterpret_cast<uint8_t const*>(json_string.data()),
                                     json_string.size()),
      cudf::get_default_stream());
  } else
    cdata = std::vector<uint8_t>(
      reinterpret_cast<uint8_t const*>(json_string.data()),
      reinterpret_cast<uint8_t const*>(json_string.data()) + json_string.size());

  auto cfilename = temp_env->get_temp_dir() + "cParseInRangeIntegers.json";
  {
    std::ofstream outfile(cfilename, std::ofstream::out);
    std::copy(cdata.begin(), cdata.end(), std::ostreambuf_iterator<char>(outfile));
  }

  constexpr int num_sources = 5;
  std::vector<std::string> cfilepaths(num_sources, cfilename);
  std::vector<cudf::host_span<char const>> hostbufs(
    num_sources,
    cudf::host_span<char const>(reinterpret_cast<char const*>(json_string.data()),
                                json_string.size()));

  // Initialize parsing options (reading json lines)
  cudf::io::json_reader_options json_lines_options =
    cudf::io::json_reader_options::builder(
      cudf::io::source_info{
        cudf::host_span<cudf::host_span<char const>>(hostbufs.data(), hostbufs.size())})
      .compression(cudf::io::compression_type::NONE)
      .lines(true);
  cudf::io::json_reader_options cjson_lines_options =
    cudf::io::json_reader_options::builder(cudf::io::source_info{cfilepaths})
      .lines(true)
      .compression(comptype)
      .recovery_mode(cudf::io::json_recovery_mode_t::FAIL);

  // Read full test data via existing, nested JSON lines reader
  cudf::io::table_with_metadata current_reader_table = cudf::io::read_json(cjson_lines_options);

  std::vector<std::unique_ptr<cudf::io::datasource>> cdatasources;
  for (auto& fp : cfilepaths) {
    cdatasources.emplace_back(cudf::io::datasource::create(fp));
  }
  auto datasources = cudf::io::datasource::create(json_lines_options.get_source().host_buffers());

  // Test for different chunk sizes
  for (auto chunk_size : {7, 10, 15, 20, 40, 50, 100, 200, 500, 1000, 2000}) {
    auto const tables = split_byte_range_reading(datasources,
                                                 cdatasources,
                                                 json_lines_options,
                                                 cjson_lines_options,
                                                 chunk_size,
                                                 cudf::get_default_stream(),
                                                 cudf::get_current_device_resource_ref());

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
