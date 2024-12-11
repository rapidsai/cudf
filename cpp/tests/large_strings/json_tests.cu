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

#include "../io/json/json_utils.cuh"
#include "io/comp/comp.hpp"
#include "large_strings_fixture.hpp"

#include <cudf_test/table_utilities.hpp>

#include <cudf/concatenate.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/json.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

struct JsonLargeReaderTest : public cudf::test::StringsLargeTest,
                             public testing::WithParamInterface<cudf::io::compression_type> {};

// Parametrize qualifying JSON tests for multiple compression types
INSTANTIATE_TEST_SUITE_P(JsonLargeReaderTest,
                         JsonLargeReaderTest,
                         ::testing::Values(cudf::io::compression_type::GZIP,
                                           cudf::io::compression_type::NONE));

TEST_P(JsonLargeReaderTest, MultiBatch)
{
  cudf::io::compression_type const comptype = GetParam();

  std::string json_string = R"(
    { "a": { "y" : 6}, "b" : [1, 2, 3], "c": 11 }
    { "a": { "y" : 6}, "b" : [4, 5   ], "c": 12 }
    { "a": { "y" : 6}, "b" : [6      ], "c": 13 }
    { "a": { "y" : 6}, "b" : [7      ], "c": 14 })";

  std::size_t const batch_size_upper_bound = std::numeric_limits<int32_t>::max() / 16;
  // set smaller batch_size to reduce file size and execution time
  setenv("LIBCUDF_JSON_BATCH_SIZE", std::to_string(batch_size_upper_bound).c_str(), 1);

  constexpr std::size_t expected_file_size = 1.5 * static_cast<double>(batch_size_upper_bound);
  std::size_t const log_repetitions =
    static_cast<std::size_t>(std::ceil(std::log2(expected_file_size / json_string.size())));

  json_string.reserve(json_string.size() * (1UL << log_repetitions));
  for (std::size_t i = 0; i < log_repetitions; i++) {
    json_string += json_string;
  }

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

  constexpr int num_sources = 2;
  std::vector<cudf::host_span<std::byte>> hostbufs(
    num_sources,
    cudf::host_span<std::byte>(reinterpret_cast<std::byte*>(json_string.data()),
                               json_string.size()));
  std::vector<cudf::host_span<std::byte>> chostbufs(
    num_sources,
    cudf::host_span<std::byte>(reinterpret_cast<std::byte*>(cdata.data()), cdata.size()));

  // Initialize parsing options (reading json lines)
  cudf::io::json_reader_options json_lines_options =
    cudf::io::json_reader_options::builder(
      cudf::io::source_info{
        cudf::host_span<cudf::host_span<std::byte>>(hostbufs.data(), hostbufs.size())})
      .lines(true)
      .compression(cudf::io::compression_type::NONE)
      .recovery_mode(cudf::io::json_recovery_mode_t::FAIL);
  cudf::io::json_reader_options cjson_lines_options =
    cudf::io::json_reader_options::builder(
      cudf::io::source_info{
        cudf::host_span<cudf::host_span<std::byte>>(chostbufs.data(), chostbufs.size())})
      .lines(true)
      .compression(comptype)
      .recovery_mode(cudf::io::json_recovery_mode_t::FAIL);

  // Read full test data via existing, nested JSON lines reader
  cudf::io::table_with_metadata current_reader_table = cudf::io::read_json(cjson_lines_options);

  auto datasources  = cudf::io::datasource::create(json_lines_options.get_source().host_buffers());
  auto cdatasources = cudf::io::datasource::create(cjson_lines_options.get_source().host_buffers());

  // Test for different chunk sizes
  std::vector<std::size_t> chunk_sizes{batch_size_upper_bound / 4,
                                       batch_size_upper_bound / 2,
                                       batch_size_upper_bound,
                                       static_cast<std::size_t>(batch_size_upper_bound * 2)};

  for (auto chunk_size : chunk_sizes) {
    auto const tables =
      split_byte_range_reading<std::int64_t>(datasources,
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

  // go back to normal batch_size
  unsetenv("LIBCUDF_JSON_BATCH_SIZE");
}
