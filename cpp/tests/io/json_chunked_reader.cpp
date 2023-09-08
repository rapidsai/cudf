/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/table_utilities.hpp>

#include <io/json/read_json.hpp>

/**
 * @brief Base test fixture for JSON reader tests
 */
struct JsonReaderTest : public cudf::test::BaseFixture {};

// function to extract first delimiter in the string in each chunk,
// collate together and form byte_range for each chunk,
// parse separately.
std::vector<cudf::io::table_with_metadata> skeleton_for_parellel_chunk_reader(
  cudf::host_span<std::unique_ptr<cudf::io::datasource>> sources,
  cudf::io::json_reader_options const& reader_opts,
  int32_t chunk_size,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  using namespace cudf::io::json::detail;
  using cudf::size_type;
  // assuming single source.
  size_t total_source_size = 0;
  for (auto const& source : sources) {
    total_source_size += source->size();
  }
  size_t num_chunks                = (total_source_size + chunk_size - 1) / chunk_size;
  constexpr size_type no_min_value = -1;

  // Get the first delimiter in each chunk.
  std::vector<size_type> first_delimiter_index(num_chunks);
  auto reader_opts_chunk = reader_opts;
  for (size_t i = 0; i < num_chunks; i++) {
    auto const chunk_start = i * chunk_size;
    reader_opts_chunk.set_byte_range_offset(chunk_start);
    reader_opts_chunk.set_byte_range_size(chunk_size);
    first_delimiter_index[i] =
      find_first_delimiter_in_chunk(sources, reader_opts_chunk, '\n', stream);
    if (first_delimiter_index[i] != no_min_value) { first_delimiter_index[i] += chunk_start; }
  }

  // Process and allocate record start, end for each worker.
  using record_range = std::pair<size_type, size_type>;
  std::vector<record_range> record_ranges;
  record_ranges.reserve(num_chunks);
  first_delimiter_index[0] = 0;
  auto prev                = first_delimiter_index[0];
  for (size_t i = 1; i < num_chunks; i++) {
    if (first_delimiter_index[i] == no_min_value) continue;
    record_ranges.push_back({prev, first_delimiter_index[i]});
    prev = first_delimiter_index[i];
  }
  record_ranges.push_back({prev, total_source_size});

  std::vector<cudf::io::table_with_metadata> tables;
  // Process each chunk in parallel.
  for (auto const& [chunk_start, chunk_end] : record_ranges) {
    if (chunk_start == -1 or chunk_end == -1) continue;
    reader_opts_chunk.set_byte_range_offset(chunk_start);
    reader_opts_chunk.set_byte_range_size(chunk_end - chunk_start);
    tables.push_back(read_json(sources, reader_opts_chunk, stream, mr));
  }
  // assume all records have same number of columns, and inferred same type. (or schema is passed)
  // TODO a step before to merge all columns, types and infer final schema.
  return tables;
}

TEST_F(JsonReaderTest, ByteRange)
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
    auto const tables = skeleton_for_parellel_chunk_reader(datasources,
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
