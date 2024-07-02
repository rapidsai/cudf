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

#include "io/json/read_json.hpp"
#include "large_strings_fixture.hpp"

#include <cudf_test/table_utilities.hpp>

#include <cudf/concatenate.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/json.hpp>
#include <cudf/utilities/span.hpp>

struct JsonLargeReaderTest : public cudf::test::StringsLargeTest {};

// function to extract first delimiter in the string in each chunk,
// collate together and form byte_range for each chunk,
// parse separately.
std::vector<cudf::io::table_with_metadata> skeleton_for_parellel_chunk_reader(
  cudf::host_span<std::unique_ptr<cudf::io::datasource>> sources,
  cudf::io::json_reader_options const& reader_opts,
  int64_t chunk_size,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  using namespace cudf::io::json::detail;
  size_t total_source_size = 0;
  for (auto const& source : sources) {
    total_source_size += source->size();
  }
  size_t num_chunks              = (total_source_size + chunk_size - 1) / chunk_size;
  constexpr int64_t no_min_value = -1;

  // Get the first delimiter in each chunk.
  std::vector<int64_t> first_delimiter_index(num_chunks);
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
  using record_range = std::pair<int64_t, int64_t>;
  std::vector<record_range> record_ranges;
  record_ranges.reserve(num_chunks);
  first_delimiter_index[0] = 0;
  auto prev                = first_delimiter_index[0];
  for (size_t i = 1; i < num_chunks; i++) {
    if (first_delimiter_index[i] == no_min_value) continue;
    record_ranges.emplace_back(prev, first_delimiter_index[i]);
    prev = first_delimiter_index[i];
  }
  record_ranges.emplace_back(prev, total_source_size);

  std::vector<cudf::io::table_with_metadata> tables;
  // Process each chunk in parallel.
  for (auto const& [chunk_start, chunk_end] : record_ranges) {
    if (chunk_start == -1 or chunk_end == -1 or
        static_cast<size_t>(chunk_start) >= total_source_size)
      continue;
    reader_opts_chunk.set_byte_range_offset(chunk_start);
    reader_opts_chunk.set_byte_range_size(chunk_end - chunk_start);
    tables.push_back(read_json(sources, reader_opts_chunk, stream, mr));
  }
  // assume all records have same number of columns, and inferred same type. (or schema is passed)
  // TODO a step before to merge all columns, types and infer final schema.
  return tables;
}

TEST_F(JsonLargeReaderTest, MultiBatch)
{
  std::string json_string             = R"(
    { "a": { "y" : 6}, "b" : [1, 2, 3], "c": 11 }
    { "a": { "y" : 6}, "b" : [4, 5   ], "c": 12 }
    { "a": { "y" : 6}, "b" : [6      ], "c": 13 }
    { "a": { "y" : 6}, "b" : [7      ], "c": 14 })";
  constexpr size_t batch_size_ub      = std::numeric_limits<int>::max();
  constexpr size_t expected_file_size = 1.5 * static_cast<double>(batch_size_ub);
  std::size_t const log_repetitions =
    static_cast<std::size_t>(std::ceil(std::log2(expected_file_size / json_string.size())));

  json_string.reserve(json_string.size() * (1UL << log_repetitions));
  for (std::size_t i = 0; i < log_repetitions; i++) {
    json_string += json_string;
  }

  constexpr int num_sources = 2;
  std::vector<cudf::host_span<std::byte>> hostbufs(
    num_sources,
    cudf::host_span<std::byte>(reinterpret_cast<std::byte*>(json_string.data()),
                               json_string.size()));

  // Initialize parsing options (reading json lines)
  cudf::io::json_reader_options json_lines_options =
    cudf::io::json_reader_options::builder(
      cudf::io::source_info{
        cudf::host_span<cudf::host_span<std::byte>>(hostbufs.data(), hostbufs.size())})
      .lines(true)
      .compression(cudf::io::compression_type::NONE)
      .recovery_mode(cudf::io::json_recovery_mode_t::FAIL);

  // Read full test data via existing, nested JSON lines reader
  cudf::io::table_with_metadata current_reader_table = cudf::io::read_json(json_lines_options);

  std::vector<std::unique_ptr<cudf::io::datasource>> datasources;
  for (auto& hb : hostbufs) {
    datasources.emplace_back(cudf::io::datasource::create(hb));
  }
  // Test for different chunk sizes
  // std::vector<size_t> chunk_sizes {5000, 10000, 20000, batch_size_ub,
  // static_cast<size_t>(batch_size_ub * 2)};
  std::vector<size_t> chunk_sizes{
    batch_size_ub / 4, batch_size_ub / 2, batch_size_ub, static_cast<size_t>(batch_size_ub * 2)};
  for (auto chunk_size : chunk_sizes) {
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
