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

#include "large_strings_fixture.hpp"

#include <cudf/io/json.hpp>

#include <fstream>

namespace {

cudf::test::TempDirTestEnvironment* const g_temp_env =
  static_cast<cudf::test::TempDirTestEnvironment*>(
    ::testing::AddGlobalTestEnvironment(new cudf::test::TempDirTestEnvironment));

}  // namespace

struct JsonLargeReaderTest : public cudf::test::StringsLargeTest {};

TEST_F(JsonLargeReaderTest, MultiBatch)
{
  std::string json_string             = R"(
    { "a": { "y" : 6}, "b" : [1, 2, 3], "c": 11 }
    { "a": { "y" : 6}, "b" : [4, 5   ], "c": 12 }
    { "a": { "y" : 6}, "b" : [6      ], "c": 13 }
    { "a": { "y" : 6}, "b" : [7      ], "c": 14 })";
  constexpr size_t expected_file_size = std::numeric_limits<int>::max() / 2;
  std::size_t const log_repetitions =
    static_cast<std::size_t>(std::ceil(std::log2(expected_file_size / json_string.size())));

  json_string.reserve(json_string.size() * (1UL << log_repetitions));
  std::size_t numrows = 4;
  for (std::size_t i = 0; i < log_repetitions; i++) {
    json_string += json_string;
    numrows <<= 1;
  }

  auto filename = g_temp_env->get_temp_dir() + "LargeishJSONFile.json";
  std::ofstream outfile(filename, std::ofstream::out);
  outfile << json_string;

  constexpr int num_sources = 10;
  std::vector<std::string> filepaths(num_sources, filename);

  // Initialize parsing options (reading json lines)
  cudf::io::json_reader_options json_lines_options =
    cudf::io::json_reader_options::builder(cudf::io::source_info{filepaths})
      .lines(true)
      .compression(cudf::io::compression_type::NONE)
      .recovery_mode(cudf::io::json_recovery_mode_t::FAIL);

  // Read full test data via existing, nested JSON lines reader
  cudf::io::table_with_metadata current_reader_table = cudf::io::read_json(json_lines_options);
  ASSERT_EQ(current_reader_table.tbl->num_rows(), numrows * num_sources);
}
