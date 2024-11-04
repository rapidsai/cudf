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

#include <cudf_test/table_utilities.hpp>

#include <cudf/concatenate.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table_view.hpp>

namespace {

cudf::test::TempDirTestEnvironment* const g_temp_env =
  static_cast<cudf::test::TempDirTestEnvironment*>(
    ::testing::AddGlobalTestEnvironment(new cudf::test::TempDirTestEnvironment));

}  // namespace

struct ParquetStringsTest : public cudf::test::StringsLargeTest {};

TEST_F(ParquetStringsTest, ReadLargeStrings)
{
  // need to create a string column larger than `threshold`
  auto const col0        = this->long_column();
  auto const column_size = cudf::strings_column_view(col0).chars_size(cudf::get_default_stream());
  auto const threshold   = column_size - 1;
  auto const expected    = cudf::table_view{{col0, col0, col0}};

  auto expected_metadata = cudf::io::table_input_metadata{expected};
  expected_metadata.column_metadata[1].set_encoding(
    cudf::io::column_encoding::DELTA_LENGTH_BYTE_ARRAY);
  expected_metadata.column_metadata[2].set_encoding(cudf::io::column_encoding::DELTA_BYTE_ARRAY);

  // set smaller threshold to reduce file size and execution time
  setenv("LIBCUDF_LARGE_STRINGS_THRESHOLD", std::to_string(threshold).c_str(), 1);

  auto const filepath = g_temp_env->get_temp_filepath("ReadLargeStrings.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .compression(cudf::io::compression_type::ZSTD)
      .stats_level(cudf::io::STATISTICS_NONE)
      .metadata(expected_metadata);
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options default_in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto const result      = cudf::io::read_parquet(default_in_opts);
  auto const result_view = result.tbl->view();
  for (auto cv : result_view) {
    auto const offsets = cudf::strings_column_view(cv).offsets();
    EXPECT_EQ(offsets.type(), cudf::data_type{cudf::type_id::INT64});
  }
  CUDF_TEST_EXPECT_TABLES_EQUAL(result_view, expected);

  // go back to normal threshold
  unsetenv("LIBCUDF_LARGE_STRINGS_THRESHOLD");
}

TEST_F(ParquetStringsTest, ChunkedReadLargeStrings)
{
  // Construct a table with one large strings column > 2GB
  auto const wide = this->wide_column();
  auto input      = cudf::concatenate(std::vector<cudf::column_view>(120000, wide));  ///< 230MB

  int constexpr multiplier = 12;
  std::vector<cudf::column_view> input_cols(multiplier, input->view());
  auto col0 = cudf::concatenate(input_cols);  ///< 2.70GB

  // Expected table
  auto const expected    = cudf::table_view{{col0->view()}};
  auto expected_metadata = cudf::io::table_input_metadata{expected};
  expected_metadata.column_metadata[0].set_encoding(
    cudf::io::column_encoding::DELTA_LENGTH_BYTE_ARRAY);

  // Write to Parquet
  auto const filepath = g_temp_env->get_temp_filepath("ChunkedReadLargeStrings.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .compression(cudf::io::compression_type::ZSTD)
      .stats_level(cudf::io::STATISTICS_NONE)
      .metadata(expected_metadata);
  cudf::io::write_parquet(out_opts);

  // Read with chunked_parquet_reader
  size_t constexpr pass_read_limit =
    size_t{8} * 1024 * 1024 *
    1024;  ///< Set to 8GB so we read almost entire table (>2GB string) in the  first subpass
           ///< and only a small amount in the second subpass.
  cudf::io::parquet_reader_options default_in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto reader = cudf::io::chunked_parquet_reader(0, pass_read_limit, default_in_opts);

  auto tables = std::vector<std::unique_ptr<cudf::table>>{};
  while (reader.has_next()) {
    tables.emplace_back(reader.read_chunk().tbl);
  }
  auto table_views = std::vector<cudf::table_view>{};
  std::transform(tables.begin(), tables.end(), std::back_inserter(table_views), [](auto& tbl) {
    return tbl->view();
  });
  auto result            = cudf::concatenate(table_views);
  auto const result_view = result->view();

  // Verify
  for (auto const& cv : result_view) {
    auto const offsets = cudf::strings_column_view(cv).offsets();
    EXPECT_EQ(offsets.type(), cudf::data_type{cudf::type_id::INT64});
  }
  EXPECT_EQ(tables.size(), 2);
  CUDF_TEST_EXPECT_TABLES_EQUAL(result_view, expected);
}
