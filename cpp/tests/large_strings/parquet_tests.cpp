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

#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>

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
  constexpr int string_width   = 1'024;
  constexpr size_t column_size = 512 * 1'024 * 1'024U;
  constexpr size_t threshold   = column_size - 1;
  constexpr int nrows          = column_size / string_width;

  auto elements       = thrust::constant_iterator<std::string_view>(std::string(string_width, 'a'));
  auto const col0     = cudf::test::strings_column_wrapper(elements, elements + nrows);
  auto const expected = cudf::table_view{{col0, col0, col0}};

  auto expected_metadata = cudf::io::table_input_metadata{expected};
  expected_metadata.column_metadata[1].set_encoding(
    cudf::io::column_encoding::DELTA_LENGTH_BYTE_ARRAY);
  expected_metadata.column_metadata[2].set_encoding(cudf::io::column_encoding::DELTA_BYTE_ARRAY);

  // set smaller threshold to reduce file size and execution time
  setenv("LIBCUDF_LARGE_STRINGS_THRESHOLD", std::to_string(threshold).c_str(), 1);

  auto const filepath = g_temp_env->get_temp_filepath("ReadLargeStrings.parquet");
  // set row group size so that there will be only one row group
  // no compression so we can easily read page data
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .compression(cudf::io::compression_type::ZSTD)
      .stats_level(cudf::io::STATISTICS_NONE)
      .metadata(expected_metadata);
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options default_in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto const result = cudf::io::read_parquet(default_in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(result.tbl->view(), expected);

  // go back to normal threshold
  unsetenv("LIBCUDF_LARGE_STRINGS_THRESHOLD");
}
