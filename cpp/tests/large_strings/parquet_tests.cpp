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
