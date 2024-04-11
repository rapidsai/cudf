/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/io/arrow_io_source.hpp>
#include <cudf/io/json.hpp>
#include <cudf/io/parquet.hpp>

#include <arrow/filesystem/filesystem.h>
#include <arrow/filesystem/s3fs.h>
#include <arrow/io/api.h>
#include <arrow/util/config.h>

#include <fstream>
#include <memory>
#include <string>

// Global environment for temporary files
auto const temp_env = static_cast<cudf::test::TempDirTestEnvironment*>(
  ::testing::AddGlobalTestEnvironment(new cudf::test::TempDirTestEnvironment));

// Base test fixture for tests
struct ArrowIOTest : public cudf::test::BaseFixture {};

TEST_F(ArrowIOTest, URIFileSystem)
{
  const std::string file_name = temp_env->get_temp_dir() + "JsonLinesFileTest.json";
  std::ofstream outfile(file_name, std::ofstream::out);
  outfile << "{\"a\":11, \"b\":1.1}\n{\"a\":22, \"b\":2.2}";
  outfile.close();

  std::string file_uri = "file://" + file_name;
  auto datasource      = std::make_unique<cudf::io::arrow_io_source>(file_uri);

  // Populate the JSON Reader Options
  cudf::io::json_reader_options options =
    cudf::io::json_reader_options::builder(cudf::io::source_info(datasource.get())).lines(true);

  // Read the JSON file from the LocalFileSystem
  cudf::io::table_with_metadata tbl = cudf::io::read_json(options);

  ASSERT_EQ(2, tbl.tbl->num_columns());
  ASSERT_EQ(2, tbl.tbl->num_rows());
}

TEST_F(ArrowIOTest, S3FileSystem)
{
  std::string s3_uri = "s3://rapidsai-data/cudf/test/tips.parquet?region=us-east-2";

  // Check to see if Arrow was built with support for S3. If not, ensure this
  // test throws. If so, validate the S3 file contents.
  auto const s3_unsupported = arrow::fs::FileSystemFromUri(s3_uri).status().IsNotImplemented();
  if (s3_unsupported) {
    EXPECT_THROW(std::make_unique<cudf::io::arrow_io_source>(s3_uri), cudf::logic_error);
  } else {
    auto datasource = std::make_unique<cudf::io::arrow_io_source>(s3_uri);

    // Populate the Parquet Reader Options
    cudf::io::source_info src(datasource.get());
    std::vector<std::string> single_column;
    single_column.insert(single_column.begin(), "total_bill");
    cudf::io::parquet_reader_options_builder builder(src);
    cudf::io::parquet_reader_options options = builder.columns(single_column).build();

    // Read the Parquet file from S3
    cudf::io::table_with_metadata tbl = cudf::io::read_parquet(options);

    ASSERT_EQ(1, tbl.tbl->num_columns());  // Only single column specified in reader_options
    ASSERT_EQ(244, tbl.tbl->num_rows());   // known number of rows from the S3 file
  }

#ifdef ARROW_S3
  if (!s3_unsupported) {
    // Verify that we are using Arrow with S3, and call finalize
    // https://github.com/apache/arrow/issues/36974
    // This needs to be in a separate conditional to ensure we call
    // finalize after all arrow_io_source instances have been deleted.
    [[maybe_unused]] auto _ = arrow::fs::EnsureS3Finalized();
  }
#endif
}

CUDF_TEST_PROGRAM_MAIN()
