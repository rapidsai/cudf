/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/cudf_gtest.hpp>
#include <tests/utilities/type_lists.hpp>
#include <tests/utilities/base_fixture.hpp>

#include <cudf/io/functions.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <cudf/io/functions.hpp>

#include <arrow/io/api.h>

#include <fstream>

#include <type_traits>

#define wrapper cudf::test::fixed_width_column_wrapper
using float_wrapper = wrapper<float>;
using float64_wrapper = wrapper<double>;
using int_wrapper = wrapper<int>;
using int8_wrapper = wrapper<int8_t>;
using int64_wrapper = wrapper<int64_t>;

namespace cudf_io = cudf::experimental::io;

cudf::test::TempDirTestEnvironment* const temp_env = static_cast<cudf::test::TempDirTestEnvironment*>(
    ::testing::AddGlobalTestEnvironment(new cudf::test::TempDirTestEnvironment));

/**
 * @brief Base test fixture for JSON reader tests
 **/
struct JsonReaderTest : public cudf::test::BaseFixture {};

TEST_F(JsonReaderTest, BasicJsonLines) {
  std::string data = "[1, 1.1]\n[2, 2.2]\n[3, 3.3]\n";
  
  // const auto df = cudf::read_json(args);
  cudf_io::read_json_args in_args{cudf_io::source_info{data.data(), data.size()}};
  in_args.lines = true;
  in_args.dtype = {"int", "float64"};
  cudf_io::table_with_metadata result = cudf_io::read_json(in_args);

  EXPECT_EQ(result.tbl->num_columns(), 2);
  EXPECT_EQ(result.tbl->num_rows(), 3);

  EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::INT32);
  EXPECT_EQ(result.tbl->get_column(1).type().id(), cudf::FLOAT64);
    
  EXPECT_EQ(result.metadata.column_names[0], "0");
  EXPECT_EQ(result.metadata.column_names[1], "1");

  auto validity = cudf::test::make_counting_transform_iterator(
      0, [](auto i) { return true; });

  cudf::test::expect_columns_equal(result.tbl->get_column(0), int_wrapper{{1, 2, 3}, validity});
  cudf::test::expect_columns_equal(result.tbl->get_column(1), float64_wrapper{{1.1, 2.2, 3.3}, validity});  
}

TEST_F(JsonReaderTest, JsonLinesStrings) {
  std::string data = "[1, 1.1, \"aa \"]\n[2, 2.2, \"  bbb\"]";

  cudf_io::read_json_args in_args{cudf_io::source_info{data.data(), data.size()}};
  in_args.lines = true;
  in_args.dtype = {"2:str", "0:int", "1:float64"};

  cudf_io::table_with_metadata result = cudf_io::read_json(in_args);

  EXPECT_EQ(result.tbl->num_columns(), 3);
  EXPECT_EQ(result.tbl->num_rows(), 2);

  EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::INT32);
  EXPECT_EQ(result.tbl->get_column(1).type().id(), cudf::FLOAT64);
  EXPECT_EQ(result.tbl->get_column(2).type().id(), cudf::STRING);

  EXPECT_EQ(result.metadata.column_names[0], "0");
  EXPECT_EQ(result.metadata.column_names[1], "1");
  EXPECT_EQ(result.metadata.column_names[2], "2");

  auto validity = cudf::test::make_counting_transform_iterator(
      0, [](auto i) { return true; });

  cudf::test::expect_columns_equal(result.tbl->get_column(0), int_wrapper{{1, 2}, validity});
  cudf::test::expect_columns_equal(result.tbl->get_column(1), float64_wrapper{{1.1, 2.2}, validity});  
  cudf::test::expect_columns_equal(result.tbl->get_column(2), cudf::test::strings_column_wrapper({"aa ", "  bbb"}));
}

TEST_F(JsonReaderTest, JsonLinesDtypeInference) {
  std::string data = "[100, 1.1, \"aa \"]\n[200, 2.2, \"  bbb\"]";

  cudf_io::read_json_args in_args{cudf_io::source_info{data.data(), data.size()}};
  in_args.lines = true;

  cudf_io::table_with_metadata result = cudf_io::read_json(in_args);

  EXPECT_EQ(result.tbl->num_columns(), 3);
  EXPECT_EQ(result.tbl->num_rows(), 2);

  EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::INT64);
  EXPECT_EQ(result.tbl->get_column(1).type().id(), cudf::FLOAT64);

  EXPECT_EQ(std::string(result.metadata.column_names[0]), "0");
  EXPECT_EQ(std::string(result.metadata.column_names[1]), "1");
  EXPECT_EQ(std::string(result.metadata.column_names[2]), "2");

  auto validity = cudf::test::make_counting_transform_iterator(
      0, [](auto i) { return true; });
    
  cudf::test::expect_columns_equal(result.tbl->get_column(0), int64_wrapper{{100, 200}, validity});
  cudf::test::expect_columns_equal(result.tbl->get_column(1), float64_wrapper{{1.1, 2.2}, validity});  
  cudf::test::expect_columns_equal(result.tbl->get_column(2), cudf::test::strings_column_wrapper({"aa ", "  bbb"}));
}

TEST_F(JsonReaderTest, JsonLinesFileInput) {
  const std::string fname = temp_env->get_temp_dir() + "JsonLinesFileTest.json";
  std::ofstream outfile(fname, std::ofstream::out);
  outfile << "[11, 1.1]\n[22, 2.2]";
  outfile.close();
  // ASSERT_TRUE(checkFile(fname));

  cudf_io::read_json_args in_args{cudf_io::source_info{fname}};  
  in_args.lines = true;

  cudf_io::table_with_metadata result = cudf_io::read_json(in_args);

  EXPECT_EQ(result.tbl->num_columns(), 2);
  EXPECT_EQ(result.tbl->num_rows(), 2);

  EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::INT64);
  EXPECT_EQ(result.tbl->get_column(1).type().id(), cudf::FLOAT64);

  EXPECT_EQ(std::string(result.metadata.column_names[0]), "0");
  EXPECT_EQ(std::string(result.metadata.column_names[1]), "1");

  auto validity = cudf::test::make_counting_transform_iterator(
      0, [](auto i) { return true; });

  cudf::test::expect_columns_equal(result.tbl->get_column(0), int64_wrapper{{11, 22}, validity});
  cudf::test::expect_columns_equal(result.tbl->get_column(1), float64_wrapper{{1.1, 2.2}, validity});    
}

TEST_F(JsonReaderTest, JsonLinesByteRange) {
  const std::string fname = temp_env->get_temp_dir() + "JsonLinesByteRangeTest.json";
  std::ofstream outfile(fname, std::ofstream::out);
  outfile << "[1000]\n[2000]\n[3000]\n[4000]\n[5000]\n[6000]\n[7000]\n[8000]\n[9000]\n";
  outfile.close();
  // ASSERT_TRUE(checkFile(fname));

  cudf_io::read_json_args in_args{cudf_io::source_info{fname}};  
  in_args.lines = true;
  in_args.byte_range_offset = 11;
  in_args.byte_range_size = 20;

  cudf_io::table_with_metadata result = cudf_io::read_json(in_args);

  EXPECT_EQ(result.tbl->num_columns(), 1);
  EXPECT_EQ(result.tbl->num_rows(), 3);
  
  EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::INT64);
  EXPECT_EQ(std::string(result.metadata.column_names[0]), "0");

  auto validity = cudf::test::make_counting_transform_iterator(
      0, [](auto i) { return true; });
  
  cudf::test::expect_columns_equal(result.tbl->get_column(0), int64_wrapper{{3000, 4000, 5000}, validity});  
}

TEST_F(JsonReaderTest, JsonLinesObjects) {
  const std::string fname = temp_env->get_temp_dir() + "JsonLinesObjectsTest.json";
  std::ofstream outfile(fname, std::ofstream::out);
  outfile << " {\"co\\\"l1\" : 1, \"col2\" : 2.0} \n";
  outfile.close();
  // ASSERT_TRUE(checkFile(fname));

  cudf_io::read_json_args in_args{cudf_io::source_info{fname}};  
  in_args.lines = true;

  cudf_io::table_with_metadata result = cudf_io::read_json(in_args);

  EXPECT_EQ(result.tbl->num_columns(), 2);
  EXPECT_EQ(result.tbl->num_rows(), 1);
  
  EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::INT64);
  EXPECT_EQ(std::string(result.metadata.column_names[0]), "co\\\"l1");
  EXPECT_EQ(result.tbl->get_column(1).type().id(), cudf::FLOAT64);  
  EXPECT_EQ(std::string(result.metadata.column_names[1]), "col2");

  auto validity = cudf::test::make_counting_transform_iterator(
      0, [](auto i) { return true; });

  cudf::test::expect_columns_equal(result.tbl->get_column(0), int64_wrapper{{1}, validity});  
  cudf::test::expect_columns_equal(result.tbl->get_column(1), float64_wrapper{{2.0}, validity});    
}

TEST_F(JsonReaderTest, JsonLinesObjectsStrings) {
  std::string data =
      "{\"col1\":100, \"col2\":1.1, \"col3\":\"aaa\"}\n"
      "{\"col1\":200, \"col2\":2.2, \"col3\":\"bbb\"}\n";

  cudf_io::read_json_args in_args{cudf_io::source_info{data.data(), data.size()}};
  in_args.lines = true;

  cudf_io::table_with_metadata result = cudf_io::read_json(in_args);

  EXPECT_EQ(result.tbl->num_columns(), 3);
  EXPECT_EQ(result.tbl->num_rows(), 2);  

  EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::INT64);
  EXPECT_EQ(result.tbl->get_column(1).type().id(), cudf::FLOAT64);
  
  EXPECT_EQ(std::string(result.metadata.column_names[0]), "col1");
  EXPECT_EQ(std::string(result.metadata.column_names[1]), "col2");
  EXPECT_EQ(std::string(result.metadata.column_names[2]), "col3");

  auto validity = cudf::test::make_counting_transform_iterator(
      0, [](auto i) { return true; });
  
  cudf::test::expect_columns_equal(result.tbl->get_column(0), int64_wrapper{{100, 200}, validity});
  cudf::test::expect_columns_equal(result.tbl->get_column(1), float64_wrapper{{1.1, 2.2}, validity});  
  cudf::test::expect_columns_equal(result.tbl->get_column(2), cudf::test::strings_column_wrapper({"aaa", "bbb"}));
}

TEST_F(JsonReaderTest, ArrowFileSource) {
  const std::string fname = temp_env->get_temp_dir() + "ArrowFileSource.csv";

  std::ofstream outfile(fname, std::ofstream::out);
  outfile << "[9]\n[8]\n[7]\n[6]\n[5]\n[4]\n[3]\n[2]\n";
  outfile.close();
  // ASSERT_TRUE(checkFile(fname));

  std::shared_ptr<arrow::io::ReadableFile> infile;
  ASSERT_TRUE(arrow::io::ReadableFile::Open(fname, &infile).ok());

  cudf_io::read_json_args in_args(cudf_io::source_info{infile});
  in_args.lines = true;
  in_args.dtype = {"int8"};
  cudf_io::table_with_metadata result = cudf_io::read_json(in_args);

  EXPECT_EQ(result.tbl->num_columns(), static_cast<cudf::size_type>(in_args.dtype.size()));
  EXPECT_EQ(result.tbl->get_column(0).type().id(), cudf::INT8);

  auto validity = cudf::test::make_counting_transform_iterator(
      0, [](auto i) { return true; });
   
  cudf::test::expect_columns_equal(result.tbl->get_column(0), int8_wrapper{{9, 8, 7, 6, 5, 4, 3, 2}, validity});    
}