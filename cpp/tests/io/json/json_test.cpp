/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cudf/cudf.h>

#include <cuda_runtime.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <sys/stat.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <cstdlib>

#include <nvstrings/NVStrings.h>

#include <tests/io/io_test_utils.hpp>
#include <io/utilities/parsing_utils.cuh>
#include <tests/utilities/cudf_test_fixtures.h>

using std::string;
using std::vector;

TempDirTestEnvironment *const temp_env =
    static_cast<TempDirTestEnvironment *>(::testing::AddGlobalTestEnvironment(new TempDirTestEnvironment));
struct gdf_json_test : GdfTest {};

TEST_F(gdf_json_test, SquareBrackets) {
  const string json_file("{columns\":[\"col 1\",\"col 2\",\"col 3\"] , "
                         "\"index\":[\"row 1\",\"row 2\"] , "
                         "\"data\":[[\"a\",1,1.0],[\"b\",2,2.0]]}");

  const gdf_size_type count = countAllFromSet(json_file.c_str(), json_file.size() * sizeof(char), {'[', ']'});
  ASSERT_TRUE(count == 10);

  device_buffer<uint64_t> d_pos(count);
  findAllFromSet(json_file.c_str(), json_file.size() * sizeof(char), {'[', ']'}, 0, d_pos.data());

  vector<uint64_t> h_pos(count);
  cudaMemcpy(h_pos.data(), d_pos.data(), count * sizeof(uint64_t), cudaMemcpyDefault);
  for (auto pos : h_pos)
    ASSERT_TRUE(json_file[pos] == '[' || json_file[pos] == ']');
}

using pos_key_pair = thrust::pair<uint64_t, char>;
TEST_F(gdf_json_test, BracketsLevels) {
  // Generate square brackets consistent with 'split' json format
  const int rows = 1000000;
  const int file_size = rows * 4 + 1;
  string json_mock("{\"columns\":[x],\"index\":[x],\"data\":[");
  const int header_size = json_mock.size();
  json_mock += string(file_size, 'x');
  json_mock[json_mock.size() - 2] = ']';
  json_mock[json_mock.size() - 1] = '}';
  for (size_t i = header_size; i < json_mock.size() - 1; i += 4) {
    json_mock[i] = '[';
    json_mock[i + 2] = ']';
  }

  vector<int16_t> expected{1, 2, 2, 2, 2, 2};
  fill_n(back_inserter(expected), rows * 2, 3);
  expected.push_back(2);
  expected.push_back(1);

  const gdf_size_type count = countAllFromSet(json_mock.c_str(), json_mock.size() * sizeof(char), {'[', ']', '{', '}'});
  device_buffer<pos_key_pair> d_pos(count);
  findAllFromSet(json_mock.c_str(), json_mock.size() * sizeof(char), {'[', ']', '{', '}'}, 0, d_pos.data());
  const auto d_lvls = getBracketLevels(d_pos.data(), count, string("[{"), string("]}"));

  vector<int16_t> h_lvls(count);
  cudaMemcpy(h_lvls.data(), d_lvls.data(), count * sizeof(int16_t), cudaMemcpyDefault);
  EXPECT_THAT(h_lvls, ::testing::ContainerEq(expected));
}

TEST_F(gdf_json_test, BasicJsonLines) {
  std::string data = "[1, 1.1]\n[2, 2.2]\n[3, 3.3]\n";

  cudf::json_read_arg args(cudf::source_info(data.c_str(), data.size()));
  args.lines = true;
  args.dtype = {"int", "float64"};

  const auto df = cudf::read_json(args);

  ASSERT_EQ(df.num_columns(), 2);
  ASSERT_EQ(df.num_rows(), 3);

  ASSERT_EQ(df.get_column(0)->dtype, GDF_INT32);
  ASSERT_EQ(df.get_column(1)->dtype, GDF_FLOAT64);

  ASSERT_EQ(std::string(df.get_column(0)->col_name), "0");
  ASSERT_EQ(std::string(df.get_column(1)->col_name), "1");

  EXPECT_THAT(gdf_host_column<int32_t>(df.get_column(0)).hostdata(), ::testing::ElementsAre(1, 2, 3));
  EXPECT_THAT(gdf_host_column<double>(df.get_column(1)).hostdata(), ::testing::ElementsAre(1.1, 2.2, 3.3));
}

TEST_F(gdf_json_test, JsonLinesStrings) {
  std::string data = "[1, 1.1, \"aa \"]\n[2, 2.2, \"  bbb\"]";

  cudf::json_read_arg args(cudf::source_info{data.c_str(), data.size()});
  args.lines = true;
  args.dtype = {"2:str", "0:int", "1:float64"};

  const auto df = cudf::read_json(args);

  ASSERT_EQ(df.num_columns(), 3);
  ASSERT_EQ(df.num_rows(), 2);

  ASSERT_EQ(df.get_column(0)->dtype, GDF_INT32);
  ASSERT_EQ(df.get_column(1)->dtype, GDF_FLOAT64);

  ASSERT_EQ(std::string(df.get_column(0)->col_name), "0");
  ASSERT_EQ(std::string(df.get_column(1)->col_name), "1");
  ASSERT_EQ(std::string(df.get_column(2)->col_name), "2");

  EXPECT_THAT(gdf_host_column<int32_t>(df.get_column(0)).hostdata(), ::testing::ElementsAre(1, 2));
  EXPECT_THAT(gdf_host_column<double>(df.get_column(1)).hostdata(), ::testing::ElementsAre(1.1, 2.2));

  checkStrColumn(df.get_column(2), {"aa ", "  bbb"});
}

TEST_F(gdf_json_test, JsonLinesDtypeInference) {
  std::string data = "[100, 1.1, \"aa \"]\n[200, 2.2, \"  bbb\"]";

  cudf::json_read_arg args(cudf::source_info{data.c_str(), data.size()});
  args.lines = true;

  const auto df = cudf::read_json(args);

  ASSERT_EQ(df.num_columns(), 3);
  ASSERT_EQ(df.num_rows(), 2);

  ASSERT_EQ(df.get_column(0)->dtype, GDF_INT64);
  ASSERT_EQ(df.get_column(1)->dtype, GDF_FLOAT64);

  ASSERT_EQ(std::string(df.get_column(0)->col_name), "0");
  ASSERT_EQ(std::string(df.get_column(1)->col_name), "1");
  ASSERT_EQ(std::string(df.get_column(2)->col_name), "2");

  EXPECT_THAT(gdf_host_column<int64_t>(df.get_column(0)).hostdata(), ::testing::ElementsAre(100, 200));
  EXPECT_THAT(gdf_host_column<double>(df.get_column(1)).hostdata(), ::testing::ElementsAre(1.1, 2.2));

  checkStrColumn(df.get_column(2), {"aa ", "  bbb"});
}

TEST_F(gdf_json_test, JsonLinesFileInput) {
  const std::string fname = temp_env->get_temp_dir() + "JsonLinesFileTest.json";
  std::ofstream outfile(fname, std::ofstream::out);
  outfile << "[11, 1.1]\n[22, 2.2]";
  outfile.close();
  ASSERT_TRUE(checkFile(fname));

  cudf::json_read_arg args(cudf::source_info{fname});
  args.lines = true;

  const auto df = cudf::read_json(args);

  ASSERT_EQ(df.num_columns(), 2);
  ASSERT_EQ(df.num_rows(), 2);

  ASSERT_EQ(df.get_column(0)->dtype, GDF_INT64);
  ASSERT_EQ(df.get_column(1)->dtype, GDF_FLOAT64);

  ASSERT_EQ(std::string(df.get_column(0)->col_name), "0");
  ASSERT_EQ(std::string(df.get_column(1)->col_name), "1");

  EXPECT_THAT(gdf_host_column<int64_t>(df.get_column(0)).hostdata(), ::testing::ElementsAre(11, 22));
  EXPECT_THAT(gdf_host_column<double>(df.get_column(1)).hostdata(), ::testing::ElementsAre(1.1, 2.2));
}

TEST_F(gdf_json_test, JsonLinesByteRange) {
  const std::string fname = temp_env->get_temp_dir() + "JsonLinesByteRangeTest.json";
  std::ofstream outfile(fname, std::ofstream::out);
  outfile << "[1000]\n[2000]\n[3000]\n[4000]\n[5000]\n[6000]\n[7000]\n[8000]\n[9000]\n";
  outfile.close();
  ASSERT_TRUE(checkFile(fname));

  cudf::json_read_arg args(cudf::source_info{fname});
  args.lines = true;
  args.byte_range_offset = 11;
  args.byte_range_size = 20;

  const auto df = cudf::read_json(args);

  ASSERT_EQ(df.num_columns(), 1);
  ASSERT_EQ(df.num_rows(), 3);

  ASSERT_EQ(df.get_column(0)->dtype, GDF_INT64);
  ASSERT_EQ(std::string(df.get_column(0)->col_name), "0");

  EXPECT_THAT(gdf_host_column<int64_t>(df.get_column(0)).hostdata(), ::testing::ElementsAre(3000, 4000, 5000));
}

TEST_F(gdf_json_test, JsonLinesObjects) {
  const std::string fname = temp_env->get_temp_dir() + "JsonLinesObjectsTest.json";
  std::ofstream outfile(fname, std::ofstream::out);
  outfile << " {\"co\\\"l1\" : 1, \"col2\" : 2.0} \n";
  outfile.close();
  ASSERT_TRUE(checkFile(fname));

  cudf::json_read_arg args(cudf::source_info{fname});
  args.lines = true;

  const auto df = cudf::read_json(args);

  ASSERT_EQ(df.num_columns(), 2);
  ASSERT_EQ(df.num_rows(), 1);

  ASSERT_EQ(df.get_column(0)->dtype, GDF_INT64);
  ASSERT_EQ(std::string(df.get_column(0)->col_name), "co\\\"l1");
  ASSERT_EQ(df.get_column(1)->dtype, GDF_FLOAT64);
  ASSERT_EQ(std::string(df.get_column(1)->col_name), "col2");

  EXPECT_THAT(gdf_host_column<int64_t>(df.get_column(0)).hostdata(), ::testing::ElementsAre(1));
  EXPECT_THAT(gdf_host_column<double>(df.get_column(1)).hostdata(), ::testing::ElementsAre(2.0));
}

TEST_F(gdf_json_test, JsonLinesObjectsStrings) {
  std::string data =
      "{\"col1\":100, \"col2\":1.1, \"col3\":\"aaa\"}\n"
      "{\"col1\":200, \"col2\":2.2, \"col3\":\"bbb\"}\n";

  cudf::json_read_arg args(cudf::source_info{data.c_str(), data.size()});
  args.lines = true;

  const auto df = cudf::read_json(args);

  ASSERT_EQ(df.num_columns(), 3);
  ASSERT_EQ(df.num_rows(), 2);

  ASSERT_EQ(df.get_column(0)->dtype, GDF_INT64);
  ASSERT_EQ(df.get_column(1)->dtype, GDF_FLOAT64);

  ASSERT_EQ(std::string(df.get_column(0)->col_name), "col1");
  ASSERT_EQ(std::string(df.get_column(1)->col_name), "col2");
  ASSERT_EQ(std::string(df.get_column(2)->col_name), "col3");

  EXPECT_THAT(gdf_host_column<int64_t>(df.get_column(0)).hostdata(), ::testing::ElementsAre(100, 200));
  EXPECT_THAT(gdf_host_column<double>(df.get_column(1)).hostdata(), ::testing::ElementsAre(1.1, 2.2));

  checkStrColumn(df.get_column(2), {"aaa", "bbb"});
}
