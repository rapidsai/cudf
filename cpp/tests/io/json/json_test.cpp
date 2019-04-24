/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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

#include <cudf.h>

#include<cuda_runtime.h>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <sys/stat.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include <cstdlib>

#include <nvstrings/NVStrings.h>

#include "io/utilities/parsing_utils.cuh"

using std::vector;
using std::string;

bool checkFile(std::string fname)
{
  struct stat st;
  return (stat(fname.c_str(), &st) ? 0 : 1);
}

template <typename T>
std::vector<T> gdf_column_to_host(gdf_column* const col) {
    auto m_hostdata = std::vector<T>(col->size);
    cudaMemcpy(m_hostdata.data(), col->data, sizeof(T) * col->size, cudaMemcpyDeviceToHost);
    return m_hostdata;
}

TEST(gdf_json_test, SquareBrackets)
{
  const string json_file("{columns\":[\"col 1\",\"col 2\",\"col 3\"] , "
    "\"index\":[\"row 1\",\"row 2\"] , "
    "\"data\":[[\"a\",1,1.0],[\"b\",2,2.0]]}");

  const gdf_size_type count = countAllFromSet(json_file.c_str(), json_file.size()*sizeof(char), {'[', ']'});
  ASSERT_TRUE(count == 10);

  device_buffer<uint64_t> d_pos(count);
  findAllFromSet(json_file.c_str(), json_file.size()*sizeof(char), {'[', ']'}, 0, d_pos.data());

  vector<uint64_t> h_pos(count);
  cudaMemcpy(h_pos.data(), d_pos.data(), count*sizeof(uint64_t), cudaMemcpyDefault);
  for (auto pos: h_pos)
    ASSERT_TRUE(json_file[pos] == '[' || json_file[pos] == ']');
}

using pos_key_pair = thrust::pair<uint64_t,char>;
TEST(gdf_json_test, BracketsLevels)
{
  // Generate square brackets consistent with 'split' json format
  const int rows = 1000000;
  const int file_size = rows * 4 + 1;
  string json_mock("{\"columns\":[x],\"index\":[x],\"data\":[");
  const int header_size = json_mock.size();
  json_mock += string(file_size, 'x');
  json_mock[json_mock.size() - 2] = ']';
  json_mock[json_mock.size() - 1] = '}';
  for (size_t i = header_size; i < json_mock.size() - 1; i += 4){
    json_mock[i] = '[';
    json_mock[i + 2] = ']';
  }

  vector<int16_t> expected{1, 2, 2, 2, 2, 2};
  fill_n(back_inserter(expected), rows*2, 3);
  expected.push_back(2);
  expected.push_back(1);

  const gdf_size_type count = countAllFromSet(json_mock.c_str(), json_mock.size()*sizeof(char), {'[', ']','{','}'});
  device_buffer<pos_key_pair> d_pos(count);
  findAllFromSet(json_mock.c_str(), json_mock.size()*sizeof(char), {'[', ']','{','}'}, 0, d_pos.data());
  const auto d_lvls = getBracketLevels(d_pos.data(), count, string("[{"), string("]}"));

  vector<int16_t> h_lvls(count);
  cudaMemcpy(h_lvls.data(), d_lvls.data(), count*sizeof(int16_t), cudaMemcpyDefault);
  EXPECT_THAT(h_lvls, ::testing::ContainerEq(expected));
}

TEST(gdf_json_test, BasicJsonLines)
{
  const char* types[]  = { "int", "float64" };
  json_read_arg args{};
  args.source = "[1, 1.1]\n[2, 2.2]\n[3, 3.3]\n";
  args.source_type = HOST_BUFFER;
  args.buffer_size = strlen(args.source);
  args.dtype = types;
  args.num_cols = 2;
  try {
    read_json(&args);
  } catch(std::exception &e){std::cerr<< e.what();}

  ASSERT_EQ(args.num_cols_out, 2);
  ASSERT_EQ(args.num_rows_out, 3);

  ASSERT_EQ(args.data[0]->dtype, GDF_INT32);
  ASSERT_EQ(args.data[1]->dtype, GDF_FLOAT64);

  ASSERT_EQ(std::string(args.data[0]->col_name), "0");
  ASSERT_EQ(std::string(args.data[1]->col_name), "1");


  const auto firstCol = gdf_column_to_host<int32_t>(args.data[0]);
  EXPECT_THAT(firstCol, ::testing::ElementsAre(1, 2, 3));
  const auto secondCol = gdf_column_to_host<double>(args.data[1]);
  EXPECT_THAT(secondCol, ::testing::ElementsAre(1.1, 2.2, 3.3));
}

TEST(gdf_json_test, JsonLinesStrings)
{
  const char* types[] = { "int", "float64", "str" };
  json_read_arg args{};
  args.source = "[1, 1.1, \"aa \"]\n[2, 2.2, \"  bbb\"]";
  args.source_type = HOST_BUFFER;
  args.buffer_size = strlen(args.source);
  args.dtype = types;
  args.num_cols = 3;
  try {
    read_json(&args);
  } catch(std::exception &e){std::cerr<< e.what();}

  ASSERT_EQ(args.num_cols_out, 3);
  ASSERT_EQ(args.num_rows_out, 2);

  ASSERT_EQ(args.data[0]->dtype, GDF_INT32);
  ASSERT_EQ(args.data[1]->dtype, GDF_FLOAT64);
  ASSERT_EQ(args.data[2]->dtype, GDF_STRING);

  ASSERT_EQ(std::string(args.data[0]->col_name), "0");
  ASSERT_EQ(std::string(args.data[1]->col_name), "1");
  ASSERT_EQ(std::string(args.data[2]->col_name), "2");

  const auto firstCol = gdf_column_to_host<int32_t>(args.data[0]);
  EXPECT_THAT(firstCol, ::testing::ElementsAre(1, 2));
  const auto secondCol = gdf_column_to_host<double>(args.data[1]);
  EXPECT_THAT(secondCol, ::testing::ElementsAre(1.1, 2.2));

  const auto stringList = reinterpret_cast<NVStrings*>(args.data[2]->data);

  ASSERT_NE(stringList, nullptr);
  const auto stringCount = stringList->size();
  ASSERT_EQ(stringCount, 2u);
  const auto stringLengths = std::unique_ptr<int[]>{ new int[stringCount] };
  ASSERT_NE(stringList->byte_count(stringLengths.get(), false), 0u);

  // Check the actual strings themselves
  auto strings = std::unique_ptr<char*[]>{ new char*[stringCount] };
  for (size_t i = 0; i < stringCount; ++i) {
    ASSERT_GT(stringLengths[i], 0);
    strings[i] = new char[stringLengths[i] + 1];
    strings[i][stringLengths[i]] = 0;
  }
  EXPECT_EQ(stringList->to_host(strings.get(), 0, stringCount), 0);

  EXPECT_STREQ(strings[0], "aa ");
  EXPECT_STREQ(strings[1], "  bbb");
  for (size_t i = 0; i < stringCount; ++i) {
    delete[] strings[i];
  }
}

TEST(gdf_json_test, JsonLinesDtypeInference)
{
  json_read_arg args{};
  args.source = "[100, 1.1, \"aa \"]\n[200, 2.2, \"  bbb\"]";
  args.source_type = HOST_BUFFER;
  args.buffer_size = strlen(args.source);
  args.num_cols = 3;
  try {
    read_json(&args);
  } catch(std::exception &e){std::cerr<< e.what();}

  ASSERT_EQ(args.num_cols_out, 3);
  ASSERT_EQ(args.num_rows_out, 2);

  ASSERT_EQ(args.data[0]->dtype, GDF_INT64);
  ASSERT_EQ(args.data[1]->dtype, GDF_FLOAT64);
  ASSERT_EQ(args.data[2]->dtype, GDF_STRING);

  ASSERT_EQ(std::string(args.data[0]->col_name), "0");
  ASSERT_EQ(std::string(args.data[1]->col_name), "1");
  ASSERT_EQ(std::string(args.data[2]->col_name), "2");

  const auto firstCol = gdf_column_to_host<int64_t>(args.data[0]);
  EXPECT_THAT(firstCol, ::testing::ElementsAre(100, 200));
  const auto secondCol = gdf_column_to_host<double>(args.data[1]);
  EXPECT_THAT(secondCol, ::testing::ElementsAre(1.1, 2.2));

  const auto stringList = reinterpret_cast<NVStrings*>(args.data[2]->data);

  ASSERT_NE(stringList, nullptr);
  const auto stringCount = stringList->size();
  ASSERT_EQ(stringCount, 2u);
  const auto stringLengths = std::unique_ptr<int[]>{ new int[stringCount] };
  ASSERT_NE(stringList->byte_count(stringLengths.get(), false), 0u);

  // Check the actual strings themselves
  auto strings = std::unique_ptr<char*[]>{ new char*[stringCount] };
  for (size_t i = 0; i < stringCount; ++i) {
    ASSERT_GT(stringLengths[i], 0);
    strings[i] = new char[stringLengths[i] + 1];
    strings[i][stringLengths[i]] = 0;
  }
  EXPECT_EQ(stringList->to_host(strings.get(), 0, stringCount), 0);

  EXPECT_STREQ(strings[0], "aa ");
  EXPECT_STREQ(strings[1], "  bbb");
  for (size_t i = 0; i < stringCount; ++i) {
    delete[] strings[i];
  }
}

TEST(gdf_json_test, JsonLinesFileInput)
{
  const char* fname  = "/tmp/CsvNumbersTest.csv";
  std::ofstream outfile(fname, std::ofstream::out);
  outfile << "[11, 1.1]\n[22, 2.2]";
  outfile.close();
  ASSERT_TRUE(checkFile(fname));

  json_read_arg args{};
  args.source = fname;
  args.source_type = FILE_PATH;
  args.num_cols = 2;
  try {
    read_json(&args);
  } catch(std::exception &e){std::cerr<< e.what();}

  ASSERT_EQ(args.num_cols_out, 2);
  ASSERT_EQ(args.num_rows_out, 2);

  ASSERT_EQ(args.data[0]->dtype, GDF_INT64);
  ASSERT_EQ(args.data[1]->dtype, GDF_FLOAT64);

  ASSERT_EQ(std::string(args.data[0]->col_name), "0");
  ASSERT_EQ(std::string(args.data[1]->col_name), "1");

  const auto firstCol = gdf_column_to_host<int64_t>(args.data[0]);
  EXPECT_THAT(firstCol, ::testing::ElementsAre(11, 22));
  const auto secondCol = gdf_column_to_host<double>(args.data[1]);
  EXPECT_THAT(secondCol, ::testing::ElementsAre(1.1, 2.2));
}