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
	const char* types[]	= { "int", "float64" };
	json_read_arg args{};
	args.source = "[1, 1.1]\n[2, 2.2]";
	args.dtype = types;
	args.num_cols = 2;
	read_json(&args);
	ASSERT_EQ(args.data[0]->dtype, GDF_INT32);

	// delete columns?
}