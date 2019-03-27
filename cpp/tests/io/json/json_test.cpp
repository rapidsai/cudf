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
using std::cout;

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
	const string h_data("{columns\":[\"col 1\",\"col 2\",\"col 3\"] , "
		"\"index\":[\"row 1\",\"row 2\"] , "
		"\"data\":[[\"a\",1,1.0],[\"b\",2,2.0]]}");

	char* d_data{};
	cudaMalloc(&d_data, h_data.size()*sizeof(char));
	cudaMemcpy(d_data, h_data.c_str(), h_data.size()*sizeof(char), cudaMemcpyDefault);

	gdf_size_type count = 0;
	auto error = countAllFromSet(h_data.c_str(), h_data.size()*sizeof(char), {'[', ']'}, count);
	ASSERT_FALSE(error);

	ll_uint_t* d_pos{};
	vector<ll_uint_t> h_pos(count);
	cudaMalloc(&d_pos, count*sizeof(ll_uint_t));

	error = findAllFromSet(h_data.c_str(), h_data.size()*sizeof(char), {'[', ']'}, 0, d_pos);
	ASSERT_FALSE(error);
	cudaMemcpy(h_pos.data(), d_pos, count*sizeof(ll_uint_t), cudaMemcpyDefault);

	cudaFree(d_data);
	cudaFree(d_pos);

	for (auto pos: h_pos)
		ASSERT_TRUE(h_data[pos] == '[' || h_data[pos] == ']');
}

TEST(gdf_json_test, BracketsLevels)
{
	// Generate square brackets consistent with 'split' json format
	const int rows = 20;
	const int file_size = rows * 4;
	string h_data("\"columns\":[x],\"index\":[x],\"data\":[");
	const int header_size = h_data.size();
	h_data += string(file_size, 'x');
	h_data[h_data.size() - 1] = ']';
	for (int i = header_size; i < h_data.size() - 1; i += 4){
		h_data[i] = '[';
		h_data[i + 2] = ']';
	}
	cout << h_data << '\n';

	char* d_data{};
	cudaMalloc(&d_data, h_data.size()*sizeof(char));
	cudaMemcpy(d_data, h_data.c_str(), h_data.size()*sizeof(char), cudaMemcpyDefault);

	gdf_size_type count = 0;
	auto error = countAllFromSet(h_data.c_str(), h_data.size()*sizeof(char), {'[', ']'}, count);
	ASSERT_FALSE(error);

	ll_uint_t* d_pos{};
	cudaMalloc(&d_pos, count*sizeof(ll_uint_t));

	error = findAllFromSet(h_data.c_str(), h_data.size()*sizeof(char), {'[', ']'}, 0, d_pos);
	ASSERT_FALSE(error);




	cudaFree(d_data);
	cudaFree(d_pos);

}
