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


TEST(gdf_json_test, stub)
{
	ASSERT_TRUE(true);
}
