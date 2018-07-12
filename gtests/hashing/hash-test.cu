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

#include <cstdlib>
#include <iostream>
#include <vector>

#include <thrust/device_vector.h>

#include "gtest/gtest.h"
#include <gdf/gdf.h>
#include <gdf/cffi/functions.h>

struct gdf_hashing_test : public ::testing::Test {

	void TearDown() {

	}
};

TEST(gdf_hashing_test, simpleTest) {

  EXPECT_TRUE( 1+1 == 2 );

}

TEST(gdf_hashing_test, twoColTest) {

	int nrows = 5;
	int ncols = 2;

	gdf_column **inputCol;
	inputCol = (gdf_column**) malloc(sizeof(gdf_column*)* ncols);
	for(size_t i=0;i<nrows;i++)
		inputCol[i] = (gdf_column*) malloc(sizeof(gdf_column*));

	gdf_column outputCol;

	inputCol[0]->dtype = GDF_INT32;
	inputCol[0]->size = nrows;

	inputCol[1]->dtype = GDF_INT64;
	inputCol[1]->size = nrows;

	outputCol.dtype = GDF_INT32;
	outputCol.size = nrows;

	// Input Data
	std::vector<int32_t> inputData1(nrows);
	inputData1[0] = 0;
	inputData1[1] = 1;
	inputData1[2] = 2;
	inputData1[3] = 3;
	inputData1[4] = 0;

	std::vector<int64_t> inputData2(nrows);
	inputData2[0] = 2;
	inputData2[1] = 34;
	inputData2[2] = 7;
	inputData2[3] = 9;
	inputData2[4] = 2;

	thrust::device_vector<int32_t> intputDataDev1(inputData1);
	thrust::device_vector<gdf_valid_type> inputValidDev(1,0);

	thrust::device_vector<int64_t> intputDataDev2(inputData2);
	// thrust::device_vector<gdf_valid_type> inputValidDev2(1,0);

	thrust::device_vector<int32_t> outDataDev(nrows);
	thrust::device_vector<gdf_valid_type> outputValidDev(1,0);

	inputCol[0]->data = thrust::raw_pointer_cast(intputDataDev1.data());
	inputCol[0]->valid = thrust::raw_pointer_cast(inputValidDev.data());

	inputCol[1]->data = thrust::raw_pointer_cast(intputDataDev2.data());
	inputCol[1]->valid = thrust::raw_pointer_cast(inputValidDev.data());

	outputCol.data = thrust::raw_pointer_cast(outDataDev.data());
	outputCol.valid = thrust::raw_pointer_cast(outputValidDev.data());

	{
		gdf_hash_func hash = GDF_HASH_MURMUR3;
		gdf_error gdfError = gdf_hash(ncols, inputCol, hash, &outputCol);

		EXPECT_TRUE( gdfError == GDF_SUCCESS );
		EXPECT_FALSE( gdfError == GDF_CUDA_ERROR );
		EXPECT_FALSE( gdfError == GDF_UNSUPPORTED_DTYPE );
		EXPECT_FALSE( gdfError == GDF_COLUMN_SIZE_MISMATCH );

		std::vector<int32_t> results(nrows+1);
		thrust::copy(outDataDev.begin(), outDataDev.end(), results.begin());

		EXPECT_TRUE( results[0] == results[nrows-1]);
	}
}
