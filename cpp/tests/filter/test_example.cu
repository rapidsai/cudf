/*
 * Copyright 2018 BlazingDB, Inc.
 *     Copyright 2018 Felipe Aramburu <felipe@blazingdb.com>
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

#include "gtest/gtest.h"

#include <iostream>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>

#include <thrust/execution_policy.h>
#include <cuda_runtime.h>
#include "helper/utils.cuh"

#include <cudf.h>
#include <cudf/functions.h>

#include "tests/utilities/cudf_test_fixtures.h"
struct Example : public GdfTest {};

/*
 ============================================================================
 Description : Compute sum of reciprocals using STL on CPU and Thrust on GPU
 ============================================================================
 */
TEST_F(Example, Equals)
{
	gdf_size_type num_elements = 8;

	char *data_left;
	char *data_right;
	char *data_out;
	rmmError_t rmm_error = RMM_ALLOC((void **)&data_left, sizeof(int8_t) * num_elements, 0);
	rmm_error = RMM_ALLOC((void **)&data_right, sizeof(int8_t) * num_elements, 0);
	rmm_error = RMM_ALLOC((void **)&data_out, sizeof(int8_t) * num_elements, 0);
	ASSERT_EQ(rmm_error, RMM_SUCCESS);

	int8_t int8_value = 2;
	thrust::device_ptr<int8_t> right_ptr = thrust::device_pointer_cast((int8_t *)data_right);
	thrust::fill(thrust::detail::make_normal_iterator(right_ptr), thrust::detail::make_normal_iterator(right_ptr + num_elements), int8_value);

	//for this simple test we will send in only 8 values
	gdf_valid_type *valid = new gdf_valid_type;

	*valid = 255;
	gdf_valid_type *valid_device;
	rmm_error = RMM_ALLOC((void **)&valid_device, 1, 0);
	cudaMemcpy(valid_device, valid, sizeof(gdf_valid_type), cudaMemcpyHostToDevice);
	
	gdf_valid_type *valid_out;
	rmm_error = RMM_ALLOC((void **)&valid_out, 1, 0);
	gdf_column lhs;
	gdf_error error = gdf_column_view_augmented(&lhs, (void *)data_left, valid_device, num_elements, GDF_INT8, 0, { TIME_UNIT_ms });
	gdf_column rhs;
	error = gdf_column_view_augmented(&rhs, (void *)data_right, valid_device, num_elements, GDF_INT8, 0, { TIME_UNIT_ms });
	gdf_column output;
	error = gdf_column_view_augmented(&output, (void *)data_out, valid_out, num_elements, GDF_INT8, 0, { TIME_UNIT_ms });
	ASSERT_EQ(error, GDF_SUCCESS);

	std::cout << "Left" << std::endl;
	print_column(&lhs);
	std::cout << "Right" << std::endl;
	print_column(&rhs);
	error = gdf_comparison(&lhs, &rhs, &output, GDF_EQUALS); // gtest!
	std::cout << "Output" << std::endl;
	print_column(&output);

	error = gdf_comparison_static_i8(&lhs, 3, &output, GDF_EQUALS);
	ASSERT_EQ(error, GDF_SUCCESS);
 
	std::cout << "Output static_i8" << std::endl;
	print_column(&output);

	RMM_FREE(data_left, 0);
	RMM_FREE(data_right, 0);
	RMM_FREE(data_out, 0);
	RMM_FREE(valid_device, 0);
	RMM_FREE(valid_out, 0); 
	delete valid;

	EXPECT_EQ(1, 1);
}
