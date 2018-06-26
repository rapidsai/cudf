/*
 ============================================================================
 Name        : testing-libgdf.cu
 Author      : felipe
 Version     :
 Copyright   : Your copyright notice
 Description : Compute sum of reciprocals using STL on CPU and Thrust on GPU
 ============================================================================
 */

#include "gtest/gtest.h"


#include <iostream>
#include <gdf/gdf.h>
#include <gdf/cffi/functions.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>

#include <thrust/execution_policy.h>
#include <cuda_runtime.h>
#include "helper/utils.cuh"

#define BIT_FIVE 0x10
#define BIT_SIX 0x20


/*void print_column(gdf_column * column){

	char * host_data_out = new char[column->size];
	char * host_valid_out;

	if(column->size % 8 != 0){
		host_valid_out = new char[(column->size + (8 - (column->size % 8)))/8];
	}else{
		host_valid_out = new char[column->size / 8];
	}


	cudaMemcpy(host_data_out,column->data,sizeof(int8_t) * column->size, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_valid_out,column->valid,sizeof(int8_t) * (column->size + GDF_VALID_BITSIZE - 1) / GDF_VALID_BITSIZE, cudaMemcpyDeviceToHost);

	std::cout<<"Printing Column"<<std::endl;

	for(int i = 0; i < column->size; i++){
		int col_position = i / 8;
		int bit_offset = i % 8;
		std::cout<<"host_data_out["<<i<<"] = "<<((int)host_data_out[i])<<" valid="<<((host_valid_out[col_position] >> bit_offset ) & 1)<<std::endl;
	}

	delete[] host_data_out;
	delete[] host_valid_out;

	std::cout<<std::endl<<std::endl;
}*/

TEST(FilterOperationTest, WithOffBits){
    
   
   gdf_size_type num_elements = 8;

	char * data_left;
	char * data_right;
	char * data_out;
	cudaError_t cuda_error = cudaMalloc((void **) &data_left,sizeof(int8_t) * num_elements);
	cuda_error = cudaMalloc((void **) &data_right,sizeof(int8_t) * num_elements);
	cuda_error = cudaMalloc((void **) &data_out,sizeof(int8_t) * num_elements);

	thrust::device_ptr<int8_t> left_ptr= thrust::device_pointer_cast((int8_t *) data_left);
	int8_t int8_value = 2;
//	thrust::fill(thrust::detail::make_normal_iterator(left_ptr), thrust::detail::make_normal_iterator(left_ptr + num_elements), int8_value);
	thrust::copy(thrust::make_counting_iterator<int8_t>(0),thrust::make_counting_iterator<int8_t>(0) + num_elements, thrust::detail::make_normal_iterator(left_ptr));




	thrust::device_ptr<int8_t> right_ptr= thrust::device_pointer_cast((int8_t *) data_right);
	int8_value = 2;
	thrust::fill(thrust::detail::make_normal_iterator(right_ptr), thrust::detail::make_normal_iterator(right_ptr + num_elements), int8_value);


	//for this simple test we will send in only 8 values
	gdf_valid_type * valid = new gdf_valid_type;


	*valid = 255;
	int num = 4;
	*valid = *valid & ~(1 << num);
	num = 6;
	*valid = *valid & ~(1 << num);

	//shold set lef tand bits 4 and 6 to off

	gdf_valid_type * valid_device;
	cuda_error = cudaMalloc((void **) &valid_device,64);
	cudaMemcpy(valid_device,valid,sizeof(gdf_valid_type),cudaMemcpyHostToDevice);
	gdf_valid_type * valid_out = new gdf_valid_type;
	cuda_error = cudaMalloc((void **) &valid_out,1);
	gdf_column lhs;
	gdf_error error = gdf_column_view(&lhs,(void *) data_left, valid_device,num_elements,GDF_INT8);
	lhs.null_count = 2;
	gdf_column rhs;
	error = gdf_column_view(&rhs,(void *) data_right, valid_device,num_elements,GDF_INT8);
	rhs.null_count = 2;
	gdf_column output;
	error = gdf_column_view(&output,(void *) data_out, valid_out,num_elements,GDF_INT8);

	std::cout<<"Left"<<std::endl;
	print_column(&lhs);
	std::cout<<"Right"<<std::endl;
	print_column(&rhs);


	error = gpu_comparison(&lhs,&rhs,&output,GDF_EQUALS);

	print_column(&output);

	error = gpu_comparison(&lhs,&rhs,&output,GDF_GREATER_THAN);

	print_column(&output);

	//copy the data on the host and compare
	thrust::device_ptr<int8_t> out_ptr = thrust::device_pointer_cast((int8_t *) output.data);





	error = gpu_comparison_static_i8(&lhs,3,&output,GDF_EQUALS);
	error = gpu_comparison(&lhs,&rhs,&output,GDF_GREATER_THAN);

	print_column(&output);

	gpu_apply_stencil(&lhs, &output, &rhs);

	print_column(&rhs);

	check_column_for_stencil_operation(&lhs, &output, &rhs);

//	cudaMemcpy(valid,output.valid,1,cudaMemcpyDeviceToHost);


	cudaFree(data_left);
	cudaFree(data_right);
	cudaFree(data_out);
	cudaFree(valid_device);
	cudaFree(valid_out);

	delete valid;

    EXPECT_EQ(1, 1);
}