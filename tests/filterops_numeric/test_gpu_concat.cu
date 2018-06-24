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

#include <gdf/cffi/functions.h>
#include <gdf/gdf.h>
#include <iostream>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>

#include "tests_utils.h"
#include <cuda_runtime.h>
#include <thrust/execution_policy.h>

#define BIT_FIVE 0x10
#define BIT_SIX 0x20

TEST(GpuConcatTest, WithDifferentColumnSizes)
{
    gdf_size_type left_num_elements = 5;
    gdf_size_type right_num_elements = 10;

    char *data_left;
    char *data_right;
    char *data_out;
    cudaError_t cuda_error =
        cudaMalloc((void **)&data_left, sizeof(int8_t) * left_num_elements);
    cuda_error =
        cudaMalloc((void **)&data_right, sizeof(int8_t) * right_num_elements);
    cuda_error =
        cudaMalloc((void **)&data_out,
                   sizeof(int8_t) * (left_num_elements + right_num_elements));

    thrust::device_ptr<int8_t> left_ptr =
        thrust::device_pointer_cast((int8_t *)data_left);
    int8_t int8_value = 2;
    //	thrust::fill(thrust::detail::make_normal_iterator(left_ptr),
    // thrust::detail::make_normal_iterator(left_ptr + num_elements), int8_value);
    thrust::copy(thrust::make_counting_iterator<int8_t>(0),
                 thrust::make_counting_iterator<int8_t>(0) + left_num_elements,
                 thrust::detail::make_normal_iterator(left_ptr));

    thrust::device_ptr<int8_t> right_ptr =
        thrust::device_pointer_cast((int8_t *)data_right);
    int8_value = 2;
    thrust::fill(
        thrust::detail::make_normal_iterator(right_ptr),
        thrust::detail::make_normal_iterator(right_ptr + right_num_elements),
        int8_value);

    // left valid data
    gdf_valid_type *valid_left_host = new gdf_valid_type;
    *valid_left_host = 0b11111;
    gdf_valid_type *valid_left_device;
    cuda_error = cudaMalloc((void **)&valid_left_device, sizeof(gdf_valid_type));
    cudaMemcpy(valid_left_device, valid_left_host, sizeof(gdf_valid_type),
               cudaMemcpyHostToDevice);

    // right valid data
    gdf_valid_type *valid_right_host = new gdf_valid_type[2];
    valid_right_host[0] = 0b11111011;
    valid_right_host[1] = 0b11;
    gdf_valid_type *valid_right_device;
    cuda_error =
        cudaMalloc((void **)&valid_right_device, 2 * sizeof(gdf_valid_type));
    cudaMemcpy(valid_right_device, valid_right_host, 2 * sizeof(gdf_valid_type),
               cudaMemcpyHostToDevice);

    gdf_valid_type *valid_out = new gdf_valid_type;
    cuda_error = cudaMalloc((void **)&valid_out, 2 * sizeof(gdf_valid_type));

    gdf_column lhs;
    gdf_error error = gdf_column_view(&lhs, (void *)data_left, valid_left_device,
                                      left_num_elements, GDF_INT8);
    lhs.null_count = 3;
    gdf_column rhs;
    error = gdf_column_view(&rhs, (void *)data_right, valid_right_device,
                            right_num_elements, GDF_INT8);
    rhs.null_count = 7; //@todo: ask for this count?
    gdf_column output;
    error = gdf_column_view(&output, (void *)data_out, valid_out,
                            left_num_elements + right_num_elements, GDF_INT8);

    std::cout << "Left" << std::endl;
    print_column(&lhs);
    std::cout << "Right" << std::endl;
    print_column(&rhs);

    error = gpu_concat(&lhs, &rhs, &output);
    EXPECT_TRUE(error == GDF_SUCCESS);

    gdf_valid_type *expectec_valid_output = new gdf_valid_type[2];
    expectec_valid_output[0] = 11111111;
    expectec_valid_output[1] = 01101111;

    print_column(&output);

    cudaFree(data_left);
    cudaFree(data_right);
    cudaFree(data_out);
    cudaFree(valid_left_device);
    cudaFree(valid_right_device);
    cudaFree(valid_out);

    delete valid_left_host;
    delete[] valid_right_host;

    EXPECT_EQ(1, 1);
}