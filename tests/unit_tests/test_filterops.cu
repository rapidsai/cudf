#include "gtest/gtest.h"
#include <iostream>
#include <gdf/gdf.h>
#include <gdf/cffi/functions.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <cuda_runtime.h>
#include <tuple>
#include "utils.cuh"

template<typename RawType, typename PointerType>
auto init_device_vector(gdf_size_type num_elements) -> std::tuple<RawType *, thrust::device_ptr<PointerType> >
{
    RawType *device_pointer;
    cudaError_t cuda_error = cudaMalloc((void **)&device_pointer, sizeof(PointerType) * num_elements);
    thrust::device_ptr<PointerType> device_wrapper = thrust::device_pointer_cast((PointerType *)device_pointer);
    return std::make_tuple(device_pointer, device_wrapper);
}


auto init_device_variable(gdf_valid_type init_value) -> gdf_valid_type *{
    gdf_valid_type *valid_device;
    gdf_valid_type * valid = new gdf_valid_type;
    *valid = init_value;
    cudaError_t cuda_error = cudaMalloc((void **)(&valid_device), sizeof(gdf_valid_type));
    cudaMemcpy(&valid_device, valid, sizeof(gdf_valid_type), cudaMemcpyHostToDevice);
    delete valid;
    return valid_device;
}

template<typename ValueType>
void test_filterops(gdf_dtype gdf_type_value) {

    gdf_size_type num_elements = 8;
    char* data_left,
        * data_right,
        * data_out;

    thrust::device_ptr<ValueType> 	left_pointer,
                                    right_pointer;
    gdf_error error;

    std::tie(data_left, left_pointer) = init_device_vector<char, ValueType>(num_elements);
    std::tie(data_right, right_pointer) = init_device_vector<char, ValueType>(num_elements);
    std::tie(data_out, std::ignore) = init_device_vector<char, int8_t>(num_elements);
    std::cout << "done init_device_vector: \n";

    using thrust::detail::make_normal_iterator;
    ValueType init_value = 2;
    thrust::fill(make_normal_iterator(right_pointer), make_normal_iterator(right_pointer + num_elements), init_value);
    std::cout << "done fill: \n";

    //auto valid_value_pointer = init_device_variable(255); // 1111 1111 
    //auto valid_out_pointer = init_device_variable(0);
    gdf_valid_type *valid = new gdf_valid_type;

    *valid = 255;
    gdf_valid_type *valid_value_pointer;
    cudaMalloc((void **)&valid_value_pointer, 1);
    cudaMemcpy(valid_value_pointer, valid, sizeof(gdf_valid_type), cudaMemcpyHostToDevice);
    
    gdf_valid_type *valid_out_pointer;
    cudaMalloc((void **)&valid_out_pointer, 1);
    std::cout << "done init_device_var: \n"; 

    gdf_column lhs, rhs, output;
    error = gdf_column_view_augmented(&lhs, (void *)data_left, valid_value_pointer, num_elements, gdf_type_value, 0);
    error = gdf_column_view_augmented(&rhs, (void *)data_right, valid_value_pointer, num_elements, gdf_type_value, 0);
    error = gdf_column_view_augmented(&output, (void *)data_out, valid_out_pointer, num_elements, GDF_INT8, 0);

    print_column(&lhs);
    print_column(&rhs);
    std::cout << "run gpu_comparison: \n";
    error = gpu_comparison(&lhs, &rhs, &output, GDF_EQUALS); // gtest!
     print_column(&output);
    
    cudaFree(data_left);
    cudaFree(data_right);
    cudaFree(data_out);
    cudaFree(valid_value_pointer);
    cudaFree(valid_out_pointer); 
}

TEST(FilterOperationsTest, WithInt8)
{
    test_filterops<int8_t>(GDF_INT8);
    EXPECT_EQ(1, 1);
}

TEST(FilterOperationsTest, WithInt16)
{
    test_filterops<int16_t>(GDF_INT16);
    EXPECT_EQ(1, 1);
}

TEST(FilterOperationsTest, WithInt32)
{
    test_filterops<int32_t>(GDF_INT32);
    EXPECT_EQ(1, 1);
}

TEST(FilterOperationsTest, WithInt64)
{
    test_filterops<int64_t>(GDF_INT64);
    EXPECT_EQ(1, 1);
}

TEST(FilterOperationsTest, WithFloat)
{
    test_filterops<float>(GDF_FLOAT32);
    EXPECT_EQ(1, 1);
}

TEST(FilterOperationsTest, WithDouble)
{
    test_filterops<double>(GDF_FLOAT64);
    EXPECT_EQ(1, 1);
}
