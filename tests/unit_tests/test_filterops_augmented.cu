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

template <typename gdf_type>
gdf_dtype gdf_template_type()
{
    return GDF_invalid;
}

template <>
gdf_dtype gdf_template_type<int8_t>()
{
    return GDF_INT8;
}

template <>
gdf_dtype gdf_template_type<int16_t>()
{
    return GDF_INT16;
}

template <>
gdf_dtype gdf_template_type<int32_t>()
{
    return GDF_INT32;
}

template <>
gdf_dtype gdf_template_type<int64_t>()
{
    return GDF_INT64;
}

template <>
gdf_dtype gdf_template_type<float>()
{
    return GDF_FLOAT32;
}

template <>
gdf_dtype gdf_template_type<double>()
{
    return GDF_FLOAT64;
}

template <typename RawType, typename PointerType>
auto init_device_vector(gdf_size_type num_elements) -> std::tuple<RawType *, thrust::device_ptr<PointerType>>
{
    RawType *device_pointer;
    cudaError_t cuda_error = cudaMalloc((void **)&device_pointer, sizeof(PointerType) * num_elements);
    thrust::device_ptr<PointerType> device_wrapper = thrust::device_pointer_cast((PointerType *)device_pointer);
    return std::make_tuple(device_pointer, device_wrapper);
}

template <typename LeftValueType, typename RightValueType>
void test_filterops_augmented()
{

    gdf_dtype gdf_left_type = gdf_template_type<LeftValueType>();
    gdf_dtype gdf_right_type = gdf_template_type<RightValueType>();

    gdf_size_type num_elements = 8;
    char *data_left,
        *data_right,
        *data_out;

    thrust::device_ptr<LeftValueType> left_pointer;
    thrust::device_ptr<RightValueType> right_pointer;
    gdf_error error;

    std::tie(data_left, left_pointer) = init_device_vector<char, LeftValueType>(num_elements);
    std::tie(data_right, right_pointer) = init_device_vector<char, RightValueType>(num_elements);
    std::tie(data_out, std::ignore) = init_device_vector<char, int8_t>(num_elements);
    std::cout << "done init_device_vector: \n";

    using thrust::detail::make_normal_iterator;
    LeftValueType init_value = 2;
    thrust::fill(make_normal_iterator(right_pointer), make_normal_iterator(right_pointer + num_elements), init_value);
    std::cout << "done fill: \n";

    gdf_valid_type *valid = new gdf_valid_type;
    *valid = 255;
    gdf_valid_type *valid_value_pointer;
    cudaMalloc((void **)&valid_value_pointer, 1);
    cudaMemcpy(valid_value_pointer, valid, sizeof(gdf_valid_type), cudaMemcpyHostToDevice);

    gdf_valid_type *valid_out_pointer;
    cudaMalloc((void **)&valid_out_pointer, 1);
    std::cout << "done init_device_var: \n";

    gdf_column lhs, rhs, output;
    error = gdf_column_view_augmented(&lhs, (void *)data_left, valid_value_pointer, num_elements, gdf_left_type, 0);
    error = gdf_column_view_augmented(&rhs, (void *)data_right, valid_value_pointer, num_elements, gdf_right_type, 0);
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

TEST(FilterOperationsTest, WithInt8AndOthers)
{
    test_filterops_augmented<int8_t, int8_t>();
    test_filterops_augmented<int8_t, int16_t>();
    test_filterops_augmented<int8_t, int32_t>();
    test_filterops_augmented<int8_t, int64_t>();
    test_filterops_augmented<int8_t, float>();
    test_filterops_augmented<int8_t, double>();
    EXPECT_EQ(1, 1);
}

TEST(FilterOperationsTest, WithInt16AndOthers)
{
    test_filterops_augmented<int16_t, int8_t>();
    test_filterops_augmented<int16_t, int16_t>();
    test_filterops_augmented<int16_t, int32_t>();
    test_filterops_augmented<int16_t, int64_t>();
    test_filterops_augmented<int16_t, float>();
    test_filterops_augmented<int16_t, double>();
    EXPECT_EQ(1, 1);
}

TEST(FilterOperationsTest, WithInt32AndOthers)
{
    test_filterops_augmented<int32_t, int8_t>();
    test_filterops_augmented<int32_t, int16_t>();
    test_filterops_augmented<int32_t, int32_t>();
    test_filterops_augmented<int32_t, int64_t>();
    test_filterops_augmented<int32_t, float>();
    test_filterops_augmented<int32_t, double>();
    EXPECT_EQ(1, 1);
}

TEST(FilterOperationsTest, WithInt64AndOthers)
{
    test_filterops_augmented<int64_t, int8_t>();
    test_filterops_augmented<int64_t, int16_t>();
    test_filterops_augmented<int64_t, int32_t>();
    test_filterops_augmented<int64_t, int64_t>();
    test_filterops_augmented<int64_t, float>();
    test_filterops_augmented<int64_t, double>();
    EXPECT_EQ(1, 1);
}

TEST(FilterOperationsTest, WithFloat32AndOthers)
{
    test_filterops_augmented<float, int8_t>();
    test_filterops_augmented<float, int16_t>();
    test_filterops_augmented<float, int32_t>();
    test_filterops_augmented<float, int64_t>();
    test_filterops_augmented<float, float>();
    test_filterops_augmented<float, double>();
    EXPECT_EQ(1, 1);
}

TEST(FilterOperationsTest, WithFloat64AndOthers)
{
    test_filterops_augmented<double, int8_t>();
    test_filterops_augmented<double, int16_t>();
    test_filterops_augmented<double, int32_t>();
    test_filterops_augmented<double, int64_t>();
    test_filterops_augmented<double, float>();
    test_filterops_augmented<double, double>();
    EXPECT_EQ(1, 1);
}
