
#ifndef GDF_TEST_UTILS
#define GDF_TEST_UTILS

#include <iostream>
#include <gdf/gdf.h>
#include <gdf/cffi/functions.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>

#include <string>
#include <functional>
#include <vector>
#include <tuple>
#include "gdf/gdf.h"

using ValueType = int8_t;

using CheckFunctionType = void(char *, bool *, int);

auto print_binary(gdf_valid_type n) -> void ;

auto chartobin(gdf_valid_type n, int size = 8) -> std::string;

//auto check_column(gdf_column * column, CheckFunctionType check_function) -> void; 
auto check_column(gdf_column *lhs, gdf_column *rhs, gdf_column *output) -> void;

auto print_column(gdf_column * column) -> void;

auto delete_gdf_column(gdf_column * column) -> void; 
 
inline auto get_number_of_bytes_for_valid (size_t column_size) -> size_t {
    return sizeof(gdf_valid_type) * (column_size + GDF_VALID_BITSIZE - 1) / GDF_VALID_BITSIZE;
}

auto gen_gdf_valid(size_t column_size) -> gdf_valid_type *;

auto gen_gdb_column(size_t column_size, ValueType init_value) -> gdf_column;

template <typename RawType, typename PointerType>
auto init_device_vector(gdf_size_type num_elements) -> std::tuple<RawType *, thrust::device_ptr<PointerType>>
{
    RawType *device_pointer;
    cudaError_t cuda_error = cudaMalloc((void **)&device_pointer, sizeof(PointerType) * num_elements);
    EXPECT_TRUE(cuda_error == cudaError::cudaSuccess);
    thrust::device_ptr<PointerType> device_wrapper = thrust::device_pointer_cast((PointerType *)device_pointer);
    return std::make_tuple(device_pointer, device_wrapper);
}


template <typename gdf_type>
inline gdf_dtype gdf_enum_type_for()
{
    return GDF_invalid;
}

template <>
inline gdf_dtype gdf_enum_type_for<int8_t>()
{
    return GDF_INT8;
}

template <>
inline gdf_dtype gdf_enum_type_for<int16_t>()
{
    return GDF_INT16;
}

template <>
inline gdf_dtype gdf_enum_type_for<int32_t>()
{
    return GDF_INT32;
}

template <>
inline gdf_dtype gdf_enum_type_for<int64_t>()
{
    return GDF_INT64;
}

template <>
inline gdf_dtype gdf_enum_type_for<float>()
{
    return GDF_FLOAT32;
}

template <>
inline gdf_dtype gdf_enum_type_for<double>()
{
    return GDF_FLOAT64;
}

#endif // GDF_TEST_UTILS
