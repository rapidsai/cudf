/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <tests/utilities/column_wrapper.cuh>
#include <tests/utilities/cudf_test_fixtures.h>
#include <utilities/wrapper_types.hpp>
#include <utilities/device_atomics.cuh>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <thrust/device_vector.h>
#include <thrust/transform.h>

#include <bitset>
#include <cstdint>
#include <random>
#include <iostream>

template<typename T>
__global__
void gpu_atomic_test(T *result, T *data, size_t size)
{
    size_t id   = blockIdx.x * blockDim.x + threadIdx.x;
    size_t step = blockDim.x * gridDim.x;

    for (; id < size; id += step) {
        atomicAdd(&result[0], data[id]);
        atomicMin(&result[1], data[id]);
        atomicMax(&result[2], data[id]);
        cudf::genericAtomicOperation(&result[3], data[id], cudf::DeviceSum{});
    }
}

template<typename T, typename BinaryOp>
__device__
T atomic_op(T* addr, T const & value, BinaryOp op)
{
    T old_value = *addr;
    T assumed;

    do {
        assumed  = old_value;
        const T new_value = op(old_value, value);

        old_value = atomicCAS(addr, assumed, new_value);
    } while (assumed != old_value);

    return old_value;
}

template<typename T>
__global__
void gpu_atomicCAS_test(T *result, T *data, size_t size)
{
    size_t id   = blockIdx.x * blockDim.x + threadIdx.x;
    size_t step = blockDim.x * gridDim.x;

    for (; id < size; id += step) {
        atomic_op(&result[0], data[id], cudf::DeviceSum{});
        atomic_op(&result[1], data[id], cudf::DeviceMin{});
        atomic_op(&result[2], data[id], cudf::DeviceMax{});
        atomic_op(&result[3], data[id], cudf::DeviceSum{});
    }
}

template<typename T>
__global__
void gpu_atomic_bitwiseOp_test(T *result, T *data, size_t size)
{
    size_t id   = blockIdx.x * blockDim.x + threadIdx.x;
    size_t step = blockDim.x * gridDim.x;

    for (; id < size; id += step) {
        atomicAnd(&result[0], data[id]);
        atomicOr(&result[1], data[id]);
        atomicXor(&result[2], data[id]);
        cudf::genericAtomicOperation(&result[3], data[id], cudf::DeviceAnd{});
    }
}

template <typename T>
struct AtomicsTest : public GdfTest
{
    void atomic_test(std::vector<int> const & v, bool is_cas_test,
        int block_size=0, int grid_size=1)
    {
        int exact[3];
        exact[0] = std::accumulate(v.begin(), v.end(), 0);
        exact[1] = *( std::min_element(v.begin(), v.end()) );
        exact[2] = *( std::max_element(v.begin(), v.end()) );
        size_t vec_size = v.size();

        // std::vector<T> v_type({6, -14, 13, 64, -13, -20, 45}));
        // use transform from std::vector<int> instead.
        std::vector<T> v_type(vec_size);
        std::transform(v.begin(), v.end(), v_type.begin(),
            [](int x) { T t(x) ; return t; } );

        std::vector<T> result_init(4);
        result_init[0] = T{0};
        result_init[1] = std::numeric_limits<T>::max();
        result_init[2] = std::numeric_limits<T>::min();
        result_init[3] = T{0};

        thrust::device_vector<T> dev_result(result_init);
        thrust::device_vector<T> dev_data(v_type);

        if( block_size == 0) block_size = vec_size;

        if( is_cas_test ){
          gpu_atomicCAS_test<<<grid_size, block_size>>>(
              dev_result.data().get(), dev_data.data().get(), vec_size);
        }else{
          gpu_atomic_test<<<grid_size, block_size>>>(
              dev_result.data().get(), dev_data.data().get(), vec_size);
        }

        thrust::host_vector<T> host_result(dev_result);
        cudaDeviceSynchronize();
        CUDA_CHECK_LAST();

        EXPECT_EQ(host_result[0], T(exact[0])) << "atomicAdd test failed";
        EXPECT_EQ(host_result[1], T(exact[1])) << "atomicMin test failed";
        EXPECT_EQ(host_result[2], T(exact[2])) << "atomicMax test failed";
        EXPECT_EQ(host_result[3], T(exact[0])) << "atomicAdd test(2) failed";
    }
};

using TestingTypes = ::testing::Types<
    int8_t, int16_t, int32_t, int64_t, float, double,
    cudf::date32, cudf::date64, cudf::timestamp, cudf::category,
    cudf::nvstring_category>;

TYPED_TEST_CASE(AtomicsTest, TestingTypes);

// tests for atomicAdd/Min/Max
TYPED_TEST(AtomicsTest, atomicOps)
{
    bool is_cas_test = false;
    std::vector<int> input_array({0, 6, 0, -14, 13, 64, -13, -20, 45});
    this->atomic_test(input_array, is_cas_test);

    std::vector<int> input_array2({6, -6, 13, 62, -11, -20, 33});
    this->atomic_test(input_array2, is_cas_test);
}

// tests for atomicCAS
TYPED_TEST(AtomicsTest, atomicCAS)
{
    bool is_cas_test = true;
    std::vector<int> input_array({0, 6, 0, -14, 13, 64, -13, -20, 45});
    this->atomic_test(input_array, is_cas_test);

    std::vector<int> input_array2({6, -6, 13, 62, -11, -20, 33});
    this->atomic_test(input_array2, is_cas_test);
}

// tests for atomicAdd/Min/Max
TYPED_TEST(AtomicsTest, atomicOpsGrid)
{
    bool is_cas_test = false;
    int block_size=3;
    int grid_size=4;

    std::vector<int> input_array({0, 6, 0, -14, 13, 64, -13, -20, 45});
    this->atomic_test(input_array, is_cas_test, block_size, grid_size);

    std::vector<int> input_array2({6, -6, 13, 62, -11, -20, 33});
    this->atomic_test(input_array2, is_cas_test, block_size, grid_size);
}

// tests for atomicCAS
TYPED_TEST(AtomicsTest, atomicCASGrid)
{
    bool is_cas_test = true;
    int block_size=3;
    int grid_size=4;

    std::vector<int> input_array({0, 6, 0, -14, 13, 64, -13, -20, 45});
    this->atomic_test(input_array, is_cas_test, block_size, grid_size);

    std::vector<int> input_array2({6, -6, 13, 62, -11, -20, 33});
    this->atomic_test(input_array2, is_cas_test, block_size, grid_size);
}

// tests for large array
TYPED_TEST(AtomicsTest, atomicOpsRandom)
{
    bool is_cas_test = false;
    int block_size=256;
    int grid_size=64;

    std::vector<int> input_array(grid_size * block_size);

    std::default_random_engine engine;
    std::uniform_int_distribution<> dist(-10, 10);
    std::generate(input_array.begin(), input_array.end(),
      [&](){ return dist(engine);} );

    this->atomic_test(input_array, is_cas_test, block_size, grid_size);
}

TYPED_TEST(AtomicsTest, atomicCASRandom)
{
    bool is_cas_test = true;
    int block_size=256;
    int grid_size=64;

    std::vector<int> input_array(grid_size * block_size);

    std::default_random_engine engine;
    std::uniform_int_distribution<> dist(-10, 10);
    std::generate(input_array.begin(), input_array.end(),
      [&](){ return dist(engine);} );

    this->atomic_test(input_array, is_cas_test, block_size, grid_size);
}

// ------------------------------------------------------------------

template <typename T>
struct AtomicsBitwiseOpTest : public GdfTest
{
    void atomic_test(std::vector<uint64_t> const & v,
        int block_size=0, int grid_size=1)
    {
        std::vector<T> identity = {T(~0ull), T(0), T(0), T(~0ull)};
        T exact[4];
        exact[0] = std::accumulate(v.begin(), v.end(), identity[0],
            [](T acc, uint64_t i) { return acc & T(i); });
        exact[1] = std::accumulate(v.begin(), v.end(), identity[1],
            [](T acc, uint64_t i) { return acc | T(i); });
        exact[2] = std::accumulate(v.begin(), v.end(), identity[2],
            [](T acc, uint64_t i) { return acc ^ T(i); });
        exact[3] = exact[0];

        size_t vec_size = v.size();

        std::vector<T> v_type(vec_size);
        std::transform(v.begin(), v.end(), v_type.begin(),
            [](uint64_t x) { T t(x) ; return t; } );

        std::vector<T> result_init(identity);


        thrust::device_vector<T> dev_result(result_init);
        thrust::device_vector<T> dev_data(v_type);

        if( block_size == 0) block_size = vec_size;

	gpu_atomic_bitwiseOp_test<T> <<<grid_size, block_size>>> (
		reinterpret_cast<T*>( dev_result.data().get() ),
		reinterpret_cast<T*>( dev_data.data().get() ),
		vec_size);

        thrust::host_vector<T> host_result(dev_result);
        cudaDeviceSynchronize();
        CUDA_CHECK_LAST();

        print_exact(exact, "exact");
        print_exact(host_result.data(), "result");


        EXPECT_EQ(host_result[0], exact[0]) << "atomicAnd test failed";
        EXPECT_EQ(host_result[1], exact[1]) << "atomicOr test failed";
        EXPECT_EQ(host_result[2], exact[2]) << "atomicXor test failed";
        EXPECT_EQ(host_result[3], exact[0]) << "atomicAnd test(2) failed";
    }

    void print_exact(const T *v, const char* msg){
        std::cout << std::hex << std::showbase;
        std::cout << "The " << msg << " = {" 
            << +v[0] << ", "
            << +v[1] << ", "
            << +v[2] << "}"
            << std::endl;
    }

};

using BitwiseOpTestingTypes = ::testing::Types<
    int8_t, int16_t, int32_t, int64_t,
    uint8_t, uint16_t, uint32_t, uint64_t
    >;

TYPED_TEST_CASE(AtomicsBitwiseOpTest, BitwiseOpTestingTypes);

TYPED_TEST(AtomicsBitwiseOpTest, atomicBitwiseOps)
{
    { // test for AND, XOR
        std::vector<uint64_t> input_array(
            {0xfcfcfcfcfcfcfc7f, 0x7f7f7f7f7f7ffc, 0xfffddffddffddfdf, 0x7f7f7f7f7f7ffc});
        this->atomic_test(input_array);
    }
    { // test for OR, XOR
        std::vector<uint64_t> input_array(
            {0x01, 0xfc02, 0x1dff03, 0x1100a0b0801d0003, 0x8000000000000000, 0x1dff03});
        this->atomic_test(input_array);
    }
}

