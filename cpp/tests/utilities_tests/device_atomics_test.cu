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

template<typename T>
__global__
void gpu_atomicadd_test(T *result, T *data, size_t size)
{
    size_t id   = blockIdx.x * blockDim.x + threadIdx.x;
    size_t step = blockDim.x * gridDim.x;

    for (; id < size; id += step) {
        atomicAdd(result, data[id]);
    }
}

// ---------------------------------------------

template <typename T>
struct AtomicsTest : public GdfTest {
};


using TestingTypes = ::testing::Types<
    int8_t, int16_t, int32_t, int64_t, float, double,
    cudf::date32, cudf::date64, cudf::timestamp, cudf::category
    >;

TYPED_TEST_CASE(AtomicsTest, TestingTypes);

template <typename T>
void AtomicsTest_atomicAdd()
{
   std::vector<int> v({6, -14, 13, 64, -13, -20, 45});
    int exact = std::accumulate(v.begin(), v.end(), 0);
    size_t vec_size = v.size();

    // std::vector<T> v_type({6, -14, 13, 64, -13, -20, 45}));
    // use transform from std::vector<int> instead.
    std::vector<T> v_type(vec_size);
    std::transform(v.begin(), v.end(), v_type.begin(),
        [](int x) { T t(x) ; return t; } );

    thrust::device_vector<T> dev_result(1);
    thrust::device_vector<T> dev_data(v_type);
    dev_result[0] = T{0};

    cudaDeviceSynchronize();
    CUDA_CHECK_LAST();


    gpu_atomicadd_test<T> <<<1, vec_size>>> (
        reinterpret_cast<T*>( dev_result.data().get() ),
        reinterpret_cast<T*>( dev_data.data().get() ),
        vec_size);

    cudaDeviceSynchronize();
    CUDA_CHECK_LAST();

    thrust::host_vector<T> host_result(dev_result);
    cudaDeviceSynchronize();

    EXPECT_EQ(host_result[0], T(exact));
}


TYPED_TEST(AtomicsTest, atomicAdd)
{
    // TODO: remove this workaround
    // At TYPED_TEST, the kernel for TypeParam of `wrapper` types won't be compiled,
    // then kenrel call failed by `cudaErrorInvalidDeviceFunction`
    // Explicit typename is requried at this point.
    if( std::is_arithmetic<TypeParam>::value ){                 AtomicsTest_atomicAdd<TypeParam>(); }
    else if( std::is_same<TypeParam, cudf::date32>::value ){    AtomicsTest_atomicAdd<cudf::date32>(); }
    else if( std::is_same<TypeParam, cudf::date64>::value ){    AtomicsTest_atomicAdd<cudf::date64>(); }
    else if( std::is_same<TypeParam, cudf::category>::value){   AtomicsTest_atomicAdd<cudf::category>(); }
    else if( std::is_same<TypeParam, cudf::timestamp>::value ){ AtomicsTest_atomicAdd<cudf::timestamp>(); }

}
