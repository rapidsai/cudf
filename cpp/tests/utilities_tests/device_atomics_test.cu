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
#if 0
    int8_t, int16_t, int32_t, int64_t, float, double,
    cudf::date32, cudf::date64, cudf::timestamp, cudf::category
#else
    int8_t, int16_t, int32_t, int64_t, float, double
#endif
    >;


TYPED_TEST_CASE(AtomicsTest, TestingTypes);

TYPED_TEST(AtomicsTest, atomicAdd)
{
    std::vector<int> v({6, -14, 13, 64, -13, -20, 45});
    int exact = std::accumulate(v.begin(), v.end(), 0);
    size_t vec_size = v.size();

    // std::vector<TypeParam> v_type({6, -14, 13, 64, -13, -20, 45}));
    // use transform from std::vector<int> instead.
    std::vector<TypeParam> v_type(vec_size);
    std::transform(v.begin(), v.end(), v_type.begin(),
        [](int x) { TypeParam t(x) ; return t; } );

    thrust::device_vector<TypeParam> dev_result(1);
    thrust::device_vector<TypeParam> dev_data(v_type);
    dev_result[0] = TypeParam{0};

    cudaDeviceSynchronize();
    CUDA_CHECK_LAST();

    gpu_atomicadd_test <<<1, vec_size>>> (
        static_cast<TypeParam*>( dev_result.data().get() ),
        static_cast<TypeParam*>( dev_data.data().get() ),
        vec_size);

    CUDA_CHECK_LAST();
    cudaDeviceSynchronize();

    thrust::host_vector<TypeParam> host_result(dev_result);
    cudaDeviceSynchronize();

    EXPECT_EQ(host_result[0], TypeParam(exact));

}
