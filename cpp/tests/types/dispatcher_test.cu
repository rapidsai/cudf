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
#include <cstdint>
#include "gtest/gtest.h"
#include <utilities/type_dispatcher.hpp>
#include <thrust/device_vector.h>
#include <cudf.h>
#include "tests/utilities/cudf_test_fixtures.h"

/**
 * @file dispatcher_test.cu
 * @brief Tests the type_dispatcher
*/

namespace{ 

struct test_functor {
  template <typename T> 
  __host__ __device__
  bool operator()(gdf_dtype type_id) {
    switch (type_id) {
    case GDF_INT8:
      return (std::is_same<T, int8_t>::value);
    case GDF_INT16:
      return (std::is_same<T, int16_t>::value);
    case GDF_INT32:
      return (std::is_same<T, int32_t>::value);
    case GDF_INT64:
      return (std::is_same<T, int64_t>::value);
    case GDF_FLOAT32:
      return (std::is_same<T, float>::value);
    case GDF_FLOAT64:
      return (std::is_same<T, double>::value);
    case GDF_CATEGORY:
      return (std::is_same<T, cudf::category>::value);
    case GDF_TIMESTAMP:
      return (std::is_same<T, cudf::timestamp>::value);
    case GDF_DATE32:
      return (std::is_same<T, cudf::date32>::value);
    case GDF_DATE64:
      return (std::is_same<T, cudf::date64>::value);
    default:
      return (false);
    }
  }
};

__global__ 
void dispatch_test_kernel(gdf_dtype type, bool * d_result)
{
  if(0 == threadIdx.x + blockIdx.x * blockDim.x)
    *d_result = cudf::type_dispatcher(type, test_functor{}, type);
}

} // anonymous namespace

struct DispatcherTest : public GdfTest 
{
  std::vector<gdf_dtype> dtype_enums{
    GDF_INT8, GDF_INT16, GDF_INT32, GDF_INT64, GDF_FLOAT32, 
    GDF_FLOAT64, GDF_DATE32, GDF_DATE64, GDF_TIMESTAMP, GDF_CATEGORY};
};

TEST_F(DispatcherTest, HostDispatchFunctor)
{
  for (auto const &t : this->dtype_enums) {
    bool result = cudf::type_dispatcher(t, test_functor{}, t);
    EXPECT_TRUE(result);
  }
}

TEST_F(DispatcherTest, DeviceDispatchFunctor)
{
  thrust::device_vector<bool> result(1);
  for (auto const& t : this->dtype_enums) {
    dispatch_test_kernel<<<1,1>>>(t, result.data().get());
    cudaDeviceSynchronize();
    EXPECT_EQ(true, result[0]);
  }
}

// These tests excerise the `assert(false)` on unsupported dtypes in the type_dispatcher
// The assert is only present if the NDEBUG macro isn't defined 
#ifndef NDEBUG

// Unsuported gdf_dtypes should cause program to exit
TEST(DispatcherDeathTest, UnsuportedTypesTest)
{
  testing::FLAGS_gtest_death_test_style="threadsafe";
  std::vector<gdf_dtype> unsupported_types{ GDF_invalid, GDF_STRING, N_GDF_TYPES};
  for (auto const &t : unsupported_types) {
    EXPECT_DEATH(cudf::type_dispatcher(t, test_functor{}, t), "");
  }
}

// Unsuported gdf_dtypes in device code should set appropriate error code
// and invalidates device context
TEST(DispatcherDeathTest, DeviceDispatchFunctor)
{
  testing::FLAGS_gtest_death_test_style="threadsafe";
  std::vector<gdf_dtype> unsupported_types{ GDF_invalid, GDF_STRING, N_GDF_TYPES};
  thrust::device_vector<bool> result(1);

  auto call_kernel = [&result](gdf_dtype t) {
    dispatch_test_kernel<<<1, 1>>>(t, result.data().get());
    auto error_code = cudaDeviceSynchronize();

    // Kernel should fail with `cudaErrorAssert` on an unsupported gdf_dtype
    // This error invalidates the current device context, so we need to kill 
    // the current process. Running with EXPECT_DEATH spawns a new process for
    // each attempted kernel launch
    EXPECT_EQ(cudaErrorAssert, error_code);
    exit(-1);
  };

  for (auto const& t : unsupported_types) {
    EXPECT_DEATH(call_kernel(t), ""); 
  }
}

#endif
