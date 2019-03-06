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
#include "cudf.h"
#include "gtest/gtest.h"
#include "utilities/error_utils.hpp"

#include <cuda_runtime_api.h>
#include <rmm/rmm.h>
#include <cstring>

// If this test fails, it means an error code was added without
// adding support to gdf_error_get_name().
TEST(ErrorTest, NameEveryError) {
  for (int i = 0; i < N_GDF_ERRORS; i++) {
    const char* res = gdf_error_get_name((gdf_error)i);
    ASSERT_EQ(0, strstr(res, "Unknown error"));
  }
}

TEST(ExpectsTest, FalseCondition) {
  EXPECT_THROW(CUDF_EXPECTS(false, "condition is false"), cudf::logic_error);
}

TEST(ExpectsTest, TrueCondition) {
  EXPECT_NO_THROW(CUDF_EXPECTS(true, "condition is true"));
}

TEST(ExpectsTest, TryCatch) {
  try {
    CUDF_EXPECTS(false, "test reason");
  } catch (cudf::logic_error const& e) {
    EXPECT_NE(nullptr, e.what());
    std::string what(e.what());
    EXPECT_NE(std::string::npos, what.find("cuDF failure at:"));
    EXPECT_NE(std::string::npos, what.find("test reason"));
  }
}

TEST(CudaTryTest, Error) {
  EXPECT_THROW(CUDA_TRY(cudaErrorLaunchFailure), cudf::cuda_error);
}
TEST(CudaTryTest, Success) { EXPECT_NO_THROW(CUDA_TRY(cudaSuccess)); }

TEST(CudaTryTest, TryCatch) {
  try {
    CUDA_TRY(cudaErrorMemoryAllocation);
  } catch (cudf::cuda_error const& e) {
    ASSERT_NE(nullptr, e.what());
    std::string what(e.what());
    EXPECT_NE(std::string::npos, what.find("CUDA error encountered at"));
    EXPECT_NE(std::string::npos, what.find("cudaErrorMemoryAllocation"));
  }
}

TEST(StreamCheck, success) {
  EXPECT_NO_THROW(cudf::detail::check_stream(0, __FILE__, __LINE__));
}

namespace {
// Some silly kernel that will cause an error
void __global__ test_kernel(int* data) { data[threadIdx.x] = threadIdx.x; }
}  // namespace

// Test the function underlying CHECK_STREAM so that it throws an exception when
// a kernel fails
TEST(StreamCheck, FailedKernel) {
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  int a;
  test_kernel<<<0, 0, 0, stream>>>(&a);
  EXPECT_THROW(cudf::detail::check_stream(0, __FILE__, __LINE__),
               cudf::cuda_error);
  cudaStreamDestroy(stream);
}

TEST(StreamCheck, CatchFailedKernel) {
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  int a;
  test_kernel<<<0, 0, 0, stream>>>(&a);
  try {
    cudf::detail::check_stream(0, __FILE__, __LINE__);
  } catch (cudf::cuda_error const& e) {
    ASSERT_NE(nullptr, e.what());
    std::string what(e.what());
    EXPECT_NE(std::string::npos, what.find("CUDA error encountered at"));
    EXPECT_NE(
        std::string::npos,
        what.find(
            "cudaErrorInvalidConfiguration invalid configuration argument"));
  }
  cudaStreamDestroy(stream);
}

// CHECK_STREAM should do nothing in a release build, even if there's an error
#ifdef NDEBUG
TEST(StreamCheck, ReleaseFailedKernel) {
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  int a;
  test_kernel<<<0, 0, 0, stream>>>(&a);
  EXPECT_NO_THROW(CHECK_STREAM(0));
  cudaStreamDestroy(stream);
}
#endif

// STREAM_CHECK only works in a non-Release build
#ifndef NDEBUG
TEST(StreamCheck, test) { EXPECT_NO_THROW(CHECK_STREAM(0)); }
#endif
