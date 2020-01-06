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
#include <cudf/utilities/error.hpp>
#include <cudf/cudf.h>

#include <rmm/rmm.h>

#include <tests/utilities/legacy/cudf_test_fixtures.h>

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
  CUDF_EXPECT_THROW_MESSAGE(CUDF_EXPECTS(false, "test reason"), 
                            "test reason");
}

TEST(CudaTryTest, Error) {
  CUDA_EXPECT_THROW_MESSAGE(CUDA_TRY(cudaErrorLaunchFailure),
                            "cudaErrorLaunchFailure unspecified launch failure");
}
TEST(CudaTryTest, Success) { EXPECT_NO_THROW(CUDA_TRY(cudaSuccess)); }

TEST(CudaTryTest, TryCatch) {
  CUDA_EXPECT_THROW_MESSAGE(CUDA_TRY(cudaErrorMemoryAllocation),
                            "cudaErrorMemoryAllocation out of memory");
}

TEST(StreamCheck, success) {
  EXPECT_NO_THROW(CHECK_CUDA(0));
}

namespace {
// Some silly kernel that will cause an error
void __global__ test_kernel(int* data) { data[threadIdx.x] = threadIdx.x; }
}  // namespace

// In a release build and without explicit synchronization, CHECK_CUDA may
// or may not fail on erroneous asynchronous CUDA calls. Invoke
// cudaStreamSynchronize to guarantee failure on error. In a non-release build,
// CHECK_CUDA deterministically fails on erroneous asynchronous CUDA
// calls.
TEST(StreamCheck, FailedKernel) {
  cudaStream_t stream;
  CUDA_TRY(cudaStreamCreate(&stream));
  int a;
  test_kernel<<<0, 0, 0, stream>>>(&a);
#ifdef NDEBUG
  CUDA_TRY(cudaStreamSynchronize(stream));
#endif
  EXPECT_THROW(CHECK_CUDA(stream), cudf::cuda_error);
  CUDA_TRY(cudaStreamDestroy(stream));
}

TEST(StreamCheck, CatchFailedKernel) {
  cudaStream_t stream;
  CUDA_TRY(cudaStreamCreate(&stream));
  int a;
  test_kernel<<<0, 0, 0, stream>>>(&a);
#ifndef NDEBUG
  CUDA_TRY(cudaStreamSynchronize(stream));
#endif
  CUDA_EXPECT_THROW_MESSAGE(CHECK_CUDA(stream),
                            "cudaErrorInvalidConfiguration "
                            "invalid configuration argument");
  CUDA_TRY(cudaStreamDestroy(stream));
}

